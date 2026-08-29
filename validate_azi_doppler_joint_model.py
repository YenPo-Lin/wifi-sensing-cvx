"""Validate Azi_DopX with the joint real-power model from thesis Section 4.3.2.

The validation deliberately keeps the production estimator unchanged.  It:

1. Generates complex CSI with one static reference path and one dynamic path.
2. Converts it to real CSI power and removes the moving average.
3. Confirms that the current Azi_DopX covariance matches an independent builder.
4. Compares the current smoothing parameters with the paper's T/2 construction.
5. Tests both spatial steering signs and a separable symmetric control signal.
6. Saves heatmaps, Doppler slices, and machine-readable metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MUSIC


LIGHT_SPEED = 3.0e8


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    window = np.ones(int(window_size), dtype=np.float64) / float(window_size)
    return np.apply_along_axis(
        lambda values: np.convolve(values, window, mode="same"),
        axis=0,
        arr=data,
    )


def generate_joint_power_model(
    *,
    num_frames: int,
    num_rx: int,
    num_scarriers: int,
    fs: float,
    f0: float,
    antenna_spacing: float,
    theta_deg: float,
    fd_hz: float,
    dynamic_amplitude: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw complex CSI, power, and moving-average-subtracted power."""

    rng = np.random.default_rng(seed)
    time_seconds = np.arange(num_frames, dtype=np.float64) / fs
    rx_indices = np.arange(num_rx, dtype=np.float64)
    subcarrier_indices = np.arange(num_scarriers, dtype=np.float64)

    psi = 1.0 - np.cos(np.deg2rad(theta_deg))
    spatial_phase = (
        2.0
        * np.pi
        * f0
        * antenna_spacing
        / LIGHT_SPEED
        * psi
        * rx_indices
    )
    # A deterministic subcarrier phase emulates different path delays while
    # preserving the same joint AoA-Doppler subspace for covariance averaging.
    subcarrier_phase = 0.13 * np.pi * subcarrier_indices
    dynamic_phase = (
        2.0 * np.pi * fd_hz * time_seconds[:, None, None]
        + spatial_phase[None, :, None]
        + subcarrier_phase[None, None, :]
    )

    raw_csi = 1.0 + dynamic_amplitude * np.exp(1j * dynamic_phase)
    raw_csi += noise_std * (
        rng.standard_normal(raw_csi.shape)
        + 1j * rng.standard_normal(raw_csi.shape)
    )

    power = np.abs(raw_csi) ** 2
    background = moving_average(power, int(round(fs * 0.5)))
    dynamic_power = power - background
    return raw_csi, power, dynamic_power


def generate_separable_control(
    *,
    num_frames: int,
    num_rx: int,
    num_scarriers: int,
    fs: float,
    f0: float,
    antenna_spacing: float,
    theta_deg: float,
    fd_hz: float,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Create a real signal with separable spatial and temporal factors.

    cos(spatial) * cos(temporal) contains both Doppler signs at the same
    positive spatial frequency.  It is a control showing the symmetric result
    expected after spatial-temporal coupling has been lost.
    """

    rng = np.random.default_rng(seed)
    time_seconds = np.arange(num_frames, dtype=np.float64) / fs
    rx_indices = np.arange(num_rx, dtype=np.float64)
    psi = 1.0 - np.cos(np.deg2rad(theta_deg))
    spatial_phase = (
        2.0
        * np.pi
        * f0
        * antenna_spacing
        / LIGHT_SPEED
        * psi
        * rx_indices
    )
    temporal = np.cos(2.0 * np.pi * fd_hz * time_seconds)
    spatial = np.cos(spatial_phase)
    control = (
        temporal[:, None, None]
        * spatial[None, :, None]
        * np.ones((1, 1, num_scarriers), dtype=np.float64)
    )
    control += noise_std * rng.standard_normal(control.shape)
    return control


def centered_segment(
    data: np.ndarray,
    frame_idx: int,
    context_len: int,
) -> tuple[np.ndarray, int, int]:
    start = int(
        np.clip(
            int(frame_idx) - context_len // 2,
            0,
            data.shape[0] - context_len,
        )
    )
    end = start + context_len
    return data[start:end], start, end


def build_covariance(
    data: np.ndarray,
    *,
    frame_idx: int,
    context_len: int,
    stream_win: int,
    dop_win: int,
    time_hop: int,
    include_last_window: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    """Independent covariance builder using (time, rx, subcarrier) data."""

    segment, start, end = centered_segment(data, frame_idx, context_len)
    stop = context_len - dop_win + int(include_last_window)
    time_starts = np.arange(0, stop, time_hop, dtype=np.int64)
    stream_starts = np.arange(
        0,
        data.shape[1] - stream_win + 1,
        dtype=np.int64,
    )
    if time_starts.size == 0 or stream_starts.size == 0:
        raise ValueError("Smoothing settings produced no snapshots")

    vector_size = stream_win * dop_win
    covariance = np.zeros((vector_size, vector_size), dtype=np.complex128)
    count = 0
    for subcarrier in range(data.shape[2]):
        for stream_start in stream_starts:
            for time_start in time_starts:
                block = segment[
                    time_start : time_start + dop_win,
                    stream_start : stream_start + stream_win,
                    subcarrier,
                ]
                vector = block.T.reshape(-1)
                covariance += np.outer(vector, vector.conj())
                count += 1

    covariance /= count
    covariance = (covariance + covariance.conj().T) / 2.0
    metadata = {
        "context_start": start,
        "context_end": end,
        "context_len": context_len,
        "dop_win": dop_win,
        "stream_win": stream_win,
        "time_slides": int(time_starts.size),
        "stream_slides": int(stream_starts.size),
        "subcarriers": int(data.shape[2]),
        "snapshots": count,
        "vector_size": vector_size,
    }
    return covariance, metadata


def music_spectrum(
    covariance: np.ndarray,
    *,
    theta_axis: np.ndarray,
    fd_axis: np.ndarray,
    stream_win: int,
    dop_win: int,
    fs: float,
    f0: float,
    antenna_spacing: float,
    spatial_sign: int,
    signal_dimension: int,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    noise_subspace = eigenvectors[:, order[signal_dimension:]]

    theta_rad = np.deg2rad(theta_axis)
    spatial_phase = (
        2.0
        * np.pi
        * f0
        * antenna_spacing
        / LIGHT_SPEED
        * (1.0 - np.cos(theta_rad))[:, None]
        * np.arange(stream_win, dtype=np.float64)[None, :]
    )
    spatial_steering = np.exp(1j * spatial_sign * spatial_phase)
    time_seconds = np.arange(dop_win, dtype=np.float64) / fs
    doppler_steering = np.exp(
        1j * 2.0 * np.pi * fd_axis[:, None] * time_seconds[None, :]
    )
    steering = np.einsum(
        "is,jt->ijst",
        spatial_steering,
        doppler_steering,
    ).reshape(theta_axis.size * fd_axis.size, stream_win * dop_win)
    steering /= np.sqrt(stream_win * dop_win)

    projection = steering.conj() @ noise_subspace
    denominator = np.sum(np.abs(projection) ** 2, axis=1)
    spectrum_db = 10.0 * np.log10(1.0 / (denominator + 1e-12))
    return spectrum_db.reshape(theta_axis.size, fd_axis.size)


def peak_metrics(
    spectrum_db: np.ndarray,
    theta_axis: np.ndarray,
    fd_axis: np.ndarray,
    theta_true: float,
    fd_true: float,
) -> dict[str, float]:
    peak_index = np.unravel_index(np.argmax(spectrum_db), spectrum_db.shape)
    theta_index = int(np.argmin(np.abs(theta_axis - theta_true)))
    positive_index = int(np.argmin(np.abs(fd_axis - abs(fd_true))))
    negative_index = int(np.argmin(np.abs(fd_axis + abs(fd_true))))
    positive_db = float(spectrum_db[theta_index, positive_index])
    negative_db = float(spectrum_db[theta_index, negative_index])
    return {
        "peak_theta_deg": float(theta_axis[peak_index[0]]),
        "peak_fd_hz": float(fd_axis[peak_index[1]]),
        "peak_db": float(spectrum_db[peak_index]),
        "true_theta_positive_fd_db": positive_db,
        "true_theta_negative_fd_db": negative_db,
        "positive_minus_negative_db": positive_db - negative_db,
    }


def current_estimator_args(
    *,
    num_rx: int,
    num_scarriers: int,
    fs: float,
    f0: float,
    antenna_spacing: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        Sdim=2,
        Sdim_energy_ratio=0.3,
        f_0=f0,
        BW=160e6,
        delta_f=2.54e6,
        fs=fs,
        antenna_spacing=antenna_spacing,
        projection="cos",
        stream_win=5,
        stream_sample_range=num_rx,
        freq_win=48,
        freq_hop=1,
        freq_sample_range=num_scarriers,
        freq_space=1,
        time_win=24,
        time_hop=1,
        time_sample_range=64,
        theta_min=0,
        theta_max=180,
        theta_step=1,
        doppler_min=-20,
        doppler_max=20,
        doppler_step=1,
        num_Rx=num_rx,
        num_scarriers=num_scarriers,
    )


def relative_db(spectrum_db: np.ndarray) -> np.ndarray:
    return spectrum_db - np.max(spectrum_db)


def save_input_figure(
    output_path: Path,
    joint_data: np.ndarray,
    separable_data: np.ndarray,
    *,
    frame_idx: int,
    fs: float,
    context_len: int,
) -> None:
    joint_segment, _, _ = centered_segment(joint_data, frame_idx, context_len)
    separable_segment, _, _ = centered_segment(separable_data, frame_idx, context_len)
    time_axis = np.arange(context_len) / fs
    values = [joint_segment[:, :, 0], separable_segment[:, :, 0]]
    limit = max(float(np.max(np.abs(value))) for value in values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    titles = [
        "Joint power cross-term: cos(temporal + spatial)",
        "Separable control: cos(temporal) x cos(spatial)",
    ]
    for axis, value, title in zip(axes, values, titles):
        image = axis.imshow(
            value.T,
            origin="lower",
            aspect="auto",
            extent=[time_axis[0], time_axis[-1], 0, value.shape[1] - 1],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel("Time in selected context (s)")
        axis.set_ylabel("Rx index")
        fig.colorbar(image, ax=axis, label="Dynamic power (linear)")
    fig.suptitle("Synthetic 8-Rx ULA inputs")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_heatmap_figure(
    output_path: Path,
    spectra: list[tuple[str, np.ndarray]],
    theta_axis: np.ndarray,
    fd_axis: np.ndarray,
    *,
    theta_true: float,
    fd_true: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    images = []
    for axis, (title, spectrum_db) in zip(axes.flat, spectra):
        image = axis.pcolormesh(
            theta_axis,
            fd_axis,
            relative_db(spectrum_db).T,
            cmap="turbo",
            shading="auto",
            vmin=-30,
            vmax=0,
        )
        images.append(image)
        axis.plot(theta_true, fd_true, "w*", markersize=9, label="Injected target")
        axis.plot(theta_true, -fd_true, "wx", markersize=7, label="Same-angle mirror")
        axis.set_title(title)
        axis.set_xlabel("Azimuth (deg)")
        axis.set_ylabel("Doppler frequency (Hz)")
        axis.set_xlim(theta_axis[0], theta_axis[-1])
        axis.set_ylim(fd_axis[0], fd_axis[-1])
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        label="Relative MUSIC power (dB)",
        shrink=0.92,
    )
    fig.suptitle("Azi-Doppler spatial-sign and smoothing validation")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_slice_figure(
    output_path: Path,
    spectra: list[tuple[str, np.ndarray]],
    theta_axis: np.ndarray,
    fd_axis: np.ndarray,
    *,
    theta_true: float,
    fd_true: float,
) -> None:
    theta_index = int(np.argmin(np.abs(theta_axis - theta_true)))
    fig, (line_axis, bar_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    deltas = []
    labels = []
    for title, spectrum_db in spectra:
        slice_db = spectrum_db[theta_index]
        slice_db = slice_db - np.max(slice_db)
        line_axis.plot(fd_axis, slice_db, linewidth=1.8, label=title)
        metrics = peak_metrics(
            spectrum_db,
            theta_axis,
            fd_axis,
            theta_true,
            fd_true,
        )
        labels.append(title)
        deltas.append(metrics["positive_minus_negative_db"])

    line_axis.axvline(fd_true, color="k", linestyle="--", linewidth=1.0)
    line_axis.axvline(-fd_true, color="k", linestyle=":", linewidth=1.0)
    line_axis.set_title(f"Doppler slice at azimuth {theta_true:.0f} deg")
    line_axis.set_xlabel("Doppler frequency (Hz)")
    line_axis.set_ylabel("Relative MUSIC power (dB)")
    line_axis.set_ylim(-35, 1)
    line_axis.grid(alpha=0.25)
    line_axis.legend(fontsize=8, ncol=2)

    positions = np.arange(len(deltas))
    colors = ["tab:blue" if value >= 0 else "tab:orange" for value in deltas]
    bars = bar_axis.bar(positions, deltas, color=colors)
    bar_axis.axhline(0, color="k", linewidth=0.8)
    bar_axis.set_xticks(positions, labels, rotation=18, ha="right")
    bar_axis.set_ylabel("P(+8 Hz) - P(-8 Hz) (dB)")
    bar_axis.set_title("Same-angle mirror rejection")
    for bar, value in zip(bars, deltas):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.7 if value >= 0 else -0.7),
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_outputs/azi_doppler_joint_model"),
    )
    parser.add_argument("--theta", type=float, default=110.0)
    parser.add_argument("--fd", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    fs = 100.0
    f0 = 5.57e9
    antenna_spacing = 0.02
    num_frames = 256
    num_rx = 8
    num_scarriers = 64
    frame_idx = num_frames // 2

    raw_csi, _, joint_power = generate_joint_power_model(
        num_frames=num_frames,
        num_rx=num_rx,
        num_scarriers=num_scarriers,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        theta_deg=cli.theta,
        fd_hz=cli.fd,
        dynamic_amplitude=0.28,
        noise_std=0.02,
        seed=cli.seed,
    )
    amplitude = np.abs(raw_csi)
    current_pipeline_input = amplitude - moving_average(
        amplitude,
        int(round(fs * 0.5)),
    )
    separable_control = generate_separable_control(
        num_frames=num_frames,
        num_rx=num_rx,
        num_scarriers=num_scarriers,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        theta_deg=cli.theta,
        fd_hz=cli.fd,
        noise_std=0.02,
        seed=cli.seed + 1,
    )

    args = current_estimator_args(
        num_rx=num_rx,
        num_scarriers=num_scarriers,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
    )
    current_estimator = MUSIC.Azi_DopX(args)
    current_covariance = current_estimator.Rxx_smooth(
        current_pipeline_input[:, None, :, :],
        frame_idx,
    )
    if current_covariance is None:
        raise RuntimeError("Current Azi_DopX did not produce a covariance matrix")
    theta_axis, fd_axis, current_estimator_spectrum = current_estimator.cal_spectrum(
        current_covariance
    )

    manual_current_covariance, current_metadata = build_covariance(
        current_pipeline_input,
        frame_idx=frame_idx,
        context_len=64,
        stream_win=5,
        dop_win=24,
        time_hop=1,
        include_last_window=False,
    )
    np.testing.assert_allclose(
        current_covariance,
        manual_current_covariance,
        rtol=1e-11,
        atol=1e-11,
    )
    # The current production cos projection uses a negative spatial exponent.
    current_manual = music_spectrum(
        current_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=24,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=-1,
        signal_dimension=2,
    )
    np.testing.assert_allclose(
        current_estimator_spectrum,
        current_manual,
        rtol=1e-10,
        atol=1e-10,
    )
    current_opposite_sign = music_spectrum(
        current_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=24,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=1,
        signal_dimension=2,
    )

    paper_covariance, paper_metadata = build_covariance(
        joint_power,
        frame_idx=frame_idx,
        context_len=50,
        stream_win=5,
        dop_win=25,
        time_hop=1,
        include_last_window=True,
    )
    paper_positive = music_spectrum(
        paper_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=25,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=1,
        signal_dimension=2,
    )
    paper_negative = music_spectrum(
        paper_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=25,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=-1,
        signal_dimension=2,
    )

    separable_covariance, separable_metadata = build_covariance(
        separable_control,
        frame_idx=frame_idx,
        context_len=50,
        stream_win=5,
        dop_win=25,
        time_hop=1,
        include_last_window=True,
    )
    separable_positive = music_spectrum(
        separable_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=25,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=1,
        signal_dimension=4,
    )
    separable_negative = music_spectrum(
        separable_covariance,
        theta_axis=theta_axis,
        fd_axis=fd_axis,
        stream_win=5,
        dop_win=25,
        fs=fs,
        f0=f0,
        antenna_spacing=antenna_spacing,
        spatial_sign=-1,
        signal_dimension=4,
    )

    spectra = [
        ("Current pipeline |H| residual, sign -", current_estimator_spectrum),
        ("Current pipeline |H| residual, sign +", current_opposite_sign),
        ("Paper T=50/25, spatial sign +", paper_positive),
        ("Paper T=50/25, spatial sign -", paper_negative),
        ("Separable control, spatial sign +", separable_positive),
        ("Separable control, spatial sign -", separable_negative),
    ]

    save_input_figure(
        cli.output_dir / "01_input_joint_vs_separable.png",
        joint_power,
        separable_control,
        frame_idx=frame_idx,
        fs=fs,
        context_len=50,
    )
    save_heatmap_figure(
        cli.output_dir / "02_heatmap_spatial_sign_validation.png",
        spectra,
        theta_axis,
        fd_axis,
        theta_true=cli.theta,
        fd_true=cli.fd,
    )
    save_slice_figure(
        cli.output_dir / "03_doppler_slice_mirror_rejection.png",
        spectra,
        theta_axis,
        fd_axis,
        theta_true=cli.theta,
        fd_true=cli.fd,
    )

    metrics = {
        "injected_target": {
            "theta_deg": cli.theta,
            "fd_hz": cli.fd,
            "num_rx": num_rx,
            "antenna_spacing_m": antenna_spacing,
            "fs_hz": fs,
        },
        "current_covariance": current_metadata,
        "paper_covariance": paper_metadata,
        "separable_covariance": separable_metadata,
        "spectra": {
            title: peak_metrics(
                spectrum,
                theta_axis,
                fd_axis,
                cli.theta,
                cli.fd,
            )
            for title, spectrum in spectra
        },
        "checks": {
            "current_covariance_matches_independent_builder": True,
            "current_spectrum_matches_negative_sign_builder": True,
            "current_covariance_shape": list(current_covariance.shape),
            "paper_covariance_shape": list(paper_covariance.shape),
        },
    }
    with (cli.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved validation outputs to: {cli.output_dir.resolve()}")


if __name__ == "__main__":
    main()
