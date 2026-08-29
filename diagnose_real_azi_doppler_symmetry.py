"""Diagnose forced Azi-Doppler symmetry on a real CSI capture.

The script compares the current real-valued amplitude/power residuals with a
complex-CSI residual, verifies the conjugate spatial-alias identity, and saves
figures plus machine-readable metrics.  It does not modify the production
MUSIC estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MUSIC
import pre_processing as pp
from main import create_parser


LIGHT_SPEED = 3.0e8


def relative_db(spectrum_db: np.ndarray) -> np.ndarray:
    return spectrum_db - np.max(spectrum_db)


def covariance_imaginary_ratio(covariance: np.ndarray) -> float:
    return float(
        np.linalg.norm(covariance.imag)
        / max(np.linalg.norm(covariance), np.finfo(float).eps)
    )


def spectrum_metrics(
    theta: np.ndarray,
    fd: np.ndarray,
    spectrum_db: np.ndarray,
    covariance: np.ndarray,
    theta_self_conjugate: float,
) -> dict[str, object]:
    theta_index = int(np.argmin(np.abs(theta - theta_self_conjugate)))
    mirror_error = spectrum_db[theta_index] - spectrum_db[theta_index, ::-1]
    peak_index = np.unravel_index(np.argmax(spectrum_db), spectrum_db.shape)
    return {
        "peak_theta_deg": float(theta[peak_index[0]]),
        "peak_fd_hz": float(fd[peak_index[1]]),
        "theta_self_grid_deg": float(theta[theta_index]),
        "same_angle_mirror_mean_abs_db": float(np.mean(np.abs(mirror_error))),
        "same_angle_mirror_max_abs_db": float(np.max(np.abs(mirror_error))),
        "covariance_imaginary_ratio": covariance_imaginary_ratio(covariance),
    }


def conjugate_alias_error(
    theta: np.ndarray,
    spectrum_db: np.ndarray,
    wavelength: float,
    spacing: float,
) -> dict[str, float]:
    q = 1.0 - np.cos(np.deg2rad(theta))
    q_mirror = wavelength / spacing - q
    valid = (q_mirror >= 0.0) & (q_mirror <= 2.0)
    theta_mirror = np.rad2deg(np.arccos(1.0 - q_mirror[valid]))
    errors = []
    for theta_index, mirror_angle in zip(np.flatnonzero(valid), theta_mirror):
        mirror_index = int(np.argmin(np.abs(theta - mirror_angle)))
        errors.extend(
            (spectrum_db[theta_index] - spectrum_db[mirror_index, ::-1]).tolist()
        )
    errors = np.asarray(errors)
    return {
        "valid_theta_min_deg": float(theta[valid][0]),
        "valid_theta_max_deg": float(theta[valid][-1]),
        "mean_abs_db": float(np.mean(np.abs(errors))),
        "max_abs_db": float(np.max(np.abs(errors))),
    }


def run_estimator(
    data: np.ndarray,
    args: argparse.Namespace,
    frame_idx: int,
    tx_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    args.azi_dopx_tx_idx = tx_idx
    estimator = MUSIC.Azi_DopX(args)
    covariance = estimator.Rxx_smooth(data, frame_idx)
    if covariance is None:
        raise RuntimeError("Azi_DopX did not produce a covariance matrix")
    theta, fd, spectrum_db = estimator.cal_spectrum(covariance)
    metadata = dict(estimator.last_meta)
    metadata["Sdim"] = int(estimator.last_Sdim)
    return theta, fd, spectrum_db, covariance, metadata


def save_ablation_heatmaps(
    output_path: Path,
    results: dict[str, dict[int, dict[str, object]]],
    theta_self: float,
    frame_idx: int,
) -> None:
    names = ["Amplitude residual", "Power residual (current)", "Complex residual"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    image = None
    for row, tx_idx in enumerate((0, 1)):
        for col, name in enumerate(names):
            result = results[name][tx_idx]
            theta = result["theta"]
            fd = result["fd"]
            spectrum_db = result["spectrum_db"]
            image = axes[row, col].pcolormesh(
                theta,
                fd,
                relative_db(spectrum_db).T,
                shading="auto",
                cmap="turbo",
                vmin=-12,
                vmax=0,
            )
            axes[row, col].axvline(
                theta_self,
                color="white",
                linestyle="--",
                linewidth=1.0,
            )
            metrics = result["metrics"]
            axes[row, col].set_title(
                f"Tx {tx_idx}: {name}\n"
                f"Sdim={result['metadata']['Sdim']}, "
                f"mirror MAE={metrics['same_angle_mirror_mean_abs_db']:.2f} dB"
            )
            axes[row, col].set_xlabel("Azimuth (deg)")
            axes[row, col].set_ylabel("Doppler frequency (Hz)")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Relative MUSIC power (dB)",
            shrink=0.94,
        )
    fig.suptitle(
        f"Real CSI Azi-Doppler preprocessing ablation @ frame {frame_idx}\n"
        f"dashed line: self-conjugate spatial angle {theta_self:.2f} deg"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_identity_figure(
    output_path: Path,
    results: dict[str, dict[int, dict[str, object]]],
    wavelength: float,
    spacing: float,
    theta_self: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    names = ["Amplitude residual", "Power residual (current)", "Complex residual"]
    colors = {
        "Amplitude residual": "tab:blue",
        "Power residual (current)": "tab:orange",
        "Complex residual": "tab:green",
    }

    power = results["Power residual (current)"][0]
    theta = power["theta"]
    fd = power["fd"]
    spectrum_db = power["spectrum_db"]
    image = axes[0, 0].pcolormesh(
        theta,
        fd,
        relative_db(spectrum_db).T,
        shading="auto",
        cmap="turbo",
        vmin=-12,
        vmax=0,
    )
    axes[0, 0].axvline(theta_self, color="white", linestyle="--")
    axes[0, 0].set_title("Tx 0 current power residual")
    axes[0, 0].set_xlabel("Azimuth (deg)")
    axes[0, 0].set_ylabel("Doppler frequency (Hz)")
    fig.colorbar(image, ax=axes[0, 0], label="Relative MUSIC power (dB)")

    for name in names:
        result = results[name][0]
        theta_index = int(np.argmin(np.abs(result["theta"] - theta_self)))
        values = result["spectrum_db"][theta_index]
        axes[0, 1].plot(
            result["fd"],
            values - values.max(),
            label=name,
            color=colors[name],
        )
    axes[0, 1].set_title(f"Tx 0 Doppler slice at {theta_self:.2f} deg")
    axes[0, 1].set_xlabel("Doppler frequency (Hz)")
    axes[0, 1].set_ylabel("Relative MUSIC power (dB)")
    axes[0, 1].set_ylim(-12, 0.5)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    q = np.linspace(max(0.0, wavelength / spacing - 2.0), 2.0, 500)
    q_mirror = wavelength / spacing - q
    theta_curve = np.rad2deg(np.arccos(1.0 - q))
    theta_mirror = np.rad2deg(np.arccos(1.0 - q_mirror))
    axes[1, 0].plot(theta_curve, theta_mirror, color="tab:purple")
    axes[1, 0].plot([0, 180], [0, 180], "k--", linewidth=0.9)
    axes[1, 0].plot(theta_self, theta_self, "ro", label="self-conjugate point")
    axes[1, 0].set_xlim(70, 180)
    axes[1, 0].set_ylim(70, 180)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("Azimuth theta (deg)")
    axes[1, 0].set_ylabel("Conjugate-alias azimuth theta' (deg)")
    axes[1, 0].set_title("q' = wavelength / d - q, q = 1 - cos(theta)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    positions = np.arange(len(names))
    width = 0.35
    real_mae = [
        results[name][0]["metrics"]["same_angle_mirror_mean_abs_db"]
        for name in names
    ]
    imag_ratio = [
        results[name][0]["metrics"]["covariance_imaginary_ratio"]
        for name in names
    ]
    axes[1, 1].bar(positions - width / 2, real_mae, width, label="mirror MAE (dB)")
    axes[1, 1].bar(positions + width / 2, imag_ratio, width, label="imaginary R ratio")
    axes[1, 1].set_xticks(positions, ["Amplitude", "Power", "Complex"])
    axes[1, 1].set_title("Tx 0 symmetry and covariance diagnostics")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Why the real-valued Azi-Doppler map folds around 110 deg")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def adjacent_phase_diagnostics(
    raw_csi: np.ndarray,
) -> tuple[dict[str, dict[str, dict[str, float | str]]], dict[int, dict[int, np.ndarray]]]:
    metrics: dict[str, dict[str, dict[str, float | str]]] = {}
    traces: dict[int, dict[int, np.ndarray]] = {}
    for tx_idx in range(min(2, raw_csi.shape[1])):
        metrics[f"tx_{tx_idx}"] = {}
        traces[tx_idx] = {}
        for rx_idx in range(raw_csi.shape[2] - 1):
            cross = np.sum(
                raw_csi[:, tx_idx, rx_idx + 1, :]
                * np.conj(raw_csi[:, tx_idx, rx_idx, :]),
                axis=1,
            )
            phase = np.unwrap(np.angle(cross))
            phase -= phase[0]
            phase_step = np.diff(phase)
            normalized_cross = cross / np.maximum(np.abs(cross), 1e-12)
            pair_type = "same_card" if rx_idx % 2 == 0 else "cross_card"
            metrics[f"tx_{tx_idx}"][f"rx_{rx_idx}_{rx_idx + 1}"] = {
                "pair_type": pair_type,
                "phase_resultant_coherence": float(np.abs(np.mean(normalized_cross))),
                "phase_step_std_rad": float(np.std(phase_step)),
                "phase_step_p95_abs_rad": float(np.percentile(np.abs(phase_step), 95)),
                "unwrapped_phase_range_rad": float(np.ptp(phase)),
            }
            traces[tx_idx][rx_idx] = phase
    return metrics, traces


def save_phase_stability_figure(
    output_path: Path,
    traces: dict[int, dict[int, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, tx_idx in enumerate(sorted(traces)):
        for rx_idx, phase in traces[tx_idx].items():
            col = 0 if rx_idx % 2 == 0 else 1
            axes[row, col].plot(phase, label=f"Rx {rx_idx}-{rx_idx + 1}")
        axes[row, 0].set_title(f"Tx {tx_idx}: same-card adjacent RX phase")
        axes[row, 1].set_title(f"Tx {tx_idx}: cross-card adjacent RX phase")
        for col in (0, 1):
            axes[row, col].set_xlabel("Frame")
            axes[row, col].set_ylabel("Unwrapped relative phase (rad)")
            axes[row, col].grid(alpha=0.25)
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Complex-CSI phase stability after physical antenna reordering")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csi-file",
        type=Path,
        default=Path("/Users/YPL/Documents/NPZ_files/20260820-172924_move_lr.npz"),
    )
    parser.add_argument("--frame", type=int, default=444)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_outputs/azi_doppler_real_symmetry"),
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(cli.csi_file)
    raw_csi = archive["csi"]

    args = create_parser().parse_args([])
    args.num_frames, args.num_Tx, args.num_Rx, args.num_scarriers = raw_csi.shape

    amplitude = np.abs(raw_csi)
    power = amplitude**2
    inputs = {
        "Amplitude residual": amplitude - pp.MA(amplitude, args.fs * 0.5),
        "Power residual (current)": power - pp.MA(power, args.fs * 0.5),
        "Complex residual": raw_csi - pp.MA(raw_csi, args.fs * 0.5),
    }

    wavelength = LIGHT_SPEED / args.f_0
    q_self = wavelength / (2.0 * args.antenna_spacing)
    if not 0.0 <= q_self <= 2.0:
        raise RuntimeError("The self-conjugate angle is outside the current scan")
    theta_self = float(np.rad2deg(np.arccos(1.0 - q_self)))

    results: dict[str, dict[int, dict[str, object]]] = {}
    metrics: dict[str, object] = {
        "source": str(cli.csi_file),
        "frame": cli.frame,
        "shape": list(raw_csi.shape),
        "f0_hz": args.f_0,
        "antenna_spacing_m": args.antenna_spacing,
        "wavelength_m": wavelength,
        "q_self": q_self,
        "theta_self_conjugate_deg": theta_self,
        "antenna_order": archive["antenna_order"].tolist(),
        "loss_rate": archive["loss_rate"].tolist(),
        "estimators": {},
    }

    phase_metrics, phase_traces = adjacent_phase_diagnostics(raw_csi)
    metrics["adjacent_rx_complex_phase"] = phase_metrics

    for name, data in inputs.items():
        results[name] = {}
        metrics["estimators"][name] = {}
        for tx_idx in range(min(2, args.num_Tx)):
            theta, fd, spectrum_db, covariance, metadata = run_estimator(
                data,
                args,
                cli.frame,
                tx_idx,
            )
            current_metrics = spectrum_metrics(
                theta,
                fd,
                spectrum_db,
                covariance,
                theta_self,
            )
            alias_metrics = conjugate_alias_error(
                theta,
                spectrum_db,
                wavelength,
                args.antenna_spacing,
            )
            current_metrics["conjugate_alias_identity"] = alias_metrics
            results[name][tx_idx] = {
                "theta": theta,
                "fd": fd,
                "spectrum_db": spectrum_db,
                "covariance": covariance,
                "metadata": metadata,
                "metrics": current_metrics,
            }
            metrics["estimators"][name][f"tx_{tx_idx}"] = {
                "metadata": metadata,
                **current_metrics,
            }

    for name in ("Amplitude residual", "Power residual (current)"):
        for tx_idx in results[name]:
            ratio = results[name][tx_idx]["metrics"]["covariance_imaginary_ratio"]
            if ratio > 1e-12:
                raise AssertionError(f"{name}, Tx {tx_idx} covariance is not real")

    save_ablation_heatmaps(
        cli.output_dir / "01_real_preprocessing_ablation.png",
        results,
        theta_self,
        cli.frame,
    )
    save_identity_figure(
        cli.output_dir / "02_spatial_alias_identity.png",
        results,
        wavelength,
        args.antenna_spacing,
        theta_self,
    )
    save_phase_stability_figure(
        cli.output_dir / "03_adjacent_rx_phase_stability.png",
        phase_traces,
    )

    metrics_path = cli.output_dir / "real_symmetry_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved diagnostics to: {cli.output_dir.resolve()}")


if __name__ == "__main__":
    main()
