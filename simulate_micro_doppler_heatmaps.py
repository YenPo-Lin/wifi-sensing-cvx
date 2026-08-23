"""Compare magnitude-only Azi-ToF and Doppler-gated Azi-ToF heatmaps."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MUSIC
import WIDFS
import pre_processing as pp


OUTPUT_DIR = Path(__file__).resolve().parent / "simulation_outputs" / "micro_doppler_weighting"


def simulation_args(num_frames, num_tx, num_rx, num_subcarriers):
    return SimpleNamespace(
        fs=100,
        f_0=5.57e9,
        BW=160e6,
        delta_f=2.54e6,
        antenna_spacing=0.02,
        projection="cos",
        num_frames=num_frames,
        num_Tx=num_tx,
        num_Rx=num_rx,
        num_subcarriers=num_subcarriers,
        avg_frames=32,
        Sdim=2,
        Sdim_energy_ratio=0.33,
        stream_win=5,
        stream_sample_range=num_rx,
        freq_win=32,
        freq_hop=2,
        freq_sample_range=num_subcarriers,
        freq_space=1,
        time_win=10,
        time_hop=2,
        time_sample_range=30,
        theta_min=0.0,
        theta_max=180.0,
        theta_step=3.0,
        tau_min=2e-9,
        tau_max=20e-9,
        tau_step=3e-10,
        doppler_min=-20.0,
        doppler_max=20.0,
        doppler_step=2.0,
        axis="ns",
        axis_flip=True,
        colorbar=True,
        pics_dir=None,
        azi_tof_dop_snapshot_norm=True,
        azi_tof_dop_epsilon=1e-8,
        widfs_band_half_width=2.0,
        widfs_weight_percentile=95.0,
        q_ref=1.0,
        # Robustness parameters
        noise_level=0.05,
        ch_gain_min=0.8, 
        ch_gain_max=1.2,
        strong_amplitude=0.3,
        weak_amplitude=0.1,
        rx_envelope_width=0.25, # lower for more deeper spatial fading
        sub_envelope_width=0.25,# lower for more deeper frequency fading
    )


def magnitude_target(args, theta_deg, tof_s, fd_hz, amplitude, envelope, time_s):
    """
    Create a real magnitude modulation matching the configured scan grids.
    # 將空間與時間相位加總後，取餘弦函數 np.cos(spatial_phase + temporal_phase)，而不是複數的指數函數
    # 模擬動態目標反射波與靜態環境反射波疊加後，在接收端產生的「純實數能量漣漪（Magnitude Ripple）」
    """
    rx_idx = np.arange(args.num_Rx)[:, None]
    sub_idx = np.arange(args.num_subcarriers)[None, :]
    theta_rad = np.deg2rad(theta_deg)

    azi_phase = (
        2.0
        * np.pi
        * args.f_0
        * (1.0 - np.cos(theta_rad))
        * args.antenna_spacing
        / 3e8
        * rx_idx
    )
    tof_phase = -2.0 * np.pi * args.delta_f * sub_idx * tof_s
    spatial_phase = azi_phase + tof_phase
    temporal_phase = 2.0 * np.pi * fd_hz * time_s[:, None, None]
    return amplitude * envelope[None, :, :] * np.cos(spatial_phase[None, :, :] + temporal_phase)


def generate_magnitude_csi(args, seed=7):
    """
    # Generate magnitude-only CSI data.
    # 模擬真實的室內干擾，在 generate_magnitude_csi 中對訊號進行了破壞。
    # 在 Rx 軸與 Subcarrier 軸上乘上了 Gaussian Envelopes，例如 weak_rx_envelope 與 weak_sub_envelope）。
    # 模擬頻率選擇性衰落與空間衰落。
    # 疊加真實世界的雜訊
    # (1) 非均勻通道增益 (Channel Gain)：給予每個 (Tx, Rx, Subcarrier) 獨立的靜態隨機增益（介於 0.8 到 1.2 之間）。
    # (2) 慢速靜態雜訊 (Slow Clutter)：加入了一個 0.7Hz 的低頻波動來模擬呼吸或微弱的環境飄移。
    # (3) 高斯白雜訊 (Gaussian Noise)：最後疊加隨機的標準常態分佈雜訊。
    """
    rng = np.random.default_rng(seed)
    time_s = np.arange(args.num_frames) / args.fs

    strong = {"theta": 40.0, "tof": 6e-9, "fd": 12.0, "amplitude": args.strong_amplitude}
    weak = {"theta": 75.0, "tof": 12e-9, "fd": +8.0, "amplitude": args.weak_amplitude}

    rx = np.arange(args.num_Rx)[:, None]
    sub = np.arange(args.num_subcarriers)[None, :]
    strong_rx_envelope = 0.05 + 0.95 * np.exp(-0.5 * ((rx - 1.5) / args.rx_envelope_width) ** 2)
    strong_sub_envelope = 0.05 + 0.95 * np.exp(-0.5 * ((sub - 8.0) / args.sub_envelope_width) ** 2)
    strong_envelope = strong_rx_envelope * strong_sub_envelope

    weak_rx_envelope = 0.05 + 0.95 * np.exp(-0.5 * ((rx - 5.5) / args.rx_envelope_width) ** 2)
    weak_sub_envelope = 0.05 + 0.95 * np.exp(-0.5 * ((sub - 23.0) / args.sub_envelope_width) ** 2)
    weak_envelope = weak_rx_envelope * weak_sub_envelope

    strong_component = magnitude_target(
        args,
        theta_deg=strong["theta"],
        tof_s=strong["tof"],
        fd_hz=strong["fd"],
        amplitude=strong["amplitude"],
        envelope=strong_envelope,
        time_s=time_s,
    )
    weak_component = magnitude_target(
        args,
        theta_deg=weak["theta"],
        tof_s=weak["tof"],
        fd_hz=weak["fd"],
        amplitude=weak["amplitude"],
        envelope=weak_envelope,
        time_s=time_s,
    )
    

    channel_gain = rng.uniform(
        args.ch_gain_min, args.ch_gain_max, size=(1, args.num_Tx, args.num_Rx, args.num_subcarriers)
    )
    baseline = 2.0 * channel_gain
    slow_clutter = 0.035 * np.sin(2.0 * np.pi * 0.7 * time_s)[:, None, None]
    noise = args.noise_level * rng.standard_normal(
        (args.num_frames, args.num_Tx, args.num_Rx, args.num_subcarriers)
    )

    strong_only_modulation = strong_component + slow_clutter
    weak_only_modulation = weak_component + slow_clutter
    modulation = strong_component + weak_component + slow_clutter
    strong_magnitude_csi = baseline * (1.0 + strong_only_modulation[:, None, :, :]) + noise
    weak_magnitude_csi = baseline * (1.0 + weak_only_modulation[:, None, :, :]) + noise
    magnitude_csi = baseline * (1.0 + modulation[:, None, :, :]) + noise
    strong_magnitude_csi = np.maximum(strong_magnitude_csi, 1e-3)
    weak_magnitude_csi = np.maximum(weak_magnitude_csi, 1e-3)
    magnitude_csi = np.maximum(magnitude_csi, 1e-3)
    components = {
        "strong": strong_magnitude_csi,
        "weak": weak_magnitude_csi,
    }
    return magnitude_csi, components, strong, weak


def normalize_magnitude(magnitude_csi, args):
    background = pp.MA(magnitude_csi, args.fs * 1.0)
    normalized = (magnitude_csi - background) / np.maximum(background, 1e-8)
    return background, normalized


def azi_tof_spectrum(csi, frame_idx, args):
    estimator = MUSIC.Azi_ToF(args)
    smoothed = estimator.cal_smoothed_csi(frame_idx, csi)
    covariance = estimator.cal_smoothed_cov(smoothed)
    tau, theta, spectrum_db = estimator.cal_spectrum(covariance)
    return theta, tau, spectrum_db


def azi_tof_doppler_cube(csi, frame_idx, args):
    estimator = MUSIC.Azi_ToF_Dop(args)
    theta, tau, fd, cube_db = estimator.cal_spectrum(csi, frame_idx)
    if cube_db is None:
        raise RuntimeError("Azi-ToF-Doppler spectrum could not be generated")
    return theta, tau, fd, cube_db


def project_doppler_gate(cube_db, fd_grid, center_fd, half_width, method="max"):
    mask = np.abs(fd_grid - center_fd) <= half_width
    if not np.any(mask):
        raise ValueError(f"No Doppler bins inside {center_fd:g} +/- {half_width:g} Hz")

    linear_cube = 10.0 ** (cube_db[:, :, mask] / 10.0)
    if method == "max":
        projected = np.max(linear_cube, axis=2)
    elif method == "sum":
        projected = np.sum(linear_cube, axis=2)
    else:
        raise ValueError(f"Unsupported projection method: {method}")
    return 10.0 * np.log10(projected + 1e-12)


def target_widfs_weight(magnitude_csi, background, frame_idx, target_fd, args):
    background_power = np.maximum(background**2, 1e-12)
    dynamic_power = magnitude_csi**2 - background_power
    filtered_power = WIDFS.apply_doppler_bandpass(dynamic_power, target_fd, args)

    window_size = 100
    start = int(np.clip(frame_idx - window_size // 2, 0, args.num_frames - window_size))
    end = start + window_size
    normalized_power = filtered_power[start:end] / background_power[start:end]
    raw_weight = WIDFS._channel_weights_from_norm_power(
        normalized_power, abs(target_fd), args
    )
    raw_weight = np.nan_to_num(raw_weight, nan=0.0, posinf=0.0, neginf=0.0)

    positive = raw_weight[raw_weight > 0]
    scale = np.percentile(positive, args.widfs_weight_percentile) if positive.size else 0.0
    if scale <= 0:
        return raw_weight, np.ones_like(raw_weight)
    return raw_weight, np.clip(raw_weight / scale, 0.0, 1.0)


def weak_target_widfs_weight(magnitude_csi, background, frame_idx, target_fd, args):
    return target_widfs_weight(magnitude_csi, background, frame_idx, target_fd, args)


def relative_db(spectrum_db):
    spectrum_db = np.asarray(spectrum_db, dtype=float)
    return spectrum_db - np.nanmax(spectrum_db)


def color_limits(values, vmin=-12.0, vmax=0.0):
    if vmin is not None and vmax is not None:
        return vmin, vmax

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 0.0

    auto_vmin = float(np.nanmin(finite)) if vmin is None else vmin
    auto_vmax = float(np.nanmax(finite)) if vmax is None else vmax
    if np.isclose(auto_vmin, auto_vmax):
        auto_vmin -= 0.5
        auto_vmax += 0.5
    return auto_vmin, auto_vmax


def draw_heatmap(ax, theta, tau, spectrum_db, title, strong, weak, vmin=-12.0, vmax=0.0):
    values = relative_db(spectrum_db).T
    plot_vmin, plot_vmax = color_limits(values, vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        theta,
        tau * 1e9,
        values,
        cmap="turbo",
        shading="auto",
        vmin=plot_vmin,
        vmax=plot_vmax,
    )
    ax.scatter(
        strong["theta"], strong["tof"] * 1e9,
        marker="o", facecolors="none", edgecolors="white", s=70, linewidths=1.5,
        label=f"Strong: {strong['fd']:+g} Hz",
    )
    ax.scatter(
        weak["theta"], weak["tof"] * 1e9,
        marker="x", color="white", s=65, linewidths=1.8,
        label=f"Weak: {weak['fd']:+g} Hz",
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("ToF (ns)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return mesh


def save_single(path, theta, tau, spectrum_db, title, strong, weak):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mesh = draw_heatmap(ax, theta, tau, spectrum_db, title, strong, weak)
    fig.colorbar(mesh, ax=ax, label="Relative spectrum (dB)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def target_pipeline(magnitude_csi, frame_idx, target_fd, args):
    background, csi_norm = normalize_magnitude(magnitude_csi, args)
    theta_2d, tau_2d, azi_tof_db = azi_tof_spectrum(csi_norm, frame_idx, args)
    theta_3d, tau_3d, fd_grid, cube_db = azi_tof_doppler_cube(
        csi_norm, frame_idx, args
    )
    gate_half_width = args.doppler_step
    unweighted_gate_db = project_doppler_gate(
        cube_db, fd_grid, target_fd, gate_half_width, method="max"
    )

    raw_weight, normalized_weight = target_widfs_weight(
        magnitude_csi, background, frame_idx, target_fd, args
    )
    weight_only_csi = csi_norm * np.sqrt(normalized_weight)[None, ...]
    theta_weight_only, tau_weight_only, fd_weight_only, weight_only_cube_db = (
        azi_tof_doppler_cube(weight_only_csi, frame_idx, args)
    )
    weight_only_gate_db = project_doppler_gate(
        weight_only_cube_db,
        fd_weight_only,
        target_fd,
        gate_half_width,
        method="max",
    )

    band_csi = WIDFS.apply_doppler_bandpass(csi_norm, target_fd, args)
    theta_band, tau_band, fd_band, band_cube_db = azi_tof_doppler_cube(
        band_csi, frame_idx, args
    )
    band_only_gate_db = project_doppler_gate(
        band_cube_db, fd_band, target_fd, gate_half_width, method="max"
    )

    weighted_csi = band_csi * np.sqrt(normalized_weight)[None, ...]
    theta_w, tau_w, fd_w, weighted_cube_db = azi_tof_doppler_cube(
        weighted_csi, frame_idx, args
    )
    weighted_gate_db = project_doppler_gate(
        weighted_cube_db, fd_w, target_fd, gate_half_width, method="max"
    )

    return {
        "background": background,
        "csi_norm": csi_norm,
        "azi_tof": (theta_2d, tau_2d, azi_tof_db),
        "unweighted": (theta_3d, tau_3d, unweighted_gate_db),
        "weight_only": (theta_weight_only, tau_weight_only, weight_only_gate_db),
        "band_only": (theta_band, tau_band, band_only_gate_db),
        "weighted": (theta_w, tau_w, weighted_gate_db),
        "fd_grid": fd_grid,
        "raw_weight": raw_weight,
        "normalized_weight": normalized_weight,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = simulation_args(num_frames=180, num_tx=1, num_rx=8, num_subcarriers=32)
    frame_idx = 90
    magnitude_csi, component_csi, strong, weak = generate_magnitude_csi(args)

    # =================================================
    print("\n=== Target Components Information ===")
    print(f"[Strong] Theta: {strong['theta']}°, ToF: {strong['tof']*1e9:.1f} ns, Doppler: {strong['fd']:+} Hz, Amp: {strong['amplitude']}")
    print(f"[Weak]   Theta: {weak['theta']}°, ToF: {weak['tof']*1e9:.1f} ns, Doppler: {weak['fd']:+} Hz, Amp: {weak['amplitude']}")
    print("=====================================\n")
    # =================================================

    weak_results = target_pipeline(magnitude_csi, frame_idx, weak["fd"], args)
    strong_results = target_pipeline(component_csi["strong"], frame_idx, strong["fd"], args)

    theta_2d, tau_2d, azi_tof_db = weak_results["azi_tof"]
    theta_3d, tau_3d, unweighted_gate_db = weak_results["unweighted"]
    theta_weight_only, tau_weight_only, weight_only_gate_db = weak_results["weight_only"]
    theta_band, tau_band, band_only_gate_db = weak_results["band_only"]
    theta_w, tau_w, weighted_gate_db = weak_results["weighted"]
    fd_grid = weak_results["fd_grid"]
    raw_weight = weak_results["raw_weight"]
    normalized_weight = weak_results["normalized_weight"]

    strong_theta_2d, strong_tau_2d, strong_azi_tof_db = strong_results["azi_tof"]
    strong_theta_3d, strong_tau_3d, strong_unweighted_gate_db = strong_results["unweighted"]
    strong_theta_weight_only, strong_tau_weight_only, strong_weight_only_gate_db = (
        strong_results["weight_only"]
    )
    strong_theta_band, strong_tau_band, strong_band_only_gate_db = strong_results["band_only"]
    strong_theta_w, strong_tau_w, strong_weighted_gate_db = strong_results["weighted"]

    save_single(
        OUTPUT_DIR / "01_azi_tof.png",
        theta_2d, tau_2d, azi_tof_db,
        "Azi-ToF: all motion", strong, weak,
    )
    save_single(
        OUTPUT_DIR / "02_azi_tof_doppler_weak_gate.png",
        theta_3d, tau_3d, unweighted_gate_db,
        f"Azi-ToF-Doppler: {weak['fd']:+g} Hz gate", strong, weak,
    )
    save_single(
        OUTPUT_DIR / "03_widfs_weighted_azi_tof_doppler_weak_gate.png",
        theta_w, tau_w, weighted_gate_db,
        f"Bandpass + WIDFS Azi-ToF-Doppler: {weak['fd']:+g} Hz gate", strong, weak,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2), constrained_layout=True)
    comparison_items = [
        (
            axes[0, 0],
            theta_2d,
            tau_2d,
            azi_tof_db,
            "Mixed signal\nAzi-ToF all motion",
        ),
        (
            axes[0, 1],
            theta_3d,
            tau_3d,
            unweighted_gate_db,
            f"Mixed signal\nAzi-ToF-Doppler {weak['fd']:+g} Hz gate",
        ),
        (
            axes[0, 2],
            theta_w,
            tau_w,
            weighted_gate_db,
            f"Mixed signal\nBandpass + WIDFS {weak['fd']:+g} Hz gate",
        ),
        (
            axes[1, 0],
            strong_theta_2d,
            strong_tau_2d,
            strong_azi_tof_db,
            "Strong component\nAzi-ToF all motion",
        ),
        (
            axes[1, 1],
            strong_theta_3d,
            strong_tau_3d,
            strong_unweighted_gate_db,
            f"Strong component\nAzi-ToF-Doppler {strong['fd']:+g} Hz gate",
        ),
        (
            axes[1, 2],
            strong_theta_w,
            strong_tau_w,
            strong_weighted_gate_db,
            f"Strong component\nBandpass + WIDFS {strong['fd']:+g} Hz gate",
        ),
    ]
    for ax, theta, tau, spectrum, title in comparison_items:
        mesh = draw_heatmap(ax, theta, tau, spectrum, title, strong, weak, vmin=None, vmax=None)
        fig.colorbar(mesh, ax=ax, label="Relative spectrum (dB)", shrink=0.88)
    fig.savefig(OUTPUT_DIR / "04_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9.2), constrained_layout=True)
    ablation_items = [
        (axes[0, 0], theta_3d, tau_3d, unweighted_gate_db, "Mixed signal\nNo preprocessing"),
        (
            axes[0, 1],
            theta_weight_only,
            tau_weight_only,
            weight_only_gate_db,
            "Mixed signal\nWIDFS weight only",
        ),
        (
            axes[0, 2],
            theta_band,
            tau_band,
            band_only_gate_db,
            "Mixed signal\nTarget bandpass only",
        ),
        (
            axes[0, 3],
            theta_w,
            tau_w,
            weighted_gate_db,
            "Mixed signal\nTarget bandpass + WIDFS",
        ),
        (
            axes[1, 0],
            strong_theta_3d,
            strong_tau_3d,
            strong_unweighted_gate_db,
            "Strong component\nNo preprocessing",
        ),
        (
            axes[1, 1],
            strong_theta_weight_only,
            strong_tau_weight_only,
            strong_weight_only_gate_db,
            "Strong component\nWIDFS weight only",
        ),
        (
            axes[1, 2],
            strong_theta_band,
            strong_tau_band,
            strong_band_only_gate_db,
            "Strong component\nTarget bandpass only",
        ),
        (
            axes[1, 3],
            strong_theta_w,
            strong_tau_w,
            strong_weighted_gate_db,
            "Strong component\nTarget bandpass + WIDFS",
        ),
    ]
    for ax, theta, tau, spectrum, title in ablation_items:
        mesh = draw_heatmap(ax, theta, tau, spectrum, title, strong, weak, vmin=None, vmax=None)
        fig.colorbar(mesh, ax=ax, label="Relative spectrum (dB)", shrink=0.88)
    fig.savefig(OUTPUT_DIR / "05_widfs_ablation.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        OUTPUT_DIR / "simulation_results.npz",
        theta=theta_3d,
        tau_ns=tau_3d * 1e9,
        fd=fd_grid,
        azi_tof_db=azi_tof_db,
        azi_tof_doppler_weak_gate_db=unweighted_gate_db,
        widfs_weight_only_weak_gate_db=weight_only_gate_db,
        target_bandpass_only_weak_gate_db=band_only_gate_db,
        weighted_azi_tof_doppler_weak_gate_db=weighted_gate_db,
        strong_azi_tof_db=strong_azi_tof_db,
        strong_azi_tof_doppler_gate_db=strong_unweighted_gate_db,
        strong_widfs_weight_only_gate_db=strong_weight_only_gate_db,
        strong_target_bandpass_only_gate_db=strong_band_only_gate_db,
        strong_weighted_azi_tof_doppler_gate_db=strong_weighted_gate_db,
        widfs_raw_weight=raw_weight,
        widfs_normalized_weight=normalized_weight,
        strong_widfs_raw_weight=strong_results["raw_weight"],
        strong_widfs_normalized_weight=strong_results["normalized_weight"],
    )

    def peak_location(theta, tau, spectrum):
        azi_idx, tof_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        return float(theta[azi_idx]), float(tau[tof_idx] * 1e9)

    print(f"Saved simulation outputs to: {OUTPUT_DIR}")
    print("Azi-ToF peak (deg, ns):", peak_location(theta_2d, tau_2d, azi_tof_db))
    print(
        "Unweighted weak-gate peak (deg, ns):",
        peak_location(theta_3d, tau_3d, unweighted_gate_db),
    )
    print(
        "WIDFS-weighted weak-gate peak (deg, ns):",
        peak_location(theta_w, tau_w, weighted_gate_db),
    )
    print(
        "WIDFS nonzero-channel fraction:",
        float(np.mean(normalized_weight > 0)),
    )


if __name__ == "__main__":
    main()
