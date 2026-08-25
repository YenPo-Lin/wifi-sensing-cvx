"""Compare magnitude-only Azi-ToF and Doppler-gated Azi-ToF heatmaps."""

from copy import copy
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Doppler_spec
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
        time_win=20,
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
        roi_gate_half_width=2.0,
        axis="ns",
        axis_flip=True,
        colorbar=True,
        pics_dir=None,
        azi_tof_dop_snapshot_norm=True,
        azi_tof_dop_epsilon=1e-8,
        widfs_band_half_width=2.0,
        widfs_weight_percentile=95.0,
        q_ref=1.0,
        cfar_training_cells=2,
        cfar_guard_cells=1,
        cfar_threshold_factor=1.05,
        cfar_top_k=5,
        cfar_min_peak_distance=1,
        cfar_min_prominence_db=0.0,
        # Robustness parameters
        noise_level=0.05,
        ch_gain_min=0.8,
        ch_gain_max=1.2,
        strong_amplitude=0.1,
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

    azi_phase = (2.0* np.pi* args.f_0* (1.0 - np.cos(theta_rad))* args.antenna_spacing/ 3e8* rx_idx)
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
        "perfect": 1.0 + strong_component + weak_component,
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


def project_doppler_mask(cube_db, mask, method="max"):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != cube_db.shape[2]:
        raise ValueError("Doppler mask must match the last spectrum dimension")
    if not np.any(mask):
        raise ValueError("Doppler projection mask does not select any bins")

    linear_cube = 10.0 ** (cube_db[:, :, mask] / 10.0)
    if method == "max":
        projected = np.max(linear_cube, axis=2)
    elif method == "sum":
        projected = np.sum(linear_cube, axis=2)
    else:
        raise ValueError(f"Unsupported projection method: {method}")
    return 10.0 * np.log10(projected + 1e-12)


def project_doppler_gate(cube_db, fd_grid, center_fd, half_width, method="max"):
    mask = np.abs(fd_grid - center_fd) <= half_width
    if not np.any(mask):
        raise ValueError(f"No Doppler bins inside {center_fd:g} +/- {half_width:g} Hz")
    return project_doppler_mask(cube_db, mask, method=method)


def doppler_roi_args(args, center_fd, half_width):
    """Copy args and restrict the 3D MUSIC Doppler scan to the target RoI bins."""
    fd_grid = np.arange(
        args.doppler_min,
        args.doppler_max + 0.5 * args.doppler_step,
        args.doppler_step,
    )
    selected = fd_grid[np.abs(fd_grid - center_fd) <= half_width]
    if selected.size == 0:
        raise ValueError(f"No Doppler bins inside {center_fd:g} +/- {half_width:g} Hz")

    roi_args = copy(args)
    roi_args.doppler_min = float(selected[0])
    roi_args.doppler_max = float(selected[-1])
    return roi_args


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

def select_amplitude_frame(magnitude_csi, frame_idx, tx_idx=0):
    """Return one CSI amplitude frame as (Rx, subcarrier)."""
    magnitude_csi = np.asarray(magnitude_csi)
    if frame_idx < 0 or frame_idx >= magnitude_csi.shape[0]:
        raise IndexError(
            f"frame_idx must be in [0, {magnitude_csi.shape[0] - 1}], got {frame_idx}"
        )

    if magnitude_csi.ndim == 4:
        if tx_idx < 0 or tx_idx >= magnitude_csi.shape[1]:
            raise IndexError(
                f"tx_idx must be in [0, {magnitude_csi.shape[1] - 1}], got {tx_idx}"
            )
        return magnitude_csi[frame_idx, tx_idx, :, :]

    if magnitude_csi.ndim == 3:
        if tx_idx != 0:
            raise ValueError("tx_idx must be 0 for CSI without a Tx dimension")
        return magnitude_csi[frame_idx, :, :]

    raise ValueError(
        "magnitude_csi must have shape (frame, Rx, subcarrier) or "
        f"(frame, Tx, Rx, subcarrier), got {magnitude_csi.shape}"
    )


def draw_amplitude_heatmap(ax, magnitude_csi, frame_idx, tx_idx=0, title=None, vmin=None, vmax=None):
    """
    畫出特定 frame 的 CSI 振幅分佈。
    X軸為 Subcarrier Index，Y軸為 Rx Index。

    Args:
        ax: matplotlib Axes 物件
        magnitude_csi: shape 為 (frame, Rx, subcarrier) 或
            (frame, Tx, Rx, subcarrier) 的陣列
        frame_idx: 要繪製的時間幀索引
        tx_idx: 發射天線索引 (預設為 0)
        title: 圖表標題
        vmin, vmax: 色階上下限
    """
    amplitude = select_amplitude_frame(magnitude_csi, frame_idx, tx_idx)

    num_rx, num_sub = amplitude.shape
    sc_idx = np.arange(num_sub)
    rx_idx = np.arange(num_rx)

    # 決定色階範圍
    if vmin is None:
        vmin = np.nanmin(amplitude)
    if vmax is None:
        vmax = np.nanmax(amplitude)

    # 使用 pcolormesh 繪製，保持與原本 heatmap 相同的視覺風格
    mesh = ax.pcolormesh(
        sc_idx,
        rx_idx,
        amplitude,
        cmap="turbo",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Rx Index")

    if title is None:
        title = f"Amplitude Heatmap (Frame {frame_idx})"
    ax.set_title(title)

    return mesh


def save_amplitude_comparison(
    path,
    perfect_csi,
    imperfect_csi,
    weighted_csi,
    frame_idx,
    tx_idx=0,
):
    """Save perfect / imperfect / weighted CSI amplitude heatmaps at one frame."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    items = [
        ("Perfect CSI", perfect_csi),
        ("Imperfect CSI", imperfect_csi),
        ("Weighted CSI", weighted_csi),
    ]
    for ax, (title, csi) in zip(axes, items):
        mesh = draw_amplitude_heatmap(
            ax,
            csi,
            frame_idx,
            tx_idx=tx_idx,
            title=f"{title} @ frame {frame_idx}",
        )
        fig.colorbar(mesh, ax=ax, label="Amplitude", shrink=0.88)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_single(path, theta, tau, spectrum_db, title, strong, weak):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mesh = draw_heatmap(ax, theta, tau, spectrum_db, title, strong, weak)
    fig.colorbar(mesh, ax=ax, label="Relative spectrum (dB)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def half_height_width(axis, values, peak_idx):
    """Return peak width at half height above the profile floor."""
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size < 2 or peak_idx < 0 or peak_idx >= values.size:
        return np.nan, True

    peak_value = values[peak_idx]
    floor = np.nanmin(values)
    threshold = floor + 0.5 * (peak_value - floor)

    left_idx = peak_idx
    while left_idx > 0 and values[left_idx] >= threshold:
        left_idx -= 1
    left_truncated = left_idx == 0 and values[left_idx] >= threshold
    if left_truncated:
        left_crossing = axis[0]
    else:
        x0, x1 = axis[left_idx], axis[left_idx + 1]
        y0, y1 = values[left_idx], values[left_idx + 1]
        left_crossing = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0 + 1e-12)

    right_idx = peak_idx
    last_idx = values.size - 1
    while right_idx < last_idx and values[right_idx] >= threshold:
        right_idx += 1
    right_truncated = right_idx == last_idx and values[right_idx] >= threshold
    if right_truncated:
        right_crossing = axis[-1]
    else:
        x0, x1 = axis[right_idx - 1], axis[right_idx]
        y0, y1 = values[right_idx - 1], values[right_idx]
        right_crossing = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0 + 1e-12)

    return float(right_crossing - left_crossing), left_truncated or right_truncated


def estimate_2d_metrics(theta, tau, spectrum_db):
    spectrum_db = np.asarray(spectrum_db, dtype=float)
    azi_idx, tof_idx = np.unravel_index(np.nanargmax(spectrum_db), spectrum_db.shape)
    linear = 10.0 ** ((spectrum_db - np.nanmax(spectrum_db)) / 10.0)
    azi_width, azi_truncated = half_height_width(
        theta, linear[:, tof_idx], azi_idx
    )
    tof_width, tof_truncated = half_height_width(
        tau * 1e9, linear[azi_idx, :], tof_idx
    )

    exclusion = (
        (np.abs(theta[:, None] - theta[azi_idx]) <= 2.0 * np.median(np.diff(theta)))
        & (
            np.abs(tau[None, :] - tau[tof_idx])
            <= 2.0 * np.median(np.diff(tau))
        )
    )
    relative_db = spectrum_db - np.nanmax(spectrum_db)
    sidelobes = relative_db[~exclusion]
    psr_db = -float(np.nanmax(sidelobes)) if sidelobes.size else np.nan
    return {
        "azi": float(theta[azi_idx]),
        "tof_ns": float(tau[tof_idx] * 1e9),
        "azi_width_deg": azi_width,
        "tof_width_ns": tof_width,
        "azi_width_truncated": azi_truncated,
        "tof_width_truncated": tof_truncated,
        "psr_db": psr_db,
    }


def estimate_3d_roi_metrics(theta, tau, fd_grid, cube_db, target_fd, half_width):
    fd_mask = np.abs(fd_grid - target_fd) <= half_width
    selected_fd = fd_grid[fd_mask]
    selected_cube = cube_db[:, :, fd_mask]
    azi_idx, tof_idx, fd_idx = np.unravel_index(
        np.nanargmax(selected_cube), selected_cube.shape
    )

    linear_profile = np.max(10.0 ** (cube_db / 10.0), axis=(0, 1))
    full_fd_idx = int(np.flatnonzero(fd_mask)[fd_idx])
    fd_width, fd_width_truncated = half_height_width(
        fd_grid, linear_profile, full_fd_idx
    )
    return {
        "azi": float(theta[azi_idx]),
        "tof_ns": float(tau[tof_idx] * 1e9),
        "fd": float(selected_fd[fd_idx]),
        "fd_width_hz": fd_width,
        "fd_width_truncated": fd_width_truncated,
    }


def target_region_contrast(theta, tau, spectrum_db, target, other):
    def local_peak(component):
        mask = (
            (np.abs(theta[:, None] - component["theta"]) <= 6.0)
            & (np.abs(tau[None, :] - component["tof"]) <= 0.9e-9)
        )
        return float(np.nanmax(spectrum_db[mask]))

    return local_peak(target) - local_peak(other)


def format_width(value, unit, truncated):
    prefix = ">=" if truncated else ""
    return f"{prefix}{value:.2f} {unit}"


def associate_detected_fd(peak_fd, target_fd, tolerance_hz):
    peak_fd = np.asarray(peak_fd, dtype=float)
    if peak_fd.size == 0:
        return None
    nearest_idx = int(np.argmin(np.abs(peak_fd - target_fd)))
    if abs(peak_fd[nearest_idx] - target_fd) > tolerance_hz:
        return None
    return float(peak_fd[nearest_idx])


def print_target_diagnostics(
    label,
    target,
    other,
    common_cube,
    roi_map,
    band_map,
    weighted_map,
    raw_weight,
    normalized_weight,
    gate_half_width,
    detected_fd,
    roi_center_fd=None,
):
    theta_cube, tau_cube, fd_grid, cube_db = common_cube
    roi_center_fd = target["fd"] if roi_center_fd is None else float(roi_center_fd)
    metrics_3d = estimate_3d_roi_metrics(
        theta_cube,
        tau_cube,
        fd_grid,
        cube_db,
        roi_center_fd,
        gate_half_width,
    )
    roi_mask = np.abs(fd_grid - roi_center_fd) <= gate_half_width
    roi_bins = fd_grid[roi_mask]

    positive_raw = raw_weight[raw_weight > 0]
    confidence = (
        float(np.percentile(positive_raw, 90.0)) if positive_raw.size else 0.0
    )
    weight_sum = float(np.sum(normalized_weight))
    effective_channels = weight_sum**2 / (
        float(np.sum(normalized_weight**2)) + 1e-12
    )
    rx_scores = np.mean(normalized_weight, axis=(0, 2))
    subcarrier_scores = np.mean(normalized_weight, axis=(0, 1))

    print(f"\n--- {label} target diagnostics ---")
    print(
        "Ground truth: "
        f"azi={target['theta']:.1f} deg, ToF={target['tof'] * 1e9:.1f} ns, "
        f"Doppler={target['fd']:+.1f} Hz, amplitude={target['amplitude']:.3f}"
    )
    print(
        "3D RoI peak: "
        f"azi={metrics_3d['azi']:.1f} deg, ToF={metrics_3d['tof_ns']:.1f} ns, "
        f"Doppler={metrics_3d['fd']:+.1f} Hz, "
        "Doppler width="
        + format_width(
            metrics_3d["fd_width_hz"],
            "Hz",
            metrics_3d["fd_width_truncated"],
        )
    )
    detected_fd_text = "not detected" if detected_fd is None else f"{detected_fd:+.1f} Hz"
    print(
        f"Estimated fd (ToF-Doppler CFAR)={detected_fd_text}; "
        f"fd RoI accumulation center={roi_center_fd:+.1f} Hz, "
        f"range=[{roi_bins[0]:+.1f}, {roi_bins[-1]:+.1f}] Hz, "
        f"bins={np.array2string(roi_bins, precision=1, separator=', ')}"
    )

    for method_name, (theta, tau, spectrum_db) in (
        ("RoI gate sum", roi_map),
        ("Target bandpass", band_map),
        ("Bandpass + WIDFS", weighted_map),
    ):
        metrics = estimate_2d_metrics(theta, tau, spectrum_db)
        contrast = target_region_contrast(
            theta, tau, spectrum_db, target, other
        )
        print(
            f"{method_name:18s}: "
            f"azi={metrics['azi']:.1f} deg "
            f"(err={metrics['azi'] - target['theta']:+.1f}), "
            f"ToF={metrics['tof_ns']:.1f} ns "
            f"(err={metrics['tof_ns'] - target['tof'] * 1e9:+.1f}), "
            "width=("
            + format_width(
                metrics["azi_width_deg"],
                "deg",
                metrics["azi_width_truncated"],
            )
            + ", "
            + format_width(
                metrics["tof_width_ns"],
                "ns",
                metrics["tof_width_truncated"],
            )
            + f"), PSR={metrics['psr_db']:.3f} dB, "
            f"target-other={contrast:+.3f} dB"
        )

    print(
        "WIDFS weight:  "
        f"confidence(P90 raw)={confidence:.4g}, "
        f"normalized mean={np.mean(normalized_weight):.3f}, "
        f"median={np.median(normalized_weight):.3f}, "
        f"nonzero={np.mean(normalized_weight > 0):.1%}, "
        f"effective channels={effective_channels:.1f}/{normalized_weight.size}"
    )
    print(
        "Weight maxima: "
        f"Rx={int(np.argmax(rx_scores))} (mean={np.max(rx_scores):.3f}), "
        f"subcarrier={int(np.argmax(subcarrier_scores))} "
        f"(mean={np.max(subcarrier_scores):.3f})"
    )


def prepare_base_analysis(magnitude_csi, frame_idx, args):
    """Compute target-independent products once for one CSI dataset."""
    background, csi_norm = normalize_magnitude(magnitude_csi, args)
    theta_2d, tau_2d, azi_tof_db = azi_tof_spectrum(csi_norm, frame_idx, args)
    theta_3d, tau_3d, fd_grid, cube_db = azi_tof_doppler_cube(
        csi_norm, frame_idx, args
    )
    all_doppler_db = project_doppler_mask(
        cube_db, np.ones_like(fd_grid, dtype=bool), method="sum"
    )
    return {
        "background": background,
        "csi_norm": csi_norm,
        "cube": (theta_3d, tau_3d, fd_grid, cube_db),
        "azi_tof": (theta_2d, tau_2d, azi_tof_db),
        "all_doppler": (theta_3d, tau_3d, all_doppler_db),
    }


def target_pipeline(
    magnitude_csi,
    frame_idx,
    target_fd,
    args,
    base_analysis=None,
):
    """Build target-specific 3D MUSIC projections while reusing the shared cube."""
    if base_analysis is None:
        base_analysis = prepare_base_analysis(magnitude_csi, frame_idx, args)

    background = base_analysis["background"]
    csi_norm = base_analysis["csi_norm"]
    theta_3d, tau_3d, fd_grid, cube_db = base_analysis["cube"]
    gate_half_width = float(
        getattr(args, "roi_gate_half_width", args.doppler_step)
    )
    roi_accumulation_db = project_doppler_gate(
        cube_db, fd_grid, target_fd, gate_half_width, method="sum"
    )
    unweighted_gate_db = project_doppler_gate(
        cube_db, fd_grid, target_fd, gate_half_width, method="max"
    )

    raw_weight, normalized_weight = target_widfs_weight(
        magnitude_csi, background, frame_idx, target_fd, args
    )
    roi_args = doppler_roi_args(args, target_fd, gate_half_width)

    band_csi = WIDFS.apply_doppler_bandpass(csi_norm, target_fd, args)
    theta_band, tau_band, fd_band, band_cube_db = azi_tof_doppler_cube(
        band_csi, frame_idx, roi_args
    )
    band_only_db = project_doppler_gate(
        band_cube_db, fd_band, target_fd, gate_half_width, method="max"
    )
    band_roi_accumulation_db = project_doppler_gate(
        band_cube_db, fd_band, target_fd, gate_half_width, method="sum"
    )

    weighted_csi = band_csi * np.sqrt(normalized_weight)[None, ...]
    theta_w, tau_w, fd_w, weighted_cube_db = azi_tof_doppler_cube(
        weighted_csi, frame_idx, roi_args
    )
    weighted_db = project_doppler_gate(
        weighted_cube_db, fd_w, target_fd, gate_half_width, method="max"
    )
    weighted_roi_accumulation_db = project_doppler_gate(
        weighted_cube_db, fd_w, target_fd, gate_half_width, method="sum"
    )

    return {
        **base_analysis,
        "roi_accumulation": (theta_3d, tau_3d, roi_accumulation_db),
        "unweighted": (theta_3d, tau_3d, unweighted_gate_db),
        "band_only": (theta_band, tau_band, band_only_db),
        "band_roi_accumulation": (theta_band, tau_band, band_roi_accumulation_db),
        "weighted": (theta_w, tau_w, weighted_db),
        "weighted_roi_accumulation": (theta_w, tau_w, weighted_roi_accumulation_db),
        "fd_grid": fd_grid,
        "band_fd_grid": fd_band,
        "weighted_fd_grid": fd_w,
        "target_fd": float(target_fd),
        "gate_half_width": gate_half_width,
        "raw_weight": raw_weight,
        "normalized_weight": normalized_weight,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = simulation_args(num_frames=180, num_tx=1, num_rx=8, num_subcarriers=32)
    frame_idx = 90
    magnitude_csi, component_csi, strong, weak = generate_magnitude_csi(args)

    print("\n=== Target Components Information ===")
    print(f"[Strong] Theta: {strong['theta']}°, ToF: {strong['tof']*1e9:.1f} ns, Doppler: {strong['fd']:+} Hz, Amp: {strong['amplitude']}")
    print(f"[Weak]   Theta: {weak['theta']}°, ToF: {weak['tof']*1e9:.1f} ns, Doppler: {weak['fd']:+} Hz, Amp: {weak['amplitude']}")
    print("3D MUSIC calls: 5 total = 1 mixed full-grid + 4 target-RoI cubes")
    print("=====================================\n")

    mixed_base = prepare_base_analysis(magnitude_csi, frame_idx, args)
    detection_csi = mixed_base["csi_norm"]
    original_pics_dir = args.pics_dir
    args.pics_dir = str(OUTPUT_DIR)
    try:
        detected_result = Doppler_spec.gen_spectrum_from_ToF_Doppler(
            detection_csi,
            frame_idx,
            args,
            method="max",
        )
    finally:
        args.pics_dir = original_pics_dir

    if detected_result is None:
        detected_peak_fd = np.array([], dtype=float)
        detected_peak_db = np.array([], dtype=float)
    else:
        detected_peak_fd, detected_peak_db = detected_result

    association_tolerance = max(
        float(args.roi_gate_half_width), float(args.doppler_step)
    )
    weak_detected_fd = associate_detected_fd(
        detected_peak_fd, weak["fd"], association_tolerance
    )
    strong_detected_fd = associate_detected_fd(
        detected_peak_fd, strong["fd"], association_tolerance
    )

    # Keep the controlled ablation centered near the known simulated component.
    # End-to-end detector results are reported separately above and below.
    fd_error_weak = +0.2
    fd_error_strong = -0.6  # 強目標誤差（可視需求調整）
    weak_results = target_pipeline(
        magnitude_csi,
        frame_idx,
        weak["fd"] + fd_error_weak,
        args,
        base_analysis=mixed_base,
    )
    strong_target_results = target_pipeline(
        magnitude_csi,
        frame_idx,
        strong["fd"] + fd_error_strong,
        args,
        base_analysis=mixed_base,
    )

    theta_2d, tau_2d, azi_tof_db = weak_results["azi_tof"]
    theta_3d, tau_3d, unweighted_gate_db = weak_results["unweighted"]
    theta_band, tau_band, band_only_gate_db = weak_results["band_only"]
    theta_w, tau_w, weighted_gate_db = weak_results["weighted"]
    fd_grid = weak_results["fd_grid"]
    raw_weight = weak_results["raw_weight"]
    normalized_weight = weak_results["normalized_weight"]

    strong_theta_3d, strong_tau_3d, strong_unweighted_gate_db = (
        strong_target_results["unweighted"]
    )
    strong_theta_w, strong_tau_w, strong_weighted_gate_db = (
        strong_target_results["weighted"]
    )

    perfect_csi = component_csi["perfect"]
    imperfect_csi = magnitude_csi
    weighted_csi = imperfect_csi * np.sqrt(normalized_weight)[None, ...]
    save_amplitude_comparison(
        OUTPUT_DIR / "00_amplitude_comparison.png",
        perfect_csi,
        imperfect_csi,
        weighted_csi,
        frame_idx,
    )


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
        f"Bandpass + WIDFS Azi-ToF: {weak['fd']:+g} Hz", strong, weak,
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
            theta_2d,
            tau_2d,
            azi_tof_db,
            "Strong target\nAzi-ToF all motion",
        ),
        (
            axes[1, 1],
            strong_theta_3d,
            strong_tau_3d,
            strong_unweighted_gate_db,
            f"Strong target\nAzi-ToF-Doppler {strong['fd']:+g} Hz gate",
        ),
        (
            axes[1, 2],
            strong_theta_w,
            strong_tau_w,
            strong_weighted_gate_db,
            f"Strong target\nBandpass + WIDFS {strong['fd']:+g} Hz",
        ),
    ]
    for ax, theta, tau, spectrum, title in comparison_items:
        mesh = draw_heatmap(ax, theta, tau, spectrum, title, strong, weak, vmin=None, vmax=None)
        fig.colorbar(mesh, ax=ax, label="Relative spectrum (dB)", shrink=0.88)
    fig.savefig(OUTPUT_DIR / "04_comparison.png", dpi=180)
    plt.close(fig)

    theta_all, tau_all, all_doppler_db = weak_results["all_doppler"]
    theta_weak_roi, tau_weak_roi, weak_roi_db = weak_results["roi_accumulation"]
    theta_weak_band, tau_weak_band, weak_band_roi_db = (
        weak_results["band_roi_accumulation"]
    )
    theta_weak_weighted, tau_weak_weighted, weak_weighted_roi_db = (
        weak_results["weighted_roi_accumulation"]
    )
    theta_strong_roi, tau_strong_roi, strong_roi_db = (
        strong_target_results["roi_accumulation"]
    )
    theta_strong_band, tau_strong_band, strong_band_roi_db = (
        strong_target_results["band_roi_accumulation"]
    )
    theta_strong_weighted, tau_strong_weighted, strong_weighted_roi_db = (
        strong_target_results["weighted_roi_accumulation"]
    )

    fig, axes = plt.subplots(2, 5, figsize=(27.5, 9.2), constrained_layout=True)
    ablation_items = [
        (axes[0, 0], theta_2d, tau_2d, azi_tof_db, "Weak target\nAzi-ToF"),
        (
            axes[0, 1],
            theta_all,
            tau_all,
            all_doppler_db,
            "Weak target\nAzi-ToF-Doppler (all fd sum)",
        ),
        (
            axes[0, 2],
            theta_weak_roi,
            tau_weak_roi,
            weak_roi_db,
            f"Weak target\n+ RoI gate sum ({weak['fd']:+g} Hz)",
        ),
        (
            axes[0, 3],
            theta_weak_band,
            tau_weak_band,
            weak_band_roi_db,
            "Weak target\n+ Target bandpass",
        ),
        (
            axes[0, 4],
            theta_weak_weighted,
            tau_weak_weighted,
            weak_weighted_roi_db,
            "Weak target\n+ Target bandpass + WIDFS",
        ),
        (axes[1, 0], theta_2d, tau_2d, azi_tof_db, "Strong target\nAzi-ToF"),
        (
            axes[1, 1],
            theta_all,
            tau_all,
            all_doppler_db,
            "Strong target\nAzi-ToF-Doppler (all fd sum)",
        ),
        (
            axes[1, 2],
            theta_strong_roi,
            tau_strong_roi,
            strong_roi_db,
            f"Strong target\n+ RoI gate sum ({strong['fd']:+g} Hz)",
        ),
        (
            axes[1, 3],
            theta_strong_band,
            tau_strong_band,
            strong_band_roi_db,
            "Strong target\n+ Target bandpass",
        ),
        (
            axes[1, 4],
            theta_strong_weighted,
            tau_strong_weighted,
            strong_weighted_roi_db,
            "Strong target\n+ Target bandpass + WIDFS",
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
        target_bandpass_only_weak_gate_db=band_only_gate_db,
        weighted_azi_tof_doppler_weak_gate_db=weighted_gate_db,
        strong_azi_tof_db=azi_tof_db,
        strong_azi_tof_doppler_gate_db=strong_unweighted_gate_db,
        strong_weighted_azi_tof_doppler_gate_db=strong_weighted_gate_db,
        widfs_raw_weight=raw_weight,
        widfs_normalized_weight=normalized_weight,
        strong_widfs_raw_weight=strong_target_results["raw_weight"],
        strong_widfs_normalized_weight=strong_target_results["normalized_weight"],
        azi_tof_doppler_all_fd_sum_db=all_doppler_db,
        weak_roi_gate_sum_db=weak_roi_db,
        weak_target_bandpass_roi_sum_db=weak_band_roi_db,
        weak_target_bandpass_widfs_roi_sum_db=weak_weighted_roi_db,
        strong_roi_gate_sum_db=strong_roi_db,
        strong_target_bandpass_roi_sum_db=strong_band_roi_db,
        strong_target_bandpass_widfs_roi_sum_db=strong_weighted_roi_db,
        strong_target_widfs_raw_weight=strong_target_results["raw_weight"],
        strong_target_widfs_normalized_weight=(
            strong_target_results["normalized_weight"]
        ),
        detected_peak_fd=detected_peak_fd,
        detected_peak_db=detected_peak_db,
    )

    print(f"Saved simulation outputs to: {OUTPUT_DIR}")
    print("\n=== Shared baseline diagnostics ===")
    print(
        f"Frame={frame_idx}, fs={args.fs:g} Hz, "
        f"Doppler grid={args.doppler_min:g}:{args.doppler_step:g}:"
        f"{args.doppler_max:g} Hz, RoI half-width={args.roi_gate_half_width:g} Hz, "
        f"bandpass half-width={args.widfs_band_half_width:g} Hz"
    )
    for method_name, theta, tau, spectrum_db in (
        ("Azi-ToF", theta_2d, tau_2d, azi_tof_db),
        ("Azi-ToF-Doppler all-fd sum", theta_all, tau_all, all_doppler_db),
    ):
        metrics = estimate_2d_metrics(theta, tau, spectrum_db)
        print(
            f"{method_name:29s}: azi={metrics['azi']:.1f} deg, "
            f"ToF={metrics['tof_ns']:.1f} ns, width=("
            + format_width(
                metrics["azi_width_deg"],
                "deg",
                metrics["azi_width_truncated"],
            )
            + ", "
            + format_width(
                metrics["tof_width_ns"],
                "ns",
                metrics["tof_width_truncated"],
            )
            + f"), PSR={metrics['psr_db']:.3f} dB"
        )

    print_target_diagnostics(
        "Weak",
        weak,
        strong,
        weak_results["cube"],
        weak_results["roi_accumulation"],
        weak_results["band_roi_accumulation"],
        weak_results["weighted_roi_accumulation"],
        weak_results["raw_weight"],
        weak_results["normalized_weight"],
        args.roi_gate_half_width,
        weak_detected_fd,
        roi_center_fd=weak_results["target_fd"],
    )
    print_target_diagnostics(
        "Strong",
        strong,
        weak,
        strong_target_results["cube"],
        strong_target_results["roi_accumulation"],
        strong_target_results["band_roi_accumulation"],
        strong_target_results["weighted_roi_accumulation"],
        strong_target_results["raw_weight"],
        strong_target_results["normalized_weight"],
        args.roi_gate_half_width,
        strong_detected_fd,
        roi_center_fd=strong_target_results["target_fd"],
    )
    print(
        "\nWidth definition: half-height above each profile floor; "
        "'>=' means the width reaches a scan boundary."
    )


if __name__ == "__main__":
    main()
