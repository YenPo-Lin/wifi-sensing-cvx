import numpy as np
from scipy.signal import butter, filtfilt
import pre_processing as pp
import MUSIC
import matplotlib.pyplot as plt
import Doppler_spec
import time

def extract_dynamic_power(raw_CSI, args):
    raw_CSI = np.abs(raw_CSI)  # take absolute value to get power
    # u_{i,j}
    background = pp.MA(raw_CSI, args.fs * 1.0)**2 
    # v_{i,j,k}
    dynamic_power = raw_CSI**2 - background 
    return background, dynamic_power

def apply_doppler_bandpass(dynamic_power, target_fd, args):
    """
    利用 scipy.signal 實作時間軸上的帶通濾波器
    """
    # 根據論文，帶通濾波器的頻寬為 |f^D| ± \Delta f (論文中 \Delta f 設為 10 Hz)
    delta_f = float(getattr(args, "widfs_band_half_width", 10.0))
    if delta_f <= 0:
        raise ValueError(f"widfs_band_half_width must be positive, got {delta_f}")

    lowcut = max(0.1, abs(target_fd) - delta_f) # 避免低頻截斷為 0
    highcut = min(args.fs / 2 - 0.1, abs(target_fd) + delta_f) # 不可超過 Nyquist 頻率
    if lowcut >= highcut:
        raise ValueError(
            f"Invalid WIDFS band for fd={target_fd:g} Hz: "
            f"lowcut={lowcut:g}, highcut={highcut:g}"
        )
    
    nyq = 0.5 * args.fs
    b, a = butter(N=3, Wn=[lowcut / nyq, highcut / nyq], btype='bandpass')
    filtered_power = filtfilt(b, a, dynamic_power, axis=0)
    return filtered_power

def filtered_dyn_power_seq(raw_CSI, target_fd, args):
    background, dynamic_power = extract_dynamic_power(raw_CSI, args)
    
    # v'_{i,j,k}
    filtered_power = apply_doppler_bandpass(dynamic_power, target_fd, args)
    
    # Normalize: v'_{i,j,k} / u_{i,j}
    norm_filtered_power = filtered_power / (background + 1e-8) 
    return norm_filtered_power

def doppler_fitting(norm_filtered_power, fd, args):
    """
    根據論文 Eq.(29) 實作矩陣最小平方擬合
    輸入的 norm_filtered_power 必須是切好 window 的資料 (例如 shape: 100, 2, 8, 64)
    """
    N_p = norm_filtered_power.shape[0] # 時間窗口大小，例如 100
    t = np.arange(N_p) / args.fs       # 時間向量 t_k
    
    # 1. 建立方程式 (29) 中的觀測矩陣 A (Shape: N_p x 2)
    A = np.column_stack([
        np.cos(2 * np.pi * abs(fd) * t),
        np.sin(2 * np.pi * abs(fd) * t)
    ])
    
    # 2. 為了支援所有硬體維度 (TX, RX, Subcarriers) 的高速平行運算，將空間與頻率維度攤平
    original_shape = norm_filtered_power.shape
    v_flattened = norm_filtered_power.reshape(N_p, -1) # Shape: (N_p, N_features)
    
    # 3. 最小平方法求解
    # 利用 Pseudo-inverse (A^+) 一次性解出所有天線與子載波的 x, y
    # A_pinv shape: (2, N_p), 乘上 v_flattened (N_p, N_features) -> (2, N_features)
    A_pinv = np.linalg.pinv(A)
    coeffs = A_pinv @ v_flattened
    
    # 4. 將解出來的 x, y 重新塑形回原本的硬體維度，如 (2, 8, 64)
    cos_coeffs = coeffs[0].reshape(original_shape[1:])
    sin_coeffs = coeffs[1].reshape(original_shape[1:])
    return cos_coeffs, sin_coeffs

def _channel_weights_from_norm_power(norm_dyn_power, center_dfs_hz, args):
    """Estimate WIDFS channel weights from a local normalized power sequence."""
    cos_coeffs, sin_coeffs = doppler_fitting(
        norm_dyn_power, center_dfs_hz, args
    )
    dyn_intensity = np.sqrt(cos_coeffs**2 + sin_coeffs**2)

    N_time = norm_dyn_power.shape[0]
    t = np.arange(N_time) / args.fs
    t_expand = t.reshape((N_time,) + (1,) * (norm_dyn_power.ndim - 1))
    angular_frequency = 2 * np.pi * abs(center_dfs_hz)
    pred_dyn_power = (
        cos_coeffs * np.cos(angular_frequency * t_expand)
        + sin_coeffs * np.sin(angular_frequency * t_expand)
    )

    residual_sq_sum = np.sum((norm_dyn_power - pred_dyn_power) ** 2, axis=0)
    mean_dyn_power = np.mean(norm_dyn_power, axis=0)
    total_sq_sum = np.sum((norm_dyn_power - mean_dyn_power) ** 2, axis=0)
    r2_score = 1.0 - residual_sq_sum / (total_sq_sum + 1e-8)
    r2_purity_score = np.maximum(r2_score, 0.0)

    ref_quality_metric = getattr(args, "q_ref", 1.0)
    return dyn_intensity * r2_purity_score * ref_quality_metric

def weight_gen(raw_csi, center_dfs_hz, args):
    # ==========================================
    # 1. Target Dynamic Intensity Extraction
    # ==========================================
    # 1-1. 取得標準化且濾波後的動態功率 r^D_{i,j,k}
    norm_dyn_power = filtered_dyn_power_seq(raw_csi, center_dfs_hz, args)
    
    # 1-2. 透過最小平方法萃取正弦與餘弦擬合係數 x_{i,j}, y_{i,j}
    dfs_channel_weights = _channel_weights_from_norm_power(
        norm_dyn_power, center_dfs_hz, args
    )
    
    # 3-2. 生成加權後的動態能量 E^D_{i,j,k} (可依據需求決定是否在此步驟直接套用)
    # 由於 norm_dyn_power 已包含正規化振幅，我們直接套用權重
    weighted_dyn_power = dfs_channel_weights * (np.abs(norm_dyn_power)**2)
    
    return dfs_channel_weights, weighted_dyn_power

def plot_power_heatmap(power_data, tx_idx=0, log_scale=True, percentile=(1, 99)):

    if power_data.ndim == 4:
        data_2d_spatial = power_data[:, tx_idx, :, :]
    elif power_data.ndim == 3:
        data_2d_spatial = power_data
    else:
        raise ValueError("不支援的維度形狀，請傳入 3D 或 4D 陣列")

    N_time, N_rx, N_sub = data_2d_spatial.shape
    heatmap_data = np.transpose(data_2d_spatial, (1, 2, 0)).reshape(N_rx * N_sub, N_time)
    heatmap_data = np.asarray(heatmap_data, dtype=float)

    if log_scale:
        heatmap_data = 10.0 * np.log10(np.abs(heatmap_data) + 1e-12)

    finite = heatmap_data[np.isfinite(heatmap_data)]
    vmin = None
    vmax = None
    if finite.size and percentile is not None:
        low, high = percentile
        vmin, vmax = np.percentile(finite, [low, high])
        if vmin == vmax:
            vmin = None
            vmax = None

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        cmap='jet',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel('Time (packet)', fontsize=10)
    ax.set_ylabel('RX - Subcarrier', fontsize=10)
    title_suffix = '' if log_scale else ''
    ax.set_title('CSI Power' + title_suffix, fontsize=12)
    
    y_ticks = np.arange(0, N_rx * N_sub, N_sub)
    ax.set_yticks(y_ticks + N_sub / 2)
    ax.set_yticklabels([f'RX {i}' for i in range(N_rx)])

    for y in y_ticks[1:]:
        ax.axhline(y=y - 0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar_label = 'Power Intensity (dB)' if log_scale else 'Power Intensity'
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()

def dfs_channel_weighting(raw_CSI, target_fd, args):
    _, weighted_power = weight_gen(raw_CSI, target_fd, args)
    plot_power_heatmap(weighted_power, tx_idx=0, log_scale=True)

def dfs_weighted_heatmap(raw_CSI, args):
    """
    Build signed-Doppler-gated Azi-ToF maps using WIDFS channel reliability.

    WIDFS estimates channel weights at |fd|. The signed Doppler value remains in
    the Azi-ToF-Doppler gate, while the weights only suppress unreliable
    magnitude channels. One weighted 3-D cube is shared by all signed gates.

    Returns a dictionary containing selected peaks, per-peak WIDFS weights,
    confidence scores, the combined channel weight, and projected heatmaps.
    """
    # ============Preprocessing================
    start_preprocessing = time.time()
    amplitude_CSI = np.nan_to_num(
        np.abs(raw_CSI), nan=0.0, posinf=0.0, neginf=0.0
    )
    background = pp.MA(amplitude_CSI, args.fs * 1.0)
    background_floor = np.maximum(background, 1e-8)
    CSI_norm = (amplitude_CSI - background) / background_floor

    end_preprocessing = time.time()
    print(f"Preprocessing Time: {end_preprocessing - start_preprocessing:.2f}s")
    # ============Preprocessing================


    total_frames = amplitude_CSI.shape[0]
    if total_frames == 0:
        raise ValueError("raw_CSI must contain at least one frame")
    frame_idx = int(np.clip(getattr(args, "frame_idx", 0), 0, total_frames - 1))
    print(f"\n========== Processing Frame {frame_idx} ==========")

    peak_result = Doppler_spec.gen_spectrum_from_ToF_Doppler(
        CSI_norm,
        frame_idx,
        args,
        method=getattr(args, "widfs_peak_projection_method", "max"),
    )
    if peak_result is None:
        print("ToF-Doppler spectrum 無法建立，跳過此 Frame。")
        return None

    peak_fd, peak_db = peak_result
    if len(peak_fd) == 0:
        print("未偵測到任何 Doppler 峰值，跳過此 Frame。")
        return None

    min_abs_fd = float(getattr(args, "widfs_min_abs_fd", 0.5))
    dynamic_mask = np.abs(peak_fd) > min_abs_fd
    target_fds = peak_fd[dynamic_mask]
    target_dbs = peak_db[dynamic_mask]

    max_targets = int(getattr(args, "widfs_max_targets", 10))
    N_targets = min(max_targets, len(target_fds))
    target_fds = target_fds[:N_targets]
    target_dbs = target_dbs[:N_targets]
    if N_targets == 0:
        print(f"沒有 |fd| > {min_abs_fd:g} Hz 的動態峰值，跳過此 Frame。")
        return None

    print(f"Selected Top-{N_targets} Dynamic DFS: {np.round(target_fds, 2)} Hz")

    window_size = int(getattr(args, "widfs_window_size", round(args.fs)))
    window_size = int(np.clip(window_size, 2, total_frames))
    start_idx = int(
        np.clip(frame_idx - window_size // 2, 0, total_frames - window_size)
    )
    end_idx = start_idx + window_size

    # Compute the MA background on the full sequence, then fit WIDFS locally.
    # This avoids zero-padding artifacts caused by recomputing MA on a short cut.
    background_power = np.maximum(background**2, 1e-12)
    dynamic_power = amplitude_CSI**2 - background_power
    per_peak_weights = []
    normalized_weights = []
    confidences = []
    weight_percentile = float(getattr(args, "widfs_weight_percentile", 95.0))
    weight_percentile = float(np.clip(weight_percentile, 1.0, 100.0))

    for i, (fd, db) in enumerate(zip(target_fds, target_dbs)):
        filtered_power = apply_doppler_bandpass(dynamic_power, fd, args)
        norm_power_window = (
            filtered_power[start_idx:end_idx]
            / background_power[start_idx:end_idx]
        )
        channel_weights = _channel_weights_from_norm_power(
            norm_power_window, abs(fd), args
        )
        channel_weights = np.nan_to_num(
            channel_weights, nan=0.0, posinf=0.0, neginf=0.0
        )
        positive_weights = channel_weights[channel_weights > 0]
        confidence = (
            float(np.percentile(positive_weights, 90.0))
            if positive_weights.size
            else 0.0
        )
        scale = (
            float(np.percentile(positive_weights, weight_percentile))
            if positive_weights.size
            else 0.0
        )
        if scale > 0:
            normalized_weight = np.clip(channel_weights / scale, 0.0, 1.0)
        else:
            normalized_weight = np.zeros_like(channel_weights)

        per_peak_weights.append(channel_weights)
        normalized_weights.append(normalized_weight)
        confidences.append(confidence)
        print(
            f"  Target {i + 1}: fd={fd:.2f} Hz, peak={db:.2f} dB, "
            f"WIDFS confidence={confidence:.4g}"
        )

    per_peak_weights = np.asarray(per_peak_weights)
    confidences = np.asarray(confidences, dtype=float)
    combined_weight = np.max(np.asarray(normalized_weights), axis=0)
    if not np.any(combined_weight > 0):
        print("WIDFS 權重全部為 0，改用等權重建立 signed Doppler heatmap。")
        combined_weight = np.ones_like(combined_weight)

    # sqrt(weight) makes the resulting covariance energy approximately follow
    # the intended WIDFS weight instead of squaring it again.
    CSI_weighted = CSI_norm * np.sqrt(combined_weight)[None, ...]

    gate_half_width = float(
        getattr(args, "widfs_gate_half_width", max(float(args.doppler_step), 1.0))
    )
    if gate_half_width <= 0:
        raise ValueError(
            f"widfs_gate_half_width must be positive, got {gate_half_width}"
        )
    doppler_min = float(args.doppler_min)
    doppler_max = float(args.doppler_max)
    signed_gates = [
        [max(doppler_min, fd - gate_half_width),
         min(doppler_max, fd + gate_half_width)]
        for fd in target_fds
    ]

    azi_tof_dop = MUSIC.Azi_ToF_Dop(args)
    heatmap_results = azi_tof_dop.gen_spectrum(
        CSI_weighted,
        frame_idx=frame_idx,
        x_axis="azi",
        y_axis="tof",
        method=getattr(args, "widfs_heatmap_projection_method", "max"),
        z_range=signed_gates,
    )

    return {
        "frame_idx": frame_idx,
        "peak_fd": target_fds,
        "peak_db": target_dbs,
        "widfs_confidence": confidences,
        "per_peak_channel_weights": per_peak_weights,
        "combined_channel_weight": combined_weight,
        "signed_doppler_gates": np.asarray(signed_gates, dtype=float),
        "heatmaps": heatmap_results,
    }
