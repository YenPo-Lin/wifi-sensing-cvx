import numpy as np
import matplotlib.pyplot as plt
import pywt
import pre_processing as pp

def sample_subcarriers(args, CSI, freq_space=16):
    if freq_space == 1:
        return CSI
    
    """
    現在的CSI(T=frames, M=8, K=2025)
    BW=160MHz, K=2025, delta_f=78.125kHz

    freq_space=16
    BW=160MHz, K=2025//16=126, delta_f=1.269MHz
    """
    num_sc = CSI.shape[-1] # 原本是 2025
    sampled_CSI = CSI[..., ::freq_space]
    indices = np.arange(0, num_sc, freq_space)
    sampled_CSI = CSI[..., indices]
    actual_K = len(indices)
    original_delta_f = args.BW / num_sc
    new_delta_f = original_delta_f * freq_space
    effective_BW = (actual_K - 1) * new_delta_f
    if args is not None:
        args.delta_f = new_delta_f
        args.num_sc = actual_K

    print(f"[Sampling] K: {actual_K} | Δf: {new_delta_f/1e6:.2f}MHz | BW: {effective_BW/1e6:.1f}MHz")

    return sampled_CSI

def plot_amp_DWT_components_time(
    CSI,
    tx_idx=0,
    rx_idx=0,
    subc_num=50,
    wavelet="db6",
    level=5,
    fs=None
):
    """
    對單一 Tx/Rx/Subcarrier 的 amplitude-time signal 做 DWT，
    並把 approximation/detail components 分橫列畫出來。

    CSI shape: (num_frames, num_tx, num_rx, num_scarriers)
    """

    # 1. 取 amplitude time-series
    amp = np.abs(CSI[:, tx_idx, rx_idx, 0:subc_num])

    # 2. DWT decomposition
    coeffs = pywt.wavedec(amp, wavelet=wavelet, level=level)

    # coeffs[0] = cA_level
    # coeffs[1] = cD_level
    # coeffs[2] = cD_level-1 ...
    # coeffs[-1] = cD1

    components = []

    # 3. reconstruction of each component to original length
    for i in range(len(coeffs)):
    
        coeffs_i = [np.zeros_like(c) for c in coeffs]
        coeffs_i[i] = coeffs[i]

        comp = pywt.waverec(coeffs_i, wavelet=wavelet)
        comp = comp[:len(amp)]

        components.append(comp)

    # 4. x-axis
    if fs is None:
        x = np.arange(len(amp))
        xlabel = "Frame index"
    else:
        x = np.arange(len(amp)) / fs
        xlabel = "Time (s)"

    # 5. labels
    labels = [f"A{level}"]
    for l in range(level, 0, -1):
        labels.append(f"D{l}")

    # 6. plot in horizontal rows
    n_rows = len(components) + 1

    plt.figure(figsize=(8, 8))

    # raw signal
    plt.subplot(n_rows, 1, 1)
    plt.plot(x, amp)
    plt.ylabel("Raw amp")
    plt.title(f"DWT Components of Amplitude | Rx{rx_idx}, Top {subc_num} subc, wavelet={wavelet}")
    plt.grid(True, alpha=0.3)

    # components
    for i, comp in enumerate(components):
        plt.subplot(n_rows, 1, i + 2)
        plt.plot(x, comp)
        plt.ylabel(labels[i])
        plt.grid(True, alpha=0.3)

    plt.xlabel(xlabel)
    plt.tight_layout()


    return amp, components, coeffs

def reconstruct_component(coeffs, wavelet, keep_idx, target_len):
    """
    coeffs order for level=L:
        coeffs[0] = A_L
        coeffs[1] = D_L
        coeffs[2] = D_{L-1}
        ...
        coeffs[L] = D_1
    """
    new_coeffs = [np.zeros_like(c) for c in coeffs]
    new_coeffs[keep_idx] = coeffs[keep_idx]

    rec = pywt.waverec(new_coeffs, wavelet=wavelet)
    return rec[:target_len]

def DWT_amp_time_(CSI, wavelet="db6", level=6):
    """
    對 CSI amplitude 沿 time axis 做 DWT，
    動態根據 level 回傳對應的 A_L, D_L ... D_1，每個 shape 都是 (T, Tx, Rx, Subc).

    CSI shape: (T, Tx, Rx, Subc)
    """

    amp = np.abs(CSI)
    T, Tx, Rx, K = amp.shape

    # 1. 動態生成 labels (例如 level=4 會生成 ["A4", "D4", "D3", "D2", "D1"])
    labels = [f"A{level}"] + [f"D{level - i}" for i in range(level)]
    
    # 2. 動態初始化 components 字典
    components = {label: np.zeros_like(amp, dtype=float) for label in labels}

    for tx in range(Tx):
        for rx in range(Rx):
            for k in range(K):
                x = amp[:, tx, rx, k]

                # 進行 DWT 分解，coeffs 長度為 level + 1
                coeffs = pywt.wavedec(
                    x,
                    wavelet=wavelet,
                    level=level
                )

                # 3. 迴圈迭代 labels，idx 剛好對應 coeffs 的 index
                for idx, label in enumerate(labels):
                    components[label][:, tx, rx, k] = reconstruct_component(
                        coeffs,
                        wavelet=wavelet,
                        keep_idx=idx,
                        target_len=T
                    )

    return components, amp

def DWT_components(CSI, target_labels= ["D6", "D5", "D4", "D3", "D2", "D1"]):
    components, amp = pp.DWT_amp_time_(CSI, wavelet="db6", level=6)
    CSI_dwt_amp = np.zeros_like(CSI, dtype=complex) 
    for label in target_labels:
        if label in components:
            CSI_dwt_amp += components[label]
    return CSI_dwt_amp

def self_sanitize(x):
    mag = np.abs(x)
    mag[mag == 0] = 1
    return x * np.conj(x) / mag

def MA(csi_amp, window_size):
    window_size = min(int(round(window_size)), len(csi_amp))
    if window_size < 1:
        raise ValueError("Moving-average window and input must be nonempty")
    window = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)

def PCA_time(CSI, window_size, k=3):
    """
    對 frame 軸做滑動視窗 PCA，並以前 k 個主成分做低秩重建後回傳 CSI。

    CSI shape: (T, Tx, Rx, Subc)
    window_size: 時間窗長度，單位為 frames
    k: 保留的主成分數
    """
    CSI = np.asarray(CSI)

    if CSI.ndim != 4:
        raise ValueError(f"Expected CSI shape (T, Tx, Rx, Subc), but got {CSI.shape}")

    window_size = int(round(window_size))
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but got {window_size}")
    if k < 1:
        raise ValueError(f"k must be at least 1, but got {k}")

    T, Tx, Rx, Subc = CSI.shape
    if T == 0:
        return CSI.copy()

    window_size = min(window_size, T)
    feature_dim = Tx * Rx * Subc
    flat_csi = CSI.reshape(T, feature_dim)

    accum = np.zeros((T, feature_dim), dtype=np.complex128)
    counts = np.zeros(T, dtype=np.float64)

    for start in range(0, T - window_size + 1):
        end = start + window_size
        block = flat_csi[start:end]

        mean = np.mean(block, axis=0, keepdims=True)
        centered = block - mean

        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        keep_k = min(int(k), s.size)

        if keep_k > 0:
            recon = (u[:, :keep_k] * s[:keep_k]) @ vh[:keep_k, :] + mean
        else:
            recon = np.repeat(mean, window_size, axis=0)

        accum[start:end] += recon
        counts[start:end] += 1.0

    counts[counts == 0] = 1.0
    recon_csi = accum / counts[:, None]

    return recon_csi.reshape(T, Tx, Rx, Subc).astype(CSI.dtype, copy=False)


def dynamic_static_ratio(CSI, args):
    """Dynamic/static amplitude-energy ratio for each adjacent Rx pair."""
    CSI = np.asarray(CSI)
    frames, _, receivers, subcarriers = CSI.shape
    if receivers % 2:
        raise ValueError("Adjacent Rx pairing requires an even number of receivers")
    window = int(args.avg_frames)
    if window < 1:
        raise ValueError("avg_frames must be positive")
    amp = np.abs(CSI.reshape(frames, CSI.shape[1], receivers // 2, 2, subcarriers))
    static = MA(amp, window)
    starts = np.arange(0, frames, window)
    static_energy = np.add.reduceat(np.sum(static ** 2, axis=(1, 4)), starts, axis=0)
    dynamic_energy = np.add.reduceat(np.sum((amp - static) ** 2, axis=(1, 4)), starts, axis=0)
    return dynamic_energy / np.maximum(static_energy, np.finfo(float).eps)


def data_check(self, C0):
    C0 = np.asarray(C0)
    if C0.ndim == 2:
        C0 = C0[:, None, None, :]
    elif C0.ndim == 3:
        C0 = C0[:, :, None, :]
    if C0.ndim != 4 or min(C0.shape) == 0:
        raise ValueError(
            f"Dop expects nonempty (frame, Tx pair, Rx pair, subcarrier) C0, got {C0.shape}"
        )
    return np.asarray(C0, dtype=np.complex128)


def _select_subcarriers(self, CSI):
    method = getattr(self.args, "sc_selection_method", "snr")
    selectors = {
        "snr": pp.sc_selection_snr,
        "witraj": pp.sc_selection_WiTraj,
    }
    if method not in selectors:
        raise ValueError(f"Unknown sc_selection_method: {method!r}")
    indices_by_tx = []
    for tx_idx in range(CSI.shape[1]):
        indices_by_rx = []
        for rx_idx in range(CSI.shape[2]):
            file_name = f"channel_quality_{method}_Tx{tx_idx}_Rx{rx_idx}.png"
            _, sc_indices = selectors[method](
                CSI[:, tx_idx, rx_idx, :], self.args, file_name=file_name
            )
            indices_by_rx.append(sc_indices)
        indices_by_tx.append(np.stack(indices_by_rx, axis=1))
    return np.stack(indices_by_tx, axis=1)


def _select_subcarriers_azi_tof(
    CSI, frame_idx, args, *, min_count=20, min_bandwidth_hz=80e6,
    frequency_offsets_hz=None,
):
    """Rank subcarriers in one frame's time context and enforce a frequency span.

    One common index set is used across Tx/Rx so that the spatial-frequency
    covariance has a single, well-defined steering vector.
    """
    CSI = np.asarray(CSI)
    if CSI.ndim != 4 or min(CSI.shape) == 0:
        raise ValueError(f"Expected nonempty (frame, Tx, Rx, SC) CSI, got {CSI.shape}")
    frames, _, _, num_sc = CSI.shape
    count = max(int(min_count), int(np.ceil(float(getattr(args, "top_sc_ratio", 0.5)) * num_sc)))
    if count > num_sc:
        raise ValueError(f"Need at least {count} subcarriers, but only {num_sc} are available")
    if frequency_offsets_hz is None:
        frequency_offsets_hz = getattr(args, "frequency_offsets_hz", None)
    if frequency_offsets_hz is None:
        frequency_offsets_hz = np.arange(num_sc) * float(args.delta_f)
    frequency_offsets_hz = np.asarray(frequency_offsets_hz, dtype=float)
    if frequency_offsets_hz.shape != (num_sc,) or not np.all(np.isfinite(frequency_offsets_hz)):
        raise ValueError("frequency_offsets_hz must contain one finite value per subcarrier")

    window = min(int(args.avg_frames), frames)
    if window < 8:
        raise ValueError("Subcarrier quality selection needs at least 8 frames")
    frame_idx = int(np.clip(frame_idx, 0, frames - 1))
    start = int(np.clip(frame_idx - window // 2, 0, frames - window))
    context = CSI[start:start + window]
    valid = np.flatnonzero(np.isfinite(context).all(axis=(0, 1, 2)))
    if valid.size < count:
        raise ValueError(f"Need {count} finite subcarriers, found {valid.size}")

    frequencies = np.fft.fftfreq(window, d=1.0 / float(args.fs))
    max_doppler = max(abs(float(args.doppler_min)), abs(float(args.doppler_max)))
    motion = (np.abs(frequencies) >= 2.0) & (np.abs(frequencies) <= max_doppler)
    if not np.any(motion):
        raise ValueError("No FFT bins in the configured Doppler motion band")
    tapered = context[..., valid] * np.hanning(window)[:, None, None, None]
    power = np.abs(np.fft.fft(tapered, axis=0)[motion]) ** 2
    peak = np.max(power, axis=0)
    floor = np.median(power, axis=0)
    quality = np.median((peak - floor) / np.maximum(floor, 1e-12), axis=(0, 1))
    ranking = valid[np.argsort(-quality, kind="stable")]
    selected = ranking[:count]
    span = np.ptp(frequency_offsets_hz[selected])
    if span < min_bandwidth_hz:
        separation = np.abs(
            frequency_offsets_hz[valid, None] - frequency_offsets_hz[None, valid]
        )
        candidate_pairs = np.argwhere(np.triu(separation >= min_bandwidth_hz, k=1))
        if candidate_pairs.size == 0:
            raise ValueError(
                f"No subcarrier set can span {min_bandwidth_hz / 1e6:g} MHz; "
                f"available span is {np.ptp(frequency_offsets_hz[valid]) / 1e6:.2f} MHz"
            )
        pair_scores = quality[candidate_pairs[:, 0]] + quality[candidate_pairs[:, 1]]
        best_pair = valid[candidate_pairs[np.argmax(pair_scores)]]
        remainder = ranking[~np.isin(ranking, best_pair)]
        selected = np.concatenate((best_pair, remainder[:count - 2]))
    return np.sort(selected), frequency_offsets_hz


def conjugate_multiplication(raw_CSI, args, *, order=None, return_reference=False):
    """Pair adjacent Rx with fixed or ratio-chosen reference, then average Tx."""
    raw_CSI = np.asarray(raw_CSI)
    if raw_CSI.ndim != 4:
        raise ValueError(f"Expected raw CSI (frame, Tx, Rx, SC), got {raw_CSI.shape}")
    num_frames, num_Tx, num_Rx, num_sc = raw_CSI.shape
    if num_frames == 0 or num_Tx % 2 or num_Rx % 2 or min(num_Tx, num_Rx, num_sc) == 0:
        raise ValueError("Conjugate multiplication requires even Tx and Rx counts")
    if order is None:
        order = getattr(args, "conjugate_order", "fixed")
    if order not in {"fixed", "ratio"}:
        raise ValueError(f"conjugate_order must be 'fixed' or 'ratio', got {order!r}")

    h0 = raw_CSI[:, :, 0::2, :]
    h1 = raw_CSI[:, :, 1::2, :]
    reference_rx = np.zeros((num_frames, num_Rx // 2), dtype=np.int8)
    if order == "ratio":
        ratio = dynamic_static_ratio(raw_CSI, args)
        reference_rx = np.repeat(
            np.argmin(ratio, axis=2), int(args.avg_frames), axis=0
        )[:num_frames]
        rx_products = np.where(
            reference_rx[:, None, :, None] == 0,
            h0 * np.conj(h1),
            h1 * np.conj(h0),
        )
    else:
        rx_products = h0 * np.conj(h1)
    result = rx_products.reshape(num_frames, num_Tx // 2, 2, num_Rx // 2, num_sc).mean(axis=2)
    if return_reference:
        return result, reference_rx
    return result
