import numpy as np
import matplotlib.pyplot as plt
import pywt

def sample_subcarriers(args, csi_amp, freq_space=16):
    if freq_space == 1:
        return csi_amp
    
    """
    現在的CSI(T=frames, M=8, K=2025)
    BW=160MHz, K=2025, delta_f=78.125kHz

    freq_space=16
    BW=160MHz, K=2025//16=126, delta_f=1.269MHz
    """
    num_subc = csi_amp.shape[-1] # 原本是 2025
    sampled_CSI = csi_amp[..., ::freq_space]
    indices = np.arange(0, num_subc, freq_space)
    sampled_CSI = csi_amp[..., indices]
    actual_K = len(indices)
    original_delta_f = args.BW / num_subc
    new_delta_f = original_delta_f * freq_space
    effective_BW = (actual_K - 1) * new_delta_f
    if args is not None:
        args.delta_f = new_delta_f
        args.num_subcarriers = actual_K

    print(f"[Sampling] K: {actual_K} | Δf: {new_delta_f/1e6:.2f}MHz | BW: {effective_BW/1e6:.1f}MHz")

    return sampled_CSI

def plot_amp_DWT_components_time(
    csi_amp,
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

    CSI shape: (num_frames, num_tx, num_rx, num_subcarriers)
    """

    # 1. 取 amplitude time-series
    amp = np.abs(csi_amp[:, tx_idx, rx_idx, 0:subc_num])

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

def DWT_amp_time_(csi_amp, wavelet="db6", level=6):
    """
    對 CSI amplitude 沿 time axis 做 DWT，
    動態根據 level 回傳對應的 A_L, D_L ... D_1，每個 shape 都是 (T, Tx, Rx, Subc).

    CSI shape: (T, Tx, Rx, Subc)
    """

    amp = np.abs(csi_amp)
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

def DWT_components(csi_amp, target_labels= ["D6", "D5", "D4", "D3", "D2", "D1"]):
    components, amp = DWT_amp_time_(csi_amp, wavelet="db6", level=6)
    csi_dwt_amp = np.zeros_like(csi_amp, dtype=complex)
    for label in target_labels:
        if label in components:
            csi_dwt_amp += components[label]
    return csi_dwt_amp

def self_sanitize(csi_amp):
    mag = np.abs(csi_amp)
    mag[mag == 0] = 1
    return csi_amp * np.conj(csi_amp) / mag

def MA(csi_amp, window_size):
    window_size = int(round(window_size))
    window = np.ones(window_size) / window_size
    
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)

def SG(csi_amp, window_size, polyorder=3):
    """
    Apply a Savitzky-Golay smoothing filter along the frame axis.

    csi_amp shape: (frame, tx, rx, subcarrier)
    """
    csi_amp = np.asarray(csi_amp)

    if csi_amp.ndim < 1:
        raise ValueError(f"Expected at least 1 dimension, but got {csi_amp.shape}")

    window_size = int(round(window_size))
    polyorder = int(round(polyorder))

    if window_size < 1:
        raise ValueError(f"window_size must be positive, but got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, but got {window_size}")
    if polyorder < 0:
        raise ValueError(f"polyorder must be non-negative, but got {polyorder}")
    if polyorder >= window_size:
        raise ValueError(
            f"polyorder must be smaller than window_size, got {polyorder} >= {window_size}"
        )
    if csi_amp.shape[0] < window_size:
        raise ValueError(
            f"window_size={window_size} cannot exceed num_frames={csi_amp.shape[0]}"
        )

    half_window = window_size // 2
    x = np.arange(-half_window, half_window + 1, dtype=float)
    vandermonde = np.vander(x, N=polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(vandermonde)[0]

    pad_width = [(half_window, half_window)] + [(0, 0)] * (csi_amp.ndim - 1)
    padded = np.pad(csi_amp, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=window_size,
        axis=0,
    )
    smoothed = np.tensordot(windows, coeffs, axes=([-1], [0]))

    return smoothed.astype(csi_amp.dtype, copy=False)

def PCA_time(csi_amp, window_size, k=3):
    """
    對 frame 軸做滑動視窗 PCA，並以前 k 個主成分做低秩重建後回傳 CSI。

    CSI shape: (T, Tx, Rx, Subc)
    window_size: 時間窗長度，單位為 frames
    k: 保留的主成分數
    """
    csi_amp = np.asarray(csi_amp)

    if csi_amp.ndim != 4:
        raise ValueError(f"Expected CSI shape (T, Tx, Rx, Subc), but got {csi_amp.shape}")

    window_size = int(round(window_size))
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but got {window_size}")
    if k < 1:
        raise ValueError(f"k must be at least 1, but got {k}")

    T, Tx, Rx, Subc = csi_amp.shape
    if T == 0:
        return csi_amp.copy()

    window_size = min(window_size, T)
    feature_dim = Tx * Rx * Subc
    flat_csi = csi_amp.reshape(T, feature_dim)

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

    return recon_csi.reshape(T, Tx, Rx, Subc).astype(csi_amp.dtype, copy=False)
