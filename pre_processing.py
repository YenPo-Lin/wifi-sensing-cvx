import numpy as np
import matplotlib.pyplot as plt
import pywt
import pre_processing as pp

def sample_subcarriers(args, CSI, freq_space=16):
    """
    現在的CSI(T=frames, M=8, K=2025)
    BW=160MHz, K=2025, delta_f=78.125kHz

    freq_space=16
    BW=160MHz, K=2025//16=126, delta_f=1.269MHz
    """
    num_subc = CSI.shape[-1] # 原本是 2025
    sampled_CSI = CSI[..., ::freq_space]
    indices = np.arange(0, num_subc, freq_space)
    sampled_CSI = CSI[..., indices]
    actual_K = len(indices)
    original_delta_f = args.BW / num_subc
    new_delta_f = original_delta_f * freq_space
    effective_BW = (actual_K - 1) * new_delta_f
    if args is not None:
        args.delta_f = new_delta_f

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

    CSI shape: (num_frames, num_tx, num_rx, num_subcarriers)
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
    window_size = int(round(window_size))
    window = np.ones(window_size) / window_size
    
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)

    """
    Remove per-frame common phase error (CPE), often dominated by residual CFO.
    CSI: (T, Tx, Rx, Subc) complex
    subc_slice: 哪些 subcarriers 用來估 common phase
    rx_slice: None=全部 Rx，或 slice/list 指定 Rx
    Returns: CSI_corr (same shape), phi0 (T,) estimated common phase (rad)
    """
    CSI_corr = CSI.copy()
    subc_slice = np.arange(CSI.shape[3])
    H = CSI_corr[:, tx]  # (T, Rx, Subc)
    

    if rx_slice is None:
        H_sel = H[:, :, subc_slice]      # (T, Rx, K)
    else:
        H_sel = H[:, rx_slice, subc_slice]

    # 用複數平均的角度估每一 frame 的 common phase
    # phi0[t] = angle( sum_{rx,subc} H_sel[t] )
    z = np.sum(H_sel, axis=(1, 2))       # (T,)
    phi0 = np.angle(z + eps)             # (T,)

    # 每一 frame 整包乘上 exp(-j*phi0[t]) 去掉共同相位
    CSI_corr[:, tx] = H * np.exp(-1j * phi0)[:, None, None]
    return CSI_corr, phi0