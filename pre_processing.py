import numpy as np

def self_sanitize(x):
    mag = np.abs(x)
    mag[mag == 0] = 1
    return x * np.conj(x) / mag

def MA(csi_amp, window_size):
    window_size = int(round(window_size))
    window = np.ones(window_size) / window_size
    
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)

def rm_SFO_PDD(
    CSI: np.ndarray,
    delta_f: float = None,
    center_x: bool = True,
    tx: int = 0,
    frames: np.ndarray = None,
    return_info: bool = False,
):
    """
    Offline batch RoArray Eq(24)(25)-style correction for ALL frames.

    CSI: (T, Tx, Rx, Subc) complex
    delta_f: subcarrier spacing (Hz). If provided, return tau_u per frame.
    center_x: center subcarrier indices to improve numerical stability.
    tx: which Tx to process (default 0). If you want all Tx, loop outside or adapt.
    frames: optional frame indices to process. None => all frames.
    return_info: if True, return dict with slope/intercept/tau_u arrays.

    Returns
    -------
    CSI_corr : np.ndarray
        Corrected CSI (same shape as input).
    info : dict (optional)
        slope[t], intercept[t], tau_u[t] (if delta_f provided)
    """
    CSI = np.asarray(CSI)
    if CSI.ndim != 4:
        raise ValueError(f"CSI must be 4D (T,Tx,Rx,Subc), got {CSI.shape}")

    T, TxN, Nrx, Nsubc = CSI.shape
    if tx < 0 or tx >= TxN:
        raise ValueError(f"tx out of range: {tx}, TxN={TxN}")

    if frames is None:
        frames = np.arange(T)
    else:
        frames = np.asarray(frames)

    # x-axis
    k = np.arange(Nsubc, dtype=np.float64)
    x = (k - k.mean()) if center_x else k
    X = np.stack([x, np.ones_like(x)], axis=1)  # (Nsubc,2)

    CSI_corr = CSI.copy()

    slope_arr = np.zeros(T, dtype=np.float64)
    intercept_arr = np.zeros(T, dtype=np.float64)
    tau_arr = np.zeros(T, dtype=np.float64) if delta_f is not None else None

    for t in frames:
        H = CSI_corr[t, tx]  # (Nrx, Nsubc)

        # unwrapped phase along subcarrier axis
        phi = np.unwrap(np.angle(H), axis=1)

        # stack all Rx samples for robust LS
        A_ls = np.tile(X, (Nrx, 1))       # (Nrx*Nsubc,2)
        b_ls = phi.reshape(-1)            # (Nrx*Nsubc,)

        slope, intercept = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]
        slope_arr[t] = slope
        intercept_arr[t] = intercept

        if delta_f is not None:
            tau_arr[t] = -slope / (2*np.pi*delta_f)

        # remove slope only (keep intercept)
        phi_corr = phi - slope * x[None, :]
        H_corr = np.abs(H) * np.exp(1j * phi_corr)

        CSI_corr[t, tx] = H_corr

    if not return_info:
        return CSI_corr

    info = {"slope": slope_arr, "intercept": intercept_arr}
    if delta_f is not None:
        info["tau_u"] = tau_arr
    return CSI_corr, info
    
def remove_CFO(CSI, rx_slice=None, tx=0, eps=1e-12):
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