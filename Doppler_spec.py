import os
import MUSIC
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, stft


DEFAULT_FS = 100.0
DEFAULT_CPI_LEN = 128


def _select_csi_channels(CSI, tx=0):
    """
    Return CSI as (frames, channels), where channels are Rx-subcarrier bins.
    """
    CSI = np.asarray(CSI)

    if CSI.ndim == 4:
        num_tx = CSI.shape[1]
        if tx is None:
            selected = CSI
        else:
            if tx < 0 or tx >= num_tx:
                raise ValueError(f"tx must be in [0, {num_tx - 1}], got {tx}")
            selected = CSI[:, tx:tx + 1, :, :]
        return selected.reshape(CSI.shape[0], -1)

    if CSI.ndim == 3:
        if tx not in (0, None):
            raise ValueError("tx can only be 0 or None when CSI has shape (frames, rx, subcarrier)")
        return CSI.reshape(CSI.shape[0], -1)

    raise ValueError(
        "Expected CSI shape (frames, tx, rx, subcarrier) or "
        f"(frames, rx, subcarrier), got {CSI.shape}"
    )


def compute_stft_doppler_spectrogram(
    CSI,
    args,
    tx=0,
    nperseg=None,
    noverlap=None,
    window="hann",
    remove_static=True,
    db=True,
):
    """
    Compute an STFT Doppler spectrogram averaged over all Rx-subcarrier channels.

    Returns:
        doppler_hz: Doppler frequency bins after fftshift.
        time_s: STFT segment times.
        spectrogram: averaged power spectrogram with shape (doppler_hz, time_s).
    """
    fs = float(args.fs)
    x = _select_csi_channels(CSI, tx=tx).astype(np.complex128, copy=False)

    if x.shape[0] < 2:
        raise ValueError(f"Need at least two frames for STFT, got {x.shape[0]}")
    if x.shape[1] == 0:
        raise ValueError("Need at least one Rx-subcarrier channel for STFT")

    if remove_static:
        x = x - np.mean(x, axis=0, keepdims=True)

    if nperseg is None:
        nperseg = int(getattr(args, "stft_nperseg", min(16, x.shape[0])))
    nperseg = int(min(max(2, nperseg), x.shape[0]))

    if noverlap is None:
        noverlap = int(args.stft_noverlap)
    noverlap = int(np.clip(noverlap, 0, nperseg - 1))

    # scipy.signal.stft works on the last axis, so transpose to
    # (channels, frames), then average power over channels.
    doppler_hz, time_s, Zxx = stft(
        x.T,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
        axis=-1,
    )

    power = np.mean(np.abs(Zxx) ** 2, axis=0)
    doppler_hz = np.fft.fftshift(doppler_hz)
    power = np.fft.fftshift(power, axes=0)

    if db:
        power = 10.0 * np.log10(power + 1e-12)

    return doppler_hz, time_s, power


def plot_stft_doppler_spectrogram(
    CSI,
    args,
    tx=0,
    nperseg=None,
    noverlap=None,
    window="hann",
    remove_static=True,
    cmap="jet",
    title="STFT Doppler spectrogram",
):
    doppler_hz, time_s, spectrogram = compute_stft_doppler_spectrogram(
        CSI,
        args,
        tx=tx,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        remove_static=remove_static,
        db=True,
    )

    doppler_min = getattr(args, "doppler_min", None)
    doppler_max = getattr(args, "doppler_max", None)
    if doppler_min is not None and doppler_max is not None:
        mask = (doppler_hz >= doppler_min) & (doppler_hz <= doppler_max)
        if np.any(mask):
            doppler_hz = doppler_hz[mask]
            spectrogram = spectrogram[mask, :]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.pcolormesh(time_s, doppler_hz, spectrogram, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler frequency (Hz)")
    ax.set_title(title)

    if getattr(args, "colorbar", True):
        fig.colorbar(im, ax=ax, label="Power (dB)")

    save_dir = getattr(args, "pics_dir", None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "doppler_spectrogram_STFT.png")
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Saved STFT Doppler spectrogram: {save_path}")

    return fig, ax, doppler_hz, time_s, spectrogram


def gen_spectrogram(CSI, args, tx=0):
    return plot_stft_doppler_spectrogram(
        CSI,
        args,
        tx=tx,
        title="STFT Doppler spectrogram",
    )


def compute_stft_doppler_spectrum(*args, **kwargs):
    return compute_stft_doppler_spectrogram(*args, **kwargs)


def plot_stft_doppler_spectrum(*args, **kwargs):
    return plot_stft_doppler_spectrogram(*args, **kwargs)


def gen_spectrum(CSI, frame_idx, tx=0):
    """
    Generate a Doppler spectrum for one CSI frame.

    The selected CPI is centered at ``frame_idx``. CSI is FFT'ed along the time
    axis, then power is averaged over all selected Rx-subcarrier channels.

    Returns:
        fig, ax: matplotlib figure and axes.
        doppler_hz: fftshift'ed Doppler frequency bins.
        spectrum_db: averaged Doppler power in dB.
        peak_doppler_hz: Doppler bins detected as local peaks.
        peak_indices: integer indices into doppler_hz/spectrum_db.
    """
    x = _select_csi_channels(CSI, tx=tx).astype(np.complex128, copy=False)
    total_frames = x.shape[0]
    if total_frames < 2:
        raise ValueError(f"Need at least two frames for Doppler FFT, got {total_frames}")
    if x.shape[1] == 0:
        raise ValueError("Need at least one Rx-subcarrier channel for Doppler FFT")

    frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
    cpi_len = min(DEFAULT_CPI_LEN, total_frames)
    start = int(np.clip(frame_idx - cpi_len // 2, 0, total_frames - cpi_len))
    end = start + cpi_len
    segment = x[start:end, :]

    # Remove static clutter/DC per channel before the Doppler FFT.
    segment = segment - np.mean(segment, axis=0, keepdims=True)
    window = np.hanning(cpi_len).reshape(-1, 1)
    segment = segment * window

    fft_data = np.fft.fftshift(np.fft.fft(segment, axis=0), axes=0)
    power = np.mean(np.abs(fft_data) ** 2, axis=1)
    spectrum_db = 10.0 * np.log10(power + 1e-12)
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(cpi_len, d=1.0 / DEFAULT_FS))

    finite = spectrum_db[np.isfinite(spectrum_db)]
    if finite.size == 0:
        peak_indices = np.array([], dtype=int)
    else:
        median = np.median(finite)
        mad = np.median(np.abs(finite - median))
        noise_scale = 1.4826 * mad
        height = median + max(3.0 * noise_scale, 3.0)
        prominence = max(noise_scale, 1.0)
        peak_indices, _ = find_peaks(spectrum_db, height=height, prominence=prominence)

    peak_doppler_hz = doppler_hz[peak_indices]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(doppler_hz, spectrum_db, linewidth=1.4)
    if peak_indices.size:
        ax.scatter(
            peak_doppler_hz,
            spectrum_db[peak_indices],
            color="tab:red",
            marker="x",
            s=45,
            label="Detected Doppler bins",
            zorder=3,
        )
        ax.legend()
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Doppler frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Doppler spectrum @ frame {frame_idx} ({start}:{end})")
    ax.grid(True, alpha=0.25)

    print("Detected Doppler bins (Hz):", np.round(peak_doppler_hz, 3))

    return fig, ax, doppler_hz, spectrum_db, peak_doppler_hz, peak_indices


def _select_csi_tx_block(CSI, tx=0):
    """
    Return CSI as (frames, tx, rx, subcarrier) for MUSIC.ToF_Doppler.
    """
    CSI = np.asarray(CSI)

    if CSI.ndim == 4:
        num_tx = CSI.shape[1]
        if tx is None:
            return CSI
        if tx < 0 or tx >= num_tx:
            raise ValueError(f"tx must be in [0, {num_tx - 1}], got {tx}")
        return CSI[:, tx:tx + 1, :, :]

    if CSI.ndim == 3:
        if tx not in (0, None):
            raise ValueError("tx can only be 0 or None when CSI has shape (frames, rx, subcarrier)")
        return CSI[:, None, :, :]

    raise ValueError(
        "Expected CSI shape (frames, tx, rx, subcarrier) or "
        f"(frames, rx, subcarrier), got {CSI.shape}"
    )


def cell_averaging_cfar(
    profile,
    training_cells=2,
    guard_cells=1,
    threshold_factor=1.05,
    top_k=None,
    min_peak_distance=1,
    min_prominence_db=0.0,
):
    """
    Detect peaks using 1-D cell-averaging CFAR on a linear-power profile.

    If top_k is provided, CFAR detections are supplemented with the strongest
    local maxima until top_k peaks are available. This is intentionally useful
    for motion gating, where missing a body-part Doppler bin is worse than
    keeping a few extra candidates.

    Returns:
        peak_indices: detected local-maximum bins above adaptive threshold.
        threshold: linear CFAR threshold for each bin, NaN at invalid edge bins.
    """
    profile = np.asarray(profile, dtype=float)
    training_cells = int(training_cells)
    guard_cells = int(guard_cells)
    threshold_factor = float(threshold_factor)
    min_peak_distance = int(min_peak_distance)
    min_prominence_db = float(min_prominence_db)

    if training_cells <= 0:
        raise ValueError(f"training_cells must be positive, got {training_cells}")
    if guard_cells < 0:
        raise ValueError(f"guard_cells must be non-negative, got {guard_cells}")
    if threshold_factor <= 0:
        raise ValueError(f"threshold_factor must be positive, got {threshold_factor}")
    if min_peak_distance <= 0:
        raise ValueError(f"min_peak_distance must be positive, got {min_peak_distance}")

    threshold = np.full(profile.shape, np.nan, dtype=float)
    peak_indices = []
    margin = training_cells + guard_cells

    for idx in range(margin, len(profile) - margin):
        left_train = profile[idx - margin:idx - guard_cells]
        right_train = profile[idx + guard_cells + 1:idx + guard_cells + 1 + training_cells]
        training = np.concatenate((left_train, right_train))
        training = training[np.isfinite(training)]
        if training.size == 0:
            continue

        noise_power = np.mean(training)
        threshold[idx] = threshold_factor * noise_power
        is_local_max = profile[idx] >= profile[idx - 1] and profile[idx] >= profile[idx + 1]
        if np.isfinite(profile[idx]) and profile[idx] > threshold[idx] and is_local_max:
            peak_indices.append(idx)

    peak_indices = np.asarray(peak_indices, dtype=int)
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        profile_db = 10.0 * np.log10(np.maximum(profile, 1e-12))
        local_peaks, _ = find_peaks(
            profile_db,
            distance=min_peak_distance,
            prominence=min_prominence_db,
        )
        local_order = np.argsort(profile_db[local_peaks])[::-1]
        selected = list(peak_indices)
        selected_set = set(selected)
        for idx in local_peaks[local_order]:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= top_k:
                break
        peak_indices = np.asarray(selected, dtype=int)

    return peak_indices, threshold


def gen_spectrum_from_ToF_Doppler(CSI, frame_idx, args, method="sum", tx=0):
    """
    Build a ToF-Doppler MUSIC spectrum, then collapse the ToF axis to Doppler.

    method:
        "sum": sum MUSIC power over all ToF bins.
        "max": keep the strongest ToF bin for each Doppler bin.
        "mean": average MUSIC power over all ToF bins.

    Returns:
        peak_fd, peak_db sorted by peak_db from high to low.
    """
    method = str(method).lower()
    if method not in ("sum", "max", "mean"):
        raise ValueError(f"method must be 'sum', 'max', or 'mean', got {method!r}")

    csi_block = _select_csi_tx_block(CSI, tx=tx)
    tof_dop = MUSIC.ToF_Doppler(args)
    Rxx = tof_dop.Rxx_smooth(csi_block, frame_idx)
    if Rxx is None:
        return None

    tau, fd, P_tof_dop = tof_dop.cal_spectrum(Rxx)

    if method == "sum":
        spectrum = np.sum(P_tof_dop, axis=0)
    elif method == "max":
        spectrum = np.max(P_tof_dop, axis=0)
    else:
        spectrum = np.mean(P_tof_dop, axis=0)

    spectrum_db = 10.0 * np.log10(np.maximum(spectrum, 1e-12))
    peak_indices, _ = cell_averaging_cfar(
        spectrum,
        training_cells=getattr(args, "cfar_training_cells", 2),
        guard_cells=getattr(args, "cfar_guard_cells", 1),
        threshold_factor=getattr(args, "cfar_threshold_factor", 1.05),
        top_k=getattr(args, "cfar_top_k", 5),
        min_peak_distance=getattr(args, "cfar_min_peak_distance", 1),
        min_prominence_db=getattr(args, "cfar_min_prominence_db", 0.0),
    )
    peak_fd = fd[peak_indices]
    peak_db = spectrum_db[peak_indices]
    order = np.argsort(peak_db)[::-1]
    peak_indices = peak_indices[order]
    peak_fd = peak_fd[order]
    peak_db = peak_db[order]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(fd, spectrum_db, linewidth=1.4)
    if peak_indices.size:
        ax.scatter(
            peak_fd,
            peak_db,
            color="tab:red",
            marker="x",
            s=45,
            label="Peak",
            zorder=3,
        )
        for f_i, db_i in zip(peak_fd, peak_db):
            ax.annotate(
                f"{f_i:g} Hz",
                xy=(f_i, db_i),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
        ax.legend()

    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Doppler frequency (Hz)")
    ax.set_ylabel("Collapsed MUSIC spectrum (dB)")
    ax.set_title(f"ToF-Doppler MUSIC -> Doppler spectrum ({method}) @ frame {frame_idx}")
    ax.grid(True, alpha=0.25)

    save_dir = getattr(args, "pics_dir", None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"{int(frame_idx):04d}_tof_doppler_{method}_doppler_spectrum.png",
        )
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Saved: {save_path}")

    print("Detected Doppler bins from ToF-Doppler MUSIC (Hz):", np.round(peak_fd, 3))
    print("Detected Doppler bins from ToF-Doppler MUSIC (dB):", np.round(peak_db, 3))

    return peak_fd, peak_db


def gen_spectrum_from_ToF_Doppler_Rx_diff(CSI, frame_idx, args, method="sum", tx=0):
    """
    Plot ToF-collapsed Doppler MUSIC spectra from each Rx on one figure.

    Each Rx is processed independently as CSI[:, selected_tx, rx:rx+1, :].
    The ToF-Doppler MUSIC heatmap is collapsed over the ToF axis using
    ``method`` and the resulting 1-D Doppler spectra are overlaid.

    Returns:
        fig, ax, fd, spectra_db, rx_peak_fds, rx_peak_indices, tau, P_tof_dop_by_rx
    """
    method = str(method).lower()
    if method not in ("sum", "max", "mean"):
        raise ValueError(f"method must be 'sum', 'max', or 'mean', got {method!r}")

    csi_block = _select_csi_tx_block(CSI, tx=tx)
    num_rx = csi_block.shape[2]
    if num_rx == 0:
        raise ValueError("Need at least one Rx to compare ToF-Doppler spectra")

    rx_count = min(8, num_rx)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    fd_ref = None
    tau_ref = None
    spectra_db = []
    rx_peak_fds = []
    rx_peak_indices = []
    P_tof_dop_by_rx = []

    for rx_idx in range(rx_count):
        csi_rx = csi_block[:, :, rx_idx:rx_idx + 1, :]
        tof_dop = MUSIC.ToF_Dop(args)
        Rxx = tof_dop.Rxx_smooth(csi_rx, frame_idx)
        if Rxx is None:
            print(f"Skip Rx {rx_idx}: not enough CSI samples")
            continue

        tau, fd, P_tof_dop = tof_dop.cal_spectrum(Rxx)
        if fd_ref is None:
            fd_ref = fd
            tau_ref = tau
        elif len(fd_ref) != len(fd) or not np.allclose(fd_ref, fd):
            raise ValueError("Doppler grid changed across Rx; cannot overlay spectra")

        if method == "sum":
            spectrum = np.sum(P_tof_dop, axis=0)
        elif method == "max":
            spectrum = np.max(P_tof_dop, axis=0)
        else:
            spectrum = np.mean(P_tof_dop, axis=0)

        spectrum_db = 10.0 * np.log10(np.maximum(spectrum, 1e-12))
        finite = spectrum_db[np.isfinite(spectrum_db)]
        if finite.size == 0:
            peak_indices = np.array([], dtype=int)
        else:
            median = np.median(finite)
            mad = np.median(np.abs(finite - median))
            noise_scale = 1.4826 * mad
            height = median + max(3.0 * noise_scale, 3.0)
            prominence = max(noise_scale, 1.0)
            peak_indices, _ = find_peaks(
                spectrum_db,
                height=height,
                prominence=prominence,
            )

        peak_fd = fd[peak_indices]
        color = cmap(rx_idx % 10)
        ax.plot(fd, spectrum_db, linewidth=1.25, color=color, label=f"Rx {rx_idx}")
        if peak_indices.size:
            ax.scatter(
                peak_fd,
                spectrum_db[peak_indices],
                color=color,
                marker="x",
                s=36,
                zorder=3,
            )

        spectra_db.append(spectrum_db)
        rx_peak_fds.append(peak_fd)
        rx_peak_indices.append(peak_indices)
        P_tof_dop_by_rx.append(P_tof_dop)
        print(f"Rx {rx_idx} Doppler bins from ToF-Doppler MUSIC (Hz):", np.round(peak_fd, 3))

    if fd_ref is None:
        plt.close(fig)
        return None

    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Doppler frequency (Hz)")
    ax.set_ylabel("Collapsed MUSIC spectrum (dB)")
    ax.set_title(
        f"ToF-Doppler MUSIC -> Doppler spectra by Rx ({method}) @ frame {frame_idx}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    save_dir = getattr(args, "pics_dir", None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"{int(frame_idx):04d}_tof_doppler_{method}_doppler_spectrum_rx_diff.png",
        )
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Saved: {save_path}")

    return (
        fig,
        ax,
        fd_ref,
        np.asarray(spectra_db),
        rx_peak_fds,
        rx_peak_indices,
        tau_ref,
        P_tof_dop_by_rx,
    )
