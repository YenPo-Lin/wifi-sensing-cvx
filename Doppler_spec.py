import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


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


def gen_spectrum(*args, **kwargs):
    return gen_spectrogram(*args, **kwargs)
