"""Utilities shared by the controlled CSI simulations."""

import numpy as np
import MUSIC


def doppler_projection(CSI, target_fd, args, window="hann", normalize=False):
    """Project every dynamic-power CSI channel onto one signed Doppler.

    This implements

        Z(fd) = sum_n h[n] * CSI[n] * exp(-j * 2*pi*fd*t[n]).

    The first axis of ``CSI`` is time; all remaining axes are treated as
    independent channels and are preserved in the returned coefficient map.
    For the simulator's ``(frame, Tx, Rx, subcarrier)`` input, the output has
    shape ``(Tx, Rx, subcarrier)``.  With ``num_Tx == 1``, ``Z_fd[0]`` is the
    ``(Rx, subcarrier)`` map used in the research formulation.

    The projection interval is centred on ``frame_idx`` and contains exactly
    ``avg_frames`` samples whenever the recording is long enough.  Near either
    boundary the interval is shifted, rather than shortened.  If the complete
    recording is shorter than ``avg_frames``, all available frames are used;
    this matches ``MUSIC.Azi_ToF.sample_csi_segment``.

    Parameters
    ----------
    CSI : array_like, shape (num_frames, ...)
        Real-valued dynamic linear-power CSI (for example, after background
        subtraction).  This is not complex CSI reconstruction.
    target_fd : float
        Signed target Doppler in Hz, e.g. ``target['fd']``.
    frame_idx : int
        Frame about which the projection context is selected.
    avg_frames : int
        Requested number of projection frames.
    fs : float
        CSI frame sampling rate in Hz.
    window : {'hann', 'rect'}, default='hann'
        Temporal weight ``h[n]``. ``'rect'`` uses equal weights.
    normalize : bool, default=False
        Divide by ``sum(h)`` when True.  The default returns the unnormalised
        sum written in the research equation; covariance-based MUSIC is
        unaffected by this common scalar.

    Returns
    -------
    Z_fd : ndarray, complex
        Doppler-conditioned coefficient map with shape ``CSI.shape[1:]``.
    """
    frame_idx = getattr(args, "plot_frame", getattr(args, "frame_idx", None))
    if frame_idx is None:
        raise AttributeError("args must define plot_frame or frame_idx")
    avg_frames = args.avg_frames
    fs = args.fs

    csi = np.asarray(CSI)
    if csi.ndim < 2:
        raise ValueError(
            "CSI must have time as its first axis and at least one channel axis"
        )
    if csi.shape[0] == 0:
        raise ValueError("CSI must contain at least one frame")
    if np.iscomplexobj(csi):
        raise ValueError(
            "CSI must be real dynamic linear power, not reconstructed complex CSI"
        )
    if not np.all(np.isfinite(csi)):
        raise ValueError("CSI must contain only finite values")

    target_fd = float(target_fd)
    fs = float(fs)
    avg_frames = int(avg_frames)
    if not np.isfinite(target_fd):
        raise ValueError(f"target_fd must be finite, got {target_fd}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if avg_frames <= 0:
        raise ValueError(f"avg_frames must be positive, got {avg_frames}")

    total_frames = csi.shape[0]
    context_len = min(avg_frames, total_frames)
    frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
    start = int(
        np.clip(
            frame_idx - context_len // 2,
            0,
            total_frames - context_len,
        )
    )
    end = start + context_len

    if window == "hann":
        # np.hanning(2) is all zeros, so use equal weights for tiny contexts.
        weights = (
            np.hanning(context_len)
            if context_len >= 3
            else np.ones(context_len)
        )
    elif window == "rect":
        weights = np.ones(context_len)
    else:
        raise ValueError(f"window must be 'hann' or 'rect', got {window!r}")

    # Absolute sample time preserves a consistent coefficient phase when the
    # projection centre changes.  target_fd keeps its sign here.
    times_s = np.arange(start, end, dtype=float) / fs
    demodulator = weights * np.exp(-1j * 2.0 * np.pi * target_fd * times_s)
    Z_fd = np.tensordot(demodulator, csi[start:end], axes=(0, 0))

    if normalize:
        Z_fd = Z_fd / np.sum(weights)
    return Z_fd


def projected_Azi_ToF(CSI, args, target_fds, window="hann", normalize=False):
    """Calculate one Doppler-conditioned Azi-ToF spectrum per target fd.

    ``doppler_projection`` collapses ``avg_frames`` into one coefficient map
    with shape ``(Tx, Rx, subcarrier)``.  A singleton projected-window axis is
    added before passing the map to ``MUSIC.Azi_ToF``, whose input contract is
    ``(window, Tx, Rx, subcarrier)``.
    """
    target_fds = np.asarray(target_fds, dtype=float).reshape(-1)
    if not np.all(np.isfinite(target_fds)):
        raise ValueError("target_fds must contain only finite values")

    plot_frame_idx = getattr(
        args,
        "plot_frame",
        getattr(args, "frame_idx", None),
    )
    if plot_frame_idx is None:
        raise AttributeError("args must define plot_frame or frame_idx")

    azi_tof = MUSIC.Azi_ToF(args)
    results = []
    for target_idx, fd in enumerate(target_fds, start=1):
        Z_fd = doppler_projection(
            CSI,
            fd,
            args,
            window=window,
            normalize=normalize,
        )
        projected_snapshot = Z_fd[np.newaxis, ...]
        title = (
            f"Projected Azimuth-ToF target {target_idx:02d} "
            f"fd {fd:+.2f} Hz"
        )
        fd_sign = "p" if fd >= 0.0 else "m"
        fd_token = f"{abs(fd):.2f}".replace(".", "p")
        file_name = (
            f"{plot_frame_idx}_projected_target_{target_idx:02d}_"
            f"fd_{fd_sign}{fd_token}Hz.png"
        )
        tau_grid, theta_grid, spectrum_db = azi_tof.gen_spectrum(
            projected_snapshot,
            frame_idx=0,
            title=title,
            plot_frame_idx=plot_frame_idx,
            file_name=file_name,
        )
        results.append(
            {
                "fd": float(fd),
                "Z_fd": Z_fd,
                "tau_grid": tau_grid,
                "theta_grid": theta_grid,
                "spectrum_db": spectrum_db,
                "file_name": file_name,
            }
        )

    return results



def _axis_cell_counts(value, name, *, minimum):
    """Normalize one integer or a ``(tau, fd)`` pair to two cell counts."""
    values = np.asarray(value)
    if values.ndim == 0:
        counts = (int(values), int(values))
    elif values.shape == (2,):
        counts = (int(values[0]), int(values[1]))
    else:
        raise ValueError(f"{name} must be an integer or a (tau, fd) pair")

    if any(count < minimum for count in counts):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must contain {qualifier} integers, got {counts}")
    return counts


def cfar_2D(
    spectrum,
    tau_grid,
    fd_grid,
    training_cells=3,
    guard_cells=1,
    threshold_factor=1.5,
    top_k=None,
    min_peak_distance=1,
):
    """Detect 2-D CA-CFAR peaks in a linear-power ToF-Doppler spectrum.

    Parameters
    ----------
    spectrum : array_like, shape (num_tau, num_fd)
        Linear MUSIC power. CFAR averaging must not be performed in dB.
    tau_grid : array_like, shape (num_tau,)
        ToF grid in seconds.
    fd_grid : array_like, shape (num_fd,)
        Signed Doppler grid in Hz.
    training_cells, guard_cells : int or tuple[int, int]
        Cell counts along the ``(tau, fd)`` axes. ``training_cells`` is the
        number of training layers outside the guard region.
    threshold_factor : float
        A cell under test must exceed ``threshold_factor * noise_power``.
    top_k : int or None
        Optional maximum number of returned CFAR detections.
    min_peak_distance : int or tuple[int, int]
        Non-maximum-suppression distance along ``(tau, fd)`` grid indices.

    Returns
    -------
    list[dict]
        Target dictionaries sorted by ``power_db`` from high to low. Each
        dictionary contains ``tau`` (seconds), ``fd`` (signed Hz), linear
        ``power``, ``power_db``, ``snr_db``, the local noise/threshold values,
        and the corresponding grid indices.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    fd_grid = np.asarray(fd_grid, dtype=float)

    if spectrum.ndim != 2:
        raise ValueError(f"spectrum must be 2-D, got shape {spectrum.shape}")
    expected_shape = (tau_grid.size, fd_grid.size)
    if spectrum.shape != expected_shape:
        raise ValueError(
            "spectrum shape must be (len(tau_grid), len(fd_grid)); "
            f"expected {expected_shape}, got {spectrum.shape}"
        )
    if tau_grid.ndim != 1 or fd_grid.ndim != 1:
        raise ValueError("tau_grid and fd_grid must both be 1-D")
    if spectrum.size == 0:
        return []
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum must contain only finite values")
    if np.any(spectrum < 0.0):
        raise ValueError("spectrum must contain non-negative linear power")

    train_tau, train_fd = _axis_cell_counts(
        training_cells, "training_cells", minimum=1
    )
    guard_tau, guard_fd = _axis_cell_counts(
        guard_cells, "guard_cells", minimum=0
    )
    distance_tau, distance_fd = _axis_cell_counts(
        min_peak_distance, "min_peak_distance", minimum=1
    )

    threshold_factor = float(threshold_factor)
    if not np.isfinite(threshold_factor) or threshold_factor <= 0.0:
        raise ValueError(
            f"threshold_factor must be positive and finite, got {threshold_factor}"
        )
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

    margin_tau = train_tau + guard_tau
    margin_fd = train_fd + guard_fd
    if spectrum.shape[0] < 2 * margin_tau + 1:
        raise ValueError("tau axis is too short for the requested CFAR window")
    if spectrum.shape[1] < 2 * margin_fd + 1:
        raise ValueError("fd axis is too short for the requested CFAR window")

    # Match the dB floor used by the MUSIC plotting path.
    eps = 1e-12
    detections = []
    for tau_idx in range(margin_tau, spectrum.shape[0] - margin_tau):
        for fd_idx in range(margin_fd, spectrum.shape[1] - margin_fd):
            patch = spectrum[
                tau_idx - margin_tau : tau_idx + margin_tau + 1,
                fd_idx - margin_fd : fd_idx + margin_fd + 1,
            ]

            training_mask = np.ones(patch.shape, dtype=bool)
            center_tau = margin_tau
            center_fd = margin_fd
            training_mask[
                center_tau - guard_tau : center_tau + guard_tau + 1,
                center_fd - guard_fd : center_fd + guard_fd + 1,
            ] = False
            noise_power = float(np.mean(patch[training_mask]))
            threshold = threshold_factor * noise_power
            cut_power = float(spectrum[tau_idx, fd_idx])

            local_patch = spectrum[
                tau_idx - 1 : tau_idx + 2,
                fd_idx - 1 : fd_idx + 2,
            ]
            is_local_max = cut_power >= float(np.max(local_patch))
            if cut_power <= threshold or not is_local_max:
                continue

            detections.append(
                {
                    "tau": float(tau_grid[tau_idx]),
                    "fd": float(fd_grid[fd_idx]),
                    "power": cut_power,
                    "power_db": float(10.0 * np.log10(max(cut_power, eps))),
                    "noise_power": noise_power,
                    "noise_power_db": float(
                        10.0 * np.log10(max(noise_power, eps))
                    ),
                    "threshold": threshold,
                    "threshold_db": float(
                        10.0 * np.log10(max(threshold, eps))
                    ),
                    "snr_db": float(
                        10.0 * np.log10(max(cut_power, eps) / max(noise_power, eps))
                    ),
                    "tau_idx": int(tau_idx),
                    "fd_idx": int(fd_idx),
                }
            )

    detections.sort(key=lambda target: target["power"], reverse=True)

    # Keep the strongest peak in each local index neighbourhood.
    targets = []
    for detection in detections:
        too_close = any(
            abs(detection["tau_idx"] - selected["tau_idx"]) <= distance_tau
            and abs(detection["fd_idx"] - selected["fd_idx"]) <= distance_fd
            for selected in targets
        )
        if too_close:
            continue
        targets.append(detection)
        if top_k is not None and len(targets) >= top_k:
            break

    for rank, target in enumerate(targets, start=1):
        target["rank"] = rank
    return targets
