"""Utilities shared by the controlled CSI simulations."""

import numpy as np


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


def design_mat(time_s, target_fd_hz=None, *, include_trend=True):
    """Build the local null or target-Doppler regression design matrix.

    ``target_fd_hz=None`` builds the null model ``M0`` with an intercept and,
    optionally, a linear trend. Passing a signed Doppler builds ``M1`` by
    appending cosine and sine columns at that frequency.

    The sample index is centred before it is used as the trend coordinate.
    This preserves the model subspace while improving numerical conditioning.
    """
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size == 0:
        raise ValueError("time_s must be a non-empty 1-D array")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain only finite values")

    columns = [np.ones(time_s.size, dtype=float)]
    if include_trend:
        sample_index = np.arange(time_s.size, dtype=float)
        sample_index -= np.mean(sample_index)
        columns.append(sample_index)

    if target_fd_hz is not None:
        target_fd_hz = float(target_fd_hz)
        if not np.isfinite(target_fd_hz):
            raise ValueError("target_fd_hz must be finite")
        if target_fd_hz == 0.0:
            raise ValueError(
                "target_fd_hz must be non-zero because its cosine column "
                "would duplicate the intercept"
            )
        phase = 2.0 * np.pi * target_fd_hz * time_s
        columns.extend((np.cos(phase), np.sin(phase)))

    return np.column_stack(columns)


def ls_solution(design_matrix, y, weights=None):
    """Solve weighted LS for one or many time-first power sequences.

    ``design_matrix`` has shape ``(num_samples, num_coefficients)`` and ``y``
    may have shape ``(num_samples,)`` or ``(num_samples, ...)``. ``weights`` is
    the diagonal of the research model's weighting matrix ``W``.

    Returns a dictionary containing ``coefficients``, ``fitted``, ``residual``,
    weighted ``sse``, matrix ``rank``, and ``singular_values``. For an ``M1``
    matrix from :func:`design_mat`, the final two coefficients are ``a`` and
    ``b`` respectively.
    """
    design_matrix = np.asarray(design_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    if design_matrix.ndim != 2:
        raise ValueError("design_matrix must be 2-D")
    if y.ndim == 0 or y.shape[0] != design_matrix.shape[0]:
        raise ValueError(
            "the first axis of y must match the number of design-matrix rows"
        )
    if not np.all(np.isfinite(design_matrix)) or not np.all(np.isfinite(y)):
        raise ValueError("design_matrix and y must contain only finite values")

    num_samples, num_coefficients = design_matrix.shape
    if num_samples < num_coefficients:
        raise ValueError(
            "weighted LS requires at least as many samples as coefficients; "
            f"got {num_samples} samples and {num_coefficients} coefficients"
        )

    if weights is None:
        weights = np.ones(num_samples, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (num_samples,):
            raise ValueError(f"weights must have shape ({num_samples},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must contain finite non-negative values")
        if not np.any(weights > 0.0):
            raise ValueError("at least one weight must be positive")

    feature_shape = y.shape[1:]
    y_flat = y.reshape(num_samples, -1)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design_matrix * sqrt_weights[:, None]
    weighted_y = y_flat * sqrt_weights[:, None]

    coefficients_flat, _, rank, singular_values = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None
    )
    fitted_flat = design_matrix @ coefficients_flat
    residual_flat = y_flat - fitted_flat
    sse_flat = np.sum(weights[:, None] * residual_flat**2, axis=0)

    coefficients = coefficients_flat.reshape((num_coefficients,) + feature_shape)
    fitted = fitted_flat.reshape(y.shape)
    residual = residual_flat.reshape(y.shape)
    sse = sse_flat.reshape(feature_shape)
    if not feature_shape:
        sse = float(sse)

    return {
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
        "sse": sse,
        "rank": int(rank),
        "singular_values": singular_values,
    }


def fitting_strength(a, b=None):
    """Return ``R = sqrt(a**2 + b**2)`` for target-Doppler coefficients.

    If ``b`` is omitted, ``a`` is interpreted as an LS coefficient array and
    its final two rows are used as the cosine and sine coefficients.
    """
    if b is None:
        coefficients = np.asarray(a, dtype=float)
        if coefficients.ndim == 0 or coefficients.shape[0] < 2:
            raise ValueError("coefficients must contain cosine and sine rows")
        a, b = coefficients[-2], coefficients[-1]

    strength = np.hypot(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if strength.ndim == 0:
        return float(strength)
    return strength


def fitting_quality(sse_0, sse_1, epsilon=1e-12):
    """Return ``Q = max(0, 1 - SSE1 / (SSE0 + epsilon))``.

    ``SSE0`` is from the offset/trend-only null model and ``SSE1`` is from the
    model that additionally contains the target-Doppler cosine/sine basis.
    Scalars or broadcast-compatible arrays are supported.
    """
    sse_0 = np.asarray(sse_0, dtype=float)
    sse_1 = np.asarray(sse_1, dtype=float)
    epsilon = float(epsilon)
    if not np.all(np.isfinite(sse_0)) or not np.all(np.isfinite(sse_1)):
        raise ValueError("sse_0 and sse_1 must contain only finite values")
    if np.any(sse_0 < 0.0) or np.any(sse_1 < 0.0):
        raise ValueError("sse_0 and sse_1 must be non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    quality = np.maximum(0.0, 1.0 - sse_1 / (sse_0 + epsilon))
    if quality.ndim == 0:
        return float(quality)
    return quality


__all__ = [
    "cfar_2D",
    "design_mat",
    "fitting_quality",
    "fitting_strength",
    "ls_solution",
]
