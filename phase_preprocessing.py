"""
Phase preprocessing utilities for CSI with shape:
    (frame, tx, rx, subcarrier)

Default Rx grouping assumes two Rx chains per NIC:
    NIC 0: Rx 0, Rx 1
    NIC 1: Rx 2, Rx 3
    NIC 2: Rx 4, Rx 5
    ...

Important:
- remove_nic_common_phase() removes only the time-varying common intercept
  drift of each NIC, so the within-NIC constant Rx phase difference is
  preserved.
- remove_SFO_PDD(mode="nic_common") removes only the time-varying common
  slope drift of each NIC. The static subcarrier slope is preserved.
- temporal_smooth_phase() performs circular moving-average smoothing along
  the frame axis for visualization.
- remove_SFO_PDD(mode="per_rx") is still available for visualization-only
  detrending, but it also removes physical ToF/cable-delay slope.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi]."""
    return np.angle(np.exp(1j * phase))


def _validate_phase_4d(csi_phase: np.ndarray) -> np.ndarray:
    """Validate phase shape (frame, tx, rx, subcarrier)."""
    phase = np.asarray(csi_phase, dtype=float)

    if phase.ndim != 4:
        raise ValueError(
            "Expected phase shape (frame, tx, rx, subcarrier), "
            f"but got {phase.shape}"
        )

    if phase.shape[-1] < 1:
        raise ValueError("The subcarrier dimension cannot be empty.")

    return phase


def _resolve_frame_times(
    num_frames: int,
    frame_times: np.ndarray | None = None,
) -> np.ndarray:
    """Return a zero-based frame-time axis."""
    if num_frames < 1:
        raise ValueError("The frame dimension cannot be empty.")

    if frame_times is None:
        times = np.arange(num_frames, dtype=float)
    else:
        times = np.asarray(frame_times, dtype=float)

    if times.shape != (num_frames,):
        raise ValueError(
            f"frame_times must have shape ({num_frames},), but got {times.shape}"
        )

    if not np.all(np.isfinite(times)):
        raise ValueError("frame_times contains NaN or infinite values.")

    times = times - times[0]

    if num_frames >= 2 and np.allclose(times, times[0]):
        raise ValueError("frame_times must vary across frames.")

    return times


def _fit_time_slope(
    values: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Fit values[frame, ...] ~= slope[...] * t + intercept[...] and return slope."""
    values = np.asarray(values, dtype=float)

    if values.shape[0] != frame_times.shape[0]:
        raise ValueError(
            "The leading dimension of values must match frame_times: "
            f"{values.shape[0]} vs {frame_times.shape[0]}"
        )

    if values.shape[0] < 2:
        return np.zeros(values.shape[1:], dtype=float)

    centered_times = frame_times - np.mean(frame_times)
    denominator = np.sum(centered_times**2)
    if denominator <= 0:
        return np.zeros(values.shape[1:], dtype=float)

    mean = np.mean(values, axis=0, keepdims=True)
    time_view = centered_times.reshape((-1,) + (1,) * (values.ndim - 1))

    slope = np.sum((values - mean) * time_view, axis=0) / denominator
    return slope


def _resolve_rx_groups(
    num_rx: int,
    rx_groups: Sequence[Sequence[int]] | None = None,
    rx_per_nic: int = 2,
) -> list[np.ndarray]:
    """Validate user-provided Rx groups or build contiguous default groups."""
    if rx_groups is None:
        if rx_per_nic < 1:
            raise ValueError("rx_per_nic must be at least 1.")

        if num_rx % rx_per_nic != 0:
            raise ValueError(
                f"num_rx={num_rx} is not divisible by rx_per_nic={rx_per_nic}. "
                "Please pass rx_groups explicitly."
            )

        rx_groups = [
            list(range(start, start + rx_per_nic))
            for start in range(0, num_rx, rx_per_nic)
        ]

    normalized_groups: list[np.ndarray] = []
    used_rx: set[int] = set()

    for group_index, group in enumerate(rx_groups):
        group_array = np.asarray(group, dtype=int).reshape(-1)

        if group_array.size == 0:
            raise ValueError(f"Rx group {group_index} is empty.")

        if np.any(group_array < 0) or np.any(group_array >= num_rx):
            raise ValueError(
                f"Rx group {group_index} contains an invalid index: "
                f"{group_array.tolist()}; valid range is [0, {num_rx - 1}]."
            )

        if np.unique(group_array).size != group_array.size:
            raise ValueError(
                f"Rx group {group_index} contains duplicate indices: "
                f"{group_array.tolist()}."
            )

        overlap = used_rx.intersection(group_array.tolist())
        if overlap:
            raise ValueError(
                f"Rx indices {sorted(overlap)} appear in more than one NIC group."
            )

        used_rx.update(group_array.tolist())
        normalized_groups.append(group_array)

    missing_rx = sorted(set(range(num_rx)) - used_rx)
    if missing_rx:
        raise ValueError(
            f"Rx indices {missing_rx} are not assigned to any NIC group."
        )

    return normalized_groups


def estimate_subcarrier_line(
    csi_phase: np.ndarray,
    subcarrier_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a linear phase model along the subcarrier axis.

    Model:
        phase[..., k] ~= intercept[...] + slope[...] * centered_index[k]

    Returns
    -------
    slope:
        Shape = csi_phase.shape[:-1]
    intercept:
        Shape = csi_phase.shape[:-1]
    """
    phase = np.asarray(csi_phase, dtype=float)

    if phase.ndim < 2:
        raise ValueError(
            "estimate_subcarrier_line expects shape (..., subcarrier), "
            f"but got {phase.shape}"
        )

    num_subcarriers = phase.shape[-1]

    if num_subcarriers < 2:
        slope = np.zeros(phase.shape[:-1], dtype=float)
        intercept = phase[..., 0].copy()
        return slope, intercept

    if not np.all(np.isfinite(phase)):
        raise ValueError(
            "csi_phase contains NaN or infinite values. "
            "Clean or interpolate invalid CSI bins before phase fitting."
        )

    if subcarrier_indices is None:
        subcarrier_indices = np.arange(num_subcarriers, dtype=float)
    else:
        subcarrier_indices = np.asarray(subcarrier_indices, dtype=float)

    if subcarrier_indices.shape != (num_subcarriers,):
        raise ValueError(
            "subcarrier_indices must have shape "
            f"({num_subcarriers},), but got {subcarrier_indices.shape}"
        )

    centered_indices = subcarrier_indices - np.mean(subcarrier_indices)
    denominator = np.sum(centered_indices**2)

    if denominator <= 0:
        raise ValueError("subcarrier_indices must contain at least two values.")

    phase_unwrapped = np.unwrap(phase, axis=-1)
    intercept = np.mean(phase_unwrapped, axis=-1)

    slope = np.sum(
        (phase_unwrapped - intercept[..., None]) * centered_indices,
        axis=-1,
    ) / denominator

    return slope, intercept


def estimate_phase_slope_intercept(
    csi_phase: np.ndarray,
    subcarrier_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Alias kept for the CFO/SFO derivation terminology."""
    return estimate_subcarrier_line(
        csi_phase,
        subcarrier_indices=subcarrier_indices,
    )


def temporal_smooth_phase(
    csi_phase: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Smooth wrapped phase over time using a circular moving average.

    Input shape:
        (frame, tx, rx, subcarrier)
    """
    phase = _validate_phase_4d(csi_phase)

    window_size = int(round(window_size))
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, but got {window_size}")

    if window_size == 1 or phase.shape[0] < 2:
        return phase.copy()

    kernel = np.ones(window_size, dtype=float) / window_size
    unit_phasor = np.exp(1j * phase)

    smoothed_phasor = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"),
        axis=0,
        arr=unit_phasor,
    )

    return np.angle(smoothed_phasor)


def remove_nic_common_phase(
    csi_phase: np.ndarray,
    frame_times: np.ndarray | None = None,
    rx_groups: Sequence[Sequence[int]] | None = None,
    rx_per_nic: int = 2,
    return_drift: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Remove the per-frame common intercept drift for each NIC.

    Input shape:
        (frame, tx, rx, subcarrier)

    When rx_groups is omitted, contiguous pairs are used:
        [[0, 1], [2, 3], [4, 5], [6, 7], ...]

    The same intercept correction is applied to all Rx chains in one NIC, so
    the within-NIC constant Rx phase difference is preserved.
    """
    phase = _validate_phase_4d(csi_phase)
    num_rx = phase.shape[2]
    groups = _resolve_rx_groups(num_rx, rx_groups, rx_per_nic)

    corrected = phase.copy()
    nic_common_drifts: list[np.ndarray] = []

    for group in groups:
        group_phase = phase[:, :, group, :]
        group_phase_unwrapped = np.unwrap(group_phase, axis=0)

        # The NIC-common intercept is the median frame-wise offset across all
        # Rx chains in the group, measured relative to the first frame.
        _, rx_intercepts = estimate_phase_slope_intercept(group_phase)
        rx_intercepts = np.unwrap(rx_intercepts, axis=0)

        delta_intercepts = rx_intercepts - rx_intercepts[0:1, :, :]
        nic_common_delta = np.median(delta_intercepts, axis=2)

        corrected[:, :, group, :] = _wrap_phase(
            group_phase_unwrapped - nic_common_delta[:, :, None, None]
        )
        nic_common_drifts.append(nic_common_delta)

    stacked_drifts = np.stack(nic_common_drifts, axis=2)
    return (corrected, stacked_drifts) if return_drift else corrected


def remove_SFO_PDD(
    csi_phase: np.ndarray,
    subcarrier_indices: np.ndarray | None = None,
    *,
    mode: str = "per_rx",
    frame_times: np.ndarray | None = None,
    rx_groups: Sequence[Sequence[int]] | None = None,
    rx_per_nic: int = 2,
    return_slope: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Remove a linear phase trend along the subcarrier axis.

    Parameters
    ----------
    mode : {"per_rx", "nic_common"}
        per_rx:
            Remove a separate slope from every (frame, tx, rx) vector.
            This is intended for visualization only. It removes physical
            ToF/cable-delay slope in addition to SFO/PDD-like slope.

        nic_common:
            Estimate one common slope drift for all Rx chains in each NIC and
            subtract only the time-varying part from the entire NIC group.

    return_slope:
        If True, return (corrected_phase, estimated_slope).
    """
    phase = np.asarray(csi_phase, dtype=float)

    if phase.ndim < 2:
        raise ValueError(
            "remove_SFO_PDD expects shape (..., subcarrier), "
            f"but got {phase.shape}"
        )

    if mode not in {"per_rx", "nic_common"}:
        raise ValueError(
            f"Unsupported mode={mode!r}; use 'per_rx' or 'nic_common'."
        )

    num_subcarriers = phase.shape[-1]

    if num_subcarriers < 2:
        corrected = phase.copy()
        empty_slope = np.zeros(phase.shape[:-1], dtype=float)
        return (corrected, empty_slope) if return_slope else corrected

    if subcarrier_indices is None:
        subcarrier_indices = np.arange(num_subcarriers, dtype=float)
    else:
        subcarrier_indices = np.asarray(subcarrier_indices, dtype=float)

    if subcarrier_indices.shape != (num_subcarriers,):
        raise ValueError(
            "subcarrier_indices must have shape "
            f"({num_subcarriers},), but got {subcarrier_indices.shape}"
        )

    centered_indices = subcarrier_indices - np.mean(subcarrier_indices)
    phase_unwrapped = np.unwrap(phase, axis=-1)

    if mode == "per_rx":
        slope, _ = estimate_subcarrier_line(
            phase_unwrapped,
            subcarrier_indices=subcarrier_indices,
        )

        linear_trend = slope[..., None] * centered_indices
        corrected = _wrap_phase(phase_unwrapped - linear_trend)

        return (corrected, slope) if return_slope else corrected

    # mode == "nic_common"
    phase_4d = _validate_phase_4d(phase)
    frame_axis = _resolve_frame_times(phase_4d.shape[0], frame_times)
    num_rx = phase_4d.shape[2]
    groups = _resolve_rx_groups(num_rx, rx_groups, rx_per_nic)

    corrected = phase_unwrapped.copy()
    nic_slopes: list[np.ndarray] = []
    time_view = frame_axis[:, None]

    for group in groups:
        group_phase = phase_unwrapped[:, :, group, :]

        rx_slopes, _ = estimate_phase_slope_intercept(
            group_phase,
            subcarrier_indices=subcarrier_indices,
        )

        # Preserve the static slope and remove only its common time drift.
        delta_rx_slopes = rx_slopes - rx_slopes[0:1, :, :]
        common_slope = np.median(delta_rx_slopes, axis=2)
        sfo_rate = _fit_time_slope(common_slope, frame_axis)
        common_slope_drift = time_view * sfo_rate[None, :]

        linear_trend = (
            common_slope_drift[:, :, None, None]
            * centered_indices[None, None, None, :]
        )

        corrected[:, :, group, :] = group_phase - linear_trend
        nic_slopes.append(common_slope)

    corrected = _wrap_phase(corrected)
    stacked_nic_slopes = np.stack(nic_slopes, axis=2)

    return (
        (corrected, stacked_nic_slopes)
        if return_slope
        else corrected
    )


def sanitize_phase(
    csi_phase: np.ndarray,
    frame_times: np.ndarray | None = None,
    rx_groups: Sequence[Sequence[int]] | None = None,
    rx_per_nic: int = 2,
    subcarrier_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Preserve within-NIC phase structure while removing common CFO/SFO drifts."""
    corrected = remove_nic_common_phase(
        csi_phase,
        frame_times=frame_times,
        rx_groups=rx_groups,
        rx_per_nic=rx_per_nic,
    )

    corrected = remove_SFO_PDD(
        corrected,
        subcarrier_indices=subcarrier_indices,
        mode="nic_common",
        frame_times=frame_times,
        rx_groups=rx_groups,
        rx_per_nic=rx_per_nic,
    )

    return corrected
