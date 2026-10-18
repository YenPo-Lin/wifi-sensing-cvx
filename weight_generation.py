import numpy as np
from scipy.optimize import minimize
import pre_processing as pp


def estimate_sinr_parameters(segment, candidate_fds, fs):
    """Fit local amplitude sinusoids and return power and residual variance.

    ``segment`` has shape ``(time, ...)`` and contains real dynamic-amplitude
    residuals. Powers are mean squared sinusoid amplitudes over its channels.
    The residual variance includes unmodelled motion and preprocessing error;
    it is an effective noise estimate for the simplified SINR model.
    """
    segment = np.asarray(segment)
    fds = np.asarray(candidate_fds, dtype=float)
    fs = float(fs)
    if segment.ndim < 2 or segment.shape[0] == 0 or np.iscomplexobj(segment):
        raise ValueError("segment must be nonempty real data with time on axis 0")
    if not np.all(np.isfinite(segment)):
        raise ValueError("segment must contain only finite values")
    if fds.ndim != 1 or fds.size == 0 or not np.all(np.isfinite(fds)):
        raise ValueError("candidate_fds must be a nonempty finite 1-D array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")

    n = segment.shape[0]
    parameter_count = 1 + 2 * fds.size  # Per-channel intercept, cos, and sin.
    if n <= parameter_count:
        raise ValueError("segment is too short to estimate noise variance")

    phase = 2.0 * np.pi * np.outer(np.arange(n) / fs, fds)
    design = np.empty((n, parameter_count), dtype=float)
    design[:, 0] = 1.0
    design[:, 1::2] = np.cos(phase)
    design[:, 2::2] = np.sin(phase)
    channels = segment.reshape(n, -1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, channels, rcond=None)
    if rank != parameter_count:
        raise ValueError(
            "candidate Dopplers cannot be separated in this time window "
            "(check duplicate or positive/negative mirror frequencies)"
        )

    powers = np.mean(
        coefficients[1::2] ** 2 + coefficients[2::2] ** 2, axis=1
    )
    residual = channels - design @ coefficients
    noise_variance = float(
        np.sum(residual ** 2) /
        (channels.shape[1] * (n - parameter_count))
    )
    return powers, noise_variance

def weight_gen(target_fd, interferer_fds, interferer_powers, noise_variance, avg_frames, fs):
    """Generate the nonnegative, target-specific SINR-optimal time window.

    This solves the fixed-gain, matched-Doppler model in Research0916 (5).md.
    ``interferer_fds`` may include both sidebands of each real-amplitude
    interferer, with its squared amplitude in ``interferer_powers`` for each
    sideband. The target's own negative-frequency mirror is still neglected
    in the fixed-gain numerator. Powers and noise variance must be in the
    amplitude domain; MUSIC/CFAR pseudo-spectrum values are not estimates.

    Parameters
    ----------
    target_fd : float
        Target Doppler frequency in Hz.
    interferer_fds, interferer_powers : 1-D array_like, shape (num_interferers,)
        Other Doppler frequencies in Hz and their nonnegative amplitude squares.
    noise_variance : float
        Nonnegative variance of the dynamic-amplitude noise per time sample.
    avg_frames : int
        Number of time samples in the projection window.
    fs : float
        Time-sample rate in Hz.

    Returns
    -------
    ndarray, shape (avg_frames,)
        Real nonnegative weights whose sum is ``avg_frames``.
    """
    if (
        isinstance(avg_frames, (bool, np.bool_))
        or not isinstance(avg_frames, (int, np.integer))
        or avg_frames < 1
    ):
        raise ValueError("avg_frames must be a positive integer")
    n = int(avg_frames)

    target_fd = float(target_fd)
    fs = float(fs)
    noise_variance = float(noise_variance)
    if not np.isfinite(target_fd):
        raise ValueError("target_fd must be finite")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be nonnegative and finite")

    fds = np.asarray(interferer_fds, dtype=float)
    powers = np.asarray(interferer_powers, dtype=float)
    if fds.ndim != 1 or powers.ndim != 1 or fds.shape != powers.shape:
        raise ValueError("interferer_fds and interferer_powers must be equal-length 1-D arrays")
    if not np.all(np.isfinite(fds)):
        raise ValueError("interferer_fds must contain only finite values")
    if not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("interferer_powers must be nonnegative and finite")

    # With no effective interference, equal weights minimize the noise term.
    if not np.any(powers > 0):
        return np.ones(n, dtype=float)

    phase = (2.0 * np.pi / fs) * np.outer(fds - target_fd, np.arange(n))
    cos_terms = np.cos(phase)
    sin_terms = np.sin(phase)
    r_q = (cos_terms.T * powers) @ cos_terms
    r_q += (sin_terms.T * powers) @ sin_terms
    r_q.flat[::n + 1] += 4.0 * noise_variance

    ones = np.ones(n, dtype=float)
    if noise_variance > 0:
        try:
            unconstrained = np.linalg.solve(r_q, ones)
        except np.linalg.LinAlgError:
            pass  # Near-singular numerical case: solve the constrained QP below.
        else:
            unconstrained *= n / unconstrained.sum()
            if np.min(unconstrained) >= -1e-10:
                weights = np.maximum(unconstrained, 0.0)
                return weights * (n / weights.sum())

    # The equality-only KKT solution can contain negative weights. In that
    # case solve the full nonnegative quadratic program, rather than clipping.
    scale = max(float(np.linalg.norm(r_q, ord=np.inf)), np.finfo(float).tiny)
    scaled_r = r_q / scale
    result = minimize(
        fun=lambda w: float(w @ scaled_r @ w),
        x0=ones,
        jac=lambda w: 2.0 * (scaled_r @ w),
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - n),
            "jac": lambda w: ones,
        }],
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    if not result.success or np.min(result.x) < -1e-7 or abs(np.sum(result.x) - n) > 1e-6:
        raise RuntimeError(f"SINR window optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)  # Remove only solver roundoff.
    return weights * (n / weights.sum())


def weight_test(raw_CSI, args):
    import MUSIC

    CSI = np.abs(raw_CSI)  # Dynamic amplitude, not linear power.
    background = pp.MA(CSI, args.avg_frames*0.1) # moving average background

    if args.preprocess == "ma":
        CSI = (CSI - background) # Original Version

        
    azi_tof = MUSIC.Azi_ToF(args)

    frame_idx = args.frame_idx
    hann_results = azi_tof.gen_Doppler_projection_spectrum(
        CSI, 
        frame_idx, 
        fd_neighbor=0.5, 
        doppler_step=0.25,
        window="hann",
        normalize=False,
        fig_name="Project_tg",
    )

    weighted_results = azi_tof.gen_Doppler_projection_spectrum(
        CSI, 
        frame_idx, 
        fd_neighbor=0.5, 
        doppler_step=0.25,
        window="weight",
        normalize=False,
        fig_name="Project_tg_weight",
    )
    return {"hann": hann_results, "weight": weighted_results}
