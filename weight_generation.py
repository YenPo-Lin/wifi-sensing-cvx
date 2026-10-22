import numpy as np
from scipy.optimize import minimize
import pre_processing as pp


def estimate_sinr_parameters(segment, candidate_fds, fs, *, times_s=None, return_offset=False):
    """Fit a real dynamic amplitude or linear-power model at every channel.

    ``segment[n, ...] = offset + sum_p(C[p, ...] exp(j 2 pi f_p t_n)
    + conj(C[p, ...]) exp(-j 2 pi f_p t_n)) + eta[n, ...]``.
    Return complex ``C`` with shape ``(num_paths, *segment.shape[1:])`` and
    residual noise variance with shape ``segment.shape[1:]``. Set
    ``return_offset=True`` to also return the fitted per-channel DC term.
    Each absolute Doppler may occur only once: a real signal cannot
    distinguish +f/-f in this fit.
    ``times_s`` must use the same time origin later passed to ``weight_gen``.
    """
    segment = np.asarray(segment)
    fds = np.asarray(candidate_fds, dtype=float)
    fs = float(fs)
    if segment.ndim < 1 or segment.shape[0] == 0 or np.iscomplexobj(segment):
        raise ValueError("segment must be nonempty real amplitude or linear power with time on axis 0")
    if not np.all(np.isfinite(segment)):
        raise ValueError("segment must contain only finite values")
    if fds.ndim != 1 or fds.size == 0 or not np.all(np.isfinite(fds)):
        raise ValueError("candidate_fds must be a nonempty finite 1-D array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")

    n = segment.shape[0]
    if times_s is None:
        times_s = np.arange(n, dtype=float) / fs
    else:
        times_s = np.asarray(times_s, dtype=float)
        if times_s.shape != (n,) or not np.all(np.isfinite(times_s)):
            raise ValueError("times_s must be a finite 1-D array of segment length")
    parameter_count = 1 + 2 * fds.size  # Per-channel intercept, cos, and sin.
    if n <= parameter_count:
        raise ValueError("segment is too short to estimate noise variance")

    phase = 2.0 * np.pi * np.outer(times_s, fds)
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

    # A cos(wt) + B sin(wt) = C exp(jwt) + C* exp(-jwt),
    # so C = (A - j B) / 2; keep its phase for coherent interference.
    path_coefficients = (coefficients[1::2] - 1j * coefficients[2::2]) / 2.0
    residual = channels - design @ coefficients
    noise_variance = np.sum(residual ** 2, axis=0) / (n - parameter_count)
    result = (
        path_coefficients.reshape((fds.size,) + segment.shape[1:]),
        noise_variance.reshape(segment.shape[1:]),
    )
    if return_offset:
        return result + (coefficients[0].reshape(segment.shape[1:]),)
    return result

def weight_gen(
    target_fd, candidate_fds, path_coefficients, noise_variance, avg_frames, fs,
    *, channel_weights=None, times_s=None, offset=None,
):
    """Return one shared, target-specific matched-Doppler SINR/QP window.

    Minimize the sum of channel interference-plus-noise powers, with
    ``u >= 0`` and ``sum(u) = avg_frames``. Matched Doppler fixes each
    matched target gain at ``C[q, ...]``. The target's negative
    mirror and both branches of every other path remain coherent inside
    ``|I_q|**2``; no phase-independence approximation is made.

    ``candidate_fds`` has shape ``(P,)`` and includes ``target_fd`` exactly
    once. It contains one signed representative for each absolute Doppler.
    ``path_coefficients`` has shape ``(P, ...)`` and holds complex
    ``C[p, ...]`` from real dynamic CSI amplitude or linear power, not complex
    CSI or averaged path powers. ``noise_variance`` and optional nonnegative
    ``channel_weights`` are scalar or broadcastable to the channel shape
    ``...``. Equal channel weights are used by default. ``times_s`` is an
    optional length-``avg_frames`` time vector, with the same origin as C.
    ``offset`` is an optional fitted real DC term; it is included in coherent
    interference when the projected amplitude residual has a local offset.
    The noise model assumes independent time samples at each channel.

    Returns a real array of shape ``(avg_frames,)`` with nonnegative entries
    and sum ``avg_frames``, suitable for a single shared Doppler projection.
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
    if not np.isfinite(target_fd):
        raise ValueError("target_fd must be finite")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    fds = np.asarray(candidate_fds, dtype=float)
    if fds.ndim != 1 or fds.size == 0 or not np.all(np.isfinite(fds)):
        raise ValueError("candidate_fds must be a nonempty finite 1-D array")
    matches = np.flatnonzero(np.isclose(fds, target_fd, rtol=1e-9, atol=1e-9))
    if matches.size != 1:
        raise ValueError("candidate_fds must contain target_fd exactly once")
    mirrored_duplicates = np.isclose(
        np.abs(fds[:, None]), np.abs(fds[None, :]), rtol=1e-9, atol=1e-9
    )
    if np.any(np.triu(mirrored_duplicates, k=1)):
        raise ValueError("candidate_fds must have unique absolute Dopplers")
    if abs(target_fd) < 1e-12:
        raise ValueError("target_fd must be nonzero to separate its two branches")

    c = np.asarray(path_coefficients, dtype=complex)
    if c.ndim < 1 or c.shape[0] != fds.size or not np.all(np.isfinite(c)):
        raise ValueError("path_coefficients must be finite complex C with first axis matching candidate_fds")
    if not np.any(np.abs(c[matches[0]]) > 0):
        raise ValueError("target path coefficient must be nonzero in at least one channel")
    channel_shape = c.shape[1:]
    if offset is not None and np.iscomplexobj(offset):
        raise ValueError("offset must be real")
    try:
        sigma2 = np.broadcast_to(np.asarray(noise_variance, dtype=float), channel_shape)
        lambdas = np.broadcast_to(
            np.asarray(1.0 if channel_weights is None else channel_weights, dtype=float),
            channel_shape,
        )
        if offset is not None:
            dc = np.broadcast_to(np.asarray(offset, dtype=float), channel_shape)
    except ValueError as exc:
        raise ValueError("noise_variance, channel_weights, and offset must match the channel shape") from exc
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 < 0):
        raise ValueError("noise_variance must be finite and nonnegative")
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0) or not np.any(lambdas > 0):
        raise ValueError("channel_weights must be finite, nonnegative, and not all zero")
    if offset is not None and not np.all(np.isfinite(dc)):
        raise ValueError("offset must be finite")

    if times_s is None:
        times_s = np.arange(n, dtype=float) / fs
    else:
        times_s = np.asarray(times_s, dtype=float)
        if times_s.shape != (n,) or not np.all(np.isfinite(times_s)):
            raise ValueError("times_s must be a finite 1-D array of avg_frames length")

    # b[n, channel] is the entire coherent interference term before weighting.
    # Only the matched target branch C_q exp(j 2 pi f_q t) is removed.
    c_flat = c.reshape(fds.size, -1)
    positive = np.exp(1j * 2.0 * np.pi * np.outer(times_s, fds - target_fd))
    negative = np.exp(-1j * 2.0 * np.pi * np.outer(times_s, fds + target_fd))
    b = positive @ c_flat + negative @ c_flat.conj()
    b -= c_flat[matches[0]][None, :]
    if offset is not None:
        b += np.exp(-1j * 2.0 * np.pi * target_fd * times_s)[:, None] * dc.reshape(1, -1)

    # For real u, |u.T b_j|^2 = u.T (Re(b_j)Re(b_j).T + Im(b_j)Im(b_j).T) u.
    # The common N^-2 factor is irrelevant to the minimizer.
    sqrt_lambda = np.sqrt(lambdas.reshape(-1))
    b_re = b.real * sqrt_lambda[None, :]
    b_im = b.imag * sqrt_lambda[None, :]
    r_q = b_re @ b_re.T + b_im @ b_im.T
    r_q.flat[::n + 1] += float(np.sum(lambdas * sigma2))

    ones = np.ones(n, dtype=float)
    if not np.any(r_q):
        return ones
    if np.sum(lambdas * sigma2) > 0:
        try:
            unconstrained = np.linalg.solve(r_q, ones)
        except np.linalg.LinAlgError:
            pass  # Near-singular numerical case: solve the constrained QP below.
        else:
            if unconstrained.sum() > 0:
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
