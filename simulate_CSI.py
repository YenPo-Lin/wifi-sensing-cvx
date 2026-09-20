"""Generate amplitude-only CSI for controlled fixed-target experiments.

The public CSI layout is ``(frame, Tx, Rx, subcarrier)``.  Each target has
constant azimuth, ToF, and Doppler; only its activity envelope changes over
time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MUSIC


def activity_envelope(args, target):
    """Return a smooth 0-to-1 raised-cosine target activity envelope."""
    time_s = np.arange(args.num_frames, dtype=float) / float(args.fs)
    start_s = float(target.get("activity_start_s", time_s[0]))
    end_s = float(target.get("activity_end_s", time_s[-1] + 1.0 / args.fs))
    ramp_s = float(target.get("activity_ramp_s", 0.0))

    if not 0.0 <= start_s < end_s:
        raise ValueError("activity requires 0 <= start_s < end_s")
    if ramp_s < 0.0 or 2.0 * ramp_s > end_s - start_s:
        raise ValueError("activity_ramp_s must fit inside the active interval")

    envelope = np.zeros_like(time_s)
    active = (time_s >= start_s) & (time_s <= end_s)
    envelope[active] = 1.0
    if ramp_s == 0.0:
        return envelope

    rising = (time_s >= start_s) & (time_s < start_s + ramp_s)
    falling = (time_s > end_s - ramp_s) & (time_s <= end_s)
    envelope[rising] = 0.5 - 0.5 * np.cos(
        np.pi * (time_s[rising] - start_s) / ramp_s
    )
    envelope[falling] = 0.5 - 0.5 * np.cos(
        np.pi * (end_s - time_s[falling]) / ramp_s
    )
    return envelope


def target_trajectory(args, target):
    """Return constant target parameters over all simulation frames."""
    missing = {"theta", "tof", "fd", "amplitude"} - target.keys()
    if missing:
        raise ValueError(
            f"target {target.get('name', '<unnamed>')} is missing: "
            + ", ".join(sorted(missing))
        )
    if args.num_frames <= 0 or args.fs <= 0:
        raise ValueError("num_frames and fs must be positive")

    theta_deg = float(target["theta"])
    tof_s = float(target["tof"])
    fd_hz = float(target["fd"])
    amplitude = float(target["amplitude"])
    if not np.all(np.isfinite([theta_deg, tof_s, fd_hz, amplitude])):
        raise ValueError("target theta, tof, fd, and amplitude must be finite")
    if not 0.0 <= theta_deg <= 180.0:
        raise ValueError("target theta must be in [0, 180] degrees")
    if tof_s <= 0.0 or amplitude < 0.0:
        raise ValueError("target tof must be positive and amplitude nonnegative")

    time_s = np.arange(args.num_frames, dtype=float) / float(args.fs)

    return {
        "time_s": time_s,
        "theta": np.full(args.num_frames, theta_deg, dtype=float),
        "tof": np.full(args.num_frames, tof_s, dtype=float),
        "fd": np.full(args.num_frames, fd_hz, dtype=float),
        "activity": activity_envelope(args, target),
    }


def plot_target_trajectories(args, targets, frame_idx):
    """Plot fixed target parameters and activity versus frame."""
    if not 0 <= frame_idx < args.num_frames:
        raise ValueError(
            f"frame_idx must be in [0, {args.num_frames - 1}], got {frame_idx}"
        )

    frame_idx = int(frame_idx)
    frames = np.arange(args.num_frames)
    target_colors = ("red", "blue")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    for target_idx, target in enumerate(targets):
        trajectory = target_trajectory(args, target)
        label = f"Target {target.get('name', target_idx + 1)}"
        if target.get("strength_label"):
            label += f" ({target['strength_label']})"
        color = target_colors[target_idx % len(target_colors)]
        panel_values = (
            trajectory["theta"],
            trajectory["tof"] * 1e9,
            trajectory["fd"],
            trajectory["activity"],
        )

        for panel_idx, values in enumerate(panel_values):
            value_at_frame = float(values[frame_idx])
            axes[panel_idx].plot(frames, values, color=color, label=label)
            axes[panel_idx].scatter(
                frame_idx,
                value_at_frame,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                s=65,
                zorder=4,
            )
            axes[panel_idx].annotate(
                f"{value_at_frame:.2f}",
                xy=(frame_idx, value_at_frame),
                xytext=(6, 7),
                textcoords="offset points",
                color=color,
                fontsize=10,
                #fontweight="bold",
            )

    axes[0].set_ylabel("Azimuth (deg)")
    axes[1].set_ylabel("ToF (ns)")
    axes[2].set_ylabel("Doppler (Hz)")
    axes[3].set_ylabel("Activity")
    axes[3].set_xlabel("Frame")
    axes[0].set_title(f"Fixed target ground truth @ frame {frame_idx}")
    for ax in axes:
        ax.axvline(frame_idx, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.3)
    if targets:
        axes[0].legend(loc="best")

    _save_figure(fig, args, f"{frame_idx:04d}_target_trajectories.png")

    return axes

def _save_figure(fig, args, filename, dpi=120):
    """Save and close a simulation figure when an output directory is set."""
    if args.pics_dir is None:
        return

    save_dir = Path(args.pics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def targets_at_frame(args, targets, frame_idx, activity_threshold=1e-3):
    """Return active targets and their instantaneous parameters."""
    frame_idx = int(np.clip(frame_idx, 0, args.num_frames - 1))
    frame_targets = []
    for target in targets:
        trajectory = target_trajectory(args, target)
        activity = float(trajectory["activity"][frame_idx])
        if activity <= activity_threshold:
            continue
        frame_targets.append(
            {
                **target,
                "theta": float(trajectory["theta"][frame_idx]),
                "tof": float(trajectory["tof"][frame_idx]),
                "fd": float(trajectory["fd"][frame_idx]),
                "activity": activity,
                "nominal_amplitude": float(target["amplitude"]),
                "amplitude": float(target["amplitude"]) * activity,
            }
        )
    return frame_targets


def target_gt_at_frame(args, targets, frame_idx):
    """Return [azi_deg, tof_s, fd_hz] for active heatmap markers."""
    return [
        [target["theta"], target["tof"], target["fd"]]
        for target in targets_at_frame(args, targets, frame_idx)
    ]


def trajectory_csi(args, target):
    """Generate one fixed target's latent complex CSI component."""
    trajectory = target_trajectory(args, target)
    steering = MUSIC.SteeringVector(args)
    a_azi = steering.steering_vector_AoA(
        target["theta"],
        stream_win=args.num_Rx,
    )
    a_tof = steering.steering_vector_ToF(
        target["tof"],
        freq_win=args.num_sc,
        freq_hop=1,
    )
    doppler_phase = np.exp(
        1j * 2.0 * np.pi * float(target["fd"]) * trajectory["time_s"]
    )

    component = (
        float(target["amplitude"])
        * trajectory["activity"][:, None, None, None]
        * doppler_phase[:, None, None, None]
        * np.exp(1j * float(target.get("initial_phase_rad", 0.0)))
        * a_azi[None, None, :, None]
        * a_tof[None, None, None, :]
    )
    return np.broadcast_to(
        component,
        (args.num_frames, args.num_Tx, args.num_Rx, args.num_sc),
    ).copy()


def target_csi(args, targets):
    """Sum the latent complex CSI of all dynamic targets."""
    total = np.zeros(
        (args.num_frames, args.num_Tx, args.num_Rx, args.num_sc),
        dtype=np.complex128,
    )
    for target in targets:
        total += trajectory_csi(args, target)
    return total


def _amplitude_csi(amplitude_csi):
    """Validate real, finite CSI amplitude in (frame, Tx, Rx, subcarrier)."""
    if np.iscomplexobj(amplitude_csi):
        raise ValueError("expected real CSI amplitude, not complex CSI")
    amplitude_csi = np.asarray(amplitude_csi, dtype=float)
    if amplitude_csi.ndim != 4 or amplitude_csi.size == 0:
        raise ValueError(
            "CSI amplitude must have non-empty shape (frame, Tx, Rx, SC); "
            f"got {amplitude_csi.shape}"
        )
    if not np.all(np.isfinite(amplitude_csi)):
        raise ValueError("CSI amplitude contains non-finite values")
    return amplitude_csi


def _normalize_mismatch_db(profile, std_db):
    profile = np.asarray(profile, dtype=float)
    profile -= np.mean(profile)
    current_std = float(np.std(profile))
    if current_std == 0.0 or std_db == 0.0:
        return np.zeros_like(profile)
    return profile * (std_db / current_std)


def Rx_mismatch(amplitude_csi, std_db=0.5, seed=None):
    """Apply a time-invariant amplitude gain to every Rx channel."""
    amplitude_csi = _amplitude_csi(amplitude_csi)
    std_db = float(std_db)
    if std_db < 0.0:
        raise ValueError("std_db must be nonnegative")
    rng = np.random.default_rng(seed)
    gain_db = _normalize_mismatch_db(
        rng.standard_normal(amplitude_csi.shape[2]), std_db
    )
    return amplitude_csi * 10.0 ** (gain_db[None, None, :, None] / 20.0)


def SC_mismatch(amplitude_csi, std_db=0.5, smooth_window=7, seed=None):
    """Apply a smooth, time-invariant amplitude gain over subcarriers."""
    amplitude_csi = _amplitude_csi(amplitude_csi)
    std_db = float(std_db)
    if std_db < 0.0:
        raise ValueError("std_db must be nonnegative")
    if not (
        isinstance(smooth_window, (int, np.integer))
        and smooth_window > 0
        and smooth_window % 2 == 1
    ):
        raise ValueError("smooth_window must be a positive odd integer")

    num_sc = amplitude_csi.shape[3]
    if num_sc == 1 or std_db == 0.0:
        gain_db = np.zeros(num_sc)
    else:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(num_sc)
        pad = smooth_window // 2
        smooth = np.convolve(
            np.pad(raw, pad, mode="reflect"),
            np.ones(smooth_window) / smooth_window,
            mode="valid",
        )
        gain_db = _normalize_mismatch_db(smooth, std_db)
    return amplitude_csi * 10.0 ** (gain_db[None, None, None, :] / 20.0)


def add_amplitude_noise(amplitude_csi, snr_db=20.0, seed=None):
    """Add real Gaussian noise in the CSI-amplitude domain.

    SNR uses mean squared amplitude before noise. Clipping keeps the observable
    amplitude nonnegative.
    """
    amplitude_csi = _amplitude_csi(amplitude_csi)
    snr_db = float(snr_db)
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    signal_power = float(np.mean(amplitude_csi**2))
    if signal_power == 0.0:
        return amplitude_csi.copy()
    noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    noise = np.random.default_rng(seed).normal(0.0, noise_std, amplitude_csi.shape)
    return np.maximum(amplitude_csi + noise, 0.0)


def simulate_csi(
    args,
    targets=None,
    static_amplitude=1.0,
    snr_db=20.0,
    rx_std_db=0.5,
    sc_std_db=0.5,
    sc_smooth_window=7,
    seed=7,
):
    """Generate final CSI amplitude with one transparent signal flow."""
    targets = targets_csi_config() if targets is None else targets
    static_amplitude = float(static_amplitude)
    if static_amplitude < 0.0:
        raise ValueError("static_amplitude must be nonnegative")

    shape = (args.num_frames, args.num_Tx, args.num_Rx, args.num_sc)
    latent_csi = np.full(shape, static_amplitude, dtype=np.complex128)
    latent_csi += target_csi(args, targets)

    # From here onward every operation is explicitly in the amplitude domain.
    amplitude_csi = np.abs(latent_csi)
    amplitude_csi = Rx_mismatch(amplitude_csi, rx_std_db, seed=seed)
    amplitude_csi = SC_mismatch(
        amplitude_csi,
        sc_std_db,
        smooth_window=sc_smooth_window,
        seed=None if seed is None else seed + 1,
    )
    return add_amplitude_noise(
        amplitude_csi,
        snr_db=snr_db,
        seed=None if seed is None else seed + 2,
    )


def targets_csi_config():
    """Return fixed close-AoA/ToF targets with distinct Doppler."""
    return [
        {
            "name": "strong",
            "strength_label": "strong",
            "theta": 80.0,
            "tof": 10e-9,
            "fd": 3.0,
            "amplitude": 0.30,
            "activity_start_s": 0.10,
            "activity_end_s": 0.60,
            "activity_ramp_s": 0.10,
        },
        {
            "name": "weak1",
            "strength_label": "weak1",
            "theta": 100.0,
            "tof": 12e-9,
            "fd": +15.0,
            "amplitude": 0.10,
            "activity_start_s": 0.20,
            "activity_end_s": 0.80,
            "activity_ramp_s": 0.10,
        },
                {
            "name": "weak1",
            "strength_label": "weak2",
            "theta": 60.0,
            "tof": 8e-9,
            "fd": -15.0,
            "amplitude": 0.10,
            "activity_start_s": 0.22,
            "activity_end_s": 0.82,
            "activity_ramp_s": 0.10,
        },
    ]


# Compatibility aliases for earlier simulation callers.
generate_simulate_csi = simulate_csi
awgn = add_amplitude_noise


__all__ = [
    "Rx_mismatch",
    "SC_mismatch",
    "activity_envelope",
    "add_amplitude_noise",
    "awgn",
    "generate_simulate_csi",
    "simulate_csi",
    "target_csi",
    "target_gt_at_frame",
    "target_trajectory",
    "targets_at_frame",
    "targets_csi_config",
    "trajectory_csi",
]
