"""Generate amplitude-only CSI for controlled multi-target experiments.

The public CSI layout is (frame, Tx, Rx, subcarrier). Each target follows one
2-D bistatic Tx-target-Rx path. AoA, ToF, and Doppler are all derived from
that path instead of being configured independently.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import MUSIC


LIGHT_SPEED_MPS = 299_792_458.0
DEFAULT_TX_POSITION_M = np.array([1.0, 0.0])
DEFAULT_RX_POSITION_M = np.array([0.0, 0.0])


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
    """Derive a target's AoA, ToF, and Doppler from one physical path.

    The configured theta/tof specify the initial position on the bistatic
    ellipse for the Tx/Rx geometry. Radial velocity plus one sinusoidal
    radial/tangential displacement produces the complete 2-D trajectory.
    Doppler is the discrete carrier-phase slope implied by the resulting ToF.
    """
    missing = {"theta", "tof", "amplitude"} - target.keys()
    if missing:
        raise ValueError(
            f"target {target.get('name', '<unnamed>')} is missing: "
            + ", ".join(sorted(missing))
        )
    if args.num_frames <= 0 or args.fs <= 0:
        raise ValueError("num_frames and fs must be positive")

    theta_0_deg = float(target["theta"])
    tof_0_s = float(target["tof"])
    amplitude = float(target["amplitude"])
    if not 0.0 <= theta_0_deg <= 180.0:
        raise ValueError("target theta must be in [0, 180] degrees")
    if tof_0_s <= 0.0 or amplitude < 0.0:
        raise ValueError("target tof must be positive and amplitude nonnegative")

    time_s = np.arange(args.num_frames, dtype=float) / float(args.fs)
    theta_0_rad = np.deg2rad(theta_0_deg)
    radial_axis = np.array([np.cos(theta_0_rad), np.sin(theta_0_rad)])
    tangential_axis = np.array([-np.sin(theta_0_rad), np.cos(theta_0_rad)])

    frequency_hz = float(target.get("motion_frequency_hz", 0.0))
    phase_rad = np.deg2rad(float(target.get("motion_phase_deg", 0.0)))
    angle = 2.0 * np.pi * frequency_hz * time_s + phase_rad
    # Subtract the initial value so theta/tof are exact at frame zero.
    oscillation = np.sin(angle) - np.sin(phase_rad)

    tx_position_m = np.asarray(
        getattr(args, "tx_position_m", DEFAULT_TX_POSITION_M), dtype=float
    )
    rx_position_m = np.asarray(
        getattr(args, "rx_position_m", DEFAULT_RX_POSITION_M), dtype=float
    )
    if tx_position_m.shape != (2,) or rx_position_m.shape != (2,):
        raise ValueError("tx_position_m and rx_position_m must be 2-D points")

    # Solve p0 = rx + radius_0*u from
    # |p0-rx| + |p0-tx| = c*tof_0.
    tx_from_rx = tx_position_m - rx_position_m
    total_path_0_m = LIGHT_SPEED_MPS * tof_0_s
    tx_rx_distance_m = float(np.linalg.norm(tx_from_rx))
    denominator = 2.0 * (
        total_path_0_m - float(np.dot(radial_axis, tx_from_rx))
    )
    if total_path_0_m <= tx_rx_distance_m or denominator <= 0.0:
        raise ValueError("target tof is too short for the configured Tx/Rx geometry")
    radius_0_m = (
        total_path_0_m**2 - tx_rx_distance_m**2
    ) / denominator

    radial_distance_m = (
        radius_0_m
        + float(target.get("radial_velocity_mps", 0.0)) * time_s
        + float(target.get("radial_motion_m", 0.0)) * oscillation
    )
    tangential_distance_m = (
        float(target.get("tangential_motion_m", 0.0)) * oscillation
    )
    position_m = (
        rx_position_m[None, :]
        + radial_distance_m[:, None] * radial_axis[None, :]
        + tangential_distance_m[:, None] * tangential_axis[None, :]
    )

    rx_to_target_m = position_m - rx_position_m[None, :]
    rx_range_m = np.linalg.norm(rx_to_target_m, axis=1)
    tx_range_m = np.linalg.norm(position_m - tx_position_m[None, :], axis=1)
    if np.any(rx_range_m <= 0.0) or np.any(tx_range_m <= 0.0):
        raise ValueError("target trajectory crosses the Tx or Rx position")

    # arccos(x/r) matches the 0-to-180 degree ULA convention used by MUSIC.
    theta_deg = np.rad2deg(
        np.arccos(np.clip(rx_to_target_m[:, 0] / rx_range_m, -1.0, 1.0))
    )
    tof_s = (rx_range_m + tx_range_m) / LIGHT_SPEED_MPS

    # H(t) contains exp(-j*2*pi*f0*tof(t)); its phase slope is Doppler.
    carrier_phase = -2.0 * np.pi * float(args.f_0) * tof_s
    fd_hz = np.zeros_like(time_s)
    if args.num_frames > 1:
        fd_hz[:-1] = np.diff(carrier_phase) * float(args.fs) / (2.0 * np.pi)
        fd_hz[-1] = fd_hz[-2]

    return {
        "time_s": time_s,
        "position_m": position_m,
        "theta": theta_deg,
        "tof": tof_s,
        "fd": fd_hz,
        "activity": activity_envelope(args, target),
    }



def plot_target_trajectories(args, targets, frame_idx):
    """Plot target trajectories versus frame and mark values at ``frame_idx``."""
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
    axes[0].set_title(f"Target ground-truth trajectories @ frame {frame_idx}")
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
    """Generate one target's latent complex CSI from its coupled trajectory."""
    trajectory = target_trajectory(args, target)
    steering = MUSIC.SteeringVector(args)
    a_azi = np.stack(
        [
            steering.steering_vector_AoA(theta, stream_win=args.num_Rx)
            for theta in trajectory["theta"]
        ]
    )
    # The time-varying carrier phase inside ToF steering produces Doppler;
    # there is no separately configured Doppler steering term.
    a_tof = np.stack(
        [
            steering.steering_vector_ToF(tof, freq_win=args.num_sc, freq_hop=1)
            for tof in trajectory["tof"]
        ]
    )

    component = (
        float(target["amplitude"])
        * trajectory["activity"][:, None, None, None]
        * np.exp(1j * float(target.get("initial_phase_rad", 0.0)))
        * a_azi[:, None, :, None]
        * a_tof[:, None, None, :]
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


def targets_csi_config(fix_motion=False):
    """Return close-AoA/ToF strong and weak targets with distinct |fd|."""
    targets = [
        {
            "name": "strong",
            "strength_label": "strong",
            "theta": 72.0,
            "tof": 11e-9,
            "amplitude": 0.30,
            "radial_velocity_mps": -0.12,
            "radial_motion_m": 0.010,
            "tangential_motion_m": 0.060,
            "motion_frequency_hz": 0.50,
            "motion_phase_deg": 0.0,
            "activity_start_s": 0.10,
            "activity_end_s": 0.60,
            "activity_ramp_s": 0.10,
        },
        {
            "name": "weak",
            "strength_label": "weak",
            "theta": 84.0,
            "tof": 14e-9,
            "amplitude": 0.10,
            "radial_velocity_mps": -0.30,
            "radial_motion_m": 0.008,
            "tangential_motion_m": 0.070,
            "motion_frequency_hz": 0.80,
            "motion_phase_deg": 30.0,
            "activity_start_s": 0.20,
            "activity_end_s": 0.80,
            "activity_ramp_s": 0.10,
        },
    ]

    if fix_motion:
        for target in targets:
            target["radial_velocity_mps"] = 0.0
            target["radial_motion_m"] = 0.0
            target["tangential_motion_m"] = 0.0
            target["motion_frequency_hz"] = 0.0
    return targets


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
