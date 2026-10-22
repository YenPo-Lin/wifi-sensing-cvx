"""Build Doppler-resolved point sets from projected Azi-ToF MUSIC maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def azi_tof_dop_pc(
    projected_spectra,
    theta_grid,
    tau_grid,
    *,
    frame_idx,
    target_id,
    pics_dir=None,
    energy_fraction=0.05,
):
    """Keep the fewest strongest pixels covering 10% of each map's magnitude sum.

    projected_spectra maps signed Doppler frequencies (Hz) to separate
    linear MUSIC pseudo-spectra of shape (azimuth, ToF). Each map is ranked
    independently before any Doppler-band sum. The point columns are
    [azimuth_deg, ToF_ns, Doppler_Hz, linear_MUSIC_pseudo_spectrum].
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if theta_grid.ndim != 1 or tau_grid.ndim != 1 or not theta_grid.size or not tau_grid.size:
        raise ValueError("theta_grid and tau_grid must be nonempty 1-D arrays")
    if not np.all(np.isfinite(theta_grid)) or not np.all(np.isfinite(tau_grid)):
        raise ValueError("theta_grid and tau_grid must contain finite values")
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must be in (0, 1]")

    clouds = []
    selected_fds = []
    input_magnitude_sums = []
    selected_magnitude_sums = []
    for fd, spectrum in sorted(projected_spectra.items()):
        fd = float(fd)
        spectrum = np.asarray(spectrum, dtype=float)
        if not np.isfinite(fd):
            raise ValueError("Doppler frequencies must be finite")
        if spectrum.shape != (theta_grid.size, tau_grid.size):
            raise ValueError("each projected spectrum must have shape (azimuth, ToF)")
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0):
            raise ValueError("projected spectra must contain finite nonnegative values")

        selected_fds.append(fd)
        flat = spectrum.ravel()
        ranked = np.argsort(-flat, kind="stable")
        total = float(np.sum(flat, dtype=np.float64))
        if not np.isfinite(total):
            raise ValueError("projected spectrum magnitude sum must be finite")
        input_magnitude_sums.append(total)
        if total == 0.0:
            selected_magnitude_sums.append(0.0)
            continue
        cumulative = np.cumsum(flat[ranked], dtype=np.float64)
        count = min(int(np.searchsorted(cumulative, energy_fraction * total) + 1), flat.size)
        ranked = ranked[:count]
        selected_magnitude_sums.append(float(cumulative[count - 1]))
        theta_idx, tau_idx = np.unravel_index(ranked, spectrum.shape)
        clouds.append(
            np.column_stack((
                theta_grid[theta_idx],
                tau_grid[tau_idx] * 1e9,
                np.full(count, fd),
                flat[ranked],
            ))
        )

    points = np.concatenate(clouds) if clouds else np.empty((0, 4), dtype=float)
    if pics_dir is None:
        return points, None

    save_dir = Path(pics_dir) / "Point_Cloud"
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(frame_idx):04d}_target_{int(target_id):02d}_azi_tof_dop_pc"
    data_path = save_dir / f"{stem}.npz"
    """
    np.savez_compressed(
        data_path,
        points=points,
        columns=np.array(["azi_deg", "tof_ns", "dop_hz", "mag_linear_pseudo"]),
        frame_idx=int(frame_idx),
        target_id=int(target_id),
        energy_fraction=float(energy_fraction),
        projected_fds=np.asarray(selected_fds, dtype=float),
        input_magnitude_sums=np.asarray(input_magnitude_sums, dtype=float),
        selected_magnitude_sums=np.asarray(selected_magnitude_sums, dtype=float),
    )

    print(f"Saved point cloud: {data_path}")
    """
    return points, str(data_path)


def plot_azi_tof_dop_pc(
    point_clouds,
    theta_grid,
    tau_grid,
    *,
    frame_idx,
    pics_dir,
    max_plot_points=60000,
    point_size=12.0,
    point_alpha=1.0,
):
    """Overlay all target point clouds in one azimuth-ToF plot, colored by Doppler."""
    if pics_dir is None:
        return None
    if max_plot_points < 1 or point_size <= 0 or not 0 <= point_alpha <= 1:
        raise ValueError("plot limits, point size, and point alpha must be valid")
    points = np.concatenate(point_clouds) if point_clouds else np.empty((0, 4), dtype=float)
    # Limit only the preview; each target's data file keeps every selected point.
    plotted = points
    if points.shape[0] > max_plot_points:
        strongest = np.argpartition(points[:, 3], -max_plot_points)[-max_plot_points:]
        plotted = points[strongest]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if plotted.size:
        # Interleave targets so overlapping coordinates do not always hide
        # the targets plotted first.
        plotted = plotted[np.random.default_rng(0).permutation(plotted.shape[0])]
        doppler_limit = float(np.max(np.abs(points[:, 2])))
        doppler_limit = doppler_limit if doppler_limit > 0 else 1.0
        scatter = ax.scatter(
            plotted[:, 0], plotted[:, 1],
            c=plotted[:, 2], cmap="bwr",
            vmin=-doppler_limit, vmax=doppler_limit,
            s=point_size, alpha=point_alpha,
        )
        fig.colorbar(scatter, ax=ax, label="Doppler (Hz)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("ToF (ns)")
    ax.set_xlim(np.min(theta_grid), np.max(theta_grid))
    ax.set_ylim(np.min(tau_grid) * 1e9, np.max(tau_grid) * 1e9)
    ax.set_title(f"Projected Azi-ToF-Doppler points | frame {frame_idx} | all targets")
    save_dir = Path(pics_dir) / "Point_Cloud"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"{int(frame_idx):04d}_all_targets_azi_tof_dop_pc.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved point cloud plot: {plot_path}")
    return str(plot_path)
