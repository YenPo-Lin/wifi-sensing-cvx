import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator


def _tof_axis_values_and_label(tau, args):
    if getattr(args, "axis", "ns") == "m":
        return tau * 3e8 / 2.0, "distance (m)"
    return tau * 1e9, "ToF (ns)"

def _axis_values_and_label(axis_name, values, args):
    if axis_name == "tof":
        return _tof_axis_values_and_label(values, args)
    if axis_name == "azi":
        return values, "Azimuth (deg)"
    if axis_name == "doppler":
        return values, "Doppler frequency (Hz)"
    raise ValueError(f"Unsupported axis name: {axis_name}")

def _plot_target_gt(ax, target_gt, x_axis, y_axis, args):
    """Plot ``[azimuth, ToF, Doppler]`` ground-truth rows on selected axes."""
    target_gt = getattr(args, "target_gt", None) if target_gt is None else target_gt
    if target_gt is None:
        return

    target_gt = np.asarray(target_gt, dtype=float)
    if target_gt.size == 0:
        return
    target_gt = np.atleast_2d(target_gt)
    if target_gt.shape[1] != 3:
        raise ValueError(
            "target_gt must have shape (num_targets, 3) with rows "
            "[azimuth_deg, tof_s, doppler_hz]."
        )

    column_by_axis = {"azi": 0, "tof": 1, "doppler": 2}
    x_gt, _ = _axis_values_and_label(
        x_axis, target_gt[:, column_by_axis[x_axis]], args
    )
    y_gt, _ = _axis_values_and_label(
        y_axis, target_gt[:, column_by_axis[y_axis]], args
    )
    ax.scatter(
        x_gt,
        y_gt,
        marker="x",
        s=50,
        color="white",
        linewidths=1.5,
        zorder=3,
    )

def plot_3D_point_cloud(
    frame_idx,
    azi_values,
    tof_values,
    doppler_values,
    P_music,
    args,
    title="cube",
    sdim=None,
    ax=None,
    save=True,
    show_colorbar=None,
    file_name=None,
):
    """Render an ``(azimuth, ToF, Doppler)`` MUSIC cube as a point cloud."""
    azi_values = np.asarray(azi_values, dtype=float)
    tof_values = np.asarray(tof_values, dtype=float)
    doppler_values = np.asarray(doppler_values, dtype=float)
    cube = np.asarray(P_music, dtype=float)

    axis_values = (azi_values, tof_values, doppler_values)
    if any(values.ndim != 1 or values.size == 0 for values in axis_values):
        raise ValueError("Azimuth, ToF, and Doppler must be non-empty 1D arrays.")

    expected_shape = (
        len(azi_values),
        len(tof_values),
        len(doppler_values),
    )
    if cube.shape != expected_shape:
        raise ValueError(
            f"Expected MUSIC cube shape {expected_shape}, got {cube.shape}."
        )

    finite_mask = np.isfinite(cube)
    finite = cube[finite_mask]
    if finite.size == 0:
        raise ValueError("P_music contains no finite values.")

    # Keep the high-energy structure visible instead of filling the entire
    # cube with opaque low-power samples. The complete cube still determines
    # the threshold and color scale.
    peak = float(np.max(finite))
    dynamic_range_db = max(
        0.0,
        float(getattr(args, "cube_dynamic_range_db", 20.0)),
    )
    percentile = float(
        np.clip(getattr(args, "cube_percentile", 82.0), 0.0, 100.0)
    )
    power_floor = max(
        peak - dynamic_range_db,
        float(np.percentile(finite, percentile)),
    )
    visible = finite_mask & (cube >= power_floor)
    azi_idx, tof_idx, doppler_idx = np.nonzero(visible)
    point_power = cube[visible]

    max_points = max(1, int(getattr(args, "cube_max_points", 60000)))
    if point_power.size > max_points:
        keep = np.argpartition(point_power, -max_points)[-max_points:]
        azi_idx = azi_idx[keep]
        tof_idx = tof_idx[keep]
        doppler_idx = doppler_idx[keep]
        point_power = point_power[keep]

    tof_plot, tof_label = _tof_axis_values_and_label(tof_values, args)
    vmin = float(np.min(point_power))
    vmax = peak
    if vmin == vmax:
        vmin = vmax - 1e-12
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    normalized_power = norm(point_power)
    point_colors = cmap(normalized_power)
    point_alpha_min = float(
        np.clip(getattr(args, "cube_point_alpha_min", 0.12), 0.0, 1.0)
    )
    point_alpha_max = float(
        np.clip(
            getattr(args, "cube_point_alpha_max", 0.95),
            point_alpha_min,
            1.0,
        )
    )
    point_alpha_gamma = max(
        1e-6,
        float(getattr(args, "cube_point_alpha_gamma", 1.8)),
    )
    point_colors[:, 3] = point_alpha_min + (
        point_alpha_max - point_alpha_min
    ) * normalized_power ** point_alpha_gamma

    if ax is None:
        fig = plt.figure(figsize=(8.4, 7.2))
        ax = fig.add_subplot(111, projection="3d")
    elif not hasattr(ax, "scatter") or not hasattr(ax, "set_zlabel"):
        raise ValueError("ax must be a Matplotlib 3D axes.")

    ax.scatter(
        azi_values[azi_idx],
        tof_plot[tof_idx],
        doppler_values[doppler_idx],
        c=point_colors,
        marker="s",
        s=max(0.1, float(getattr(args, "cube_point_size", 3.0))),
        linewidths=0,
        depthshade=False,
    )

    ax.set_xlabel("Azimuth (deg)", labelpad=8)
    ax.set_ylabel(tof_label, labelpad=8)
    ax.set_zlabel("Doppler (Hz)", labelpad=2)
    ax.set_xlim(float(np.min(azi_values)), float(np.max(azi_values)))
    ax.set_ylim(float(np.min(tof_plot)), float(np.max(tof_plot)))
    ax.set_zlim(float(np.min(doppler_values)), float(np.max(doppler_values)))
    # View the cube from the opposite side so the Doppler axis is on the left.
    ax.view_init(elev=24, azim=-125)
    ax.set_box_aspect((1.20, 1.05, 1.15))

    # Fine, unobtrusive cube grid similar to the reference visualization.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=8))
        axis.line.set_color("black")
        axis.line.set_linewidth(1.0)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["color"] = (0.10, 0.10, 0.10, 0.22)
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.10, 0.10, 0.10, 0.22))
        axis.pane.set_linewidth(0.35)
    ax.tick_params(labelsize=8, width=0.5, pad=1)

    full_title = title + f" @ frame {frame_idx}"
    if sdim is not None:
        full_title += f" Sdim {int(sdim)}"
    ax.set_title(full_title, fontsize=9, pad=14)

    if show_colorbar is None:
        show_colorbar = bool(args.colorbar)
    if show_colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array(point_power)
        colorbar = ax.figure.colorbar(
            scalar_map,
            ax=ax,
            pad=0.10,
            shrink=0.72,
        )
        colorbar.outline.set_visible(False)
        colorbar.set_label("Power (dB)", rotation=270, labelpad=15)

    if save and args.pics_dir is not None:
        save_dir = os.path.join(args.pics_dir, "Azi_ToF_Doppler")
        os.makedirs(save_dir, exist_ok=True)
        if file_name is None:
            file_name = f"{frame_idx:04d}.png"
        save_path = os.path.join(save_dir, file_name)
        ax.figure.savefig(
            save_path,
            dpi=160,
            bbox_inches="tight",
            pad_inches=0.20,
        )
        plt.close(ax.figure)
        print(f"Saved: {save_path}")

    return ax

def plot_spectrum(
    frame_idx,
    axis0_values,
    axis1_values,
    P_music,
    args,
    title="",
    x_axis=None,
    y_axis=None,
    sdim=None,
    spectrum_axes=None,
    ax=None,
    save=True,
    show_colorbar=None,
    target_gt=None,
    file_name=None,
):
    if spectrum_axes is None or len(spectrum_axes) != 2:
        raise ValueError("spectrum_axes=(axis0, axis1) is required.")

    axis0, axis1 = spectrum_axes
    if {x_axis, y_axis} != {axis0, axis1}:
        raise ValueError("x_axis/y_axis must match spectrum_axes.")

    axis0_values = np.asarray(axis0_values)
    axis1_values = np.asarray(axis1_values)
    P_music = np.asarray(P_music)
    if P_music.shape != (len(axis0_values), len(axis1_values)):
        raise ValueError(
            f"Expected spectrum shape {(len(axis0_values), len(axis1_values))}, "
            f"got {P_music.shape}."
        )

    values_by_axis = {axis0: axis0_values, axis1: axis1_values}
    x_values, x_label = _axis_values_and_label(x_axis, values_by_axis[x_axis], args)
    y_values, y_label = _axis_values_and_label(y_axis, values_by_axis[y_axis], args)
    plot_values = P_music.T if x_axis == axis0 else P_music

    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(
        x_values,
        y_values,
        plot_values,
        cmap='jet',
        shading='auto',
    )
    if show_colorbar is None:
        show_colorbar = bool(args.colorbar)
    if show_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax)
        colorbar.outline.set_visible(False)
        colorbar.set_label('Power (dB)', rotation=270, labelpad=15)

    _plot_target_gt(ax, target_gt, x_axis, y_axis, args)

    # Remove the default black frame around the heatmap. Cell edges are also
    # disabled above to avoid dark seams when the raster image is rescaled.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 畫出網格線 (選擇性開啟，用於觀察 Grid Refinement 的分佈)
    #ax.set_xticks(x_values, minor=True)
    #ax.set_yticks(y_values, minor=True)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0.2, alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    full_title = title + ' @ frame ' + str(frame_idx)
    if sdim is not None:
        full_title += ' Sdim ' + str(int(sdim))
    ax.set_title(full_title, fontsize=8)

    # --- save figures ---
    if save and args.pics_dir is not None:
        output_folder_by_axes = {
            frozenset(("azi", "tof")): "Azi_ToF",
            frozenset(("azi", "doppler")): "Azi_Doppler",
            frozenset(("tof", "doppler")): "ToF_Doppler",
        }
        save_dir = os.path.join(
            args.pics_dir,
            output_folder_by_axes[frozenset(spectrum_axes)],
        )
        os.makedirs(save_dir, exist_ok=True)
        file_name = file_name or f"{frame_idx:04d}.png"
        save_path = os.path.join(save_dir, file_name)
        ax.figure.savefig(save_path, dpi=100)
        plt.close(ax.figure)
        print(f"Saved: {save_path}")

    return ax
