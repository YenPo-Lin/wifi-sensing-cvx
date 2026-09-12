import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def _percentile_limits(x, low=1.0, high=99.0):
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None, None

    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def _tof_axis_values_and_label(tau, args):
    if getattr(args, "axis", "ns") == "m":
        return tau * 3e8 / 2.0, "distance (m)"
    return tau * 1e9, "ToF (ns)"


def _normalize_axis_name(axis_name):
    aliases = {
        "aoa": "azi",
        "azimuth": "azi",
        "theta": "azi",
        "azi": "azi",
        "tof": "tof",
        "tau": "tof",
        "distance": "tof",
        "doppler": "doppler",
        "fd": "doppler",
    }
    normalized = aliases.get(str(axis_name).lower())
    if normalized is None:
        raise ValueError(f"Unsupported axis name: {axis_name}")
    return normalized


def _axis_values_and_label(axis_name, values, args):
    axis_name = _normalize_axis_name(axis_name)
    if axis_name == "tof":
        return _tof_axis_values_and_label(values, args)
    if axis_name == "azi":
        return values, "Azimuth (deg)"
    if axis_name == "doppler":
        return values, "Doppler frequency (Hz)"
    raise ValueError(f"Unsupported axis name: {axis_name}")


def plot_heatmap(
    frame_idx,
    x_values,
    y_values,
    heatmap,
    args,
    title="",
    cmap="jet",
    x_axis="",
    y_axis="",
    file_suffix=None,
    sdim=None,
):
    x_values, x_label = _axis_values_and_label(x_axis, np.asarray(x_values), args)
    y_values, y_label = _axis_values_and_label(y_axis, np.asarray(y_values), args)
    heatmap = np.asarray(heatmap)
    expected_shape = (len(y_values), len(x_values))
    if heatmap.shape != expected_shape:
        raise ValueError(
            f"Expected heatmap shape {expected_shape}, but got {heatmap.shape}"
        )

    plt.figure()
    plt.pcolormesh(x_values, y_values, heatmap, cmap=cmap, shading="auto")
    if args.colorbar:
        plt.colorbar()

    plt.gca().set_xticks(x_values, minor=True)
    plt.gca().set_yticks(y_values, minor=True)
    plt.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    full_title = title + " @ frame " + str(frame_idx)
    if sdim is not None:
        full_title += " Sdim " + str(int(sdim))
    plt.title(full_title, fontsize=8)

    save_dir = args.pics_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{frame_idx:04d}.png"
        if file_suffix:
            filename = f"{frame_idx:04d}_{file_suffix}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Saved: {save_path}")


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

def plot_3D_stack_spectrum(
    frame_idx,
    axis0_values,
    axis1_values,
    axis2_values,
    P_music,
    args,
    title="",
    x_axis=None,
    y_axis=None,
    z_axis=None,
    sdim=None,
    spectrum_axes=None,
    ax=None,
    save=True,
    show_colorbar=None,
    alpha=0.02,
):
    """Plot a 3D MUSIC cube as translucent heatmap slices along ``z_axis``.

    ``P_music`` follows ``spectrum_axes`` order. The displayed x/y/z axes may
    be any permutation of those three axes. Every z-axis bin is rendered as
    one layer, so the layer count follows doppler_min/max/step when Doppler is
    selected as the z axis.
    """
    if spectrum_axes is None or len(spectrum_axes) != 3:
        raise ValueError(
            "spectrum_axes=(axis0, axis1, axis2) is required for 3D plotting."
        )

    spectrum_axes = tuple(map(_normalize_axis_name, spectrum_axes))
    if len(set(spectrum_axes)) != 3:
        raise ValueError("spectrum_axes must contain three distinct axes.")

    x_axis = _normalize_axis_name(x_axis or spectrum_axes[0])
    y_axis = _normalize_axis_name(y_axis or spectrum_axes[1])
    z_axis = _normalize_axis_name(z_axis or spectrum_axes[2])
    display_axes = (x_axis, y_axis, z_axis)
    if set(display_axes) != set(spectrum_axes):
        raise ValueError("x_axis/y_axis/z_axis must match spectrum_axes.")

    axis_values = tuple(
        np.asarray(values, dtype=float)
        for values in (axis0_values, axis1_values, axis2_values)
    )
    P_music = np.asarray(P_music, dtype=float)
    expected_shape = tuple(len(values) for values in axis_values)
    if P_music.shape != expected_shape:
        raise ValueError(
            f"Expected 3D spectrum shape {expected_shape}, got {P_music.shape}."
        )
    if any(values.ndim != 1 or values.size == 0 for values in axis_values):
        raise ValueError("All three spectrum axes must be non-empty 1D arrays.")

    values_by_axis = dict(zip(spectrum_axes, axis_values))
    x_values, x_label = _axis_values_and_label(
        x_axis, values_by_axis[x_axis], args
    )
    y_values, y_label = _axis_values_and_label(
        y_axis, values_by_axis[y_axis], args
    )
    z_values, z_label = _axis_values_and_label(
        z_axis, values_by_axis[z_axis], args
    )

    # Reorder the cube to a simple (x, y, z) layout for surface rendering.
    permutation = tuple(spectrum_axes.index(axis_name) for axis_name in display_axes)
    cube_xyz = np.transpose(P_music, permutation)
    finite = cube_xyz[np.isfinite(cube_xyz)]
    if finite.size == 0:
        raise ValueError("P_music contains no finite values.")
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if vmin == vmax:
        vmax = vmin + 1e-12

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)

    # Every Doppler/z bin is one surface layer. Only the in-plane x/y grid may
    # be downsampled for rendering speed; no z-axis feature is discarded.
    slice_indices = np.arange(len(z_values), dtype=int)

    max_axis_points = max(2, int(getattr(args, "stack_axis_max_points", 80)))
    x_indices = np.unique(
        np.linspace(0, len(x_values) - 1, min(max_axis_points, len(x_values))).round().astype(int)
    )
    y_indices = np.unique(
        np.linspace(0, len(y_values) - 1, min(max_axis_points, len(y_values))).round().astype(int)
    )
    x_plot = x_values[x_indices]
    y_plot = y_values[y_indices]
    X, Y = np.meshgrid(x_plot, y_plot, indexing="ij")

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    elif not hasattr(ax, "plot_surface"):
        raise ValueError("ax must be a Matplotlib 3D axes.")

    for slice_idx in slice_indices:
        plane = cube_xyz[np.ix_(x_indices, y_indices, [slice_idx])][..., 0]
        normalized = norm(plane)
        facecolors = cmap(normalized)
        # Every Doppler layer uses the same fixed opacity supplied by ``alpha``.
        facecolors[..., 3] = alpha
        Z = np.full_like(X, z_values[slice_idx], dtype=float)
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    target_gt = getattr(args, "target_gt", None)
    if target_gt is not None:
        target_gt = np.asarray(target_gt, dtype=float)
        if target_gt.size:
            target_gt = np.atleast_2d(target_gt)
            if target_gt.shape[1] != 3:
                raise ValueError(
                    "target_gt must have rows [azimuth_deg, tof_s, doppler_hz]."
                )
            gt_by_axis = {
                "azi": target_gt[:, 0],
                "tof": target_gt[:, 1],
                "doppler": target_gt[:, 2],
            }
            x_gt, _ = _axis_values_and_label(x_axis, gt_by_axis[x_axis], args)
            y_gt, _ = _axis_values_and_label(y_axis, gt_by_axis[y_axis], args)
            z_gt, _ = _axis_values_and_label(z_axis, gt_by_axis[z_axis], args)
            ax.scatter(
                x_gt,
                y_gt,
                z_gt,
                marker="x",
                s=55,
                color="white",
                linewidths=1.8,
                depthshade=False,
            )

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_zlabel(z_label, labelpad=8)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(float(y_values[0]), float(y_values[-1]))
    ax.set_zlim(float(z_values[0]), float(z_values[-1]))
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1.15, 1.15, 1.4))

    full_title = (title or "3D stacked MUSIC spectrum") + f" @ frame {frame_idx}"
    if sdim is not None:
        full_title += f" Sdim {int(sdim)}"
    ax.set_title(full_title, fontsize=9, pad=16)

    if show_colorbar is None:
        show_colorbar = bool(args.colorbar)
    if show_colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array(finite)
        colorbar = ax.figure.colorbar(scalar_map, ax=ax, pad=0.10, shrink=0.72)
        colorbar.outline.set_visible(False)
        colorbar.set_label("Power (dB)", rotation=270, labelpad=15)

    if save and args.pics_dir is not None:
        os.makedirs(args.pics_dir, exist_ok=True)
        save_title = title or "3D stacked MUSIC spectrum"
        save_path = os.path.join(args.pics_dir, f"{save_title} {frame_idx:04d}.png")
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
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
):
    if spectrum_axes is None or len(spectrum_axes) != 2:
        raise ValueError("spectrum_axes=(axis0, axis1) is required.")

    axis0, axis1 = map(_normalize_axis_name, spectrum_axes)
    x_axis = _normalize_axis_name(x_axis or axis1)
    y_axis = _normalize_axis_name(y_axis or axis0)
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
    save_dir = args.pics_dir
    if save and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{title} {frame_idx:04d}.png"
        )
        ax.figure.savefig(save_path, dpi=100)
        plt.close(ax.figure)
        print(f"Saved: {save_path}")

    return ax
