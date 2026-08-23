import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


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
    plt.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + " @ frame " + str(frame_idx), fontsize=8)

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


def plot_spectrum(
    frame_idx,
    tau,
    theta,
    P_music,
    args,
    title="",
    cmap='jet',
    x_axis=None,
    y_axis=None,
):
    #peaks = find_Peaks.find_AoA_ToF_peaks(P_music, theta, tau)
    tau = np.asarray(tau)
    theta = np.asarray(theta)
    P_music = np.asarray(P_music)

    if x_axis == "":
        x_axis = None
    if y_axis == "":
        y_axis = None

    if x_axis is None and y_axis is None:
        axis_flip = getattr(args, "axis_flip", False)
        if axis_flip:
            x_axis, y_axis = "azi", "tof"
        else:
            x_axis, y_axis = "tof", "azi"
    elif x_axis is None or y_axis is None:
        raise ValueError("x_axis and y_axis must be provided together.")

    x_axis = _normalize_axis_name(x_axis)
    y_axis = _normalize_axis_name(y_axis)
    if x_axis == y_axis:
        raise ValueError(f"x_axis and y_axis must be different, got {x_axis}")
    if "tof" not in (x_axis, y_axis):
        raise ValueError("One axis must be 'tof' because the first spectrum axis is tau.")

    if x_axis == "tof":
        x_values, x_label = _axis_values_and_label(x_axis, tau, args)
        y_values, y_label = _axis_values_and_label(y_axis, theta, args)
        plot_values = P_music
    else:
        x_values, x_label = _axis_values_and_label(x_axis, theta, args)
        y_values, y_label = _axis_values_and_label(y_axis, tau, args)
        plot_values = P_music.T

    expected_shape = (len(y_values), len(x_values))
    if plot_values.shape != expected_shape:
        raise ValueError(
            "Spectrum shape does not match requested axes: "
            f"got {plot_values.shape}, expected {expected_shape} "
            f"for x_axis={x_axis}, y_axis={y_axis}."
        )

    plt.figure()
    plt.pcolormesh(x_values, y_values, plot_values, cmap=cmap, shading='auto')
    if args.colorbar:
        plt.colorbar()
    '''
    for o in range(len(args.x_obj)):
        plt.scatter(args.gt_taus[o][frame_idx], args.gt_AoAs[o][frame_idx], marker='x', s=50,color = obj_colors[o])
    '''
    # 畫出網格線 (選擇性開啟，用於觀察 Grid Refinement 的分佈)
    plt.gca().set_xticks(x_values, minor=True)
    plt.gca().set_yticks(y_values, minor=True)
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + ' @ frame ' + str(frame_idx), fontsize = 8)

    # --- save figures ---
    save_dir = args.pics_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{frame_idx:04d}.png"
        )
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Saved: {save_path}")
