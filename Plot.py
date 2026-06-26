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


def plot_spectrum(frame_idx, tau, theta, P_music, args, title="", prefix="Azi-ToF", other_axis_label="theta (deg)"):
    #peaks = find_Peaks.find_AoA_ToF_peaks(P_music, theta, tau)
    tau_axis, tau_label = _tof_axis_values_and_label(tau, args)
    axis_flip = getattr(args, "axis_flip", False)
    plt.figure()
    if axis_flip:
        plt.pcolormesh(theta, tau_axis, P_music.T, cmap='jet', shading='auto')
    else:
        plt.pcolormesh(tau_axis, theta, P_music, cmap='jet', shading='auto')
    if args.colorbar:
        plt.colorbar()
    '''
    for o in range(len(args.x_obj)):
        plt.scatter(args.gt_taus[o][frame_idx], args.gt_AoAs[o][frame_idx], marker='x', s=50,color = obj_colors[o])
    '''
    # 畫出網格線 (選擇性開啟，用於觀察 Grid Refinement 的分佈)
    if axis_flip:
        plt.gca().set_xticks(theta, minor=True)
        plt.gca().set_yticks(tau_axis, minor=True)
    else:
        plt.gca().set_xticks(tau_axis, minor=True)
        plt.gca().set_yticks(theta, minor=True)
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)

    if axis_flip:
        plt.xlabel(other_axis_label)
        plt.ylabel(tau_label)
    else:
        plt.xlabel(tau_label)
        plt.ylabel(other_axis_label)
    plt.title(prefix + ' ' + title + ' @ frame ' + str(frame_idx), fontsize = 8)

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


def plot_csi(csi, csi2=None, csi3=None, tx=0, title=None, amp_db=True, unwrap_phase=False, margin=50):
    """
    Plot CSI heatmaps.

    Single input:
        plot amplitude heatmap and rank-k PCA reconstructed amplitude heatmap.

    Three inputs:
        plot three amplitude heatmaps in one figure, e.g. MA / DWT / PCA results.
    """
    def _prepare_single_csi(csi_input):
        csi_input = np.asarray(csi_input)

        if csi_input.ndim == 4:
            total_frames, num_tx, num_rx, num_subc = csi_input.shape
            if tx < 0 or tx >= num_tx:
                raise ValueError(f"tx must be between 0 and {num_tx - 1}, but got {tx}")
            x = csi_input[:, tx, :, :]
            tx_label_local = f"Tx {tx}"
        elif csi_input.ndim == 3:
            if tx != 0:
                raise ValueError("tx can only be 0 when csi already has shape (frames, rx, subcarrier)")
            total_frames, num_rx, num_subc = csi_input.shape
            x = csi_input
            tx_label_local = "selected Tx"
        else:
            raise ValueError(
                "Expected CSI shape (frames, tx, rx, subcarrier) or "
                f"(frames, rx, subcarrier), but got {csi_input.shape}"
            )

        if total_frames == 0 or num_rx == 0 or num_subc == 0:
            raise ValueError(f"CSI must be non-empty, but got {x.shape}")

        if margin < 0:
            raise ValueError(f"margin must be non-negative, but got {margin}")
        if margin > 0:
            if total_frames <= 2 * margin:
                raise ValueError(
                    f"margin={margin} is too large for total_frames={total_frames}; "
                    "need total_frames > 2 * margin"
                )
            x = x[margin:-margin, :, :]

        return x, tx_label_local, num_rx, num_subc


    multi_inputs = [arr for arr in [csi, csi2, csi3] if arr is not None]

    if len(multi_inputs) >= 2:
        prepared = [_prepare_single_csi(arr) for arr in multi_inputs]
        first_shape = prepared[0][0].shape
        for x_i, _, _, _ in prepared[1:]:
            if x_i.shape != first_shape:
                raise ValueError(f"All CSI inputs must have the same cropped shape, but got {first_shape} and {x_i.shape}")

        tx_label = prepared[0][1]
        num_rx = prepared[0][2]
        num_subc = prepared[0][3]

        amp_maps = []
        for x_i, _, _, _ in prepared:
            amp_i = np.abs(x_i)
            amp_i = 20 * np.log10(amp_i + 1e-12) if amp_db else amp_i
            amp_maps.append(np.transpose(amp_i, (1, 2, 0)).reshape(num_rx * num_subc, x_i.shape[0]))

        stacked = np.concatenate([m.reshape(-1) for m in amp_maps])
        amp_vmin, amp_vmax = _percentile_limits(stacked)

        subplot_titles = ["MA", "DWT", f"PCA"]
        if len(amp_maps) != 3:
            subplot_titles = [f"CSI {i + 1}" for i in range(len(amp_maps))]

        fig, axes = plt.subplots(
            len(amp_maps),
            1,
            figsize=(8, 2 * len(amp_maps)),
            sharex=True,
            constrained_layout=True,
        )
        if len(amp_maps) == 1:
            axes = [axes]

        rx_centers = np.arange(num_rx) * num_subc + (num_subc - 1) / 2.0
        for idx, (ax, amp_map) in enumerate(zip(axes, amp_maps)):
            im = ax.imshow(
                amp_map,
                aspect="auto",
                origin="lower",
                cmap="jet",
                interpolation="nearest",
                vmin=amp_vmin,
                vmax=amp_vmax,
            )
            ax.set_title(f"{subplot_titles[idx]}, {tx_label}")
            ax.set_ylabel("Rx-Subcarrier bin")
            ax.set_yticks(rx_centers)
            ax.set_yticklabels([f"Rx{r}" for r in range(num_rx)])
            for r in range(1, num_rx):
                ax.axhline(r * num_subc - 0.5, color="white", linewidth=0.6, alpha=0.7)
            fig.colorbar(im, ax=ax, label="Amplitude (dB)" if amp_db else "Amplitude")

        axes[-1].set_xlabel("Frame index")
        fig.suptitle(title if title is not None else f"Preprocessing CSI heatmaps ({tx_label})")
        return fig, axes

    x, tx_label, num_rx, num_subc = _prepare_single_csi(csi)
    num_frames = x.shape[0]

    amp_linear = np.abs(x)
    amp = 20 * np.log10(amp_linear + 1e-12) if amp_db else amp_linear

    amp_map = np.transpose(amp, (1, 2, 0)).reshape(num_rx * num_subc, num_frames)
    amp_vmin, amp_vmax = _percentile_limits(amp_map)

    pca_recon_linear = np.zeros_like(amp_linear)
    for rx_idx in range(num_rx):
        rx_amp = amp_linear[:, rx_idx, :]
        rx_mean = np.mean(rx_amp, axis=0, keepdims=True)
        rx_centered = rx_amp - rx_mean

        if rx_centered.shape[0] >= 2 and rx_centered.shape[1] >= 1:
            u, s, vt = np.linalg.svd(rx_centered, full_matrices=False)
            if s.size > 0:
                keep_k = min(1, s.size)
                rank_k = (u[:, :keep_k] * s[:keep_k]) @ vt[:keep_k, :]
                pca_recon_linear[:, rx_idx, :] = rank_k + rx_mean
            else:
                pca_recon_linear[:, rx_idx, :] = rx_mean
        else:
            pca_recon_linear[:, rx_idx, :] = rx_amp

    pca_recon = 20 * np.log10(np.maximum(pca_recon_linear, 1e-12)) if amp_db else pca_recon_linear
    pca_recon_map = np.transpose(pca_recon, (1, 2, 0)).reshape(num_rx * num_subc, num_frames)
    pca_vmin, pca_vmax = _percentile_limits(pca_recon_map)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        amp_map,
        aspect="auto",
        origin="lower",
        cmap="jet",
        interpolation="nearest",
        vmin=amp_vmin,
        vmax=amp_vmax,
    )
    axes[0].set_title(f"Amplitude along time, {tx_label}")
    axes[0].set_ylabel("Rx-Subcarrier bin")
    fig.colorbar(im0, ax=axes[0], label="Amplitude (dB)" if amp_db else "Amplitude")

    im1 = axes[1].imshow(
        pca_recon_map,
        aspect="auto",
        origin="lower",
        cmap="jet",
        interpolation="nearest",
        vmin=pca_vmin,
        vmax=pca_vmax,
    )
    axes[1].set_title(f"PCA reconstructed amplitude, {tx_label}")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Rx-Subcarrier bin")
    fig.colorbar(im1, ax=axes[1], label="Amplitude (dB)" if amp_db else "Amplitude")

    for r in range(1, num_rx):
        axes[0].axhline(r * num_subc - 0.5, color="white", linewidth=0.6, alpha=0.7)
        axes[1].axhline(r * num_subc - 0.5, color="white", linewidth=0.6, alpha=0.7)

    rx_centers = np.arange(num_rx) * num_subc + (num_subc - 1) / 2.0
    axes[0].set_yticks(rx_centers)
    axes[0].set_yticklabels([f"Rx{r}" for r in range(num_rx)])
    axes[1].set_yticks(rx_centers)
    axes[1].set_yticklabels([f"Rx{r}" for r in range(num_rx)])

    fig.suptitle(title if title is not None else f"CSI heatmaps ({tx_label})")

    return fig, axes
