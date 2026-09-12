"""Compare magnitude-only Azi-ToF and Doppler-gated Azi-ToF heatmaps."""


import matplotlib
import argparse
matplotlib.use("Agg")

import numpy as np
import MUSIC
import Plot
import pre_processing as pp
from simulate_CSI import (
    simulate_csi,
    target_gt_at_frame,
    plot_target_trajectories,
    targets_csi_config,
)
from simulate_utils import (
    cfar_2D,
    design_mat,
    fitting_quality,
    fitting_strength,
    ls_solution,
)


def simulation_args():
    parser = argparse.ArgumentParser()
    # ---- CSI parameters ----
    parser.add_argument('--num_Tx', type=int, default=1)
    parser.add_argument('--num_Rx', type=int, default=5)
    parser.add_argument('--num_sc', type=int, default=64)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--f_0', type=float, default=5.57e9)
    parser.add_argument('--BW', type=float, default=160e6)
    parser.add_argument('--delta_f', type=float, default=2.54e6)
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--antenna_spacing', type=float, default=0.02)

    # ---- MUSIC settings ----
    parser.add_argument('--preprocess', type=str, default='ma', choices=['ma', 'dwt', 'pca'])
    # plotted frame
    #parser.add_argument('--frame_idx', type=int, default=660)
    # MUSIC signal dimension
    parser.add_argument('--Sdim', type=int, default=8)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.9)
    parser.add_argument('--avg_frames', type=int, default=100)
    parser.add_argument('--fitting_window', type=int, default=100)
    parser.add_argument('--projection', type=str, default='cos', choices=['sin', 'cos'])

    parser.add_argument('--stream_win', type=int, default=3)
    parser.add_argument('--stream_sample_range', type=int, default=5) #all Rx

    parser.add_argument('--freq_win', type=int, default=40) #block size = freq_win // freq_hop
    parser.add_argument('--freq_hop', type=int, default=4)
    parser.add_argument('--freq_sample_range', type=int, default=64) #all subcarriers
    parser.add_argument('--freq_space', type=int, default=1) # if freq resampling


    parser.add_argument('--time_win', type=int, default=20)
    parser.add_argument('--time_hop', type=int, default=1)
    parser.add_argument('--time_sample_range', type=int, default= 100)

    # Azimuth grid
    parser.add_argument('--theta_min', type=float, default= 0)
    parser.add_argument('--theta_max', type=float, default= 180)
    parser.add_argument('--theta_step', type=int, default=3)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='ns', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=2e-9)
    parser.add_argument('--tau_max', type=float, default=20e-9)
    parser.add_argument('--tau_step', type=float, default=3e-10)
    # Doppler grid
    parser.add_argument('--doppler_min', type=float, default=-30)
    parser.add_argument('--doppler_max', type=float, default=30)
    parser.add_argument('--doppler_step', type=float, default=1)

    # 2-D CA-CFAR settings (applied to linear ToF-Doppler MUSIC power)
    parser.add_argument('--cfar_training_cells', type=int, default=3)
    parser.add_argument('--cfar_guard_cells', type=int, default=1)
    parser.add_argument('--cfar_threshold_factor', type=float, default=1.5)
    parser.add_argument('--cfar_top_k', type=int, default=10)
    parser.add_argument('--cfar_min_peak_distance', type=int, default=3)

    # Doppler spectrogram settings
    parser.add_argument('--stft_nperseg', type=int, default=64)
    parser.add_argument('--stft_noverlap', type=int, default=63) # hop 1

    # Filled from the instantaneous target states at the plotted frame.
    parser.set_defaults(target_gt=None)
    # heatmap axis (X: Azi, Y: TOF if True)
    parser.add_argument('--axis_flip', type=bool, default=True)
    parser.add_argument('--colorbar', type=bool, default=True)


    # ---- 圖片保存路徑 ----
    parser.add_argument('--pics_dir', type=str, default="simulation_outputs")

    # ----3D MUSIC accumulation method ----
    parser.add_argument('--method', type=str, default='sum', choices=['sum', 'max', 'mean', 'weighted'],)

    args = parser.parse_args()
    # MUSIC.SteeringVector still uses the legacy ``num_scarriers`` name.
    args.num_sc = args.num_sc
    return args


def main():
    args = simulation_args()
    targets = targets_csi_config()
    plotted_frame = args.num_frames//2
    args.target_gt = target_gt_at_frame(args, targets, plotted_frame)
    plot_target_trajectories(args, targets, plotted_frame)
    csi_amplitude = simulate_csi(
        args,
        targets=targets,
        static_amplitude=1.0,
        snr_db=10.0,
        rx_std_db=0.5,
        sc_std_db=0.5,
        sc_smooth_window=7,
        seed=7,
    )
    CSI = csi_amplitude**2  # linear CSI power used by this research
    background = pp.MA(CSI, args.fs * 0.5)

    if args.preprocess == "ma":
        CSI = (CSI - background) # Original Version
        #CSI = (CSI - background) / (background + 1e-8) # Normalized dynamic residual

    # 2D MUSIC
    azi_tof = MUSIC.Azi_ToF(args)
    azi_tof.gen_spectrum(CSI, plotted_frame)



    tof_dop = MUSIC.ToF_Dop(args)
    Rxx = tof_dop.Rxx_smooth(CSI, plotted_frame)
    tau_grid, fd_grid, P_tof_dop = tof_dop.cal_spectrum(Rxx)
    P_tof_dop_db = 10.0 * np.log10(np.maximum(P_tof_dop, 1e-12))
    # 2D CFAR to find target peaks in the ToF-Doppler spectrum
    detected_targets = cfar_2D(
        P_tof_dop,
        tau_grid,
        fd_grid,
        training_cells=args.cfar_training_cells,
        guard_cells=args.cfar_guard_cells,
        threshold_factor=args.cfar_threshold_factor,
        top_k=args.cfar_top_k,
        min_peak_distance=args.cfar_min_peak_distance,
    )
    print("2-D CFAR targets (power descending):")
    for target in detected_targets:
        print(
            f"  #{target['rank']}: tau={target['tau'] * 1e9:.2f} ns, "
            f"fd={target['fd']:+.2f} Hz, "
            f"power={target['power_db']:.2f} dB, "
            f"SNR={target['snr_db']:.2f} dB"
        )

    # Search for the best local LS window inside an args.fitting_window context
    # centred on plotted_frame. args.time_win/time_hop control the local fit.
    fitting_context_size = int(args.fitting_window)
    local_window_size = int(args.time_win)
    local_window_hop = int(args.time_hop)
    if local_window_size < 4:
        raise ValueError("time_win must be at least 4 for target-Doppler fitting")
    if fitting_context_size < local_window_size:
        raise ValueError("fitting_window cannot be smaller than time_win")
    if local_window_hop <= 0:
        raise ValueError("time_hop must be positive")

    fitting_half_window = fitting_context_size // 2
    fitting_start = plotted_frame - fitting_half_window
    fitting_end = fitting_start + fitting_context_size
    if fitting_start < 0 or fitting_end > CSI.shape[0]:
        raise ValueError(
            f"fitting search range {fitting_start}:{fitting_end} exceeds CSI "
            f"frame range 0:{CSI.shape[0]}"
        )

    window_starts = np.arange(
        fitting_start,
        fitting_end - local_window_size + 1,
        local_window_hop,
        dtype=int,
    )
    window_ends = window_starts + local_window_size
    window_centers = (window_starts + window_ends - 1) / 2.0
    local_half_window = local_window_size // 2
    local_time_s = (
        np.arange(local_window_size, dtype=float) - local_half_window
    ) / args.fs
    window_weights = np.hanning(local_window_size)
    X0 = design_mat(local_time_s)

    # M0 is independent of target Doppler, so calculate it once per window.
    y_windows = [CSI[start:end] for start, end in zip(window_starts, window_ends)]
    null_sse = np.stack(
        [ls_solution(X0, y_window, window_weights)["sse"] for y_window in y_windows],
        axis=0,
    )
    print(
        f"Target-Doppler fitting search range: {fitting_start}:{fitting_end} "
        f"(centre frame {plotted_frame}), local window={local_window_size}, "
        f"hop={local_window_hop}"
    )

    for target in detected_targets:
        target_fd_hz = target["fd"]
        if np.isclose(target_fd_hz, 0.0):
            target["fitting"] = None
            print(f"  Skip fitting target #{target['rank']}: fd is 0 Hz")
            continue

        X1 = design_mat(local_time_s, target_fd_hz=target_fd_hz)
        cos_coefficients = []
        sin_coefficients = []
        strengths = []
        target_sse = []
        for y_window in y_windows:
            fit1 = ls_solution(X1, y_window, window_weights)
            a = fit1["coefficients"][-2]
            b = fit1["coefficients"][-1]
            cos_coefficients.append(a)
            sin_coefficients.append(b)
            strengths.append(fitting_strength(a, b))
            target_sse.append(fit1["sse"])

        cos_coefficients = np.stack(cos_coefficients, axis=0)
        sin_coefficients = np.stack(sin_coefficients, axis=0)
        strengths = np.stack(strengths, axis=0)
        target_sse = np.stack(target_sse, axis=0)
        qualities = fitting_quality(null_sse, target_sse)

        # Normalize every channel along the local-window axis before channel
        # aggregation, so its temporal peak has unit strength.
        strength_scale = np.max(strengths, axis=0, keepdims=True)
        normalized_strengths = np.divide(
            strengths,
            strength_scale,
            out=np.zeros_like(strengths),
            where=strength_scale > 1e-12,
        )
        channel_scores = normalized_strengths * qualities
        channel_axes = tuple(range(1, channel_scores.ndim))
        mean_strength = np.mean(strengths, axis=channel_axes)
        aggregate_strength = np.mean(normalized_strengths, axis=channel_axes)
        mean_quality = np.mean(qualities, axis=channel_axes)
        mean_score = np.mean(channel_scores, axis=channel_axes)
        best_window_idx = int(np.argmax(mean_score))
        best_start = int(window_starts[best_window_idx])
        best_end = int(window_ends[best_window_idx])
        best_center = float(window_centers[best_window_idx])

        target["fitting"] = {
            "search_start_idx": fitting_start,
            "search_end_idx": fitting_end,
            "window_start_idx": window_starts.copy(),
            "window_end_idx": window_ends.copy(),
            "window_center_idx": window_centers.copy(),
            "window_center_time_s": window_centers / args.fs,
            "cos_coefficient": cos_coefficients,
            "sin_coefficient": sin_coefficients,
            "strength": strengths,
            "strength_scale": strength_scale[0],
            "normalized_strength": normalized_strengths,
            "quality": qualities,
            "score": channel_scores,
            "mean_strength": mean_strength,
            "aggregate_strength": aggregate_strength,
            "mean_quality": mean_quality,
            "mean_score": mean_score,
            "best_window_idx": best_window_idx,
            "best_window_start_idx": best_start,
            "best_window_end_idx": best_end,
            "best_window_center_idx": best_center,
        }

        print(
            f"  Fitting target #{target['rank']} ({target_fd_hz:+.2f} Hz): "
            f"best window=[{best_start}:{best_end}), "
            f"R_norm={aggregate_strength[best_window_idx]:.4f}, "
            f"Q_mean={mean_quality[best_window_idx]:.4f}, "
            f"score={mean_score[best_window_idx]:.4f}"
        )

    """
    Plot.plot_spectrum(
        plotted_frame,
        tau_grid,
        fd_grid,
        P_tof_dop_db,
        args,
        title="ToF-Doppler",
        x_axis="doppler",
        y_axis="tof",
        sdim=args.Sdim,
        spectrum_axes=("tof", "doppler"),
    )
    """
    #azi_dop = MUSIC.Azi_Dop(args)
    #azi_dop.gen_spectrum(CSI, plotted_frame)

    # 3D MUSIC uses the shared Plot.plot_spectrum() rendering path.
    # azi_tof_dop = MUSIC.Azi_ToF_Dop(args)
    # azi_tof_dop.gen_spectrum(CSI, plotted_frame, method=args.method)

if __name__ == "__main__":
    main()
