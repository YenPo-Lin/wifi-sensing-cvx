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
    doppler_projection,
    projected_Azi_ToF
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
    parser.add_argument('--plot_frame', type=int, default=50)
    # MUSIC signal dimension
    parser.add_argument('--Sdim', type=int, default=10)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.9)
    parser.add_argument('--avg_frames', type=int, default=100)
    parser.add_argument('--fitting_window_rage', type=int, default=100) # search context
    parser.add_argument('--fitting_window', type=int, default=40) # local LS samples
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
    parser.add_argument('--theta_step', type=int, default=1)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='ns', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=2e-9)
    parser.add_argument('--tau_max', type=float, default=20e-9)
    parser.add_argument('--tau_step', type=float, default=1e-10)
    # Doppler grid
    parser.add_argument('--doppler_min', type=float, default=-30)
    parser.add_argument('--doppler_max', type=float, default=30)
    parser.add_argument('--doppler_step', type=float, default=1)

    # 2-D CA-CFAR settings (applied to linear ToF-Doppler MUSIC power)
    parser.add_argument('--cfar_training_cells', type=int, default=3)
    parser.add_argument('--cfar_guard_cells', type=int, default=1)
    parser.add_argument('--cfar_threshold_factor', type=float, default=1.5)
    parser.add_argument('--cfar_top_k', type=int, default=10)
    parser.add_argument('--cfar_min_peak_distance', type=int, default=2)

    # Relative hysteresis thresholds for fixed-fd temporal activity detection.
    parser.add_argument('--activity_low_ratio', type=float, default=0.15)
    parser.add_argument('--activity_high_ratio', type=float, default=0.5)
    parser.add_argument('--activity_baseline_quantile', type=float, default=0.1)
    parser.add_argument('--activity_max_gap_windows', type=int, default=3)
    parser.add_argument('--activity_min_windows', type=int, default=3)

    # Doppler spectrogram settings
    parser.add_argument('--stft_nperseg', type=int, default=64)
    parser.add_argument('--stft_noverlap', type=int, default=63) # hop 1

    # Filled from the instantaneous target states at the plotted frame.
    parser.set_defaults(target_gt=None)
    # heatmap axis (X: Azi, Y: TOF if True)
    parser.add_argument('--axis_flip', type=bool, default=True)
    parser.add_argument('--colorbar', type=bool, default=True)


    # ---- 圖片保存路徑 ----
    # ``os.makedirs("")`` is invalid, so keep a non-empty output directory.
    parser.add_argument('--pics_dir', type=str, default="pic")

    # ----3D MUSIC accumulation method ----
    parser.add_argument('--method', type=str, default='sum', choices=['sum', 'max', 'mean', 'weighted'],)

    args = parser.parse_args()
    # MUSIC.SteeringVector still uses the legacy ``num_scarriers`` name.
    args.num_sc = args.num_sc
    return args


def main():
    args = simulation_args()
    targets = targets_csi_config()
    args.target_gt = target_gt_at_frame(args, targets, args.plot_frame)
    plot_target_trajectories(args, targets, args.plot_frame)

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

    CSI = csi_amplitude **2  # linear CSI power used by this research
    background = pp.MA(CSI, args.fs * 0.5)

    if args.preprocess == "ma":
        #CSI = (CSI - background) # Original Version
        CSI = (CSI - background) / (background + 1e-8) # Normalized dynamic residual

    # 2D MUSIC

    ## Azimuth-Time of Flight spectrum
    azi_tof = MUSIC.Azi_ToF(args)
    azi_tof.gen_spectrum(CSI, args.plot_frame)


    ## Doppler-Time of Flight spectrum
    tof_dop = MUSIC.ToF_Dop(args)
    tof_dop.gen_spectrum(CSI, args.plot_frame)
    Rxx = tof_dop.Rxx_smooth(CSI, args.plot_frame)
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
        )

    target_fds = [target["fd"] for target in detected_targets]
    projected_results = projected_Azi_ToF(
        CSI,
        args,
        target_fds,
        window="rect",
        normalize=False,
    )
    for target, projected_result in zip(detected_targets, projected_results):
        target["projected_azi_tof"] = projected_result

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

    return detected_targets

if __name__ == "__main__":
    main()
