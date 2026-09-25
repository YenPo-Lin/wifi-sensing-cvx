import numpy as np
import pre_processing as pp
import MUSIC
import time
from simulate_utils import projected_Azi_ToF




def signal_processing(raw_CSI, args):

    start_preprocessing = time.time()
    #CSI = pp.self_sanitize(raw_CSI) # = np.abs(raw_CSI) # remove NaN and Inf
    CSI = np.abs(raw_CSI)**2# take absolute value to get power
    background = pp.MA(CSI, args.fs * 0.5)

    if args.preprocess == "ma":
        #CSI = (CSI - background) # Original Version
        CSI = (CSI - background) / (background + 1e-8) # Normalized dynamic residual
    elif args.preprocess == "dwt":
        CSI = pp.DWT_components(CSI, target_labels= ["", "D5", "D4", "D3", "D2", ""])
    elif args.preprocess == "pca":
        CSI  -= background
        CSI = pp.PCA_time(CSI, args.fs *0.5, k=3)

    #CSI = np.mean(CSI, axis=1, keepdims=True) # average over Tx

    # Re sampling
    # CSI = pp.sample_subcarriers(args, CSI, freq_space=args.freq_space)

    end_preprocessing = time.time()
    print(f"Preprocessing Method: {args.preprocess} | Time: {end_preprocessing - start_preprocessing:.2f}s")

    #WIDFS.plot_power_heatmap(CSI)
    #WIDFS.dfs_channel_weighting(raw_CSI, target_fd=0, args=args)


    azi_tof = MUSIC.Azi_ToF(args)
    azi_dop = MUSIC.Azi_Dop(args)
    azi_tof_dop = MUSIC.Azi_ToF_Dop(args)


    frame_idx = args.frame_idx

    print(f"Processing Frame {frame_idx}... ")

    #Doppler_spec.gen_spectrogram(CSI, args)
    #Doppler_spec.gen_spectrum(CSI, frame_idx)
    # Use all TX channels when enabling either ToF-Doppler branch.
    #Doppler_spec.gen_spectrum_from_ToF_Doppler(CSI, frame_idx, args, method="max", tx=None)
    #Doppler_spec.gen_spectrum_from_ToF_Doppler_Rx_diff(CSI, frame_idx, args, method="max")
    
    # 3D spectrum
    _ = azi_tof_dop.gen_spectrum(CSI, frame_idx, method="sum", fig_name="3D_MUSIC_sum",)
    # 3D spectrum with Doppler RoI
    roi_results = azi_tof_dop.gen_Doppler_RoI_specturm(CSI, frame_idx, fd_range=2, fig_name="3D_MUSIC_RoI")
    targets = [result["target"] for result in roi_results]


    # 2D Azi-ToF spectrum
    azi_tof.gen_spectrum(
        CSI,
        frame_idx,
        x_axis="azi",
        y_axis="tof",
    )
    _ = azi_tof.gen_Doppler_projection_spectrum(CSI, frame_idx)
