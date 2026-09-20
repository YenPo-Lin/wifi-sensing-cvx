import numpy as np
import pre_processing as pp
import MUSIC
import time
from simulate_utils import projected_Azi_ToF




def signal_processing(raw_CSI, args):

    start_preprocessing = time.time()
    #CSI = pp.self_sanitize(raw_CSI) # = np.abs(raw_CSI) # remove NaN and Inf
    CSI = np.abs(raw_CSI)# take absolute value to get power
    background = pp.MA(CSI, args.fs * 0.5)

    if args.preprocess == "ma":
        CSI = (CSI - background) # Original Version
        #CSI = (CSI - background) / (background + 1e-8) # Normalized dynamic residual
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


    tof_dop = MUSIC.ToF_Dop(args)
    azi_tof = MUSIC.Azi_ToF(args)
    azi_dop = MUSIC.Azi_Dop(args)
    azi_tof_dop = MUSIC.Azi_ToF_Dop(args)

    
    frame_idx = args.frame_idx

    print(f"Processing Frame {frame_idx}... ")

    #Doppler_spec.gen_spectrogram(CSI, args)
    #Doppler_spec.gen_spectrum(CSI, frame_idx)
    # Use all TX channels so this branch builds the same ToF-Doppler
    # covariance matrix as tof_dop.gen_spectrum(...) below.
    #Doppler_spec.gen_spectrum_from_ToF_Doppler(CSI, frame_idx, args, method="max", tx=None)
    #Doppler_spec.gen_spectrum_from_ToF_Doppler_Rx_diff(CSI, frame_idx, args, method="max")
    
    azi_tof_dop.gen_spectrum(CSI, frame_idx, method="sum")
    azi_tof.gen_spectrum(CSI, frame_idx, x_axis="azi", y_axis="tof")
    targets = tof_dop.gen_spectrum_return_target(CSI, frame_idx, x_axis="doppler", y_axis="tof")

    for target in targets:
        print(
            f"  #{target['rank']}: tau={target['tau'] * 1e9:.2f} ns, "
            f"fd={target['fd']:+.2f} Hz, "
            f"power={target['power_db']:.2f} dB, "
        )

    target_fds = [target["fd"] for target in targets]
    projected_results = projected_Azi_ToF(
        CSI,
        args,
        target_fds,
        window="rect",
        normalize=False,
    )
    for target, projected_result in zip(targets, projected_results):
        target["projected_azi_tof"] = projected_result

    azi_tof_dop.gen_sub_specturm(
        CSI,
        frame_idx,
        targets,
        fd_band=args.fd_band,
        method="sum",
    )
    azi_dop.gen_spectrum(CSI, frame_idx, x_axis="azi", y_axis="doppler")
