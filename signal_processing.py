import numpy as np
import pre_processing as pp
import MUSIC
import Plot
import LASSO
import os
import time
import Doppler_spec




def signal_processing(raw_CSI, args):

    start_preprocessing = time.time()
    CSI = pp.self_sanitize(raw_CSI)

    #CSI = np.mean(CSI, axis=1, keepdims=True)
    if args.preprocess == "ma":
        CSI  -= pp.MA(CSI, args.fs * 1.0)
        pass
    elif args.preprocess == "dwt":
        CSI = pp.DWT_components(CSI, target_labels= ["", "D5", "D4", "D3", "D2", ""])
    elif args.preprocess == "pca":
        CSI  -= pp.MA(CSI, args.fs * 0.5)
        CSI = pp.PCA_time(CSI, args.fs *0.5, k=3)

    # Re sampling
    CSI = pp.sample_subcarriers(args, CSI, freq_space=args.freq_space)

    end_preprocessing = time.time()
    print(f"Preprocessing Method: {args.preprocess} | Time: {end_preprocessing - start_preprocessing:.2f}s")



    if args.show_preprocessing_methods == True:
        CSI_MA  = CSI - pp.MA(CSI, args.fs * 1.0)
        CSI_DWT = pp.DWT_components(CSI, target_labels= ["D", "D5", "D4", "D3", "D2", "D"])
        CSI_PCA = pp.PCA_time(CSI_MA, args.fs * 1.0, k=3)
        Plot.plot_csi(CSI_MA, CSI_DWT, CSI_PCA,tx=0)

    tof_dop = MUSIC.ToF_Doppler(args)
    azi_tof = MUSIC.Azi_ToF(args)
    azi_dop = MUSIC.Azi_Dop(args)

    
    frame_idx = 888

    print(f"Processing Frame {frame_idx}... ")

    #Doppler_spec.gen_spectrogram(CSI, args)

    for i in range(1000, 1500, 10):
        frame_idx = i
        print(f"Processing Frame {frame_idx}... ")
        azi_tof.gen_spectrum(CSI, frame_idx, x_axis="azimuth", y_axis="tof")





    #azi_tof.gen_spectrum(CSI, frame_idx, x_axis="azi", y_axis="tof")
    #tof_dop.gen_spectrum(CSI, frame_idx, x_axis="doppler", y_axis="tof")
    #azi_dop.gen_spectrum(CSI, frame_idx, x_axis="azi", y_axis="doppler")
    #LASSO.gen_L2_LASSO_prob(CSI, args, frame_idx)
    #LASSO.gen_MUSIC_weight_L2_LASSO_prob(CSI, args, frame_idx)




    
