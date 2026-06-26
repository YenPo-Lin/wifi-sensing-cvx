import numpy as np
import pre_processing as pp
import MUSIC
import Plot
import LASSO
import os
import time




def signal_processing(raw_CSI, args):

    start_preprocessing = time.time()
    CSI = pp.self_sanitize(raw_CSI)

    #CSI = np.mean(CSI, axis=1, keepdims=True)
    if args.preprocess == "ma":
        CSI  -= pp.MA(CSI, args.fs * 1.0)
        pass
    elif args.preprocess == "dwt":
        CSI = pp.DWT_components(CSI, target_labels= ["D", "D5", "D4", "D3", "D2", "D"])
    elif args.preprocess == "pca":
        CSI  -= pp.MA(CSI, args.fs * 0.5)
        CSI = pp.PCA_time(CSI, args.fs * 1.0, k=3)

    # Re sampling
    CSI = pp.sample_subcarriers(args, CSI, freq_space=args.freq_space)

    end_preprocessing = time.time()
    print(f"Preprocessing Time: {end_preprocessing - start_preprocessing:.2f}s")



    if args.show_preprocessing_methods == True:
        CSI_MA  = CSI - pp.MA(CSI, args.fs * 1.0)
        CSI_DWT = pp.DWT_components(CSI, target_labels= ["D", "D5", "D4", "D3", "D2", "D"])
        CSI_PCA = pp.PCA_time(CSI_MA, args.fs * 1.0, k=3)
        Plot.plot_csi(CSI_MA, CSI_DWT, CSI_PCA,tx=0)

    tof_dop = MUSIC.ToF_Doppler(args)
    azi_tof = MUSIC.Azi_ToF(args)



    
    frame_idx = 1666

    print(f"Processing Frame {frame_idx}... ")

    #_,_,_ = LASSO.gen_L2_LASSO_prob(CSI_dwt, args, frame_idx, Nrx=args.num_Rx, Nsubc=args.num_subcarriers, lam=args.lam)

    #azi_tof.gen_MUSIC_spectrum(frame_idx, CSI)
    #tof_dop.gen_spectrum(CSI, frame_idx)
    start = time.time()
    """
    for i in range(1000, 1100, 10):
        azi_tof.gen_spectrum(CSI, i)

    etime = time.time() - start
    print(f"Average Time per frame: {(etime/10):.2f} (s)")
    """
    start = time.time()
    tof_dop.gen_spectrum(CSI, frame_idx)
    etime = time.time() - start
    print(f"Average ToF-Doppler Time per frame: {(etime/10):.2f} (s)")

    start = time.time()
    azi_tof.gen_spectrum(CSI, frame_idx)
    etime = time.time() - start
    print(f"Average Azimuth-ToF Time per frame: {(etime/10):.2f} (s)")




    
