import numpy as np
import pre_processing as pp
import MUSIC
import LASSO
import os
import time




def signal_processing(raw_CSI, args):
    azi_tof = MUSIC.Azi_ToF(args)
    tof_dop = MUSIC.ToF_Doppler(args)

    CSI = pp.self_sanitize(raw_CSI)
    #CSI = np.mean(CSI, axis=1, keepdims=True)
    if args.preprocess == "ma":
        CSI  -= pp.MA(CSI, args.fs * 1.0)
    elif args.preprocess == "dwt":
        CSI = pp.DWT_components(CSI, target_labels= ["D", "D5", "D4", "D3", "D2", "D"])

    # Re sampling
    CSI = pp.sample_subcarriers(args, CSI, freq_space=args.freq_space)


    
    frame_idx = 1300

    print(f"Processing Frame {frame_idx}... ")

    #_,_,_ = LASSO.gen_L2_LASSO_prob(CSI_dwt, args, frame_idx, Nrx=args.num_Rx, Nsubc=args.num_subcarriers, lam=args.lam)

    #azi_tof.gen_MUSIC_spectrum(frame_idx, CSI)
    start = time.time()
    #tof_dop.gen_spectrum(CSI, frame_idx)
    for i in range(800, 1200, 60):
        tof_dop.gen_spectrum(CSI, i)
    #azi_tof.gen_spectrum(CSI, frame_idx)
    end = time.time()
    print(f"Heatmap Computation Time: {end - start}")



    
