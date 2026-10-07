import numpy as np
import pre_processing as pp
import MUSIC
import time
import point_cloud

def weight_gen():
    pass


def weight_test(raw_CSI, args):

    CSI = np.abs(raw_CSI)# take absolute value to get power
    background = pp.MA(CSI, args.avg_frames) # moving average background

    if args.preprocess == "ma":
        CSI = (CSI - background) # Original Version

        
    azi_tof = MUSIC.Azi_ToF(args)

    frame_idx = args.frame_idz
    results = azi_tof.gen_Doppler_projection_spectrum(
        CSI, 
        frame_idx, 
        fd_neighbor=0.5, 
        doppler_step=0.25
        window="hann",
        normalize=False,
        fig_name="Project_tg",
    )

    # weight generation

    weight = g

    results = azi_tof.gen_Doppler_projection_spectrum(
        CSI, 
        frame_idx, 
        fd_neighbor=0.5, 
        doppler_step=0.25
        window="weight",
        normalize=False,
        fig_name="Project_tg_weight",
    )
