import numpy as np
import pre_processing as pp
import MUSIC
import time
from beamform_rd_map import Beamform_RD_Map
from stability_check import RDStabilityCheck
import Doppler_spec




def signal_processing(raw_CSI, BG, args):

    raw_CSI_amp = np.abs(raw_CSI)
    BG = np.asarray(BG)
    if BG.ndim != 4 or BG.shape[0] == 0 or BG.shape[1:] != raw_CSI_amp.shape[1:]:
        raise ValueError("BG must match the CSI Tx/Rx/subcarrier shape and contain frames")
    background_CSI = np.mean(np.abs(BG), axis=0, keepdims=True)
    CSI2 = raw_CSI_amp - background_CSI

    if args.preprocess == "bg":
        CSI = CSI2

    elif args.preprocess == "ma":
        static_CSI = pp.MA(raw_CSI_amp, args.avg_frames)
        CSI = (raw_CSI_amp - static_CSI)
    else:
        raise ValueError("preprocess must be 'ma' or 'bg'")


    Doppler_spec.gen_spectrogram(CSI, args, tx=0)

    frame_idx = args.frame_idx

    dop = MUSIC.Doppler(args)
    azi_tof = MUSIC.Azi_ToF(args)
    aod_tof = MUSIC.AoD_ToF(args)
    dop_proj_azi_tof = MUSIC.DopplerProject_Azi_ToF(args, raw_CSI)
    tof_dop = MUSIC.ToF_Dop(args)
    #azi_dop = MUSIC.Azi_Dop(args)
    azi_tof_dop = MUSIC.Azi_ToF_Dop(args)
    beamformer = Beamform_RD_Map(args)




    #InterCard_RD_Map(args).gen_spectrum(raw_CSI, frame_idx)

    #dop.gen_spectrum(CSI, frame_idx, sc_select=True, )




    tof_dop.gen_spectrum(CSI, frame_idx)
    azi_tof.gen_spectrum(CSI2, frame_idx)
    #dop_proj_azi_tof.gen_beamform_ToF_Dop(CSI, frame_idx, file_name="beamform")
    #azi_tof_dop.gen_spectrum(CSI, frame_idx)



    result = beamformer.gen_beamform(
        CSI_for_RA=CSI2,
        CSI_for_RD=CSI,
        frame_idx=frame_idx,
        plot_beamform=True
        )

    stability_checker = RDStabilityCheck(args, beamformer)
    fd_step = 0.5
    stable_targets = stability_checker.stable_targets_from_RDmap(
        CSI, frame_idx, observation_interval=args.fs * 0.01,
        center_result=result, peak_source="candidate", fd_step=fd_step,
        CSI_for_RA=CSI2,
    )

    #dop_proj_azi_tof.gen_Doppler_specific_spectrum(CSI, frame_idx, stable_targets, fd_neighbor=0.5, fd_step=0.5, window="rect")


    results = dop_proj_azi_tof.gen_Doppler_projected_spectrum(
        CSI2, frame_idx, fd_neighbor=None, fd_step=fd_step, window="rect",
        beamform_result=result,
        fuse=True, stable_targets=stable_targets,
    )



    



    # 2D Azi-ToF spectrum
    for frame_idx in range(300, 2000, 40):
        #tof_dop.gen_spectrum(CSI, frame_idx)
        azi_tof.gen_spectrum(CSI2, frame_idx)
        """
        result = beamformer.gen_beamform(
            CSI_for_RA=CSI2,
            CSI_for_RD=CSI,
            frame_idx=frame_idx,
            plot_beamform=False
            )

        stability_checker = RDStabilityCheck(args, beamformer)
        fd_step = 0.5
        stable_targets = stability_checker.stable_targets_from_RDmap(
            CSI, frame_idx, observation_interval=args.fs * 0.01,
            center_result=result, peak_source="candidate", fd_step=fd_step,
            CSI_for_RA=CSI2,
        )

        #dop_proj_azi_tof.gen_Doppler_specific_spectrum(CSI, frame_idx, stable_targets, fd_neighbor=0.5, fd_step=0.5, window="rect")


        results = dop_proj_azi_tof.gen_Doppler_projected_spectrum(
            CSI2, frame_idx, fd_neighbor=None, fd_step=fd_step, window="rect",
            beamform_result=result,
            fuse=True, stable_targets=stable_targets,
        )
        """


        #result = beamformer.gen_beamform(CSI, frame_idx)
        #dop_proj_azi_tof.gen_beamform_ToF_Dop(CSI, frame_idx, file_name="beamform")
        #result = beamformer.gen_beamform(CSI, frame_idx)
        #tof_dop.gen_spectrum(CSI, frame_idx)

        #azi_tof.gen_spectrum(CSI, frame_idx)
        #result = beamformer.gen_beamform(CSI, frame_idx)
        #aod_tof.gen_spectrum(CSI, frame_idx)

        #azi_dop.gen_spectrum(CSI, frame_idx)
        # 3D spectrum
        #_ = azi_tof_dop.gen_spectrum(CSI, frame_idx, method="sum", fig_name="3D_MUSIC",)

        # 3D spectrum with Doppler RoI
        #roi_results = azi_tof_dop.gen_Doppler_RoI_specturm(CSI, frame_idx, fd_range=2, fig_name="3D_MUSIC_RoI")
        #targets = [result["target"] for result in roi_results]

        pass
