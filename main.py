import os
import time
import argparse
import numpy as np
from signal_processing import signal_processing



def create_parser():
    parser = argparse.ArgumentParser()

    # npz 文件路徑
    file_path = "/Users/YPL/Documents/NPZ_files/20261002-205358_ten_clap_256.npz"
    parser.add_argument('--csi_file', type=str, default=file_path)
    
    # ---- CSI parameters ----
    parser.add_argument('--f_0', type=float, default=5.57e9)
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--antenna_spacing', type=float, default=0.02)
    
    # ---- MUSIC settings ----
    parser.add_argument('--preprocess', type=str, default='ma', choices=['ma', 'dwt', 'pca'])
    # plotted frame
    parser.add_argument('--frame_idx', type=int, default=1100)
    # MUSIC signal dimension
    parser.add_argument('--Dop_Sdim', type=int, default=None)
    parser.add_argument('--Azi_ToF_Sdim', type=int, default=7)
    parser.add_argument('--ToF_Dop_Sdim', type=int, default=None)
    parser.add_argument('--Azi_Dop_Sdim', type=int, default=None)
    parser.add_argument('--Azi_ToF_Dop_Sdim', type=int, default=None)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.8)
    parser.add_argument('--avg_frames', type=int, default=100)
    parser.add_argument('--projection', type=str, default='cos', choices=['sin', 'cos'])

    parser.add_argument('--stream_win', type=int, default=5) ## Azimuth Steering Vector length = stream_win
    parser.add_argument('--stream_sample_range', type=int, default=8) #all Rx

    parser.add_argument('--freq_win', type=int, default=240) 
    parser.add_argument('--freq_hop', type=int, default=10) ## ToF Steering Vector length = freq_win//freq_hop 
    parser.add_argument('--freq_sample_range', type=int, default=256) #all subcarriers
    parser.add_argument('--freq_space', type=int, default=1) # if freq resampling


    parser.add_argument('--time_win', type=int, default= 20)
    parser.add_argument('--time_hop', type=int, default=1) ## Doppler Steering Vector length = time_win//time_hop
    parser.add_argument('--time_sample_range', type=int, default=50) 
    parser.add_argument('--conti_hop', type=int, default=1)
    parser.add_argument('--top_sc_ratio', type=float, default=0.5)
    parser.add_argument('--sc_selection_method', choices=['snr'], default='snr')
    parser.add_argument('--conjugate_order', choices=['fixed', 'ratio'], default='ratio')

    # Azimuth grid
    parser.add_argument('--theta_min', type=float, default= 0)
    parser.add_argument('--theta_max', type=float, default= 180)
    parser.add_argument('--theta_step', type=int, default=3)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='m', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=0e-9)
    parser.add_argument('--tau_max', type=float, default=10e-9)
    parser.add_argument('--tau_step', type=float, default=2e-10)
    # Doppler grid
    parser.add_argument('--doppler_min', type=float, default=-30)
    parser.add_argument('--doppler_max', type=float, default=30)
    parser.add_argument('--doppler_step', type=float, default=1)

    # Doppler spectrogram settings
    parser.add_argument('--stft_nperseg', type=int, default=64)
    parser.add_argument('--stft_noverlap', type=int, default=63) # hop 1

    # heatmap axis (X: Azi, Y: TOF if True)
    parser.add_argument('--axis_flip', type=bool, default=True)
    parser.add_argument('--colorbar', type=bool, default=True)

    # 3D MUSIC cube display

    parser.add_argument('--cube_dynamic_range_db', type=float, default=10.0)
    # 數值變小：保留範圍更窄，圖形更集中於峰值。
    # 數值變大：保留更多低功率點，point cloud 更散。
    parser.add_argument('--cube_percentile', type=float, default=95.0)
    # 只保留功率位於前 10% 的點。
    parser.add_argument('--cube_max_points', type=int, default=60000)
    parser.add_argument('--cube_point_size', type=float, default=3.0)
    parser.add_argument('--cube_point_alpha_min', type=float, default=0.12)
    parser.add_argument('--cube_point_alpha_max', type=float, default=5.0)
    parser.add_argument('--cube_point_alpha_gamma', type=float, default=5)
    # gamma 越高，藍色區域越透明，黃紅色峰值仍接近不透明

    # Doppler Projection settings
    parser.add_argument('--tof_gating', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tof_gate', type=float, default=3e-9)
    
    # ---- 圖片保存路徑 ----
    parser.add_argument('--pics_dir', type=str, default=None)
    
    # ----CFAR ----
    parser.add_argument('--cfar_training_cells', type=int, default=2)
    parser.add_argument('--cfar_guard_cells', type=int, default=1)
    parser.add_argument('--cfar_threshold_factor', type=float, default=1.05)
    parser.add_argument('--cfar_top_k', type=int, default=20)
    parser.add_argument('--cfar_min_peak_distance', type=int, default=1)
    parser.add_argument('--cfar_min_prominence_db', type=float, default=0.0)
    parser.add_argument('--fd_band', type=float, default=2.0)


    
    return parser


if __name__ == '__main__':
    # 解析命令行參數
    parser = create_parser()
    args = parser.parse_args()


    if not os.path.isfile(args.csi_file):
        raise FileNotFoundError(f"❌ Cannot find CSI: {args.csi_file}")
    else:
        if args.pics_dir is None:
            npz_name = os.path.splitext(os.path.basename(args.csi_file))[0]
            args.pics_dir = os.path.join(os.getcwd(), npz_name)

        with np.load(args.csi_file, allow_pickle=False) as metadata:
            if "csi" not in metadata.files:
                raise KeyError(
                    f"NPZ files not in ：{args.csi_file}; "
                    f"keys: {metadata.files}"
                )

            CSI = metadata["csi"]
            if CSI.ndim != 4:
                raise ValueError(
                    "CSI needs to be 4D array (frame, Tx, Rx, subcarrier), "
                    f"{CSI.shape}。"
                )

            print(f"📁 LOADING: {os.path.basename(args.csi_file)}")
            print(
                f"✅ CSI{CSI.shape} | dtype:{CSI.dtype} | "
                f"Frames:{CSI.shape[0] / args.fs:.2f}s | Fs:{args.fs}Hz"
            )
            args.num_frames, args.num_Tx, args.num_Rx, args.num_sc = CSI.shape

            if "reference_rx_system_ns" in metadata.files:
                timestamp_ns = metadata["reference_rx_system_ns"]
                duration_s = (timestamp_ns[-1] - timestamp_ns[0]) / 1e9
                effective_fs = (len(timestamp_ns) - 1) / duration_s
                print(f"Duration            : {duration_s:.3f} s")
                print(f"Sampling Rate       : {effective_fs:.3f} Hz")

            if {"overall_packet_loss", "overall_total_packet"}.issubset(metadata.files):
                print(
                    f"Packet Loss         : {metadata['overall_packet_loss'].item()} / "
                    f"{metadata['overall_total_packet'].item()}"
                )

            if "nominal_bandwidth_hz" in metadata.files:
                print(
                    f"Bandwidth           : "
                    f"{metadata['nominal_bandwidth_hz'].item() / 1e6:.1f} MHz"
                )
                args.BW = float(metadata["nominal_bandwidth_hz"].item())
            if "frequency_spacing_hz" in metadata.files:
                print(
                    f"Frequency Spacing   : "
                    f"{metadata['frequency_spacing_hz'].item() / 1e6:.6f} MHz"
                )
                args.delta_f = float(metadata["frequency_spacing_hz"].item())
            if "frequency_offsets_hz" in metadata.files:
                offsets = np.asarray(metadata["frequency_offsets_hz"], dtype=float)
                if offsets.shape != (CSI.shape[-1],) or not np.all(np.isfinite(offsets)):
                    raise ValueError("frequency_offsets_hz must match the CSI subcarrier axis")
                args.frequency_offsets_hz = offsets
            elif args.delta_f is None and "frequency_spacing_hz" in metadata.files:
                args.delta_f = float(metadata["frequency_spacing_hz"].item())



    print("🥶 Start Signal Processing...")
    start = time.time()
    signal_processing(CSI, args)
    
    elapsed_time = time.time() - start
    print(f"🥶 End Signal Processing: {elapsed_time:.2f} (s)")
