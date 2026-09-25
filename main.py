import os
import time
import argparse
import numpy as np
from signal_processing import signal_processing



def create_parser():
    parser = argparse.ArgumentParser()

    # npz 文件路徑
    file_path = "/Users/YPL/Documents/NPZ_files/20260820-172856_move_fb.npz"
    parser.add_argument('--csi_file', type=str, default=file_path)
    
    # ---- CSI parameters ----
    parser.add_argument('--f_0', type=float, default=5.57e9)
    parser.add_argument('--BW', type=float, default=160e6)
    parser.add_argument('--delta_f', type=float, default=2.54e6) 
    # 160 M /2024 = 79.05 kHz 
    # 160 M /63 =  2.54 MHz
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--antenna_spacing', type=float, default=0.02)
    
    # ---- MUSIC settings ----
    parser.add_argument('--preprocess', type=str, default='ma', choices=['ma', 'dwt', 'pca'])
    # plotted frame
    parser.add_argument('--frame_idx', type=int, default=660)
    # MUSIC signal dimension
    parser.add_argument('--Azi_ToF_Sdim', type=int, default=None)
    parser.add_argument('--ToF_Dop_Sdim', type=int, default=2)
    parser.add_argument('--Azi_Dop_Sdim', type=int, default=None)
    parser.add_argument('--Azi_ToF_Dop_Sdim', type=int, default=None)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.60)
    parser.add_argument('--avg_frames', type=int, default=50)
    parser.add_argument('--projection', type=str, default='cos', choices=['sin', 'cos'])

    parser.add_argument('--stream_win', type=int, default=5)
    parser.add_argument('--stream_sample_range', type=int, default=8) #all Rx

    parser.add_argument('--freq_win', type=int, default=40) #block size = freq_win // freq_hop
    parser.add_argument('--freq_hop', type=int, default=3)
    parser.add_argument('--freq_sample_range', type=int, default=64) #all subcarriers
    parser.add_argument('--freq_space', type=int, default=1) # if freq resampling


    parser.add_argument('--time_win', type=int, default=20)
    parser.add_argument('--time_hop', type=int, default=1)
    parser.add_argument('--time_sample_range', type=int, default=50) #100 frames

    # Azimuth grid
    parser.add_argument('--theta_min', type=float, default= 20)
    parser.add_argument('--theta_max', type=float, default= 160)
    parser.add_argument('--theta_step', type=int, default=2)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='ns', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=2e-9)
    parser.add_argument('--tau_max', type=float, default=16e-9)
    parser.add_argument('--tau_step', type=float, default=2e-10)
    # Doppler grid
    parser.add_argument('--doppler_min', type=float, default=-20)
    parser.add_argument('--doppler_max', type=float, default=20)
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
    parser.add_argument('--cube_point_size', type=float, default=4.0)
    parser.add_argument('--cube_point_alpha_min', type=float, default=0.1)
    parser.add_argument('--cube_point_alpha_max', type=float, default=3.0)
    parser.add_argument('--cube_point_alpha_gamma', type=float, default=2.5)
    # gamma 越高，藍色區域越透明，黃紅色峰值仍接近不透明
    
    
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
        raise FileNotFoundError(f"❌ 找不到 CSI 檔案: {args.csi_file}")
    else:
        if args.pics_dir is None:
            npz_name = os.path.splitext(os.path.basename(args.csi_file))[0]
            args.pics_dir = os.path.join(os.getcwd(), npz_name)

        # ---- Load CSI ----
        print(f"📁 LOADING: {os.path.basename(args.csi_file)}")
        data = np.load(args.csi_file)
        CSI = data[data.files[0]]

        # ---- Read CSI dimensions ----
        args.num_frames, args.num_Tx, args.num_Rx, args.num_sc = CSI.shape
        print(f"✅ CSI{CSI.shape} | Frames:{args.num_frames/args.fs:.2f}s | Fs:{args.fs}Hz")

    # ---- 開始信號處理 ----
    print("🥶 Start Signal Processing...")
    start = time.time()
    signal_processing(CSI, args)
    
    elapsed_time = time.time() - start
    print(f"🥶 End Signal Processing: {elapsed_time:.2f} (s)")
