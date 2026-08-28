import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from signal_processing import signal_processing
import WIDFS


def create_parser():
    parser = argparse.ArgumentParser()

    # npz 文件路徑
    file_path = "/Users/YPL/Documents/NPZ_files/20260821-125845_big_swing.npz"
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
    parser.add_argument('--frame_idx', type=int, default=1300)
    # MUSIC signal dimension
    parser.add_argument('--Sdim', type=int, default=None)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.8)
    parser.add_argument('--tof_dop_Sdim', type=int, default=None)
    parser.add_argument('--tof_dop_Sdim_energy_ratio', type=float, default=None)
    parser.add_argument('--avg_frames', type=int, default=50)
    parser.add_argument('--projection', type=str, default='cos', choices=['sin', 'cos'])

    parser.add_argument('--stream_win', type=int, default=5)
    parser.add_argument('--stream_sample_range', type=int, default=8) #all Rx

    parser.add_argument('--freq_win', type=int, default=48) #block size = freq_win // freq_hop
    parser.add_argument('--freq_hop', type=int, default=3)
    parser.add_argument('--freq_sample_range', type=int, default=64) #all subcarriers
    parser.add_argument('--freq_space', type=int, default=1) # if freq resampling


    parser.add_argument('--time_win', type=int, default=24)
    parser.add_argument('--time_hop', type=int, default=1)
    parser.add_argument('--time_sample_range', type=int, default=100) #100 frames

    # Azimuth grid
    parser.add_argument('--theta_min', type=float, default= 0)
    parser.add_argument('--theta_max', type=float, default= 180)
    parser.add_argument('--theta_step', type=int, default=3)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='ns', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=2e-9)
    parser.add_argument('--tau_max', type=float, default=20e-9)
    parser.add_argument('--tau_step', type=float, default=3e-10)
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
    
    
    # ---- 圖片保存路徑 ----
    parser.add_argument('--pics_dir', type=str, default=None)
    
    # ----CFAR ----
    parser.add_argument('--cfar_training_cells', type=int, default=2)
    parser.add_argument('--cfar_guard_cells', type=int, default=1)
    parser.add_argument('--cfar_threshold_factor', type=float, default=1.05)
    parser.add_argument('--cfar_top_k', type=int, default=20)
    parser.add_argument('--cfar_min_peak_distance', type=int, default=1)
    parser.add_argument('--cfar_min_prominence_db', type=float, default=0.0)

    # ---- WIDFS signed Doppler gating ----
    parser.add_argument('--widfs_min_abs_fd', type=float, default=0.5)
    parser.add_argument('--widfs_max_targets', type=int, default=10)
    parser.add_argument('--widfs_window_size', type=int, default=100)
    parser.add_argument('--widfs_band_half_width', type=float, default=10.0)
    parser.add_argument('--widfs_gate_half_width', type=float, default=1.0)
    parser.add_argument('--widfs_weight_percentile', type=float, default=95.0)
    parser.add_argument(
        '--widfs_peak_projection_method',
        type=str,
        default='max',
        choices=['sum', 'max', 'mean'],
    )
    parser.add_argument(
        '--widfs_heatmap_projection_method',
        type=str,
        default='max',
        choices=['sum', 'max', 'mean'],
    )

    
    return parser


if __name__ == '__main__':
    # 解析命令行參數
    parser = create_parser()
    args = parser.parse_args()

    target_path = args.csi_file
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"❌ 找不到 CSI 檔案: {target_path}")
    
    # ---- Load CSI ----
    print(f"📁 LOADING: {os.path.basename(target_path)}")
    data = np.load(target_path)
    CSI = data[data.files[0]]

    # ---- Read CSI dimensions ----
    args.num_frames = CSI.shape[0]
    args.num_Tx = CSI.shape[1]
    args.num_Rx = CSI.shape[2]
    args.num_subcarriers = CSI.shape[3]
    #args.delta_f = args.BW / args.num_subcarriers
    
    print(f"✅ CSI{CSI.shape} | Frames:{args.num_frames/args.fs:.2f}s | Fs:{args.fs}Hz")

    # ---- 開始信號處理 ----
    print("🥶 Start Signal Processing...")
    start = time.time()

    #WIDFS.dfs_weighted_heatmap(CSI, args)
    WIDFS.dfs_weighted_heatmap(CSI, args)
    #signal_processing(CSI, args)
    
    elapsed_time = time.time() - start
    print(f"🥶 End Signal Processing: {elapsed_time:.2f} (s)")

    plt.show()
