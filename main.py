import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from signal_processing import signal_processing


def create_parser():
    parser = argparse.ArgumentParser()

    # npz 文件路徑
    parser.add_argument(
        '--csi_file', 
        type=str, 
        default="/Users/YPL/Documents/Experiments/CSI_files/intermediates/left-hand-311/merged_csi/left-hand-311.npz"
        )
    
    # ---- CSI parameters ----
    parser.add_argument('--f_0', type=float, default=5.57e9)
    parser.add_argument('--BW', type=float, default=160e6)
    parser.add_argument('--delta_f', type=float, default=78.125e3) # 160 M /2024 = 78.125 kHz
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--antenna_spacing', type=float, default=0.015)
    
    # ---- MUSIC settings ----
    parser.add_argument('--preprocess', type=str, default='ma', choices=['ma', 'dwt'])
    parser.add_argument('--Sdim', type=int, default=12)
    parser.add_argument('--projection', type=str, default='sin', choices=['sin', 'cos'])

    parser.add_argument('--antenna_win', type=int, default=5)
    parser.add_argument('--subc_win', type=int, default=1024)
    parser.add_argument('--subc_stride', type=int, default=16)
    parser.add_argument('--theta_min', type=float, default= -90)
    parser.add_argument('--theta_max', type=float, default= 90)
    parser.add_argument('--theta_step', type=int, default=3)
    parser.add_argument('--tau_min', type=float, default=0)
    parser.add_argument('--tau_max', type=float, default=20e-9)
    parser.add_argument('--tau_step', type=float, default=5e-10)
    
    # ---- 圖片保存路徑 ----
    parser.add_argument('--pics_dir', type=str, default=None)
    
    # ---- L2 LASSO 設置 ----
    parser.add_argument('--energy_thresh', type=float, default=0.98)
    parser.add_argument('--multi_frame', type=int, default=11)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--tol', type=float, default=5e-3)
    parser.add_argument('--lam', type=float, default=0.4)
    
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

    signal_processing(CSI, args)
    
    elapsed_time = time.time() - start
    print(f"🥶 End Signal Processing: {elapsed_time:.2f} (s)")

    plt.show()