import argparse
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from signal_processing import signal_processing




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File Config
    parser.add_argument('--foldername', type = str, default = "/Users/YPL/Documents/CSI_files/intermediates/")
    parser.add_argument('--csi_name', type = str, default = "20260224_rh_cir")#left-hand-311 #20260224_rh_cir
    parser.add_argument('--file_name', type = str, default = "*.npz")

    # CSI config
    parser.add_argument('--f_0', type = float, default = 5e9) #5e9
    parser.add_argument('--BW', type = float, default = 160e6)
    parser.add_argument('--fs', type = float, default = 100)

    #args
    args = parser.parse_args()

    # ---- Load CSI ----
    search_path = os.path.join(args.foldername, args.csi_name, "merged_csi", args.file_name)
    npz_files = glob.glob(search_path)

    if len(npz_files) == 0:
        raise FileNotFoundError(f"找不到任何 .npz 檔案於: {args.foldername}")
    elif len(npz_files) > 1:
        print(f"⚠️ 警告：發現多個 .npz 檔案，將自動選擇第一個：{os.path.basename(npz_files[0])}")
    
    target_path = npz_files[0]
    print(f"📁 Loading: {os.path.basename(target_path)}")
    data = np.load(target_path)
    CSI = data[data.files[0]]

    # ----  Tx (2 -> 1) ----
    CSI = np.mean(CSI, axis=1, keepdims=True)
    
    


    # ---- CSI Config From Data ----
    args.num_frames = CSI.shape[0]
    args.num_Tx = CSI.shape[1]
    args.num_Rx = CSI.shape[2]
    args.d = 0.015  # antenna spacing
    args.num_subcarriers = CSI.shape[3]
    args.delta_f = args.BW / args.num_subcarriers
    args.time = args.num_frames / args.fs
    print(f"CSI{CSI.shape} | T:{args.time:.2f}s | fs:{args.fs}")

    # ---- Heatmap Setting ----
    args.projection = 'sin' # 'sin' or 'cos'
    args.subc_stride = 2
    args.stream_win=(args.num_Rx + 2) // 2 
    args.subc_win = args.num_subcarriers // 2
    # ---- Pics Dir ----
    #args.pics_dir = "/Users/YPL/Documents/Experiments/pics/"
    args.pics_dir = None

    # ---- STFT Settings ----



    # Signal Processing
    start = time.time()
    signal_processing(CSI, args)
    
    end = time.time()
    print(f"Execution time: {(end - start):.2f} seconds")

    plt.show()