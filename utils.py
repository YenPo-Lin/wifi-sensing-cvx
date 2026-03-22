import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from utils import*
import cvxpy as cvx
import LASSO

from tqdm import tqdm
import MUSIC

def save_all_MUSIC_spectrum_as_mat(CSI, args, f_start=0, f_end=5, f_step=1):
    P_all = []

    frame_indices = range(f_start, f_end, f_step)
    print(f"🌀 Starting MUSIC processing for {len(frame_indices)} frames...")

    for f_idx in tqdm(frame_indices, desc="Processing Frames", unit="frame"):
            # MUSIC
            x = MUSIC.cal_smoothed_csi(f_idx, CSI, args, avg=True)
            x_cov = MUSIC.cal_smoothed_cov(x)
            _, _, P_music_x = MUSIC.cal_spectrum(x_cov, args)
            P_all.append(1/P_music_x)

    P_all = np.array(P_all)
    #print(f"P_all shape: {P_all.shape}")  # (num_frames, len(theta_grid), len(tau_grid))

    # 封裝數據字典
    mat_data = {
        'P_all': P_all,
        'file_name': args.csi_name,
        'subc_stride': args.subc_stride,
        'stream_win': args.stream_win,
        'subc_win': args.subc_win,
        'theta_min': args.theta_min,
        'theta_max': args.theta_max,
        'theta_step': args.theta_step,
        'tau_min': args.tau_min,
        'tau_max': args.tau_max,
        'tau_step': args.tau_step
    }
    save_dir = args.mat_data_path
    os.makedirs(save_dir, exist_ok=True)
    mat_path = os.path.join(save_dir, f"(MUSIC){args.csi_name}.mat")
    sio.savemat(mat_path, mat_data)
    print(f"✨ Saved {args.csi_name}. Size: {P_all.shape}. Dir: {mat_path}")

def save_all_LASSO_spectrum_as_mat(CSI, args, f_start=100, f_end=105, f_step=1):
    P_all = []
    A, theta_grid, tau_grid = LASSO.build_dictionary(args)

    frame_indices = range(f_start, f_end, f_step)
    print(f"🌀 Starting LASSO processing for {len(frame_indices)} frames...")
    for frame_idx in tqdm(frame_indices, desc="Processing Frames", unit="frame"):
        Y = LASSO.build_Y_packets(CSI, frame_idx, args.num_subcarriers, K_frame=args.multi_frame)
        # --- SVD & Dynamic K Selection ---
        #print(f"Y.shape before SVD {Y.shape}")
        U, S, Vh = np.linalg.svd(Y, full_matrices=False)
        # 方法 1: 使用能量比例
        energy_thresh = args.energy_thresh
        S_sq = S**2
        K_subspace = np.searchsorted(np.cumsum(S_sq), np.sum(S_sq) * energy_thresh) + 1
        K_subspace = np.clip(K_subspace, 2, 10)
        #print(f"Dynamically selected K={K_subspace} (Energy Thresh={energy_thresh})")

        Y = U[:, :K_subspace] @ np.diag(S[:K_subspace])
        #print(f"Y.shape after SVD{Y.shape}")
        

        #print("🐌 Solving Group L2 Lasso... ")

        X_cvx = LASSO.FISTA.FISTA_group_Lasso(A, Y, args.lam, max_iter=args.max_iter, tol=args.tol, verbose=False)
        # X_cvx.shape(G * K)
        X_cvx = np.linalg.norm(X_cvx, axis=1)
        X_cvx = np.abs(X_cvx).reshape(len(theta_grid), len(tau_grid))
        
        pad_theta = 1  # 角度軸邊界
        pad_tau = 1    # 延遲軸邊界
        
        # 將邊緣設為 0
        X_cvx[:pad_theta, :] = 0  # 上邊界
        X_cvx[-pad_theta:, :] = 0 # 下邊界
        X_cvx[:, :pad_tau] = 0    # 左邊界
        X_cvx[:, -pad_tau:] = 0   # 右邊界
        
        P_all.append(X_cvx)

    P_all = np.array(P_all)
    #print(f"P_all shape: {P_all.shape}")  # (num_frames, len(theta_grid), len(tau_grid))

    # 封裝數據字典
    mat_data = {
        'P_all': P_all,
        'file_name': args.csi_name,
        #'subc_stride': args.subc_stride,
        #'stream_win': args.stream_win,
        #'subc_win': args.subc_win,
        'theta_min': args.theta_min,
        'theta_max': args.theta_max,
        'theta_step': args.theta_step,
        'tau_min': args.tau_min,
        'tau_max': args.tau_max,
        'tau_step': args.tau_step,
        'energy_thresh': args.energy_thresh,
        'multi_frame': args.multi_frame,
        'max_iter': args.max_iter,
        'tol': args.tol,
        'lam': args.lam
    }
    save_dir = args.mat_data_path
    os.makedirs(save_dir, exist_ok=True)
    mat_path = os.path.join(save_dir, f"(LASSO){args.csi_name}.mat")
    sio.savemat(mat_path, mat_data)
    print(f"✨ Saved {args.csi_name}. Size: {P_all.shape}. Dir: {mat_path}")


def steering_vector_AoA(theta_i, args, stream_win):
    fc = args.f_0
    theta_i = np.deg2rad(theta_i)

    if args.projection == "sin":
        sv = np.exp(+2j * np.pi * fc * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_win))
    elif args.projection == "cos":
        sv = np.exp(2j * np.pi * fc * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_win))
    return sv.flatten()

def steering_vector_ToF(tau_i, args, subc_win, subc_stride=1):
    row_size = subc_win // subc_stride
    sub_idx = np.arange(0, row_size) * subc_stride  #[0 4 8 ... 252] 64 points
    # carrier delay
    #const_phase = np.exp(-2j * np.pi * args.f_0 * tau_i)
    # subcarrier frequency phase
    sv = np.exp(-2j * np.pi * (args.f_0 +args.delta_f * sub_idx) * tau_i) #64 points #args.f_0 + 
    #sv = subc_phase* const_phase
    return sv.flatten()

def steering_vector_AoA_ToF(theta_i, tau_j, args, stream_win, subc_win, subc_stride=1):
        delta_f = args.delta_f
        theta_i = np.deg2rad(theta_i)
        
        # ----------------------
        # Subcarrier (ToF + carrier) phase
        # ----------------------
        row_size = subc_win // subc_stride #256//4=64
        sub_idx = np.arange(0, row_size) * subc_stride  #[0 4 8 ... 252] 64 points

        # carrier delay
        #const_phase = np.exp(-2j * np.pi * fc * tau_j)
        # subcarrier frequency phase
        exp_omega = np.exp(-2j * np.pi * (args.f_0+ delta_f * sub_idx) * tau_j) #64 points


        # ----------------------
        # Spatial phase (AoA) #3
        # ----------------------
        if args.projection == "sin":
            exp_phi = np.exp(+2j * np.pi * args.f_0 * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_win))
        elif args.projection == "cos":
            exp_phi = np.exp(2j * np.pi * args.f_0 * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_win))
        
        # ----------------------
        # Steering vector = Kronecker of spatial and subcarrier vectors
        # ----------------------
        steering_vector = np.kron(exp_phi, exp_omega) #3*64=192
        #print(f"Steering vector shape: {steering_vector.shape}")
        return steering_vector.flatten()

def find_peaks(P, theta_grid, tau_grid,
                            ratio=0.3,        # 只留 >= ratio*max 的 peaks
                            K_max=50,
                            theta_margin=3, tau_margin=3,
                            min_dist=3, print_peaks=True):
    P = P.copy()
    N_theta, N_tau = P.shape

    # 邊界 mask
    mask = np.ones_like(P, dtype=bool)
    mask[:theta_margin, :] = False
    mask[-theta_margin:, :] = False
    mask[:, :tau_margin] = False
    mask[:, -tau_margin:] = False

    P_work = np.where(mask, P, -np.inf)

    # 以「全圖最大」定門檻
    max0 = np.max(P_work)
    if not np.isfinite(max0):
        return []

    thr = ratio * max0

    peaks = []
    for _ in range(K_max):
        idx = np.argmax(P_work)
        v = P_work.flat[idx]
        if (not np.isfinite(v)) or (v < thr):
            break

        i_theta, i_tau = np.unravel_index(idx, P_work.shape)
        peaks.append((i_theta, i_tau, theta_grid[i_theta], tau_grid[i_tau], v))

        # NMS 抑制鄰近
        i_min = max(i_theta - min_dist, 0)
        i_max = min(i_theta + min_dist + 1, N_theta)
        j_min = max(i_tau - min_dist, 0)
        j_max = min(i_tau + min_dist + 1, N_tau)
        P_work[i_min:i_max, j_min:j_max] = -np.inf
    if print_peaks:
            print(f"{'Index':<5} | {'Theta (deg)':<12} | {'Tau (s)':<12} | {'Magnitude'}")
            print("-" * 50)
            for i, p in enumerate(peaks):
                # p 的內容為 (i_theta, i_tau, theta_val, tau_val, value)
                _, _, theta_val, tau_val, val = p
                print(f"#{i+1:<4} | {theta_val:<12.2f} | {tau_val:<12.2e} | {val:.4f}")

    return peaks
