import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def save_as_mat(tau_grid, theta_grid, x_cvx, frame_idx, title=""):
    mat_data = {
        'tau': tau_grid,
        'theta': theta_grid,
        'P_music': x_cvx,
        'frame_idx': frame_idx
    }
    save_dir = "/Users/YPL/Documents/Experiments/cvx_mat/"
    os.makedirs(save_dir, exist_ok=True)
    # 存為 mat 文件，檔名與圖片對應
    mat_path = os.path.join(save_dir, f"spectrum_{frame_idx:04d}.mat")
    sio.savemat(mat_path, mat_data)
    print(f"Saved: {mat_path}")

def plot_spectrum(frame_idx, tau, theta, P_music, args, title=""):
    #peaks = find_Peaks.find_AoA_ToF_peaks(P_music, theta, tau)
    plt.figure()
    plt.pcolormesh(tau, theta, P_music, cmap = 'jet', shading = 'auto')
    plt.colorbar()
    '''
    for o in range(len(args.x_obj)):
        plt.scatter(args.gt_taus[o][frame_idx], args.gt_AoAs[o][frame_idx], marker='x', s=50,color = obj_colors[o])
    '''
    # 畫出網格線 (選擇性開啟，用於觀察 Grid Refinement 的分佈)
    plt.gca().set_xticks(tau, minor=True)
    plt.gca().set_yticks(theta, minor=True)
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)

    plt.xlabel('tau (s)')
    plt.ylabel('theta (deg)')
    plt.title('AoA-ToF '+ title + ' @ frame ' + str(frame_idx), fontsize = 8)

    # --- save figures ---
    save_dir = args.pics_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{frame_idx:04d}.png"
        )
        plt.savefig(save_path, dpi=100)
        plt.close()

def plot_phase_along_subcarrier(CSI, args, frame_idx, title_prefix=""):
    """
    只畫某一個 frame 的每個 Rx：phase(angle) vs subcarrier index
    CSI shape: (T, Tx, Rx, Subc) complex
    """
    H = CSI[frame_idx, 0]  # (Rx, Subc)
    Nrx, Nsubc = H.shape
    sub_idx = np.arange(Nsubc)

    plt.figure(figsize=(10,4))
    
    for r in range(min(args.num_Rx, Nrx)):
        ph = np.angle(H[r, :])
        # ⭐️ unwrapping
        ph = np.unwrap(ph)
        plt.plot(sub_idx, ph, label=f"Rx{r}")

    plt.xlabel("Subcarrier index")
    plt.ylabel("Phase (rad)")
    plt.title(f"{title_prefix}Phase vs Subcarrier | frame={frame_idx}")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_phase_along_time(CSI, args, subc_idx=0, title_prefix=""):
    """
    只畫某一個 frame 的每個 Rx：phase(angle) vs subcarrier index
    CSI shape: (T, Tx, Rx, Subc) complex
    """
    H = CSI[:, 0, :, subc_idx]  # (Rx, Subc)
    Nrx = CSI.shape[2]
    frame_idx = np.arange(CSI.shape[0])

    plt.figure(figsize=(10,4))
    
    for r in range(min(args.num_Rx, Nrx)):
        ph = np.angle(H[:, r])
        # ⭐️ unwrapping
        ph = np.unwrap(ph)
        plt.plot(frame_idx, ph, label=f"Rx{r}")

    plt.xlabel("time index")
    plt.ylabel("Phase (rad)")
    plt.title(f"{title_prefix}Phase vs Time | subcarrier={subc_idx}")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_phases_along_time(CSI, rx_idx=0, title_prefix=""):
    """
    畫出第 rx_idx 個 Rx 的前30個 subcarrier 的 phase(angle) vs time
    """
    H = CSI[:, 0, rx_idx, 0:30]  # (Rx, Subc)
    Nsubc = 30
    frame_idx = np.arange(CSI.shape[0])

    plt.figure(figsize=(10,4))
    
    for r in range(Nsubc):
        ph = np.angle(H[:, r])
        # ⭐️ unwrapping
        ph = np.unwrap(ph)
        plt.plot(frame_idx, ph, label=f"subcarrier{r}")

    plt.xlabel("time index")
    plt.ylabel("Phase (rad)")
    plt.title(f"{title_prefix}top 30 subcarrier Phase vs Time | Rx={rx_idx}")
    #plt.legend(ncol=5, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_amps_along_time(CSI, rx_idx=0, title_prefix=""):
    """
    畫出第 rx_idx 個 Rx 的前30個 subcarrier 的 phase(angle) vs time
    """
    H = CSI[:, 0, rx_idx, 0:30]  # (Rx, Subc)
    Nsubc = 30
    frame_idx = np.arange(CSI.shape[0])

    plt.figure(figsize=(10,4))
    
    for r in range(Nsubc):
        ph = 10 * np.log10(np.abs(H[:, r]))
        # ⭐️ unwrapping
        ph = np.unwrap(ph)
        plt.plot(frame_idx, ph, label=f"subcarrier{r}")

    plt.xlabel("time index")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{title_prefix}top 30 subcarrier Amp vs Time | Rx={rx_idx}")
    #plt.legend(ncol=5, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
