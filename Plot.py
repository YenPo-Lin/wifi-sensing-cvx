import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


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
    plt.title('Azi-ToF '+ title + ' @ frame ' + str(frame_idx), fontsize = 8)

    # --- save figures ---
    save_dir = args.pics_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{frame_idx:04d}.png"
        )
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Saved: {save_path}")

def plot_spectrum3D(frame_idx, tau, theta, P_music, args, title=""):
    # 建立畫布，並指定為 3D 投影
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 建立 2D 網格 (plot_surface 需要 X, Y 都是 2D 矩陣)
    if tau.ndim == 1 and theta.ndim == 1:
        Tau_grid, Theta_grid = np.meshgrid(tau, theta)
    else:
        Tau_grid, Theta_grid = tau, theta

    # 2. 避免微小虛數引發警告，取絕對值
    Z = np.abs(P_music)

    # 3. 繪製 3D 曲面 (山谷/地形圖效果)
    # rstride 與 cstride 控制網格粗細，cmap 延續你原本的 'jet'
    surf = ax.plot_surface(Tau_grid, Theta_grid, Z, cmap='jet', 
                           linewidth=0, antialiased=True, shade=True)

    # 加上 Colorbar，並調整大小比例避免過大
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # 4. 設定視角 (仰角 elev, 方位角 azim)，可根據需求調整以看清楚山谷深度
    ax.view_init(elev=30, azim=-45)

    # 5. 設定標籤與標題 (延續前面改為 ns 的單位)
    ax.set_xlabel('ToF (ns)', labelpad=10)
    ax.set_ylabel('Azi (deg)', labelpad=10)
    ax.set_zlabel('Spectrum Power', labelpad=10)
    ax.set_title('3D Azi-ToF ' + title + ' @ frame ' + str(frame_idx), fontsize=10)

