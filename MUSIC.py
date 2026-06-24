import numpy as np
import utils
import Plot
import matplotlib.pyplot as plt

def gen_MUSIC_spectrum(frame_idx, CSI, args, avg=True, title="MUSIC"):
    x = cal_smoothed_csi(frame_idx, CSI, args, avg)
    x_cov = cal_smoothed_cov(x)
    print(f"x_cov.shape={x_cov.shape}")
    tau_x, theta_x, P_music_x = cal_spectrum(x_cov, args)

    #Plot.save_as_mat(tau_x, theta_x, P_music_x, frame_idx)
    Plot.plot_spectrum(frame_idx, tau_x, theta_x, P_music_x, args, title=title)

# ⭐️直接生成 1D Azi or ToF MUSIC 。
# 呼叫畫 Azi
def gen_1D_MUSIC_spectrum(CSI, frame_idx, args, Sdim=2, spec_type='Azi', gt_AoA= ""):
    covs = []
    for f in range(frame_idx - 10, frame_idx + 10 + 1):
        covs.append(CSI[f, 0, :]@CSI[f, 0, :].conj().T)  # Tx=0
    cov = np.mean(covs, axis=0)
        # Eigen decomposition
    eig_val, eig_vec = np.linalg.eigh(cov)

    # Sort from large to small
    idx_order = eig_val.argsort()[::-1]
    eig_val = eig_val[idx_order]
    eig_vec = eig_vec[:, idx_order]

    # Noise subspace
    En = eig_vec[:, Sdim:]

    # Azi grid
    theta = np.arange(args.theta_min, args.theta_max + args.theta_step, args.theta_step)

    num_Rx = CSI.shape[2]
    SV = np.zeros((len(theta), num_Rx), dtype=complex)

    for i, theta_i in enumerate(theta):
        SV[i, :] = utils.steering_vector_AoA(theta_i, args, num_Rx)

    # MUSIC denominator: || a^H En ||^2
    A = SV @ En
    PP = np.sum(np.abs(A) ** 2, axis=1)

    P_music = 10 * np.log10(1.0 / (PP + 1e-12))

    return theta, P_music, eig_val

def smooth_csi(csi, num_Rx, num_subcarriers, stream_win, sub_win, sub_stride):
    """
    Build Toeplitz-like smoothed CSI for AoA–ToF. (Use Tx=0)
    """
    # block_size = num_subc_per_block
    block_size = sub_win // sub_stride
    H_list = []
    for r in range(num_Rx):
        H = np.zeros((block_size, block_size), dtype=complex)
        for i in range(block_size):
            start = i * sub_stride
            end = min(start + block_size, num_subcarriers)
            start = end - block_size
            H[i, :] = csi[0, r, start:end]
        H_list.append(H)

    Ncol = stream_win if (num_Rx % 2 == 1) else (stream_win - 1)

    smoothed_csi = np.block([
        [H_list[i + j] for j in range(Ncol)]
        for i in range(stream_win)
    ])
    return smoothed_csi

def cal_smoothed_csi(frame_idx, CSI, args, avg=True):
        smoothed_CSIs = []
        if avg:
            #avg_frame = args.nperseg
            avg_frame = int(args.fs * 0.2)
            for i in range(avg_frame):
                smoothed_csi = smooth_csi(
                    CSI[frame_idx -avg_frame//2 + i], 
                    num_Rx = args.num_Rx, 
                    num_subcarriers = args.num_subcarriers, 
                    stream_win = args.antenna_win, 
                    sub_win = args.subc_win, 
                    sub_stride = args.subc_stride
                    )
                smoothed_CSIs.append(smoothed_csi)

        else:
            smoothed_csi = smooth_csi(
                CSI[frame_idx], 
                num_Rx = args.num_Rx, 
                num_subcarriers = args.num_subcarriers, 
                stream_win = args.antenna_win, 
                sub_win = args.subc_win, 
                sub_stride = args.subc_stride
                )
            smoothed_CSIs.append(smoothed_csi)

        return np.array(smoothed_CSIs)

def cal_smoothed_cov(smoothed_csi):
    #step1. Covariance matrix
    if len(smoothed_csi.shape) == 2:
        smoothed_csi = np.asarray(smoothed_csi, dtype=complex)
        cov = smoothed_csi @ smoothed_csi.conj().T
    elif len(smoothed_csi.shape) == 3:
        cov = 0
        for i in range(smoothed_csi.shape[0]):
            temp_x = np.asarray(smoothed_csi[i], dtype=complex)
            cov += temp_x @ temp_x.conj().T
        cov /= smoothed_csi.shape[0]
    return cov

def cal_spectrum(smoothed_cov, args):
    #print(f"smoothed_cov.shape={smoothed_cov.shape}")
    #step2. Eigen decomposition
    eig_val, eig_vec = np.linalg.eigh(smoothed_cov)
    eig_vec = eig_vec.astype(complex)
    idx_order = eig_val.argsort()[::-1]
    eig_val = eig_val[idx_order]
    eig_vec = eig_vec[:, idx_order]
    
    '''
    eig_val_db = 10 * np.log10(eig_val / np.max(eig_val))
    plt.figure()
    plt.plot(idx_order, eig_val_db, marker='o')
    plt.xlabel('Eigenvalue index (sorted)')
    plt.ylabel('Eigenvalue (dB, normalized)')
    plt.title('Eigenvalue Spectrum in dB')
    plt.grid(True)
    '''

    # Noise subspace
    Sdim = args.Sdim
    #print(f"Selected signal subspace dimension Sdim={Sdim}.")
    #print(f"top 20 eigen vals{eig_val[:20]}")
    N_dim = eig_val.shape[0] - Sdim
    E_n = eig_vec[:, -N_dim:]
    #P_n = E_n @ E_n.conj().T

    # theta candidate
    theta = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
    # tau candidate
    tau = np.arange(args.tau_min, args.tau_max, args.tau_step) #40 points

    # steering_vector length:
    sv_len = args.antenna_win * (args.subc_win // args.subc_stride)
    # calculate all steering vectors at once:
    Steering_Vectors = np.zeros((len(theta), len(tau), sv_len), dtype=complex)
    for i in range(len(theta)):
        for j in range(len(tau)):
            sv = utils.steering_vector_AoA_ToF(theta[i],tau[j],args, args.antenna_win, args.subc_win, args.subc_stride)
            Steering_Vectors[i,j,:] = sv
            #sv_aoa = Azi.steering_vector_AoA(theta[i], args)
            #sv_tof = ToF.steering_vector_ToF(tau[j], args)
            #sv = np.kron(sv_aoa, sv_tof).flatten()
            #Steering_Vectors[i,j,:] = sv

    #print("Steering_Vectors.shape:", Steering_Vectors.shape)

    # MUSIC spectrum
    #Pn = EnEn^H and PP = s^T @ Pn @ s = s^T @ EnEn^H @ s
    #let a = s^T @ En and PP = a^* @ a = |a|^2

    # 1)
    SV_flat = Steering_Vectors.reshape(len(theta) * len(tau), sv_len)

    # SV_flat 的第 p 列，就是某一組 (θ_i, τ_j) 的 steering vector：
    # p = i * num_tau + j 對應 (i, j)

    # 2) 投影到 noise subspace: (T*K, N_dim)
    A = SV_flat @ E_n     # E_n: (N, N_dim)

    # 3) 每個 (θ,τ) 的分母：‖E_n^H s‖²
    PP_flat = np.sum(np.abs(A)**2, axis=1)

    # 4) reshape 回 (θ, τ)
    PP = PP_flat.reshape(len(theta), len(tau))

    # 5) MUSIC spectrum
    P_music = 10 * np.log10(1.0 / PP)

    return tau, theta, P_music
