import numpy as np
import os
from tqdm import tqdm
import Plot
import matplotlib.pyplot as plt

def estimate_Sdim(Rxx, energy_ratio=0.88, min_dim=1, max_dim=None):
    arr = np.asarray(Rxx)

    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Rxx must be square, got {arr.shape}")
        eig_val = np.linalg.eigvalsh(arr)
    elif arr.ndim == 1:
        eig_val = arr
    else:
        raise ValueError(f"Invalid Rxx shape: {arr.shape}")

    eig_val = np.sort(np.real(eig_val))[::-1]
    eig_val = np.maximum(eig_val, 1e-12)

    M = eig_val.size
    if M < 2:
        raise ValueError("Need at least two eigenvalues.")

    if max_dim is None:
        max_dim = M - 1

    min_dim = int(np.clip(min_dim, 1, M - 1))
    max_dim = int(np.clip(max_dim, min_dim, M - 1))
    energy_ratio = float(np.clip(energy_ratio, 1e-6, 1.0))

    cumulative = np.cumsum(eig_val) / np.sum(eig_val)
    Sdim = np.searchsorted(cumulative, energy_ratio) + 1
    return int(np.clip(Sdim, min_dim, max_dim))

def resolve_Sdim(args, eig_val, label="MUSIC"):
    M = len(eig_val)

    if getattr(args, "Sdim", None) is not None:
        Sdim = int(args.Sdim)
    else:
        energy_ratio = getattr(
            args,
            "Sdim_energy_ratio",
            getattr(args, "Sdim_energy_ratio", 0.88),
        )
        Sdim = estimate_Sdim(
            eig_val,
            energy_ratio=energy_ratio,
            min_dim=getattr(args, "Sdim_min", 1),
            max_dim=M - 1,
        )
        print(f"Estimated Sdim={Sdim} (energy_ratio={energy_ratio:.2f})")

    Sdim = int(np.clip(Sdim, 1, M - 1))
    return Sdim

class SteeringVector:
    def __init__(self, args):
        self.args = args
        self.f_0 = args.f_0
        self.projection = args.projection
        self.antenna_spacing = args.antenna_spacing
        self.stream_win = args.stream_win
        self.freq_win = args.freq_win
        self.freq_hop = args.freq_hop
        self.dop_win = args.time_win
        self.fs = args.fs
        self.delta_f = float(getattr(args, "delta_f", args.BW / args.num_subcarriers))

    def steering_vector_AoA(self, theta_i, stream_win=None):
        if stream_win is None:
            stream_win = self.stream_win

        theta_i = np.deg2rad(theta_i)
        if self.projection == "sin":
            sv = np.exp(
                +2j
                * np.pi
                * self.f_0
                * np.sin(theta_i)
                * self.antenna_spacing
                / 3e8
                * np.arange(stream_win)
            )
        elif self.projection == "cos":
            sv = np.exp(
                2j
                * np.pi
                * self.f_0
                * (1 - np.cos(theta_i))
                * self.antenna_spacing
                / 3e8
                * np.arange(stream_win)
            )
        else:
            raise ValueError(f"Unsupported projection: {self.projection}")
        return sv.flatten()

    def steering_vector_ToF(self, tau_i, freq_win=None, freq_hop=None):
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop

        row_size = freq_win // freq_hop
        sub_idx = np.arange(0, row_size) * freq_hop
        sv = np.exp(-2j * np.pi * (self.f_0 + self.delta_f * sub_idx) * tau_i)
        return sv.flatten()

    def steering_vector_AoA_ToF(self, theta_i, tau_j, stream_win=None, freq_win=None, freq_hop=None):
        if stream_win is None:
            stream_win = self.stream_win
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop

        phase_azi = self.steering_vector_AoA(theta_i, stream_win)
        phase_tof = self.steering_vector_ToF(tau_j, freq_win, freq_hop)
        steering_vector = np.kron(phase_azi, phase_tof)
        return steering_vector.flatten()

    def steering_vector_ToF_Dop(self, tau, fd):
        """
        ToF-Doppler steering vector.
        phase_tof = +2*pi*Δf*m*tau
        phase_dop = -2*pi*fd*t/fs
        sv = exp(-j * (phase_tof + phase_dop))

        注意 flatten order = (subcarrier, time).reshape(-1)。
        """
        m_idx = np.arange(self.freq_win)[:, None]
        t_idx = (np.arange(self.dop_win) / self.fs)[None, :]
        phase_tof = 2 * np.pi * self.delta_f * m_idx * tau
        phase_dop = -2 * np.pi * fd * t_idx
        sv = np.exp(-1j * (phase_tof + phase_dop))
        return sv.reshape(-1) / np.sqrt(self.freq_win * self.dop_win)

    def steering_vector_AoA_Dop(self, theta_i, fd, stream_win=None, dop_win=None):
        if stream_win is None:
            stream_win = self.stream_win
        if dop_win is None:
            dop_win = self.dop_win

        exp_phi = self.steering_vector_AoA(theta_i, stream_win)
        t_idx = np.arange(dop_win) / self.fs
        exp_dop = np.exp(1j * 2 * np.pi * fd * t_idx)
        steering_vector = np.kron(exp_phi, exp_dop)
        return steering_vector.flatten() / np.sqrt(stream_win * dop_win)


class Azi_ToF:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.fs = args.fs
        self.avg_frames = args.avg_frames
        self.num_Rx = args.num_Rx
        self.num_subcarriers = args.num_subcarriers
        self.stream_win = args.stream_win
        self.stream_sample_range = min(args.stream_sample_range, args.num_Rx)
        self.freq_win = args.freq_win
        self.freq_hop = args.freq_hop
        self.freq_sample_range = min(args.freq_sample_range, args.num_subcarriers)
        self.Sdim = args.Sdim
        self.tau = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.theta = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)

        if self.stream_win <= 0:
            raise ValueError(f"stream_win must be positive, got {self.stream_win}")
        if self.stream_sample_range <= 0:
            raise ValueError(
                f"stream_sample_range must be positive, got {self.stream_sample_range}"
            )
        if self.stream_win > self.stream_sample_range:
            raise ValueError(
                f"stream_win={self.stream_win} cannot exceed "
                f"stream_sample_range={self.stream_sample_range}"
            )

        if self.freq_win <= 0:
            raise ValueError(f"freq_win must be positive, got {self.freq_win}")
        if self.freq_hop <= 0:
            raise ValueError(f"freq_hop must be positive, got {self.freq_hop}")
        if self.freq_sample_range <= 0:
            raise ValueError(
                f"freq_sample_range must be positive, got {self.freq_sample_range}"
            )

        self.block_size = self.freq_win // self.freq_hop
        if self.block_size <= 0:
            raise ValueError(
                f"freq_win // freq_hop must be positive, got {self.block_size}"
            )
        if self.block_size > self.freq_sample_range:
            raise ValueError(
                f"block_size={self.block_size} cannot exceed "
                f"freq_sample_range={self.freq_sample_range}"
            )

        self.ncol = (
            self.stream_win
            if (self.stream_sample_range % 2 == 1)
            else (self.stream_win - 1)
        )
        if self.ncol <= 0:
            raise ValueError(f"Invalid ncol={self.ncol}")
        if self.stream_win + self.ncol - 1 > self.stream_sample_range:
            raise ValueError(
                "stream_sample_range is too small for current stream_win: "
                f"stream_win={self.stream_win}, ncol={self.ncol}, "
                f"stream_sample_range={self.stream_sample_range}"
            )

    def gen_spectrum(self, CSI, frame_idx, x_axis="azi", y_axis="tof"):
        x = self.cal_smoothed_csi(frame_idx, CSI)
        Rxx = self.cal_smoothed_cov(x)
        tau_x, theta_x, P_music_x = self.cal_spectrum(Rxx)
        Plot.plot_spectrum(
            frame_idx,
            tau_x,
            theta_x,
            P_music_x,
            self.args,
            title="Azi-ToF",
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def smooth_csi(self, csi):
        """
        Build Toeplitz-like smoothed CSI for AoA–ToF. (Use Tx=0)
        """
        H_list = []
        for r in range(self.stream_sample_range):
            H = np.zeros((self.block_size, self.block_size), dtype=complex)
            for i in range(self.block_size):
                start = i * self.freq_hop
                end = min(start + self.block_size, self.freq_sample_range)
                start = end - self.block_size
                H[i, :] = csi[0, r, start:end]
            H_list.append(H)

        smoothed_csi = np.block([
            [H_list[i + j] for j in range(self.ncol)]
            for i in range(self.stream_win)
        ])
        return smoothed_csi

    def cal_smoothed_csi(self, frame_idx, CSI):
            smoothed_CSIs = []
            avg_frames = self.avg_frames
            for i in range(avg_frames):
                smoothed_csi = self.smooth_csi(CSI[frame_idx -avg_frames//2 + i])
                smoothed_CSIs.append(smoothed_csi)

            return np.array(smoothed_CSIs)

    def cal_smoothed_cov(self, smoothed_csi):
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

    def cal_spectrum(self, smoothed_cov):
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
        Sdim = resolve_Sdim(self.args, eig_val)
        #print(f"Selected signal subspace dimension Sdim={Sdim}.")
        #print(f"top 20 eigen vals{eig_val[:20]}")
        N_dim = eig_val.shape[0] - Sdim
        E_n = eig_vec[:, -N_dim:]
        #P_n = E_n @ E_n.conj().T

        # theta candidate
        theta = self.theta
        # tau candidate
        tau = self.tau

        # steering_vector length:
        sv_len = self.stream_win * (self.freq_win // self.freq_hop)
        # calculate all steering vectors at once:
        Steering_Vectors = np.zeros((len(theta), len(tau), sv_len), dtype=complex)
        for i in range(len(theta)):
            for j in range(len(tau)):
                sv = self.steering_vector.steering_vector_AoA_ToF(
                    theta[i],
                    tau[j],
                    self.stream_win,
                    self.freq_win,
                    self.freq_hop,
                )
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
        P_music = 10 * np.log10(1.0 / PP + 1e-12)

        return tau, theta, P_music



class ToF_Doppler:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "Sdim", None)
        self.num_Rx = getattr(args, "num_Rx", None)
        self.num_subcarriers = int(args.num_subcarriers)
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(getattr(args, "freq_hop", 1)))
        self.freq_space = max(1, int(getattr(args, "freq_space", 1)))
        self.freq_sample_range = int(
            min(getattr(args, "freq_sample_range", args.num_subcarriers), args.num_subcarriers)
        )
        self.dop_win = int(args.time_win)
        self.time_sample_range = int(
            max(getattr(args, "time_sample_range", self.dop_win), self.dop_win)
        )
        self.time_win = self.time_sample_range
        self.fs = float(args.fs)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.delta_f = float(getattr(args, "delta_f", args.BW / args.num_subcarriers))

        self.epsilon = float(getattr(args, "tof_dop_epsilon", getattr(args, "epsilon", 1e-5)))
        self.tau_chunk = int(getattr(args, "tau_chunk", 12))
        self.snapshot_norm = bool(getattr(args, "tof_dop_snapshot_norm", True))
        self.dynamic_range_db = float(getattr(args, "dynamic_range_db", 35.0))
        self.floor_percentile = float(getattr(args, "floor_percentile", 5.0))
        self.last_meta = None

        if self.freq_sample_range <= 0:
            raise ValueError(
                f"freq_sample_range must be positive, got {self.freq_sample_range}"
            )
        # Align with 3Dspectrum_Exp:
        # CSI is sampled once outside this class by sample_subcarriers(..., freq_space).
        # Here we only compute how many sampled bins are available to use.
        self.sampled_freq_count = len(
            np.arange(0, self.freq_sample_range, self.freq_space)
        )
        if self.freq_win > self.sampled_freq_count:
            raise ValueError(
                f"freq_win={self.freq_win} cannot exceed "
                f"sampled_freq_count={self.sampled_freq_count}"
            )
        if self.dop_win <= 0:
            raise ValueError(f"time_win must be positive, got {self.dop_win}")
        if self.time_sample_range < self.dop_win:
            raise ValueError(
                f"time_sample_range={self.time_sample_range} cannot be smaller than "
                f"time_win={self.dop_win}"
            )

    def sample_CSI_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end

    def Rxx_smooth(self, CSI, frame_idx):
        T_dop = self.dop_win
        L_subc = self.freq_win
        stride = self.freq_hop

        csi_segment, start, end = self.sample_CSI_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :, :self.sampled_freq_count]
        context_len, num_tx, num_rx, K_subc = csi_segment.shape

        if context_len < T_dop or K_subc < L_subc:
            print(f"Warning: Not enough samples for ToF-Doppler smoothing at {frame_idx}")
            return None

        num_time_slides = context_len - T_dop + 1
        num_freq_slides = ((K_subc - L_subc) // stride) + 1

        # Snapshots = Tx * Rx * 子載波滑動數 * 時間滑動數
        total_snapshots = num_tx * num_rx * num_freq_slides * num_time_slides
        sv_len = T_dop * L_subc
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)

        idx = 0
        for tx in range(num_tx):
            for rx in range(num_rx):
                for i in range(num_freq_slides):
                    start_subc = i * stride
                    for t in range(num_time_slides):
                        block = csi_segment[t:t + T_dop, tx, rx, start_subc:start_subc + L_subc]

                        # block: (time, subcarrier) -> (subcarrier, time) -> flatten
                        v = block.T.reshape(-1)
                        if self.snapshot_norm:
                            v = v / (np.linalg.norm(v) + 1e-12)
                        X[:, idx] = v
                        idx += 1

        Rxx = (X @ X.conj().T) / max(idx, 1)
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        self.last_meta = {
            "context": (start, end),
            "num_tx": num_tx,
            "num_rx": num_rx,
            "num_subc_slides": num_freq_slides,
            "num_time_slides": num_time_slides,
            "num_snapshots": idx,
            "vec_len": sv_len,
        }
        print(
            f"ToF-Dop Rxx: {Rxx.shape}, snapshots={idx}, "
            f"context={start}:{end}, subc_slides={num_freq_slides}, time_slides={num_time_slides}"
        )
        return Rxx

    def steering_matrix_chunk(self, tau_chunk):
        A = np.empty(
            (len(tau_chunk) * len(self.fd_grid), self.freq_win * self.dop_win),
            dtype=np.complex128,
        )
        row = 0
        for tau in tau_chunk:
            for fd in self.fd_grid:
                A[row] = self.steering_vector.steering_vector_ToF_Dop(tau, fd)
                row += 1
        return A

    def cal_spectrum(self, Rxx):
        print(f"ToF-Doppler Covariance Matrix shape = {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        # Signal subspace projection, same convention as XMUSIC_ToF_Dop/Guan.
        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="ToF-Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_s = eig_vec[:, :Sdim]

        tau = self.tau_grid
        fd = self.fd_grid

        PP = np.empty((len(tau), len(fd)), dtype=float)
        for start in tqdm(range(0, len(tau), self.tau_chunk), desc="Calculating ToF-Doppler Spectrum"):
            end = min(start + self.tau_chunk, len(tau))
            tau_chunk = tau[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)

            # P = 1 / (1 - a^H Es Es^H a + epsilon)
            A = SV_chunk.conj() @ E_s
            aEEa = np.real(np.sum(A * np.conj(A), axis=1))
            aEEa = np.clip(aEEa, -1.0, 1.0)
            PP[start:end] = (1.0 / (1.0 - aEEa + self.epsilon)).reshape(len(tau_chunk), len(fd))

        return tau, fd, PP

    def gen_spectrum(self, CSI, frame_idx, x_axis="doppler", y_axis="tof"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return
        tau, fd, P_tof_dop = self.cal_spectrum(Rxx)
        Plot.plot_spectrum(
            frame_idx,
            tau,
            fd,
            P_tof_dop.T,
            self.args,
            title="ToF-Doppler",
            x_axis=x_axis,
            y_axis=y_axis,
        )


class Azi_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "Sdim", None)
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_sample_range = int(
            min(getattr(args, "freq_sample_range", args.num_subcarriers), args.num_subcarriers)
        )
        self.dop_win = int(args.time_win)
        self.time_hop = max(1, int(getattr(args, "time_hop", 1)))
        self.time_sample_range = int(
            max(getattr(args, "time_sample_range", self.dop_win), self.dop_win)
        )
        self.fs = float(args.fs)
        self.theta = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
        self.fd_grid = np.arange(
            args.doppler_min,
            args.doppler_max + 0.5 * args.doppler_step,
            args.doppler_step,
        )
        self.snapshot_norm = bool(getattr(args, "azi_dop_snapshot_norm", True))
        self.epsilon = float(getattr(args, "azi_dop_epsilon", 1e-12))
        self.last_meta = None

        if self.stream_sample_range <= 0:
            raise ValueError(
                f"stream_sample_range must be positive, got {self.stream_sample_range}"
            )
        if self.freq_sample_range <= 0:
            raise ValueError(
                f"freq_sample_range must be positive, got {self.freq_sample_range}"
            )
        if self.stream_win <= 0:
            raise ValueError(f"stream_win must be positive, got {self.stream_win}")
        if self.stream_win > self.stream_sample_range:
            raise ValueError(
                f"stream_win={self.stream_win} cannot exceed "
                f"stream_sample_range={self.stream_sample_range}"
            )
        if self.dop_win <= 0:
            raise ValueError(f"time_win must be positive, got {self.dop_win}")
        if self.time_sample_range < self.dop_win:
            raise ValueError(
                f"time_sample_range={self.time_sample_range} cannot be smaller than "
                f"time_win={self.dop_win}"
            )

    def sample_CSI_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end

    def Rxx_smooth(self, CSI, frame_idx):
        csi_segment, start, end = self.sample_CSI_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.freq_sample_range]
        context_len, num_tx, num_rx, num_subc = csi_segment.shape

        if context_len < self.dop_win:
            print(f"Warning: Not enough samples for Azi-Doppler smoothing at {frame_idx}")
            return None

        csi_segment = np.asarray(csi_segment, dtype=np.complex128)
        csi_segment = csi_segment - np.mean(csi_segment, axis=0, keepdims=True)
        taper = np.hanning(self.dop_win)[None, :]

        time_starts = np.arange(0, context_len - self.dop_win + 1, self.time_hop)
        stream_starts = np.arange(0, num_rx - self.stream_win + 1)
        total_snapshots = num_tx * num_subc * len(stream_starts) * len(time_starts)
        sv_len = self.stream_win * self.dop_win
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)

        idx = 0
        for tx in range(num_tx):
            for subc in range(num_subc):
                for stream_start in stream_starts:
                    stream_end = stream_start + self.stream_win
                    for time_start in time_starts:
                        time_end = time_start + self.dop_win
                        block = csi_segment[
                            time_start:time_end,
                            tx,
                            stream_start:stream_end,
                            subc,
                        ]
                        v = (block.T * taper).reshape(-1)
                        if self.snapshot_norm:
                            v = v / (np.linalg.norm(v) + 1e-12)
                        X[:, idx] = v
                        idx += 1

        Rxx = (X @ X.conj().T) / max(idx, 1)
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        self.last_meta = {
            "context": (start, end),
            "num_tx": num_tx,
            "num_rx": num_rx,
            "num_subc": num_subc,
            "num_stream_slides": len(stream_starts),
            "num_time_slides": len(time_starts),
            "num_snapshots": idx,
            "vec_len": sv_len,
        }
        print(
            f"Azi-Dop Rxx: {Rxx.shape}, snapshots={idx}, "
            f"context={start}:{end}, stream_slides={len(stream_starts)}, "
            f"time_slides={len(time_starts)}"
        )
        return Rxx

    def steering_matrix(self, theta, fd):
        sv_len = self.stream_win * self.dop_win
        Steering_Vectors = np.zeros((len(theta), len(fd), sv_len), dtype=complex)
        for i in range(len(theta)):
            for j in range(len(fd)):
                sv = self.steering_vector.steering_vector_AoA_Dop(
                    theta[i],
                    fd[j],
                    self.stream_win,
                    self.dop_win,
                )
                Steering_Vectors[i, j, :] = sv
        return Steering_Vectors

    def cal_spectrum(self, Rxx):
        print(f"Azi-Doppler Covariance Matrix shape = {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_n = eig_vec[:, Sdim:]
        if E_n.size == 0:
            E_n = eig_vec[:, -1:]

        theta = self.theta
        fd = self.fd_grid

        sv_len = self.stream_win * self.dop_win
        Steering_Vectors = self.steering_matrix(theta, fd)
        SV_flat = Steering_Vectors.reshape(len(theta) * len(fd), sv_len)
        A = SV_flat.conj() @ E_n
        PP_flat = np.sum(np.abs(A) ** 2, axis=1)
        P_music = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon))
        P_music = P_music.reshape(len(theta), len(fd))

        return theta, fd, P_music

    def gen_spectrum(self, CSI, frame_idx, x_axis="azi", y_axis="doppler"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return
        theta, fd, P_azi_dop = self.cal_spectrum(Rxx)
        Plot.plot_heatmap(
            frame_idx,
            theta,
            fd,
            P_azi_dop.T,
            self.args,
            title="Azi-Doppler",
            x_axis=x_axis,
            y_axis=y_axis,
            file_suffix="azi_doppler",
        )


Azi_Doppler = Azi_Dop
