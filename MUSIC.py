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

def resolve_Sdim(
    args,
    eig_val,
    label="MUSIC",
    sdim_attr="Sdim",
    energy_ratio_attr="Sdim_energy_ratio",
):
    M = len(eig_val)

    if getattr(args, sdim_attr, None) is not None:
        Sdim = int(getattr(args, sdim_attr))
    else:
        energy_ratio = getattr(
            args,
            energy_ratio_attr,
            getattr(args, "Sdim_energy_ratio", 0.88),
        )
        if energy_ratio is None:
            energy_ratio = getattr(args, "Sdim_energy_ratio", 0.88)
        Sdim = estimate_Sdim(
            eig_val,
            energy_ratio=energy_ratio,
            min_dim=getattr(args, "Sdim_min", 1),
            max_dim=M - 1,
        )
        print(f"{label}: Estimated Sdim={Sdim} (energy_ratio={energy_ratio:.2f})")

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
        self.time_win = args.time_win
        self.time_hop = args.time_hop
        self.time_sample_range = args.time_sample_range
        self.fs = args.fs
        self.delta_f = float(getattr(args, "delta_f", args.BW / args.num_scarriers))

    def steering_vector_AoA(self, theta_i, stream_win=None):
        if stream_win is None:
            stream_win = self.stream_win

        theta_i = np.deg2rad(theta_i)
        if self.projection == "sin":
            sv = np.exp(-2j * np.pi* self.f_0* np.sin(theta_i)* self.antenna_spacing/ 3e8* np.arange(stream_win))
        elif self.projection == "cos":
            sv = np.exp(-2j* np.pi* self.f_0* (1-np.cos(theta_i))* self.antenna_spacing/ 3e8* np.arange(stream_win))
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

    def steering_vector_Dop(self, fd, dop_win=None):
        if dop_win is None:
            dop_win = self.time_win
        t_idx = np.arange(dop_win) / self.fs
        sv = np.exp(1j * 2 * np.pi * fd * t_idx)
        return sv.flatten()

    def steering_vector_AoA_ToF(self, theta_i, tau_j, stream_win=None, freq_win=None, freq_hop=None):
        if stream_win is None:
            stream_win = self.stream_win
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop

        sv_azi = self.steering_vector_AoA(theta_i, stream_win)
        sv_tof = self.steering_vector_ToF(tau_j, freq_win, freq_hop)
        steering_vector = np.kron(sv_azi, sv_tof)
        return steering_vector.flatten()

    def steering_vector_ToF_Dop(
        self,
        tau,
        fd,
        freq_win=None,
        freq_hop=None,
        time_win=None,
    ):
        """
        ToF-Doppler steering vector.
        phase_tof = +2*pi*Δf*m*tau
        phase_dop = +2*pi*fd*t/fs
        sv = exp(-j*phase_tof + j*phase_dop)

        注意 flatten order = (subcarrier, time).reshape(-1)。
        """
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop
        if time_win is None:
            time_win = self.time_win

        freq_win = int(freq_win)
        freq_hop = int(freq_hop)
        time_win = int(time_win)
        if freq_win <= 0 or freq_hop <= 0 or time_win <= 0:
            raise ValueError(
                "freq_win, freq_hop, and time_win must all be positive"
            )

        # Match Azi_ToF: freq_win is the frequency-aperture span and
        # freq_hop is the spacing between virtual frequency sensors.
        freq_row_count = freq_win // freq_hop
        if freq_row_count <= 0:
            raise ValueError(
                f"freq_win // freq_hop must be positive, got {freq_row_count}"
            )
        m_idx = (np.arange(freq_row_count) * freq_hop)[:, None]
        t_idx = (np.arange(time_win) / self.fs)[None, :]
        phase_tof = 2 * np.pi * self.delta_f * m_idx * tau
        phase_dop = 2 * np.pi * fd * t_idx
        sv = np.exp(-1j * (phase_tof - phase_dop))

        return sv.reshape(-1) / np.sqrt(freq_row_count * time_win)

class Azi_ToF:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.fs = args.fs
        self.avg_frames = args.avg_frames
        self.num_Rx = args.num_Rx
        self.num_scarriers = args.num_scarriers
        self.stream_win = args.stream_win
        self.stream_sample_range = min(args.stream_sample_range, args.num_Rx)
        self.freq_win = args.freq_win
        self.freq_hop = args.freq_hop
        self.freq_sample_range = min(args.freq_sample_range, args.num_scarriers)
        self.Sdim = args.Sdim
        self.last_Sdim = None
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

        # Frequency-smoothing rows are the virtual sensors used by the ToF
        # steering vector: 0, freq_hop, ..., (block_size - 1) * freq_hop.
        # The remaining contiguous bins form independent snapshot columns.
        # Keeping these dimensions separate avoids clamping a row to an
        # incorrect subcarrier index when the old square block would overrun.
        self.freq_snapshot_count = (
            self.freq_sample_range
            - (self.block_size - 1) * self.freq_hop
        )
        if self.freq_snapshot_count <= 0:
            required = (self.block_size - 1) * self.freq_hop + 1
            raise ValueError(
                "Not enough subcarriers for the requested frequency aperture: "
                f"need at least {required}, got {self.freq_sample_range} "
                f"(block_size={self.block_size}, freq_hop={self.freq_hop})"
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
            theta_x,
            tau_x,
            P_music_x,
            self.args,
            title="Azi-ToF",
            x_axis=x_axis,
            y_axis=y_axis,
            sdim=self.last_Sdim,
            spectrum_axes=("azi", "tof"),
        )

    def smooth_csi(self, csi):
        """
        Build steering-consistent frequency/spatial smoothing for AoA–ToF.

        Each per-RX matrix has shape
        ``(block_size, freq_snapshot_count)``. Row ``i`` starts at
        subcarrier ``i * freq_hop``, exactly matching the ToF steering-vector
        indices. Columns are contiguous frequency snapshots. (Use Tx=0.)
        """
        csi = np.asarray(csi)
        if csi.ndim != 3:
            raise ValueError(
                "Azi_ToF.smooth_csi expects shape (tx, rx, subcarrier), "
                f"got {csi.shape}"
            )
        if csi.shape[0] < 1:
            raise ValueError("Azi_ToF.smooth_csi requires at least one Tx")
        if csi.shape[1] < self.stream_sample_range:
            raise ValueError(
                f"CSI has {csi.shape[1]} Rx channels, but "
                f"stream_sample_range={self.stream_sample_range}"
            )
        if csi.shape[2] < self.freq_sample_range:
            raise ValueError(
                f"CSI has {csi.shape[2]} subcarriers, but "
                f"freq_sample_range={self.freq_sample_range}"
            )

        H_list = []
        for r in range(self.stream_sample_range):
            H = np.empty(
                (self.block_size, self.freq_snapshot_count),
                dtype=complex,
            )
            for i in range(self.block_size):
                start = i * self.freq_hop
                end = start + self.freq_snapshot_count
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
        # Standard sample covariance: every smoothing column from every frame
        # is one snapshot and contributes with equal weight.
        smoothed_csi = np.asarray(smoothed_csi, dtype=complex)
        if smoothed_csi.ndim == 2:
            total_snapshots = smoothed_csi.shape[1]
            if total_snapshots <= 0:
                raise ValueError("Azi-ToF smoothing produced no snapshots")
            return (
                smoothed_csi @ smoothed_csi.conj().T
            ) / total_snapshots

        if smoothed_csi.ndim == 3:
            total_snapshots = smoothed_csi.shape[0] * smoothed_csi.shape[2]
            if total_snapshots <= 0:
                raise ValueError("Azi-ToF smoothing produced no snapshots")
            cov = np.zeros(
                (smoothed_csi.shape[1], smoothed_csi.shape[1]),
                dtype=complex,
            )
            for temp_x in smoothed_csi:
                cov += temp_x @ temp_x.conj().T
            return cov / total_snapshots

        raise ValueError(
            "smoothed_csi must have shape (row, snapshot) or "
            f"(frame, row, snapshot), got {smoothed_csi.shape}"
        )

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
        self.last_Sdim = Sdim
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
                sv = self.steering_vector.steering_vector_AoA_ToF(theta[i],tau[j],self.stream_win,self.freq_win,self.freq_hop,)
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

class ToF_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "Sdim", None)

        # Steering/smoothing aperture.
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, args.freq_hop)
        freq_space = max(1, args.freq_space)

        # Number of frequency samples available to the smoother. Preserve the
        # existing external-resampling convention without storing three fields.
        freq_limit = int(min(getattr(args, "freq_sample_range", args.num_scarriers), args.num_scarriers))
        self.num_freq_samples = len(np.arange(0, freq_limit, freq_space))

        # Doppler aperture and input context.
        self.time_win = int(args.time_win)
        self.time_sample_range = int(max(getattr(args, "time_sample_range", self.time_win), self.time_win))

        # Search grid and estimator controls.
        self.tau_chunk = 10
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, "tof_dop_epsilon", getattr(args, "epsilon", 1e-5)))

        # Configuration validation.
        freq_win_points = self.freq_win // self.freq_hop
        if freq_win_points <= 0:
            raise ValueError(
                f"freq_win // freq_hop must be positive, got {freq_win_points}"
            )
        if self.num_freq_samples <= 0:
            raise ValueError("No frequency samples are available")
        freq_offsets = np.arange(freq_win_points) * self.freq_hop
        freq_aperture_span = int(freq_offsets[-1]) + 1
        if freq_aperture_span > self.num_freq_samples:
            raise ValueError(
                "Frequency aperture cannot exceed the available subcarriers: "
                f"need {freq_aperture_span}, got {self.num_freq_samples}"
            )
        if self.time_win <= 0:
            raise ValueError(f"time_win must be positive, got {self.time_win}")
        if self.time_sample_range < self.time_win:
            raise ValueError(
                f"time_sample_range={self.time_sample_range} cannot be smaller than "
                f"time_win={self.time_win}"
            )

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames) 
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end # (context_len, num_tx, num_rx, num_sc)

    def Rxx_smooth(self, CSI, frame_idx):
        time_win = self.time_win
        freq_win = self.freq_win # if 48
        freq_hop = self.freq_hop # if 3
        freq_win_points = freq_win // freq_hop # 48 // 3 = 16 sample points per window
        freq_win = np.arange(freq_win_points) * freq_hop # new freq_win: [0, 3, 6, ..., 45]

        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :, :self.num_freq_samples]
        context_len, num_tx, num_rx, num_sc = csi_segment.shape

        
        # Shift the snapshot start by one subcarrier. freq_hop controls the
        # spacing inside the frequency steering aperture, not this slide.
        num_time_slides = context_len - time_win + 1
        num_freq_slides = num_sc - int(freq_win[-1])

        # Snapshots = Tx * Rx * num_time_slides * num_freq_slides
        total_snapshots = num_tx * num_rx * num_freq_slides * num_time_slides
        sv_len = time_win * freq_win_points

        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)

        idx = 0
        for tx in range(num_tx): # 2 Tx
            for rx in range(num_rx): # 8 Rx
                for i in range(num_freq_slides): # num_freq_slides
                    for t in range(num_time_slides):
                        block = csi_segment[t:(t + time_win), tx, rx, i + freq_win]

                        v = block.T.reshape(-1)
                        X[:, idx] = v
                        # X= [v_1, v_2, ...v_total_snapshots]
                        idx += 1 # final idx = total_snapshots

        Rxx = (X @ X.conj().T) / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0 # symmetrize # 數值穩定處理
        return Rxx

    def steering_matrix_chunk(self, tau_chunk=10):
        # 表示每次取 tau_chunk 個 ToF 候選值進行計算
        # 對這一批 tau 值，搭配全部 Doppler grid
        # 產生所有 ToF–Doppler steering vectors
        A = np.empty((len(tau_chunk) * len(self.fd_grid), (self.freq_win // self.freq_hop) * self.time_win),dtype=np.complex128)
        # A.shape = (tau_chunk * fd_grid, sv_len)
        row = 0
        for tau in tau_chunk:
            for fd in self.fd_grid:
                A[row] = self.steering_vector.steering_vector_ToF_Dop(tau,fd,self.freq_win,self.freq_hop,self.time_win)
                row += 1
        return A

    def cal_spectrum(self, Rxx):
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        # Signal subspace projection, same convention as XMUSIC_ToF_Dop/Guan.
        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args,eig_val)
            self.Sdim = Sdim
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_s = eig_vec[:, :Sdim]

        tau_grid = self.tau_grid
        fd_grid = self.fd_grid

        PP = np.empty((len(tau_grid), len(fd_grid)), dtype=float)

        for start in tqdm(range(0, len(tau_grid), self.tau_chunk), desc="Calculating ToF-Doppler Spectrum"):
            end = min(start + self.tau_chunk, len(tau_grid))
            tau_chunk = tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)

            A = SV_chunk.conj() @ E_s
            aEEa = np.real(np.sum(A * np.conj(A), axis=1))
            aEEa = np.clip(aEEa, -1.0, 1.0)
            PP[start:end] = (1.0 / (1.0 - aEEa + self.epsilon)).reshape(len(tau_chunk), len(fd_grid))

        return tau_grid, fd_grid, PP

    def gen_spectrum(self, CSI, frame_idx, x_axis="doppler", y_axis="tof"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop = self.cal_spectrum(Rxx)
        P_tof_dop_db = 10.0 * np.log10(np.maximum(P_tof_dop, 1e-12))
        Plot.plot_spectrum(
            frame_idx,
            tau_grid,
            fd_grid,
            P_tof_dop_db,
            self.args,
            title="ToF-Doppler",
            x_axis=x_axis,
            y_axis=y_axis,
            sdim=self.Sdim,
            spectrum_axes=("tof", "doppler"),
        )

class Azi_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "Sdim", None)
        self.last_Sdim = None
        self.last_meta = None

        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_sample_range = int(min(getattr(args, "freq_sample_range", args.num_scarriers), args.num_scarriers))
        self.input_time_win = int(args.time_sample_range)
        self.time_win = int(args.time_win)
        self.time_hop = max(1, int(getattr(args, "time_hop", 1)))
        self.freq_space = max(1, int(getattr(args, "freq_space", 1)))

        self.theta = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)

        self.epsilon = float(getattr(args, "azi_dop_epsilon", 1e-12))

        if not 0 < self.stream_win <= self.stream_sample_range:
            raise ValueError(
                "Require 0 < stream_win <= stream_sample_range, got "
                f"{self.stream_win} and {self.stream_sample_range}"
            )
        if self.freq_sample_range <= 0:
            raise ValueError(f"freq_sample_range must be positive, got {self.freq_sample_range}")
        if not 0 < self.time_win < self.input_time_win:
            raise ValueError(
                "Require 0 < time_win < input_time_win, got "
                f"{self.time_win} and {self.input_time_win}"
            )

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.input_time_win, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        # Center an even-length window as [n - N//2, n + N//2 - 1].
        # Near either boundary, shift the complete window instead of shrinking it.
        start = int(np.clip(frame_idx - context_len // 2,0,total_frames - context_len,))
        end = start + context_len
        return CSI[start:end], start, end

    def Rxx_smooth(self, CSI, frame_idx):
        if CSI.ndim != 4:
            raise ValueError("Azi_DopX expects CSI shape (frame, tx, rx, subcarrier), "f"got {CSI.shape}")

        time_win = self.time_win
        stream_win = self.stream_win

        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:,:,:self.stream_sample_range,:self.freq_sample_range]
        csi_segment = csi_segment[:, :, :, ::self.freq_space]
        csi_segment = np.asarray(csi_segment, dtype=np.complex128)
        context_len, num_tx, num_rx, num_sc = csi_segment.shape
        if context_len < self.input_time_win:
            print(f"Warning: Not enough samples for Azi-Doppler smoothing at {frame_idx}")
            return None

        time_starts = np.arange(0, context_len - time_win, self.time_hop)
        stream_starts = np.arange(0, num_rx - stream_win + 1)
        num_time_slides = len(time_starts)
        num_stream_slides = len(stream_starts)

        # Snapshots = Tx * subcarrier * spatial slides * time slides
        total_snapshots = num_tx * num_sc * num_stream_slides * num_time_slides
        sv_len = stream_win * time_win
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)

        idx = 0
        for tx in range(num_tx):
            for subc in range(num_sc):
                for stream_start in stream_starts:
                    for time_start in time_starts:
                        block = csi_segment[time_start:(time_start + time_win),tx,stream_start:(stream_start + stream_win),subc,]

                        v = block.T.reshape(-1)
                        X[:, idx] = v
                        idx += 1

        Rxx = (X @ X.conj().T) / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        return Rxx

    def steering_matrix(self, theta, fd):
        sv_len = self.stream_win * self.time_win
        A = np.empty(
            (len(theta) * len(fd), sv_len),
            dtype=np.complex128,
        )

        row = 0
        for theta_i in theta:
            sv_aoa = self.steering_vector.steering_vector_AoA(
                theta_i,
                self.stream_win,
            )
            for fd_i in fd:
                sv_dop = self.steering_vector.steering_vector_Dop(
                    fd_i,
                    self.time_win,
                )
                A[row] = np.kron(sv_aoa, sv_dop) / np.sqrt(sv_len)
                row += 1

        return A

    def cal_spectrum(self, Rxx):
        print(f"Azi-Doppler Covariance Matrix shape = {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        self.last_Sdim = Sdim
            
        E_n = eig_vec[:, Sdim:]
        if E_n.size == 0:
            E_n = eig_vec[:, -1:]

        theta = self.theta
        fd = self.fd_grid

        # Each row corresponds to one (theta, fd) steering vector.
        Steering_Vectors = self.steering_matrix(theta, fd)
        A = Steering_Vectors.conj() @ E_n
        
        PP_flat = np.sum(np.abs(A) ** 2, axis=1)
        P_music = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon))
        P_music = P_music.reshape(len(theta), len(fd))

        return theta, fd, P_music

    def gen_spectrum(self, CSI, frame_idx, x_axis="azi", y_axis="doppler"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return
        
        theta_grid, fd_grid, P_azi_dop = self.cal_spectrum(Rxx)
        
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            fd_grid,
            P_azi_dop,
            self.args,
            title="Azi-Doppler",
            x_axis=x_axis,
            y_axis=y_axis,
            sdim=self.last_Sdim,
            spectrum_axes=("azi", "doppler"),
        )


class Azi_ToF_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "Sdim", None)
        self.last_Sdim = None
        self.last_meta = None
        self.last_cube = None
        self.last_axes = None

        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(getattr(args, "freq_hop", 1)))
        self.freq_sample_range = int(min(getattr(args, "freq_sample_range", args.num_scarriers), args.num_scarriers))
        self.time_win = int(args.time_win)
        self.time_sample_range = int(max(getattr(args, "time_sample_range", self.time_win), self.time_win))
        self.time_hop = max(1, int(getattr(args, "time_hop", 1)))

        self.theta_grid = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, "azi_tof_dop_epsilon", 1e-12))

        freq_win_points = self.freq_win // self.freq_hop
        if not 0 < self.stream_win <= self.stream_sample_range:
            raise ValueError(
                "Require 0 < stream_win <= stream_sample_range, got "
                f"{self.stream_win} and {self.stream_sample_range}"
            )
        if freq_win_points <= 0:
            raise ValueError(
                f"freq_win // freq_hop must be positive, got {freq_win_points}"
            )
        freq_offsets = np.arange(freq_win_points) * self.freq_hop
        freq_aperture_span = int(freq_offsets[-1]) + 1
        if freq_aperture_span > self.freq_sample_range:
            raise ValueError(
                "Frequency aperture cannot exceed the available subcarriers: "
                f"need {freq_aperture_span}, got {self.freq_sample_range}"
            )
        if not 0 < self.time_win <= self.time_sample_range:
            raise ValueError(
                "Require 0 < time_win <= time_sample_range, got "
                f"{self.time_win} and {self.time_sample_range}"
            )

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end

    def Rxx_smooth(self, CSI, frame_idx):
        if CSI.ndim != 4:
            raise ValueError(
                "Azi_ToF_Dop expects CSI shape (frame, tx, rx, subcarrier), "
                f"got {CSI.shape}"
            )

        stream_win = self.stream_win
        time_win = self.time_win
        freq_win_points = self.freq_win // self.freq_hop
        freq_offsets = np.arange(freq_win_points) * self.freq_hop

        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[
            :,
            :,
            :self.stream_sample_range,
            :self.freq_sample_range,
        ]
        csi_segment = np.asarray(csi_segment, dtype=np.complex128)
        context_len, num_tx, num_rx, num_sc = csi_segment.shape

        if context_len < time_win:
            print(f"Warning: Not enough samples for 3D smoothing at {frame_idx}")
            return None

        time_starts = np.arange(0, context_len - time_win + 1, self.time_hop)
        stream_starts = np.arange(0, num_rx - stream_win + 1)
        num_time_slides = len(time_starts)
        num_stream_slides = len(stream_starts)
        num_freq_slides = num_sc - int(freq_offsets[-1])

        # Snapshots = Tx * spatial slides * frequency slides * time slides
        total_snapshots = (
            num_tx
            * num_stream_slides
            * num_freq_slides
            * num_time_slides
        )
        sv_len = stream_win * freq_win_points * time_win
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)

        idx = 0
        for tx in range(num_tx):
            for stream_start in stream_starts:
                for freq_start in range(num_freq_slides):
                    subcarrier_indices = freq_start + freq_offsets
                    for time_start in time_starts:
                        block = csi_segment[
                            time_start:(time_start + time_win),
                            tx,
                            stream_start:(stream_start + stream_win),
                            :,
                        ][:, :, subcarrier_indices]

                        # (time, stream, frequency) -> (stream, frequency, time)
                        v = block.transpose(1, 2, 0).reshape(-1)
                        X[:, idx] = v
                        idx += 1

        assert idx == total_snapshots, (
            f"Azi-ToF-Dop snapshot count mismatch: idx={idx}, "
            f"expected={total_snapshots}"
        )
        Rxx = (X @ X.conj().T) / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0

        self.last_meta = {
            "context": (start, end),
            "num_tx": num_tx,
            "num_rx": num_rx,
            "num_sc": num_sc,
            "num_stream_slides": num_stream_slides,
            "num_freq_slides": num_freq_slides,
            "num_time_slides": num_time_slides,
            "num_snapshots": total_snapshots,
            "vec_len": sv_len,
        }
        print(
            f"Azi-ToF-Dop Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, stream_slides={num_stream_slides}, "
            f"subc_slides={num_freq_slides}, time_slides={num_time_slides}"
        )
        return Rxx

    def steering_matrix(self, theta_i, tau, fd):
        freq_win_points = self.freq_win // self.freq_hop
        sv_len = self.stream_win * freq_win_points * self.time_win
        A = np.empty(
            (len(tau) * len(fd), sv_len),
            dtype=np.complex128,
        )

        sv_aoa = self.steering_vector.steering_vector_AoA(
            theta_i,
            self.stream_win,
        )
        row = 0
        for tau_i in tau:
            for fd_i in fd:
                sv_tof_dop = self.steering_vector.steering_vector_ToF_Dop(
                    tau_i,
                    fd_i,
                    self.freq_win,
                    self.freq_hop,
                    self.time_win,
                )
                A[row] = np.kron(sv_aoa, sv_tof_dop) / np.sqrt(
                    self.stream_win
                )
                row += 1

        return A

    def cal_spectrum(self, Rxx):
        print(f"Azi-ToF-Doppler Covariance Matrix shape = {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-ToF-Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        self.last_Sdim = Sdim
            
        E_n = eig_vec[:, Sdim:]
        if E_n.size == 0:
            E_n = eig_vec[:, -1:]

        theta = self.theta_grid
        tau = self.tau_grid
        fd = self.fd_grid

        P_music = np.zeros((len(theta), len(tau), len(fd)), dtype=float)

        # Process one theta plane at a time to avoid allocating the full 3D
        # steering grid at once.
        for i, th in enumerate(tqdm(theta, desc="Calculating 3D MUSIC")):
            A = self.steering_matrix(th, tau, fd)
            proj = A.conj() @ E_n
            PP_flat = np.sum(np.abs(proj) ** 2, axis=1)
            P_music[i, :, :] = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon)).reshape(len(tau), len(fd))

        self.last_cube = P_music
        self.last_axes = {"azi": theta, "tof": tau, "doppler": fd, "tof_ns": tau * 1e9}

        return theta, tau, fd, P_music

    @staticmethod
    def _combine_axis(values, axis, method, axis_values=None):
        method = str(method).lower()
        if method == "sum":
            return np.sum(values, axis=axis)
        if method == "max":
            return np.max(values, axis=axis)
        if method == "mean":
            return np.mean(values, axis=axis)
        if method == "weighted":
            weights = np.abs(np.asarray(axis_values, dtype=float))
            weights = weights / (np.sum(weights) + 1e-12)
            shape = [1] * values.ndim
            shape[axis] = weights.size
            return np.sum(values * weights.reshape(shape), axis=axis)
        raise ValueError(f"Unsupported method: {method}")

    def gen_spectrum(self, CSI, frame_idx, method="sum"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return []
        theta, tau, fd, P_music_db = self.cal_spectrum(Rxx)

        # Projection must be performed in linear power, not in dB.
        P_music = 10.0 ** (P_music_db / 10.0)
        azi_tof_db = 10.0 * np.log10(
            self._combine_axis(P_music, axis=2, method=method, axis_values=fd)
            + 1e-12
        )
        tof_dop_db = 10.0 * np.log10(
            self._combine_axis(P_music, axis=0, method=method, axis_values=theta)
            + 1e-12
        )
        azi_dop_db = 10.0 * np.log10(
            self._combine_axis(P_music, axis=1, method=method, axis_values=tau)
            + 1e-12
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        panels = [
            {
                "title": "Azi-ToF",
                "axis0_values": theta,
                "axis1_values": tau,
                "heatmap_db": azi_tof_db,
                "spectrum_axes": ("azi", "tof"),
                "x_axis": "azi",
                "y_axis": "tof",
                "projected_axis": "doppler",
            },
            {
                "title": "ToF-Doppler",
                "axis0_values": tau,
                "axis1_values": fd,
                "heatmap_db": tof_dop_db,
                "spectrum_axes": ("tof", "doppler"),
                "x_axis": "doppler",
                "y_axis": "tof",
                "projected_axis": "azi",
            },
            {
                "title": "Azi-Doppler",
                "axis0_values": theta,
                "axis1_values": fd,
                "heatmap_db": azi_dop_db,
                "spectrum_axes": ("azi", "doppler"),
                "x_axis": "azi",
                "y_axis": "doppler",
                "projected_axis": "tof",
            },
        ]

        results = []
        for ax, panel in zip(axes, panels):
            Plot.plot_spectrum(
                frame_idx,
                panel["axis0_values"],
                panel["axis1_values"],
                panel["heatmap_db"],
                self.args,
                title=panel["title"],
                x_axis=panel["x_axis"],
                y_axis=panel["y_axis"],
                sdim=self.last_Sdim,
                spectrum_axes=panel["spectrum_axes"],
                ax=ax,
                save=False,
                show_colorbar=True,
            )
            results.append({
                "fig": fig,
                "ax": ax,
                "heatmap_db": panel["heatmap_db"],
                "x_axis": panel["x_axis"],
                "y_axis": panel["y_axis"],
                "projected_axis": panel["projected_axis"],
                "method": method,
            })

        if self.args.pics_dir is not None:
            os.makedirs(self.args.pics_dir, exist_ok=True)
            save_path = os.path.join(
                self.args.pics_dir,
                f"{frame_idx:04d}_azi_tof_dop_{method}.png",
            )
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"Saved: {save_path}")

        return results
