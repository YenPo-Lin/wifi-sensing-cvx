import numpy as np
import os
from tqdm import tqdm
import Plot
import matplotlib.pyplot as plt
from simulate_utils import cfar_2D

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
        #print(f"{label}: Estimated Sdim={Sdim} (energy_ratio={energy_ratio:.2f})")

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
        self.delta_f = float(getattr(args, "delta_f", args.BW / args.num_sc))

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
        self.Sdim = getattr(
            args,
            "Azi_ToF_Sdim",
            getattr(args, "Sdim", None),
        )

        # Steering/smoothing aperture.
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(
            min(args.stream_sample_range, args.num_Rx)
        )
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(args.freq_hop))
        self.freq_space = max(1, int(args.freq_space))
        self.avg_frames = max(1, int(args.avg_frames))

        # Search grid and estimator controls.
        self.tau_chunk = max(
            1,
            int(getattr(args, "azi_tof_tau_chunk", 10)),
        )
        self.theta_grid = np.arange(
            args.theta_min,
            args.theta_max + 0.5 * args.theta_step,
            args.theta_step,
        )
        self.tau_grid = np.arange(
            args.tau_min,
            args.tau_max,
            args.tau_step,
        )
        self.epsilon = float(
            getattr(
                args,
                "azi_tof_epsilon",
                getattr(args, "epsilon", 1e-12),
            )
        )

        # Number of frequency samples available to the smoother. Preserve the
        # existing external-resampling convention without storing three fields.
        freq_limit = int(
            min(
                getattr(args, "freq_sample_range", args.num_sc),
                args.num_sc,
            )
        )
        self.num_freq_samples = len(
            np.arange(0, freq_limit, self.freq_space)
        )

        # Configuration validation.
        self.freq_win_points = self.freq_win // self.freq_hop
        if self.freq_win_points <= 0:
            raise ValueError(
                "freq_win // freq_hop must be positive, got "
                f"{self.freq_win_points}"
            )
        if self.num_freq_samples <= 0:
            raise ValueError("No frequency samples are available")
        if self.stream_win <= 0:
            raise ValueError(
                f"stream_win must be positive, got {self.stream_win}"
            )
        if self.stream_win > self.stream_sample_range:
            raise ValueError(
                f"stream_win={self.stream_win} cannot exceed "
                f"stream_sample_range={self.stream_sample_range}"
            )

        self.freq_offsets = (
            np.arange(self.freq_win_points) * self.freq_hop
        )
        self.freq_aperture_span = int(self.freq_offsets[-1]) + 1
        if self.freq_aperture_span > self.num_freq_samples:
            raise ValueError(
                "Frequency aperture cannot exceed the available subcarriers: "
                f"need {self.freq_aperture_span}, "
                f"got {self.num_freq_samples}"
            )

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.avg_frames, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2,0,total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end

    def Rxx_smooth(self, CSI, frame_idx):
        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.num_freq_samples]
        context_len, num_tx, num_rx, num_sc = csi_segment.shape

        if num_tx < 1:
            raise ValueError("Azi_ToFX requires at least one Tx")

        # The spatial and frequency snapshot starts both slide by one sample.
        # freq_hop controls spacing inside the frequency aperture only.
        num_stream_slides = num_rx - self.stream_win + 1
        num_freq_slides = num_sc - int(self.freq_offsets[-1])

        sv_len = self.stream_win * self.freq_win_points
        total_snapshots = (context_len* num_tx* num_stream_slides* num_freq_slides)

        if total_snapshots <= 0:
            raise ValueError("Azi_ToFX smoothing produced no snapshots")

        X = np.empty((sv_len, total_snapshots),dtype=np.complex128,)

        idx = 0
        for t in range(context_len):
            for tx in range(num_tx):
                for stream_start in range(num_stream_slides):
                    for freq_start in range(num_freq_slides):
                        block = csi_segment[t,tx,stream_start:stream_start + self.stream_win,:]
                        block = block[:, freq_start + self.freq_offsets]
                        # block.shape = (stream_win, freq_win_points).
                        # Row-major flatten matches a_azi kron a_tof.
                        X[:, idx] = block.reshape(-1)
                        idx += 1

        if idx != total_snapshots:
            raise RuntimeError(f"Snapshot mismatch: {idx} != {total_snapshots}")

        Rxx = (X @ X.conj().T) / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        print(
            f"Azi-ToF     Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, tx={num_tx}, "
            f"stream_slides={num_stream_slides}, "
            f"freq_slides={num_freq_slides}"
        )
        return Rxx

    def steering_matrix_chunk(self, tau_chunk):
        """Build unit-norm AoA-ToF steering rows for one ToF chunk."""
        sv_len = self.stream_win * self.freq_win_points
        A = np.empty((len(self.theta_grid) * len(tau_chunk), sv_len),dtype=np.complex128,)

        row = 0
        normalization = np.sqrt(sv_len)
        for theta in self.theta_grid:
            for tau in tau_chunk:
                A[row] = (self.steering_vector.steering_vector_AoA_ToF(theta,tau,self.stream_win,self.freq_win,self.freq_hop,))/normalization
                row += 1
        return A

    def cal_spectrum(self, Rxx, return_sdim=False):
        """Calculate the linear AoA-ToF MUSIC spectrum in ToF chunks."""
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        expected_dim = self.stream_win * self.freq_win_points
        if Rxx.shape != (expected_dim, expected_dim):
            raise ValueError(f"Expected Rxx shape {(expected_dim, expected_dim)}, "f"got {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-ToF")
            self.Sdim = Sdim
        else:
            Sdim = int(np.clip(self.Sdim, 1, expected_dim - 1))
        E_s = eig_vec[:, Sdim:]

        PP = np.empty((len(self.theta_grid), len(self.tau_grid)),dtype=float)
        for start in range(0, len(self.tau_grid), self.tau_chunk):

            end = min(start + self.tau_chunk, len(self.tau_grid))
            tau_chunk = self.tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)

            projection = SV_chunk.conj() @ E_s
            signal_projection = np.sum(np.abs(projection) ** 2,axis=1).real
            signal_projection = np.clip(signal_projection, 0.0, 1.0)
            denominator = np.maximum(
                np.sum(np.abs(projection) ** 2, axis=1).real,
                self.epsilon,
            )
            PP[:, start:end] = (1.0 / denominator).reshape(len(self.theta_grid),len(tau_chunk))

        result = (self.tau_grid, self.theta_grid, PP)
        if return_sdim:
            return (*result, Sdim)
        return result

    def gen_spectrum(
        self,
        CSI,
        frame_idx,
        x_axis="azi",
        y_axis="tof",
        title="Azimuth-ToF",
        file_name=None,
    ):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = self.cal_spectrum(
            Rxx,
            return_sdim=True,
        )
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))

        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            tau_grid,
            P_azi_tof_db,
            self.args,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=file_name,
        )

        return tau_grid, theta_grid, P_azi_tof_db


    def gen_Doppler_projection_spectrum(self, CSI, frame_idx, window="hann", normalize=False):
        """Calculate one Doppler-conditioned Azi-ToF spectrum per target fd.

        ``doppler_projection`` collapses ``avg_frames`` into one coefficient map
        with shape ``(Tx, Rx, subcarrier)``.  A singleton projected-window axis is
        added before passing the map to ``MUSIC.Azi_ToF``, whose input contract is
        ``(window, Tx, Rx, subcarrier)``.
        """
        tof_dop = ToF_Dop(self.args)
        targets = tof_dop.estimate_target_tof_doppler(CSI, frame_idx)
        if not targets:
            print("No targets available for Azi-ToF Doppler-band projection.")
            return []

        azi_tof = Azi_ToF(self.args)
        results = []

        for target in targets:
            fd = target['fd']

            Z_fd = azi_tof.doppler_projection(CSI, fd, self.args, window, normalize)

            projected_snapshot = Z_fd[np.newaxis, ...]
            title = (f"Azimuth-ToF Project onto  tg #{target['rank']:02d} " f"fd {target['fd']:+.2f} Hz")

            file_name = (
                f"{frame_idx:04d}_projected_tg{target['rank']:02d}_"
                f"[{fd:+.2f}].png"
            )
            tau_grid, theta_grid, spectrum_db = azi_tof.gen_spectrum(
                projected_snapshot,
                frame_idx=frame_idx,
                title=title,
                file_name=file_name,
            )
            results.append(
                {
                    "fd": float(fd),
                    "Z_fd": Z_fd,
                    "tau_grid": tau_grid,
                    "theta_grid": theta_grid,
                    "spectrum_db": spectrum_db,
                    "file_name": file_name,
                }
            )

        return results


    @staticmethod
    def doppler_projection(CSI, target_fd, args, window="hann", normalize=False):
        """Project every dynamic-power CSI channel onto one signed Doppler.

        This implements

            Z(fd) = sum_n h[n] * CSI[n] * exp(-j * 2*pi*fd*t[n]).

        The first axis of ``CSI`` is time; all remaining axes are treated as
        independent channels and are preserved in the returned coefficient map.
        For the simulator's ``(frame, Tx, Rx, subcarrier)`` input, the output has
        shape ``(Tx, Rx, subcarrier)``.  With ``num_Tx == 1``, ``Z_fd[0]`` is the
        ``(Rx, subcarrier)`` map used in the research formulation.

        The projection interval is centred on ``frame_idx`` and contains exactly
        ``avg_frames`` samples whenever the recording is long enough.  Near either
        boundary the interval is shifted, rather than shortened.  If the complete
        recording is shorter than ``avg_frames``, all available frames are used;
        this matches ``MUSIC.Azi_ToF.sample_csi_segment``.

        Parameters
        ----------
        CSI : array_like, shape (num_frames, ...)
            Real-valued dynamic linear-power CSI (for example, after background
            subtraction).  This is not complex CSI reconstruction.
        target_fd : float
            Signed target Doppler in Hz, e.g. ``target['fd']``.
        frame_idx : int
            Frame about which the projection context is selected.
        avg_frames : int
            Requested number of projection frames.
        fs : float
            CSI frame sampling rate in Hz.
        window : {'hann', 'rect'}, default='hann'
            Temporal weight ``h[n]``. ``'rect'`` uses equal weights.
        normalize : bool, default=False
            Divide by ``sum(h)`` when True.  The default returns the unnormalised
            sum written in the research equation; covariance-based MUSIC is
            unaffected by this common scalar.

        Returns
        -------
        Z_fd : ndarray, complex
            Doppler-conditioned coefficient map with shape ``CSI.shape[1:]``.
        """
        frame_idx = getattr(args, "plot_frame", getattr(args, "frame_idx", None))
        if frame_idx is None:
            raise AttributeError("args must define plot_frame or frame_idx")
        avg_frames = args.avg_frames
        fs = args.fs

        csi = np.asarray(CSI)
        if csi.ndim < 2:
            raise ValueError(
                "CSI must have time as its first axis and at least one channel axis"
            )
        if csi.shape[0] == 0:
            raise ValueError("CSI must contain at least one frame")
        if np.iscomplexobj(csi):
            raise ValueError(
                "CSI must be real dynamic linear power, not reconstructed complex CSI"
            )
        if not np.all(np.isfinite(csi)):
            raise ValueError("CSI must contain only finite values")

        target_fd = float(target_fd)
        fs = float(fs)
        avg_frames = int(avg_frames)
        if not np.isfinite(target_fd):
            raise ValueError(f"target_fd must be finite, got {target_fd}")
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"fs must be positive and finite, got {fs}")
        if avg_frames <= 0:
            raise ValueError(f"avg_frames must be positive, got {avg_frames}")

        total_frames = csi.shape[0]
        context_len = min(avg_frames, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(
            np.clip(
                frame_idx - context_len // 2,
                0,
                total_frames - context_len,
            )
        )
        end = start + context_len

        if window == "hann":
            # np.hanning(2) is all zeros, so use equal weights for tiny contexts.
            weights = (
                np.hanning(context_len)
                if context_len >= 3
                else np.ones(context_len)
            )
        elif window == "rect":
            weights = np.ones(context_len)
        else:
            raise ValueError(f"window must be 'hann' or 'rect', got {window!r}")

        # Absolute sample time preserves a consistent coefficient phase when the
        # projection centre changes.  target_fd keeps its sign here.
        times_s = np.arange(start, end, dtype=float) / fs
        demodulator = weights * np.exp(-1j * 2.0 * np.pi * target_fd * times_s)
        Z_fd = np.tensordot(demodulator, csi[start:end], axes=(0, 0))

        if normalize:
            Z_fd = Z_fd / np.sum(weights)
        return Z_fd






class ToF_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(
            args,
            "ToF_Dop_Sdim",
            getattr(args, "Sdim", None),
        )
        # For 2D CFAR
        self.training_cells = args.cfar_training_cells
        self.guard_cells = args.cfar_guard_cells
        self.threshold_factor = args.cfar_threshold_factor
        self.top_k = args.cfar_top_k
        self.min_peak_distance = args.cfar_min_peak_distance


        # Steering/smoothing aperture.
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, args.freq_hop)
        freq_space = max(1, args.freq_space)

        # Number of frequency samples available to the smoother. Preserve the
        # existing external-resampling convention without storing three fields.
        freq_limit = int(min(getattr(args, "freq_sample_range", args.num_sc), args.num_sc))
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
        total_snapshots = num_time_slides * num_tx * num_rx * num_freq_slides
        sv_len = time_win * freq_win_points

        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)
        sv_len = time_win * freq_win_points

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
        print(
            f"ToF-Doppler Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, tx={num_tx}, "
            f"freq_slides={num_freq_slides}, "
            f"time_slides={num_time_slides}"
        )
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

        for start in range(0, len(tau_grid), self.tau_chunk):
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

    def estimate_target_tof_doppler(self, CSI, frame_idx):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop = self.cal_spectrum(Rxx)
        detected_targets= cfar_2D(
            P_tof_dop,
            tau_grid,
            fd_grid,
            training_cells=self.training_cells,
            guard_cells=self.guard_cells,
            threshold_factor=self.threshold_factor,
            top_k=self.top_k,
            min_peak_distance=self.min_peak_distance,
        )
        return detected_targets


class Azi_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(
            args,
            "Azi_Dop_Sdim",
            getattr(args, "Sdim", None),
        )

        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_sample_range = int(min(getattr(args, "freq_sample_range", args.num_sc), args.num_sc))
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
        print(
            f"Azi-Doppler Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, tx={num_tx}, "
            f"stream_slides={num_stream_slides}, "
            f"time_slides={num_time_slides}"
        )
        return Rxx

    def steering_matrix(self, theta, fd):
        sv_len = self.stream_win * self.time_win
        A = np.empty((len(theta) * len(fd), sv_len),dtype=np.complex128,)

        row = 0
        for theta_i in theta:
            sv_azi = self.steering_vector.steering_vector_AoA(theta_i,self.stream_win)
            for fd_i in fd:
                sv_dop = self.steering_vector.steering_vector_Dop(fd_i,self.time_win)
                A[row] = np.kron(sv_azi, sv_dop) / np.sqrt(sv_len)
                row += 1

        return A

    def cal_spectrum(self, Rxx, return_sdim=False):
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

        # Each row corresponds to one (theta, fd) steering vector.
        Steering_Vectors = self.steering_matrix(theta, fd)
        A = Steering_Vectors.conj() @ E_n
        
        PP_flat = np.sum(np.abs(A) ** 2, axis=1)
        P_music = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon))
        P_music = P_music.reshape(len(theta), len(fd))

        result = (theta, fd, P_music)
        if return_sdim:
            return (*result, Sdim)
        return result

    def gen_spectrum(self, CSI, frame_idx, x_axis="azi", y_axis="doppler"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return
        
        theta_grid, fd_grid, P_azi_dop, Sdim = self.cal_spectrum(
            Rxx,
            return_sdim=True,
        )
        
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            fd_grid,
            P_azi_dop,
            self.args,
            title="Azimuth-Doppler",
            x_axis=x_axis,
            y_axis=y_axis,
            sdim=Sdim,
            spectrum_axes=("azi", "doppler"),
        )

class Azi_ToF_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(
            args,
            "Azi_ToF_Dop_Sdim",
            getattr(args, "Sdim", None),
        )

        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(getattr(args, "freq_hop", 1)))
        self.freq_sample_range = int(min(getattr(args, "freq_sample_range", args.num_sc), args.num_sc))
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
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.freq_sample_range]
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
        total_snapshots = (num_tx * num_stream_slides * num_freq_slides * num_time_slides)
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
                A[row] = np.kron(sv_aoa, sv_tof_dop) / np.sqrt(self.stream_win)
                row += 1

        return A

    def cal_spectrum(self, Rxx, return_sdim=False):
        print(f"Azi-ToF-Doppler Covariance Matrix shape = {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-ToF-Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))

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

        result = (theta, tau, fd, P_music)
        if return_sdim:
            return (*result, Sdim)
        return result

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

    def gen_spectrum(self, CSI, frame_idx, method="sum", fig_name=None):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        theta, tau, fd, P_music_db, Sdim = self.cal_spectrum(Rxx, return_sdim=True)
        Plot.plot_3D_point_cloud(
            frame_idx,
            theta,
            tau,
            fd,
            P_music_db,
            self.args,
            title="3D Point Cloud",
            sdim=Sdim,
            file_name=f"{frame_idx:04d}_3D_MUSIC_PointCloud.png",
        )
        # Projection must be performed in linear power, not in dB.
        P_music = 10.0 ** (P_music_db / 10.0)
        azi_tof_db = 10.0 * np.log10(self._combine_axis(P_music, axis=2, method=method, axis_values=fd)+ 1e-12)
        tof_dop_db = 10.0 * np.log10(self._combine_axis(P_music, axis=0, method=method, axis_values=theta)+ 1e-12)
        azi_dop_db = 10.0 * np.log10(self._combine_axis(P_music, axis=1, method=method, axis_values=tau)+ 1e-12)

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

        # Build three independent 6.4 x 4.8 inch SubFigures side by side.
        # Each one therefore gets the exact same axes/colorbar layout as the
        # standalone 2D MUSIC figure, not merely one third of a wide canvas.
        fig = plt.figure(figsize=(19.2, 4.8))
        subfigures = fig.subfigures(1, 3, wspace=0)
        axes = np.asarray([subfigure.subplots() for subfigure in subfigures])
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
                sdim=Sdim,
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
            save_dir = os.path.join(self.args.pics_dir, "Azi_ToF_Doppler")
            os.makedirs(save_dir, exist_ok=True)
            if fig_name is None:
                fig_name = f"3D_MUSIC_{method}"
            save_path = os.path.join(save_dir, f"{frame_idx:04d}_{fig_name}.png")
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"Saved: {save_path}")

        return results

    def gen_Doppler_RoI_specturm(self, CSI, frame_idx, fd_range=2, fig_name=None):
        """Generate one Azi-ToF map per detected target Doppler ROI.

        For a target at ``target['fd']``, all 3D MUSIC bins inside
        ``target_fd +/- fd_range`` are summed in linear power along the
        Doppler axis.  Each target result is plotted and returned separately.
        """
        fd_range = float(fd_range)
        if not np.isfinite(fd_range) or fd_range < 0.0:
            raise ValueError(f"fd_range must be a finite non-negative value, got {fd_range}")

        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return []
        theta, tau, fd_grid, P_music_db, Sdim = self.cal_spectrum(
            Rxx,
            return_sdim=True,
        )

        # Detect target (ToF, Doppler) pairs on the independent ToF-Doppler
        # spectrum. ToF_Dop expects the shared argument namespace, not this
        # Azi_ToF_Dop estimator instance.
        tof_dop = ToF_Dop(self.args)
        targets = tof_dop.estimate_target_tof_doppler(CSI, frame_idx)
        if not targets:
            print("No targets available for Azi-ToF Doppler-band projection.")
            return []

        # The 3D spectrum is stored in dB. Doppler-bin accumulation must be
        # performed in linear power, followed by one dB conversion per target.
        P_music = 10.0 ** (P_music_db / 10.0)
        results = []
        for target in targets:
            if "fd" not in target:
                raise KeyError("Each target must contain an 'fd' value.")
            print(
                f"  #{target['rank']}: tau={target['tau'] * 1e9:.2f} ns, "
                f"fd={target['fd']:+.2f} Hz, "
                f"power={target['power_db']:.2f} dB"
            )

            target_fd = float(target["fd"])
            fd_mask = np.abs(fd_grid - target_fd) <= fd_range + 1e-12
            if not np.any(fd_mask):
                print(
                    f"Skip target fd={target_fd:+.2f} Hz: "
                    f"no Doppler bins within ±{fd_range:.2f} Hz."
                )
                continue

            selected_fd = fd_grid[fd_mask]
            azi_tof = np.sum(P_music[:, :, fd_mask], axis=2)
            azi_tof_db = 10.0 * np.log10(np.maximum(azi_tof, 1e-12))

            rank = int(target["rank"])
            if fig_name is None:
                name = "Doppler_RoI"
            else:
                name = os.path.splitext(os.path.basename(str(fig_name)))[0]
            fd_min = target_fd - fd_range
            fd_max = target_fd + fd_range
            file_name = (
                f"{frame_idx:04d}_{name}_tg{rank:02d}_"
                f"[{fd_min:.2f}, {fd_max:.2f}].png"
            )
            title = (
                f"Azi-ToF target #{rank}: "
                f"fd={target_fd:+.2f}±{fd_range:.2f} Hz (sum)"
            )
            Plot.plot_spectrum(
                frame_idx,
                theta,
                tau,
                azi_tof_db,
                self.args,
                title=title,
                x_axis="azi",
                y_axis="tof",
                sdim=Sdim,
                spectrum_axes=("azi", "tof"),
                file_name=file_name,
            )

            result = {
                "target": target,
                "target_fd": target_fd,
                "fd_band": fd_range,
                "selected_fd": selected_fd,
                "theta": theta,
                "tau": tau,
                "spectrum_db": azi_tof_db,
                "method": "sum",
                "file_name": file_name,
            }
            results.append(result)
            print(
                f"Azi-ToF target #{rank}: fd={target_fd:+.2f} Hz, "
                f"band=[{selected_fd[0]:+.2f}, {selected_fd[-1]:+.2f}] Hz, "
                f"bins={selected_fd.size}, method=sum"
            )

        return results
