import numpy as np
import os
from tqdm import tqdm
import pre_processing as pp
import Plot
import matplotlib.pyplot as plt
import weight_generation



def _axis_cell_counts(value, name, *, minimum):
    """Normalize one integer or a ``(tau, fd)`` pair to two cell counts."""
    values = np.asarray(value)
    if values.ndim == 0:
        counts = (int(values), int(values))
    elif values.shape == (2,):
        counts = (int(values[0]), int(values[1]))
    else:
        raise ValueError(f"{name} must be an integer or a (tau, fd) pair")

    if any(count < minimum for count in counts):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must contain {qualifier} integers, got {counts}")
    return counts


def cfar_2D(
    spectrum,
    tau_grid,
    fd_grid,
    training_cells=3,
    guard_cells=1,
    threshold_factor=1.5,
    top_k=None,
    min_peak_distance=1,
):
    """Detect 2-D CA-CFAR peaks in a linear-power ToF-Doppler spectrum."""
    spectrum = np.asarray(spectrum, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    fd_grid = np.asarray(fd_grid, dtype=float)

    if spectrum.ndim != 2:
        raise ValueError(f"spectrum must be 2-D, got shape {spectrum.shape}")
    expected_shape = (tau_grid.size, fd_grid.size)
    if spectrum.shape != expected_shape:
        raise ValueError(
            "spectrum shape must be (len(tau_grid), len(fd_grid)); "
            f"expected {expected_shape}, got {spectrum.shape}"
        )
    if tau_grid.ndim != 1 or fd_grid.ndim != 1:
        raise ValueError("tau_grid and fd_grid must both be 1-D")
    if spectrum.size == 0:
        return []
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum must contain only finite values")
    if np.any(spectrum < 0.0):
        raise ValueError("spectrum must contain non-negative linear power")

    train_tau, train_fd = _axis_cell_counts(training_cells, "training_cells", minimum=1)
    guard_tau, guard_fd = _axis_cell_counts(guard_cells, "guard_cells", minimum=0)
    distance_tau, distance_fd = _axis_cell_counts(min_peak_distance, "min_peak_distance", minimum=1)

    threshold_factor = float(threshold_factor)
    if not np.isfinite(threshold_factor) or threshold_factor <= 0.0:
        raise ValueError(f"threshold_factor must be positive and finite, got {threshold_factor}")
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

    margin_tau = train_tau + guard_tau
    margin_fd = train_fd + guard_fd
    if spectrum.shape[0] < 2 * margin_tau + 1:
        raise ValueError("tau axis is too short for the requested CFAR window")
    if spectrum.shape[1] < 2 * margin_fd + 1:
        raise ValueError("fd axis is too short for the requested CFAR window")

    eps = 1e-12
    detections = []
    for tau_idx in range(margin_tau, spectrum.shape[0] - margin_tau):
        for fd_idx in range(margin_fd, spectrum.shape[1] - margin_fd):
            patch = spectrum[
                tau_idx - margin_tau : tau_idx + margin_tau + 1,
                fd_idx - margin_fd : fd_idx + margin_fd + 1,
            ]

            training_mask = np.ones(patch.shape, dtype=bool)
            center_tau = margin_tau
            center_fd = margin_fd
            training_mask[
                center_tau - guard_tau : center_tau + guard_tau + 1,
                center_fd - guard_fd : center_fd + guard_fd + 1,
            ] = False
            noise_power = float(np.mean(patch[training_mask]))
            threshold = threshold_factor * noise_power
            cut_power = float(spectrum[tau_idx, fd_idx])

            local_patch = spectrum[
                tau_idx - 1 : tau_idx + 2,
                fd_idx - 1 : fd_idx + 2,
            ]
            is_local_max = cut_power >= float(np.max(local_patch))
            if cut_power <= threshold or not is_local_max:
                continue

            detections.append(
                {
                    "tau": float(tau_grid[tau_idx]),
                    "fd": float(fd_grid[fd_idx]),
                    "power": cut_power,
                    "power_db": float(10.0 * np.log10(max(cut_power, eps))),
                    "noise_power": noise_power,
                    "noise_power_db": float(
                        10.0 * np.log10(max(noise_power, eps))
                    ),
                    "threshold": threshold,
                    "threshold_db": float(
                        10.0 * np.log10(max(threshold, eps))
                    ),
                    "snr_db": float(
                        10.0 * np.log10(max(cut_power, eps) / max(noise_power, eps))
                    ),
                    "tau_idx": int(tau_idx),
                    "fd_idx": int(fd_idx),
                }
            )

    detections.sort(key=lambda target: target["power"], reverse=True)

    targets = []
    for detection in detections:
        too_close = any(
            abs(detection["tau_idx"] - selected["tau_idx"]) <= distance_tau
            and abs(detection["fd_idx"] - selected["fd_idx"]) <= distance_fd
            for selected in targets
        )
        if too_close:
            continue
        targets.append(detection)
        if top_k is not None and len(targets) >= top_k:
            break

    for rank, target in enumerate(targets, start=1):
        target["rank"] = rank
    return targets


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
        self.delta_f = args.delta_f

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

class Doppler:

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.fs = args.fs
        self.avg_frames = getattr(args, "avg_frames", 100)
        self.Sdim = getattr(args, "Dop_Sdim", getattr(args, "Sdim", None))

        # Doppler aperture and input context.
        self.time_win = int(args.time_win)
        self.time_sample_range = int(max(getattr(args, "time_sample_range", self.time_win), self.time_win))
        self.time_hop = int(getattr(args, "time_hop", 1))

        # Search grid and estimator controls.

        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, "dop_epsilon", getattr(args, "epsilon", 1e-12)))
        self._steering = np.array([
            self.steering_vector.steering_vector_Dop(fd, self.time_win)
            for fd in self.fd_grid
        ]) / np.sqrt(self.time_win)

        # Configuration validation.

        if self.time_win < 2:
            raise ValueError(f"time_win must be at least 2, got {self.time_win}")
        if self.time_sample_range < self.time_win:
            raise ValueError(f"time_sample_range={self.time_sample_range} cannot be smaller than "f"time_win={self.time_win}")
        if self.time_hop < 1:
            raise ValueError("time_hop must be positive")

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return CSI[start:end], start, end

    def data_check(self, CSI):
        return pp.data_check(self, CSI)

    def _select_subcarriers(self, CSI):
        """Rank each C0 link by Doppler-band peak-to-median power ratio."""
        if getattr(self.args, "sc_selection_method", "snr") != "snr":
            raise ValueError("This Doppler pipeline supports sc_selection_method='snr'")
        CSI = self.data_check(CSI)
        frames, tx_pairs, rx_pairs, num_sc = CSI.shape
        ratio = float(getattr(self.args, "top_sc_ratio", 0.2))
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("top_sc_ratio must be in (0, 1]")
        count = max(1, int(np.ceil(ratio * num_sc)))
        segment_size = self.time_sample_range
        window_size = min(max(self.avg_frames, self.time_win), frames)
        if window_size < 8:
            raise ValueError("Doppler-SNR selection needs at least 8 frames")
        frequencies = np.fft.fftfreq(window_size, d=1.0 / self.fs)
        motion = (np.abs(frequencies) >= 2.0) & (
            np.abs(frequencies) <= min(max(abs(self.fd_grid[0]), abs(self.fd_grid[-1])), self.fs / 2)
        )
        if np.count_nonzero(motion) < 3:
            raise ValueError("Doppler-SNR selection needs at least 3 motion bins")
        indices = np.empty((frames, tx_pairs, rx_pairs, count), dtype=int)
        taper = np.hanning(window_size)[:, None]
        for start in range(0, frames, segment_size):
            end = min(start + segment_size, frames)
            center = (start + end - 1) // 2
            context_start = int(np.clip(center - window_size // 2, 0, frames - window_size))
            context = CSI[context_start:context_start + window_size]
            for tx_idx in range(tx_pairs):
                for rx_idx in range(rx_pairs):
                    channels = context[:, tx_idx, rx_idx, :]
                    power = np.abs(np.fft.fft(channels * taper, axis=0)) ** 2
                    band_power = power[motion]
                    signal = np.max(band_power, axis=0)
                    noise = np.median(band_power, axis=0)
                    score = (signal - noise) / np.maximum(noise, 1e-12)
                    score[~np.isfinite(channels).all(axis=0)] = -np.inf
                    selected = np.sort(np.argsort(-score, kind="stable")[:count])
                    indices[start:end, tx_idx, rx_idx] = selected
        return indices

    def Rxx_smooth(self, CSI, frame_idx, verbose=True):
        channels = self.data_check(CSI)
        if channels.shape[0] < self.time_win:
            raise ValueError(f"Need at least time_win={self.time_win} frames, got {channels.shape[0]}")
        segment, start, end = self.sample_csi_segment(channels, frame_idx)
        time_starts = range(0, segment.shape[0] - self.time_win + 1, self.time_hop)
        num_channels = int(np.prod(segment.shape[1:]))
        num_tx, num_rx, num_sc = segment.shape[1:]
        X = np.empty((self.time_win, len(time_starts) * num_channels), dtype=np.complex128)
        for slide, time_start in enumerate(time_starts):
            X[:, slide * num_channels:(slide + 1) * num_channels] = segment[
                time_start:time_start + self.time_win
            ].reshape(self.time_win, num_channels)

        Rxx = (X @ X.conj().T) / X.shape[1]
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        if verbose:
            print(
                f"Doppler Rxx: {Rxx.shape}, snapshots={X.shape[1]}, "
                f"context={start}:{end}, Tx={num_tx}, Rx={num_rx}, subcarriers={num_sc}, "
                f"time_slides={len(time_starts)}"
            )
        return Rxx

    def cal_spectrum(self, Rxx, return_sdim=False):
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        if Rxx.shape != (self.time_win, self.time_win):
            raise ValueError(
                f"Expected Rxx shape {(self.time_win, self.time_win)}, got {Rxx.shape}"
            )
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[order], eig_vec[:, order]

        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Doppler")
        else:
            Sdim = int(np.clip(self.Sdim, 1, self.time_win - 1))
        E_n = eig_vec[:, Sdim:]
        denominator = np.maximum(
            np.sum(np.abs(self._steering.conj() @ E_n) ** 2, axis=1).real,
            self.epsilon,
        )
        spectrum = 1.0 / denominator
        result = (self.fd_grid, spectrum)
        if return_sdim:
            return (*result, Sdim)
        return result

    def gen_spectrum(self, CSI, frame_idx, sc_select=False, plot=True):
        """Use the selected frame's SC set across its whole covariance context."""
        CSI = self.data_check(CSI)
        if sc_select:
            sc_indices = self._select_subcarriers(CSI)
            selected = sc_indices[int(np.clip(frame_idx, 0, CSI.shape[0] - 1))]
            self.last_selected_indices = selected
            if plot:
                print(f"Frame {frame_idx}: selected {selected.shape[-1]}/{CSI.shape[-1]} "
                      f"subcarriers per link ({getattr(self.args, 'sc_selection_method', 'snr')}): "
                      f"{selected.tolist()}")
            CSI = np.take_along_axis(CSI, selected[None, ...], axis=3)
        Rxx = self.Rxx_smooth(CSI, frame_idx, verbose=plot)

        fd_grid, spectrum, Sdim = self.cal_spectrum(Rxx, return_sdim=True)
        spectrum_db = 10.0 * np.log10(np.maximum(spectrum, 1e-12))
        if plot:
            Plot.plot_1D_spectrum(
                frame_idx, fd_grid, spectrum_db, self.args,
                title="Doppler MUSIC", sdim=Sdim,
            )
        return fd_grid, spectrum_db

    def gen_conti_spectrum(self, raw_CSI, sc_select=False):
        """Compute a continuous MUSIC map for each Tx/Rx-pair C0 channel."""
        CSI, reference_rx = pp.conjugate_multiplication(
            raw_CSI, self.args, return_reference=True
        )
        CSI = CSI - pp.MA(CSI, self.avg_frames)

        total_frames = CSI.shape[0]
        if total_frames < self.time_win:
            raise ValueError(f"Need at least time_win={self.time_win} frames, got {total_frames}")

        context_len = min(self.time_sample_range, total_frames)
        half_context = context_len // 2
        conti_hop = int(getattr(self.args, "conti_hop", 1))

        if conti_hop < 1:
            raise ValueError("conti_hop must be positive")

        last_center = total_frames - (context_len - half_context)
        frame_indices = np.arange(half_context, last_center + 1, conti_hop)
        sc_indices = None
        if sc_select:
            sc_indices = self._select_subcarriers(CSI)
            self.selected_indices_by_frame = sc_indices[frame_indices]
        spectrum = np.empty((CSI.shape[1], CSI.shape[2], self.fd_grid.size, frame_indices.size), dtype=float)
        for tx_idx in range(CSI.shape[1]):
            for rx_idx in range(CSI.shape[2]):
                link = CSI[:, tx_idx, rx_idx, :]
                for column, frame_idx in enumerate(frame_indices):
                    frame_CSI = (
                        np.take_along_axis(link, sc_indices[frame_idx, tx_idx, rx_idx][None, :], axis=1)
                        if sc_select else link
                    )
                    _, spectrum[tx_idx, rx_idx, :, column] = self.gen_spectrum(
                        frame_CSI, int(frame_idx), plot=False
                    )

        used_sc = sc_indices.shape[-1] if sc_select else CSI.shape[-1]
        print(f"Continuous Doppler: using {used_sc}/{CSI.shape[-1]} "
              f"subcarriers per link, {CSI.shape[1] * CSI.shape[2]} links for {frame_indices.size} frames"
              + (f" ({getattr(self.args, 'sc_selection_method', 'snr')})" if sc_select else ""))
        time_s = frame_indices / float(self.args.fs)
        conjugate_order = getattr(self.args, "conjugate_order", "fixed")
        Plot.plot_conti_spectrum(
            time_s, self.fd_grid, spectrum, self.args,
            reference_rx=reference_rx[frame_indices],
            file_name="continuous_ratio.png" if conjugate_order == "ratio" else "continuous.png",
        )
        return time_s, self.fd_grid, spectrum

class Azi_ToF:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args,"Azi_ToF_Sdim",getattr(args, "Sdim", None),)

        # Steering/smoothing aperture.
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(args.freq_hop))
        self.freq_space = max(1, int(args.freq_space))
        self.avg_frames = max(1, int(args.avg_frames))

        # Search grid and estimator controls.
        self.tau_chunk = max(1, int(getattr(args, "azi_tof_tau_chunk", 10)),)
        self.theta_grid = np.arange(args.theta_min, args.theta_max + 0.5 * args.theta_step, args.theta_step)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)

        # Half-width of the Stage 1 / Stage 2 ToF consistency gate, in seconds.
        self.tof_gate = float(getattr(args, "tof_gate", 10e-9))
        if not np.isfinite(self.tof_gate) or self.tof_gate < 0:
            raise ValueError("tof_gate must be finite and non-negative")


        # Number of frequency samples available to the smoother. Preserve the
        # existing external-resampling convention without storing three fields.
        freq_limit = int(min(getattr(args, "freq_sample_range", args.num_sc), args.num_sc,))
        self.num_freq_samples = len(np.arange(0, freq_limit, self.freq_space))

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

        self.freq_offsets = (np.arange(self.freq_win_points) * self.freq_hop)
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
        """
        print(
            f"Azi-ToF     Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, tx={num_tx}, "
            f"stream_slides={num_stream_slides}, "
            f"freq_slides={num_freq_slides}"
        )
        """
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

    def cal_spectrum(
        self,
        Rxx,
        return_sdim=True,
        sdim_attr=None,
        energy_ratio_attr=None,
        label="Azi-ToF",
    ):
        """Calculate the linear AoA-ToF MUSIC spectrum in ToF chunks."""
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        expected_dim = self.stream_win * self.freq_win_points
        if Rxx.shape != (expected_dim, expected_dim):
            raise ValueError(f"Expected Rxx shape {(expected_dim, expected_dim)}, "f"got {Rxx.shape}")

        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        if sdim_attr is not None or energy_ratio_attr is not None:
            Sdim = resolve_Sdim(
                self.args,
                eig_val,
                label=label,
                sdim_attr=sdim_attr or "Azi_ToF_Sdim",
                energy_ratio_attr=energy_ratio_attr or "Sdim_energy_ratio",
            )
        elif self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label="Azi-ToF")
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
            denominator = np.maximum(np.sum(np.abs(projection) ** 2, axis=1).real, 1e-12)
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
        plot=True,
        return_sdim=False,
        sdim_attr=None,
        energy_ratio_attr=None,
        sdim_label="Azi-ToF",
    ):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = self.cal_spectrum(
            Rxx,
            return_sdim=True,
            sdim_attr=sdim_attr,
            energy_ratio_attr=energy_ratio_attr,
            label=sdim_label,
        )
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))

        if plot:
            Plot.plot_spectrum(
                frame_idx,
                x_values= theta_grid,
                y_values = tau_grid,
                P_music = P_azi_tof_db.T,
                args = self.args,
                title = title,
                sdim = Sdim,
                spectrum_axes=(x_axis, y_axis),
                file_name=file_name,
            )

        result = (tau_grid, theta_grid, P_azi_tof_db)
        if return_sdim:
            return (*result, Sdim)
        return result


    def gen_Doppler_projection_spectrum(
        self,
        CSI,
        frame_idx,
        fd_neighbor=0,
        doppler_step=1.0,
        window="hann",
        normalize=False,
        fig_name="Project_tg",
    ):
        """Generate one frequency-band-summed Azi-ToF map per target."""
        return self._doppler_projection_band_spectra(
            CSI,
            frame_idx,
            fd_neighbor=fd_neighbor,
            doppler_step=doppler_step,
            window=window,
            normalize=normalize,
            fig_name=fig_name,
            plot=True,
        )

    def return_Doppler_projection_spectrum(
        self,
        CSI,
        frame_idx,
        fd_neighbor=0,
        doppler_step=1.0,
        window="hann",
        normalize=False,
        fig_name="Project_tg",
    ):
        """Return one frequency-band-summed Azi-ToF map per target."""
        return self._doppler_projection_band_spectra(
            CSI,
            frame_idx,
            fd_neighbor=fd_neighbor,
            doppler_step=doppler_step,
            window=window,
            normalize=normalize,
            fig_name=fig_name,
            plot=False,
        )

    def _doppler_projection_band_spectra(
        self,
        CSI,
        frame_idx,
        fd_neighbor,
        doppler_step,
        window,
        normalize,
        fig_name,
        plot,
    ):
        """Sum projected Azi-ToF maps over each target's Doppler band.

        ``doppler_projection`` collapses ``avg_frames`` into one coefficient map
        with shape ``(Tx, Rx, subcarrier)``.  A singleton projected-window axis is
        added before passing the map to ``MUSIC.Azi_ToF``, whose input contract is
        ``(window, Tx, Rx, subcarrier)``. ``fd_neighbor`` is the Doppler radius
        in Hz, and frequencies spaced by ``doppler_step`` are projected from
        ``fd - fd_neighbor`` through ``fd + fd_neighbor``. Every unique frequency
        is calculated once. For each target, its maps are converted back to linear
        power, summed, and converted to dB to produce one output map at the target's
        detected center Doppler.
        """
        if isinstance(fd_neighbor, (bool, np.bool_)):
            raise ValueError("fd_neighbor must be non-negative and finite")
        fd_neighbor = float(fd_neighbor)
        if not np.isfinite(fd_neighbor) or fd_neighbor < 0.0:
            raise ValueError("fd_neighbor must be non-negative and finite")
        doppler_step = float(doppler_step)
        if not np.isfinite(doppler_step) or doppler_step <= 0.0:
            raise ValueError("doppler_step must be positive and finite")
        neighbor_bin_count = int(
            np.floor(fd_neighbor / doppler_step + 1e-12)
        )

        tof_dop = ToF_Dop(self.args)
        targets = tof_dop.estimate_target_tof_doppler(CSI, frame_idx)
        if not targets:
            print("No targets available for Azi-ToF Doppler-band projection.")
            return []

        # Store each target's band membership while deduplicating the actual
        # projection frequencies globally.
        projection_bins = {}
        target_bands = []
        for target in targets:
            center_fd = float(target["fd"])
            band_members = []
            for offset_bin in range(-neighbor_bin_count, neighbor_bin_count + 1):
                requested_fd = center_fd + offset_bin * doppler_step

                if (requested_fd < self.args.doppler_min - 1e-12 or requested_fd > self.args.doppler_max + 1e-12):
                    continue
                fd_idx = int(np.rint((requested_fd - self.args.doppler_min) / doppler_step))

                fd = float(self.args.doppler_min + fd_idx * doppler_step)
                projection_bins.setdefault(fd_idx, fd)
                band_members.append((fd_idx, offset_bin))
            target_bands.append((target, band_members))

        print(
            f"Doppler projection: {len(targets)} targets use "
            f"{len(projection_bins)} unique frequencies and produce at most "
            f"{len(targets)} band-summed maps "
            f"(fd_neighbor={fd_neighbor:g} Hz, "
            f"doppler_step={doppler_step:g} Hz)"
        )

        azi_tof = Azi_ToF(self.args)
        projected_maps = {}
        for fd_idx, fd in projection_bins.items():
            Z_fd = azi_tof.doppler_projection(CSI, frame_idx, fd, self.args, window, normalize)

            projected_snapshot = Z_fd[np.newaxis, ...]
            tau_grid, theta_grid, spectrum_db, Sdim = azi_tof.gen_spectrum(
                projected_snapshot,
                frame_idx=frame_idx,
                plot=False,
                return_sdim=True,
                sdim_attr="Projected_Azi_ToF_Sdim",
                energy_ratio_attr="Projected_Azi_ToF_Sdim_ratio",
                sdim_label="Projected Azi-ToF",
            )
            projected_maps[fd_idx] = {
                "fd": fd,
                "Z_fd": Z_fd,
                "spectrum_db": spectrum_db,
                "Sdim": int(Sdim),
            }

        results = []
        for target, band_members in target_bands:
            if not band_members:
                continue

            valid_members = []
            rejected_fds = []
            for fd_idx, offset_bin in band_members:
                projected_map = projected_maps[fd_idx]
                if (
                    getattr(self.args, "tof_gating", True)
                    and not self.tof_gating_check(
                        projected_map["spectrum_db"],
                        target["tau"],
                        self.tof_gate,
                    )
                ):
                    rejected_fds.append(float(projected_map["fd"]))
                    continue
                valid_members.append((fd_idx, offset_bin))

            center_fd = float(target["fd"])
            target_rank = int(target["rank"])
            if not valid_members:
                print(
                    f"⚠️ Target #{target_rank:02d} fd {center_fd:+.2f} Hz: "
                    "all projected frequencies failed the "
                    f"{self.tof_gate * 1e9:.2f} ns ToF gate. Skip this target."
                )
                continue

            band_maps = [
                projected_maps[fd_idx] for fd_idx, _ in valid_members
            ]
            spectrum_linear_sum = np.sum(
                [10.0 ** (item["spectrum_db"] / 10.0) for item in band_maps],
                axis=0,
            )
            spectrum_db = 10.0 * np.log10(
                np.maximum(spectrum_linear_sum, 1e-12)
            )
            projection_fds = [float(item["fd"]) for item in band_maps]

            if rejected_fds:
                print(
                    f"Target #{target_rank:02d} fd {center_fd:+.2f} Hz: "
                    f"ToF gate kept {len(valid_members)}/{len(band_members)} "
                    f"projected frequencies; rejected "
                    + ", ".join(f"{fd:+.2f}" for fd in rejected_fds)
                    + " Hz"
                )

            file_name = (
                f"{frame_idx:04d}_{fig_name}{target_rank:02d}_"
                f"[{center_fd:+.2f}].png"
            )
            if plot:
                title = (
                    f"Azimuth-ToF Projected Band Sum fd {center_fd:+.2f} Hz "
                    f"[{min(projection_fds):+.2f}, "
                    f"{max(projection_fds):+.2f}] Hz"
                )
                Plot.plot_spectrum(
                    frame_idx,
                    theta_grid,
                    tau_grid,
                    spectrum_db.T,
                    self.args,
                    title=title,
                    spectrum_axes=("azi", "tof"),
                    file_name=file_name,
                )
                """
                # 畫圖用
                Plot.plot_clean_spectrum(
                    frame_idx,
                    theta_grid,
                    tau_grid,
                    spectrum_db.T,
                    self.args,
                    file_name=f"{os.path.splitext(file_name)[0]}_clean.png",
                )
                """
            results.append(
                {
                    "fd": center_fd,
                    "fd_idx": int(target["fd_idx"]),
                    "fd_neighbor": fd_neighbor,
                    "doppler_step": doppler_step,
                    "target": target,
                    "targets": [target],
                    "target_ranks": [target_rank],
                    "matched_target_ranks": [target_rank],
                    "neighbor_offsets": [
                        offset for _, offset in valid_members
                    ],
                    "neighbor_frequency_offsets_hz": [
                        offset * doppler_step
                        for _, offset in valid_members
                    ],
                    "projection_fd_indices": [
                        fd_idx for fd_idx, _ in valid_members
                    ],
                    "projection_fds": projection_fds,
                    "rejected_projection_fds": rejected_fds,
                    "projection_sdims": [
                        item["Sdim"] for item in band_maps
                    ],
                    "Z_fds": [item["Z_fd"] for item in band_maps],
                    "tau_grid": tau_grid,
                    "theta_grid": theta_grid,
                    "spectrum_db": spectrum_db,
                    "file_name": file_name,
                }
            )

        if not results:
            print("No Doppler-projected target bands passed the ToF gate.")
        return results


    @staticmethod
    def doppler_projection(CSI, frame_idx, target_fd, args, window="hann", normalize=False):
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
        CSI : ndarray, shape (num_frames, ...)
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
        avg_frames = args.avg_frames
        fs = args.fs

        target_fd = float(target_fd)
        fs = float(fs)
        avg_frames = int(avg_frames)
        if not np.isfinite(target_fd):
            raise ValueError(f"target_fd must be finite, got {target_fd}")
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"fs must be positive and finite, got {fs}")
        if avg_frames <= 0:
            raise ValueError(f"avg_frames must be positive, got {avg_frames}")

        total_frames = CSI.shape[0]
        context_len = min(avg_frames, total_frames)
        #frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        segment = CSI[start:end]
        if not np.all(np.isfinite(segment)):
            raise ValueError("CSI projection segment must contain only finite values")

        if window == "hann":
            # np.hanning(2) is all zeros, so use equal weights for tiny contexts.
            weights = (
                np.hanning(context_len)
                if context_len >= 3
                else np.ones(context_len)
            )
        elif window == "rect":
            weights = np.ones(context_len)

        elif window == "weight":
            weights = weight_generation.weight_gen(target_fd)
        else:
            raise ValueError(f"window must be 'hann' or 'rect', got {window!r}")

        # Absolute sample time preserves a consistent coefficient phase when the
        # projection centre changes.  target_fd keeps its sign here.
        times_s = np.arange(start, end, dtype=float) / fs
        demodulator = weights * np.exp(-1j * 2.0 * np.pi * target_fd * times_s)
        Z_fd = np.tensordot(demodulator, segment, axes=(0, 0))

        if normalize:
            Z_fd = Z_fd / np.sum(weights)
        return Z_fd

    def tof_gating_check(self, Z, tau, gating_range):
        """Check whether the strongest Azi-ToF peak matches a Stage 1 ToF.

        ``Z`` has shape (azimuth, ToF) and may be on a dB scale. ``tau`` and
        ``gating_range`` are in seconds. The peak is found over the full map
        before applying the gate, so a peak forced into the gate cannot pass
        the consistency check by construction.
        """
        spectrum = np.asarray(Z)
        expected_shape = (self.theta_grid.size, self.tau_grid.size)
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"Azi-ToF spectrum must have shape {expected_shape}, "
                f"got {spectrum.shape}"
            )
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("Azi-ToF spectrum must contain only finite values")

        tau = float(tau)
        gating_range = float(gating_range)
        if not np.isfinite(tau):
            raise ValueError("target tau must be finite")
        if not np.isfinite(gating_range) or gating_range < 0:
            raise ValueError("gating_range must be finite and non-negative")

        _, tau_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        return bool(abs(self.tau_grid[tau_idx] - tau) <= gating_range)

class ToF_Dop:
    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, "ToF_Dop_Sdim", getattr(args, "Sdim", None),)

        self.num_Rx = args.num_Rx
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
        """
        print(
            f"ToF-Doppler Rxx: {Rxx.shape}, snapshots={total_snapshots}, "
            f"context={start}:{end}, tx={num_tx}, "
            f"freq_slides={num_freq_slides}, "
            f"time_slides={num_time_slides}"
        )
        """
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

    def cal_spectrum(
        self,
        Rxx,
        return_sdim=False,
        sdim_attr=None,
        energy_ratio_attr=None,
        label="ToF-Doppler",
    ):
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx_order], eig_vec[:, idx_order]

        # Signal subspace projection, same convention as XMUSIC_ToF_Dop/Guan.
        if sdim_attr is not None or energy_ratio_attr is not None:
            Sdim = resolve_Sdim(
                self.args,
                eig_val,
                label=label,
                sdim_attr=sdim_attr or "ToF_Dop_Sdim",
                energy_ratio_attr=energy_ratio_attr or "Sdim_energy_ratio",
            )
        elif self.Sdim is None:
            Sdim = resolve_Sdim(self.args,eig_val)
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

        result = (tau_grid, fd_grid, PP)
        if return_sdim:
            return (*result, Sdim)
        return result


    def gen_spectrum(self, CSI, frame_idx, x_axis="doppler", y_axis="tof", plot=True):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(Rxx, return_sdim=True)

        if not plot:
            return tau_grid, fd_grid, P_tof_dop
        else:
            Plot.plot_spectrum(
                frame_idx,
                x_values = fd_grid,
                y_values = tau_grid,
                P_music = P_tof_dop,
                args = self.args,
                title="ToF-Doppler",
                sdim=Sdim,
                spectrum_axes=(x_axis, y_axis),
            )

    def estimate_target_tof_doppler(self, CSI, frame_idx, use_beamforming=True):
        if use_beamforming:
            tau_grid, fd_grid, P_tof_dop = self.return_spectrum_Azi_beamforming_spectrum(CSI, frame_idx)
        else:
            tau_grid, fd_grid, P_tof_dop = self.gen_spectrum(CSI, frame_idx, plot=False)
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

    def gen_spectrum_Azi_beamforming_spectrum(self, CSI, frame_idx):
        """Print the strongest azimuth in the Azi-ToF MUSIC spectrum."""

        """
        Iteration #0
        Azi-Tof spectrum & ToF-Doppler spectrum
        """
        iter = 0
        azi_tof = Azi_ToF(self.args)
        Rxx = azi_tof.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = azi_tof.cal_spectrum(Rxx, return_sdim=True)
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))
        #idx_theta, idx_tau = np.unravel_index(np.argmax(P_azi_tof),P_azi_tof.shape,)
        #max_theta = float(theta_grid[idx_theta])
        #max_tau = float(tau_grid[idx_tau])
        """
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            tau_grid,
            P_azi_tof_db.T,
            self.args,
            title=f"Azi-ToF iter {iter}",
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
            )
        """

        tof_dop = ToF_Dop(self.args)
        Rxx = tof_dop.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(Rxx, return_sdim=True)
        P_tof_dop_db = 10.0 * np.log10(np.maximum(P_tof_dop, 1e-12))
        Plot.plot_spectrum(
            frame_idx,
            fd_grid,
            tau_grid,
            P_tof_dop_db,
            self.args,
            title=f"ToF-Doppler iter {iter}",
            sdim=Sdim,
            spectrum_axes=("doppler", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
        )
        """
        Iteration #1
        Beamformed ToF-Doppler spectrum (Azimuth beamforming)
        Beamformed Azi-ToF spectrum (ToF beamforming)
        """
        iter += 1

        CSI_bf = self.beamform_azimuth(CSI, P_azi_tof)
        beam_theta = self.last_beamform_theta
        Rxx_bf = self.Rxx_smooth(CSI_bf, frame_idx)
        tau_grid, fd_grid, P_tof_dop_bf, Sdim = self.cal_spectrum(
            Rxx_bf,
            return_sdim=True,
            sdim_attr="Beamformed_ToF_Dop_Sdim",
            energy_ratio_attr="Beamformed_ToF_Dop_Sdim_ratio",
            label="Beamformed ToF-Doppler",
        )
        P_tof_dop_db = 10.0 * np.log10(np.maximum(P_tof_dop_bf, 1e-12))
        Plot.plot_spectrum(
            frame_idx,
            fd_grid,
            tau_grid,
            P_tof_dop_db,
            self.args,
            title=f"ToF-Doppler iter {iter} Beamformed {beam_theta:.2f} deg",
            sdim=Sdim,
            spectrum_axes=("doppler", "tof"),
            file_name = f"{frame_idx:04d}-Iter{iter}.png"
        )

        # Plot the 1-D array response of the selected azimuth beam.
        w = self.steering_vector.steering_vector_AoA(
            beam_theta, stream_win=self.num_Rx
        )
        w = w / np.linalg.norm(w)
        theta_grid_lobe = np.arange(
            self.args.theta_min, self.args.theta_max + 0.1, 0.1
        )
        pattern = np.array(
            [
                np.abs(
                    np.vdot(
                        w,
                        self.steering_vector.steering_vector_AoA(
                            theta, stream_win=self.num_Rx
                        ),
                    )
                )
                ** 2
                for theta in theta_grid_lobe
            ]
        )
        pattern_db = 10.0 * np.log10(np.maximum(pattern, 1e-12))
        pattern_db = np.clip(pattern_db - np.max(pattern_db), -40.0, 0.0)

        fig_lobe, ax_lobe = plt.subplots(
            figsize=(8, 5), subplot_kw={"projection": "polar"}
        )
        theta_rad = np.deg2rad(theta_grid_lobe)
        ax_lobe.plot(theta_rad, pattern_db, color="#1f77b4", linewidth=2.0)
        ax_lobe.axvline(
            np.deg2rad(beam_theta),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
        )
        ax_lobe.set_theta_zero_location("W")
        ax_lobe.set_theta_direction(-1)
        ax_lobe.set_thetamin(0)
        ax_lobe.set_thetamax(180)
        ax_lobe.set_ylim(-40, 0)
        ax_lobe.set_yticks([-40, -30, -20, -10, 0])
        ax_lobe.set_yticklabels(["-40 dB", "-30", "-20", "-10", "0"])
        ax_lobe.set_rlabel_position(0)
        xticks_deg = [0, 30, 60, 90, 120, 150, 180]
        ax_lobe.set_xticks(np.deg2rad(xticks_deg))
        ax_lobe.set_xticklabels([f"{degree}°" for degree in xticks_deg])
        ax_lobe.set_title(
            f"Azimuth Beam Lobe @ frame {frame_idx} "
            f"(beam {beam_theta:.2f}°)",
            pad=20,
        )
        ax_lobe.grid(True, linewidth=0.8, alpha=0.6)
        fig_lobe.tight_layout()

        pics_dir = getattr(self.args, "pics_dir", None)
        if pics_dir is not None:
            lobe_save_dir = os.path.join(pics_dir, "Beamform_Lobe")
            os.makedirs(lobe_save_dir, exist_ok=True)
            save_path = os.path.join(
                lobe_save_dir,
                f"{frame_idx:04d}_Azimuth_Lobe_Polar.png",
            )
            fig_lobe.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved Polar Beam Lobe to {save_path}")
        plt.close(fig_lobe)

        """
        目前這種「累積 hard rank-1 projection」繼續 iteration 幾乎沒有意義。
        你的兩個 projection 都具有冪等性：

        P_\theta^2=P_\theta,\qquad P_\tau^2=P_\tau

        目前流程相當於： X_1=P_\theta X

        接著： X_2=P_\theta X P_\tau

        再次使用相同方向與 ToF：
        P_\theta X_2P_\tau= P_\theta^2XP_\tau^2= P_\theta XP_\tau= X_2

        所以第一次同時套用 Azimuth 與 ToF projection 後，資料已被壓進：
        operatorname{span}(a_\theta)\otimes\operatorname{span}(a_\tau)

        Azi–ToF map 變成一個點，是 projection 強制產生的結果，不代表解析度真的提高。
        先前數值也顯示 Iter1 covariance 第一特徵值占比為 1.0，確實已經是 rank 1。

        CSI_bf = self.beamform_tof(CSI_bf, P_tof_dop_bf)
        Rxx = azi_tof.Rxx_smooth(CSI_bf, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = azi_tof.cal_spectrum(Rxx, return_sdim=True,)
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))
        #idx_theta, idx_tau = np.unravel_index(np.argmax(P_azi_tof),P_azi_tof.shape,)
        #max_theta = float(theta_grid[idx_theta])
        #max_tau = float(tau_grid[idx_tau])
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            tau_grid,
            P_azi_tof_db.T,
            self.args,
            title=f"Azi-ToF iter {iter}",
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
            )
        """


    def return_spectrum_Azi_beamforming_spectrum(self, CSI, frame_idx):
        """Print the strongest azimuth in the Azi-ToF MUSIC spectrum."""

        """
        Iteration #0
        Azi-Tof spectrum & ToF-Doppler spectrum
        """
        iter = 0
        azi_tof = Azi_ToF(self.args)
        Rxx = azi_tof.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = azi_tof.cal_spectrum(Rxx,return_sdim=True)
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))
        #idx_theta, idx_tau = np.unravel_index(np.argmax(P_azi_tof),P_azi_tof.shape,)
        #max_theta = float(theta_grid[idx_theta])
        #max_tau = float(tau_grid[idx_tau])
        """
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            tau_grid,
            P_azi_tof_db.T,
            self.args,
            title=f"Azi-ToF iter {iter}",
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
            )
        """

        tof_dop = ToF_Dop(self.args)
        Rxx = tof_dop.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(Rxx, return_sdim=True)

        """
        Plot.plot_spectrum(
            frame_idx,
            fd_grid,
            tau_grid,
            P_tof_dop_db,
            self.args,
            title=f"ToF-Doppler iter {iter}",
            sdim=Sdim,
            spectrum_axes=("doppler", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
        )
        """
        """
        Iteration #1
        Beamformed ToF-Doppler spectrum (Azimuth beamforming)
        Beamformed Azi-ToF spectrum (ToF beamforming)
        """
        iter += 1

        CSI_bf = self.beamform_azimuth(CSI, P_azi_tof)
        beam_theta = self.last_beamform_theta
        Rxx_bf = self.Rxx_smooth(CSI_bf, frame_idx)
        tau_grid, fd_grid, P_tof_dop_bf, Sdim = self.cal_spectrum(
            Rxx_bf,
            return_sdim=True,
            sdim_attr="Beamformed_ToF_Dop_Sdim",
            energy_ratio_attr="Beamformed_ToF_Dop_Sdim_ratio",
            label="Beamformed ToF-Doppler",
        )
        #P_tof_dop_db = 10.0 * np.log10(np.maximum(P_tof_dop_bf, 1e-12))
        """
        Plot.plot_spectrum(
            frame_idx,
            fd_grid,
            tau_grid,
            P_tof_dop_db,
            self.args,
            title=f"ToF-Doppler iter {iter} Beamformed {beam_theta:.2f} deg",
            sdim=Sdim,
            spectrum_axes=("doppler", "tof"),
            file_name = f"{frame_idx:04d}-Iter{iter}.png"
        )
        """

        """
        目前這種「累積 hard rank-1 projection」繼續 iteration 幾乎沒有意義。
        你的兩個 projection 都具有冪等性：

        P_\theta^2=P_\theta,\qquad P_\tau^2=P_\tau

        目前流程相當於： X_1=P_\theta X

        接著： X_2=P_\theta X P_\tau

        再次使用相同方向與 ToF：
        P_\theta X_2P_\tau= P_\theta^2XP_\tau^2= P_\theta XP_\tau= X_2

        所以第一次同時套用 Azimuth 與 ToF projection 後，資料已被壓進：
        operatorname{span}(a_\theta)\otimes\operatorname{span}(a_\tau)

        Azi–ToF map 變成一個點，是 projection 強制產生的結果，不代表解析度真的提高。
        先前數值也顯示 Iter1 covariance 第一特徵值占比為 1.0，確實已經是 rank 1。

        CSI_bf = self.beamform_tof(CSI_bf, P_tof_dop_bf)
        Rxx = azi_tof.Rxx_smooth(CSI_bf, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = azi_tof.cal_spectrum(Rxx, return_sdim=True,)
        P_azi_tof_db = 10.0 * np.log10(np.maximum(P_azi_tof, 1e-12))
        #idx_theta, idx_tau = np.unravel_index(np.argmax(P_azi_tof),P_azi_tof.shape,)
        #max_theta = float(theta_grid[idx_theta])
        #max_tau = float(tau_grid[idx_tau])
        Plot.plot_spectrum(
            frame_idx,
            theta_grid,
            tau_grid,
            P_azi_tof_db.T,
            self.args,
            title=f"Azi-ToF iter {iter}",
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=f"{frame_idx:04d}-Iter{iter}.png"
            )
        """
        return tau_grid, fd_grid, P_tof_dop_bf



    def beamform_tof(self, CSI, P_tof_dop_bf):
        """Estimate 1-D ToF and project CSI onto its frequency steering.

        ``P_tof_dop_bf`` must be a linear-power spectrum with shape
        (ToF, Doppler). Its maximum over Doppler forms a one-dimensional ToF
        profile. The strongest ToF selects a rank-1 frequency projection.

        The returned CSI keeps shape (frame, Tx, Rx, subcarrier).
        """
        csi = np.asarray(CSI)
        spectrum = np.asarray(P_tof_dop_bf, dtype=float)

        if csi.ndim != 4:
            raise ValueError(
                "CSI must have shape (frame, Tx, Rx, subcarrier), "
                f"got {csi.shape}"
            )
        expected_shape = (self.tau_grid.size, self.fd_grid.size)
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"P_tof_dop_bf must have shape {expected_shape}, "
                f"got {spectrum.shape}"
            )
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
            raise ValueError(
                "P_tof_dop_bf must contain finite, non-negative linear power"
            )

        # P_tof_dop_bf is indexed as (ToF, Doppler). Preserve the strongest
        # Doppler response at each ToF to obtain a one-dimensional ToF profile.
        P_tof = np.max(spectrum, axis=1)
        if float(np.max(P_tof)) <= 0.0:
            raise ValueError("P_tof_dop_bf must contain positive power")

        peak_idx = int(np.argmax(P_tof))
        beam_tau = float(self.tau_grid[peak_idx])
        num_sc = csi.shape[-1]
        steering = self.steering_vector.steering_vector_ToF(
            beam_tau,
            freq_win=num_sc,
            freq_hop=1,
        ).reshape(-1, 1)
        norm_sq = float(np.vdot(steering, steering).real)
        projection_matrix = (
            steering @ steering.conj().T
        ) / max(norm_sq, 1e-12)

        # Contract the input subcarrier axis with the projector's input axis.
        # The output axis is already (frame, Tx, Rx, subcarrier).
        CSI_beamformed = np.tensordot(
            csi,
            projection_matrix,
            axes=([3], [1]),
        )

        self.last_tof_grid = self.tau_grid
        self.last_tof_profile = P_tof
        self.last_tof_weight_matrix = projection_matrix
        self.last_beamform_tau = beam_tau
        return CSI_beamformed

    def beamform_azimuth(self, CSI, P_azi_tof):
        """Estimate 1-D azimuth from Azi-ToF and form one coherent beam.

        ``P_azi_tof`` is a linear-power spectrum with shape (theta, ToF).
        Its maximum over ToF forms a 1-D azimuth profile. The strongest
        direction selects one rank-1 spatial projection.

        The returned shape remains (frame, Tx, Rx, subcarrier), matching
        ``beamform_theta`` and preserving Rx channels for downstream snapshots.
        """
        csi = np.asarray(CSI)
        spectrum = np.asarray(P_azi_tof, dtype=float)
        theta_grid = np.arange(
            self.args.theta_min,
            self.args.theta_max + 0.5 * self.args.theta_step,
            self.args.theta_step,
        )
        tau_grid = self.tau_grid

        if csi.ndim != 4:
            raise ValueError(
                "CSI must have shape (frame, Tx, Rx, subcarrier), "
                f"got {csi.shape}"
            )
        if csi.shape[2] != self.num_Rx:
            raise ValueError(
                f"Expected {self.num_Rx} Rx channels, got {csi.shape[2]}"
            )
        expected_shape = (theta_grid.size, tau_grid.size)
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"P_azi_tof must have shape {expected_shape}, "
                f"got {spectrum.shape}"
            )
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
            raise ValueError(
                "P_azi_tof must contain finite, non-negative linear power"
            )

        # P_azi_tof is indexed as (theta, ToF). Preserve the strongest ToF
        # response at each angle to obtain a one-dimensional azimuth profile.
        P_azi = np.max(spectrum, axis=1)
        if float(np.max(P_azi)) <= 0.0:
            raise ValueError("P_azi_tof must contain positive power")

        peak_idx = int(np.argmax(P_azi))
        beam_theta = float(theta_grid[peak_idx])
        steering = self.steering_vector.steering_vector_AoA(
            beam_theta,
            stream_win=self.num_Rx,
        ).reshape(-1, 1)
        norm_sq = float(np.vdot(steering, steering).real)
        projection_matrix = (
            steering @ steering.conj().T
        ) / max(norm_sq, 1e-12)
        CSI_beamformed = self.beamform_theta(csi, beam_theta)

        self.last_azimuth_grid = theta_grid
        self.last_azimuth_profile = P_azi
        self.last_azimuth_weight_matrix = projection_matrix
        self.last_beamform_theta = beam_theta
        return CSI_beamformed

    def beamform_theta(self, CSI, max_theta):

        sv = self.steering_vector.steering_vector_AoA(max_theta, stream_win=self.num_Rx).reshape(-1, 1)

        norm_sq = np.sum(np.abs(sv)**2)

        P_para = (sv @ sv.conj().T) / norm_sq

        CSI_proj = np.tensordot(CSI, P_para, axes=([2], [1]))

        CSI_proj = np.transpose(CSI_proj, (0, 1, 3, 2))

        return CSI_proj


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
        
        if (x_axis, y_axis) == ("azi", "doppler"):
            x_values, y_values = theta_grid, fd_grid
            display_spectrum = P_azi_dop.T
        elif (x_axis, y_axis) == ("doppler", "azi"):
            x_values, y_values = fd_grid, theta_grid
            display_spectrum = P_azi_dop
        else:
            raise ValueError("Azi_Dop axes must be ('azi', 'doppler') or reversed")
        Plot.plot_spectrum(
            frame_idx,
            x_values,
            y_values,
            display_spectrum,
            self.args,
            title="Azimuth-Doppler",
            sdim=Sdim,
            spectrum_axes=(x_axis, y_axis),
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
        self.time_hop = max(1, int(getattr(args, "time_hop", 1)))
        self.avg_frames = args.avg_frames

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
        if self.time_win <= 0:
            raise ValueError(f"time_win must be positive, got {self.time_win}")

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))

        if total_frames < self.time_win:
            return CSI, 0, total_frames

        max_rxx = 1 + (total_frames - self.time_win) // self.time_hop
        num_rxx = min(self.avg_frames, max_rxx)
        context_len = self.time_win + (num_rxx - 1) * self.time_hop

        # For an even avg_frames, use centres
        # [frame_idx - avg_frames//2, ..., frame_idx + avg_frames//2 - 1].
        requested_start = (
            frame_idx
            - (num_rxx // 2) * self.time_hop
            - self.time_win // 2
        )
        start = int(np.clip(requested_start, 0, total_frames - context_len))
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

        # One local Rxx is formed at every time_start from all Tx, spatial,
        # and frequency-smoothed snapshots. The final Rxx is their average.
        snapshots_per_rxx = num_tx * num_stream_slides * num_freq_slides
        total_snapshots = snapshots_per_rxx * num_time_slides
        sv_len = stream_win * freq_win_points * time_win
        Rxx = np.zeros((sv_len, sv_len), dtype=np.complex128)

        for time_start in time_starts:
            X_r = np.empty((sv_len, snapshots_per_rxx), dtype=np.complex128)
            idx = 0
            for tx in range(num_tx):
                for stream_start in stream_starts:
                    for freq_start in range(num_freq_slides):
                        subcarrier_indices = freq_start + freq_offsets
                        block = csi_segment[
                            time_start:(time_start + time_win),
                            tx,
                            stream_start:(stream_start + stream_win),
                            :,
                        ][:, :, subcarrier_indices]

                        # (time, stream, frequency) -> (stream, frequency, time)
                        v = block.transpose(1, 2, 0).reshape(-1)
                        X_r[:, idx] = v
                        idx += 1

            assert idx == snapshots_per_rxx, (
                f"Azi-ToF-Dop snapshot count mismatch: idx={idx}, "
                f"expected={snapshots_per_rxx}"
            )
            Rxx += (X_r @ X_r.conj().T) / snapshots_per_rxx

        Rxx /= num_time_slides
        Rxx = (Rxx + Rxx.conj().T) / 2.0

        print(
            f"Azi-ToF-Dop Rxx: {Rxx.shape}, averaged_Rxx={num_time_slides}, "
            f"snapshots_per_Rxx={snapshots_per_rxx}, total_snapshots={total_snapshots}, "
            f"context={start}:{end}, stream_slides={num_stream_slides}, "
            f"subc_slides={num_freq_slides}"
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
                "data_axes": ("azi", "tof"),
                "x_axis": "azi",
                "y_axis": "tof",
                "projected_axis": "doppler",
            },
            {
                "title": "ToF-Doppler",
                "axis0_values": tau,
                "axis1_values": fd,
                "heatmap_db": tof_dop_db,
                "data_axes": ("tof", "doppler"),
                "x_axis": "doppler",
                "y_axis": "tof",
                "projected_axis": "azi",
            },
            {
                "title": "Azi-Doppler",
                "axis0_values": theta,
                "axis1_values": fd,
                "heatmap_db": azi_dop_db,
                "data_axes": ("azi", "doppler"),
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
            x_values = panel["axis0_values"]
            y_values = panel["axis1_values"]
            P_music = panel["heatmap_db"].T
            if panel["data_axes"] == (panel["y_axis"], panel["x_axis"]):
                x_values, y_values = y_values, x_values
                P_music = panel["heatmap_db"]
            Plot.plot_spectrum(
                frame_idx,
                x_values,
                y_values,
                P_music,
                self.args,
                title=panel["title"],
                sdim=Sdim,
                spectrum_axes=(panel["x_axis"], panel["y_axis"]),
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
        All target RoI masks are also combined with logical OR, then the union
        is summed once and saved as one additional Azi-ToF map.
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
        combined_fd_mask = np.zeros(fd_grid.shape, dtype=bool)
        if fig_name is None:
            name = "Doppler_RoI"
        else:
            name = os.path.splitext(os.path.basename(str(fig_name)))[0]

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

            combined_fd_mask |= fd_mask
            selected_fd = fd_grid[fd_mask]
            azi_tof = np.sum(P_music[:, :, fd_mask], axis=2)
            azi_tof_db = 10.0 * np.log10(np.maximum(azi_tof, 1e-12))

            rank = int(target["rank"])
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
                azi_tof_db.T,
                self.args,
                title=title,
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

        if not np.any(combined_fd_mask):
            print("No Doppler bins available in the target RoI union.")
            return results

        # Sum the union of all target RoIs once. Overlapping Doppler bins are
        # represented by one True entry and therefore cannot be double-counted.
        spectrum_linear_sum = np.sum(
            P_music[:, :, combined_fd_mask],
            axis=2,
        )
        spectrum_sum_db = 10.0 * np.log10(
            np.maximum(spectrum_linear_sum, 1e-12)
        )
        sum_file_name = f"{frame_idx:04d}_{name}_sum.png"
        Plot.plot_spectrum(
            frame_idx,
            theta,
            tau,
            spectrum_sum_db.T,
            self.args,
            title="Azimuth-ToF Sum over Target Doppler-RoI Union",
            sdim=Sdim,
            spectrum_axes=("azi", "tof"),
            file_name=sum_file_name,
        )

        return results
