import numpy as np
import os
from tqdm import tqdm
import pre_processing as pp
import Plot
import matplotlib.pyplot as plt
import weight_generation
import point_cloud
from scipy.signal import peak_widths



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
                    "noise_power_db": float(10.0 * np.log10(max(noise_power, eps))),
                    "threshold_db": float(10.0 * np.log10(max(threshold, eps))),
                    "snr_db": float(10.0 * np.log10(max(cut_power, eps) / max(noise_power, eps))),
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

class Azi_AoD_ToF:
    """Amplitude-domain AoA-AoD-ToF pseudo-spectrum with two Tx per snapshot."""

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = min(int(args.stream_sample_range), int(args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = int(args.freq_hop)
        self.freq_space = int(args.freq_space)
        self.avg_frames = max(1, int(args.avg_frames))
        self.tau_chunk = max(1, int(getattr(args, "azi_tof_tau_chunk", 10)))
        self.theta_grid = np.arange(
            args.theta_min, args.theta_max + 0.5 * args.theta_step, args.theta_step
        )
        self.aod_grid = self.theta_grid.copy()
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        tx_spacing = getattr(args, "tx_antenna_spacing", None)
        self.tx_antenna_spacing = float(
            args.antenna_spacing if tx_spacing is None else tx_spacing
        )

        if int(args.num_Tx) != 2:
            raise ValueError("Azi_AoD_ToF requires exactly two Tx antennas")
        if not 0 < self.stream_win <= self.stream_sample_range:
            raise ValueError("stream_win must be within the available Rx aperture")
        if self.freq_hop < 1 or self.freq_space < 1 or self.tx_antenna_spacing <= 0:
            raise ValueError("freq_hop, freq_space, and Tx spacing must be positive")
        self.freq_win_points = self.freq_win // self.freq_hop
        if self.freq_win_points < 1:
            raise ValueError("freq_win // freq_hop must be positive")
        self.freq_offsets = np.arange(self.freq_win_points) * self.freq_hop
        self.freq_limit = min(int(args.freq_sample_range), int(args.num_sc))
        self.num_freq_samples = len(range(0, self.freq_limit, self.freq_space))
        if self.freq_offsets[-1] >= self.num_freq_samples:
            raise ValueError("Frequency aperture exceeds the selected subcarriers")
        if not self.theta_grid.size or not self.tau_grid.size:
            raise ValueError("AoA/AoD and ToF grids must be nonempty")

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.avg_frames, total_frames)
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        return CSI[start:start + context_len], start, start + context_len

    def Rxx_smooth(self, CSI, frame_idx):
        """Smooth over Rx/frequency starts; never slide or average the Tx axis."""
        csi = np.asarray(CSI)
        if csi.ndim != 4 or csi.shape[1] != 2:
            raise ValueError("CSI must have shape (frame, 2 Tx, Rx, subcarrier)")
        if np.iscomplexobj(csi):
            raise ValueError("Azi_AoD_ToF expects real CSI amplitude or its signed residual")
        if csi.shape[0] < 1 or csi.shape[2] < self.stream_sample_range:
            raise ValueError("CSI has too few frames or Rx channels")
        if csi.shape[3] < self.freq_limit or not np.all(np.isfinite(csi)):
            raise ValueError("CSI has too few subcarriers or contains non-finite values")
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < csi.shape[0]:
            raise ValueError("frame_idx is outside CSI")

        segment, _, _ = self.sample_csi_segment(csi, frame_idx)
        segment = segment[:, :, :self.stream_sample_range, :self.freq_limit:self.freq_space]
        num_rx_slides = self.stream_sample_range - self.stream_win + 1
        num_freq_slides = self.num_freq_samples - int(self.freq_offsets[-1])
        snapshot_dim = 2 * self.stream_win * self.freq_win_points
        snapshot_count = segment.shape[0] * num_rx_slides * num_freq_slides
        X = np.empty((snapshot_dim, snapshot_count), dtype=np.complex128)

        idx = 0
        for t in range(segment.shape[0]):
            for rx_start in range(num_rx_slides):
                for freq_start in range(num_freq_slides):
                    block = segment[t, :, rx_start:rx_start + self.stream_win, :]
                    X[:, idx] = block[:, :, freq_start + self.freq_offsets].reshape(-1)
                    idx += 1
        Rxx = (X @ X.conj().T) / snapshot_count
        return (Rxx + Rxx.conj().T) / 2.0

    def _aod_steering(self):
        theta = np.deg2rad(self.aod_grid)
        if self.args.projection == "sin":
            projected = np.sin(theta)
        elif self.args.projection == "cos":
            projected = 1.0 - np.cos(theta)
        else:
            raise ValueError(f"Unsupported projection: {self.args.projection}")
        phase = -2j * np.pi * self.args.f_0 * self.tx_antenna_spacing / 3e8
        return np.exp(phase * projected[:, None] * np.arange(2)[None, :])

    def steering_matrix_chunk(self, tau_chunk):
        """Unit-norm AoA-ToF steering in Rx-major, subcarrier-minor order."""
        rx = np.stack([
            self.steering_vector.steering_vector_AoA(theta, self.stream_win)
            for theta in self.theta_grid
        ])
        subcarrier_offsets = self.freq_space * self.freq_offsets
        tof = np.exp(
            -2j * np.pi *
            (self.args.f_0 + self.args.delta_f * subcarrier_offsets[None, :]) *
            np.asarray(tau_chunk)[:, None]
        )
        return (rx[:, None, :, None] * tof[None, :, None, :]).reshape(
            len(self.theta_grid) * len(tau_chunk),
            self.stream_win * self.freq_win_points,
        ) / np.sqrt(self.stream_win * self.freq_win_points)

    def cal_spectrum(self, Rxx):
        """Return linear MUSIC values with axes (AoA, AoD, ToF)."""
        dim = 2 * self.stream_win * self.freq_win_points
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        if Rxx.shape != (dim, dim):
            raise ValueError(f"Expected Rxx shape {(dim, dim)}, got {Rxx.shape}")
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[order], eig_vec[:, order]
        sdim = resolve_Sdim(
            self.args, eig_val, label="AoA-AoD-ToF",
            sdim_attr="Azi_AoD_ToF_Sdim",
            energy_ratio_attr="Sdim_energy_ratio",
        )
        E_n = eig_vec[:, sdim:]
        rx_freq_dim = dim // 2
        tx_phase = self._aod_steering()[:, 1]
        cube = np.empty(
            (len(self.theta_grid), len(self.aod_grid), len(self.tau_grid)),
            dtype=float,
        )

        for start in range(0, len(self.tau_grid), self.tau_chunk):
            stop = min(start + self.tau_chunk, len(self.tau_grid))
            B = self.steering_matrix_chunk(self.tau_grid[start:stop])
            q0 = B.conj() @ E_n[:rx_freq_dim]
            q1 = B.conj() @ E_n[rx_freq_dim:]
            base = np.sum(np.abs(q0) ** 2 + np.abs(q1) ** 2, axis=1).real
            cross = np.sum(q0 * q1.conj(), axis=1)
            denominator = (base[None, :] + 2.0 * np.real(tx_phase[:, None] * cross[None, :])) / 2.0
            cube[:, :, start:stop] = (
                1.0 / np.maximum(denominator, 1e-12)
            ).reshape(len(self.aod_grid), len(self.theta_grid), stop - start).transpose(1, 0, 2)
        return self.tau_grid, self.theta_grid, self.aod_grid, cube, sdim

    def gen_spectrum(self, CSI, frame_idx, plot=True, file_name="Azi_AoD_ToF", title="AoD-summed Azi-ToF amplitude pseudo-spectrum"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, aod_grid, cube, sdim = self.cal_spectrum(Rxx)
        # Sum in the linear domain; summing dB values changes the spectrum.
        projected = np.sum(cube, axis=1)
        if plot:
            Plot.plot_spectrum(
                frame_idx, x_values=theta_grid, y_values=tau_grid,
                P_music=10.0 * np.log10(np.maximum(projected.T, 1e-12)),
                args=self.args, title=title, sdim=sdim,
                spectrum_axes=("azi", "tof"), file_name=file_name,
            )
        return tau_grid, theta_grid, projected, sdim

class AoD_ToF:
    """Amplitude-domain AoD-ToF pseudo-spectrum without frequency resampling."""

    def __init__(self, args):
        self.args = args
        self.avg_frames = max(1, int(args.avg_frames))
        self.num_rx = int(args.num_Rx)
        self.freq_win = int(args.freq_win)
        self.freq_hop = int(args.freq_hop)
        if self.freq_hop < 1:
            raise ValueError("freq_hop must be positive")
        self.freq_win_points = self.freq_win // self.freq_hop
        self.freq_limit = min(int(args.freq_sample_range), int(args.num_sc))
        self.theta_grid = np.arange(args.theta_min, args.theta_max + 0.5 * args.theta_step, args.theta_step)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.tau_chunk = max(1, int(getattr(args, "aod_tof_tau_chunk", 10)))
        tx_spacing = getattr(args, "tx_antenna_spacing", None)
        self.tx_antenna_spacing = float(args.antenna_spacing if tx_spacing is None else tx_spacing)

        if int(args.num_Tx) != 2:
            raise ValueError("AoD_ToF requires exactly two Tx antennas")
        if self.num_rx < 1 or self.freq_win_points < 1 or self.freq_limit < 1:
            raise ValueError("Rx count and frequency aperture must be positive")
        self.freq_offsets = np.arange(self.freq_win_points) * self.freq_hop
        if self.freq_offsets[-1] >= self.freq_limit:
            raise ValueError("Frequency aperture exceeds the selected subcarriers")
        if not np.isfinite(self.tx_antenna_spacing) or self.tx_antenna_spacing <= 0:
            raise ValueError("Tx antenna spacing must be positive and finite")
        if not self.theta_grid.size or not self.tau_grid.size:
            raise ValueError("AoD and ToF grids must be nonempty")

    def Rxx_smooth(self, CSI, frame_idx):
        """Slide only over frequency; keep both Tx in each Rx snapshot."""
        csi = np.asarray(CSI)
        if csi.ndim != 4 or csi.shape[1] != 2:
            raise ValueError("CSI must have shape (frame, 2 Tx, Rx, subcarrier)")
        if np.iscomplexobj(csi):
            raise ValueError("AoD_ToF expects real CSI amplitude or its signed residual")
        if csi.shape[0] < 1 or csi.shape[2] < self.num_rx or csi.shape[3] < self.freq_limit:
            raise ValueError("CSI has too few frames, Rx channels, or subcarriers")
        if not np.all(np.isfinite(csi)):
            raise ValueError("CSI contains non-finite values")
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < csi.shape[0]:
            raise ValueError("frame_idx is outside CSI")

        context_len = min(self.avg_frames, csi.shape[0])
        start = int(np.clip(frame_idx - context_len // 2, 0, csi.shape[0] - context_len))
        segment = csi[start:start + context_len, :, :self.num_rx, :self.freq_limit]
        num_freq_slides = self.freq_limit - int(self.freq_offsets[-1])
        snapshot_count = context_len * self.num_rx * num_freq_slides
        dim = 2 * self.freq_win_points
        X = np.empty((dim, snapshot_count), dtype=np.complex128)
        idx = 0
        for t in range(context_len):
            for rx in range(self.num_rx):
                for freq_start in range(num_freq_slides):
                    # Tx-major, subcarrier-minor ordering matches a_AoD ⊗ a_ToF.
                    block = segment[t, :, rx, :]
                    X[:, idx] = block[:, freq_start + self.freq_offsets].reshape(-1)
                    idx += 1
        Rxx = (X @ X.conj().T) / snapshot_count
        return (Rxx + Rxx.conj().T) / 2.0

    def steering_matrix_chunk(self, tau_chunk):
        """Unit-norm AoD-ToF rows in Tx-major, subcarrier-minor order."""
        theta = np.deg2rad(self.theta_grid)
        if self.args.projection == "sin":
            projected = np.sin(theta)
        elif self.args.projection == "cos":
            projected = 1.0 - np.cos(theta)
        else:
            raise ValueError(f"Unsupported projection: {self.args.projection}")
        tx_phase = -2j * np.pi * self.args.f_0 * self.tx_antenna_spacing / 3e8
        aod = np.exp(tx_phase * projected[:, None] * np.arange(2)[None, :])
        tof = np.exp(
            -2j * np.pi *
            (self.args.f_0 + self.args.delta_f * self.freq_offsets[None, :]) *
            np.asarray(tau_chunk)[:, None]
        )
        return (aod[:, None, :, None] * tof[None, :, None, :]).reshape(
            len(self.theta_grid) * len(tau_chunk), 2 * self.freq_win_points
        ) / np.sqrt(2 * self.freq_win_points)

    def cal_spectrum(self, Rxx):
        """Return linear MUSIC values with axes (AoD, ToF)."""
        dim = 2 * self.freq_win_points
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        if Rxx.shape != (dim, dim):
            raise ValueError(f"Expected Rxx shape {(dim, dim)}, got {Rxx.shape}")
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[order], eig_vec[:, order]
        sdim = resolve_Sdim(
            self.args, eig_val, label="AoD-ToF",
            sdim_attr="AoD_ToF_Sdim", energy_ratio_attr="Sdim_energy_ratio",
        )
        E_n = eig_vec[:, sdim:]
        spectrum = np.empty((len(self.theta_grid), len(self.tau_grid)), dtype=float)
        for start in range(0, len(self.tau_grid), self.tau_chunk):
            stop = min(start + self.tau_chunk, len(self.tau_grid))
            steering = self.steering_matrix_chunk(self.tau_grid[start:stop])
            projection = steering.conj() @ E_n
            denominator = np.sum(np.abs(projection) ** 2, axis=1).real
            spectrum[:, start:stop] = (1.0 / np.maximum(denominator, 1e-12)).reshape(len(self.theta_grid), stop - start)
        return self.tau_grid, self.theta_grid, spectrum, sdim

    def gen_spectrum(self, CSI, frame_idx, plot=True, file_name="AoD_ToF", title="AoD-ToF-spectrum"):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, aod_grid, spectrum, sdim = self.cal_spectrum(Rxx)
        if plot:
            Plot.plot_spectrum(
                frame_idx, x_values=aod_grid, y_values=tau_grid,
                P_music=10.0 * np.log10(np.maximum(spectrum.T, 1e-12)),
                args=self.args, title=title, sdim=sdim,
                spectrum_axes=("aod", "tof"), file_name=file_name,
            )
        return tau_grid, aod_grid, spectrum, sdim

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
            Sdim = resolve_Sdim(self.args, eig_val, label=label)
        else:
            Sdim = int(np.clip(self.Sdim, 1, expected_dim - 1))
        E_s = eig_vec[:, Sdim:]

        P_azi_tof = np.empty((len(self.theta_grid), len(self.tau_grid)),dtype=float)
        for start in range(0, len(self.tau_grid), self.tau_chunk):

            end = min(start + self.tau_chunk, len(self.tau_grid))
            tau_chunk = self.tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)

            projection = SV_chunk.conj() @ E_s
            signal_projection = np.sum(np.abs(projection) ** 2,axis=1).real
            signal_projection = np.clip(signal_projection, 0.0, 1.0)
            denominator = np.maximum(np.sum(np.abs(projection) ** 2, axis=1).real, 1e-12)
            P_azi_tof[:, start:end] = (1.0 / denominator).reshape(len(self.theta_grid),len(tau_chunk))

        return self.tau_grid, self.theta_grid, P_azi_tof, Sdim

    def gen_spectrum(self, CSI, frame_idx, plot=True, *, sdim_attr=None, energy_ratio_attr=None, label="Azi-ToF"):

        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = self.cal_spectrum(
            Rxx,
            sdim_attr=sdim_attr,
            energy_ratio_attr=energy_ratio_attr,
            label=label,
        )

        if plot:
            Plot.plot_spectrum(
                frame_idx,
                x_values= theta_grid,
                y_values = tau_grid,
                P_music = 10.0 * np.log10(np.maximum(P_azi_tof.T, 1e-12)),
                args = self.args,
                title = "Azimuth-ToF",
                sdim = Sdim,
                spectrum_axes=("azi", "tof"),
                file_name=None,
            )
        else:
            return tau_grid, theta_grid, P_azi_tof, Sdim

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

    def cal_spectrum(self, Rxx, sdim_attr=None, energy_ratio_attr=None, label="ToF-Doppler",):
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

        P_tof_dop = np.empty((len(tau_grid), len(fd_grid)), dtype=float)

        for start in range(0, len(tau_grid), self.tau_chunk):
            end = min(start + self.tau_chunk, len(tau_grid))
            tau_chunk = tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)

            A = SV_chunk.conj() @ E_s
            aEEa = np.real(np.sum(A * np.conj(A), axis=1))
            aEEa = np.clip(aEEa, -1.0, 1.0)
            P_tof_dop[start:end] = (1.0 / (1.0 - aEEa + self.epsilon)).reshape(len(tau_chunk), len(fd_grid))

        return tau_grid, fd_grid, P_tof_dop, Sdim


    def gen_spectrum(self, CSI, frame_idx, plot=True):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(Rxx)
        if plot:
            Plot.plot_spectrum(
                frame_idx,
                x_values = fd_grid,
                y_values = tau_grid,
                P_music = 10.0 * np.log10(np.maximum(P_tof_dop, 1e-12)),
                args = self.args,
                title="ToF-Doppler",
                sdim=Sdim,
                spectrum_axes=("doppler", "tof"),
                file_name=None,
            )
        else:
            return tau_grid, fd_grid, P_tof_dop, Sdim

    def gen_beamformed_spectrum(self, CSI, frame_idx, title=None, file_name=None, plot=True):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(
            Rxx,
            sdim_attr="Beamformed_ToF_Dop_Sdim",
            energy_ratio_attr="Beamformed_ToF_Dop_Sdim_ratio",
            label="Beamformed ToF-Doppler",
        )
        if plot:
            Plot.plot_spectrum(
                frame_idx,
                x_values = fd_grid,
                y_values = tau_grid,
                P_music = 10.0 * np.log10(np.maximum(P_tof_dop, 1e-12)),
                args = self.args,
                title=title,
                sdim=Sdim,
                spectrum_axes=("doppler", "tof"),
                file_name=file_name,
            )
        else:
            return tau_grid, fd_grid, P_tof_dop, Sdim

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
        """
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
        """
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

class DopplerProject_Azi_ToF:
    def __init__(self, args, raw_CSI=None):
        self.args = args
        self.raw_CSI = raw_CSI
        self.steering_vector = SteeringVector(args)
        self.num_Rx = args.num_Rx
        self.Sdim = getattr(args, "ToF_Dop_Sdim", getattr(args, "Sdim", None))

        self.training_cells = args.cfar_training_cells
        self.guard_cells = args.cfar_guard_cells
        self.threshold_factor = args.cfar_threshold_factor
        self.top_k = args.cfar_top_k
        self.min_peak_distance = args.cfar_min_peak_distance

        self.theta_grid = np.arange(
            args.theta_min,
            args.theta_max + 0.5 * args.theta_step,
            args.theta_step,
        )
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.tof_gate = float(getattr(args, "tof_gate", 10e-9))
        if not np.isfinite(self.tof_gate) or self.tof_gate < 0:
            raise ValueError("tof_gate must be finite and non-negative")
    
    def beamform_azimuthX(self, CSI, P_azi_tof, plot=False):
        """Estimate 1-D azimuth from Azi-ToF and form one coherent beam.

        ``P_azi_tof`` has shape (theta, ToF).
        Only ToF bins beyond 3e-9 seconds contribute to the
        1-D azimuth profile. Its strongest direction selects one rank-1
        spatial projection.

        The returned CSI keeps shape (frame, Tx, Rx, subcarrier).
        When plot=True, draw the normalized array response of that projection.
        """
        csi = np.asarray(CSI)

        theta_grid = self.theta_grid
        tof_mask = self.tau_grid > 3e-9
        if not np.any(tof_mask):
            raise ValueError("Azimuth beamforming requires ToF bins beyond 3e-9 s")
        P_azi = np.max(P_azi_tof[:, tof_mask], axis=1)


        peak_idx = int(np.argmax(P_azi))
        beam_theta = float(theta_grid[peak_idx])
        steering = self.steering_vector.steering_vector_AoA(beam_theta, stream_win=self.num_Rx).reshape(-1, 1)
        norm_sq = float(np.vdot(steering, steering).real)
        projection_matrix = (steering @ steering.conj().T) / max(norm_sq, 1e-12)
        CSI_beamformed = np.tensordot(csi, projection_matrix, axes=([2], [1]))
        CSI_beamformed = np.transpose(CSI_beamformed, (0, 1, 3, 2))

        self.last_azimuth_grid = theta_grid
        self.last_azimuth_profile = P_azi
        self.last_azimuth_weight_matrix = projection_matrix
        self.last_beamform_theta = beam_theta

        if plot:
            scan_theta = np.linspace(theta_grid[0], theta_grid[-1], max(361, theta_grid.size))
            scan_steering = np.stack([
                self.steering_vector.steering_vector_AoA(theta, stream_win=self.num_Rx)
                for theta in scan_theta
            ])
            gain_linear = np.abs(scan_steering @ steering.conj().ravel()) ** 2
            gain_linear /= norm_sq * np.sum(np.abs(scan_steering) ** 2, axis=1)
            gain_db = 10.0 * np.log10(np.clip(gain_linear, 1e-6, 1.0))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(scan_theta, gain_db, label="Beam pattern")
            ax.axvline(beam_theta, color="black", linestyle="--", label=f"Selected {beam_theta:g}°")
            ax.set(xlabel="Azimuth (deg)", ylabel="Normalized gain (dB)",
                   title="Azimuth beam pattern", xlim=(scan_theta[0], scan_theta[-1]),
                   ylim=(-60, 1))
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()

            if self.args.pics_dir is not None:
                save_dir = os.path.join(self.args.pics_dir, "Beamform_Pattern")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"beam_pattern_{beam_theta:+.2f}deg.png")
                fig.savefig(save_path, dpi=120)
                plt.close(fig)
                print(f"Saved: {save_path}")
            else:
                plt.show(block=False)
        return CSI_beamformed, beam_theta
    
    def doppler_projectionX(
        self, CSI, frame_idx, target_fd, args, window="hann", normalize=False,
        *, candidate_fds=None, weight_center_fd=None, weight_cache=None,
    ):
        """Project every real dynamic CSI amplitude channel onto one signed Doppler.

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
            Real-valued dynamic-amplitude residual. This is not complex CSI
            reconstruction.
        target_fd : float
            Signed target Doppler in Hz, e.g. ``target['fd']``.
        frame_idx : int
            Frame about which the projection context is selected.
        avg_frames : int
            Requested number of projection frames.
        fs : float
            CSI frame sampling rate in Hz.
        window : {'hann', 'hamming', 'rect', 'weight', 'weight_amp'}, default='hann'
            Temporal weight ``h[n]``. ``'rect'`` uses equal weights;
            ``'weight'`` fits power coefficients, while ``'weight_amp'`` fits
            the projected amplitude residual; both solve a target-specific QP.
        candidate_fds : array_like, optional
            Detected path Dopplers. Real-signal mirror pairs are fitted once.
        weight_center_fd : float, optional
            Doppler matched by the QP; defaults to ``target_fd``. Neighboring
            projection frequencies may share this target's temporal weight.
        weight_cache : dict, optional
            Per-call cache shared by projections of the same target set.
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
        elif window == "hamming":
            weights = np.hamming(context_len)
        elif window == "rect":
            weights = np.ones(context_len)

        elif window in ("weight", "weight_amp"):
            if window == "weight" and (self.raw_CSI is None or np.shape(self.raw_CSI) != np.shape(CSI)):
                raise ValueError("window='weight' requires aligned raw_CSI in the constructor")
            center_fd = float(target_fd if weight_center_fd is None else weight_center_fd)
            if not np.isfinite(center_fd) or abs(center_fd) < 1e-12:
                raise ValueError(f"window={window!r} requires a nonzero finite centre Doppler")
            if weight_cache is None:
                weight_cache = {}
            paths = np.asarray([center_fd] if candidate_fds is None else candidate_fds, dtype=float)
            if paths.ndim != 1 or not np.all(np.isfinite(paths)):
                raise ValueError("candidate_fds must be a finite 1-D array")
            positive_fds = []
            for fd in np.r_[paths, center_fd]:
                if abs(fd) < 1e-12:  # DC is fitted separately from Doppler paths.
                    continue
                if not any(np.isclose(abs(fd), known, rtol=1e-9, atol=1e-9) for known in positive_fds):
                    positive_fds.append(abs(fd))
            positive_fds = np.asarray(positive_fds, dtype=float)
            fit_key = ("fit_amp" if window == "weight_amp" else "fit", start, end, tuple(positive_fds))
            if fit_key not in weight_cache:
                if window == "weight_amp":
                    # Fit exactly the real amplitude residual being projected.
                    weight_cache[fit_key] = weight_generation.estimate_sinr_parameters(
                        segment, positive_fds, fs, return_offset=True
                    )
                else:
                    power_CSI = np.abs(self.raw_CSI) ** 2
                    if args.preprocess == "ma":
                        power_CSI -= pp.MA(power_CSI, args.avg_frames)
                    elif args.preprocess == "dwt":
                        power_CSI = pp.DWT_components(
                            power_CSI, target_labels=["", "D5", "D4", "D3", "D2", ""]
                        )
                    elif args.preprocess == "pca":
                        power_CSI -= pp.MA(power_CSI, args.avg_frames)
                        power_CSI = pp.PCA_time(power_CSI, args.fs * 0.5, k=3)
                    weight_cache[fit_key] = weight_generation.estimate_sinr_parameters(
                        power_CSI[start:end], positive_fds, fs
                    )
            if window == "weight_amp":
                coefficients, noise_variance, offset = weight_cache[fit_key]
            else:
                coefficients, noise_variance = weight_cache[fit_key]

            window_key = ("window", fit_key, center_fd)
            if window_key not in weight_cache:
                model_fds = positive_fds.copy()
                path_idx = int(np.flatnonzero(np.isclose(
                    model_fds, abs(center_fd), rtol=1e-9, atol=1e-9
                ))[0])
                if center_fd < 0:
                    model_fds[path_idx] = center_fd
                    model_coefficients = coefficients.copy()
                    model_coefficients[path_idx] = coefficients[path_idx].conj()
                else:
                    model_coefficients = coefficients
                weight_cache[window_key] = weight_generation.weight_gen(
                    center_fd, model_fds, model_coefficients, noise_variance,
                    context_len, fs, **({"offset": offset} if window == "weight_amp" else {}),
                )
            weights = weight_cache[window_key]
        else:
            raise ValueError(f"window must be 'hann', 'hamming', 'rect', 'weight', or 'weight_amp', got {window!r}")

        # Absolute sample time preserves a consistent coefficient phase when the
        # projection centre changes.  target_fd keeps its sign here.
        times_s = np.arange(start, end, dtype=float) / fs
        demodulator = weights * np.exp(-1j * 2.0 * np.pi * target_fd * times_s)
        Z_fd = np.tensordot(demodulator, segment, axes=(0, 0))

        if normalize:
            Z_fd = Z_fd / np.sum(weights)
        return Z_fd
    
    def tof_gating_checkX(self, Z, tau, gating_range):
        """Check whether the strongest Azi-ToF peak matches a Stage 1 ToF.

        ``Z`` has shape (azimuth, ToF). ``tau`` and ``gating_range`` are in
        seconds. The peak is found over the full map
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

    def gen_beamform_ToF_Dop(self, CSI, frame_idx, file_name=None):
        azi_tof = Azi_ToF(self.args)
        tof_dop = ToF_Dop(self.args)
        # 1. Dectect target (fd, tau) from Azimuth beamformed ToF-Doppler Map
        ## (1) Generate Azi-ToF spectrum 
        _, _, P_azi_tof, _ = azi_tof.gen_spectrum(CSI, frame_idx, plot=False)

        ## (2) Beamforming Azimuth ToF-Doppler Map
        CSI_bf, beam_theta = self.beamform_azimuthX(CSI, P_azi_tof)
        tof_dop.gen_beamformed_spectrum(
            CSI_bf,
            frame_idx,
            plot=True,
            title=f"ToF-Doppler Beamform {beam_theta:g} deg",
            file_name=file_name,
        )


    def detect_beamform_targets(self, beamform_result, fd_step, *, select=True):
        """Detect candidate-beam RD peaks; select center targets if requested."""
        tau_grid = np.asarray(beamform_result["tau_grid"])
        fd_grid = np.asarray(beamform_result["fd_grid"])
        beam_maps = np.asarray(beamform_result["maps"])
        if beam_maps.ndim != 3 or beam_maps.shape[1:] != (len(tau_grid), len(fd_grid)):
            raise ValueError("beamform_result maps must have shape (beam, ToF, Doppler)")
        if len(beamform_result["theta_candidates"]) != beam_maps.shape[0]:
            raise ValueError("beamform_result angles must match its RD maps")
        min_snr_db = float(getattr(self.args, "beamform_candidate_min_snr_db", 0.5))
        if not np.isfinite(min_snr_db):
            raise ValueError("beamform_candidate_min_snr_db must be finite")
        beam_peaks = []
        for beam_idx, beam_map in enumerate(beam_maps):
            peaks = cfar_2D(
                beam_map, tau_grid, fd_grid,
                training_cells=self.training_cells,
                guard_cells=self.guard_cells,
                threshold_factor=self.threshold_factor,
                top_k=None,
                min_peak_distance=self.min_peak_distance,
            )
            beam_peaks.extend(
                {**peak, "beam_idx": beam_idx}
                for peak in peaks if peak["snr_db"] >= min_snr_db
            )
        beam_peaks.sort(key=lambda peak: peak["power"], reverse=True)
        if not select:
            return beam_peaks
        detected_targets = []
        for peak in beam_peaks:
            if any(
                abs(peak["fd"] - chosen["fd"]) <= fd_step + 1e-12
                and abs(peak["tau"] - chosen["tau"]) <= max(self.args.tau_step, 0.6e-9) + 1e-12
                for chosen in detected_targets
            ):
                continue
            detected_targets.append(peak)
            if self.top_k is not None and len(detected_targets) >= self.top_k:
                break
        for rank, peak in enumerate(detected_targets, start=1):
            peak["rank"] = rank
        return detected_targets

    def gen_Doppler_projected_spectrum(
        self, CSI, frame_idx, fd_neighbor, fd_step, window="hann", *,
        beamform_result=None, fuse=True, stable_targets="auto"
    ):
        """Project amplitude CSI with fixed or target-specific QP windows.

        ``CSI`` remains the original real amplitude residual for target detection
        and projection. With ``window='weight'``, the constructor's ``raw_CSI``
        supplies a separate linear-power residual used only to estimate C and noise.
        With ``window='weight_amp'``, C, noise, and DC offset are fitted from
        ``CSI`` itself. Both QPs match each target's centre Doppler and reuse
        one shared weight for that target's neighboring band frequencies.
        With ``beamform_result``, detect peaks directly in each candidate beam's
        RD map and estimate each Doppler band from that beam's half-prominence
        width. Duplicate peaks across beams are merged.
        ``fd_neighbor=None`` then uses one Doppler bin as the minimum half-width.
        With ``fuse=False``, keep each frequency that passes the ToF gate as
        a separate Azi-ToF result and plot instead of summing its band.
        With ``beamform_result``, stability is checked on each candidate beam
        by default. A supplied ``stable_targets`` list reuses an earlier check;
        ``stable_targets=None`` explicitly skips it.
        """
        if window not in ("rect", "hann", "hamming", "weight", "weight_amp"):
            raise ValueError("window must be 'rect', 'hann', 'hamming', 'weight', or 'weight_amp'")
        fd_step = float(fd_step)
        if not np.isfinite(fd_step) or fd_step <= 0.0:
            raise ValueError("doppler_step must be positive and finite")
        azi_tof = Azi_ToF(self.args)
        if beamform_result is None:
            if stable_targets is not None and stable_targets != "auto":
                raise ValueError("stable_targets requires beamform_result")
            # Preserve the original single-angle diagnostic path.
            tof_dop = ToF_Dop(self.args)
            _, _, P_azi_tof, _ = azi_tof.gen_spectrum(CSI, frame_idx, plot=False)
            CSI_bf, beam_theta = self.beamform_azimuthX(CSI, P_azi_tof)
            tau_grid, fd_grid, P_tof_dop, Sdim = tof_dop.gen_beamformed_spectrum(
                CSI_bf, frame_idx, plot=False
            )
            detected_targets = cfar_2D(
                P_tof_dop,
                tau_grid,
                fd_grid,
                training_cells=self.training_cells,
                guard_cells=self.guard_cells,
                threshold_factor=self.threshold_factor,
                top_k=self.top_k,
                min_peak_distance=self.min_peak_distance,
            )
            target_label = np.asarray(
                [(target["tau"], target["fd"]) for target in detected_targets],
                dtype=float,
            ).reshape(-1, 2)
            Plot.plot_spectrum(
                frame_idx,
                x_values=fd_grid,
                y_values=tau_grid,
                P_music=10.0 * np.log10(np.maximum(P_tof_dop, 1e-12)),
                args=self.args,
                title=f"ToF-Doppler Beamform {beam_theta:g} deg",
                sdim=Sdim,
                spectrum_axes=("doppler", "tof"),
                target_label=target_label,
                file_name="label_beamform",
            )
        else:
            if int(beamform_result["frame_idx"]) != int(frame_idx):
                raise ValueError("beamform_result must come from the requested frame")
            if isinstance(stable_targets, str):
                if stable_targets != "auto":
                    raise ValueError("stable_targets must be 'auto', None, or a tracked target list")
                from stability_check import RDStabilityCheck

                stable_targets = RDStabilityCheck(self.args).stable_targets_from_RDmap(
                    CSI, frame_idx, observation_interval=self.args.fs * 0.1,
                    center_result=beamform_result, peak_source="candidate", fd_step=fd_step,
                )
            tau_grid = np.asarray(beamform_result["tau_grid"])
            fd_grid = np.asarray(beamform_result["fd_grid"])
            beam_maps = np.asarray(beamform_result["maps"])
            detected_targets = self.detect_beamform_targets(beamform_result, fd_step)
            if stable_targets is not None:
                stable_by_key = {
                    (int(target["beam_idx"]), int(target["tau_idx"]), int(target["fd_idx"])): target
                    for target in stable_targets
                }
                stable_candidates = []
                for target in detected_targets:
                    identity = (target["beam_idx"], target["tau_idx"], target["fd_idx"])
                    if identity not in stable_by_key:
                        continue
                    tracked = stable_by_key[identity]
                    stable_candidates.append({
                        **target,
                        **{key: tracked[key] for key in
                           ("hit_count", "observation_count", "required_hits", "stability_ratio")
                           if key in tracked},
                    })
                detected_targets = stable_candidates
                print(f"RD stability: {len(detected_targets)} candidate targets passed")

        if not detected_targets:
            print("No targets available for Azi-ToF Doppler-band projection.")
            return []
        if fd_neighbor is None:
            if beamform_result is None:
                raise ValueError("fd_neighbor=None requires beamform_result")
            min_half_width = fd_step
        else:
            if isinstance(fd_neighbor, (bool, np.bool_)):
                raise ValueError("fd_neighbor must be non-negative and finite")
            min_half_width = float(fd_neighbor)
            if not np.isfinite(min_half_width) or min_half_width < 0.0:
                raise ValueError("fd_neighbor must be non-negative and finite")
        max_half_width = float(getattr(self.args, "beamform_fd_max_halfwidth_hz", 3.0))
        if not np.isfinite(max_half_width) or max_half_width <= 0.0:
            raise ValueError("beamform_fd_max_halfwidth_hz must be positive and finite")
        max_half_width = max(max_half_width, min_half_width)
        neighbor_bin_count = int(np.floor(min_half_width / fd_step + 1e-12))

        projection_bins = {}
        target_bands = []
        for target in detected_targets:
            center_fd = float(target["fd"])
            band_members = []
            if beamform_result is None:
                requested_fds = [
                    center_fd + offset * fd_step
                    for offset in range(-neighbor_bin_count, neighbor_bin_count + 1)
                ]
            else:
                tau_idx, fd_idx = int(target["tau_idx"]), int(target["fd_idx"])
                beam_idx = int(target["beam_idx"])
                profile_db = 10.0 * np.log10(np.maximum(beam_maps[beam_idx, tau_idx], 1e-12))
                left, right = center_fd - min_half_width, center_fd + min_half_width
                if (0 < fd_idx < len(fd_grid) - 1
                        and profile_db[fd_idx] > profile_db[fd_idx - 1]
                        and profile_db[fd_idx] > profile_db[fd_idx + 1]):
                    width = peak_widths(profile_db, [fd_idx], rel_height=0.5)
                    if np.all(np.isfinite(width[2:4])) and width[0][0] > 0:
                        left = min(left, float(np.interp(
                            width[2][0], np.arange(len(fd_grid)), fd_grid
                        )))
                        right = max(right, float(np.interp(
                            width[3][0], np.arange(len(fd_grid)), fd_grid
                        )))
                # Broad weak peaks can merge into a stronger neighbor. Bound
                # their half-prominence width and stop at the peak midpoint.
                left = max(left, center_fd - max_half_width)
                right = min(right, center_fd + max_half_width)
                for other in detected_targets:
                    other_fd = float(other["fd"])
                    if other_fd < center_fd:
                        left = max(left, (center_fd + other_fd) / 2.0)
                    elif other_fd > center_fd:
                        right = min(right, (center_fd + other_fd) / 2.0)
                target["fd_band_hz"] = (left, right)
                target["beam_theta"] = float(beamform_result["theta_candidates"][beam_idx])
                requested_fds = np.arange(
                    self.args.doppler_min, self.args.doppler_max + fd_step / 2.0, fd_step
                )
                requested_fds = requested_fds[
                    (requested_fds >= left - 1e-12) & (requested_fds <= right + 1e-12)
                ]
                print(
                    f"⚡️ RD peak fd {center_fd:+.2f} Hz: beam "
                    f"{beamform_result['theta_candidates'][beam_idx]:g}°, "
                    f"Doppler band [{left:+.2f}, {right:+.2f}] Hz"
                    + (f", stability {target['hit_count']}/{target['observation_count']} frames"
                       if "hit_count" in target else "")
                )
            for requested_fd in requested_fds:
                if (requested_fd < self.args.doppler_min - 1e-12
                        or requested_fd > self.args.doppler_max + 1e-12):
                    continue
                fd_idx = int(np.rint((requested_fd - self.args.doppler_min) / fd_step))
                fd = float(self.args.doppler_min + fd_idx * fd_step)
                projection_bins.setdefault(fd_idx, fd)
                band_members.append(fd_idx)
            target_bands.append((target, band_members))

        print(
            f"Doppler projection: {len(detected_targets)} targets, "
            f"{len(projection_bins)} frequencies "
            f"(minimum half-width={min_half_width:g} Hz, step={fd_step:g} Hz)"
        )
        if not projection_bins:
            return []

        # Pass only detected centres; the weighted modes fit their respective
        # real signals and generate target-specific weights.
        center_fds = []
        for target, members in target_bands:
            if members:
                center_idx = min(members, key=lambda idx: abs(projection_bins[idx] - target["fd"]))
                center_fds.append(projection_bins[center_idx])
            else:
                center_fds.append(None)
        candidate_fds = [fd for fd in center_fds if fd is not None]
        weight_cache = {}

        def projected_spectrum(fd, center_fd=None):
            Z_fd = self.doppler_projectionX(
                CSI, frame_idx, fd, self.args, window=window,
                candidate_fds=candidate_fds,
                weight_center_fd=center_fd,
                weight_cache=weight_cache,
            )
            return azi_tof.gen_spectrum(
                Z_fd[np.newaxis, ...],
                frame_idx=frame_idx,
                plot=False,
                sdim_attr="Projected_Azi_ToF_Sdim",
                energy_ratio_attr="Projected_Azi_ToF_Sdim_ratio",
                label="Projected Azi-ToF",
            )

        # A weighted target owns its projections, even if another target uses
        # the same fd: the two target-specific windows can differ.
        projected_maps = {}
        if window in ("weight", "weight_amp"):
            for target_idx, (_, members) in enumerate(target_bands):
                for fd_idx in members:
                    tau_grid, theta_grid, spectrum_linear, _ = projected_spectrum(
                        projection_bins[fd_idx], center_fds[target_idx]
                    )
                    projected_maps[target_idx, fd_idx] = spectrum_linear
        else:
            for fd_idx, fd in projection_bins.items():
                tau_grid, theta_grid, spectrum_linear, _ = projected_spectrum(fd)
                projected_maps[fd_idx] = spectrum_linear

        # ToF Gating: check each frequency against each target's Stage 1 ToF.
        results = []
        maps = []
        for target_idx, (target, band_members) in enumerate(target_bands):
            center_fd = float(target["fd"])
            target_rank = int(target["rank"])
            map_key = lambda fd_idx: (target_idx, fd_idx) if window in ("weight", "weight_amp") else fd_idx
            valid_indices = [
                fd_idx for fd_idx in band_members
                if not getattr(self.args, "tof_gating", True)
                or self.tof_gating_checkX(projected_maps[map_key(fd_idx)], target["tau"], self.tof_gate)
            ]
            if not valid_indices:
                print(
                    f"⚠️ Target #{target_rank:02d} fd {center_fd:+.2f} Hz: "
                    f"no frequencies passed the {self.tof_gate * 1e9:.2f} ns ToF gate."
                )
                continue

            if not fuse:
                for fd_idx in valid_indices:
                    projection_fd = projection_bins[fd_idx]
                    spectrum_db = 10.0 * np.log10(
                        np.maximum(projected_maps[map_key(fd_idx)], 1e-12)
                    )
                    results.append(
                        {
                            "fd": center_fd,
                            "projection_fd": projection_fd,
                            "target_rank": target_rank,
                            "target_tau": float(target["tau"]),
                            "tau_grid": tau_grid,
                            "theta_grid": theta_grid,
                            "spectrum_db": spectrum_db,
                            "projection_fds": [projection_fd],
                            "fd_band_hz": target.get("fd_band_hz"),
                            "beam_theta": target.get("beam_theta"),
                            "hit_count": target.get("hit_count"),
                            "observation_count": target.get("observation_count"),
                            "required_hits": target.get("required_hits"),
                            "stability_ratio": target.get("stability_ratio"),
                        }
                    )
                    Plot.plot_spectrum(
                        frame_idx,
                        x_values=theta_grid,
                        y_values=tau_grid,
                        P_music=spectrum_db.T,
                        args=self.args,
                        title=(f"Target #{target_rank:02d} | peak {center_fd:+.2f} Hz"
                               f"\nProjection {projection_fd:+.2f} Hz"),
                        spectrum_axes=("azi", "tof"),
                        target_label=np.empty((0, 2)),
                        file_name=f"Project_target{target_rank:02d}_bin{fd_idx:03d}",
                    )
                continue

            if len(valid_indices) == 1:
                spectrum_linear_sum = projected_maps[map_key(valid_indices[0])]
            else:
                spectrum_linear_sum = np.sum([projected_maps[map_key(fd_idx)] for fd_idx in valid_indices], axis=0)
            spectrum_db = 10.0 * np.log10(np.maximum(spectrum_linear_sum, 1e-12))

            projection_fds = [projection_bins[fd_idx] for fd_idx in valid_indices]
            title = f"Target #{target_rank:02d} | fd {center_fd:+.2f} Hz"
            if len(valid_indices) == 1:
                title += "\nAzi-ToF projection"
            else:
                title += (
                    f"\nBand [{min(projection_fds):+.2f}, "
                    f"{max(projection_fds):+.2f}] Hz"
                )
            maps.append(
                {
                    "frame_idx": frame_idx,
                    "x_values": theta_grid,
                    "y_values": tau_grid,
                    "P_music": spectrum_db.T,  # plot_spectrum expects (ToF, azimuth).
                    "title": title,
                }
            )
            results.append(
                {
                    "fd": center_fd,
                    "tau_grid": tau_grid,
                    "theta_grid": theta_grid,
                    "spectrum_db": spectrum_db,
                    "projection_fds": projection_fds,
                    "fd_band_hz": target.get("fd_band_hz"),
                    "beam_theta": target.get("beam_theta"),
                    "hit_count": target.get("hit_count"),
                    "observation_count": target.get("observation_count"),
                    "required_hits": target.get("required_hits"),
                    "stability_ratio": target.get("stability_ratio"),
                }
            )

        if not results:
            print("No Doppler-projected target bands passed the ToF gate.")
            return results

        if fuse:
            Plot.plot_multi_spectrum(maps, self.args)
        return results

    def gen_Doppler_specific_spectrum(self, CSI, frame_idx, stable_targets, fd_neighbor, fd_step, window="hann", constrain_weight=False):
        """Plot one Doppler-projected Azi-ToF map per stable fused-RD peak.

        A shared temporal weight is applied to every Tx/Rx/subcarrier channel.
        By default it projects each target-band frequency with the selected
        window. With constrain_weight=True, it also nulls the target center's
        opposite-sign mirror and both signs of other stable targets when the
        constraints remain well-conditioned.
        """
        if window not in ("hann", "hamming", "rect"):
            raise ValueError("Doppler projection supports 'hann', 'hamming', or 'rect' windows")
        csi = np.asarray(CSI)
        if csi.ndim != 4 or np.iscomplexobj(csi) or csi.shape[0] == 0:
            raise ValueError("CSI must be real amplitude with shape (frame, Tx, Rx, subcarrier)")
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < csi.shape[0]:
            raise ValueError("frame_idx is outside CSI")
        if not isinstance(stable_targets, list):
            raise ValueError("stable_targets must be a list from RDStabilityCheck")
        fd_neighbor = float(fd_neighbor)
        fd_step = float(fd_step)
        if not np.isfinite(fd_neighbor) or fd_neighbor < 0:
            raise ValueError("fd_neighbor must be non-negative and finite")
        if not np.isfinite(fd_step) or fd_step <= 0:
            raise ValueError("fd_step must be positive and finite")

        targets = []
        for stable_target in stable_targets:
            fd, tau = float(stable_target["fd"]), float(stable_target["tau"])
            if not np.isfinite(fd) or not np.isfinite(tau):
                raise ValueError("target Doppler and ToF must be finite")
            targets.append({"id": int(stable_target["id"]), "fd": fd, "tau": tau})
        if not targets:
            print("No stable RD targets available for Doppler-specific projection.")
            return []

        fs = float(self.args.fs)
        context_len = min(int(self.args.avg_frames), csi.shape[0])
        if not np.isfinite(fs) or fs <= 0 or context_len <= 0:
            raise ValueError("fs and avg_frames must be positive")
        start = int(np.clip(frame_idx - context_len // 2, 0, csi.shape[0] - context_len))
        segment = csi[start:start + context_len]
        if not np.all(np.isfinite(segment)):
            raise ValueError("CSI projection segment must contain only finite values")
        times_s = np.arange(start, start + context_len, dtype=float) / fs
        if window == "hann":
            taper = np.hanning(context_len) if context_len >= 3 else np.ones(context_len)
        elif window == "hamming":
            taper = np.hamming(context_len)
        else:
            taper = np.ones(context_len)

        neighbor_bins = int(np.floor(fd_neighbor / fd_step + 1e-12))
        bands = []
        for target in targets:
            bins = {}
            for offset in range(-neighbor_bins, neighbor_bins + 1):
                requested_fd = target["fd"] + offset * fd_step
                if not self.args.doppler_min - 1e-12 <= requested_fd <= self.args.doppler_max + 1e-12:
                    continue
                index = int(np.rint((requested_fd - self.args.doppler_min) / fd_step))
                bins[index] = float(self.args.doppler_min + index * fd_step)
            bands.append(bins)

        def temporal_weight(fd, target_index):
            """Return the windowed projection weight, with optional nulls."""
            steering = np.exp(1j * 2.0 * np.pi * fd * times_s)
            reference = taper * steering
            if not constrain_weight:
                return reference, []
            center_fd = targets[target_index]["fd"]
            mirror_fd = -center_fd
            columns = [
                np.exp(1j * 2.0 * np.pi * mirror_fd * times_s)
            ] if abs(center_fd) > 1e-9 else []

            def constrained_weight(constraints):
                if not constraints:
                    return reference
                matrix = np.column_stack(constraints)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                if singular_values[-1] <= 1e-6 * singular_values[0]:
                    return None
                # Project the tapered matched filter away from all nulls,
                # then restore its gain at the desired signed frequency.
                perpendicular = reference - matrix @ np.linalg.lstsq(
                    matrix, reference, rcond=None
                )[0]
                response = np.vdot(perpendicular, steering)
                if abs(response) <= 1e-10 * np.sum(taper):
                    return None
                candidate = perpendicular * np.conj(np.sum(taper) / response)
                noise_gain = np.vdot(candidate, candidate).real / np.vdot(reference, reference).real
                if not np.isfinite(noise_gain) or noise_gain > 4.0:
                    return None
                return candidate

            weight = constrained_weight(columns)
            if weight is None:
                # Near DC the target and its mirror can become indistinguishable.
                return None, []
            null_fds = []
            for other_index, other in enumerate(targets):
                if other_index == target_index:
                    continue
                other_fd = abs(other["fd"])
                # Real amplitude cannot distinguish two tracks at the same
                # absolute Doppler, including a +f/-f mirror pair.
                if abs(other_fd - abs(center_fd)) < 1e-9 or any(
                    abs(other_fd - known) < 1e-9 for known in null_fds
                ):
                    continue
                new_columns = [np.exp(1j * 2.0 * np.pi * other_fd * times_s)]
                if other_fd > 1e-9:
                    new_columns.append(new_columns[0].conj())
                candidate = constrained_weight(columns + new_columns)
                if candidate is None:
                    continue
                columns.extend(new_columns)
                null_fds.append(other_fd)
                weight = candidate
            applied_nulls = ([mirror_fd] if abs(center_fd) > 1e-9 else [])
            for other_fd in null_fds:
                applied_nulls.append(other_fd)
                if other_fd > 1e-9:
                    applied_nulls.append(-other_fd)
            return weight, applied_nulls

        azi_tof = Azi_ToF(self.args)
        results, maps = [], []
        #point_clouds = []
        for target_index, (target, band) in enumerate(zip(targets, bands)):
            if not band:
                continue
            projected_maps = {}
            nulled_by_bin = {}
            for index, fd in band.items():
                weight, null_fds = temporal_weight(fd, target_index)
                if weight is None:
                    print(
                        f"Target #{target['id']:02d}: cannot stably null the center mirror "
                        f"at {-target['fd']:+.2f} Hz while passing {fd:+.2f} Hz"
                    )
                    continue
                z_fd = np.tensordot(weight.conj(), segment, axes=(0, 0))
                tau_grid, theta_grid, spectrum, _ = azi_tof.gen_spectrum(
                    z_fd[np.newaxis, ...],
                    frame_idx=frame_idx,
                    plot=False,
                    sdim_attr="Projected_Azi_ToF_Sdim",
                    energy_ratio_attr="Projected_Azi_ToF_Sdim_ratio",
                    label="Projected Azi-ToF",
                )
                projected_maps[index] = spectrum
                nulled_by_bin[index] = null_fds

            if not projected_maps:
                continue
            """
            points, point_cloud_path = point_cloud.azi_tof_dop_pc(
                {band[index]: projected_maps[index] for index in projected_maps},
                theta_grid,
                tau_grid,
                frame_idx=frame_idx,
                target_id=target["id"],
                pics_dir=getattr(self.args, "pics_dir", None),
            )
            if points.size:
                point_clouds.append(points)
            """

            valid_indices = [
                index for index in projected_maps
                if not getattr(self.args, "tof_gating", True)
                or self.tof_gating_checkX(projected_maps[index], target["tau"], self.tof_gate)
            ]
            if not valid_indices:
                print(
                    f"Target #{target['id']:02d} fd {target['fd']:+.2f} Hz: "
                    "no frequencies passed the ToF gate."
                )
                continue
            spectrum_linear = np.sum([projected_maps[index] for index in valid_indices], axis=0)
            spectrum_db = 10.0 * np.log10(np.maximum(spectrum_linear, 1e-12))
            projection_fds = [band[index] for index in valid_indices]
            title = f"Target #{target['id']:02d} | fd {target['fd']:+.2f} Hz"
            if len(valid_indices) == 1:
                title += "\nAzi-ToF projection"
            else:
                title += f"\nBand [{min(projection_fds):+.2f}, {max(projection_fds):+.2f}] Hz"
            maps.append({
                "frame_idx": frame_idx,
                "x_values": theta_grid,
                "y_values": tau_grid,
                "P_music": spectrum_db.T,
                "title": title,
            })
            results.append({
                "id": target["id"],
                "fd": target["fd"],
                "tau_grid": tau_grid,
                "theta_grid": theta_grid,
                "spectrum_db": spectrum_db,
                "projection_fds": projection_fds,
                "nulled_fds_by_bin": {band[index]: nulled_by_bin[index] for index in valid_indices},
                #"point_cloud": points,
                #"point_cloud_path": point_cloud_path,
            })
        """
        if point_clouds:
            point_cloud.plot_azi_tof_dop_pc(
                point_clouds,
                theta_grid,
                tau_grid,
                frame_idx=frame_idx,
                pics_dir=getattr(self.args, "pics_dir", None),
                max_plot_points=int(getattr(self.args, "point_cloud_max_points", 60000)),
                point_size=float(getattr(self.args, "point_cloud_point_size", 12.0)),
                point_alpha=float(getattr(self.args, "point_cloud_point_alpha", 1.0)),
            )
        """
        if results:
            Plot.plot_multi_spectrum(maps, self.args, file_name="Specific_Project_targets")
        else:
            print("No Doppler-specific target bands passed the ToF gate.")
        return results



    
