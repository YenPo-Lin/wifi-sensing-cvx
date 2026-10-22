"""Check which center-frame RD peaks persist across nearby frames.

The active pipeline passes dynamic CSI amplitude to Beamform_RD_Map. Its
MUSIC RD maps are pseudo-spectra; a stable peak is a persistent
scattering feature, not an identified body part.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import MUSIC
from beamform_rd_map import Beamform_RD_Map


class SpatialStabilityCheck:
    pass

class RDStabilityCheck:
    def __init__(self, args, beamformer=None):
        self.args = args
        self.beamformer = beamformer or Beamform_RD_Map(args)

    def stable_targets_from_RDmap(
        self, CSI_for_RD, frame_idx, observation_interval=None, *, center_result=None,
        peak_source="fused", fd_step=None, CSI_for_RA=None
    ):
        """Return center RD peaks that pass hit-count and time-ratio thresholds.

        ``observation_interval`` is an offset in CSI frames. Check only the
        center and its two neighbors at ``frame_idx ± round(observation_interval)``;
        out-of-range or duplicate frame indices are omitted. Every check uses
        the angle candidates selected at the center. With ``peak_source='candidate'``,
        center peaks come from individual beam maps and neighbors must match
        on the same beam. ``peak_source='fused'`` retains the original behavior.
        Neighboring peaks are matched one-to-one within the configured ToF
        and Doppler gates. A target needs at least ``track_min_hits`` matches
        and strictly more than ``track_min_ratio`` of observed frame centers.
        """
        csi = np.asarray(CSI_for_RD)
        candidate_csi = csi if CSI_for_RA is None else np.asarray(CSI_for_RA)
        if csi.ndim != 4 or csi.shape[0] == 0:
            raise ValueError("CSI_for_RD must have shape (frame, Tx, Rx, subcarrier)")
        if candidate_csi.shape != csi.shape:
            raise ValueError("CSI_for_RA must have the same shape as CSI_for_RD")
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < csi.shape[0]:
            raise ValueError("frame_idx is outside CSI")
        if peak_source not in ("fused", "candidate"):
            raise ValueError("peak_source must be 'fused' or 'candidate'")
        if peak_source == "candidate":
            if fd_step is None or not np.isfinite(fd_step) or fd_step <= 0:
                raise ValueError("candidate tracking requires positive finite fd_step")
            projector = MUSIC.DopplerProject_Azi_ToF(self.args)

        if observation_interval is None:
            observation_interval = 0.02 * float(self.args.fs)
        if not np.isfinite(observation_interval) or observation_interval < 0:
            raise ValueError("observation_interval must be finite and non-negative")
        radius = int(round(observation_interval))
        observation_frames = sorted({
            current_frame
            for current_frame in (frame_idx - radius, frame_idx, frame_idx + radius)
            if 0 <= current_frame < csi.shape[0]
        })

        tau_gate = float(self.args.track_tau_gate_ns) * 1e-9
        fd_gate = float(self.args.track_fd_gate_hz)
        min_hits = int(self.args.track_min_hits)
        min_ratio = float(getattr(self.args, "track_min_ratio", 0.5))
        min_snr_db = float(self.args.track_cfar_min_snr_db)
        if not np.isfinite(tau_gate) or tau_gate <= 0:
            raise ValueError("track_tau_gate_ns must be finite and positive")
        if not np.isfinite(fd_gate) or fd_gate <= 0:
            raise ValueError("track_fd_gate_hz must be finite and positive")
        if min_hits < 1 or not np.isfinite(min_snr_db):
            raise ValueError("track_min_hits must be positive and CFAR SNR must be finite")
        if not np.isfinite(min_ratio) or not 0.0 <= min_ratio < 1.0:
            raise ValueError("track_min_ratio must be finite and in [0, 1)")
        observation_count = len(observation_frames)
        required_hits = max(min_hits, int(np.floor(min_ratio * observation_count)) + 1)

        if center_result is None:
            center_result = self.beamformer.gen_beamform(candidate_csi, csi, frame_idx, plot=False)
        tau_grid = np.asarray(center_result["tau_grid"])
        fd_grid = np.asarray(center_result["fd_grid"])

        def detect(result, *, select=True):
            if not np.array_equal(result["tau_grid"], tau_grid) or not np.array_equal(
                result["fd_grid"], fd_grid
            ):
                raise ValueError("neighboring RD maps must share the center grids")
            if peak_source == "candidate":
                return projector.detect_beamform_targets(result, fd_step, select=select)
            peaks = MUSIC.cfar_2D(
                result["fused_map"], tau_grid, fd_grid,
                training_cells=self.args.track_cfar_training_cells,
                guard_cells=self.args.track_cfar_guard_cells,
                threshold_factor=self.args.track_cfar_threshold_factor,
                top_k=self.args.track_cfar_top_k,
                min_peak_distance=self.args.track_cfar_min_peak_distance,
            )
            return [peak for peak in peaks if peak["snr_db"] >= min_snr_db]

        center_peaks = detect(center_result)
        print(
            f"RD stability gate: at least {required_hits}/{observation_count} frames "
            f"(more than {min_ratio:.0%}, minimum {min_hits} hits)"
        )
        if not center_peaks or required_hits > observation_count:
            return []
        observed_frames = [[frame_idx] for _ in center_peaks]
        for current_frame in observation_frames:
            if current_frame == frame_idx:
                continue
            result = self.beamformer.gen_beamform(
                candidate_csi, csi, current_frame, candidate_reference=center_result, plot=False
            )
            detections = detect(result, select=False)
            if not detections:
                continue
            costs = np.full((len(center_peaks), len(detections)), 1e6, dtype=float)
            for i, center in enumerate(center_peaks):
                for j, candidate in enumerate(detections):
                    if (peak_source == "candidate"
                            and center["beam_idx"] != candidate["beam_idx"]):
                        continue
                    dtau = abs(center["tau"] - candidate["tau"]) / tau_gate
                    dfd = abs(center["fd"] - candidate["fd"]) / fd_gate
                    if dtau <= 1 and dfd <= 1:
                        costs[i, j] = dtau**2 + dfd**2
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] < 1e6:
                    observed_frames[row].append(current_frame)

        stable_targets = [
            {
                **peak,
                "id": int(peak["rank"]),
                "hit_count": len(frames),
                "observation_count": observation_count,
                "required_hits": required_hits,
                "stability_ratio": len(frames) / observation_count,
                "observed_frames": sorted(frames),
            }
            for peak, frames in zip(center_peaks, observed_frames)
            if len(frames) >= required_hits
        ]
        return sorted(stable_targets, key=lambda target: target["fd"])
