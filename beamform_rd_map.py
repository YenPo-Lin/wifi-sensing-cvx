"""Angle-conditioned MUSIC ToF-Doppler maps for the existing CSI pipeline.

The active ``signal_processing`` path passes dynamic CSI amplitude residuals.
For that input, the angular weights and delay maps are pseudo-spatial features;
physical coherent AoA/ToF beamforming requires calibrated complex CSI.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

import MUSIC
import Plot


class Beamform_RD_Map:
    def __init__(self, args):
        self.args = args
        self.angle_min = 20.0
        self.angle_max = 160.0
        self.min_angle_separation = 10.0
        self.tof_half_width = 0.6e-9
        self.fused_min_snr_db = float(getattr(args, "beamform_fused_min_snr_db", 0.4))
        if not np.isfinite(self.fused_min_snr_db):
            raise ValueError("beamform_fused_min_snr_db must be finite")
        self.azi_tof = MUSIC.Azi_ToF(args)
        self.tof_dop = MUSIC.ToF_Dop(args)
        self.steering = MUSIC.SteeringVector(args)

    def _angle_candidates(self, tau_grid, theta_grid, spectrum, max_candidates):
        """Rank local Azi-ToF peaks and suppress nearby angle/ToF duplicates."""
        tau_grid = np.asarray(tau_grid)
        theta_grid = np.asarray(theta_grid)
        spectrum = np.asarray(spectrum)
        if spectrum.shape != (theta_grid.size, tau_grid.size):
            raise ValueError("Azi-ToF spectrum shape must be (theta, tau)")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("Azi-ToF spectrum contains non-finite values")
        angle_mask = (theta_grid >= self.angle_min) & (theta_grid <= self.angle_max)
        if not np.any(angle_mask):
            raise ValueError("Azi-ToF theta grid has no samples in 20-160 degrees")
        tof_mask = tau_grid > 3e-9  # Match the existing azimuth beamforming gate.
        if not np.any(tof_mask):
            raise ValueError("Azi-ToF tau grid has no samples above 3 ns")
        smoothed = gaussian_filter(spectrum, sigma=(1.0, 1.0))
        search_values = smoothed[np.ix_(angle_mask, tof_mask)]
        background = float(np.median(search_values))
        peak_floor = background + 0.1 * (float(search_values.max()) - background)
        local_maxima = (
            (smoothed == maximum_filter(smoothed, size=(3, 3), mode="nearest"))
            & (smoothed > peak_floor) & angle_mask[:, None] & tof_mask[None, :]
        )
        peaks = np.argwhere(local_maxima)
        peaks = sorted(peaks, key=lambda ij: smoothed[tuple(ij)], reverse=True)

        # N Rx can satisfy at most N independent linear constraints. The cap
        # also bounds the number of expensive RD maps.
        selected = []
        for theta_idx, tau_idx in peaks:
            theta = float(theta_grid[theta_idx])
            tau = float(tau_grid[tau_idx])
            if any(
                abs(theta - existing_theta) < self.min_angle_separation
                and abs(tau - existing_tau) <= self.tof_half_width
                for existing_theta, existing_tau, _ in selected
            ):
                continue
            selected.append((theta, tau, float(smoothed[theta_idx, tau_idx])))
            if len(selected) >= max_candidates:
                break
        if not selected:
            raise ValueError("Azi-ToF spectrum has no local peak in the search region")

        angles, delays, scores = (np.asarray(values) for values in zip(*selected))
        return angles, delays, scores, (float(angles.min()), float(angles.max()))

    def _null_steering_weights(self, theta, other_angles, num_rx):
        """Minimum-norm beam with target gain and stable angular nulls."""
        target = self.steering.steering_vector_AoA(theta, stream_win=num_rx)
        columns = [target]
        nulled = []
        for other in other_angles:
            if abs(other - theta) < self.min_angle_separation or any(
                abs(other - prior) < self.min_angle_separation for prior in nulled
            ):
                continue
            if len(columns) >= num_rx:
                break
            trial = np.column_stack(
                columns + [self.steering.steering_vector_AoA(other, stream_win=num_rx)]
            )
            gram = trial.conj().T @ trial
            if np.linalg.cond(gram) > 1e4:
                continue
            desired = np.zeros(trial.shape[1], dtype=complex)
            desired[0] = np.sqrt(num_rx)  # Same target gain as the former beam.
            weights = trial @ np.linalg.solve(gram, desired)
            if np.linalg.norm(weights) > 3.0:
                continue
            columns.append(trial[:, -1])
            nulled.append(float(other))
        matrix = np.column_stack(columns)
        desired = np.zeros(matrix.shape[1], dtype=complex)
        desired[0] = np.sqrt(num_rx)
        weights = matrix @ np.linalg.solve(matrix.conj().T @ matrix, desired)
        return weights, np.asarray(nulled, dtype=float)

    def _plot_beamform_pattern(self, frame_idx, angles, weights, null_angles):
        """Plot 0-180 degree polar patterns from the applied Rx weights."""
        scan_theta = np.linspace(0.0, 180.0, 721)
        steering = np.stack([
            self.steering.steering_vector_AoA(theta, stream_win=weights.shape[1])
            for theta in scan_theta
        ])
        fig, axes = plt.subplots(
            1, len(angles), figsize=(6.0 * len(angles), 3.7),
            squeeze=False,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.88, wspace=0.06)
        theta_rad = np.deg2rad(scan_theta)
        for ax, theta, weight, nulled in zip(axes[0, :], angles, weights, null_angles):
            response = np.abs(steering @ weight.conj())
            response /= np.max(response)
            gain_db = 20.0 * np.log10(np.maximum(response, 1e-12))
            radius = np.clip((gain_db + 25.0) / 25.0, 0.0, 1.0)
            ax.add_patch(Wedge((0, 0), 1, 0, 180, facecolor="white",
                               edgecolor="black", linewidth=1.0))
            for ring_db in (-20, -15, -10, -5):
                ring_radius = (ring_db + 25.0) / 25.0
                ax.plot(-ring_radius * np.cos(theta_rad), ring_radius * np.sin(theta_rad),
                        color="0.70", linewidth=0.8)
            for grid_theta in np.deg2rad(np.arange(0, 181, 30)):
                ax.plot([0, -np.cos(grid_theta)], [0, np.sin(grid_theta)],
                        color="0.70", linewidth=0.8)
            ax.plot(-radius * np.cos(theta_rad), radius * np.sin(theta_rad),
                    color="#0072BD", linewidth=1.8)
            for angle_label in range(0, 181, 30):
                angle_rad = np.deg2rad(angle_label)
                label_x = -1.08 * np.cos(angle_rad)
                label_y = 1.08 * np.sin(angle_rad)
                alignment = "right" if angle_label == 0 else "left" if angle_label == 180 else "center"
                ax.text(label_x, label_y, f"{angle_label}°", color="black",
                        fontsize=10, ha=alignment, va="center")
            for tick_db in (0, -5, -10, -15, -20, -25):
                tick_x = -(tick_db + 25.0) / 25.0
                ax.text(tick_x, -0.065, f"{tick_db}", color="black",
                        fontsize=9, ha="center", va="top")
            ax.text(-0.5, -0.18, "Normalized gain (dB)", color="black",
                    fontsize=9, ha="center", va="top")
            null_text = ", ".join(f"{angle:g}°" for angle in nulled)
            title = f"Beam {theta:g}°"
            if null_text:
                title += f"  |  Null {null_text}"
            ax.text(0, 1.20, title, ha="center", va="bottom", color="black",
                    fontsize=11, clip_on=False)
            ax.set_xlim(-1.20, 1.20)
            ax.set_ylim(-0.24, 1.24)
            ax.set_aspect("equal")
            ax.axis("off")
        if getattr(self.args, "pics_dir", None) is not None:
            save_dir = os.path.join(self.args.pics_dir, "Beamform_Pattern")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{frame_idx:04d}_beamform_pattern.png")
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show(block=False)

    def gen_beamform(self, CSI_for_RA, CSI_for_RD, frame_idx, *, candidate_reference=None, plot=True, plot_beamform=False):
        """Return separate angle maps and a diagnostic mean map.

        Both inputs have shape (frame, Tx, Rx, subcarrier). CSI_for_RA selects
        the Azi-ToF angle candidates; CSI_for_RD produces the beamformed RD maps.
        ``time_sample_range`` frames
        and ``freq_sample_range`` contiguous subcarriers are used exactly; the
        time window shifts at recording edges. MUSIC uses the configured ToF
        and Doppler search grids. The Azi-ToF candidate estimator retains its
        own ``avg_frames`` context. ``candidate_reference`` reuses a center
        frame's angle/ToF candidates for nearby frames without re-estimating
        Azi-ToF. Nulls here use the assumed steering model on amplitude data;
        they are not calibrated physical angular nulls.
        """
        csi = np.asarray(CSI_for_RD)
        candidate_csi = np.asarray(CSI_for_RA)
        if csi.ndim != 4 or candidate_csi.shape != csi.shape:
            raise ValueError("CSI_for_RA and CSI_for_RD must have the same 4D shape")
        total_frames, num_tx, num_rx, num_sc = csi.shape
        time_count = int(self.args.time_sample_range)
        freq_count = int(self.args.freq_sample_range)
        if time_count < 2 or freq_count < 2:
            raise ValueError("time_sample_range and freq_sample_range must be at least 2")
        if total_frames < time_count or num_sc < freq_count:
            raise ValueError(
                f"Need at least {time_count} frames and {freq_count} subcarriers; "
                f"got {total_frames} and {num_sc}"
            )
        if num_tx < 1 or num_rx < 1:
            raise ValueError("CSI must contain at least one Tx and one Rx")
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < total_frames:
            raise ValueError(f"frame_idx must be in [0, {total_frames - 1}]")
        start = int(np.clip(frame_idx - time_count // 2, 0, total_frames - time_count))
        end = start + time_count
        segment = csi[start:end, :, :, :freq_count]
        if not np.all(np.isfinite(segment)):
            raise ValueError("Selected CSI segment contains non-finite values")

        if candidate_reference is None:
            tau_azi, theta_grid, azi_spectrum, _ = self.azi_tof.gen_spectrum(
                candidate_csi, frame_idx, plot=False
            )
            angles, delays, scores, angle_range = self._angle_candidates(
                tau_azi, theta_grid, azi_spectrum, num_rx
            )
            theta_peak, tau_peak = float(angles[0]), float(delays[0])
            angle_text = ", ".join(f"{theta:g}°" for theta in angles)
            print(f"⚡️ AoA candidates [{angle_text}]")
        else:
            angles = np.asarray(candidate_reference["theta_candidates"], dtype=float)
            delays = np.asarray(candidate_reference["tau_candidates"], dtype=float)
            scores = np.asarray(candidate_reference["theta_scores"], dtype=float)
            if angles.ndim != 1 or not 1 <= angles.size <= num_rx or not np.all(np.isfinite(angles)):
                raise ValueError("candidate_reference must contain 1 to num_rx finite angles")
            if np.any((angles < self.angle_min) | (angles > self.angle_max)):
                raise ValueError("candidate_reference angles must be in 20-160 degrees")
            if (scores.shape != angles.shape or delays.shape != angles.shape
                    or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(delays))):
                raise ValueError("candidate_reference scores and delays must match the angles")
            theta_peak = float(candidate_reference["target_theta"])
            tau_peak = float(candidate_reference["target_tau"])
            angle_range = tuple(candidate_reference["candidate_angle_range"])

        maps = []
        sdims = []
        null_angles = []
        beam_weights = []
        for index, theta in enumerate(angles):
            weights, nulled = self._null_steering_weights(
                theta, np.delete(angles, index), num_rx
            )
            beam = np.einsum(
                "r,turs->tus", weights.conj(), segment,
                optimize=True,
            )
            tau_grid, fd_grid, power, sdim = self.tof_dop.gen_beamformed_spectrum(
                beam[:, :, np.newaxis, :], frame_idx - start, plot=False,
            )
            maps.append(power)
            sdims.append(sdim)
            null_angles.append(nulled)
            beam_weights.append(weights)
            if candidate_reference is None:
                null_text = ", ".join(f"{angle:g}°" for angle in nulled)
                #print(f"Stage1: beam {theta:g}° nulls [{null_text}]")

        maps = np.stack(maps, axis=0)  # Independent input channels: (angle, ToF, Doppler).
        fused_map = np.mean(maps, axis=0)

        if plot_beamform:
            self._plot_beamform_pattern(
                frame_idx, angles, np.stack(beam_weights, axis=0), null_angles
            )

        if plot and getattr(self.args, "pics_dir", None) is not None:
            maps_db = 10.0 * np.log10(np.maximum(maps, 1e-12))
            num_cols = min(5, len(angles))
            num_rows = (len(angles) + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                num_rows, num_cols,
                figsize=(5.5 * num_cols, 4.5 * num_rows),
                squeeze=False,
                constrained_layout=True,
            )
            color_limits = (float(maps_db.min()), float(maps_db.max()))
            for ax, theta, spectrum_db in zip(axes.flat, angles, maps_db):
                Plot.plot_spectrum(
                    frame_idx, fd_grid, tau_grid,
                    spectrum_db, self.args,
                    title=f"MUSIC ToF-Doppler, azimuth {theta:g} deg",
                    spectrum_axes=("doppler", "tof"),
                    ax=ax, save=False,
                )
                ax.collections[0].set_clim(*color_limits)
            for ax in axes.flat[len(angles):]:
                ax.set_visible(False)
            save_dir = os.path.join(self.args.pics_dir, "ToF_Doppler")
            os.makedirs(save_dir, exist_ok=True)
            candidate_path = os.path.join(
                save_dir, f"{frame_idx:04d}_beamform_candidate.png"
            )
            fig.savefig(candidate_path, dpi=100)
            plt.close(fig)
            print(f"Saved: {candidate_path}")
            Plot.plot_spectrum(
                frame_idx, fd_grid, tau_grid,
                10.0 * np.log10(np.maximum(fused_map, 1e-12)), self.args,
                title="Fused MUSIC ToF-Doppler",
                spectrum_axes=("doppler", "tof"),
                file_name="beamform_fused",
            )

        result = {
            "frame_idx": frame_idx,
            "theta_candidates": angles,
            "tau_candidates": delays,
            "theta_scores": scores,
            "target_theta": theta_peak,
            "target_tau": tau_peak,
            "candidate_angle_range": angle_range,
            "tau_grid": tau_grid,
            "fd_grid": fd_grid,
            "maps": maps,
            "fused_map": fused_map,
            "sdims": np.asarray(sdims, dtype=int),
            "null_angles": null_angles,
            "frame_range": (start, end),
        }
        if candidate_reference is None:
            targets = MUSIC.DopplerProject_Azi_ToF(self.args).detect_beamform_targets(
                result, float(self.args.doppler_step)
            )
            fd_text = ", ".join(f"{target['fd']:+.2f} Hz" for target in targets)
            print(f"⚡️ Doppler Targets [{fd_text}]")
        return result

class Beamform_Dop_Map:
    pass
