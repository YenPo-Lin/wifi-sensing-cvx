"""Compare magnitude-only Azi-ToF and Doppler-gated Azi-ToF heatmaps."""
import csv
from pathlib import Path

from weight_generation import weight_test

import matplotlib
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MUSIC
import Plot
import pre_processing as pp
from simulate_CSI import (
    simulate_csi,
    target_gt_at_frame,
    plot_target_trajectories,
    targets_csi_config,
    trajectory_csi,
)


def simulation_args():
    parser = argparse.ArgumentParser()
    # ---- CSI parameters ----
    parser.add_argument('--num_Tx', type=int, default=1)
    parser.add_argument('--num_Rx', type=int, default=5)
    parser.add_argument('--num_sc', type=int, default=64)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--f_0', type=float, default=5.57e9)
    parser.add_argument('--BW', type=float, default=160e6)
    parser.add_argument('--delta_f', type=float, default=2.54e6)
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--antenna_spacing', type=float, default=0.02)

    # ---- MUSIC settings ----
    parser.add_argument('--preprocess', type=str, default='ma', choices=['ma', 'dwt', 'pca'])
    # plotted frame
    parser.add_argument('--frame_idx', type=int, default=50)
    # MUSIC signal dimension
    parser.add_argument('--Sdim', type=int, default=None)
    parser.add_argument('--Sdim_energy_ratio', type=float, default=0.2)
    parser.add_argument('--avg_frames', type=int, default=100)
    parser.add_argument('--fitting_window_rage', type=int, default=100) # search context
    parser.add_argument('--fitting_window', type=int, default=40) # local LS samples
    parser.add_argument('--projection', type=str, default='cos', choices=['sin', 'cos'])

    parser.add_argument('--stream_win', type=int, default=3)
    parser.add_argument('--stream_sample_range', type=int, default=5) #all Rx

    parser.add_argument('--freq_win', type=int, default=40) #block size = freq_win // freq_hop
    parser.add_argument('--freq_hop', type=int, default=4)
    parser.add_argument('--freq_sample_range', type=int, default=64) #all subcarriers
    parser.add_argument('--freq_space', type=int, default=1) # if freq resampling


    parser.add_argument('--time_win', type=int, default=20)
    parser.add_argument('--time_hop', type=int, default=1)
    parser.add_argument('--time_sample_range', type=int, default= 100)

    # Azimuth grid
    parser.add_argument('--theta_min', type=float, default= 0)
    parser.add_argument('--theta_max', type=float, default= 180)
    parser.add_argument('--theta_step', type=int, default=2)
    # Time of Flight grid
    parser.add_argument('--axis', type=str, default='ns', choices=['ns', 'm'])
    parser.add_argument('--tau_min', type=float, default=2e-9)
    parser.add_argument('--tau_max', type=float, default=20e-9)
    parser.add_argument('--tau_step', type=float, default=2e-10)
    # Doppler grid
    parser.add_argument('--doppler_min', type=float, default=-30)
    parser.add_argument('--doppler_max', type=float, default=30)
    parser.add_argument('--doppler_step', type=float, default=1)

    # 2-D CA-CFAR settings (applied to linear ToF-Doppler MUSIC power)
    parser.add_argument('--cfar_training_cells', type=int, default=3)
    parser.add_argument('--cfar_guard_cells', type=int, default=1)
    parser.add_argument('--cfar_threshold_factor', type=float, default=1.5)
    parser.add_argument('--cfar_top_k', type=int, default=10)
    parser.add_argument('--cfar_min_peak_distance', type=int, default=2)

    # Relative hysteresis thresholds for fixed-fd temporal activity detection.
    parser.add_argument('--activity_low_ratio', type=float, default=0.15)
    parser.add_argument('--activity_high_ratio', type=float, default=0.5)
    parser.add_argument('--activity_baseline_quantile', type=float, default=0.1)
    parser.add_argument('--activity_max_gap_windows', type=int, default=3)
    parser.add_argument('--activity_min_windows', type=int, default=3)

    # Doppler spectrogram settings
    parser.add_argument('--stft_nperseg', type=int, default=64)
    parser.add_argument('--stft_noverlap', type=int, default=63) # hop 1

    # Filled from the instantaneous target states at the plotted frame.
    parser.set_defaults(target_gt=None)
    # heatmap axis (X: Azi, Y: TOF if True)
    parser.add_argument('--axis_flip', type=bool, default=True)
    parser.add_argument('--colorbar', type=bool, default=True)


    # ---- 圖片保存路徑 ----
    # ``os.makedirs("")`` is invalid, so keep a non-empty output directory.
    parser.add_argument('--pics_dir', type=str, default="pic")

    # ----3D MUSIC accumulation method ----
    parser.add_argument('--method', type=str, default='sum', choices=['sum', 'max', 'mean', 'weighted'],)
    parser.add_argument(
        '--reference-model-check', action='store_true',
        help='Compare amplitude-projected Azi-ToF with one versus two static paths.',
    )

    args = parser.parse_args()
    # MUSIC.SteeringVector still uses the legacy ``num_scarriers`` name.
    args.num_sc = args.num_sc
    return args


def run_reference_model_check(args):
    """Test whether amplitude projection follows the assumed Azi-ToF steering.

    The oracle row removes the *known simulated* static-channel phase. It is
    only a diagnostic and is not available from amplitude-only measurements.
    The experiment is noiseless so any peak shift comes from the signal model.
    """
    # Fixed settings keep the Doppler at an integer number of cycles in the
    # projection window and isolate spatial-model mismatch from leakage/noise.
    args = argparse.Namespace(**vars(args))
    args.num_frames = 300
    args.num_Tx = 1
    args.num_Rx = 8
    args.num_sc = 64
    args.fs = 100
    args.avg_frames = 100
    args.frame_idx = 150
    args.stream_sample_range = 8
    args.stream_win = 5
    args.freq_sample_range = 64
    args.freq_win = 58
    args.freq_hop = 3
    args.theta_step = 2
    args.tau_step = 0.2e-9
    args.Projected_Azi_ToF_Sdim = 1

    steering = MUSIC.SteeringVector(args)

    def path_response(theta_deg, tof_s, amplitude=1.0, phase_rad=0.0):
        azi = steering.steering_vector_AoA(theta_deg, stream_win=args.num_Rx)
        tof = steering.steering_vector_ToF(tof_s, freq_win=args.num_sc, freq_hop=1)
        return amplitude * np.exp(1j * phase_rad) * azi[:, None] * tof[None, :]

    target = {
        'name': 'moving', 'theta': 80.0, 'tof': 14e-9, 'fd': 8.0,
        'amplitude': 0.02, 'activity_start_s': 0.0,
        'activity_end_s': args.num_frames / args.fs, 'activity_ramp_s': 0.0,
    }
    dynamic = trajectory_csi(args, target)
    moving_steering = path_response(target['theta'], target['tof'])
    primary = path_response(0.0, 6e-9)
    secondary = path_response(120.0, 10e-9, phase_rad=0.7)
    relative_steering = moving_steering * np.conj(primary)
    ratios = (0.0, 0.25, 0.5, 0.65, 0.8, 1.0, 1.5, 2.0)

    projector = MUSIC.DopplerProject_Azi_ToF(args)
    azi_tof = MUSIC.Azi_ToF(args)
    rows = []
    maps = {}

    def coherence(reference, measured):
        denominator = np.linalg.norm(reference) * np.linalg.norm(measured)
        return float(abs(np.vdot(reference, measured)) / denominator)

    def spectrum_peak(projected):
        tau, theta, spectrum, sdim = azi_tof.gen_spectrum(
            projected[np.newaxis, ...], args.frame_idx, plot=False,
            sdim_attr='Projected_Azi_ToF_Sdim', label='Projected Azi-ToF',
        )
        index = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        return {
            'theta_deg': float(theta[index[0]]),
            'tof_ns': float(tau[index[1]] * 1e9),
            'sdim': int(sdim),
        }, spectrum

    for ratio in ratios:
        static = primary + ratio * secondary
        amplitude = np.abs(static[None, None, :, :] + dynamic)
        dynamic_amplitude = amplitude - pp.MA(amplitude, args.avg_frames)
        projected = projector.doppler_projectionX(
            dynamic_amplitude, args.frame_idx, target['fd'], args,
            window='rect',
        )
        measured = projected[0]

        # First-order amplitude model: C = D * conj(S) / (2 |S|).
        static_magnitude = np.abs(static)
        linearized = (
            target['amplitude'] * moving_steering
            * np.conj(static) / (2.0 * static_magnitude)
        )
        raw_peak, raw_map = spectrum_peak(projected)

        # Remove the simulated static phase to test the suspected cause.
        oracle_projected = projected * (static / static_magnitude)[None, :, :]
        oracle_peak, oracle_map = spectrum_peak(oracle_projected)
        rows.append({
            'secondary_to_primary_ratio': ratio,
            'relative_steering_coherence': coherence(relative_steering, measured),
            'linearized_model_coherence': coherence(linearized, measured),
            'near_static_null_fraction': float(np.mean(
                static_magnitude < 2.0 * target['amplitude']
            )),
            'raw_peak_theta_deg': raw_peak['theta_deg'],
            'raw_peak_tof_ns': raw_peak['tof_ns'],
            'oracle_peak_theta_deg': oracle_peak['theta_deg'],
            'oracle_peak_tof_ns': oracle_peak['tof_ns'],
            'sdim': raw_peak['sdim'],
        })
        maps[ratio] = (raw_map, oracle_map)

    print('Static reference: 0 deg / 6 ns; moving target: 80 deg / 14 ns / 8 Hz')
    print('Single-reference expectation: 80 deg / 8 ns; oracle: 80 deg / 14 ns')
    for row in rows:
        print(
            f"second/static={row['secondary_to_primary_ratio']:.2f} "
            f"coherence={row['relative_steering_coherence']:.3f} "
            f"linearized={row['linearized_model_coherence']:.4f} "
            f"near_null={100 * row['near_static_null_fraction']:.1f}% "
            f"raw=({row['raw_peak_theta_deg']:.0f} deg, "
            f"{row['raw_peak_tof_ns']:.1f} ns) "
            f"oracle=({row['oracle_peak_theta_deg']:.0f} deg, "
            f"{row['oracle_peak_tof_ns']:.1f} ns)"
        )

    if args.pics_dir is not None:
        output_dir = Path(args.pics_dir) / 'reference_model_check'
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'reference_model_check.csv'
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        shown_ratios = (0.0, 0.65, 1.0, 1.5)
        fig, axes = plt.subplots(2, len(shown_ratios), figsize=(15, 7),
                                 sharex=True, sharey=True, constrained_layout=True)
        for col, ratio in enumerate(shown_ratios):
            for row_idx, spectrum in enumerate(maps[ratio]):
                db = 10.0 * np.log10(np.maximum(spectrum, 1e-12))
                # Exact noiseless MUSIC can have a nearly singular one-pixel
                # peak, so use the map median to show the surrounding shape.
                db -= np.median(db)
                mesh = axes[row_idx, col].pcolormesh(
                    azi_tof.tau_grid * 1e9, azi_tof.theta_grid, db,
                    shading='auto', cmap='magma', vmin=0, vmax=15,
                )
                expected_tof = 8.0 if row_idx == 0 else 14.0
                axes[row_idx, col].scatter(expected_tof, 80.0, marker='x',
                                           color='cyan', s=55, linewidths=1.5)
                axes[row_idx, col].set_title(
                    f"second/static={ratio:g} | "
                    f"{'amplitude' if row_idx == 0 else 'oracle'}"
                )
                axes[row_idx, col].set_xlabel('ToF (ns)')
            axes[0, col].set_ylabel('Azimuth (deg)' if col == 0 else '')
            axes[1, col].set_ylabel('Azimuth (deg)' if col == 0 else '')
        fig.colorbar(mesh, ax=axes,
                     label='MUSIC pseudo-spectrum (dB above map median; capped at 15)')
        figure_path = output_dir / 'reference_model_maps.png'
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {csv_path}')
        print(f'Saved: {figure_path}')
    return rows


def run_unbeamformed_projection(CSI, frame_idx, args):
    """Detect Doppler targets without beamforming, then project each target."""
    tof_dop = MUSIC.ToF_Dop(args)
    tau_grid, fd_grid, spectrum, sdim = tof_dop.gen_spectrum(
        CSI, frame_idx, plot=False
    )
    detected_targets = MUSIC.cfar_2D(
        spectrum,
        tau_grid,
        fd_grid,
        training_cells=args.cfar_training_cells,
        guard_cells=args.cfar_guard_cells,
        threshold_factor=args.cfar_threshold_factor,
        top_k=args.cfar_top_k,
        min_peak_distance=args.cfar_min_peak_distance,
    )
    target_labels = np.asarray(
        [(target["tau"], target["fd"]) for target in detected_targets],
        dtype=float,
    ).reshape(-1, 2)
    Plot.plot_spectrum(
        frame_idx,
        x_values=fd_grid,
        y_values=tau_grid,
        P_music=10.0 * np.log10(np.maximum(spectrum, 1e-12)),
        args=args,
        title="ToF-Doppler (no beamforming)",
        spectrum_axes=("doppler", "tof"),
        sdim=sdim,
        target_label=target_labels,
        file_name="label_unbeamformed",
    )

    projector = MUSIC.DopplerProject_Azi_ToF(args)
    azi_tof = MUSIC.Azi_ToF(args)
    maps = []
    results = []
    for target in detected_targets:
        projected = projector.doppler_projectionX(
            CSI, frame_idx, target["fd"], args, window="hann"
        )
        projected_tau, projected_theta, projected_spectrum, projected_sdim = (
            azi_tof.gen_spectrum(
                projected[np.newaxis, ...],
                frame_idx,
                plot=False,
                sdim_attr="Projected_Azi_ToF_Sdim",
                energy_ratio_attr="Projected_Azi_ToF_Sdim_ratio",
                label="Projected Azi-ToF",
            )
        )
        spectrum_db = 10.0 * np.log10(np.maximum(projected_spectrum, 1e-12))
        maps.append({
            "frame_idx": frame_idx,
            "x_values": projected_theta,
            "y_values": projected_tau,
            "P_music": spectrum_db.T,
            "title": f"Target #{target['rank']:02d} | fd {target['fd']:+.2f} Hz",
        })
        results.append({
            "target": target,
            "fd": target["fd"],
            "tau_grid": projected_tau,
            "theta_grid": projected_theta,
            "spectrum_db": spectrum_db,
            "sdim": projected_sdim,
        })

    print(f"Unbeamformed Doppler projection: {len(results)} targets")
    if maps:
        Plot.plot_multi_spectrum(maps, args)
    return results


def main():
    args = simulation_args()
    if args.reference_model_check:
        return run_reference_model_check(args)
    targets = targets_csi_config()
    args.target_gt = target_gt_at_frame(args, targets, args.frame_idx)
    plot_target_trajectories(args, targets, args.frame_idx)

    csi_amplitude = simulate_csi(
        args,
        targets=targets,
        static_amplitude=1.0,
        snr_db=5.0,
        rx_std_db=0.5,
        sc_std_db=0.5,
        sc_smooth_window=7,
        seed=7,
    )

    CSI = csi_amplitude  # Keep the current CSI-amplitude projection path.

    #weight_test(CSI, args)
    background = pp.MA(CSI, args.fs * 0.5)

    if args.preprocess == "ma":
        #CSI = (CSI - background) # Original Version
        CSI = (CSI - background)# / (background + 1e-8) # Normalized dynamic residual

    frame_idx = args.frame_idx
    """
    # 3D Azi-ToF-Doppler MUSIC and its Doppler-RoI Azi-ToF maps.
    azi_tof_dop = MUSIC.Azi_ToF_Dop(args)
    _ = azi_tof_dop.gen_spectrum(CSI, frame_idx, method="sum", fig_name="3D_MUSIC_sum",)
    roi_results = azi_tof_dop.gen_Doppler_RoI_specturm(CSI, frame_idx, fd_range=2, fig_name="3D_MUSIC_RoI",)

    # Direct and Doppler-conditioned 2D Azi-ToF MUSIC.
    azi_tof = MUSIC.Azi_ToF(args)
    azi_tof.gen_spectrum(CSI, frame_idx, x_axis="azi", y_axis="tof")
    projected_results = azi_tof.gen_Doppler_projection_spectrum(CSI, frame_idx)



    azi_dop = MUSIC.Azi_Dop(args)
    tof_dop = MUSIC.ToF_Dop(args)
    _ = azi_dop.gen_spectrum(CSI, frame_idx)
    _ = tof_dop.    azi_dop = MUSIC.Azi_Dop(args)
    tof_dop = MUSIC.ToF_Dop(args)
    _ = tof_dop.gen_spectrum(CSI, frame_idx)

    #detected_targets = [result["target"] for result in roi_results]
    #for target, projected_result in zip(detected_targets, projected_results):
    #    target["projected_azi_tof"] = projected_result
    #return detected_targets
    """
    results = run_unbeamformed_projection(CSI, frame_idx, args)



if __name__ == "__main__":
    main()
