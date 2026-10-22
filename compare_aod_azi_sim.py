"""Controlled amplitude-CSI comparison of AoD-ToF and Azi-ToF MUSIC.

The moving target has the same injected AoA and AoD.  A separate static path
provides the interference needed for angle/ToF structure to survive taking
the CSI magnitude.  This script does not change the production simulator.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MUSIC
import pre_processing as pp
from simulate_CSI import simulate_csi


def comparison_args(rx_window):
    """Keep the relevant settings equal to main.py except for Rx aperture."""
    return Namespace(
        f_0=5.57e9,
        delta_f=2.54e6,
        fs=100,
        antenna_spacing=0.02,
        tx_antenna_spacing=0.02,
        projection="cos",
        num_Tx=2,
        num_Rx=8,
        num_sc=64,
        avg_frames=100,
        time_win=20,
        time_hop=1,
        time_sample_range=50,
        stream_win=rx_window,
        stream_sample_range=8,
        freq_win=50,
        freq_hop=2,
        freq_sample_range=64,
        freq_space=1,
        theta_min=0.0,
        theta_max=180.0,
        theta_step=3.0,
        tau_min=1e-9,
        tau_max=15e-9,
        tau_step=0.2e-9,
        Azi_ToF_Sdim=2,
        AoD_ToF_Sdim=2,
    )


def simulate_amplitude(args, theta_deg=105.0, tof_s=10e-9, fd_hz=5.0,
                       target_amplitude=0.35, static_amplitude=1.0,
                       snr_db=None, seed=7):
    """Generate |H_static + H_target| with independent Tx and Rx steering."""
    frames = 160
    tx = np.arange(args.num_Tx)[None, :, None, None]
    rx = np.arange(args.num_Rx)[None, None, :, None]
    sc = np.arange(args.num_sc)[None, None, None, :]
    t = np.arange(frames)[:, None, None, None] / args.fs
    projected = 1.0 - np.cos(np.deg2rad(theta_deg))
    tx_spacing = args.tx_antenna_spacing
    spatial_phase = -2j * np.pi * args.f_0 * projected / 3e8 * (
        args.antenna_spacing * rx + tx_spacing * tx
    )
    tof_phase = -2j * np.pi * (args.f_0 + args.delta_f * sc) * tof_s
    doppler_phase = 2j * np.pi * fd_hz * t
    target = target_amplitude * np.exp(spatial_phase + tof_phase + doppler_phase)
    amplitude = np.abs(static_amplitude + target)
    if snr_db is not None:
        noise_std = np.sqrt(np.mean(amplitude**2) / 10 ** (snr_db / 10))
        amplitude = np.maximum(
            amplitude + np.random.default_rng(seed).normal(0, noise_std, amplitude.shape),
            0.0,
        )
    return amplitude


def spectrum_pair(args, residual, frame_idx=80):
    azi = MUSIC.Azi_ToF(args)
    aod = MUSIC.AoD_ToF(args)
    tau_azi, theta_azi, p_azi, sdim_azi = azi.gen_spectrum(
        residual, frame_idx, plot=False
    )
    tau_aod, theta_aod, p_aod, sdim_aod = aod.gen_spectrum(
        residual, frame_idx, plot=False
    )
    np.testing.assert_allclose(tau_azi, tau_aod)
    np.testing.assert_allclose(theta_azi, theta_aod)
    return tau_azi, theta_azi, p_azi, p_aod, sdim_azi, sdim_aod


def peak(theta_grid, tau_grid, spectrum, tof_truth=10e-9):
    """Report both global peak and peak within 2 ns of injected ToF."""
    ai, ti = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    band = np.abs(tau_grid - tof_truth) <= 2e-9
    local_ai, local_ti = np.unravel_index(
        np.argmax(spectrum[:, band]), spectrum[:, band].shape
    )
    return (theta_grid[ai], tau_grid[ti] * 1e9), (
        theta_grid[local_ai], tau_grid[band][local_ti] * 1e9
    )


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_angle = 105.0
    truth_tof_ns = 10.0
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    results = []
    cases = ("matched 2 Rx", "matched 5 Rx", "current simulator 5 Rx")
    for row, case in enumerate(cases):
        rx_window = 2 if row == 0 else 5
        args = comparison_args(rx_window)
        if row < 2:
            amplitude = simulate_amplitude(args, theta_deg=truth_angle,
                                           tof_s=truth_tof_ns * 1e-9, snr_db=20)
        else:
            # Existing simulate_CSI.py broadcasts the target equally over Tx.
            args.num_frames = 160
            amplitude = simulate_csi(
                args,
                targets=[dict(theta=truth_angle, tof=truth_tof_ns * 1e-9,
                              fd=5.0, amplitude=0.35)],
                static_amplitude=1.0, snr_db=20,
                rx_std_db=0.0, sc_std_db=0.0,
            )
        residual = amplitude - pp.MA(amplitude, args.avg_frames)
        tau, theta, p_azi, p_aod, s_azi, s_aod = spectrum_pair(args, residual)
        if row == 1:
            reverse_tau, reverse_theta, reverse_p, _ = MUSIC.AoD_ToF(args).gen_spectrum(
                residual[:, ::-1], 80, plot=False
            )
            reversed_peak = peak(reverse_theta, reverse_tau, reverse_p)[0]
        for col, (name, spectrum, sdim) in enumerate((
            ("Azi-ToF", p_azi, s_azi), ("AoD-ToF", p_aod, s_aod)
        )):
            global_peak, tof_band_peak = peak(theta, tau, spectrum)
            results.append((case, name, sdim, global_peak, tof_band_peak))
            ax = axes[row, col]
            db = 10 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-12))
            im = ax.imshow(db.T, origin="lower", aspect="auto",
                           extent=(theta[0], theta[-1], tau[0] * 1e9,
                                   tau[-1] * 1e9), cmap="magma", vmin=-20,
                           vmax=0)
            ax.scatter(truth_angle, truth_tof_ns, marker="x", s=70,
                       color="cyan", linewidths=2, label="injected truth")
            ax.scatter(*tof_band_peak, marker="o", facecolors="none",
                       edgecolors="white", s=65, linewidths=1.5,
                       label="peak near truth ToF")
            ax.set_title(f"{name} | {case} | Sdim {sdim}")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("ToF (ns)")
    fig.colorbar(im, ax=axes, label="Relative pseudo-spectrum (dB)")
    path = output_dir / "aod_vs_azi_tof_sim.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    for case, name, sdim, global_peak, near_peak in results:
        print(f"{case} {name} Sdim={sdim}: "
              f"global peak={global_peak[0]:.0f} deg, {global_peak[1]:.1f} ns; "
              f"near-ToF peak={near_peak[0]:.0f} deg, {near_peak[1]:.1f} ns")
    print(f"Injected truth: AoA=AoD={truth_angle:.0f} deg, ToF={truth_tof_ns:.1f} ns")
    print(f"Reversed Tx order (matched 5 Rx): AoD peak="
          f"{reversed_peak[0]:.0f} deg, {reversed_peak[1]:.1f} ns")
    args = comparison_args(5)
    no_reference = simulate_amplitude(args, static_amplitude=0.0, snr_db=None)
    no_reference_residual = no_reference - pp.MA(no_reference, args.avg_frames)
    print("No static reference, noise-free: interior residual RMS="
          f"{np.sqrt(np.mean(no_reference_residual[50:110] ** 2)):.2e}")
    angle_errors = {"Azi-ToF": [], "AoD-ToF": []}
    tof_errors = {"Azi-ToF": [], "AoD-ToF": []}
    for angle_truth in (60.0, 90.0, 120.0):
        for tof_truth_ns in (6.0, 10.0, 14.0):
            amplitude = simulate_amplitude(
                args, theta_deg=angle_truth, tof_s=tof_truth_ns * 1e-9,
                snr_db=20,
            )
            residual = amplitude - pp.MA(amplitude, args.avg_frames)
            tau, theta, p_azi, p_aod, _, _ = spectrum_pair(args, residual)
            for name, spectrum in (("Azi-ToF", p_azi), ("AoD-ToF", p_aod)):
                found_angle, found_tof_ns = peak(
                    theta, tau, spectrum, tof_truth_ns * 1e-9
                )[0]
                angle_errors[name].append(abs(found_angle - angle_truth))
                tof_errors[name].append(abs(found_tof_ns - tof_truth_ns))
    for name in ("Azi-ToF", "AoD-ToF"):
        print(f"9-point sweep {name}: max angle error="
              f"{max(angle_errors[name]):.0f} deg, max ToF error="
              f"{max(tof_errors[name]):.1f} ns")
    print(f"Saved: {path}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path,
                        default=Path("simulation_outputs/aod_azi_comparison"))
    run(parser.parse_args().output_dir)
