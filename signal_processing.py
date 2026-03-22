import numpy as np
import pre_processing
import Plot
import MUSIC
import LASSO
import os
import scipy.io as sio
import utils


def signal_processing(CSI, args):
    print("🤓 Starting Signal Processing... ")



    CSI = pre_processing.self_sanitize(CSI)
    CSI -= pre_processing.MA(CSI, window_size = args.fs* 0.5)
    #Plot.plot_phase_along_subcarrier(CSI, args, frame_idx=700, title_prefix=" Phase vs Subcarrier ")
    #Plot.plot_phase_along_time(CSI, args, subc_idx=0, title_prefix=" Phase vs Time ")
    #Plot.plot_phases_along_time(CSI, rx_idx=0, title_prefix=" Phase vs Time ")
    #Plot.plot_amps_along_time(CSI, rx_idx=0, title_prefix=" Amplitude vs Time ")
    


    '''
    lam = 0.12
    A, theta_grid, tau_grid = utils.build_dictionary(args, theta_min=-90, theta_max=90, theta_step=4, tau_min=0.5e-8, tau_max=2.5e-8, tau_step=4e-10)
    y = utils.build_Y_packets(CSI, frame_idx, args.num_subcarriers, K_frame=1)
    print(f"CSI vec size: {len(y)}, θ grid size: {len(theta_grid)},  τ grid size: {len(tau_grid)}")

    x_cvx = FISTA.FISTA_Lasso(A, y, lam, max_iter=3000, tol=1e-3, verbose=True)
    P = np.abs(x_cvx).reshape(len(theta_grid), len(tau_grid))
    peaks = utils.find_peaks(P, theta_grid, tau_grid)
    itheta, itau, theta_val, tau_val, _ = peaks[0]
    itheta1, itau1, theta_val1, tau_val1, _ = peaks[1]
    itheta2, itau2, theta_val2, tau_val2, _ = peaks[2]
    
    Plot.plot_spectrum(frame_idx, tau_grid, theta_grid, P, args, title=f"global peaks CVXPY lam={lam}")

    y1_hat_top = utils.reconstruct(A, x_cvx, peaks[1], theta_grid, tau_grid, radius=1)
    y2_hat_top = utils.reconstruct(A, x_cvx, peaks[2], theta_grid, tau_grid, radius=1)
    r2 = y - y1_hat_top- y2_hat_top

    theta_min = theta_val - 10/2
    theta_max = theta_val + 10/2
    tau_min = tau_val - 1e-9/2
    tau_max = tau_val + 1e-9/2
    theta_step = 0.2
    tau_step = 2e-11

    A_fne, theta_grid_fine, tau_grid_fine = utils.build_dictionary(args, theta_min, theta_max, theta_step, tau_min, tau_max, tau_step)
    x2 = FISTA.FISTA_Lasso(A_fne, r2, lam, max_iter=3000, tol=1e-3, verbose=True)
    P2 = np.abs(x2).reshape(len(theta_grid_fine), len(tau_grid_fine))
    Plot.plot_spectrum(frame_idx, tau_grid_fine, theta_grid_fine, P2, args, title="Refine strongest peak")

    Plot.save_as_mat(tau_grid_fine, theta_grid_fine, x2, frame_idx, title="Refine1")


    



    #subtract the contribution of the top peak
    y1_hat_top = utils.reconstruct(A, x_cvx, itheta, itau, theta_grid, tau_grid, radius=1)
    r1 = y - y1_hat_top


    x2 = FISTA.FISTA_Lasso(A, r1, lam, max_iter=3000, tol=1e-3, verbose=True)
    P2 = np.abs(x2).reshape(len(theta_grid), len(tau_grid))
    peaks = utils.find_peaks(P2, theta_grid, tau_grid)
    itheta, itau, theta_val, tau_val, _ = peaks[0]
    
    Plot.plot_spectrum(frame_idx, tau_grid, theta_grid, P2, args, title="Residual after subtracting strongest peak")

    theta_min = theta_val - 10/2
    theta_max = theta_val + 10/2
    tau_min = tau_val - 1e-9/2
    tau_max = tau_val + 1e-9/2
    theta_step = 0.2
    tau_step = 2e-11

    A_fne, theta_grid_fine, tau_grid_fine = utils.build_dictionary(args, theta_min, theta_max, theta_step, tau_min, tau_max, tau_step)
    x2 = FISTA.FISTA_Lasso(A_fne, r1, lam, max_iter=3000, tol=1e-3, verbose=True)
    P2 = np.abs(x2).reshape(len(theta_grid_fine), len(tau_grid_fine))
    Plot.plot_spectrum(frame_idx, tau_grid_fine, theta_grid_fine, P2, args, title="Refine 2nd strongest peak")

    '''


    frame_idx = 0
    #print(f"🚀 Processing Frame {frame_idx}... ")


    #_,_,_ = LASSO.gen_L2_LASSO_prob(CSI, args, frame_idx, Nrx=args.num_Rx, Nsubc=args.num_subcarriers, lam=args.lam)
    #MUSIC.gen_MUSIC_spectrum(frame_idx, CSI, args, avg=True, title=f"MUSIC @ frame{frame_idx}")

    utils.save_all_MUSIC_spectrum_as_mat(CSI, args, f_start=100, f_end=105, f_step=1)
    utils.save_all_LASSO_spectrum_as_mat(CSI, args, f_start=100, f_end=105, f_step=1)



    



