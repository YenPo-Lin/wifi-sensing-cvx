import numpy as np
import MUSIC
import Plot

def build_overcomplete_dictionary(args):

    theta_grid = np.arange(args.theta_min, args.theta_max + 1, args.lasso_azi_step)
    tau_grid = np.arange(args.tau_min, args.tau_max, args.lasso_tau_step)

    G = len(theta_grid) * len(tau_grid)
    L = args.num_Rx * args.num_subcarriers
    A = np.zeros((L, G), dtype=np.complex128)
    steering_vector = MUSIC.SteeringVector(args)

    idx = 0
    for theta_i in theta_grid:
        for tau_i in tau_grid:
            v = steering_vector.steering_vector_AoA_ToF(
                theta_i,
                tau_i,
                stream_win= args.num_Rx,
                freq_win= args.num_subcarriers,
                freq_hop= 1,
            )
            A[:, idx] = v / (np.linalg.norm(v) + 1e-12)
            idx += 1
    return A, theta_grid, tau_grid

def build_Y_packets(CSI, frame_idx, avg_frames, svd_frames):
    if svd_frames < 1:
        raise ValueError("K_frame must be >= 1")

    num_frames = CSI.shape[0]
    if not (0 <= frame_idx < num_frames):
        raise IndexError(f"frame_idx out of range: {frame_idx}, num_frames={num_frames}")

    avg_frames = max(1, int(avg_frames))
    svd_frames = max(1, int(svd_frames))

    half_snapshots = svd_frames // 2
    frame_indices = np.arange(frame_idx - half_snapshots, frame_idx + half_snapshots + 1)
    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < num_frames)]

    avg_half = avg_frames // 2
    Ys = []
    for t in frame_indices:
        start = max(0, t - avg_half)
        end = min(num_frames, t + avg_half + 1)
        y = np.mean(CSI[start:end, 0, :, :], axis=0).reshape(-1)
        y = y / (np.linalg.norm(y) + 1e-12)
        Ys.append(y)

    if len(Ys) == 1:
        return Ys[0]
    return np.stack(Ys, axis=1)

def gen_L2_LASSO_prob(CSI, args, frame_idx):

    A, theta_grid, tau_grid = build_overcomplete_dictionary(args)
    Y = build_Y_packets(
        CSI,
        frame_idx,
        avg_frames= args.avg_frames, # 每一個 snapshot 的平均(denoising)
        svd_frames= args.svd_frames, # number of snapshots for SVD
    )
    if Y.ndim == 1:
        Y = Y[:, None]

    # --- SVD & Dynamic K Selection ---
    # print(f"Y.shape before SVD {Y.shape}") (num_Rx * num_subcarriers, K)
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    N_subc = args.num_subcarriers
    N_rx = args.num_Rx
    # 使用能量比例
    energy_thresh = args.Sdim_energy_ratio
    S_sq = S**2
    K_subspace = np.searchsorted(np.cumsum(S_sq), np.sum(S_sq) * energy_thresh) + 1
    K_subspace = np.clip(K_subspace, 1, min(Y.shape[1], 10))
    print(f"Dynamically selected K={K_subspace} (Energy Thresh={energy_thresh})")

    Y = U[:, :K_subspace] @ np.diag(S[:K_subspace])
    print(f"Y.shape after SVD{Y.shape}") #(num_Rx * num_subcarriers, ?)
    

    print("🐌 Solving Group L2 Lasso... ")

    X_cvx = FISTA.FISTA_group_Lasso(A, Y, args.lam, max_iter=args.max_iter, tol=args.tol, verbose=True)
    # X_cvx.shape(G * K)
    X_cvx = np.linalg.norm(X_cvx, axis=1)
    X_cvx = np.abs(X_cvx).reshape(len(theta_grid), len(tau_grid))

    Plot.plot_spectrum(frame_idx, tau_grid, theta_grid, X_cvx, args, title=f"Group Lasso λ {args.lam}")

    return X_cvx, theta_grid, tau_grid

def gen_MUSIC_weight_L2_LASSO_prob(CSI, args, frame_idx):

    # Run MUSIC first
    music = MUSIC.Azi_ToF(args)
    x = music.cal_smoothed_csi(frame_idx, CSI)
    Rxx = music.cal_smoothed_cov(x)
    tau_x, theta_x, P_music_x = music.cal_spectrum(Rxx)

    # print(P_music_x.shape) (azi_grid x tau_grid)
    # P_music = 10 * np.log10(1.0 / PP)
    # Return to Linear Scale
    P_music_x = 10.0 ** (P_music_x / 10.0)
    # Normalize
    P_music_x = P_music_x / (np.max(P_music_x) + 1e-12)
    # MUSIC strong -> small penalty, weak -> large penalty
    weight_map = (1.0 / (P_music_x + 1e-6)) ** args.music_reweight_alpha
    # Average to 1
    weight_map = weight_map / (np.mean(weight_map) + 1e-12)
    Plot.plot_spectrum(frame_idx, tau_x, theta_x, weight_map, args, cmap="gray", title="MUSIC weight map")

    # flatten
    w = weight_map.reshape(-1)

    if args.lasso_azi_step != args.theta_step:
        raise ValueError("theta_step and lasso_azi_step must be the same")
    if args.lasso_tau_step != args.tau_step:
        raise ValueError("tau_step and lasso_tau_step must be the same")


    A, theta_grid, tau_grid = build_overcomplete_dictionary(args)
    Y = build_Y_packets(
        CSI,
        frame_idx,
        avg_frames= args.avg_frames, # 每一個 snapshot 的平均(denoising)
        svd_frames= args.svd_frames, # number of snapshots for SVD
    )
    if Y.ndim == 1:
        Y = Y[:, None]

    # --- SVD & Dynamic Selection ---
    # print(f"Y.shape before SVD {Y.shape}") (num_Rx * num_subcarriers, K)
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    # 使用能量比例
    energy_thresh = args.Sdim_energy_ratio
    S_sq = S**2
    K_subspace = np.searchsorted(np.cumsum(S_sq), np.sum(S_sq) * energy_thresh) + 1
    K_subspace = np.clip(K_subspace, 1, min(Y.shape[1], 10))
    print(f"Dynamically selected K={K_subspace} (Energy Thresh={energy_thresh})")

    Y = U[:, :K_subspace] @ np.diag(S[:K_subspace])
    print(f"Y.shape after SVD{Y.shape}") #(num_Rx * num_subcarriers, ?)
    

    print("🐌 Solving Group L2 Lasso... ")

    X_cvx = FISTA.FISTA_group_Lasso(
        A,
        Y,
        args.lam,
        weights=w,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
    )
    # X_cvx.shape(G * K)
    X_cvx = np.linalg.norm(X_cvx, axis=1)
    X_cvx = np.abs(X_cvx).reshape(len(theta_grid), len(tau_grid))

    X_cvx[0, :] = 0
    X_cvx[-1, :] = 0
    X_cvx[:, 0] = 0
    X_cvx[:, -1] = 0

    Plot.plot_spectrum(
        frame_idx, 
        tau_grid, 
        theta_grid, 
        X_cvx, 
        args, 
        title=f"Weighted Group Lasso λ {args.lam}, α{args.music_reweight_alpha}")

    return X_cvx, theta_grid, tau_grid

class FISTA:
    @staticmethod
    def prox_op(x: np.ndarray, thr: float, eps: float = 1e-12) -> np.ndarray:
        """
        Complex soft-thresholding (prox of L1 norm on complex coefficients):
            prox_{thr ||.||_1}(x) = max(0, 1 - thr/|x|) * x
        Keeps phase, shrinks magnitude.
        """
        mag = np.abs(x)
        scale = np.maximum(0.0, 1.0 - thr / (mag + eps))
        return scale * x

    @staticmethod 
    def estimate_max_eigenval(A: np.ndarray, iters: int = 30, eps: float = 1e-12) -> float:
        """
        Estimate Lipschitz constant L = ||A||_2^2 by power iteration.
        Works for complex A.
        """
        G = A.shape[1]
        v = (np.random.randn(G) + 1j * np.random.randn(G)).astype(np.complex128)
        v /= (np.linalg.norm(v) + eps)

        for _ in range(iters):
            v = A.conj().T @ (A @ v)
            v /= (np.linalg.norm(v) + eps)

        Av = A @ v
        L = float(np.real(np.vdot(Av, Av)) + eps)  # ||A v||^2
        return L

    @staticmethod
    def FISTA_Lasso(A, y, lam=0.08, max_iter=1000, tol=1e-5, lipschitz_iters=30, verbose=False,):
        """
        slove min_s 0.5||y - A s||_2^2 + lam ||s||_1
        f(s) = 0.5||y - A s||_2^2
        g(s) = lam ||s||_1
        

        FISTA 程式設計流程
        INPUT: A, y, lam, max_iter, tol
        OUTPUT: s
        Step 0: 初始化
        s = 0, z = s, t = 1, L = max eigenvalue of A^H A

        Step 1: 梯度
        grad = A^H (A z - y)

        Step 2: prox-gradient 迭代更新
        s_new = soft_thresh_complex(z - (1/L) * grad, lam / L)

        Step 3: 更新動量係數
        t_new = 0.5 * (1 + sqrt(1 + 4 t^2))

        Step 4: 更新加速點
        z = s_new + ((t - 1) / t_new) * (s_new - s)

        Step 5: 收斂檢查
        if ||s_new - s|| / (||s|| + eps) < tol: break
        """
        #Step 0: 初始化
        A = np.asarray(A, dtype=np.complex128)
        y = np.asarray(y, dtype=np.complex128).reshape(-1)
        N, M = A.shape
        if y.shape[0] != N:
            raise ValueError(f"Dimension mismatch: A is ({N},{M}) but y is {y.shape}")
        s = np.zeros(M, dtype=np.complex128)
        z = s.copy()
        t = 1.0
        L = FISTA.estimate_max_eigenval(A, iters=lipschitz_iters)

        prev_obj = None

        #Step 1-5: Main loop
        for it in range(max_iter):
            grad = A.conj().T @ (A @ z - y)
            # step=1/L
            s_new = FISTA.prox_op(z - (1/L) * grad, lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            z = s_new + ((t - 1.0) / (t_new + 1e-12)) * (s_new - s)

            # convergence check
            rel = np.linalg.norm(s_new - s) / (np.linalg.norm(s) + 1e-12)
            s = s_new
            t = t_new

            if verbose and (it % 100 == 0 or it == max_iter - 1):
                r = y - A @ s
                obj = 0.5 * np.vdot(r, r).real + lam * np.sum(np.abs(s))
                if prev_obj is None:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}")
                else:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}, Δobj={obj-prev_obj:.3e}")
                prev_obj = obj

            if rel < tol:
                if verbose:
                    print(f"FISTA converged @ iter {it}, rel={rel:.3e}")
                break
        return s
    
    @staticmethod
    def prox_op_group(Z, thr, weights=None):
        # Z: (G, K)
        # thr = lam / L
        row_norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        if weights is None:
            thr_vec = thr
        else:
            weights = np.asarray(weights, dtype=float).reshape(-1, 1)
            thr_vec = thr * weights
        return np.maximum(0.0, 1.0 - thr_vec / row_norm) * Z

    @staticmethod
    def FISTA_group_Lasso(A, Y, lam=0.1, weights=None, max_iter=1000, tol=1e-5, verbose=True):
        """
        Solve: min_X 0.5||AX - Y||_F^2 + lam * sum_i ||X_i||_2
        A: (L, G) complex
        Y: (L, K) complex
        X: (G, K) complex
        """
        L = FISTA.estimate_max_eigenval(A)  # Lipschitz constant of grad (A^H A)
        G = A.shape[1]
        K = Y.shape[1]
        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if weights.shape[0] != G:
                raise ValueError(f"weights length must be {G}, got {weights.shape[0]}")
        X = np.zeros((G, K), dtype=np.complex128)
        Z = X.copy()
        t = 1.0

        prev_obj = None
        for it in range(max_iter):
            # grad of 0.5||AZ - Y||_F^2 is A^H(AZ - Y)
            grad = A.conj().T @ (A @ Z - Y)
            X_new = FISTA.prox_op_group(Z - (1.0 / L) * grad, (1.0 / L) * lam, weights=weights)

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            Z = X_new + ((t - 1) / t_new) * (X_new - X)
            t = t_new

            # convergence check
            rel = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + 1e-12)
            X = X_new

            if verbose and (it % 250 == 0 or it == max_iter - 1):
                data = 0.5 * np.linalg.norm(A @ X - Y, 'fro')**2
                row_norm = np.linalg.norm(X, axis=1)
                if weights is None:
                    reg = lam * np.sum(row_norm)
                else:
                    reg = lam * np.sum(weights * row_norm)
                obj = data + reg
                if prev_obj is None:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}")
                else:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}, Δobj={obj-prev_obj:.3e}")
                prev_obj = obj

            if rel < tol:
                break

        return X
