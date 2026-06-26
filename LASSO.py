import numpy as np
import MUSIC
import Plot

def build_dictionary(args, theta_min=-90, theta_max=90, theta_step=4, tau_min=0, tau_max=1.5e-8, tau_step=4e-10):
    # theta candidate
    theta_grid = np.arange(args.theta_min, args.theta_max + 1, args.theta_step) #181 points

    # tau candidate
    tau_grid =np.arange(args.tau_min, args.tau_max, args.tau_step) # 40 points


    L = args.num_Rx * args.num_subcarriers
    #print(f"θ grid size: {len(theta_grid)},  τ grid size: {len(tau_grid)}")
    G = len(theta_grid) * len(tau_grid)
    # Dictionary A
    A = np.zeros((L, G), dtype=np.complex128)
    steering_vector = MUSIC.SteeringVector(args)
    idx = 0
    for i in theta_grid:
        for j in tau_grid:
            v = np.kron(
                steering_vector.steering_vector_AoA(i, args.num_Rx),
                steering_vector.steering_vector_ToF(j, args.num_subcarriers, 1),
            )
            A[:, idx] = v / (np.linalg.norm(v) + 1e-12)  # normalize columns
            idx += 1
    return A, theta_grid, tau_grid

def build_dictionary_adp(Nrx, Nsubc, args, thteta_min=-90, theta_max=90, theta_step=3, tau_min=0, tau_max=2.5e-8, tau_step=4e-10):
    # theta candidate
    theta_grid = np.arange(thteta_min, theta_max + 1, theta_step) #181 points

    # tau candidate
    tau_grid =np.arange(tau_min, tau_max + 1e-12, tau_step) # 40 points


    L = Nrx * Nsubc
    #print(f"θ grid size: {len(theta_grid)},  τ grid size: {len(tau_grid)}")
    G = len(theta_grid) * len(tau_grid)
    # Dictionary A
    A = np.zeros((L, G), dtype=np.complex128)
    steering_vector = MUSIC.SteeringVector(args)
    idx = 0
    for i in theta_grid:
        for j in tau_grid:
            v = np.kron(
                steering_vector.steering_vector_AoA(i, args.num_Rx),
                steering_vector.steering_vector_ToF(j, Nsubc, 1),
            )
            A[:, idx] = v / (np.linalg.norm(v) + 1e-12)  # normalize columns
            idx += 1
    return A, theta_grid, tau_grid

def build_Y_packets(CSI, frame_idx, Nsubc, K_frame=11):
    if K_frame > 1:
        # 取 frame_idx 前後各 K//2 個（可自行改成只取往前）
        half = K_frame // 2
        idxs = np.arange(frame_idx - half, frame_idx + half + 1)
        idxs = idxs[(idxs >= 0) & (idxs < CSI.shape[0])]

        Ys = []
        for t in idxs:
            y = np.mean(CSI[t-10:t+10, 0, :, 0:Nsubc], axis=0).reshape(-1)   
            y = y / (np.linalg.norm(y) + 1e-12)
            Ys.append(y)
        Y = np.stack(Ys, axis=1)   # shape: (L, K_eff)
        return Y
    elif K_frame == 1:
        y = CSI[frame_idx, 0, :, 0:Nsubc].reshape(-1)
        y = y / (np.linalg.norm(y) + 1e-12)
        return y.reshape(-1)
    else:
        raise ValueError("K_frame must be >= 1")
  
def gen_L1_LASSO_prob(CSI, args, frame_idx, Nrx, Nsubc, lam=0.1):

    A, theta_grid, tau_grid = build_dictionary(args)
    y = build_Y_packets(CSI, frame_idx, Nsubc, K_frame=1)
    print(f"CSI vec size: {len(y)}, θ grid size: {len(theta_grid)},  τ grid size: {len(tau_grid)}")

    # CVXPY
    print("🐌 Solving CVXPY... ")
    #x_cvx, _ = LASSOResult.solve_L1_LASSO_cvx(A, y, lam, solver="SCS", eps=1e-5, max_iter=3000, verbose=True)
    x_cvx = FISTA.FISTA_Lasso(A, y, lam, max_iter=3000, tol=1e-3, verbose=True)
    x_cvx = np.abs(x_cvx).reshape(len(theta_grid), len(tau_grid))

    Plot.save_as_mat(tau_grid, theta_grid, x_cvx, frame_idx)



    Plot.plot_spectrum(frame_idx, tau_grid, theta_grid, x_cvx, args, title=f"CVXPY lam={lam}")

def gen_L2_LASSO_prob(CSI, args, frame_idx, Nrx, Nsubc, lam=0.1):

    A, theta_grid, tau_grid = build_dictionary(args)
    Y = build_Y_packets(CSI, frame_idx, Nsubc, K_frame=args.multi_frame)

    # --- SVD & Dynamic K Selection ---
    print(f"Y.shape before SVD {Y.shape}")
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    # 方法 1: 使用能量比例
    energy_thresh = args.energy_thresh
    S_sq = S**2
    K_subspace = np.searchsorted(np.cumsum(S_sq), np.sum(S_sq) * energy_thresh) + 1
    K_subspace = np.clip(K_subspace, 2, 10)
    print(f"Dynamically selected K={K_subspace} (Energy Thresh={energy_thresh})")

    Y = U[:, :K_subspace] @ np.diag(S[:K_subspace])
    print(f"Y.shape after SVD{Y.shape}")
    

    print("🐌 Solving Group L2 Lasso... ")

    X_cvx = FISTA.FISTA_group_Lasso(A, Y, lam, max_iter=args.max_iter, tol=args.tol, verbose=True)
    # X_cvx.shape(G * K)
    X_cvx = np.linalg.norm(X_cvx, axis=1)
    X_cvx = np.abs(X_cvx).reshape(len(theta_grid), len(tau_grid))

    pad_theta = 2  # 角度軸邊界
    pad_tau = 2    # 延遲軸邊界
    
    # 將邊緣設為 0
    X_cvx[:pad_theta, :] = 0  # 上邊界
    X_cvx[-pad_theta:, :] = 0 # 下邊界
    X_cvx[:, :pad_tau] = 0    # 左邊界
    X_cvx[:, -pad_tau:] = 0   # 右邊界

    #Plot.save_as_mat(tau_grid, theta_grid, X_cvx, frame_idx)

    Plot.plot_spectrum(frame_idx, tau_grid, theta_grid, X_cvx, args, title=f"Group Lasso lam={lam}")

    return X_cvx, theta_grid, tau_grid

def reconstruct(A, x_esti, peaks_i, theta_grid, tau_grid, radius=1):
    idx_theta, idx_tau, _, _, _ = peaks_i
    x_top = np.zeros_like(x_esti)
    for i in range(max(0, idx_theta - radius), min(len(theta_grid), idx_theta + radius+1)):
        for j in range(max(0, idx_tau - radius), min(len(tau_grid), idx_tau + radius+1)):
            idx2 = i * len(tau_grid) + j
            x_top[idx2] = x_esti[idx2]
    return A @ x_top

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
    def prox_op_group(Z, thr):
        # Z: (G, K)
        # thr = lam / L
        row_norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return np.maximum(0.0, 1.0 - thr / row_norm)* Z

    @staticmethod
    def FISTA_group_Lasso(A, Y, lam=0.1, max_iter=1000, tol=1e-5, verbose=True):
        """
        Solve: min_X 0.5||AX - Y||_F^2 + lam * sum_i ||X_i||_2
        A: (L, G) complex
        Y: (L, K) complex
        X: (G, K) complex
        """
        L = FISTA.estimate_max_eigenval(A)  # Lipschitz constant of grad (A^H A)
        G = A.shape[1]
        K = Y.shape[1]
        X = np.zeros((G, K), dtype=np.complex128)
        Z = X.copy()
        t = 1.0

        prev_obj = None
        for it in range(max_iter):
            # grad of 0.5||AZ - Y||_F^2 is A^H(AZ - Y)
            grad = A.conj().T @ (A @ Z - Y)
            X_new = FISTA.prox_op_group(Z - (1.0 / L) * grad, (1.0 / L) * lam)

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            Z = X_new + ((t - 1) / t_new) * (X_new - X)
            t = t_new

            # convergence check
            rel = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + 1e-12)
            X = X_new

            if verbose and (it % 250 == 0 or it == max_iter - 1):
                data = 0.5 * np.linalg.norm(A @ X - Y, 'fro')**2
                reg = lam * np.sum(np.linalg.norm(X, axis=1))
                obj = data + reg
                if prev_obj is None:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}")
                else:
                    print(f"[{it:4d}] rel={rel:.3e}, obj={obj:.6e}, Δobj={obj-prev_obj:.3e}")
                prev_obj = obj

            if rel < tol:
                break

        return X
