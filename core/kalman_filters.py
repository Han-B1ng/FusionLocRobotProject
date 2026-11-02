import numpy as np


def _predict(x_prev, P_prev, F, Q):
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.T + Q
    return x_pred, P_pred


def _update(x_pred, P_pred, z, H, R):
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    residual = z - H @ x_pred
    x_update = x_pred + K @ residual
    P_update = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_update, P_update, residual, K


def _rts_smooth(obs_states, obs_P, obs_F, obs_Q):
    N = len(obs_states)
    x_smooth = [None] * N
    P_smooth = [None] * N

    x_smooth[-1] = obs_states[-1].copy()
    P_smooth[-1] = obs_P[-1].copy()

    for k in range(N - 2, -1, -1):
        F_k = obs_F[k]
        Q_k = obs_Q[k]

        x_pred, P_pred = _predict(obs_states[k], obs_P[k], F_k, Q_k)
        C = obs_P[k] @ F_k.T @ np.linalg.inv(P_pred)

        x_smooth[k] = obs_states[k] + C @ (x_smooth[k + 1] - x_pred)
        P_smooth[k] = obs_P[k] + C @ (P_smooth[k + 1] - P_pred) @ C.T

    return np.array(x_smooth), np.array(P_smooth)


def fuse_sensors(t1, x1, y1, t2, x2, y2,
                 target_freq=10.0,
                 R1_est=None, R2_est=None,
                 ar1_alpha=0.0, ar1_bias_var=0.01):
    from config import filter_config

    dt = 1.0 / target_freq
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    t_grid = np.arange(t_start, t_end + dt, dt)
    n = len(t_grid)

    x1_interp = np.interp(t_grid, t1, x1)
    y1_interp = np.interp(t_grid, t1, y1)
    x2_interp = np.interp(t_grid, t2, x2)
    y2_interp = np.interp(t_grid, t2, y2)

    if R1_est is None:
        R1 = np.diag(filter_config.R1)
    else:
        R1 = R1_est
    if R2_est is None:
        R2 = np.diag(filter_config.R2)
    else:
        R2 = R2_est

    Q_cfg = filter_config.Q
    q_pos_x = Q_cfg[0]  # X位置过程噪声
    q_pos_y = Q_cfg[1]  # Y位置过程噪声
    q_vel_x = Q_cfg[2]  # X速度过程噪声
    q_vel_y = Q_cfg[3]  # Y速度过程噪声
    q_bias = ar1_bias_var  # AR1偏差噪声

    obs_states = []
    obs_P = []
    obs_F = []
    obs_Q = []
    R_matrices = []

    x0 = np.array([x1_interp[0], 0.0, y1_interp[0], 0.0, 0.0, 0.0], dtype=np.float64)
    P0 = np.diag([1.0, 0.5, 1.0, 0.5, 0.1, 0.1])

    x_est = x0
    P_est = P0

    for i in range(n):
        rho = np.exp(-ar1_alpha * dt)
        F = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, rho, 0],
            [0, 0, 0, 0, 0, rho]
        ], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0, 0] = q_pos_x * dt ** 2
        Q[0, 1] = q_pos_x * dt
        Q[1, 0] = q_pos_x * dt
        Q[1, 1] = q_vel_x
        Q[2, 2] = q_pos_y * dt ** 2
        Q[2, 3] = q_pos_y * dt
        Q[3, 2] = q_pos_y * dt
        Q[3, 3] = q_vel_y
        Q[4, 4] = q_bias * (1 - rho ** 2)
        Q[5, 5] = q_bias * (1 - rho ** 2)

        x_pred, P_pred = _predict(x_est, P_est, F, Q)

        z1 = np.array([x1_interp[i], y1_interp[i]])
        H1 = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0]], dtype=np.float64)
        x1_update, P1_update, _, _ = _update(x_pred, P_pred, z1, H1, R1)

        z2 = np.array([x2_interp[i], y2_interp[i]])
        H2 = np.array([[1, 0, 0, 0, -1, 0],
                       [0, 0, 1, 0, 0, -1]], dtype=np.float64)
        x_final, P_final, _, _ = _update(x1_update, P1_update, z2, H2, R2)

        obs_states.append(x_final.copy())
        obs_P.append(P_final.copy())
        obs_F.append(F.copy())
        obs_Q.append(Q.copy())
        R_matrices.append([R1, R2])

        x_est = x_final
        P_est = P_final

    x_smooth, P_smooth = _rts_smooth(obs_states, obs_P, obs_F, obs_Q)

    bias_x_arr = x_smooth[:, 4]
    bias_y_arr = x_smooth[:, 5]

    return t_grid, x_smooth[:, 0], x_smooth[:, 2], bias_x_arr, bias_y_arr


def estimate_adaptive_R(
        t1: np.ndarray,
        x1_d: np.ndarray,
        y1_d: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        bias_x: float = 0.0,
        bias_y: float = 0.0,
        method: str = "mad",
        upper_bound_multiplier: float = 5.0,
) -> tuple:
    from config import filter_config

    def _mad(x: np.ndarray) -> float:
        return float(np.median(np.abs(x - np.median(x)))) * 1.4826

    dt1 = np.diff(t1)
    dt1 = np.where(dt1 > 0, dt1, np.median(dt1))

    vx1 = np.diff(x1_d) / dt1
    vy1 = np.diff(y1_d) / dt1

    ax1 = np.diff(vx1) / dt1[:-1]
    ay1 = np.diff(vy1) / dt1[:-1]

    if method == "mad":
        dt_med = float(np.median(dt1))
        sigma_ax = _mad(ax1) * dt_med ** 2
        sigma_ay = _mad(ay1) * dt_med ** 2
    else:
        dt_med = float(np.median(dt1))
        sigma_ax = float(np.std(ax1)) * dt_med ** 2
        sigma_ay = float(np.std(ay1)) * dt_med ** 2

    R1 = np.diag([max(sigma_ax ** 2, 1e-6),
                  max(sigma_ay ** 2, 1e-6)])

    dx_comp = dx - bias_x
    dy_comp = dy - bias_y

    if method == "mad":
        sx = _mad(dx_comp)
        sy = _mad(dy_comp)
        rho = _mad(dx_comp * dy_comp) / (sx * sy + 1e-12)
        rho = np.clip(rho, -0.99, 0.99)
    else:
        sx = float(np.std(dx_comp))
        sy = float(np.std(dy_comp))
        rho = float(np.corrcoef(dx_comp, dy_comp)[0, 1])

    R2 = np.array([
        [sx ** 2, rho * sx * sy],
        [rho * sx * sy, sy ** 2],
    ])
    R2 = (R2 + R2.T) / 2
    eigvals = np.linalg.eigvalsh(R2)
    if np.min(eigvals) < 1e-8:
        R2 += np.eye(2) * (1e-8 - np.min(eigvals))

    default_R1 = np.diag(filter_config.R1)
    default_R2 = np.diag(filter_config.R2)

    R1_upper = default_R1 * upper_bound_multiplier
    R2_upper = default_R2 * upper_bound_multiplier

    R1 = np.clip(R1, a_min=np.diag(default_R1), a_max=np.diag(R1_upper))
    R2 = np.clip(R2, a_min=np.diag(default_R2), a_max=np.diag(R2_upper))

    R1 = np.diag(np.diag(R1))
    R2 = np.diag(np.diag(R2))

    return R1, R2


def estimate_ar1_params(dx, dy, dt_ref=0.1):

    def fit_ar1(res):
        n = len(res)
        if n < 10:
            return 0.0, np.var(res)
        x_prev = res[:-1]
        x_curr = res[1:]
        rho = np.corrcoef(x_prev, x_curr)[0, 1]
        rho = np.clip(rho, 0.01, 0.999)
        alpha = -np.log(rho) / dt_ref
        var = np.var(res)
        return alpha, var

    alpha_x, var_x = fit_ar1(dx)
    alpha_y, var_y = fit_ar1(dy)
    alpha = (alpha_x + alpha_y) / 2.0
    bias_var = (var_x + var_y) / 2.0
    return alpha, bias_var
