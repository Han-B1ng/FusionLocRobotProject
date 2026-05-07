# file: core/kalman_filters.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : EKF 传感器融合：6维状态 [x,y,vx,vy,bx,by]，恒速度模型

"""
卡尔曼滤波融合模块。

状态向量：[x, y, vx, vy, bx, by]
  - (x, y)   : 位置 (m)
  - (vx, vy)  : 速度 (m/s)
  - (bx, by)  : 传感器 2 相对传感器 1 的系统偏差 (m)

观测模型：
  - 传感器 1: z = [x, y]            （直接观测位置）
  - 传感器 2: z = [x + bx, y + by]  （位置 + 系统偏差）

融合策略：
  将两路观测合并按时间排序，逐条执行预测-更新，
  最后在 10Hz 目标网格上输出纯预测结果。

自适应观测噪声：
  当传入 R1_est / R2_est 时，使用数据驱动的观测噪声替代
  FilterConfig 中的固定经验值。R1_est 和 R2_est 为 2×2 矩阵，
  通常由 stage2_problem2.py 中从残差统计量估计得到。

  R2 估计的关键修正：
    残差 dx = x_aligned - x_ref 包含两部分：
      - 系统偏差（确定性常量，由 EKF 的 bx/by 状态量建模）
      - 随机噪声（需要被 R2 捕获的不确定性）
    因此在计算 R2 之前，必须先去除残差的中位数（系统偏差的鲁棒估计），
    仅对去均值后的随机波动部分计算协方差，避免系统偏差混入观测噪声。

依赖：numpy, config.py
被依赖：stage2_problem2.py, stage3_problem3.py
"""

from typing import Optional, Tuple

import numpy as np

from config import filter_config, time_config


# ============================================================
#  主函数：传感器融合
# ============================================================
def fuse_sensors(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    target_freq: float = 10.0,
    R1_est: Optional[np.ndarray] = None,
    R2_est: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """使用扩展卡尔曼滤波融合两个传感器的数据。

    Parameters
    ----------
    t1, x1, y1 : np.ndarray
        传感器 1 的时间戳 (s)、X/Y 坐标 (m)。
        已完成时间对齐和去噪。
    t2, x2, y2 : np.ndarray
        传感器 2 的时间戳 (s)、X/Y 坐标 (m)。
        已完成时间对齐和去噪。
    target_freq : float
        目标输出频率 (Hz)，默认 10.0。
    R1_est : np.ndarray or None
        传感器 1 的观测噪声矩阵 (2×2)。
        若为 None，使用 FilterConfig 中的 R1 固定值。
        通常由传感器 1 坐标差分的高频部分协方差估计得到。
    R2_est : np.ndarray or None
        传感器 2 的观测噪声矩阵 (2×2)。
        若为 None，使用 FilterConfig 中的 R2 固定值。
        通常由对齐后残差（去中位数）的协方差矩阵估计得到。

    Returns
    -------
    t_grid : np.ndarray
        目标时间网格 (s)。
    x_fused : np.ndarray
        融合后 X 坐标 (m)。
    y_fused : np.ndarray
        融合后 Y 坐标 (m)。
    bias_x_arr : np.ndarray
        估计的 X 方向偏差序列 (m)。
    bias_y_arr : np.ndarray
        估计的 Y 方向偏差序列 (m)。
    """
    t1 = np.asarray(t1, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    # ======================================================
    # Step 1: 合并观测并按时间排序
    # ======================================================
    observations = []
    for i in range(len(t1)):
        observations.append((t1[i], x1[i], y1[i], 1))
    for i in range(len(t2)):
        observations.append((t2[i], x2[i], y2[i], 2))
    observations.sort(key=lambda o: o[0])

    # ======================================================
    # Step 2: 从 config 读取滤波参数，构建矩阵
    #     若传入了 R1_est / R2_est，则覆盖默认值
    # ======================================================
    fc = filter_config

    P = np.diag(fc.P0)          # 6×6 初始协方差
    Q = np.diag(fc.Q)           # 6×6 过程噪声

    # 观测噪声：优先使用外部传入的自适应估计值
    if R1_est is not None:
        R1 = np.asarray(R1_est, dtype=np.float64)
    else:
        R1 = np.diag(fc.R1)     # 2×2 传感器1观测噪声

    if R2_est is not None:
        R2 = np.asarray(R2_est, dtype=np.float64)
    else:
        R2 = np.diag(fc.R2)     # 2×2 传感器2观测噪声

    # 观测矩阵
    # 传感器 1: z = [x, y] = H1 @ state
    H1 = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float64)

    # 传感器 2: z = [x + bx, y + by] = H2 @ state
    H2 = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)

    # ======================================================
    # Step 3: 初始化状态
    #     [x0, y0, vx=0, vy=0, bx=0, by=0]
    #     初始位置取传感器 1 第一个观测值
    # ======================================================
    x_state = np.array([
        x1[0], y1[0],  # 位置
        0.0, 0.0,       # 速度
        0.0, 0.0,       # 偏差
    ], dtype=np.float64)

    # ======================================================
    # Step 4: 遍历所有观测，执行预测-更新
    # ======================================================
    t_last = observations[0][0]
    obs_states = []   # 存储 (观测时间, 更新后状态)
    obs_P = []        # 存储 (观测时间, 更新后协方差)

    for t_obs, xo, yo, sensor_id in observations:
        # --- 预测 ---
        dt = t_obs - t_last
        if dt > 0:
            x_state, P = _predict(x_state, P, dt, Q)

        # --- 更新 ---
        z = np.array([xo, yo], dtype=np.float64)
        if sensor_id == 1:
            H, R = H1, R1
        else:
            H, R = H2, R2

        x_state, P = _update(x_state, P, z, H, R)

        obs_states.append((t_obs, x_state.copy()))
        obs_P.append((t_obs, P.copy()))
        t_last = t_obs

    # ======================================================
    # Step 5: 在 10Hz 目标网格上输出融合结果
    #     对每个目标时刻，找到最近的观测状态，
    #     用恒速度模型纯预测到目标时刻
    # ======================================================
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    dt_target = 1.0 / target_freq

    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    # 构建观测时间数组用于查找
    obs_times = np.array([s[0] for s in obs_states])

    x_fused = np.zeros(len(t_grid))
    y_fused = np.zeros(len(t_grid))
    bias_x_arr = np.zeros(len(t_grid))
    bias_y_arr = np.zeros(len(t_grid))

    for i, t_target in enumerate(t_grid):
        # 找到最近的不超过 t_target 的观测索引
        idx = np.searchsorted(obs_times, t_target, side="right") - 1
        idx = max(0, min(idx, len(obs_states) - 1))

        t_ref, x_ref = obs_states[idx]

        # 纯预测到目标时刻
        dt_pred = t_target - t_ref
        if dt_pred > 1e-9:
            x_pred, _ = _predict(x_ref, P, dt_pred, Q)
        else:
            x_pred = x_ref

        x_fused[i] = x_pred[0]
        y_fused[i] = x_pred[1]
        bias_x_arr[i] = x_pred[4]
        bias_y_arr[i] = x_pred[5]

    return t_grid, x_fused, y_fused, bias_x_arr, bias_y_arr


# # ============================================================
# #  自适应观测噪声估计（供 stage2_problem2.py 调用）
# # ============================================================
# def estimate_adaptive_R(
#     t1: np.ndarray,
#     x1: np.ndarray,
#     y1: np.ndarray,
#     dx_residual: np.ndarray,
#     dy_residual: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """从数据中自适应估计观测噪声矩阵 R1 和 R2。
#
#     R1 估计策略：
#       对传感器 1 去噪后的位置做一阶前向差分估计速度，
#       计算速度的 2×2 样本协方差，取对角线平均值作为 R1 对角元素。
#       原理：位置测量的高频抖动反映了观测噪声量级，
#       速度协方差的对角均值可作为位置观测噪声的保守近似。
#
#     R2 估计策略（已修正）：
#       对齐后残差 dx = x_aligned - x_ref 包含两部分：
#         (a) 系统偏差（确定性常量，由 EKF 的 bx/by 状态量建模）
#         (b) 随机噪声（需要被 R2 捕获的不确定性）
#       因此先去除残差的中位数（系统偏差的鲁棒估计），
#       仅对去均值后的随机波动部分计算 2×2 协方差作为 R2。
#       这样 R2 只反映传感器 2 的随机测量噪声，不混入系统偏差。
#
#       修正前：R2 = cov(dx, dy)         — 包含系统偏差，对角约 77 m²
#       修正后：R2 = cov(dx-median(dx), dy-median(dy))  — 仅随机噪声，对角约 1~5 m²
#
#     Parameters
#     ----------
#     t1 : np.ndarray
#         传感器 1 时间戳 (s)。
#     x1, y1 : np.ndarray
#         传感器 1 去噪后的位置 (m)。
#     dx_residual, dy_residual : np.ndarray
#         对齐后传感器 2 相对传感器 1 的 X/Y 残差 (m)。
#
#     Returns
#     -------
#     R1_est : np.ndarray
#         传感器 1 观测噪声矩阵 (2×2)。
#     R2_est : np.ndarray
#         传感器 2 观测噪声矩阵 (2×2)，仅含随机噪声成分。
#     """
#     t1 = np.asarray(t1, dtype=np.float64)
#     x1 = np.asarray(x1, dtype=np.float64)
#     y1 = np.asarray(y1, dtype=np.float64)
#     dx_residual = np.asarray(dx_residual, dtype=np.float64)
#     dy_residual = np.asarray(dy_residual, dtype=np.float64)
#
#     # --- R1 估计 ---
#     # 一阶前向差分估计速度
#     dt1 = np.diff(t1)
#     vx1 = np.diff(x1) / dt1
#     vy1 = np.diff(y1) / dt1
#
#     # 速度协方差的对角线平均值，取 1/2 作为位置观测噪声的近似
#     # 原理：Var(x_obs) ≈ 0.5 * Var(v) * dt^2 的量级
#     # 这里直接用速度方差的对角平均作为 R1 对角元素（保守估计）
#     speed_cov = np.cov(vx1, vy1)  # 2×2
#     r1_diag = np.trace(speed_cov) / 2.0  # 对角线平均
#     r1_diag = max(r1_diag, 1e-4)  # 下限保护
#
#     R1_est = np.array([
#         [r1_diag, 0.0],
#         [0.0, r1_diag],
#     ], dtype=np.float64)
#
#     # --- R2 估计（修正：先去除系统偏差）---
#     # 残差 = 系统偏差（常量）+ 随机噪声
#     # 去除中位数后，仅保留随机噪声成分
#     dx_centered = dx_residual - np.median(dx_residual)
#     dy_centered = dy_residual - np.median(dy_residual)
#
#     # 对去均值后的随机波动计算 2×2 样本协方差矩阵
#     residual_centered = np.vstack([dx_centered, dy_centered])  # 2×N
#     R2_est = np.cov(residual_centered)  # 2×2
#
#     # 确保正定（对角线下限保护）
#     R2_est[0, 0] = max(R2_est[0, 0], 1e-4)
#     R2_est[1, 1] = max(R2_est[1, 1], 1e-4)
#
#     return R1_est, R2_est
# core/kalman_filters.py

import numpy as np
from config import filter_config


def _mad(x: np.ndarray) -> float:
    """中位绝对偏差 (MAD)，高斯等效标准差 = MAD × 1.4826。"""
    return float(np.median(np.abs(x - np.median(x)))) * 1.4826


def _covariance_from_residuals(
    dx: np.ndarray,
    dy: np.ndarray,
    method: str = "mad",
) -> np.ndarray:
    """从残差序列估计 2×2 观测噪声协方差矩阵。

    Parameters
    ----------
    dx, dy : 残差序列
    method : "mad" 使用中位绝对偏差（鲁棒）
             "std" 使用标准差（经典）

    Returns
    -------
    R : (2, 2) 协方差矩阵
    """
    if method == "mad":
        sx = _mad(dx)
        sy = _mad(dy)
        # 协方差也用 MAD 风格的鲁棒估计
        rho = _mad(dx * dy) / (sx * sy + 1e-12)
        rho = np.clip(rho, -0.99, 0.99)
    else:
        sx = float(np.std(dx))
        sy = float(np.std(dy))
        rho = float(np.corrcoef(dx, dy)[0, 1])

    R = np.array([
        [sx ** 2,        rho * sx * sy],
        [rho * sx * sy,  sy ** 2      ],
    ])
    return R


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
    """自适应观测噪声估计（改进版）。

    改进点：
      1. 使用 MAD 替代方差，对极值更鲁棒
      2. 接收 bias_x/bias_y 参数，在估计 R 前补偿系统偏差
      3. 对 R 施加上界约束，防止过度放大

    Parameters
    ----------
    t1, x1_d, y1_d : 传感器 1 的时间与去噪坐标
    dx, dy : 传感器 2 相对于传感器 1 的残差（已去中位数）
    bias_x, bias_y : 系统偏差估计值，用于补偿残差中的残留偏差
    method : "mad" 或 "std"
    upper_bound_multiplier : R 对角元素的上界 = 默认值 × 此倍数

    Returns
    -------
    R1 : (2, 2) 传感器 1 的观测噪声协方差
    R2 : (2, 2) 传感器 2 的观测噪声协方差
    """
    # ---- R1: 从传感器 1 的速度差分估计 ----
    dt1 = np.diff(t1)
    dt1 = np.where(dt1 > 0, dt1, np.median(dt1))

    vx1 = np.diff(x1_d) / dt1
    vy1 = np.diff(y1_d) / dt1

    # 二阶差分（加速度噪声）更直接反映观测噪声
    ax1 = np.diff(vx1) / dt1[:-1]
    ay1 = np.diff(vy1) / dt1[:-1]

    if method == "mad":
        # 加速度噪声 → 等效位置噪声（量纲缩放因子 ≈ dt²）
        dt_med = float(np.median(dt1))
        sigma_ax = _mad(ax1) * dt_med ** 2
        sigma_ay = _mad(ay1) * dt_med ** 2
    else:
        dt_med = float(np.median(dt1))
        sigma_ax = float(np.std(ax1)) * dt_med ** 2
        sigma_ay = float(np.std(ay1)) * dt_med ** 2

    R1 = np.diag([max(sigma_ax ** 2, 1e-6),
                   max(sigma_ay ** 2, 1e-6)])

    # ---- R2: 从对齐后残差估计（补偿系统偏差后） ----
    # 补偿残留偏差
    dx_comp = dx - bias_x
    dy_comp = dy - bias_y

    R2 = _covariance_from_residuals(dx_comp, dy_comp, method=method)
    # 保底正定
    R2 = (R2 + R2.T) / 2
    eigvals = np.linalg.eigvalsh(R2)
    if np.min(eigvals) < 1e-8:
        R2 += np.eye(2) * (1e-8 - np.min(eigvals))

    # ---- 上界约束 ----
    default_R1 = np.diag(filter_config.R1)  # [[0.5, 0], [0, 0.5]]
    default_R2 = np.diag(filter_config.R2)  # [[0.3, 0], [0, 0.3]]

    R1_upper = default_R1 * upper_bound_multiplier
    R2_upper = default_R2 * upper_bound_multiplier

    # 逐元素 clip：不超过上界，不小于默认值（保守策略）
    R1 = np.clip(R1, a_min=np.diag(default_R1), a_max=np.diag(R1_upper))
    R2 = np.clip(R2, a_min=np.diag(default_R2), a_max=np.diag(R2_upper))

    # 保持对角形式（如果原始设计是对角矩阵）
    R1 = np.diag(np.diag(R1))
    R2 = np.diag(np.diag(R2))

    return R1, R2


# ============================================================
#  内部：预测步
# ============================================================
def _predict(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """恒速度模型预测。

    状态转移矩阵 F(dt)：
        [[1, 0, dt, 0,  0, 0],
         [0, 1, 0,  dt, 0, 0],
         [0, 0, 1,  0,  0, 0],
         [0, 0, 0,  1,  0, 0],
         [0, 0, 0,  0,  1, 0],
         [0, 0, 0,  0,  0, 1]]

    预测公式：
        x_pred = F @ x
        P_pred = F @ P @ F^T + Q
    """
    F = np.eye(6)
    F[0, 2] = dt
    F[1, 3] = dt

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred


# ============================================================
#  内部：更新步（标准卡尔曼增益公式）
# ============================================================
def _update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """标准卡尔曼滤波更新步。

    公式：
        y = z - H @ x_pred          （新息/残差）
        S = H @ P_pred @ H^T + R    （新息协方差）
        K = P_pred @ H^T @ S^{-1}   （卡尔曼增益）
        x = x_pred + K @ y          （状态更新）
        P = (I - K @ H) @ P_pred    （协方差更新）
    """
    I6 = np.eye(6)

    y_res = z - H @ x_pred                       # 新息
    S = H @ P_pred @ H.T + R                     # 新息协方差
    K = P_pred @ H.T @ np.linalg.inv(S)          # 卡尔曼增益

    x_upd = x_pred + K @ y_res                   # 状态更新
    P_upd = (I6 - K @ H) @ P_pred                # 协方差更新

    return x_upd, P_upd
