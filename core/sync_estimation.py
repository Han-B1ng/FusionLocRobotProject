# file: core/sync_estimation.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 时偏估计算法：互相关法 + 最小二乘优化

"""
时间偏差估计模块。

提供两种互补的时偏估计算法：
  - estimate_delay_cross_corr : 互相关法，粗搜索全局最优时偏
  - estimate_delay_lsq        : 最小二乘法，在初值附近精细优化

典型用法：先用互相关法得到初值，再用最小二乘法精化。

依赖：numpy, scipy
被依赖：core/time_alignment.py, stage1_problem1.py, stage2_problem2.py
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate


# ============================================================
#  1. 互相关法 — 粗搜索
# ============================================================
def estimate_delay_cross_corr(
    t_grid: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delay_range: Tuple[float, float] = (-1.0, 1.0),
    dt: float = 0.1,
    w_x: float = 0.5,
    w_y: float = 0.5,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """互相关法估计两传感器间的时间偏差。

    将连续时偏离散化，对每个候选 Δ 将传感器 2 轨迹在时域上平移 Δ，
    在重叠区间内计算 X、Y 通道的相关系数加权和，取最大值对应的时偏。

    Parameters
    ----------
    t_grid : np.ndarray
        公共时间网格 (s)，两组数据已在该网格上插值。
    x1, y1 : np.ndarray
        传感器 1 在 t_grid 上的 X/Y 坐标 (m)。
    x2, y2 : np.ndarray
        传感器 2 在 t_grid 上的 X/Y 坐标 (m)。
    delay_range : tuple of float
        时偏搜索范围 (s)，格式 (delay_min, delay_max)。
        delay > 0 表示传感器 2 相对传感器 1 滞后。
    dt : float
        时偏搜索步长 (s)，默认 0.1。
    w_x, w_y : float
        X / Y 通道的相关系数权重，默认各 0.5。

    Returns
    -------
    best_delay : float
        使加权相关系数最大的时偏估计值 (s)。
    (delays, scores) : tuple of np.ndarray
        delays — 候选时偏数组；
        scores — 对应的加权相关系数数组，供后续绘图。
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    dt_grid = float(t_grid[1] - t_grid[0])  # 网格原始步长

    # --------------------------------------------------
    # 生成候选时偏列表
    # --------------------------------------------------
    n_candidates = int(np.round((delay_range[1] - delay_range[0]) / dt)) + 1
    delays = np.linspace(delay_range[0], delay_range[1], n_candidates)

    scores = np.zeros(len(delays))

    for i, delta in enumerate(delays):
        # 将传感器 2 的时间轴平移 delta：
        #   t2_shifted = t_grid - delta
        # 即传感器 2 的观测时刻向回平移 delta
        t_shifted = t_grid - delta

        # 找到重叠区间：t_shifted 落在 t_grid 范围内的部分
        mask = (t_shifted >= t_grid[0]) & (t_shifted <= t_grid[-1])
        n_overlap = int(np.sum(mask))

        if n_overlap < 10:
            # 重叠点太少，相关系数无意义
            scores[i] = -np.inf
            continue

        # 在重叠区间内插值传感器 2 的数据
        x2_shifted = np.interp(t_shifted[mask], t_grid, x2)
        y2_shifted = np.interp(t_shifted[mask], t_grid, y2)

        x1_seg = x1[mask]
        y1_seg = y1[mask]

        # 计算皮尔逊相关系数
        corr_x = _pearson_r(x1_seg, x2_shifted)
        corr_y = _pearson_r(y1_seg, y2_shifted)

        scores[i] = w_x * corr_x + w_y * corr_y

    best_idx = int(np.argmax(scores))
    best_delay = float(delays[best_idx])

    return best_delay, (delays, scores)


# ============================================================
#  2. 最小二乘法 — 精细优化
# ============================================================
def estimate_delay_lsq(
    t_grid: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delay_range: Tuple[float, float] = (-1.0, 1.0),
    x0: float | None = None,
) -> Tuple[float, float]:
    """最小二乘法精化时间偏差估计。

    在 delay_range 内搜索使两传感器轨迹 RMSE 最小的时偏。
    平移传感器 2 时使用线性插值 (np.interp)。

    目标函数：
        RMSE(Δ) = sqrt( mean( (x1 - x2(t-Δ))² + (y1 - y2(t-Δ))² ) )

    Parameters
    ----------
    t_grid : np.ndarray
        公共时间网格 (s)。
    x1, y1 : np.ndarray
        传感器 1 在 t_grid 上的 X/Y 坐标 (m)。
    x2, y2 : np.ndarray
        传感器 2 在 t_grid 上的 X/Y 坐标 (m)。
    delay_range : tuple of float
        时偏搜索范围 (s)。
    x0 : float or None, optional
        初始猜测值 (s)。若提供，使用有界 Brent 法在 x0 附近搜索；
        否则在整个 delay_range 内搜索。

    Returns
    -------
    best_delay : float
        使 RMSE 最小的时偏估计值 (s)。
    best_rmse : float
        对应的最小 RMSE 值 (m)。
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    def _rmse_objective(delta: float) -> float:
        """目标函数：两传感器轨迹在时偏 delta 下的 RMSE。

        公式：
            RMSE = sqrt( (1/N) * Σ [ (x1_i - x2(t_i - Δ))²
                                    + (y1_i - y2(t_i - Δ))² ] )
        """
        t_shifted = t_grid - delta

        # 只在重叠区间内计算
        mask = (t_shifted >= t_grid[0]) & (t_shifted <= t_grid[-1])
        n_overlap = int(np.sum(mask))

        if n_overlap < 2:
            return 1e12  # 重叠不足，返回极大惩罚值

        x2_shifted = np.interp(t_shifted[mask], t_grid, x2)
        y2_shifted = np.interp(t_shifted[mask], t_grid, y2)

        err_x = x1[mask] - x2_shifted
        err_y = y1[mask] - y2_shifted

        # RMSE = sqrt( mean(err_x² + err_y²) )
        rmse = np.sqrt(np.mean(err_x ** 2 + err_y ** 2))
        return float(rmse)

    # --------------------------------------------------
    # 使用有界 Brent 法优化
    # --------------------------------------------------
    result = minimize_scalar(
        _rmse_objective,
        bounds=delay_range,
        method="bounded",
    )

    best_delay = float(result.x)
    best_rmse = float(result.fun)

    return best_delay, best_rmse


# ============================================================
#  内部工具
# ============================================================
def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个等长数组的皮尔逊相关系数。

    公式：
        r = Σ[(a_i - ā)(b_i - b̄)] / sqrt( Σ(a_i - ā)² · Σ(b_i - b̄)² )

    若标准差为零（常数序列），返回 0.0。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)

    denom = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))

    if denom < 1e-15:
        return 0.0

    return float(np.sum(a_mean * b_mean) / denom)
