# file: core/time_alignment.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 时间对齐主流程：插值 + 时偏估计 + 生成10Hz对齐轨迹

"""
时间对齐主流程模块。

将两个不同步、不同频率的传感器数据对齐到统一的 10Hz 时间网格，
并融合为单一轨迹输出。

流程：
  1. 插值到公共网格（粗对齐）
  2. 互相关法估计时偏（粗搜索）
  3. 最小二乘法精化时偏（精细优化）
  4. 修正传感器 2 时间戳并重插值
  5. 加权平均融合

依赖：core/interpolation.py, core/sync_estimation.py, config.py
被依赖：stage1_problem1.py, stage2_problem2.py, stage3_problem3.py
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from config import alignment_config, time_config
from core.interpolation import interp_to_common_grid, resample_to_target
from core.sync_estimation import (
    estimate_delay_cross_corr,
    estimate_delay_lsq,
)


# ============================================================
#  主函数：时间对齐 + 融合
# ============================================================
def align_sensors(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    target_freq: Optional[float] = None,
    delay_range: Optional[Tuple[float, float]] = None,
    method: str = "cubic",
    w1: float = 0.5,
    w2: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """将两组传感器数据对齐并融合为统一的 10Hz 轨迹。

    Parameters
    ----------
    t1, x1, y1 : np.ndarray
        传感器 1 的时间戳 (s)、X/Y 坐标 (m)。
    t2, x2, y2 : np.ndarray
        传感器 2 的时间戳 (s)、X/Y 坐标 (m)。
    target_freq : float or None
        目标输出频率 (Hz)，默认从 config.time_config.target_freq 读取。
    delay_range : tuple of float or None
        时偏搜索范围 (s)，默认从 config.alignment_config.delay_range 读取。
    method : str
        插值方法，'cubic' 或 'linear'，默认 'cubic'。
    w1, w2 : float
        融合时传感器 1 / 传感器 2 的权重，默认各 0.5（等权）。
        例如 w1=0.4, w2=0.6 表示传感器 2 精度更高。

    Returns
    -------
    delay_fine : float
        精化后的时偏估计值 (s)。
        delay > 0 表示传感器 2 相对传感器 1 滞后。
    t_grid : np.ndarray
        最终输出的 10Hz 时间网格 (s)。
    x_fused : np.ndarray
        融合后的 X 坐标 (m)。
    y_fused : np.ndarray
        融合后的 Y 坐标 (m)。
    """
    # 参数默认值从 config 读取
    if target_freq is None:
        target_freq = time_config.target_freq
    if delay_range is None:
        delay_range = alignment_config.delay_range

    t1 = np.asarray(t1, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    # ==========================================================
    # Step 1: 插值到公共网格（粗对齐，不矫正时偏）
    # ==========================================================
    t_grid_coarse, x1_c, y1_c, x2_c, y2_c = interp_to_common_grid(
        t1, x1, y1,
        t2, x2, y2,
        target_freq=target_freq,
        method=method,
    )

    # ==========================================================
    # Step 2: 互相关法 — 粗搜索时偏
    # ==========================================================
    delay_coarse, (delays, scores) = estimate_delay_cross_corr(
        t_grid_coarse, x1_c, y1_c, x2_c, y2_c,
        delay_range=delay_range,
        dt=alignment_config.corr_window / 20.0,  # 窗口内约 20 个采样点
    )

    # ==========================================================
    # Step 3: 最小二乘法 — 精化时偏
    #     以互相关结果为初值，在其 ±0.2s 范围内精细搜索
    # ==========================================================
    lsq_margin = 0.2  # s
    lsq_bounds = (
        max(delay_range[0], delay_coarse - lsq_margin),
        min(delay_range[1], delay_coarse + lsq_margin),
    )

    delay_fine, rmse_fine = estimate_delay_lsq(
        t_grid_coarse, x1_c, y1_c, x2_c, y2_c,
        delay_range=lsq_bounds,
    )

    print(
        f"[align_sensors] 时偏估计: "
        f"粗搜索 = {delay_coarse:+.4f} s, "
        f"精化 = {delay_fine:+.4f} s, "
        f"RMSE = {rmse_fine:.4f} m"
    )

    # ==========================================================
    # Step 4: 用精化时偏修正传感器 2 时间戳，重插值
    #     t2_corrected = t2 - delay_fine
    #     即将传感器 2 的观测时刻向前平移 delay_fine
    # ==========================================================
    t2_corrected = t2 - delay_fine

    # 确定最终输出网格的时间范围（取传感器 1 与修正后传感器 2 的交集）
    t_start = max(t1.min(), t2_corrected.min())
    t_end = min(t1.max(), t2_corrected.max())

    if t_start >= t_end:
        raise ValueError(
            f"[align_sensors] 修正后时间范围无交集: "
            f"s1=[{t1.min():.2f}, {t1.max():.2f}] vs "
            f"s2_corrected=[{t2_corrected.min():.2f}, {t2_corrected.max():.2f}]"
        )

    # 生成精确的 10Hz 输出网格
    dt_target = 1.0 / target_freq
    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    # 传感器 1 在输出网格上插值
    x1_aligned = np.interp(t_grid, t1, x1)
    y1_aligned = np.interp(t_grid, t1, y1)

    # 传感器 2 在修正时间轴上插值到输出网格
    x2_aligned = np.interp(t_grid, t2_corrected, x2)
    y2_aligned = np.interp(t_grid, t2_corrected, y2)

    # ==========================================================
    # Step 5: 加权平均融合
    #     x_fused = w1 * x1_aligned + w2 * x2_aligned
    #     y_fused = w1 * y1_aligned + w2 * y2_aligned
    # ==========================================================
    x_fused = w1 * x1_aligned + w2 * x2_aligned
    y_fused = w1 * y1_aligned + w2 * y2_aligned

    # ==========================================================
    # 平滑性检查：检测融合轨迹中的异常跳变
    # ==========================================================
    _check_smoothness(t_grid, x_fused, y_fused, target_freq)

    print(
        f"[align_sensors] 输出网格: "
        f"[{t_grid[0]:.2f}, {t_grid[-1]:.2f}] s, "
        f"{len(t_grid)} 点, "
        f"频率 = {target_freq:.0f} Hz"
    )

    return delay_fine, t_grid, x_fused, y_fused


# ============================================================
#  内部工具：平滑性检查
# ============================================================
def _check_smoothness(
    t_grid: np.ndarray,
    x_fused: np.ndarray,
    y_fused: np.ndarray,
    target_freq: float,
    sigma_factor: float = 5.0,
) -> None:
    """检查融合轨迹是否存在异常跳变，若有则打印警告。

    判据：相邻点的速度突变（加速度）超过 σ_factor 倍标准差。

    Parameters
    ----------
    t_grid : np.ndarray
        时间网格 (s)。
    x_fused, y_fused : np.ndarray
        融合轨迹的 X/Y 坐标 (m)。
    target_freq : float
        目标频率 (Hz)。
    sigma_factor : float
        异常判定倍数，默认 5.0。
    """
    if len(t_grid) < 3:
        return

    dt = np.diff(t_grid)
    dt[dt == 0] = 1e-10  # 防除零

    # 一阶差分 → 速度
    vx = np.diff(x_fused) / dt
    vy = np.diff(y_fused) / dt
    speed = np.sqrt(vx ** 2 + vy ** 2)

    # 二阶差分 → 加速度
    if len(speed) < 2:
        return

    dt_mid = (dt[:-1] + dt[1:]) / 2.0
    dt_mid[dt_mid == 0] = 1e-10

    ax = np.diff(vx) / dt_mid
    ay = np.diff(vy) / dt_mid
    accel = np.sqrt(ax ** 2 + ay ** 2)

    # 检测异常加速度
    mu = np.mean(accel)
    sigma = np.std(accel)
    threshold = mu + sigma_factor * sigma
    outlier_idx = np.where(accel > threshold)[0]

    if len(outlier_idx) > 0:
        warnings.warn(
            f"[align_sensors] 融合轨迹中检测到 {len(outlier_idx)} 个"
            f"异常跳变点 (加速度阈值={threshold:.3f} m/s²)。"
            f"建议检查时偏估计或数据质量。"
        )
    else:
        print("[align_sensors] 融合轨迹平滑性检查通过。")
