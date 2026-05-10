# file: core/time_alignment.py


import warnings
from typing import Optional, Tuple

import numpy as np

from config import alignment_config, time_config
from core.interpolation import interp_to_common_grid, resample_to_target
from core.sync_estimation import (
    estimate_delay_cross_corr,
    estimate_delay_lsq,
)


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

    t_grid_coarse, x1_c, y1_c, x2_c, y2_c = interp_to_common_grid(
        t1, x1, y1,
        t2, x2, y2,
        target_freq=target_freq,
        method=method,
    )

    delay_coarse, (delays, scores) = estimate_delay_cross_corr(
        t_grid_coarse, x1_c, y1_c, x2_c, y2_c,
        delay_range=delay_range,
        dt=alignment_config.corr_window / 20.0,  # 窗口内约 20 个采样点
    )

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

    t2_corrected = t2 - delay_fine

    t_start = max(t1.min(), t2_corrected.min())
    t_end = min(t1.max(), t2_corrected.max())

    if t_start >= t_end:
        raise ValueError(
            f"[align_sensors] 修正后时间范围无交集: "
            f"s1=[{t1.min():.2f}, {t1.max():.2f}] vs "
            f"s2_corrected=[{t2_corrected.min():.2f}, {t2_corrected.max():.2f}]"
        )

    dt_target = 1.0 / target_freq
    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    x1_aligned = np.interp(t_grid, t1, x1)
    y1_aligned = np.interp(t_grid, t1, y1)

    x2_aligned = np.interp(t_grid, t2_corrected, x2)
    y2_aligned = np.interp(t_grid, t2_corrected, y2)

    x_fused = w1 * x1_aligned + w2 * x2_aligned
    y_fused = w1 * y1_aligned + w2 * y2_aligned

    _check_smoothness(t_grid, x_fused, y_fused, target_freq)

    print(
        f"[align_sensors] 输出网格: "
        f"[{t_grid[0]:.2f}, {t_grid[-1]:.2f}] s, "
        f"{len(t_grid)} 点, "
        f"频率 = {target_freq:.0f} Hz"
    )

    return delay_fine, t_grid, x_fused, y_fused, delays, scores


def _check_smoothness(
    t_grid: np.ndarray,
    x_fused: np.ndarray,
    y_fused: np.ndarray,
    target_freq: float,
    sigma_factor: float = 5.0,
) -> None:
    if len(t_grid) < 3:
        return

    dt = np.diff(t_grid)
    dt[dt == 0] = 1e-10  # 防除零

    vx = np.diff(x_fused) / dt
    vy = np.diff(y_fused) / dt
    speed = np.sqrt(vx ** 2 + vy ** 2)

    if len(speed) < 2:
        return

    dt_mid = (dt[:-1] + dt[1:]) / 2.0
    dt_mid[dt_mid == 0] = 1e-10

    ax = np.diff(vx) / dt_mid
    ay = np.diff(vy) / dt_mid
    accel = np.sqrt(ax ** 2 + ay ** 2)

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
