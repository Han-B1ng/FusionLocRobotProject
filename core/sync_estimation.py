# file: core/sync_estimation.py


from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate


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
    t_grid = np.asarray(t_grid, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    dt_grid = float(t_grid[1] - t_grid[0])  # 网格原始步长

    n_candidates = int(np.round((delay_range[1] - delay_range[0]) / dt)) + 1
    delays = np.linspace(delay_range[0], delay_range[1], n_candidates)

    scores = np.zeros(len(delays))

    for i, delta in enumerate(delays):
        t_shifted = t_grid - delta

        mask = (t_shifted >= t_grid[0]) & (t_shifted <= t_grid[-1])
        n_overlap = int(np.sum(mask))

        if n_overlap < 10:
            scores[i] = -np.inf
            continue

        x2_shifted = np.interp(t_shifted[mask], t_grid, x2)
        y2_shifted = np.interp(t_shifted[mask], t_grid, y2)

        x1_seg = x1[mask]
        y1_seg = y1[mask]

        corr_x = _pearson_r(x1_seg, x2_shifted)
        corr_y = _pearson_r(y1_seg, y2_shifted)

        scores[i] = w_x * corr_x + w_y * corr_y

    best_idx = int(np.argmax(scores))
    best_delay = float(delays[best_idx])

    return best_delay, (delays, scores)


def estimate_delay_lsq(
    t_grid: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delay_range: Tuple[float, float] = (-1.0, 1.0),
    x0: float | None = None,
) -> Tuple[float, float]:
    t_grid = np.asarray(t_grid, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    def _rmse_objective(delta: float) -> float:
        t_shifted = t_grid - delta

        mask = (t_shifted >= t_grid[0]) & (t_shifted <= t_grid[-1])
        n_overlap = int(np.sum(mask))

        if n_overlap < 2:
            return 1e12  # 重叠不足，返回极大惩罚值

        x2_shifted = np.interp(t_shifted[mask], t_grid, x2)
        y2_shifted = np.interp(t_shifted[mask], t_grid, y2)

        err_x = x1[mask] - x2_shifted
        err_y = y1[mask] - y2_shifted

        rmse = np.sqrt(np.mean(err_x ** 2 + err_y ** 2))
        return float(rmse)

    result = minimize_scalar(
        _rmse_objective,
        bounds=delay_range,
        method="bounded",
    )

    best_delay = float(result.x)
    best_rmse = float(result.fun)

    return best_delay, best_rmse


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)

    denom = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))

    if denom < 1e-15:
        return 0.0

    return float(np.sum(a_mean * b_mean) / denom)
