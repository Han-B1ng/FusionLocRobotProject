# file: core/interpolation.py


from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def resample_to_target(
    t_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    target_freq: float,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_raw = np.asarray(t_raw, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)

    valid = np.isfinite(t_raw) & np.isfinite(x_raw) & np.isfinite(y_raw)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        t_raw = t_raw[valid]
        x_raw = x_raw[valid]
        y_raw = y_raw[valid]

    if len(t_raw) < 2:
        raise ValueError(
            f"[resample_to_target] 有效数据不足 2 点，无法插值。"
        )

    eff_method = method
    if method == "cubic" and len(t_raw) < 4:
        eff_method = "linear"

    dt_target = 1.0 / target_freq
    t_min = float(t_raw.min())
    t_max = float(t_raw.max())

    n_steps = int(np.floor((t_max - t_min) / dt_target))
    t_out = t_min + np.arange(n_steps + 1) * dt_target

    t_out = np.clip(t_out, t_min, t_max)

    x_out = _interpolate_1d(t_raw, x_raw, t_out, eff_method)
    y_out = _interpolate_1d(t_raw, y_raw, t_out, eff_method)

    return t_out, x_out, y_out


def interp_to_common_grid(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    target_freq: float,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)

    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())

    if t_start >= t_end:
        raise ValueError(
            f"[interp_to_common_grid] 两组数据时间范围无交集: "
            f"[{t1.min():.2f}, {t1.max():.2f}] vs "
            f"[{t2.min():.2f}, {t2.max():.2f}]"
        )

    dt_target = 1.0 / target_freq
    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    x1_out = _interpolate_1d(
        *_clean(t1, x1), t_grid, method
    ) if _has_nan(t1, x1) else _interpolate_1d(t1, x1, t_grid, method)
    y1_out = _interpolate_1d(
        *_clean(t1, y1), t_grid, method
    ) if _has_nan(t1, y1) else _interpolate_1d(t1, y1, t_grid, method)

    x2_out = _interpolate_1d(
        *_clean(t2, x2), t_grid, method
    ) if _has_nan(t2, x2) else _interpolate_1d(t2, x2, t_grid, method)
    y2_out = _interpolate_1d(
        *_clean(t2, y2), t_grid, method
    ) if _has_nan(t2, y2) else _interpolate_1d(t2, y2, t_grid, method)

    return t_grid, x1_out, y1_out, x2_out, y2_out


def _has_nan(t: np.ndarray, v: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(t)) or np.any(~np.isfinite(v)))


def _clean(
    t: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(v)
    return t[mask], v[mask]


def _interpolate_1d(
    t_src: np.ndarray,
    v_src: np.ndarray,
    t_query: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "cubic":
        cs = CubicSpline(t_src, v_src, bc_type="natural")
        return cs(t_query)
    else:
        f = interp1d(
            t_src, v_src,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # 仅在极微小浮点越界时触发
        )
        return f(t_query)
