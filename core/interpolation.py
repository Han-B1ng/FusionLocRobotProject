# file: core/interpolation.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 重采样与插值工具：支持线性、三次样条，将不同频率数据统一到目标频率

"""
插值与重采样模块。

提供两个核心函数：
  - resample_to_target : 将单组传感器数据重采样到目标频率
  - interp_to_common_grid : 将两组传感器数据插值到公共时间网格

依赖：numpy, scipy
被依赖：core/time_alignment.py, stage1_problem1.py
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


# ============================================================
#  1. 单组数据重采样
# ============================================================
def resample_to_target(
    t_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    target_freq: float,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将原始传感器数据重采样到目标频率。

    Parameters
    ----------
    t_raw : np.ndarray
        原始时间戳数组 (s)，要求单调递增。
    x_raw : np.ndarray
        原始 X 坐标数组 (m)，与 t_raw 等长。
    y_raw : np.ndarray
        原始 Y 坐标数组 (m)，与 t_raw 等长。
    target_freq : float
        目标输出频率 (Hz)，例如 10.0。
    method : str, optional
        插值方法，'linear' 或 'cubic'，默认 'cubic'。
        当数据点数 < 4 时自动退化为 'linear'。

    Returns
    -------
    t_out : np.ndarray
        目标时间网格数组 (s)。
    x_out : np.ndarray
        插值后的 X 坐标数组 (m)。
    y_out : np.ndarray
        插值后的 Y 坐标数组 (m)。
    """
    # --------------------------------------------------
    # Step 1: 输入清洗 — 剔除 NaN，确保单调
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Step 2: 自动退化 — 数据点数 < 4 时 cubic → linear
    # --------------------------------------------------
    eff_method = method
    if method == "cubic" and len(t_raw) < 4:
        eff_method = "linear"

    # --------------------------------------------------
    # Step 3: 生成目标时间网格（不外推）
    #     t_min + k * dt_target,  k = 0, 1, 2, ...
    #     最后一个网格点 ≤ t_raw.max()
    # --------------------------------------------------
    dt_target = 1.0 / target_freq
    t_min = float(t_raw.min())
    t_max = float(t_raw.max())

    n_steps = int(np.floor((t_max - t_min) / dt_target))
    t_out = t_min + np.arange(n_steps + 1) * dt_target

    # 裁剪：防止浮点累积导致最后一点微超
    t_out = np.clip(t_out, t_min, t_max)

    # --------------------------------------------------
    # Step 4: 构建插值器并求值
    # --------------------------------------------------
    x_out = _interpolate_1d(t_raw, x_raw, t_out, eff_method)
    y_out = _interpolate_1d(t_raw, y_raw, t_out, eff_method)

    return t_out, x_out, y_out


# ============================================================
#  2. 双组数据插值到公共网格
# ============================================================
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
    """将两组传感器数据插值到相同的时间网格。

    时间网格取两组数据时间范围的 **交集**，避免外推。

    Parameters
    ----------
    t1, x1, y1 : np.ndarray
        传感器 1 的时间戳 (s)、X 坐标 (m)、Y 坐标 (m)。
    t2, x2, y2 : np.ndarray
        传感器 2 的时间戳 (s)、X 坐标 (m)、Y 坐标 (m)。
    target_freq : float
        目标输出频率 (Hz)。
    method : str, optional
        插值方法，'linear' 或 'cubic'，默认 'cubic'。

    Returns
    -------
    t_grid : np.ndarray
        公共时间网格 (s)。
    x1_out : np.ndarray
        传感器 1 插值后 X (m)。
    y1_out : np.ndarray
        传感器 1 插值后 Y (m)。
    x2_out : np.ndarray
        传感器 2 插值后 X (m)。
    y2_out : np.ndarray
        传感器 2 插值后 Y (m)。
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)

    # --------------------------------------------------
    # 取时间范围交集
    # --------------------------------------------------
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())

    if t_start >= t_end:
        raise ValueError(
            f"[interp_to_common_grid] 两组数据时间范围无交集: "
            f"[{t1.min():.2f}, {t1.max():.2f}] vs "
            f"[{t2.min():.2f}, {t2.max():.2f}]"
        )

    # --------------------------------------------------
    # 生成公共网格
    # --------------------------------------------------
    dt_target = 1.0 / target_freq
    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    # --------------------------------------------------
    # 分别对两组数据构建插值器并在公共网格求值
    # --------------------------------------------------
    # 对第一组数据：先清洗再插值
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


# ============================================================
#  内部工具
# ============================================================
def _has_nan(t: np.ndarray, v: np.ndarray) -> bool:
    """检查时间或数值数组中是否存在 NaN。"""
    return bool(np.any(~np.isfinite(t)) or np.any(~np.isfinite(v)))


def _clean(
    t: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """剔除含 NaN 的行，返回清洗后的 (t, v)。"""
    mask = np.isfinite(t) & np.isfinite(v)
    return t[mask], v[mask]


def _interpolate_1d(
    t_src: np.ndarray,
    v_src: np.ndarray,
    t_query: np.ndarray,
    method: str,
) -> np.ndarray:
    """一维插值的统一封装。

    Parameters
    ----------
    t_src : np.ndarray
        源时间戳 (单调递增)。
    v_src : np.ndarray
        源数值。
    t_query : np.ndarray
        查询时间点。
    method : str
        'linear' 使用 scipy.interpolate.interp1d；
        'cubic'  使用 scipy.interpolate.CubicSpline。

    Returns
    -------
    v_out : np.ndarray
        插值结果。

    Notes
    -----
    公式位置说明：
      - linear: v(t) = v_i + (v_{i+1} - v_i) / (t_{i+1} - t_i) * (t - t_i)
        对应 interp1d(kind='linear')
      - cubic: S_i(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3
        对应 CubicSpline 自然边界条件
    """
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
