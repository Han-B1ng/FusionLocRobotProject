# file: core/robust_stats.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 鲁棒统计工具：系统偏差估计、异常检测、显著性检验

"""
鲁棒统计模块。

提供三个核心功能：
  - estimate_systematic_bias : 估计传感器间的系统偏差
      支持三种方法：'median'（中位数）、'trimmed'（10%截尾均值）、
      'robust_mean'（10%截尾均值，与 'trimmed' 等价，别名方便调用）
  - detect_anomalies          : 基于 MAD 的异常点检测
  - bias_significance_test    : Wilcoxon 符号秩检验判断偏差是否显著

方法对比说明：
  - 'median'      ：中位数估计，对单个极端离群值最鲁棒，但效率约 64%
  - 'trimmed'/'robust_mean'：10% 截尾均值，在鲁棒性与效率之间取得平衡
  - 当两种方法的偏差估计差异 < 0.1 m 时，说明偏差估计稳健，极值影响小
  - 当差异 >= 0.1 m 时，提示偏差受极值影响，建议采用截尾均值

依赖：numpy, scipy
被依赖：stage2_problem2.py, stage3_problem3.py
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import median_abs_deviation, wilcoxon


# ============================================================
#  1. 系统偏差估计
# ============================================================
def estimate_systematic_bias(
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    method: str = "median",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """估计两个传感器之间的系统偏差。

    计算残差 dx = x_aligned - x_ref, dy = y_aligned - y_ref，
    然后用指定方法估计固定偏差。

    支持的 method：
      - 'median'      : 中位数，对离群值最鲁棒
      - 'trimmed'     : 10% 截尾均值（去掉两端各 10% 后求均值）
      - 'robust_mean' : 与 'trimmed' 等价，为方便调用提供的别名

    Parameters
    ----------
    x_aligned, y_aligned : np.ndarray
        对齐后的待估传感器坐标 (m)，例如传感器 2。
    x_ref, y_ref : np.ndarray
        参考传感器坐标 (m)，例如传感器 1。
    method : str
        估计方法：'median'、'trimmed' 或 'robust_mean'。

    Returns
    -------
    bias_x : float
        X 方向系统偏差估计值 (m)。
    bias_y : float
        Y 方向系统偏差估计值 (m)。
    dx : np.ndarray
        X 方向残差序列 (m)。
    dy : np.ndarray
        Y 方向残差序列 (m)。

    Raises
    ------
    ValueError
        输入数组长度不一致，或 method 不支持。
    """
    x_aligned = np.asarray(x_aligned, dtype=np.float64)
    y_aligned = np.asarray(y_aligned, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    n = len(x_aligned)
    if not (n == len(y_aligned) == len(x_ref) == len(y_ref)):
        raise ValueError(
            f"[estimate_systematic_bias] 输入长度不一致: "
            f"{len(x_aligned)}, {len(y_aligned)}, "
            f"{len(x_ref)}, {len(y_ref)}"
        )

    # 残差 = 待估传感器 - 参考传感器
    dx = x_aligned - x_ref
    dy = y_aligned - y_ref

    if method == "median":
        # 中位数估计：对离群值鲁棒
        bias_x = float(np.median(dx))
        bias_y = float(np.median(dy))
    elif method in ("trimmed", "robust_mean"):
        # 10% 截尾均值：去掉两端各 10% 后求均值
        # 'robust_mean' 是 'trimmed' 的别名，行为完全相同
        bias_x = float(_trimmed_mean(dx, proportion=0.10))
        bias_y = float(_trimmed_mean(dy, proportion=0.10))
    else:
        raise ValueError(
            f"[estimate_systematic_bias] 不支持的 method='{method}'，"
            f"请使用 'median'、'trimmed' 或 'robust_mean'。"
        )

    return bias_x, bias_y, dx, dy


# ============================================================
#  2. 偏差估计方法对比（供 stage2_problem2.py 调用）
# ============================================================
def compare_bias_methods(
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    consistency_threshold: float = 0.1,
) -> dict:
    """对比中位数与截尾均值两种偏差估计方法。

    分别用 'median' 和 'robust_mean' 估计偏差，计算差异，
    并根据 consistency_threshold 判断两种方法是否一致。

    Parameters
    ----------
    x_aligned, y_aligned : np.ndarray
        对齐后的待估传感器坐标 (m)。
    x_ref, y_ref : np.ndarray
        参考传感器坐标 (m)。
    consistency_threshold : float
        一致性判断阈值 (m)，默认 0.1。
        当两种方法偏差差异 < 此值时认为一致。

    Returns
    -------
    result : dict
        包含以下字段：
        - 'median'    : (bias_x, bias_y)   中位数估计
        - 'robust_mean': (bias_x, bias_y)  截尾均值估计
        - 'diff_x'    : float              X 方向差异绝对值
        - 'diff_y'    : float              Y 方向差异绝对值
        - 'is_consistent' : bool           两种方法是否一致
        - 'message'   : str                结论提示
    """
    # 中位数方法
    bx_med, by_med, _, _ = estimate_systematic_bias(
        x_aligned, y_aligned, x_ref, y_ref, method="median"
    )
    # 截尾均值方法
    bx_trim, by_trim, _, _ = estimate_systematic_bias(
        x_aligned, y_aligned, x_ref, y_ref, method="robust_mean"
    )

    diff_x = abs(bx_med - bx_trim)
    diff_y = abs(by_med - by_trim)

    is_consistent = diff_x < consistency_threshold and diff_y < consistency_threshold

    if is_consistent:
        message = "偏差估计稳健，两种方法一致"
    else:
        message = "偏差估计受极值影响，建议采用截尾均值"

    return {
        "median": (bx_med, by_med),
        "robust_mean": (bx_trim, by_trim),
        "diff_x": diff_x,
        "diff_y": diff_y,
        "is_consistent": is_consistent,
        "message": message,
    }


# ============================================================
#  3. 异常点检测
# ============================================================
def detect_anomalies(
    residuals: np.ndarray,
    threshold: float = 3.0,
) -> List[int]:
    """基于 MAD 的异常点检测。

    使用中位绝对偏差 (MAD) 估计标准差尺度，
    计算每个残差的 z 分数，超过阈值的标记为异常。

    z_i = |r_i - median(r)| / MAD(r)

    Parameters
    ----------
    residuals : np.ndarray
        残差序列。
    threshold : float
        z 分数阈值，默认 3.0。

    Returns
    -------
    anomaly_indices : list of int
        异常点的索引列表。
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    # scipy.stats.median_abs_deviation(scale='normal')
    # 等价于 MAD / 0.6745，输出为标准差尺度
    mad = median_abs_deviation(residuals, scale="normal")

    if mad < 1e-15:
        # MAD 为零说明数据几乎恒定，无异常
        return []

    med = np.median(residuals)
    z = np.abs(residuals - med) / mad

    anomaly_indices = np.where(z > threshold)[0].tolist()

    return anomaly_indices


# ============================================================
#  4. 偏差显著性检验
# ============================================================
def bias_significance_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Wilcoxon 符号秩检验：判断系统偏差是否显著不同于零。

    H₀: 中位数偏差 = 0
    H₁: 中位数偏差 ≠ 0（双侧检验）

    Parameters
    ----------
    residuals : np.ndarray
        残差序列 dx 或 dy。
    alpha : float
        显著性水平，默认 0.05。

    Returns
    -------
    is_significant : bool
        若 p_value < alpha 则偏差显著。
    p_value : float
        检验 p 值。

    Notes
    -----
    使用 scipy.stats.wilcoxon，method='approx'。
    当残差全为零或样本量过小时，返回 (False, 1.0)。
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    # 去除零残差（Wilcoxon 检验不处理零差值）
    non_zero = residuals[residuals != 0.0]

    if len(non_zero) < 5:
        # 样本量太小，检验无意义
        return False, 1.0

    try:
        result = wilcoxon(non_zero, alternative="two-sided", method="approx")
        p_value = float(result.pvalue)
    except ValueError:
        # 某些边界情况（如所有差值同号且样本极小）
        return False, 1.0

    is_significant = p_value < alpha

    return is_significant, p_value


# ============================================================
#  内部工具
# ============================================================
def _trimmed_mean(
    data: np.ndarray,
    proportion: float = 0.10,
) -> float:
    """截尾均值：去掉两端各 proportion 比例后求均值。

    Parameters
    ----------
    data : np.ndarray
        输入数据。
    proportion : float
        两端各截去的比例，默认 0.10（即各去 10%）。

    Returns
    -------
    mean : float
        截尾均值。
    """
    data = sort_arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(data)
    k = int(np.floor(n * proportion))

    if 2 * k >= n:
        # 截去太多，退化为中位数
        return float(np.median(data))

    trimmed = data[k: n - k]
    return float(np.mean(trimmed))
