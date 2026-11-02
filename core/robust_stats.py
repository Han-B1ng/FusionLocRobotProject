# file: core/robust_stats.py


from typing import List, Tuple

import numpy as np
from scipy.stats import median_abs_deviation, wilcoxon


def estimate_systematic_bias(
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    method: str = "median",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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

    dx = x_aligned - x_ref
    dy = y_aligned - y_ref

    if method == "median":
        bias_x = float(np.median(dx))
        bias_y = float(np.median(dy))
    elif method in ("trimmed", "robust_mean"):
        bias_x = float(_trimmed_mean(dx, proportion=0.10))
        bias_y = float(_trimmed_mean(dy, proportion=0.10))
    else:
        raise ValueError(
            f"[estimate_systematic_bias] 不支持的 method='{method}'，"
            f"请使用 'median'、'trimmed' 或 'robust_mean'。"
        )

    return bias_x, bias_y, dx, dy


def compare_bias_methods(
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    consistency_threshold: float = 0.1,
) -> dict:
    bx_med, by_med, _, _ = estimate_systematic_bias(
        x_aligned, y_aligned, x_ref, y_ref, method="median"
    )
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


def detect_anomalies(
    residuals: np.ndarray,
    threshold: float = 3.0,
) -> List[int]:
    residuals = np.asarray(residuals, dtype=np.float64)

    mad = median_abs_deviation(residuals, scale="normal")

    if mad < 1e-15:
        return []

    med = np.median(residuals)
    z = np.abs(residuals - med) / mad

    anomaly_indices = np.where(z > threshold)[0].tolist()

    return anomaly_indices


def bias_significance_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    residuals = np.asarray(residuals, dtype=np.float64)

    non_zero = residuals[residuals != 0.0]

    if len(non_zero) < 5:
        return False, 1.0

    try:
        result = wilcoxon(non_zero, alternative="two-sided", method="approx")
        p_value = float(result.pvalue)
    except ValueError:
        return False, 1.0

    is_significant = p_value < alpha

    return is_significant, p_value


def _trimmed_mean(
    data: np.ndarray,
    proportion: float = 0.10,
) -> float:
    data = sort_arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(data)
    k = int(np.floor(n * proportion))

    if 2 * k >= n:
        return float(np.median(data))

    trimmed = data[k: n - k]
    return float(np.mean(trimmed))
