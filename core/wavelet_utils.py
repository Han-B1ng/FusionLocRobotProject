# file: core/wavelet_utils.py


import warnings
from typing import Optional, Tuple

import numpy as np

try:
    import pywt
except ImportError:
    raise ImportError(
        "[wavelet_utils] 缺少 PyWavelets 库，请安装：\n"
        "  pip install PyWavelets\n"
        "安装后即可使用小波去噪功能。"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
    threshold_method: str = "universal",
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if n < 2:
        raise ValueError(
            f"[wavelet_denoise] 信号长度过短 ({n})，至少需要 2 个点。"
        )

    if mode not in ("soft", "hard"):
        raise ValueError(
            f"[wavelet_denoise] mode 必须为 'soft' 或 'hard'，"
            f"实际为 '{mode}'。"
        )

    if threshold_method not in ("universal", "bayes"):
        raise ValueError(
            f"[wavelet_denoise] threshold_method 必须为 'universal' 或 'bayes'，"
            f"实际为 '{threshold_method}'。"
        )

    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(level, 10))

    min_len = 2 ** level
    if n < min_len:
        level = max(1, int(np.floor(np.log2(n))))
        warnings.warn(
            f"[wavelet_denoise] 数据长度 {n} 不足以支持原定层数，"
            f"自动降为 {level} 层。"
        )

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    if threshold_method == "universal":
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = sigma * np.sqrt(2.0 * np.log(n))

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], thresh, mode=mode)

    elif threshold_method == "bayes":
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            sigma_noise = np.median(np.abs(detail)) / 0.6745
            sigma_noise_sq = sigma_noise ** 2

            var_detail = np.var(detail)
            sigma_coeff_sq = max(var_detail - sigma_noise_sq, 0.0)
            sigma_coeff = np.sqrt(sigma_coeff_sq)

            if sigma_coeff > 1e-15:
                thresh_k = sigma_noise_sq / sigma_coeff
            else:
                thresh_k = 0.0

            coeffs[i] = pywt.threshold(coeffs[i], thresh_k, mode=mode)

    rec = pywt.waverec(coeffs, wavelet)
    denoised = rec[:n]

    return denoised


def denoise_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
    threshold_method: str = "universal",
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError(
            f"[denoise_trajectory] x({len(x)}) 与 y({len(y)}) 长度不一致。"
        )

    x_d = wavelet_denoise(
        x, wavelet=wavelet, level=level, mode=mode,
        threshold_method=threshold_method,
    )
    y_d = wavelet_denoise(
        y, wavelet=wavelet, level=level, mode=mode,
        threshold_method=threshold_method,
    )

    return x_d, y_d


def adaptive_denoise_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "db4",
    threshold_method: str = "universal",
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)

    if n < 128:
        warnings.warn(
            f"[adaptive_denoise] 信号长度 {n} < 128，"
            f"退化为移动平均滤波 (窗口=3)。"
        )
        kernel = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        x_out = np.convolve(x, kernel, mode="same")
        y_out = np.convolve(y, kernel, mode="same")
    else:
        x_out, y_out = denoise_trajectory(
            x, y, wavelet=wavelet, threshold_method=threshold_method,
        )

    return x_out, y_out


def compare_denoise_configs(
    x: np.ndarray,
    y: np.ndarray,
    wavelet_list: Tuple[str, ...] = ("db4", "sym5"),
    thresh_methods: Tuple[str, ...] = ("universal", "bayes"),
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    results = {}

    for wv in wavelet_list:
        for tm in thresh_methods:
            x_d, y_d = denoise_trajectory(
                x, y, wavelet=wv, threshold_method=tm,
            )

            var_x = float(np.var(x_d))
            var_y = float(np.var(y_d))

            if len(x_d) >= 3:
                accel_x = np.diff(x_d, n=2)
                accel_y = np.diff(y_d, n=2)
                accel_var_x = float(np.var(accel_x))
                accel_var_y = float(np.var(accel_y))
            else:
                accel_var_x = float("inf")
                accel_var_y = float("inf")

            results[(wv, tm)] = {
                "var_x": var_x,
                "var_y": var_y,
                "accel_var_x": accel_var_x,
                "accel_var_y": accel_var_y,
            }

    return results


def plot_wavelet_decomp(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    if not _HAS_MPL:
        raise ImportError(
            "[plot_wavelet_decomp] 需要 matplotlib，请安装：\n"
            "  pip install matplotlib"
        )

    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(level, 6))

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    n_panels = len(coeffs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels))

    if n_panels == 1:
        axes = [axes]

    labels = [f"cA_{level} (approx)"] + [
        f"cD_{level - i} (detail)" for i in range(n_panels - 1)
    ]

    for ax, coeff, label in zip(axes, coeffs, labels):
        ax.plot(coeff, linewidth=0.8, color="#2563EB")
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_title(
        f"Wavelet Decomposition — {wavelet}, level={level}",
        fontsize=12, fontweight="bold",
    )
    axes[-1].set_xlabel("Coefficient Index", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_wavelet_decomp] 已保存: {save_path}")
    else:
        plt.show()
