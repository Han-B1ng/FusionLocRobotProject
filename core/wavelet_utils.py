# file: core/wavelet_utils.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 小波去噪工具：支持通用阈值法与 BayesShrink 阈值策略

"""
小波去噪模块。

提供三个层次的去噪接口：
  - wavelet_denoise         : 单通道信号小波去噪（核心算法）
  - denoise_trajectory      : 对 X/Y 轨迹分别去噪
  - adaptive_denoise_trajectory : 自适应选择小波基，短信号自动退化为平滑滤波

阈值策略：
  - 'universal'（通用阈值法）：
      σ = MAD(cD₁) / 0.6745
      thr = σ · √(2·ln N)
  - 'bayes'（BayesShrink）：
      对每层细节系数独立计算：
        σ_noise = median(|cD_k|) / 0.6745
        σ_coeff = √(max(var(cD_k) - σ_noise², 0))
        thr_k   = σ_noise² / σ_coeff   (σ_coeff > 0 时)
                = 0                     (σ_coeff = 0 时)

依赖：numpy, PyWavelets (pywt)
被依赖：stage2_problem2.py, stage3_problem3.py
"""

import warnings
from typing import Optional, Tuple

import numpy as np

# --------------------------------------------------
#  友好的 pywt 导入
# --------------------------------------------------
try:
    import pywt
except ImportError:
    raise ImportError(
        "[wavelet_utils] 缺少 PyWavelets 库，请安装：\n"
        "  pip install PyWavelets\n"
        "安装后即可使用小波去噪功能。"
    )

# 可选：matplotlib 用于调试绘图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ============================================================
#  1. 核心：单通道小波去噪
# ============================================================
def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
    threshold_method: str = "universal",
) -> np.ndarray:
    """对一维信号进行小波去噪。

    支持两种阈值策略：
      - 'universal'：通用阈值 thr = σ · √(2·ln N)
      - 'bayes'    ：BayesShrink 逐层自适应阈值

    算法步骤：
      1. 对信号进行离散小波分解 (DWT)
      2. 根据 threshold_method 计算阈值
      3. 对各层细节系数施加阈值（保留近似系数不变）
      4. 小波重构得到去噪信号

    Parameters
    ----------
    signal : np.ndarray
        输入一维信号，长度 N。
    wavelet : str
        小波基名称，默认 'db4'（Daubechies-4）。
        常用选项：'db4', 'db6', 'sym4', 'sym5', 'coif3', 'bior3.5'。
    level : int or None
        分解层数。若为 None，使用 pywt.dwt_max_level 自动计算。
    mode : str
        阈值模式：'soft'（软阈值）或 'hard'（硬阈值）。

        - 软阈值：sgn(c) · max(|c| - thr, 0)，连续但有偏
        - 硬阈值：c · I(|c| > thr)，无偏但不连续
    threshold_method : str
        阈值计算策略：
        - 'universal'：通用阈值法（默认），全局统一阈值
        - 'bayes'    ：BayesShrink，逐层自适应阈值

    Returns
    -------
    denoised : np.ndarray
        去噪后的信号，与输入等长。

    Raises
    ------
    ValueError
        信号长度小于 2，或 mode 不是 'soft'/'hard'，
        或 threshold_method 不是 'universal'/'bayes'。
    """
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

    # --------------------------------------------------
    # Step 1: 确定分解层数
    # --------------------------------------------------
    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(level, 10))

    # 数据点太少无法分解时退化
    min_len = 2 ** level
    if n < min_len:
        level = max(1, int(np.floor(np.log2(n))))
        warnings.warn(
            f"[wavelet_denoise] 数据长度 {n} 不足以支持原定层数，"
            f"自动降为 {level} 层。"
        )

    # --------------------------------------------------
    # Step 2: 离散小波分解
    #     coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    #     coeffs[0]  = 近似系数 (cA)
    #     coeffs[1:] = 各层细节系数 (cD)
    # --------------------------------------------------
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # --------------------------------------------------
    # Step 3 & 4: 根据阈值策略计算阈值并施加
    # --------------------------------------------------
    if threshold_method == "universal":
        # --- 通用阈值法 ---
        # σ = median(|cD₁|) / 0.6745，其中 coeffs[-1] 是第一层细节系数
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        # thr = σ · √(2 · ln N)
        thresh = sigma * np.sqrt(2.0 * np.log(n))

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], thresh, mode=mode)

    elif threshold_method == "bayes":
        # --- BayesShrink 逐层自适应阈值 ---
        # 对每层细节系数 cD_k：
        #   σ_noise  = median(|cD_k|) / 0.6745
        #   σ_coeff  = √(max(var(cD_k) - σ_noise², 0))
        #   thr_k    = σ_noise² / σ_coeff   (σ_coeff > 0)
        #            = 0                     (σ_coeff = 0)
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            # 噪声标准差估计（基于该层细节系数的 MAD）
            sigma_noise = np.median(np.abs(detail)) / 0.6745
            sigma_noise_sq = sigma_noise ** 2

            # 信号系数标准差估计
            var_detail = np.var(detail)
            sigma_coeff_sq = max(var_detail - sigma_noise_sq, 0.0)
            sigma_coeff = np.sqrt(sigma_coeff_sq)

            if sigma_coeff > 1e-15:
                # BayesShrink 阈值 = σ_noise² / σ_coeff
                thresh_k = sigma_noise_sq / sigma_coeff
            else:
                # σ_coeff ≈ 0，该层无信号成分，阈值设为 0（不做阈值处理）
                thresh_k = 0.0

            coeffs[i] = pywt.threshold(coeffs[i], thresh_k, mode=mode)

    # --------------------------------------------------
    # Step 5: 小波重构，切片为原始长度
    #     waverec 可能返回比原信号多 1 个点（偶数补齐）
    # --------------------------------------------------
    rec = pywt.waverec(coeffs, wavelet)
    denoised = rec[:n]

    return denoised


# ============================================================
#  2. 轨迹去噪：分别处理 X 和 Y
# ============================================================
def denoise_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
    threshold_method: str = "universal",
) -> Tuple[np.ndarray, np.ndarray]:
    """对轨迹的 X / Y 分量分别进行小波去噪。

    Parameters
    ----------
    x, y : np.ndarray
        轨迹的 X / Y 坐标数组 (m)，长度须一致。
    wavelet : str
        小波基名称，默认 'db4'。
    level : int or None
        分解层数，默认自动确定。
    mode : str
        阈值模式：'soft' 或 'hard'。
    threshold_method : str
        阈值计算策略：'universal'（默认）或 'bayes'。

    Returns
    -------
    x_denoised, y_denoised : np.ndarray
        去噪后的 X / Y 坐标数组，与输入等长。

    Raises
    ------
    ValueError
        x 与 y 长度不一致。
    """
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


# ============================================================
#  3. 自适应去噪：自动选小波基 + 短信号退化
# ============================================================
def adaptive_denoise_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "db4",
    threshold_method: str = "universal",
) -> Tuple[np.ndarray, np.ndarray]:
    """自适应轨迹去噪：短信号退化为移动平均，长信号使用小波去噪。

    策略：
      - 信号长度 < 128 ：退化为 3 点移动平均滤波
      - 信号长度 >= 128：使用小波去噪

    Parameters
    ----------
    x, y : np.ndarray
        轨迹的 X / Y 坐标数组 (m)。
    wavelet : str
        小波基名称，默认 'db4'。
    threshold_method : str
        阈值计算策略：'universal'（默认）或 'bayes'。

    Returns
    -------
    x_out, y_out : np.ndarray
        去噪 / 平滑后的 X / Y 坐标数组，与输入等长。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)

    if n < 128:
        # 短信号退化为移动平均
        # np.convolve kernel [1/3, 1/3, 1/3], mode='same' 保持等长
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


# ============================================================
#  4. 去噪参数对比实验（供 stage2_problem2.py 调用）
# ============================================================
def compare_denoise_configs(
    x: np.ndarray,
    y: np.ndarray,
    wavelet_list: Tuple[str, ...] = ("db4", "sym5"),
    thresh_methods: Tuple[str, ...] = ("universal", "bayes"),
) -> dict:
    """对比不同小波基与阈值策略组合的去噪效果。

    对每种组合计算：
      - 去噪后 X/Y 方差（越小表示越平滑）
      - 加速度方差（差分平滑性指标，越小表示越平滑）

    Parameters
    ----------
    x, y : np.ndarray
        原始轨迹 X / Y 坐标 (m)。
    wavelet_list : tuple of str
        待比较的小波基列表。
    thresh_methods : tuple of str
        待比较的阈值策略列表。

    Returns
    -------
    results : dict
        键为 (wavelet, threshold_method) 的元组，
        值为 dict，包含 'var_x', 'var_y', 'accel_var_x', 'accel_var_y'。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    results = {}

    for wv in wavelet_list:
        for tm in thresh_methods:
            x_d, y_d = denoise_trajectory(
                x, y, wavelet=wv, threshold_method=tm,
            )

            # 去噪后方差
            var_x = float(np.var(x_d))
            var_y = float(np.var(y_d))

            # 加速度方差（二阶差分平滑性）
            #  accel_x = diff(diff(x_d))，反映曲率/抖动程度
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


# ============================================================
#  5. 可选：调试用小波分解可视化
# ============================================================
def plot_wavelet_decomp(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """绘制信号的小波分解各层系数，用于调试。

    Parameters
    ----------
    signal : np.ndarray
        输入一维信号。
    wavelet : str
        小波基名称。
    level : int or None
        分解层数，默认自动确定（调试图最多 6 层）。
    save_path : str or None
        若提供，保存图片到该路径；否则 plt.show()。

    Raises
    ------
    ImportError
        matplotlib 未安装时无法绘图。
    """
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
