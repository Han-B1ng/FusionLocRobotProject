# file: sensitivity_analysis.py
# @Author : Han_B1ng
# @Time : 2026/5/9
# @Description : Q矩阵过程噪声敏感性分析 —— 独立脚本
#                在问题2代表性数据上扫描Q对角缩放因子，
#                记录融合RMSE并生成敏感性曲线图与控制台报告。

"""
╔══════════════════════════════════════════════════════╗
║  Q矩阵敏感性分析                                       ║
╚══════════════════════════════════════════════════════╝

扫描Q矩阵对角元素的缩放因子 scale ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}，
在问题2数据上循环运行EKF融合，记录RMSE并绘制敏感性曲线。

运行方式：
    python sensitivity_analysis.py

依赖：
    - 问题2数据（附件2）需存在于 data/ 目录
    - cleaned_data.pkl 已由 stage0_eda.py 生成（可选加速）
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 确保项目根目录在 sys.path 中 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    alignment_config,
    data_path,
    filter_config,
    plot_config,
    time_config,
    PLOT_DIR,
    TABLE_DIR,
    INTERMEDIATE_DIR,
    ensure_dirs,
)
from core.kalman_filters import estimate_adaptive_R, estimate_ar1_params
from core.robust_stats import (
    bias_significance_test,
    estimate_systematic_bias,
)
from core.time_alignment import align_sensors
from core.wavelet_utils import (
    compare_denoise_configs,
    denoise_trajectory,
)

# ============================================================
#  全局样式
# ============================================================
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
plot_config.apply_style()

_COLOR_LINE = plot_config.COLORS[3]       # 蓝绿色主曲线
_COLOR_BASELINE = plot_config.COLORS[6]   # 朱红色基线标记

# ── Q矩阵缩放因子扫描范围（六个水平）──
Q_SCALES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# ── 为加速分析，可选择是否跳过消融/对比等非核心步骤 ──
SKIP_DENOISE_COMPARISON = False  # True = 使用默认小波参数，跳过自动选择


# ============================================================
#  数据加载（复用 stage2 逻辑）
# ============================================================
def load_problem2_data() -> Tuple[np.ndarray, ...]:
    """加载附件2的两个传感器工作表，返回 (t1, x1, y1, t2, x2, y2)。"""
    file_path = data_path.path2
    if not file_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break

    print(f"[SA] 加载文件：{file_path}")

    df1 = pd.read_excel(file_path, sheet_name="方式1(4Hz)", engine="openpyxl")
    df2 = pd.read_excel(file_path, sheet_name="方式2(5Hz)", engine="openpyxl")

    col_map = {"时间(s)": "t", "X坐标(m)": "x", "Y坐标(m)": "y"}
    df1 = df1.rename(columns=col_map)
    df2 = df2.rename(columns=col_map)

    for df in (df1, df2):
        for col in ("t", "x", "y"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    t1 = df1["t"].values.astype(np.float64)
    x1 = df1["x"].values.astype(np.float64)
    y1 = df1["y"].values.astype(np.float64)
    t2 = df2["t"].values.astype(np.float64)
    x2 = df2["x"].values.astype(np.float64)
    y2 = df2["y"].values.astype(np.float64)

    print(f"[SA] 传感器1：{len(t1)} 点, [{t1[0]:.2f}, {t1[-1]:.2f}] s")
    print(f"[SA] 传感器2：{len(t2)} 点, [{t2[0]:.2f}, {t2[-1]:.2f}] s")
    return t1, x1, y1, t2, x2, y2


# ============================================================
#  预处理管线（去噪 + 时间对齐 + 偏差估计，仅执行一次）
# ============================================================
def preprocess_pipeline(
    t1: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, x2: np.ndarray, y2: np.ndarray,
) -> Dict:
    """执行去噪、时间对齐、偏差估计，返回供后续EKF融合复用的预处理结果。

    这些步骤与Q矩阵无关，仅需执行一次以节省计算时间。
    """
    print("\n" + "=" * 60)
    print("  [预处理] 小波去噪参数选择（仅一次）")
    print("=" * 60)

    # ── 小波去噪 ──
    if SKIP_DENOISE_COMPARISON:
        best_wl1, best_tm1 = "db4", "universal"
        best_wl2, best_tm2 = "db4", "universal"
        print("[SA] 使用默认小波参数（跳过自动选择）")
    else:
        res1 = compare_denoise_configs(
            x1, y1, wavelet_list=("db4", "sym5"),
            thresh_methods=("universal", "bayes"),
        )
        best_key = min(res1, key=lambda k: (
            res1[k]["accel_var_x"] + res1[k]["accel_var_y"]
        ))
        best_wl1, best_tm1 = best_key

        res2 = compare_denoise_configs(
            x2, y2, wavelet_list=("db4", "sym5"),
            thresh_methods=("universal", "bayes"),
        )
        best_key = min(res2, key=lambda k: (
            res2[k]["accel_var_x"] + res2[k]["accel_var_y"]
        ))
        best_wl2, best_tm2 = best_key

    print(f"[SA] 传感器1 最优小波: {best_wl1}/{best_tm1}")
    print(f"[SA] 传感器2 最优小波: {best_wl2}/{best_tm2}")

    x1_d, y1_d = denoise_trajectory(
        x1, y1, wavelet=best_wl1, threshold_method=best_tm1,
    )
    x2_d, y2_d = denoise_trajectory(
        x2, y2, wavelet=best_wl2, threshold_method=best_tm2,
    )

    # ── 时间对齐 ──
    print("\n" + "=" * 60)
    print("  [预处理] 时间对齐（仅一次）")
    print("=" * 60)

    delay, _, _, _ = align_sensors(
        t1, x1_d, y1_d, t2, x2_d, y2_d,
        target_freq=time_config.target_freq,
        delay_range=alignment_config.delay_range,
        method=alignment_config.method, w1=0.5, w2=0.5,
    )
    print(f"[SA] 估计时间偏差：{delay:+.4f} s")

    # 对齐后的统一时间网格
    t2_corrected = t2 - delay
    t_start = max(t1.min(), t2_corrected.min())
    t_end = min(t1.max(), t2_corrected.max())
    dt_target = 1.0 / time_config.target_freq
    n_steps = int(np.floor((t_end - t_start) / dt_target))
    t_grid = t_start + np.arange(n_steps + 1) * dt_target
    t_grid = np.clip(t_grid, t_start, t_end)

    x1_aligned = np.interp(t_grid, t1, x1_d)
    y1_aligned = np.interp(t_grid, t1, y1_d)
    x2_aligned = np.interp(t_grid, t2_corrected, x2_d)
    y2_aligned = np.interp(t_grid, t2_corrected, y2_d)

    # ── 系统偏差估计 ──
    print("\n" + "=" * 60)
    print("  [预处理] 系统偏差估计（仅一次）")
    print("=" * 60)

    bias_x, bias_y, dx, dy = estimate_systematic_bias(
        x2_aligned, y2_aligned, x1_aligned, y1_aligned, method="median",
    )
    print(f"[SA] 系统偏差：bias_x={bias_x:+.4f} m, bias_y={bias_y:+.4f} m")

    # ── AR(1) 偏差漂移参数 ──
    ar1_alpha, ar1_bias_var = estimate_ar1_params(dx, dy, dt_ref=0.1)
    print(f"[SA] AR(1): alpha={ar1_alpha:.4f}, bias_var={ar1_bias_var:.6f}")

    # ── 自适应观测噪声 ──
    R1_est, R2_est = estimate_adaptive_R(
        t1, x1_d, y1_d, dx, dy,
        bias_x=bias_x, bias_y=bias_y, method="mad",
    )

    return {
        "t1": t1, "x1_d": x1_d, "y1_d": y1_d,
        "t2": t2_corrected, "x2_d": x2_d, "y2_d": y2_d,
        "t_grid": t_grid,
        "x1_aligned": x1_aligned, "y1_aligned": y1_aligned,
        "delay": delay,
        "bias_x": bias_x, "bias_y": bias_y,
        "ar1_alpha": ar1_alpha, "ar1_bias_var": ar1_bias_var,
        "R1_est": R1_est, "R2_est": R2_est,
        "best_wl1": best_wl1, "best_tm1": best_tm1,
        "best_wl2": best_wl2, "best_tm2": best_tm2,
    }


# ============================================================
#  单次融合 + RMSE 评估
# ============================================================
def run_fusion_with_q_scale(
    prep: Dict,
    q_scale: float,
) -> Dict:
    """使用指定的Q缩放因子执行一次EKF融合，计算RMSE。

    Parameters
    ----------
    prep : dict
        preprocess_pipeline() 返回的预处理结果。
    q_scale : float
        Q矩阵对角元素的缩放因子。

    Returns
    -------
    dict : 包含 rmse, rmse_x, rmse_y, t_grid, x_fused, y_fused 等
    """
    from core.kalman_filters import fuse_sensors

    # ── 核心操作：用 dataclasses.replace 创建缩放后的 FilterConfig ──
    #    绕过 frozen=True 限制，比 monkey-patch 更安全
    q_orig = filter_config.Q
    q_scaled = tuple(v * q_scale for v in q_orig)

    scaled_cfg = dataclasses.replace(filter_config, Q=q_scaled)
    # 临时替换 config 模块中的 filter_config 引用
    import config
    old_cfg = config.filter_config
    config.filter_config = scaled_cfg

    try:
        # 执行EKF融合（fuse_sensors 内部 from config import filter_config
        # 读取到的是我们临时替换后的 scaled_cfg）
        t_grid, x_fused, y_fused, _, _ = fuse_sensors(
            prep["t1"], prep["x1_d"], prep["y1_d"],
            prep["t2"], prep["x2_d"], prep["y2_d"],
            target_freq=time_config.target_freq,
            R1_est=prep["R1_est"],
            R2_est=prep["R2_est"],
            ar1_alpha=prep["ar1_alpha"],
            ar1_bias_var=prep["ar1_bias_var"],
        )
    finally:
        # 恢复原始配置
        config.filter_config = old_cfg

    # ── 以传感器1去噪后轨迹为参考计算RMSE ──
    x_ref = np.interp(t_grid, prep["t1"], prep["x1_d"])
    y_ref = np.interp(t_grid, prep["t1"], prep["y1_d"])

    error_x = x_fused - x_ref
    error_y = y_fused - y_ref

    rmse_x = float(np.sqrt(np.mean(error_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(error_y ** 2)))
    rmse = float(np.sqrt(np.mean(error_x ** 2 + error_y ** 2)))

    return {
        "q_scale": q_scale,
        "rmse": rmse,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "t_grid": t_grid,
        "x_fused": x_fused,
        "y_fused": y_fused,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "error_x": error_x,
        "error_y": error_y,
    }


# ============================================================
#  批量扫描
# ============================================================
def sweep_q_scales(
    prep: Dict,
    scales: List[float],
) -> List[Dict]:
    """对所有Q缩放因子执行融合扫描，返回结果列表。

    Parameters
    ----------
    prep : dict
        预处理结果。
    scales : list[float]
        Q缩放因子列表。

    Returns
    -------
    list[dict] : 每个缩放因子的结果字典。
    """
    results = []
    n = len(scales)

    print("\n" + "=" * 60)
    print("  [扫描] Q矩阵缩放因子扫描")
    print("=" * 60)
    print(f"  {'scale':>8s}  {'RMSE (m)':>10s}  {'RMSE_X (m)':>12s}"
          f"  {'RMSE_Y (m)':>12s}")
    print("  " + "-" * 48)

    for i, scale in enumerate(scales):
        t_start = time.time()
        res = run_fusion_with_q_scale(prep, scale)
        elapsed = time.time() - t_start

        print(f"  {scale:>8.1f}  {res['rmse']:>10.4f}"
              f"  {res['rmse_x']:>12.4f}  {res['rmse_y']:>12.4f}"
              f"  ({elapsed:.1f}s)")

        results.append(res)

    # ── 找出最优 ──
    best = min(results, key=lambda r: r["rmse"])
    baseline = [r for r in results if r["q_scale"] == 1.0]
    baseline_rmse = baseline[0]["rmse"] if baseline else best["rmse"]

    print("  " + "-" * 48)
    print(f"  最优 Q scale = {best['q_scale']:.1f}, RMSE = {best['rmse']:.4f} m")
    if baseline:
        delta_pct = (best["rmse"] - baseline_rmse) / baseline_rmse * 100
        print(f"  相对默认 (scale=1.0) 改善: {delta_pct:+.2f}%")

    print("=" * 60)
    return results


# ============================================================
#  可视化
# ============================================================
def plot_sensitivity_curves(
    results: List[Dict],
    save_dir: Path,
) -> None:
    """绘制Q矩阵敏感性分析曲线图。

    生成两张图：
      1. RMSE vs Q Scale —— 主敏感性曲线（线图 + 最优/默认标注）
      2. 双轴误差分解 —— RMSE_X 和 RMSE_Y 并排对比
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    scales = np.array([r["q_scale"] for r in results])
    rmse = np.array([r["rmse"] for r in results])
    rmse_x = np.array([r["rmse_x"] for r in results])
    rmse_y = np.array([r["rmse_y"] for r in results])

    best_idx = int(np.argmin(rmse))
    baseline_idx = int(np.argmin(np.abs(scales - 1.0)))

    # ══════════════════════════════════════════════
    #  图1：主敏感性曲线
    # ══════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        scales, rmse,
        color=_COLOR_LINE, linewidth=plot_config.linewidth,
        marker="o", markersize=8, markerfacecolor="white",
        markeredgecolor=_COLOR_LINE, markeredgewidth=2,
        zorder=3,
    )

    # 默认值标注（scale=1.0）
    ax.scatter(
        scales[baseline_idx], rmse[baseline_idx],
        s=160, marker="s", facecolor=_COLOR_BASELINE,
        edgecolors="black", linewidths=1.2, zorder=5,
        label=f"默认配置 (scale=1.0)\nRMSE={rmse[baseline_idx]:.4f} m",
    )

    # 最优值标注
    ax.scatter(
        scales[best_idx], rmse[best_idx],
        s=180, marker="*", facecolor=plot_config.COLORS[4],
        edgecolors="black", linewidths=1.2, zorder=5,
        label=f"最优配置 (scale={scales[best_idx]:.1f})\nRMSE={rmse[best_idx]:.4f} m",
    )

    # 相对改善标注
    if best_idx != baseline_idx:
        delta_pct = (rmse[best_idx] - rmse[baseline_idx]) / rmse[baseline_idx] * 100
        mid_x = (scales[best_idx] + scales[baseline_idx]) / 2
        mid_y = (rmse[best_idx] + rmse[baseline_idx]) / 2
        ax.annotate(
            f"{delta_pct:+.1f}%",
            (mid_x, mid_y),
            fontsize=11, fontweight="bold", color=_COLOR_BASELINE,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Q矩阵对角缩放因子 (log scale)", fontsize=plot_config.label_fontsize)
    ax.set_ylabel("融合轨迹 RMSE (m)", fontsize=plot_config.label_fontsize)
    ax.set_title("Q矩阵过程噪声 — 敏感性分析 (问题2数据)",
                 fontsize=plot_config.title_fontsize, fontweight="bold")
    ax.legend(loc="best", fontsize=plot_config.legend_fontsize,
              frameon=plot_config.legend_frameon)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_dir / "sensitivity_q_rmse.png", dpi=plot_config.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[SA] 图1已保存: {save_dir / 'sensitivity_q_rmse.png'}")

    # ══════════════════════════════════════════════
    #  图2：X/Y 方向误差分解
    # ══════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        scales, rmse_x,
        color=plot_config.COLORS[1], linewidth=plot_config.linewidth,
        marker="^", markersize=7, label=f"RMSE_X (最优 scale={scales[np.argmin(rmse_x)]:.1f})",
    )
    ax.plot(
        scales, rmse_y,
        color=plot_config.COLORS[2], linewidth=plot_config.linewidth,
        marker="v", markersize=7, label=f"RMSE_Y (最优 scale={scales[np.argmin(rmse_y)]:.1f})",
    )
    ax.plot(
        scales, rmse,
        color=_COLOR_LINE, linewidth=plot_config.linewidth + 0.5,
        marker="o", markersize=8, markerfacecolor="white",
        markeredgecolor=_COLOR_LINE, markeredgewidth=2,
        label=f"RMSE 合成",
        zorder=3,
    )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
               label="默认配置 (scale=1.0)")

    ax.set_xscale("log")
    ax.set_xlabel("Q矩阵对角缩放因子 (log scale)", fontsize=plot_config.label_fontsize)
    ax.set_ylabel("RMSE (m)", fontsize=plot_config.label_fontsize)
    ax.set_title("Q矩阵敏感性 — X/Y方向误差分解",
                 fontsize=plot_config.title_fontsize, fontweight="bold")
    ax.legend(loc="best", fontsize=plot_config.legend_fontsize - 1,
              frameon=plot_config.legend_frameon)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_dir / "sensitivity_q_decomposition.png", dpi=plot_config.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[SA] 图2已保存: {save_dir / 'sensitivity_q_decomposition.png'}")


# ============================================================
#  结果保存
# ============================================================
def save_results_table(
    results: List[Dict],
    save_path: Path,
) -> None:
    """将敏感性扫描结果保存为Excel表格。"""
    rows = []
    for r in results:
        rows.append({
            "Q缩放因子": r["q_scale"],
            "RMSE (m)": round(r["rmse"], 6),
            "RMSE_X (m)": round(r["rmse_x"], 6),
            "RMSE_Y (m)": round(r["rmse_y"], 6),
        })

    df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"[SA] 结果表已保存: {save_path}")


# ============================================================
#  run() 入口（供 main.py 通过 import 调用）
# ============================================================
def run() -> None:
    """执行完整的Q矩阵敏感性分析流程。

    此函数可由 main.py 通过 ``from sensitivity_analysis import run; run()``
    调用，也可由本脚本直接运行时触发。
    """
    t_total_start = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  Q矩阵过程噪声 — 敏感性分析（独立脚本）          ║")
    print("╚" + "═" * 58 + "╝")
    print(f"[SA] Q原始对角元: {filter_config.Q}")
    print(f"[SA] 扫描因子: {Q_SCALES}")
    print(f"[SA] 目标频率: {time_config.target_freq} Hz")

    # ── 确保输出目录存在 ──
    ensure_dirs()

    # ── 步骤1：加载问题2数据 ──
    t1, x1, y1, t2, x2, y2 = load_problem2_data()

    # ── 步骤2：预处理（去噪+对齐+偏差，仅执行一次）──
    prep = preprocess_pipeline(t1, x1, y1, t2, x2, y2)

    # ── 步骤3：Q缩放因子扫描 ──
    results = sweep_q_scales(prep, Q_SCALES)

    # ── 步骤4：保存结果表格 ──
    save_results_table(
        results,
        Path(TABLE_DIR) / "sensitivity_q_results.xlsx",
    )

    # ── 步骤5：绘制敏感性曲线 ──
    plot_sensitivity_curves(results, Path(PLOT_DIR))

    # ── 步骤6：汇总 ──
    t_total = time.time() - t_total_start
    print(f"\n[SA] 敏感性分析完成，总耗时: {t_total:.1f}s")
    print(f"[SA] 图表目录: {PLOT_DIR}")
    print(f"[SA] 结果表格: {TABLE_DIR}/sensitivity_q_results.xlsx")


# ============================================================
#  直接运行入口
# ============================================================
if __name__ == "__main__":
    run()
