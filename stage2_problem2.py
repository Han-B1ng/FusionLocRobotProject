# file: stage2_problem2.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 问题2求解：加载附件2 → 去噪 → 时间对齐 → 偏差估计 → EKF融合 → 输出

"""
阶段 2 — 问题 2：含噪声 + 系统偏差的传感器融合。

附件 2 的两类传感器数据存在随机测量噪声和固定系统偏差。
本模块完成：
  1. 加载附件 2 的两个传感器 sheet
  2. 小波去噪（含去噪参数对比实验：2 小波基 × 2 阈值策略 = 4 种组合）
  3. 时间对齐（估计时偏）
  4. 系统偏差估计与显著性检验（含 median vs robust_mean 对比）
  5. EKF 融合输出 10Hz 轨迹（含自适应 R 与默认 R 对比）
  6. 保存 Excel + 可视化

依赖：config.py, core/wavelet_utils.py, core/time_alignment.py,
      core/robust_stats.py, core/kalman_filters.py
后续：stage3 复用本模块流程处理实际数据
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import alignment_config, data_path, filter_config, time_config
from core.kalman_filters import estimate_adaptive_R, fuse_sensors
from core.robust_stats import (
    bias_significance_test,
    compare_bias_methods,
    detect_anomalies,
    estimate_systematic_bias,
)
from core.time_alignment import align_sensors
from core.wavelet_utils import (
    adaptive_denoise_trajectory,
    compare_denoise_configs,
    denoise_trajectory,
)

# ============================================================
#  全局绘图样式
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

_COLOR_S1 = "#2563EB"
_COLOR_S2 = "#DC2626"
_COLOR_FUSED = "#16A34A"


# ============================================================
#  数据加载
# ============================================================
def load_problem2_data() -> tuple:
    """加载附件 2 的两个传感器 sheet。

    Returns
    -------
    t1, x1, y1, t2, x2, y2 : np.ndarray
    """
    file_path = data_path.path2

    if not file_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break

    print(f"[Problem2] 加载文件: {file_path}")

    df1 = pd.read_excel(
        file_path, sheet_name="方式1(4Hz)", engine="openpyxl"
    )
    df2 = pd.read_excel(
        file_path, sheet_name="方式2(5Hz)", engine="openpyxl"
    )

    col_map = {
        "时间(s)": "t", "时间": "t", "Time": "t", "time": "t", "t": "t",
        "X坐标(m)": "x", "X坐标": "x", "X": "x", "x": "x",
        "Y坐标(m)": "y", "Y坐标": "y", "Y": "y", "y": "y",
    }
    df1 = df1.rename(columns=col_map)[["t", "x", "y"]]
    df2 = df2.rename(columns=col_map)[["t", "x", "y"]]

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

    print(
        f"[Problem2] 传感器1: {len(t1)} 点, "
        f"[{t1[0]:.2f}, {t1[-1]:.2f}] s"
    )
    print(
        f"[Problem2] 传感器2: {len(t2)} 点, "
        f"[{t2[0]:.2f}, {t2[-1]:.2f}] s"
    )

    return t1, x1, y1, t2, x2, y2


# ============================================================
#  去噪参数对比实验
# ============================================================
def run_denoise_comparison(
    x: np.ndarray,
    y: np.ndarray,
    sensor_name: str,
) -> tuple:
    """对单个传感器执行去噪参数对比实验。

    比较 2 种小波基 × 2 种阈值策略 = 4 种组合，
    以加速度方差（平滑性）为评价指标选出最优组合。

    Parameters
    ----------
    x, y : np.ndarray
        原始轨迹坐标 (m)。
    sensor_name : str
        传感器名称，用于打印标识。

    Returns
    -------
    best_wavelet : str
        最优小波基名称。
    best_thresh : str
        最优阈值策略名称。
    """
    wavelet_list = ("db4", "sym5")
    thresh_methods = ("universal", "bayes")

    results = compare_denoise_configs(
        x, y,
        wavelet_list=wavelet_list,
        thresh_methods=thresh_methods,
    )

    # 表头
    header = (
        f"  {'wavelet':<10} {'thresh_method':<15} "
        f"{'var_x':>10} {'var_y':>10} "
        f"{'accel_var_x':>12} {'accel_var_y':>12}"
    )
    sep = "  " + "-" * 72

    print(f"\n  [{sensor_name}] 去噪参数对比:")
    print(header)
    print(sep)

    for (wv, tm), metrics in results.items():
        print(
            f"  {wv:<10} {tm:<15} "
            f"{metrics['var_x']:>10.4f} {metrics['var_y']:>10.4f} "
            f"{metrics['accel_var_x']:>12.4f} {metrics['accel_var_y']:>12.4f}"
        )

    # 最优组合：加速度方差之和最小（最平滑）
    best_key = min(
        results,
        key=lambda k: results[k]["accel_var_x"] + results[k]["accel_var_y"],
    )
    best_wavelet, best_thresh = best_key
    best_metrics = results[best_key]

    print(sep)
    print(
        f"  最优组合: wavelet='{best_wavelet}', "
        f"threshold_method='{best_thresh}' "
        f"(accel_var_sum="
        f"{best_metrics['accel_var_x'] + best_metrics['accel_var_y']:.4f})"
    )

    return best_wavelet, best_thresh


# ============================================================
#  绘图
# ============================================================
def plot_problem2_results(
    t1: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, x2: np.ndarray, y2: np.ndarray,
    t_grid: np.ndarray,
    x_fused: np.ndarray, y_fused: np.ndarray,
    bias_x_arr: np.ndarray, bias_y_arr: np.ndarray,
    dx: np.ndarray, dy: np.ndarray,
    delay: float,
    output_dir: Path,
) -> None:
    """绘制问题 2 的所有结果图。"""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- 图 1：原始数据 + 融合轨迹 ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].scatter(t1, x1, s=3, color=_COLOR_S1, alpha=0.5, label="Sensor1 X")
    axes[0].scatter(t2, x2, s=3, color=_COLOR_S2, alpha=0.5, label="Sensor2 X")
    axes[0].plot(t_grid, x_fused, color=_COLOR_FUSED, linewidth=1.0,
                 label="Fused X")
    axes[0].set_ylabel("X (m)", fontsize=12)
    axes[0].set_title(
        f"Problem 2 — Fused Trajectory (delay={delay:+.4f}s)",
        fontsize=14, fontweight="bold",
    )
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].scatter(t1, y1, s=3, color=_COLOR_S1, alpha=0.5, label="Sensor1 Y")
    axes[1].scatter(t2, y2, s=3, color=_COLOR_S2, alpha=0.5, label="Sensor2 Y")
    axes[1].plot(t_grid, y_fused, color=_COLOR_FUSED, linewidth=1.0,
                 label="Fused Y")
    axes[1].set_ylabel("Y (m)", fontsize=12)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_trajectory.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 图 2：残差直方图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(dx, bins=50, color=_COLOR_S1, alpha=0.7, edgecolor="white")
    axes[0].axvline(np.median(dx), color="red", linestyle="--",
                    label=f"median={np.median(dx):.3f}m")
    axes[0].set_xlabel("dx (m)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Residual X", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].hist(dy, bins=50, color=_COLOR_S2, alpha=0.7, edgecolor="white")
    axes[1].axvline(np.median(dy), color="red", linestyle="--",
                    label=f"median={np.median(dy):.3f}m")
    axes[1].set_xlabel("dy (m)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Residual Y", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_residuals.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 图 3：偏差随时间变化 ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t_grid, bias_x_arr, color=_COLOR_S1, linewidth=0.8)
    axes[0].set_ylabel("Bias X (m)", fontsize=12)
    axes[0].set_title(
        "Estimated Systematic Bias Over Time",
        fontsize=14, fontweight="bold",
    )

    axes[1].plot(t_grid, bias_y_arr, color=_COLOR_S2, linewidth=0.8)
    axes[1].set_ylabel("Bias Y (m)", fontsize=12)
    axes[1].set_xlabel("Time (s)", fontsize=12)

    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_bias.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"[Problem2] 图表已保存至 {figures_dir}")


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    output_dir = data_path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ======================================================
    # Step 1: 加载数据
    # ======================================================
    print("[Problem2] 数据加载完成")
    t1, x1, y1, t2, x2, y2 = load_problem2_data()

    # ======================================================
    # Step 2: 去噪参数对比实验
    #     2 小波基 (db4, sym5) × 2 阈值策略 (universal, bayes)
    #     = 4 种组合，以加速度方差（平滑性）为指标选最优
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 2: 小波去噪参数对比实验")
    print("=" * 60)

    print(f"\n  传感器1 去噪前 X 方差: {np.var(x1):.4f}, "
          f"Y 方差: {np.var(y1):.4f}")
    print(f"  传感器2 去噪前 X 方差: {np.var(x2):.4f}, "
          f"Y 方差: {np.var(y2):.4f}")

    # 传感器 1 对比
    best_wl1, best_tm1 = run_denoise_comparison(x1, y1, "传感器1")

    # 传感器 2 对比
    best_wl2, best_tm2 = run_denoise_comparison(x2, y2, "传感器2")

    # 用最优组合分别去噪
    x1_d, y1_d = denoise_trajectory(
        x1, y1, wavelet=best_wl1, threshold_method=best_tm1,
    )
    x2_d, y2_d = denoise_trajectory(
        x2, y2, wavelet=best_wl2, threshold_method=best_tm2,
    )

    print(f"\n  传感器1 去噪后 X 方差: {np.var(x1_d):.4f}, "
          f"Y 方差: {np.var(y1_d):.4f}")
    print(f"  传感器2 去噪后 X 方差: {np.var(x2_d):.4f}, "
          f"Y 方差: {np.var(y2_d):.4f}")

    # ======================================================
    # Step 3: 时间对齐（获取时偏）
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 3: 时间对齐")
    print("=" * 60)

    delay, t_align, x_fused_init, y_fused_init = align_sensors(
        t1, x1_d, y1_d,
        t2, x2_d, y2_d,
        target_freq=time_config.target_freq,
        delay_range=alignment_config.delay_range,
        method=alignment_config.method,
        w1=0.5, w2=0.5,
    )
    print(f"[Problem2] 估计时间偏差: {delay:+.4f} s")

    # 手动对齐两组数据到公共网格（供偏差估计使用）
    t2_corrected = t2 - delay
    t_start_align = max(t1.min(), t2_corrected.min())
    t_end_align = min(t1.max(), t2_corrected.max())
    dt_target = 1.0 / time_config.target_freq
    n_steps = int(np.floor((t_end_align - t_start_align) / dt_target))
    t_grid_align = t_start_align + np.arange(n_steps + 1) * dt_target
    t_grid_align = np.clip(t_grid_align, t_start_align, t_end_align)

    x1_aligned = np.interp(t_grid_align, t1, x1_d)
    y1_aligned = np.interp(t_grid_align, t1, y1_d)
    x2_aligned = np.interp(t_grid_align, t2_corrected, x2_d)
    y2_aligned = np.interp(t_grid_align, t2_corrected, y2_d)

    # ======================================================
    # Step 4: 系统偏差估计（含方法对比）
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 4: 系统偏差估计")
    print("=" * 60)

    # --- 4a: median vs robust_mean 对比 ---
    bias_cmp = compare_bias_methods(
        x2_aligned, y2_aligned,
        x1_aligned, y1_aligned,
        consistency_threshold=0.1,
    )

    print(f"\n  中位数估计:     bias_x={bias_cmp['median'][0]:+.4f} m, "
          f"bias_y={bias_cmp['median'][1]:+.4f} m")
    print(f"  截尾均值估计:   bias_x={bias_cmp['robust_mean'][0]:+.4f} m, "
          f"bias_y={bias_cmp['robust_mean'][1]:+.4f} m")
    print(f"  差异: Δx={bias_cmp['diff_x']:.4f} m, "
          f"Δy={bias_cmp['diff_y']:.4f} m")
    print(f"  结论: {bias_cmp['message']}")

    # 最终用于后续融合的偏差采用中位数估计值（保持原逻辑不变）
    bias_x, bias_y = bias_cmp["median"]

    # --- 4b: 用中位数方法获取残差序列（供异常检测和后续使用）---
    _, _, dx, dy = estimate_systematic_bias(
        x2_aligned, y2_aligned,
        x1_aligned, y1_aligned,
        method="median",
    )

    print(f"\n[Problem2] 最终采用系统偏差 (median): "
          f"bias_x={bias_x:+.4f} m, bias_y={bias_y:+.4f} m")

    # --- 4c: 异常点检测 ---
    anomalies_x = detect_anomalies(dx, threshold=3.0)
    anomalies_y = detect_anomalies(dy, threshold=3.0)
    print(f"[Problem2] 异常点: X方向 {len(anomalies_x)} 个, "
          f"Y方向 {len(anomalies_y)} 个")

    # ======================================================
    # Step 5: 偏差显著性检验
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 5: 偏差显著性检验")
    print("=" * 60)

    sig_x, p_x = bias_significance_test(dx, alpha=0.05)
    sig_y, p_y = bias_significance_test(dy, alpha=0.05)

    print(f"[Problem2] 偏差显著性检验: "
          f"dx p={p_x:.4f} ({'显著' if sig_x else '不显著'}), "
          f"dy p={p_y:.4f} ({'显著' if sig_y else '不显著'})")

    if not sig_x and not sig_y:
        print("[Problem2] 提示: 未检测到显著系统偏差，"
              "但以下估计值仍作为参考输出。")

    # ======================================================
    # Step 6: 自适应观测噪声估计
    #     R1：从传感器 1 速度差分协方差估计
    #     R2：从对齐后残差协方差估计
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 6: 自适应观测噪声估计")
    print("=" * 60)

    R1_est, R2_est = estimate_adaptive_R(
        t1, x1_d, y1_d,
        dx, dy,
    )

    print(f"\n  默认 R1 对角: [{filter_config.R1[0]:.4f}, "
          f"{filter_config.R1[1]:.4f}]")
    print(f"  自适应 R1:\n{R1_est}")
    print(f"\n  默认 R2 对角: [{filter_config.R2[0]:.4f}, "
          f"{filter_config.R2[1]:.4f}]")
    print(f"  自适应 R2:\n{R2_est}")

    # ======================================================
    # Step 7: EKF 融合（默认 R vs 自适应 R 对比）
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 7: EKF 融合")
    print("=" * 60)

    t2_for_fuse = t2 - delay

    # --- 7a: 默认 R 融合 ---
    print("\n  [默认 R] 融合中...")
    t_grid_def, x_fused_def, y_fused_def, _, _ = fuse_sensors(
        t1, x1_d, y1_d,
        t2_for_fuse, x2_d, y2_d,
        target_freq=time_config.target_freq,
    )

    # --- 7b: 自适应 R 融合 ---
    print("  [自适应 R] 融合中...")
    t_grid_adp, x_fused_adp, y_fused_adp, bias_x_arr, bias_y_arr = (
        fuse_sensors(
            t1, x1_d, y1_d,
            t2_for_fuse, x2_d, y2_d,
            target_freq=time_config.target_freq,
            R1_est=R1_est,
            R2_est=R2_est,
        )
    )

    # --- 7c: 对比两种融合结果 ---
    #     用传感器 1 作为参考，计算融合轨迹与参考的残差方差
    #     取两者公共时间范围
    common_len = min(len(t_grid_def), len(t_grid_adp))
    x_ref_interp = np.interp(
        t_grid_def[:common_len], t1, x1_d,
    )
    y_ref_interp = np.interp(
        t_grid_def[:common_len], t1, y1_d,
    )

    resid_var_def = (
        np.var(x_fused_def[:common_len] - x_ref_interp)
        + np.var(y_fused_def[:common_len] - y_ref_interp)
    )
    resid_var_adp = (
        np.var(x_fused_adp[:common_len] - x_ref_interp)
        + np.var(y_fused_adp[:common_len] - y_ref_interp)
    )

    print(f"\n  默认 R 融合残差方差:   {resid_var_def:.6f}")
    print(f"  自适应 R 融合残差方差: {resid_var_adp:.6f}")

    if resid_var_adp < resid_var_def:
        print("  -> 自适应 R 更优，采用自适应 R 融合结果。")
        t_grid = t_grid_adp
        x_fused = x_fused_adp
        y_fused = y_fused_adp
    else:
        print("  -> 默认 R 更优或持平，保持默认 R 融合结果。")
        t_grid = t_grid_def
        x_fused = x_fused_def
        y_fused = y_fused_def
        # 用默认 R 重新融合以获取偏差序列
        _, _, _, bias_x_arr, bias_y_arr = fuse_sensors(
            t1, x1_d, y1_d,
            t2_for_fuse, x2_d, y2_d,
            target_freq=time_config.target_freq,
        )

    print(f"\n[Problem2] 融合完成，生成 {time_config.target_freq:.0f}Hz 轨迹 "
          f"{len(t_grid)} 点")

    # ======================================================
    # Step 8: 保存结果
    # ======================================================
    df_result = pd.DataFrame({
        "Time(s)":   np.round(t_grid, 4),
        "X(m)":      np.round(x_fused, 6),
        "Y(m)":      np.round(y_fused, 6),
        "bias_x(m)": np.round(bias_x_arr, 6),
        "bias_y(m)": np.round(bias_y_arr, 6),
    })

    xlsx_path = output_dir / "Problem2_10Hz.xlsx"
    df_result.to_excel(xlsx_path, index=False, engine="openpyxl")
    size_kb = xlsx_path.stat().st_size / 1024
    print(f"[Problem2] 结果已保存至 {xlsx_path}  ({size_kb:.1f} KB)")

    # ======================================================
    # Step 9: 绘图
    # ======================================================
    plot_problem2_results(
        t1, x1, y1, t2, x2, y2,
        t_grid, x_fused, y_fused,
        bias_x_arr, bias_y_arr,
        dx, dy, delay,
        output_dir,
    )

    # ======================================================
    # 汇总输出
    # ======================================================
    print("\n" + "=" * 60)
    print("  问题 2 结果汇总")
    print("=" * 60)
    print(f"  时间偏差:         {delay:+.6f} s")
    print(f"  系统偏差 X:       {bias_x:+.6f} m")
    print(f"  系统偏差 Y:       {bias_y:+.6f} m")
    print(f"  偏差显著性 X:     p={p_x:.4f} ({'显著' if sig_x else '不显著'})")
    print(f"  偏差显著性 Y:     p={p_y:.4f} ({'显著' if sig_y else '不显著'})")
    print(f"  去噪最优(传感器1): wavelet='{best_wl1}', thresh='{best_tm1}'")
    print(f"  去噪最优(传感器2): wavelet='{best_wl2}', thresh='{best_tm2}'")
    print(f"  自适应 R 融合:    {'是' if resid_var_adp < resid_var_def else '否'}")
    print(f"  输出点数:         {len(t_grid)}")
    print(f"  时间范围:         [{t_grid[0]:.2f}, {t_grid[-1]:.2f}] s")
    print(f"  输出频率:         {time_config.target_freq:.0f} Hz")
    print("=" * 60)

    print("\n[Problem2] 问题 2 求解完毕。")
