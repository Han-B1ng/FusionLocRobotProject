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

def iterative_bias_estimation(
    x2_aligned: np.ndarray,
    y2_aligned: np.ndarray,
    x1_aligned: np.ndarray,
    y1_aligned: np.ndarray,
    max_iter: int = 5,
    threshold: float = 3.0,
) -> tuple:
    dx_all = x2_aligned - x1_aligned
    dy_all = y2_aligned - y1_aligned
    mask = np.ones(len(dx_all), dtype=bool)

    for iteration in range(max_iter):
        dx_clean = dx_all[mask]
        dy_clean = dy_all[mask]

        if len(dx_clean) < 10:
            break

        bias_x = float(np.median(dx_clean))
        bias_y = float(np.median(dy_clean))

        dx_res = dx_all - bias_x
        dy_res = dy_all - bias_y

        sigma_x = np.std(dx_res[mask])
        sigma_y = np.std(dy_res[mask])

        new_outliers = (np.abs(dx_res) > threshold * sigma_x) | \
                       (np.abs(dy_res) > threshold * sigma_y)

        n_new = int(np.sum(new_outliers & mask))
        if n_new == 0:
            break

        mask = mask & ~new_outliers

    bias_x = float(np.median(dx_all[mask]))
    bias_y = float(np.median(dy_all[mask]))

    dx = dx_all - bias_x
    dy = dy_all - bias_y

    return bias_x, bias_y, dx, dy, mask

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

def load_problem2_data() -> tuple:
    file_path = data_path.path2

    if not file_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break

    print(f"[Problem2] 加载文件: {file_path}")

    # 分别读取两个工作表
    df1 = pd.read_excel(file_path, sheet_name='方式1(4Hz)', engine="openpyxl")
    df2 = pd.read_excel(file_path, sheet_name='方式2(5Hz)', engine="openpyxl")

    # 统一列名
    col_map = {
        '时间(s)': 't',
        'X坐标(m)': 'x',
        'Y坐标(m)': 'y',
    }
    df1 = df1.rename(columns=col_map)
    df2 = df2.rename(columns=col_map)

    # 数据清洗
    for df in (df1, df2):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    t1 = df1['t'].values.astype(np.float64)
    x1 = df1['x'].values.astype(np.float64)
    y1 = df1['y'].values.astype(np.float64)
    t2 = df2['t'].values.astype(np.float64)
    x2 = df2['x'].values.astype(np.float64)
    y2 = df2['y'].values.astype(np.float64)

    print(f"[Problem2] 传感器1 (方式1, 4Hz): {len(t1)} 点")
    print(f"[Problem2] 传感器2 (方式2, 5Hz): {len(t2)} 点")
    return t1, x1, y1, t2, x2, y2

def run_denoise_comparison(x: np.ndarray, y: np.ndarray, sensor_name: str) -> tuple:
    wavelet_list = ("db4", "sym5")
    thresh_methods = ("universal", "bayes")

    results = compare_denoise_configs(
        x, y, wavelet_list=wavelet_list, thresh_methods=thresh_methods,
    )

    header = f"  {'wavelet':<10} {'thresh_method':<15} {'var_x':>10} {'var_y':>10} {'accel_var_x':>12} {'accel_var_y':>12}"
    sep = "  " + "-" * 72
    print(f"\n  [{sensor_name}] 去噪参数对比:")
    print(header)
    print(sep)

    for (wv, tm), metrics in results.items():
        print(f"  {wv:<10} {tm:<15} {metrics['var_x']:>10.4f} {metrics['var_y']:>10.4f} {metrics['accel_var_x']:>12.4f} {metrics['accel_var_y']:>12.4f}")

    best_key = min(results, key=lambda k: results[k]["accel_var_x"] + results[k]["accel_var_y"])
    best_wavelet, best_thresh = best_key
    best_metrics = results[best_key]

    print(sep)
    print(f"  最优组合: wavelet='{best_wavelet}', threshold_method='{best_thresh}'")
    return best_wavelet, best_thresh

def plot_problem2_results(
    t1, x1, y1, t2, x2, y2,
    t_grid, x_fused, y_fused, bias_x_arr, bias_y_arr,
    dx, dy, delay, output_dir: Path,
):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    axes[0].scatter(t1, x1, s=3, color=_COLOR_S1, alpha=0.5, label="Sensor1 X")
    axes[0].scatter(t2, x2, s=3, color=_COLOR_S2, alpha=0.5, label="Sensor2 X")
    axes[0].plot(t_grid, x_fused, color=_COLOR_FUSED, lw=1, label="Fused X")
    axes[0].set_ylabel("X (m)")
    axes[0].set_title(f"Problem2 Trajectory (delay={delay:.4f}s)")
    axes[0].legend()

    axes[1].scatter(t1, y1, s=3, color=_COLOR_S1, alpha=0.5, label="Sensor1 Y")
    axes[1].scatter(t2, y2, s=3, color=_COLOR_S2, alpha=0.5, label="Sensor2 Y")
    axes[1].plot(t_grid, y_fused, color=_COLOR_FUSED, lw=1, label="Fused Y")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_trajectory.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(dx, bins=50, color=_COLOR_S1, alpha=0.7)
    axes[0].axvline(np.median(dx), c='r', ls='--', label=f"median={np.median(dx):.3f}")
    axes[0].set_xlabel("dx (m)")
    axes[0].legend()

    axes[1].hist(dy, bins=50, color=_COLOR_S2, alpha=0.7)
    axes[1].axvline(np.median(dy), c='r', ls='--', label=f"median={np.median(dy):.3f}")
    axes[1].set_xlabel("dy (m)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_residuals.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2,1,figsize=(14,8), sharex=True)
    axes[0].plot(t_grid, bias_x_arr, c=_COLOR_S1, lw=0.8)
    axes[0].set_ylabel("Bias X (m)")
    axes[1].plot(t_grid, bias_y_arr, c=_COLOR_S2, lw=0.8)
    axes[1].set_ylabel("Bias Y (m)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(figures_dir / "Problem2_bias.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    output_dir = data_path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("[Problem2] 数据加载完成")
    t1, x1, y1, t2, x2, y2 = load_problem2_data()

    print("\n" + "=" * 60)
    print("  Step 2: 小波去噪参数对比实验")
    print("=" * 60)
    print(f"\n  传感器1 去噪前 X 方差: {np.var(x1):.4f}, Y 方差: {np.var(y1):.4f}")
    print(f"  传感器2 去噪前 X 方差: {np.var(x2):.4f}, Y 方差: {np.var(y2):.4f}")

    best_wl1, best_tm1 = run_denoise_comparison(x1, y1, "传感器1")
    best_wl2, best_tm2 = run_denoise_comparison(x2, y2, "传感器2")

    x1_d, y1_d = denoise_trajectory(x1, y1, wavelet=best_wl1, threshold_method=best_tm1)
    x2_d, y2_d = denoise_trajectory(x2, y2, wavelet=best_wl2, threshold_method=best_tm2)

    print(f"\n  传感器1 去噪后 X 方差: {np.var(x1_d):.4f}, Y 方差: {np.var(y1_d):.4f}")
    print(f"  传感器2 去噪后 X 方差: {np.var(x2_d):.4f}, Y 方差: {np.var(y2_d):.4f}")

    print("\n" + "=" * 60)
    print("  Step 3: 时间对齐")
    print("=" * 60)
    delay, t_align, x_fused_init, y_fused_init = align_sensors(
        t1, x1_d, y1_d, t2, x2_d, y2_d,
        target_freq=time_config.target_freq,
        delay_range=alignment_config.delay_range,
        method=alignment_config.method, w1=0.5, w2=0.5
    )
    print(f"[Problem2] 估计时间偏差: {delay:+.4f} s")

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

    print("\n" + "=" * 60)
    print("  Step 4: 系统偏差估计")
    print("=" * 60)
    bias_cmp = compare_bias_methods(x2_aligned, y2_aligned, x1_aligned, y1_aligned, consistency_threshold=0.1)
    print(f"\n  中位数估计:     bias_x={bias_cmp['median'][0]:+.4f} m, bias_y={bias_cmp['median'][1]:+.4f} m")
    print(f"  截尾均值估计:   bias_x={bias_cmp['robust_mean'][0]:+.4f} m, bias_y={bias_cmp['robust_mean'][1]:+.4f} m")

    bias_x, bias_y, dx, dy, clean_mask = iterative_bias_estimation(x2_aligned, y2_aligned, x1_aligned, y1_aligned)
    n_outliers = int(np.sum(~clean_mask))
    print(f"\n[Problem2] 迭代剔除完成，剔除 {n_outliers} 个异常点")
    print(f"[Problem2] 最终系统偏差: bias_x={bias_x:+.4f} m, bias_y={bias_y:+.4f} m")

    bias_x, bias_y = bias_cmp["median"]
    _, _, dx, dy = estimate_systematic_bias(x2_aligned, y2_aligned, x1_aligned, y1_aligned, method="median")
    print(f"\n[Problem2] 最终采用系统偏差 (median): bias_x={bias_x:+.4f} m, bias_y={bias_y:+.4f} m")

    anomalies_x = detect_anomalies(dx, threshold=3.0)
    anomalies_y = detect_anomalies(dy, threshold=3.0)
    print(f"[Problem2] 异常点: X方向 {len(anomalies_x)} 个, Y方向 {len(anomalies_y)} 个")

    print("\n" + "=" * 60)
    print("  Step 5: 偏差显著性检验")
    print("=" * 60)
    sig_x, p_x = bias_significance_test(dx, alpha=0.05)
    sig_y, p_y = bias_significance_test(dy, alpha=0.05)
    print(f"[Problem2] 偏差显著性检验: dx p={p_x:.4f} ({'显著' if sig_x else '不显著'}), dy p={p_y:.4f} ({'显著' if sig_y else '不显著'})")

    # ======================================================
    # Step 5.5: AR(1) 偏差漂移建模
    # ======================================================
    print("\n" + "=" * 60)
    print("  Step 5.5: AR(1) 偏差漂移建模")
    print("=" * 60)

    from core.kalman_filters import estimate_ar1_params

    ar1_alpha, ar1_bias_var = estimate_ar1_params(dx, dy, dt_ref=0.1)
    ar1_rho = np.exp(-ar1_alpha * 0.1)

    print(f"  AR(1) 系数 ρ = {ar1_rho:.4f}")
    print(f"  均值回复速率 α = {ar1_alpha:.4f} /s")
    print(f"  平稳方差 σ_b² = {ar1_bias_var:.6f} m²")
    if ar1_rho > 0.95:
        print("  结论: 偏差高度持续（接近恒定），AR(1) ≈ 常数模型")
    elif ar1_rho > 0.5:
        print("  结论: 偏差缓慢漂移，AR(1) 建模有意义")
    else:
        print("  结论: 偏差快速变化，AR(1) 建模有效")

    print("\n" + "=" * 60)
    print("  Step 6: 自适应观测噪声估计")
    print("=" * 60)
    R1_est, R2_est = estimate_adaptive_R(t1, x1_d, y1_d, dx, dy)
    print(f"\n  默认 R1 对角: [{filter_config.R1[0]:.4f}, {filter_config.R1[1]:.4f}]")
    print(f"  自适应 R1:\n{R1_est}")
    print(f"\n  默认 R2 对角: [{filter_config.R2[0]:.4f}, {filter_config.R2[1]:.4f}]")
    print(f"  自适应 R2:\n{R2_est}")

    print("\n" + "=" * 60)
    print("  Step 7: EKF 融合")
    print("=" * 60)
    t2_for_fuse = t2 - delay

    # --- 7a: 默认 R 融合
    print("\n  [默认 R] 融合中...")
    t_grid_def, x_fused_def, y_fused_def, _, _ = fuse_sensors(
        t1, x1_d, y1_d, t2_for_fuse, x2_d, y2_d,
        target_freq=time_config.target_freq,
        ar1_alpha=ar1_alpha,
        ar1_bias_var=ar1_bias_var
    )

    # --- 7b: 自适应 R 融合
    print("  [自适应 R] 融合中...")
    t_grid_adp, x_fused_adp, y_fused_adp, bias_x_arr, bias_y_arr = fuse_sensors(
        t1, x1_d, y1_d, t2_for_fuse, x2_d, y2_d,
        target_freq=time_config.target_freq,
        R1_est=R1_est, R2_est=R2_est,
        ar1_alpha=ar1_alpha,
        ar1_bias_var=ar1_bias_var
    )

    common_len = min(len(t_grid_def), len(t_grid_adp))
    x_ref_interp = np.interp(t_grid_def[:common_len], t1, x1_d)
    y_ref_interp = np.interp(t_grid_def[:common_len], t1, y1_d)

    resid_var_def = np.var(x_fused_def[:common_len]-x_ref_interp) + np.var(y_fused_def[:common_len]-y_ref_interp)
    resid_var_adp = np.var(x_fused_adp[:common_len]-x_ref_interp) + np.var(y_fused_adp[:common_len]-y_ref_interp)

    print(f"\n  默认 R 融合残差方差:   {resid_var_def:.6f}")
    print(f"  自适应 R 融合残差方差: {resid_var_adp:.6f}")

    if resid_var_adp < resid_var_def:
        print("  -> 自适应 R 更优，采用自适应 R 融合结果。")
        t_grid, x_fused, y_fused = t_grid_adp, x_fused_adp, y_fused_adp
    else:
        print("  -> 默认 R 更优或持平，保持默认 R 融合结果。")
        t_grid, x_fused, y_fused = t_grid_def, x_fused_def, y_fused_def
        _, _, _, bias_x_arr, bias_y_arr = fuse_sensors(
            t1, x1_d, y1_d, t2_for_fuse, x2_d, y2_d,
            target_freq=time_config.target_freq,
            ar1_alpha=ar1_alpha,
            ar1_bias_var=ar1_bias_var
        )

    print(f"\n[Problem2] 融合完成，生成 {time_config.target_freq:.0f}Hz 轨迹 {len(t_grid)} 点")

    df_result = pd.DataFrame({
        "Time(s)": np.round(t_grid,4),
        "X(m)": np.round(x_fused,6),
        "Y(m)": np.round(y_fused,6),
        "bias_x(m)": np.round(bias_x_arr,6),
        "bias_y(m)": np.round(bias_y_arr,6)
    })
    xlsx_path = output_dir / "Problem2_10Hz.xlsx"
    df_result.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"[Problem2] 结果已保存至 {xlsx_path}")

    plot_problem2_results(t1,x1,y1,t2,x2,y2, t_grid,x_fused,y_fused, bias_x_arr,bias_y_arr, dx,dy,delay, output_dir)

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
    print("=" * 60)
    print("\n[Problem2] 问题 2 求解完毕。")