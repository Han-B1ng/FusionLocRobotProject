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
        if len(dx_clean) < 10: break
        bias_x = float(np.median(dx_clean))
        bias_y = float(np.median(dy_clean))
        dx_res = dx_all - bias_x
        dy_res = dy_all - bias_y
        sigma_x = np.std(dx_res[mask])
        sigma_y = np.std(dy_res[mask])
        new_outliers = (np.abs(dx_res) > threshold * sigma_x) | (np.abs(dy_res) > threshold * sigma_y)
        n_new = int(np.sum(new_outliers & mask))
        if n_new == 0: break
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


def load_problem3_data() -> tuple:
    file_path = data_path.path3
    if not file_path.exists():
        for ext in [".xlsx", ".xls", ".csv"]:
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break
    print(f"[Problem3] 加载文件: {file_path}")
    df1 = pd.read_excel(file_path, sheet_name="方式1(4Hz)", engine="openpyxl")
    df2 = pd.read_excel(file_path, sheet_name="方式2(5Hz)", engine="openpyxl")
    col_map = {"时间(s)": "t", "X坐标(m)": "x", "Y坐标(m)": "y"}
    df1 = df1.rename(columns=col_map)[["t", "x", "y"]]
    df2 = df2.rename(columns=col_map)[["t", "x", "y"]]
    for df in [df1, df2]:
        for c in ["t", "x", "y"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)

    # 🔥 核心修复：添加.copy() 解决只读数组错误
    t1 = df1["t"].values.copy().astype(np.float64)
    x1 = df1["x"].values.copy().astype(np.float64)
    y1 = df1["y"].values.copy().astype(np.float64)
    t2 = df2["t"].values.copy().astype(np.float64)
    x2 = df2["x"].values.copy().astype(np.float64)
    y2 = df2["y"].values.copy().astype(np.float64)

    print(f"[Problem3] 传感器1: {len(t1)} 点")
    print(f"[Problem3] 传感器2: {len(t2)} 点")
    return t1, x1, y1, t2, x2, y2


def estimate_coarse_time_offset(t1, x1, y1, t2, x2, y2, search_range=(-500, 800), coarse_step=5, fine_step=0.01,
                                min_overlap=20):
    def _mse(d):
        t2s = t2 + d
        st, ed = max(t1.min(), t2s.min()), min(t1.max(), t2s.max())
        if ed - st < min_overlap: return np.inf
        tc = np.linspace(st, ed, max(60, int((ed - st) * 2)))
        return np.mean((np.interp(tc, t1, x1) - np.interp(tc, t2s, x2)) ** 2 + (
                    np.interp(tc, t1, y1) - np.interp(tc, t2s, y2)) ** 2)

    ds = np.arange(*search_range, coarse_step)
    cs = np.array([_mse(d) for d in ds])
    bd = ds[np.argmin(cs)]
    dsf = np.arange(bd - coarse_step, bd + coarse_step, fine_step)
    csf = np.array([_mse(d) for d in dsf])
    return float(dsf[np.argmin(csf)]), float(csf.min())


def run_denoise_comparison(x, y, sensor_name):
    wvs = ["db4", "sym5"]
    tms = ["universal", "bayes"]
    res = compare_denoise_configs(x, y, wvs, tms)
    best = min(res, key=lambda k: res[k]["accel_var_x"] + res[k]["accel_var_y"])
    return best


def plot_problem3_results(t1, x1, y1, t2, x2, y2, t_grid, x_fused, y_fused, bias_x_arr, bias_y_arr, dx, dy, delay,
                          output_dir):
    d = output_dir / "figures"
    d.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax[0].scatter(t1, x1, s=3, c=_COLOR_S1, alpha=0.5, label="S1 X")
    ax[0].scatter(t2, x2, s=3, c=_COLOR_S2, alpha=0.5, label="S2 X")
    ax[0].plot(t_grid, x_fused, c=_COLOR_FUSED, lw=1, label="Fused X")
    ax[1].scatter(t1, y1, s=3, c=_COLOR_S1, alpha=0.5, label="S1 Y")
    ax[1].scatter(t2, y2, s=3, c=_COLOR_S2, alpha=0.5, label="S2 Y")
    ax[1].plot(t_grid, y_fused, c=_COLOR_FUSED, lw=1, label="Fused Y")
    fig.savefig(d / "Problem3_trajectory.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    output_dir = data_path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    t1, x1, y1, t2, x2, y2 = load_problem3_data()

    print("\n" + "=" * 60)
    print("  Step 2: 小波去噪参数对比实验")
    print("=" * 60)
    best1 = run_denoise_comparison(x1, y1, "传感器1")
    best2 = run_denoise_comparison(x2, y2, "传感器2")
    best_wl1, best_tm1 = best1
    best_wl2, best_tm2 = best2
    x1_d, y1_d = denoise_trajectory(x1, y1, wavelet=best_wl1, threshold_method=best_tm1)
    x2_d, y2_d = denoise_trajectory(x2, y2, wavelet=best_wl2, threshold_method=best_tm2)

    print("\n" + "=" * 60)
    print("  Step 2.5: 粗略时间偏移估计")
    print("=" * 60)
    coarse_off, _ = estimate_coarse_time_offset(t1, x1_d, y1_d, t2, x2_d, y2_d)
    t2_shifted = t2 + coarse_off
    print(f"粗偏移: {coarse_off:.2f}s")

    print("\n" + "=" * 60)
    print("  Step 3: 精细时间对齐")
    print("=" * 60)
    fine_delay, _, _, _ = align_sensors(t1, x1_d, y1_d, t2_shifted, x2_d, y2_d, target_freq=time_config.target_freq)
    delay = fine_delay - coarse_off
    print(f"总时间偏差: {delay:.4f}s")

    t2c = t2 - delay
    sta, end = max(t1.min(), t2c.min()), min(t1.max(), t2c.max())
    dt = 1 / time_config.target_freq
    t_align = sta + np.arange(int((end - sta) / dt) + 1) * dt
    x1a = np.interp(t_align, t1, x1_d)
    y1a = np.interp(t_align, t1, y1_d)
    x2a = np.interp(t_align, t2c, x2_d)
    y2a = np.interp(t_align, t2c, y2_d)

    print("\n" + "=" * 60)
    print("  Step 4: 系统偏差估计")
    print("=" * 60)
    bias_cmp = compare_bias_methods(x2a, y2a, x1a, y1a)
    bias_x, bias_y, dx, dy, mask = iterative_bias_estimation(x2a, y2a, x1a, y1a)
    bias_x, bias_y = bias_cmp["median"]
    _, _, dx, dy = estimate_systematic_bias(x2a, y2a, x1a, y1a, method="median")

    print("\n" + "=" * 60)
    print("  Step 5: 偏差显著性检验")
    print("=" * 60)
    sig_x, p_x = bias_significance_test(dx)
    sig_y, p_y = bias_significance_test(dy)
    print(f"dx p={p_x:.4f}, dy p={p_y:.4f}")

    # ======================================================
    # Step 5.5: AR(1) 偏差漂移建模 ✅ 已插入
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
    R1_est, R2_est = estimate_adaptive_R(t1, x1_d, y1_d, dx, dy, bias_x=bias_x, bias_y=bias_y, method="mad")

    print("\n" + "=" * 60)
    print("  Step 7: EKF 融合")
    print("=" * 60)
    t2f = t2 - delay

    # --- 7a 默认 R ✅ 已加 AR1
    print("\n[默认 R] 融合中...")
    tgd, xfd, yfd, _, _ = fuse_sensors(
        t1, x1_d, y1_d, t2f, x2_d, y2_d,
        target_freq=time_config.target_freq,
        ar1_alpha=ar1_alpha, ar1_bias_var=ar1_bias_var
    )

    # --- 7b 自适应 R ✅ 已加 AR1
    print("[自适应 R] 融合中...")
    tga, xfa, yfa, bxa, bya = fuse_sensors(
        t1, x1_d, y1_d, t2f, x2_d, y2_d,
        target_freq=time_config.target_freq,
        R1_est=R1_est, R2_est=R2_est,
        ar1_alpha=ar1_alpha, ar1_bias_var=ar1_bias_var
    )

    cl = min(len(tgd), len(tga))
    xri = np.interp(tgd[:cl], t1, x1_d)
    yri = np.interp(tgd[:cl], t1, y1_d)
    rvdef = np.var(xfd[:cl] - xri) + np.var(yfd[:cl] - yri)
    rvadp = np.var(xfa[:cl] - xri) + np.var(yfa[:cl] - yri)
    print(f"默认 R: {rvdef:.6f}, 自适应 R: {rvadp:.6f}")

    if rvadp < rvdef:
        tg, xf, yf = tga, xfa, yfa
    else:
        tg, xf, yf = tgd, xfd, yfd
        _, _, _, bxa, bya = fuse_sensors(
            t1, x1_d, y1_d, t2f, x2_d, y2_d,
            target_freq=time_config.target_freq,
            ar1_alpha=ar1_alpha, ar1_bias_var=ar1_bias_var
        )

    df = pd.DataFrame({
        "Time(s)": np.round(tg, 4), "X(m)": np.round(xf, 6),
        "Y(m)": np.round(yf, 6), "bias_x(m)": np.round(bxa, 6), "bias_y(m)": np.round(bya, 6)
    })
    df.to_excel(output_dir / "Problem3_10Hz.xlsx", index=False, engine="openpyxl")

    plot_problem3_results(t1, x1, y1, t2, x2, y2, tg, xf, yf, bxa, bya, dx, dy, delay, output_dir)

    print("\n" + "=" * 60)
    print("  问题3 汇总")
    print("=" * 60)
    print(f"时间偏差: {delay:.6f}s")
    print(f"系统偏差: x={bias_x:.4f}, y={bias_y:.4f}")
    print("=" * 60)
    print("[Problem3] 完毕")