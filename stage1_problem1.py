# file: stage1_problem1.py


import matplotlib
matplotlib.use("Agg")
import config  # 触发 config.py 中的字体配置
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import data_path, time_config, alignment_config, plot_config, TABLE_DIR, PLOT_DIR, ensure_dirs
from core.time_alignment import align_sensors

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plot_config.apply_style()


_COLOR_S1 = "#2563EB"   # 方式1 — 蓝
_COLOR_S2 = "#DC2626"   # 方式2 — 红
_COLOR_FUSED = "#16A34A"  # 融合轨迹 — 绿


def load_problem1_data() -> tuple:
    file_path = data_path.path1

    if not file_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break

    print(f"[stage1] 加载文件：{file_path}")

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
        f"[stage1] 传感器1：{len(t1)} 个采样点，"
        f"[{t1[0]:.2f}, {t1[-1]:.2f}] s"
    )
    print(
        f"[stage1] 传感器2：{len(t2)} 个采样点，"
        f"[{t2[0]:.2f}, {t2[-1]:.2f}] s"
    )

    return t1, x1, y1, t2, x2, y2


def save_result(
    t_grid: np.ndarray,
    x_fused: np.ndarray,
    y_fused: np.ndarray,
    output_path: Path,
) -> None:
    df = pd.DataFrame({
        "Time(s)": np.round(t_grid, 4),
        "X(m)":    np.round(x_fused, 6),
        "Y(m)":    np.round(y_fused, 6),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    size_kb = output_path.stat().st_size / 1024
    print(f"[stage1] 已保存：{output_path}（{size_kb:.1f} KB）")


def plot_time_deviation(
    delays: np.ndarray,
    scores: np.ndarray,
    delay_fine: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(delays, scores, color=_COLOR_S1, linewidth=1.0, alpha=0.9)
    ax.axvline(delay_fine, color=_COLOR_S2, linewidth=1.5,
               linestyle="--", alpha=0.85,
               label=f"估计时偏 = {delay_fine:+.4f} s")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)

    best_idx = np.argmax(scores)
    ax.plot(delays[best_idx], scores[best_idx], "o",
            color=_COLOR_S2, markersize=8, markeredgecolor="white",
            markeredgewidth=1.0, zorder=5)

    ax.set_xlabel("候选时偏 (s)", fontsize=12)
    ax.set_ylabel("加权相关系数", fontsize=12)
    ax.set_title("问题1 — 时间偏差估计（互相关曲线）",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(delays[0], delays[-1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] 已保存：{output_path}")


def plot_comparison(
    t1: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, x2: np.ndarray, y2: np.ndarray,
    t_grid: np.ndarray, x_fused: np.ndarray, y_fused: np.ndarray,
    delay_fine: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.scatter(
        t1, x1, s=4, color=_COLOR_S1, alpha=0.6,
        label="传感器1 (4Hz)", marker="o",
    )
    ax1.scatter(
        t2, x2, s=4, color=_COLOR_S2, alpha=0.6,
        label="传感器2 (5Hz)", marker="x",
    )
    ax1.set_ylabel("X (m)", fontsize=12)
    ax1.set_title(
        "问题1 — 对齐前原始传感器数据",
        fontsize=14, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=10)

    ax2 = axes[1]
    ax2.plot(
        t_grid, x_fused,
        color=_COLOR_FUSED, linewidth=1.2, alpha=0.9,
        label=f"融合 10Hz (延迟={delay_fine:+.4f}s)",
    )
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("X (m)", fontsize=12)
    ax2.set_title(
        "问题1 — 对齐后融合轨迹",
        fontsize=14, fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] 已保存：{output_path}")


if __name__ == "__main__":
    t1, x1, y1, t2, x2, y2 = load_problem1_data()

    delay_fine, t_grid, x_fused, y_fused, delays, scores = align_sensors(
        t1, x1, y1,
        t2, x2, y2,
        target_freq=time_config.target_freq,
        delay_range=alignment_config.delay_range,
        method=alignment_config.method,
        w1=0.5, w2=0.5,
    )

    print("\n" + "=" * 50)
    print("  问题 1 结果")
    print("=" * 50)
    print(f"  估计时间偏差：{delay_fine:+.6f} s")
    print(f"  输出轨迹点数：{len(t_grid)}")
    print(f"  时间范围：[{t_grid[0]:.2f}, {t_grid[-1]:.2f}] s")
    print(f"  输出频率：{time_config.target_freq:.0f} Hz")
    print("=" * 50)

    output_xlsx = Path(TABLE_DIR) / "Problem1_10Hz.xlsx"
    save_result(t_grid, x_fused, y_fused, output_xlsx)

    output_td = Path(PLOT_DIR) / "Problem1_time_deviation.png"
    plot_time_deviation(delays, scores, delay_fine, output_td)

    output_fig = Path(PLOT_DIR) / "Problem1_trajectory.png"
    plot_comparison(
        t1, x1, y1,
        t2, x2, y2,
        t_grid, x_fused, y_fused,
        delay_fine,
        output_fig,
    )

    import pickle
    from config import INTERMEDIATE_DIR

    vx = np.gradient(x_fused, t_grid)
    vy = np.gradient(y_fused, t_grid)
    speed = np.sqrt(vx**2 + vy**2)

    result_p1 = {
        "t1": t1, "x1": x1, "y1": y1,
        "t2": t2, "x2": x2, "y2": y2,
        "t_fused": t_grid, "x_fused": x_fused, "y_fused": y_fused,
        "t_ref": t1, "x_ref": x1, "y_ref": y1,
        "error_x": x_fused - np.interp(t_grid, t1, x1),
        "error_y": y_fused - np.interp(t_grid, t1, y1),
        "t_error": t_grid,
        "speed": speed,
        "t_speed": t_grid,
        "delay": delay_fine,
        "cc_delays": delays,
        "cc_scores": scores,
        "t2_orig": t2,
    }

    pkl_path = Path(INTERMEDIATE_DIR) / "result_problem1.pkl"
    with open(pkl_path, "wb") as _f:
        pickle.dump(result_p1, _f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[stage1] 可视化数据已保存 → {pkl_path}")

    print("\n[stage1] 问题1求解完毕。")
