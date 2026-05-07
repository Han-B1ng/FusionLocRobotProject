# file: stage1_problem1.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 问题1求解：加载附件1 → 时间对齐 → 输出时间偏差与10Hz轨迹

"""
阶段 1 — 问题 1：无噪声时间对齐。

附件 1 的两类传感器数据无噪声影响，但存在设备开机先后导致的时间偏差。
本模块完成：
  1. 加载附件 1 的两个传感器 sheet
  2. 估计时间偏差（互相关 + 最小二乘）
  3. 对齐并融合为 10Hz 轨迹
  4. 输出 Excel + 对比图

依赖：core/time_alignment.py, config.py
后续：stage2 复用本模块的对齐框架处理含噪声数据
"""

import matplotlib
matplotlib.use("Agg")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import data_path, time_config, alignment_config
from core.time_alignment import align_sensors

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

# 传感器配色
_COLOR_S1 = "#2563EB"   # 方式1 — 蓝
_COLOR_S2 = "#DC2626"   # 方式2 — 红
_COLOR_FUSED = "#16A34A"  # 融合轨迹 — 绿


# ============================================================
#  数据加载
# ============================================================
def load_problem1_data() -> tuple:
    """加载附件 1 的两个传感器 sheet。

    Returns
    -------
    t1, x1, y1 : np.ndarray
        传感器 1 的时间戳 (s)、X/Y 坐标 (m)。
    t2, x2, y2 : np.ndarray
        传感器 2 的时间戳 (s)、X/Y 坐标 (m)。
    """
    file_path = data_path.path1

    # 兼容 .csv / .xlsx
    if not file_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = file_path.with_suffix(ext)
            if alt.exists():
                file_path = alt
                break

    print(f"[stage1] 加载文件: {file_path}")

    df1 = pd.read_excel(
        file_path, sheet_name="方式1(4Hz)", engine="openpyxl"
    )
    df2 = pd.read_excel(
        file_path, sheet_name="方式2(5Hz)", engine="openpyxl"
    )

    # 列名标准化
    col_map = {
        "时间(s)": "t", "时间": "t", "Time": "t", "time": "t", "t": "t",
        "X坐标(m)": "x", "X坐标": "x", "X": "x", "x": "x",
        "Y坐标(m)": "y", "Y坐标": "y", "Y": "y", "y": "y",
    }
    df1 = df1.rename(columns=col_map)[["t", "x", "y"]]
    df2 = df2.rename(columns=col_map)[["t", "x", "y"]]

    # 类型清洗
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
        f"[stage1] 传感器1: {len(t1)} 点, "
        f"[{t1[0]:.2f}, {t1[-1]:.2f}] s"
    )
    print(
        f"[stage1] 传感器2: {len(t2)} 点, "
        f"[{t2[0]:.2f}, {t2[-1]:.2f}] s"
    )

    return t1, x1, y1, t2, x2, y2


# ============================================================
#  结果保存
# ============================================================
def save_result(
    t_grid: np.ndarray,
    x_fused: np.ndarray,
    y_fused: np.ndarray,
    output_path: Path,
) -> None:
    """将 10Hz 融合轨迹保存为 Excel。

    列名：Time(s), X(m), Y(m)
    """
    df = pd.DataFrame({
        "Time(s)": np.round(t_grid, 4),
        "X(m)":    np.round(x_fused, 6),
        "Y(m)":    np.round(y_fused, 6),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    size_kb = output_path.stat().st_size / 1024
    print(f"[stage1] 已保存: {output_path}  ({size_kb:.1f} KB)")


# ============================================================
#  绘图：对齐前后对比
# ============================================================
def plot_comparison(
    t1: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, x2: np.ndarray, y2: np.ndarray,
    t_grid: np.ndarray, x_fused: np.ndarray, y_fused: np.ndarray,
    delay_fine: float,
    output_path: Path,
) -> None:
    """绘制对齐前后轨迹对比图。

    子图 1：原始两个传感器数据（散点）
    子图 2：融合后的 10Hz 轨迹（实线）
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --------------------------------------------------
    # 子图 1：原始传感器数据
    # --------------------------------------------------
    ax1 = axes[0]
    ax1.scatter(
        t1, x1, s=4, color=_COLOR_S1, alpha=0.6,
        label="Sensor1 (4Hz)", marker="o",
    )
    ax1.scatter(
        t2, x2, s=4, color=_COLOR_S2, alpha=0.6,
        label="Sensor2 (5Hz)", marker="x",
    )
    ax1.set_ylabel("X (m)", fontsize=12)
    ax1.set_title(
        "Problem 1 — Raw Sensor Data (Before Alignment)",
        fontsize=14, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=10)

    # --------------------------------------------------
    # 子图 2：融合后的 10Hz 轨迹
    # --------------------------------------------------
    ax2 = axes[1]
    ax2.plot(
        t_grid, x_fused,
        color=_COLOR_FUSED, linewidth=1.2, alpha=0.9,
        label=f"Fused 10Hz (delay={delay_fine:+.4f}s)",
    )
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("X (m)", fontsize=12)
    ax2.set_title(
        "Problem 1 — Fused Trajectory (After Alignment)",
        fontsize=14, fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] 已保存: {output_path}")


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    # ---- 加载数据 ----
    t1, x1, y1, t2, x2, y2 = load_problem1_data()

    # ---- 时间对齐 + 融合 ----
    #     问题 1 无噪声，等权重融合
    delay_fine, t_grid, x_fused, y_fused = align_sensors(
        t1, x1, y1,
        t2, x2, y2,
        target_freq=time_config.target_freq,
        delay_range=alignment_config.delay_range,
        method=alignment_config.method,
        w1=0.5, w2=0.5,
    )

    # ---- 打印结果 ----
    print("\n" + "=" * 50)
    print("  问题 1 结果")
    print("=" * 50)
    print(f"  估计时间偏差: {delay_fine:+.6f} s")
    print(f"  输出轨迹点数: {len(t_grid)}")
    print(f"  时间范围: [{t_grid[0]:.2f}, {t_grid[-1]:.2f}] s")
    print(f"  输出频率: {time_config.target_freq:.0f} Hz")
    print("=" * 50)

    # ---- 保存 Excel ----
    output_xlsx = data_path.output_dir / "Problem1_10Hz.xlsx"
    save_result(t_grid, x_fused, y_fused, output_xlsx)

    # ---- 绘图 ----
    output_fig = data_path.output_dir / "figures" / "Problem1_trajectory.png"
    plot_comparison(
        t1, x1, y1,
        t2, x2, y2,
        t_grid, x_fused, y_fused,
        delay_fine,
        output_fig,
    )

    print("\n[stage1] 问题 1 求解完毕。")

