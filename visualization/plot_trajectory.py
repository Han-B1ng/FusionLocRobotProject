# file: visualization/plot_trajectory.py
# @Description : 融合轨迹对比可视化模块
# 依赖：matplotlib, numpy, pathlib
# 上游：stage1~3 的融合结果（时间对齐后的轨迹、误差序列、速度序列）
# 下游：被 main.py 或各 stage 脚本调用

"""
visualization/plot_trajectory.py
=================================
提供三类轨迹可视化函数：
  1. plot_trajectory_comparison — 二维轨迹对比（x-y 平面）
  2. plot_error_time_series     — 融合误差随时间变化
  3. plot_velocity_profile      — 速度曲线 + 可选任务窗口标注
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  全局样式
# ============================================================
plt.rcParams.update({
    "font.sans-serif":    ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.0,
    "font.size":          10,
})

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

# ---- 论文风格配色 ----
_COLOR_S1     = "#2563EB"   # 传感器 1 — 蓝
_COLOR_S2     = "#DC2626"   # 传感器 2 — 红
_COLOR_FUSED  = "#059669"   # 融合轨迹 — 绿
_COLOR_REF    = "#6B7280"   # 参考 / 真值 — 灰
_COLOR_TASK   = "#F59E0B"   # 任务窗口 — 琥珀

_DPI = 150


# ============================================================
#  辅助：确保目录存在
# ============================================================
def _ensure_parent(path: Path) -> None:
    """创建 path 的父目录（若不存在）。"""
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
#  1. 二维轨迹对比图（x-y 平面）
# ============================================================
def plot_trajectory_comparison(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    t_fused: Optional[np.ndarray] = None,
    x_fused: Optional[np.ndarray] = None,
    y_fused: Optional[np.ndarray] = None,
    t_ref: Optional[np.ndarray] = None,
    x_ref: Optional[np.ndarray] = None,
    y_ref: Optional[np.ndarray] = None,
    save_path: Union[str, Path] = "output/figures/trajectory_comparison.png",
    title: str = "轨迹对比",
    mark_endpoints: bool = True,
) -> None:
    """将原始传感器轨迹与融合轨迹绘制在同一张 x-y 平面图上。

    Parameters
    ----------
    t1, x1, y1 : np.ndarray
        传感器 1 的时间、X、Y 坐标序列。
    t2, x2, y2 : np.ndarray
        传感器 2 的时间、X、Y 坐标序列。
    t_fused, x_fused, y_fused : np.ndarray, optional
        融合轨迹。若为 None 则仅绘制两路原始轨迹。
    t_ref, x_ref, y_ref : np.ndarray, optional
        参考 / 真值轨迹（问题 1 已知无噪声数据）。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    mark_endpoints : bool, default=True
        是否用圆点标注每条轨迹的起点和终点。

    Notes
    -----
    - 传感器 1 实线蓝、传感器 2 实线红、融合虚线绿、参考点线灰。
    - 起点用空心圆 ``○``，终点用实心圆 ``●``。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # ---- 绘制各轨迹 ----
    trajectories: List[Tuple[np.ndarray, np.ndarray, str, str, str]] = []
    # (x, y, color, linestyle, label)

    trajectories.append((x1, y1, _COLOR_S1, "-", "传感器 1 (4 Hz)"))
    trajectories.append((x2, y2, _COLOR_S2, "-", "传感器 2 (5 Hz)"))

    if x_fused is not None and y_fused is not None:
        trajectories.append(
            (x_fused, y_fused, _COLOR_FUSED, "--", "融合轨迹")
        )

    if x_ref is not None and y_ref is not None:
        trajectories.append(
            (x_ref, y_ref, _COLOR_REF, "-.", "参考轨迹")
        )

    for xi, yi, color, ls, label in trajectories:
        ax.plot(
            xi, yi,
            color=color, linestyle=ls, linewidth=1.0, alpha=0.85,
            label=label,
        )

    # ---- 标注起点 / 终点 ----
    if mark_endpoints:
        for xi, yi, color, _, label in trajectories:
            # 起点 — 空心圆
            ax.plot(
                xi[0], yi[0],
                marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5,
                linestyle="None", zorder=5,
            )
            # 终点 — 实心圆
            ax.plot(
                xi[-1], yi[-1],
                marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=1.5,
                linestyle="None", zorder=5,
            )

    # ---- 样式 ----
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_trajectory_comparison] 已保存: {save_path}")


# ============================================================
#  2. 融合误差随时间变化
# ============================================================
def plot_error_time_series(
    t: np.ndarray,
    error_x: np.ndarray,
    error_y: np.ndarray,
    save_path: Union[str, Path] = "output/figures/error_timeseries.png",
    title: str = "融合轨迹误差",
    error_ref_x: Optional[np.ndarray] = None,
    error_ref_y: Optional[np.ndarray] = None,
    ref_label: str = "参考误差",
) -> None:
    """绘制融合轨迹与参考轨迹在 X / Y 方向的误差随时间变化。

    Parameters
    ----------
    t : np.ndarray
        时间轴（秒）。
    error_x : np.ndarray
        X 方向误差序列（融合 − 参考），单位 m。
    error_y : np.ndarray
        Y 方向误差序列（融合 − 参考），单位 m。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    error_ref_x, error_ref_y : np.ndarray, optional
        第二组误差（如校正前 vs 校正后），用于对比。
    ref_label : str
        第二组误差的图例标签。

    Notes
    -----
    - 上子图为 X 误差，下子图为 Y 误差。
    - 零线用黑色虚线标注。
    - 右侧文本框显示 RMSE 和最大绝对误差。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, (ax_x, ax_y) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True
    )

    # ---- 主误差曲线 ----
    ax_x.plot(t, error_x, color=_COLOR_FUSED, linewidth=0.8, alpha=0.9,
              label="X 误差")
    ax_y.plot(t, error_y, color=_COLOR_FUSED, linewidth=0.8, alpha=0.9,
              label="Y 误差")

    # ---- 可选：第二组误差 ----
    if error_ref_x is not None:
        ax_x.plot(t[:len(error_ref_x)], error_ref_x,
                  color=_COLOR_REF, linewidth=0.8, alpha=0.7,
                  linestyle="--", label=ref_label)
    if error_ref_y is not None:
        ax_y.plot(t[:len(error_ref_y)], error_ref_y,
                  color=_COLOR_REF, linewidth=0.8, alpha=0.7,
                  linestyle="--", label=ref_label)

    # ---- 零线 ----
    ax_x.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_y.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)

    # ---- 统计信息 ----
    def _stats_text(err: np.ndarray) -> str:
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        max_abs = float(np.max(np.abs(err)))
        return (
            f"RMSE   = {rmse:.4f} m\n"
            f"MAE    = {mae:.4f} m\n"
            f"Max|e| = {max_abs:.4f} m"
        )

    ax_x.text(
        0.98, 0.95, _stats_text(error_x),
        transform=ax_x.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        family="monospace",
    )
    ax_y.text(
        0.98, 0.95, _stats_text(error_y),
        transform=ax_y.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        family="monospace",
    )

    # ---- 样式 ----
    ax_x.set_ylabel("X 误差 (m)", fontsize=11)
    ax_x.set_title(f"{title} — X 方向", fontsize=12, fontweight="bold")
    ax_x.legend(loc="upper left", fontsize=9, framealpha=0.9)

    ax_y.set_ylabel("Y 误差 (m)", fontsize=11)
    ax_y.set_xlabel("时间 (s)", fontsize=11)
    ax_y.set_title(f"{title} — Y 方向", fontsize=12, fontweight="bold")
    ax_y.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_error_time_series] 已保存: {save_path}")


# ============================================================
#  3. 速度曲线 + 可选任务窗口
# ============================================================
def plot_velocity_profile(
    t: np.ndarray,
    speed: np.ndarray,
    save_path: Union[str, Path] = "output/figures/velocity_profile.png",
    title: str = "融合轨迹速度曲线",
    task_windows: Optional[List[Dict[str, float]]] = None,
    speed_limit: Optional[float] = None,
) -> None:
    """绘制融合后轨迹的速度曲线，并可选标注任务执行窗口。

    Parameters
    ----------
    t : np.ndarray
        时间轴（秒）。
    speed : np.ndarray
        合成速率序列，单位 m/s，即 ``sqrt(vx² + vy²)``。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    task_windows : list of dict, optional
        任务执行区间列表，每个元素形如::

            {'t_start': 100.0, 't_end': 105.0, 'label': '目标A-射击'}

        在图上以半透明色块 + 标签标注。
    speed_limit : float, optional
        速度上限（如射击约束 2.0 m/s），以水平虚线标注。

    Notes
    -----
    - 主曲线为蓝色实线。
    - 任务窗口为琥珀色半透明矩形。
    - 速度限制为红色虚线。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    # ---- 主速度曲线 ----
    ax.plot(
        t, speed,
        color=_COLOR_S1, linewidth=0.9, alpha=0.9,
        label="合成速率",
    )

    # ---- 任务窗口标注 ----
    if task_windows:
        for idx, win in enumerate(task_windows):
            t_start = win.get("t_start", 0.0)
            t_end = win.get("t_end", 0.0)
            label = win.get("label", f"任务 {idx + 1}")

            ax.axvspan(
                t_start, t_end,
                color=_COLOR_TASK, alpha=0.15, linewidth=0,
            )
            # 窗口顶部标签（避免重叠：交替上下偏移）
            y_pos = 0.95 - (idx % 3) * 0.08
            ax.text(
                (t_start + t_end) / 2, y_pos,
                label,
                transform=ax.get_xaxis_transform(),
                fontsize=7, ha="center", va="top",
                color="#92400E", fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=_COLOR_TASK, alpha=0.25,
                ),
            )

    # ---- 速度限制线 ----
    if speed_limit is not None:
        ax.axhline(
            speed_limit, color=_COLOR_S2, linestyle="--", linewidth=1.0,
            alpha=0.7, label=f"速度限制 = {speed_limit:.1f} m/s",
        )

    # ---- 统计摘要 ----
    speed_mean = float(np.mean(speed))
    speed_max = float(np.max(speed))
    stats_text = (
        f"均值 = {speed_mean:.3f} m/s\n"
        f"峰值 = {speed_max:.3f} m/s"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        family="monospace",
    )

    # ---- 样式 ----
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_ylabel("速率 (m/s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_velocity_profile] 已保存: {save_path}")
