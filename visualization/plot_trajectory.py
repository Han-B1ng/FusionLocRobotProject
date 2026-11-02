# file: visualization/plot_trajectory.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import plot_config, PLOT_DIR

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plot_config.apply_style()

plt.rcParams.update({
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.0,
    "font.size":          10,
})

_COLOR_S1     = "#2563EB"   # 传感器 1 — 蓝
_COLOR_S2     = "#DC2626"   # 传感器 2 — 红
_COLOR_FUSED  = "#059669"   # 融合轨迹 — 绿
_COLOR_REF    = "#6B7280"   # 参考 / 真值 — 灰
_COLOR_TASK   = "#F59E0B"   # 任务窗口 — 琥珀

_DPI = 150


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "trajectory_comparison.png"),
    title: str = "轨迹对比",
    mark_endpoints: bool = True,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    trajectories: List[Tuple[np.ndarray, np.ndarray, str, str, str]] = []

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

    if mark_endpoints:
        for xi, yi, color, _, label in trajectories:
            ax.plot(
                xi[0], yi[0],
                marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5,
                linestyle="None", zorder=5,
            )
            ax.plot(
                xi[-1], yi[-1],
                marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=1.5,
                linestyle="None", zorder=5,
            )

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_trajectory_comparison] 已保存: {save_path}")


def plot_error_time_series(
    t: np.ndarray,
    error_x: np.ndarray,
    error_y: np.ndarray,
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "error_timeseries.png"),
    title: str = "融合轨迹误差",
    error_ref_x: Optional[np.ndarray] = None,
    error_ref_y: Optional[np.ndarray] = None,
    ref_label: str = "参考误差",
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, (ax_x, ax_y) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True
    )

    ax_x.plot(t, error_x, color=_COLOR_FUSED, linewidth=0.8, alpha=0.9,
              label="X 误差")
    ax_y.plot(t, error_y, color=_COLOR_FUSED, linewidth=0.8, alpha=0.9,
              label="Y 误差")

    if error_ref_x is not None:
        ax_x.plot(t[:len(error_ref_x)], error_ref_x,
                  color=_COLOR_REF, linewidth=0.8, alpha=0.7,
                  linestyle="--", label=ref_label)
    if error_ref_y is not None:
        ax_y.plot(t[:len(error_ref_y)], error_ref_y,
                  color=_COLOR_REF, linewidth=0.8, alpha=0.7,
                  linestyle="--", label=ref_label)

    ax_x.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_y.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)

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


def plot_velocity_profile(
    t: np.ndarray,
    speed: np.ndarray,
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "velocity_profile.png"),
    title: str = "融合轨迹速度曲线",
    task_windows: Optional[List[Dict[str, float]]] = None,
    speed_limit: Optional[float] = None,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    ax.plot(
        t, speed,
        color=_COLOR_S1, linewidth=0.9, alpha=0.9,
        label="合成速率",
    )

    if task_windows:
        for idx, win in enumerate(task_windows):
            t_start = win.get("t_start", 0.0)
            t_end = win.get("t_end", 0.0)
            label = win.get("label", f"任务 {idx + 1}")

            ax.axvspan(
                t_start, t_end,
                color=_COLOR_TASK, alpha=0.15, linewidth=0,
            )
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

    if speed_limit is not None:
        ax.axhline(
            speed_limit, color=_COLOR_S2, linestyle="--", linewidth=1.0,
            alpha=0.7, label=f"速度限制 = {speed_limit:.1f} m/s",
        )

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

    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_ylabel("速率 (m/s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_velocity_profile] 已保存: {save_path}")


def plot_velocity_heatmap_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "velocity_heatmap.png"),
    title: str = "速度热力轨迹",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "RdYlBu_r",
) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    save_path = Path(save_path)
    _ensure_parent(save_path)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    speed_seg = 0.5 * (speed[:-1] + speed[1:])

    if vmin is None:
        vmin = float(np.percentile(speed_seg, 2))
    if vmax is None:
        vmax = float(np.percentile(speed_seg, 98))

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    lc = LineCollection(
        segments, cmap=cmap, norm=norm,
        linewidths=plot_config.linewidth, alpha=0.9,
    )
    lc.set_array(speed_seg)
    ax.add_collection(lc)

    ax.plot(
        x[0], y[0],
        marker="o", markersize=10,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.5, zorder=5, label="起点",
    )
    ax.plot(
        x[-1], y[-1],
        marker="o", markersize=10,
        markerfacecolor="black", markeredgecolor="black",
        markeredgewidth=1.5, zorder=5, label="终点",
    )

    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 2, y.max() + 2)
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(lc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("速率 (m/s)", fontsize=plot_config.label_fontsize)
    cbar.ax.tick_params(labelsize=plot_config.tick_fontsize)

    ax.set_xlabel("X (m)", fontsize=plot_config.label_fontsize)
    ax.set_ylabel("Y (m)", fontsize=plot_config.label_fontsize)
    ax.set_title(title, fontsize=plot_config.title_fontsize, fontweight="bold")
    ax.legend(
        loc="best", fontsize=plot_config.legend_fontsize,
        frameon=plot_config.legend_frameon,
    )

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_velocity_heatmap_trajectory] 已保存: {save_path}")
