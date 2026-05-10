# file: visualization/plot_results.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

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
    "grid.alpha":         0.25,
    "grid.linewidth":     0.4,
    "font.size":          10,
})

_COLOR_TRAJ   = "#2563EB"   # 融合轨迹 — 蓝
_COLOR_SHOOT  = "#DC2626"   # 射击任务 — 红
_COLOR_PHOTO  = "#2563EB"   # 拍照任务 — 蓝
_COLOR_TARGET = "#6B7280"   # 目标点 — 灰
_COLOR_PREP   = "#E5E7EB"   # 准备时段 — 浅灰
_COLOR_BG     = "#FAFAFA"   # 背景

_DPI = 150


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_task_field(task: Any, *candidates: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        for name in candidates:
            if name in task:
                return task[name]
        return default
    for name in candidates:
        val = getattr(task, name, None)
        if val is not None:
            return val
    return default


def plot_tasks_on_trajectory(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    tasks: List[Any],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "tasks_on_trajectory.png"),
    t: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    title: str = "任务执行位置",
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    ax.plot(
        traj_x, traj_y,
        color=_COLOR_TRAJ, linewidth=0.8, alpha=0.6,
        label="融合轨迹",
    )

    if targets is not None:
        targets = np.asarray(targets)
        if targets.ndim == 2 and targets.shape[1] >= 3:
            tx, ty = targets[:, 1], targets[:, 2]
            tids = targets[:, 0]
        elif targets.ndim == 2 and targets.shape[1] == 2:
            tx, ty = targets[:, 0], targets[:, 1]
            tids = np.arange(1, len(tx) + 1)
        else:
            tx, ty, tids = np.array([]), np.array([]), np.array([])

        ax.scatter(
            tx, ty,
            marker="D", s=36, c=_COLOR_TARGET, alpha=0.5,
            edgecolors="white", linewidths=0.5,
            label="目标点", zorder=3,
        )
        for tid, xi, yi in zip(tids, tx, ty):
            ax.annotate(
                f"T{int(tid)}",
                (xi, yi), fontsize=6, color=_COLOR_TARGET,
                xytext=(5, 5), textcoords="offset points",
            )

    shoot_count = 0
    photo_count = 0

    for task in tasks:
        ttype = str(_extract_task_field(task, "task_type", "type", default="")).lower()
        tid = _extract_task_field(task, "target_id", "target", "id", default="?")
        tx_exec = _extract_task_field(task, "x", "pos_x", default=None)
        ty_exec = _extract_task_field(task, "y", "pos_y", default=None)
        t_exec = _extract_task_field(task, "t_exec", "t_execute", "time", default=None)

        if tx_exec is None and t is not None and t_exec is not None:
            idx = np.argmin(np.abs(t - t_exec))
            tx_exec = traj_x[idx]
            ty_exec = traj_y[idx]

        if tx_exec is None or ty_exec is None:
            continue

        is_shoot = "shoot" in ttype or "射击" in ttype
        is_photo = "photo" in ttype or "拍照" in ttype

        if is_shoot:
            ax.scatter(
                tx_exec, ty_exec,
                marker="^", s=100, c=_COLOR_SHOOT,
                edgecolors="white", linewidths=0.8,
                zorder=6,
            )
            ax.annotate(
                f"S-{tid}",
                (tx_exec, ty_exec), fontsize=7, fontweight="bold",
                color=_COLOR_SHOOT,
                xytext=(8, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
            )
            shoot_count += 1

        elif is_photo:
            ax.scatter(
                tx_exec, ty_exec,
                marker="o", s=80, c=_COLOR_PHOTO,
                edgecolors="white", linewidths=0.8,
                zorder=6,
            )
            ax.annotate(
                f"P-{tid}",
                (tx_exec, ty_exec), fontsize=7, fontweight="bold",
                color=_COLOR_PHOTO,
                xytext=(8, -12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
            )
            photo_count += 1

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")

    legend_handles = [
        plt.Line2D([0], [0], color=_COLOR_TRAJ, linewidth=0.8, alpha=0.6,
                    label="融合轨迹"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=_COLOR_SHOOT,
                    markersize=9, linestyle="None", label=f"射击 ({shoot_count})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLOR_PHOTO,
                    markersize=8, linestyle="None", label=f"拍照 ({photo_count})"),
    ]
    if targets is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="D", color="w",
                        markerfacecolor=_COLOR_TARGET, markersize=7,
                        linestyle="None", label="目标点")
        )
    ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_tasks_on_trajectory] 已保存: {save_path}")


def plot_task_gantt(
    tasks: List[Any],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "task_gantt.png"),
    title: str = "任务调度甘特图",
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if not tasks:
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "无任务数据", transform=ax.transAxes,
                fontsize=14, ha="center", va="center", color="#9CA3AF")
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_task_gantt] 已保存（空）: {save_path}")
        return

    parsed: List[Dict[str, Any]] = []
    for task in tasks:
        ttype = str(_extract_task_field(task, "task_type", "type", default="")).lower()
        tid = _extract_task_field(task, "target_id", "target", "id", default="?")
        t_prep_start = _extract_task_field(task, "t_prep_start", default=None)
        t_exec_start = _extract_task_field(task, "t_exec_start", "t_start", default=None)
        t_exec_end = _extract_task_field(task, "t_exec_end", "t_end", default=None)
        prep_dur = _extract_task_field(task, "prep_duration", "prep_time", default=0.0)

        if t_exec_start is None:
            continue

        if t_prep_start is None:
            t_prep_start = t_exec_start - float(prep_dur)

        is_shoot = "shoot" in ttype or "射击" in ttype
        label = f"{'射击' if is_shoot else '拍照'}-{tid}"

        parsed.append({
            "label":        label,
            "t_prep_start": float(t_prep_start),
            "t_exec_start": float(t_exec_start),
            "t_exec_end":   float(t_exec_end) if t_exec_end is not None else float(t_exec_start),
            "is_shoot":     is_shoot,
        })

    parsed.sort(key=lambda d: d["t_exec_start"])

    n_tasks = len(parsed)

    fig, ax = plt.subplots(
        figsize=(14, max(3, 0.5 * n_tasks + 2)),
        constrained_layout=True,
    )

    bar_height = 0.6

    for i, p in enumerate(parsed):
        y = n_tasks - 1 - i  # 从上到下

        color_exec = _COLOR_SHOOT if p["is_shoot"] else _COLOR_PHOTO

        prep_width = p["t_exec_start"] - p["t_prep_start"]
        if prep_width > 0:
            ax.barh(
                y, prep_width, left=p["t_prep_start"],
                height=bar_height, color=_COLOR_PREP,
                edgecolor="#D1D5DB", linewidth=0.5,
            )

        exec_width = p["t_exec_end"] - p["t_exec_start"]
        if exec_width <= 0:
            exec_width = 0.3  # 最小宽度保证可见
        ax.barh(
            y, exec_width, left=p["t_exec_start"],
            height=bar_height, color=color_exec, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        ax.text(
            p["t_exec_start"] + exec_width / 2, y,
            f'{p["t_exec_start"]:.1f}s',
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold",
        )

    y_labels = [p["label"] for p in reversed(parsed)]
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_PREP, edgecolor="#D1D5DB",
                  label="准备时段"),
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_SHOOT, alpha=0.85,
                  label="射击执行"),
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_PHOTO, alpha=0.85,
                  label="拍照执行"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_task_gantt] 已保存: {save_path}")


def plot_heading_diversity(
    target_id: Union[int, str],
    headings: np.ndarray,
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "heading_diversity.png"),
    title: Optional[str] = None,
    min_angle_diff: float = 60.0,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    headings = np.asarray(headings, dtype=float)
    headings = headings % 360.0
    n = len(headings)

    if title is None:
        title = f"目标 {target_id} — 拍照航向角多样性"

    sorted_idx = np.argsort(headings)
    sorted_headings = headings[sorted_idx]
    diffs = np.diff(sorted_headings)
    wrap_diff = 360.0 - sorted_headings[-1] + sorted_headings[0]
    all_diffs = np.append(diffs, wrap_diff)
    min_diff_val = float(np.min(all_diffs))
    is_violated = min_diff_val < min_angle_diff

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)

    ax_polar = fig.add_subplot(121, projection="polar")

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.6 * i / max(n - 1, 1)) for i in range(n)]

    for i, (h, c) in enumerate(zip(headings, colors)):
        theta = np.deg2rad(h)
        ax_polar.plot(
            [theta, theta], [0, 1.0],
            color=c, linewidth=2.5, alpha=0.9,
        )
        ax_polar.plot(
            theta, 1.0,
            marker="o", markersize=8, color=c,
            markeredgecolor="white", markeredgewidth=0.8,
        )
        ax_polar.annotate(
            f"#{i + 1}\n{h:.1f}°",
            (theta, 1.15), fontsize=7, ha="center", va="center",
            fontweight="bold",
        )

    if n >= 2 and is_violated:
        min_idx = int(np.argmin(all_diffs))
        if min_idx < n - 1:
            h1 = sorted_headings[min_idx]
            h2 = sorted_headings[min_idx + 1]
        else:
            h1 = sorted_headings[-1]
            h2 = sorted_headings[0]

        theta1 = np.deg2rad(h1)
        theta2 = np.deg2rad(h2)
        arc_thetas = np.linspace(theta1, theta2, 50)
        ax_polar.plot(
            arc_thetas, np.full_like(arc_thetas, 0.5),
            color=_COLOR_SHOOT, linewidth=3, alpha=0.6,
        )
        ax_polar.annotate(
            f"Δ={min_diff_val:.1f}°\n<{min_angle_diff:.0f}°",
            ((theta1 + theta2) / 2, 0.5),
            fontsize=7, color=_COLOR_SHOOT, fontweight="bold",
            ha="center", va="bottom",
        )

    ax_polar.set_ylim(0, 1.4)
    ax_polar.set_rticks([])
    ax_polar.set_title("航向角分布", fontsize=11, fontweight="bold", pad=15)

    ax_text = fig.add_subplot(122)
    ax_text.axis("off")

    summary_lines = [
        f"目标 ID:  {target_id}",
        f"拍摄次数: {n}",
        "",
        "各次航向角:",
    ]
    for i, h in enumerate(headings):
        summary_lines.append(f"  第 {i + 1} 次:  {h:7.2f}°")

    summary_lines.append("")
    summary_lines.append("相邻角度差（升序）:")
    for i, idx in enumerate(sorted_idx[:-1]):
        d = diffs[i]
        flag = " ⚠" if d < min_angle_diff else ""
        summary_lines.append(
            f"  {sorted_headings[i]:.1f}° → {sorted_headings[i + 1]:.1f}°  "
            f"Δ = {d:.2f}°{flag}"
        )
    flag_wrap = " ⚠" if wrap_diff < min_angle_diff else ""
    summary_lines.append(
        f"  {sorted_headings[-1]:.1f}° → {sorted_headings[0]:.1f}°  "
        f"Δ = {wrap_diff:.2f}°（环绕）{flag_wrap}"
    )

    summary_lines.append("")
    summary_lines.append(f"最小角度差: {min_diff_val:.2f}°")
    summary_lines.append(f"约束要求:   ≥ {min_angle_diff:.0f}°")

    if is_violated:
        summary_lines.append("")
        summary_lines.append("⚠ 存在违反角度差异约束的拍摄对！")
    else:
        summary_lines.append("")
        summary_lines.append("✓ 所有拍摄对满足角度差异约束。")

    text_content = "\n".join(summary_lines)
    ax_text.text(
        0.05, 0.95, text_content,
        transform=ax_text.transAxes, fontsize=9,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", alpha=0.9),
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_heading_diversity] 已保存: {save_path}")


def plot_task_gantt_enhanced(
    tasks: List[Any],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "task_gantt_enhanced.png"),
    title: str = "任务调度甘特图（增强版）",
    candidate_windows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    from matplotlib.lines import Line2D

    save_path = Path(save_path)
    _ensure_parent(save_path)

    if not tasks:
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.text(
            0.5, 0.5, "无任务数据", transform=ax.transAxes,
            fontsize=14, ha="center", va="center", color="#9CA3AF",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_task_gantt_enhanced] 已保存（空）: {save_path}")
        return

    parsed: List[Dict[str, Any]] = []
    for task in tasks:
        ttype = str(
            _extract_task_field(task, "task_type", "type", default="")
        ).lower()
        tid = _extract_task_field(task, "target_id", "target", "id", default="?")
        t_prep_start = _extract_task_field(
            task, "t_prep_start", "t_start_prep", default=None,
        )
        t_exec_start = _extract_task_field(
            task, "t_exec_start", "t_exec", "t_execute", default=None,
        )
        t_exec_end = _extract_task_field(
            task, "t_exec_end", "t_end", default=None,
        )
        prep_dur = _extract_task_field(
            task, "prep_duration", "prep_time", default=0.0,
        )

        if t_exec_start is None:
            continue

        if t_prep_start is None:
            t_prep_start = t_exec_start - float(prep_dur)

        is_shoot = "shoot" in ttype or "射击" in ttype
        label = f"{'射击' if is_shoot else '拍照'}-{tid}"

        parsed.append({
            "label":        label,
            "t_prep_start": float(t_prep_start),
            "t_exec_start": float(t_exec_start),
            "t_exec_end":   (float(t_exec_end)
                             if t_exec_end is not None
                             else float(t_exec_start)),
            "is_shoot":     is_shoot,
            "tid":          tid,
        })

    parsed.sort(key=lambda d: d["t_exec_start"])
    n_tasks = len(parsed)

    fig, ax = plt.subplots(
        figsize=(14, max(3, 0.5 * n_tasks + 2)),
        constrained_layout=True,
    )

    bar_height = 0.6

    if candidate_windows:
        scheduled_set = set()
        for p in parsed:
            scheduled_set.add((str(p["tid"]), p["t_exec_start"]))

        all_tids = list(
            dict.fromkeys(p["tid"] for p in parsed)
        )
        tid_to_y = {
            tid: n_tasks - 1 - i for i, tid in enumerate(all_tids)
        }

        for cw in candidate_windows:
            cw_tid = _extract_task_field(
                cw, "target_id", "target", "id", default=None,
            )
            cw_texec = _extract_task_field(
                cw, "t_exec", "t_exec_start", "t_execute", default=None,
            )
            cw_tstart = _extract_task_field(
                cw, "t_start", "t_start_prep", default=None,
            )
            cw_type = str(
                _extract_task_field(cw, "task_type", "type", default="")
            ).lower()

            if cw_tid is None or cw_texec is None:
                continue

            cw_tid_str = str(cw_tid)
            if (cw_tid_str, float(cw_texec)) in scheduled_set:
                continue
            if cw_tid_str not in tid_to_y:
                continue

            y = tid_to_y[cw_tid_str]
            t_s = (float(cw_tstart)
                   if cw_tstart is not None
                   else float(cw_texec) - 1.5)
            t_e = float(cw_texec) + 0.3

            is_shoot_cw = "shoot" in cw_type or "射击" in cw_type
            color_dash = "#FCA5A5" if is_shoot_cw else "#93C5FD"

            ax.barh(
                y, t_e - t_s, left=t_s,
                height=bar_height * 0.4,
                color="none", edgecolor=color_dash,
                linewidth=0.8, linestyle="--", alpha=0.6,
            )

    for i, p in enumerate(parsed):
        y = n_tasks - 1 - i
        color_exec = _COLOR_SHOOT if p["is_shoot"] else _COLOR_PHOTO

        prep_width = p["t_exec_start"] - p["t_prep_start"]
        if prep_width > 0:
            ax.barh(
                y, prep_width, left=p["t_prep_start"],
                height=bar_height, color=_COLOR_PREP,
                edgecolor="#D1D5DB", linewidth=0.5,
            )

        exec_width = p["t_exec_end"] - p["t_exec_start"]
        if exec_width <= 0:
            exec_width = 0.3
        ax.barh(
            y, exec_width, left=p["t_exec_start"],
            height=bar_height, color=color_exec, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        marker = "^" if p["is_shoot"] else "o"
        marker_color = _COLOR_SHOOT if p["is_shoot"] else _COLOR_PHOTO
        ax.plot(
            p["t_exec_start"] + exec_width / 2, y,
            marker=marker, markersize=8,
            markerfacecolor=marker_color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=6,
        )

        ax.text(
            p["t_exec_start"] + exec_width / 2, y,
            f'{p["t_exec_start"]:.1f}s',
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold",
        )

    y_labels = [p["label"] for p in reversed(parsed)]
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_PREP,
                  edgecolor="#D1D5DB", label="准备时段"),
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_SHOOT,
                  alpha=0.85, label="射击执行"),
        Rectangle((0, 0), 1, 1, facecolor=_COLOR_PHOTO,
                  alpha=0.85, label="拍照执行"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=_COLOR_SHOOT, markersize=8,
               linestyle="None", label="射击标记 ▲"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_COLOR_PHOTO, markersize=8,
               linestyle="None", label="拍照标记 ●"),
    ]
    if candidate_windows:
        legend_handles.append(
            Line2D([0], [0], color="#FCA5A5", linewidth=1,
                   linestyle="--", label="候选窗口（未选中）"),
        )
    ax.legend(
        handles=legend_handles, loc="lower right",
        fontsize=9, framealpha=0.9,
    )

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_task_gantt_enhanced] 已保存: {save_path}")
