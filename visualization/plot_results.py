# file: visualization/plot_results.py
# @Description : 问题4 任务规划结果可视化模块
# 依赖：matplotlib, numpy, pathlib
# 上游：stage4_problem4.py 产出的任务调度方案、融合轨迹、目标点坐标
# 下游：被 main.py 或 stage4 脚本调用

"""
visualization/plot_results.py
==============================
提供三类任务规划结果可视化函数：
  1. plot_tasks_on_trajectory  — 二维轨迹上叠加任务执行标记
  2. plot_task_gantt           — 任务调度甘特图
  3. plot_heading_diversity    — 拍照航向角多样性圆图 / 玫瑰图
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# ============================================================
#  全局样式
# ============================================================
plt.rcParams.update({
    "font.sans-serif":    ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.4,
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
_COLOR_TRAJ   = "#2563EB"   # 融合轨迹 — 蓝
_COLOR_SHOOT  = "#DC2626"   # 射击任务 — 红
_COLOR_PHOTO  = "#2563EB"   # 拍照任务 — 蓝
_COLOR_TARGET = "#6B7280"   # 目标点 — 灰
_COLOR_PREP   = "#E5E7EB"   # 准备时段 — 浅灰
_COLOR_BG     = "#FAFAFA"   # 背景

_DPI = 150


# ============================================================
#  辅助
# ============================================================
def _ensure_parent(path: Path) -> None:
    """创建保存路径的父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_task_field(task: Any, *candidates: str, default: Any = None) -> Any:
    """从 dict 或对象中按候选字段名依次取值。

    兼容 dict、dataclass、命名元组等多种任务表示。
    """
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


# ============================================================
#  1. 轨迹上叠加任务执行位置
# ============================================================
def plot_tasks_on_trajectory(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    tasks: List[Any],
    save_path: Union[str, Path] = "output/figures/tasks_on_trajectory.png",
    t: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    title: str = "任务执行位置",
) -> None:
    """在二维轨迹上标记射击和拍照任务的执行位置。

    Parameters
    ----------
    traj_x, traj_y : np.ndarray
        融合轨迹的 X / Y 坐标序列（米）。
    tasks : list
        任务列表，每个元素为 dict 或对象，需包含以下字段::

            task_type  : str   — 'shoot' 或 'photo'
            t_exec     : float — 任务执行时刻（秒）
            x, y       : float — 执行位置坐标（米）
            target_id  : int/str — 目标编号
            t_start    : float — 任务窗口起始（可选，用于时间条）
            t_end      : float — 任务窗口结束（可选）

    save_path : str 或 Path
        图片保存路径。
    t : np.ndarray, optional
        轨迹时间轴。若提供则用于从轨迹中插值定位执行点
        （当 task 中无 x/y 字段时的后备方案）。
    targets : np.ndarray, optional
        目标点坐标，shape ``(N, 2)`` 或 ``(N, 3)`` 含 [id, x, y]。
        若提供则以灰色菱形标注所有目标点。
    title : str
        图表标题。

    Notes
    -----
    - 射击：红色三角形 ``▲`` + 红色标签。
    - 拍照：蓝色圆形 ``●`` + 蓝色标签。
    - 每个标记旁以小字标注目标编号。
    - 右下角附图例和任务数量统计。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # ---- 融合轨迹 ----
    ax.plot(
        traj_x, traj_y,
        color=_COLOR_TRAJ, linewidth=0.8, alpha=0.6,
        label="融合轨迹",
    )

    # ---- 目标点（可选）----
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

    # ---- 任务标记 ----
    shoot_count = 0
    photo_count = 0

    for task in tasks:
        ttype = str(_extract_task_field(task, "task_type", "type", default="")).lower()
        tid = _extract_task_field(task, "target_id", "target", "id", default="?")
        tx_exec = _extract_task_field(task, "x", "pos_x", default=None)
        ty_exec = _extract_task_field(task, "y", "pos_y", default=None)
        t_exec = _extract_task_field(task, "t_exec", "t_execute", "time", default=None)

        # 若无坐标，尝试从轨迹时间轴插值
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

    # ---- 样式 ----
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")

    # 手工图例补充
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


# ============================================================
#  2. 任务甘特图
# ============================================================
def plot_task_gantt(
    tasks: List[Any],
    save_path: Union[str, Path] = "output/figures/task_gantt.png",
    title: str = "任务调度甘特图",
) -> None:
    """绘制任务调度甘特图：横轴时间、纵轴任务，区分准备与执行时段。

    Parameters
    ----------
    tasks : list
        任务列表，每个元素为 dict 或对象，需包含::

            task_type  : str   — 'shoot' / 'photo'
            target_id  : int/str — 目标编号
            t_prep_start : float — 准备开始时刻（秒）
            t_exec_start : float — 执行开始时刻（秒）
            t_exec_end   : float — 执行结束时刻（秒）

        若无 t_prep_start，则用 prep_duration 倒推。

    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。

    Notes
    -----
    - 每行一个任务，从上到下按执行时间排序。
    - 浅灰色条为准备时段，彩色条为执行时段。
    - 射击红色、拍照蓝色。
    - 横轴标注绝对时间（秒）。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if not tasks:
        # 空任务：生成占位图
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "无任务数据", transform=ax.transAxes,
                fontsize=14, ha="center", va="center", color="#9CA3AF")
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_task_gantt] 已保存（空）: {save_path}")
        return

    # ---- 提取任务信息并按执行时间排序 ----
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

        # 倒推准备开始时刻
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

    # 按执行开始时间排序
    parsed.sort(key=lambda d: d["t_exec_start"])

    n_tasks = len(parsed)

    # ---- 绘图 ----
    fig, ax = plt.subplots(
        figsize=(14, max(3, 0.5 * n_tasks + 2)),
        constrained_layout=True,
    )

    bar_height = 0.6

    for i, p in enumerate(parsed):
        y = n_tasks - 1 - i  # 从上到下

        color_exec = _COLOR_SHOOT if p["is_shoot"] else _COLOR_PHOTO

        # 准备时段 — 浅灰
        prep_width = p["t_exec_start"] - p["t_prep_start"]
        if prep_width > 0:
            ax.barh(
                y, prep_width, left=p["t_prep_start"],
                height=bar_height, color=_COLOR_PREP,
                edgecolor="#D1D5DB", linewidth=0.5,
            )

        # 执行时段 — 彩色
        exec_width = p["t_exec_end"] - p["t_exec_start"]
        if exec_width <= 0:
            exec_width = 0.3  # 最小宽度保证可见
        ax.barh(
            y, exec_width, left=p["t_exec_start"],
            height=bar_height, color=color_exec, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        # 标注时间
        ax.text(
            p["t_exec_start"] + exec_width / 2, y,
            f'{p["t_exec_start"]:.1f}s',
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold",
        )

    # ---- 样式 ----
    y_labels = [p["label"] for p in reversed(parsed)]
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # 图例
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


# ============================================================
#  3. 拍照航向角多样性图（圆图 / 玫瑰图）
# ============================================================
def plot_heading_diversity(
    target_id: Union[int, str],
    headings: np.ndarray,
    save_path: Union[str, Path] = "output/figures/heading_diversity.png",
    title: Optional[str] = None,
    min_angle_diff: float = 60.0,
) -> None:
    """对给定拍照目标，绘制不同次拍摄的航向角圆图，展示角度多样性。

    Parameters
    ----------
    target_id : int 或 str
        目标编号，用于标题和文件命名。
    headings : np.ndarray
        各次拍摄的航向角序列（**度**，0~360 或 -180~180 均可）。
    save_path : str 或 Path
        图片保存路径。
    title : str, optional
        自定义标题。若为 None 则自动生成。
    min_angle_diff : float, default=60.0
        拍照最小航向角差异约束（度），在图上以弧段标注。

    Notes
    -----
    - 左侧：极坐标圆图，每个方向一条射线，颜色深浅区分拍摄次序。
    - 右侧：文本摘要，列出各次航向角及相邻角度差。
    - 若任意两次拍摄角度差 < min_angle_diff，以红色标注违反约束。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    headings = np.asarray(headings, dtype=float)
    # 统一映射到 [0, 360)
    headings = headings % 360.0
    n = len(headings)

    if title is None:
        title = f"目标 {target_id} — 拍照航向角多样性"

    # ---- 排序用于角度差计算 ----
    sorted_idx = np.argsort(headings)
    sorted_headings = headings[sorted_idx]
    # 相邻角度差（含首尾环绕差）
    diffs = np.diff(sorted_headings)
    wrap_diff = 360.0 - sorted_headings[-1] + sorted_headings[0]
    all_diffs = np.append(diffs, wrap_diff)
    min_diff_val = float(np.min(all_diffs))
    is_violated = min_diff_val < min_angle_diff

    # ---- 绘图：双面板 ----
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)

    # ---- 左：极坐标圆图 ----
    ax_polar = fig.add_subplot(121, projection="polar")

    # 极坐标方向：0° 在上方，顺时针
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    # 颜色映射：按拍摄次序渐变
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

    # 最小角度差异弧段标注
    if n >= 2 and is_violated:
        # 找到违反约束的最小差对
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

    # ---- 右：文本摘要 ----
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
    # 首尾环绕差
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
