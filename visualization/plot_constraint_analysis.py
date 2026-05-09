# file: visualization/plot_constraint_analysis.py
# @Description : 约束分析可视化模块（漏斗图 / 桑基图）
# 依赖：matplotlib, numpy, pathlib
# 上游：stage4_problem4.py 产出的约束统计数据（constraint_stats.xlsx）
# 下游：被 main.py 或 stage4 脚本调用

"""
visualization/plot_constraint_analysis.py
==========================================
提供约束淘汰过程的可视化函数：
  1. plot_constraint_funnel  — 约束漏斗图（各阶段候选数递减）
  2. plot_constraint_bar     — 约束淘汰比例水平条形图
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from config import plot_config, PLOT_DIR
# 应用中文字体配置
plot_config.apply_style()


# ============================================================
#  辅助
# ============================================================
def _ensure_parent(path: Path) -> None:
    """创建保存路径的父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
#  1. 约束漏斗图
# ============================================================
def plot_constraint_funnel(
    stage_labels: List[str],
    stage_counts: List[int],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "constraint_funnel.png"),
    title: str = "约束漏斗图 — 候选窗口筛选过程",
    color_start: str = "#0072B2",
    color_end: str = "#D55E00",
    show_pct: bool = True,
    show_abs: bool = True,
) -> None:
    """绘制约束漏斗图，展示各约束阶段对候选窗口的逐层淘汰。

    Parameters
    ----------
    stage_labels : list[str]
        各阶段名称，从上到下排列。示例::

            ["全部候选窗口",
             "时间冲突剔除",
             "速度约束过滤",
             "距离约束过滤",
             "角度约束过滤",
             "最终调度结果"]

    stage_counts : list[int]
        各阶段剩余窗口数量，须与 stage_labels 等长且单调递减。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    color_start : str
        漏斗顶部颜色（浅色）。
    color_end : str
        漏斗底部颜色（深色）。
    show_pct : bool
        是否在标签旁显示相对第一阶段的百分比。
    show_abs : bool
        是否在标签旁显示绝对数量。

    Notes
    -----
    - 漏斗由梯形色块构成，宽度与该阶段数量成正比。
    - 每层右侧标注阶段名、数量、占比。
    - 左侧标注淘汰数和淘汰率（非第一阶段）。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    n_stages = len(stage_labels)
    if n_stages < 2:
        raise ValueError("至少需要 2 个阶段才能绘制漏斗图")
    if len(stage_counts) != n_stages:
        raise ValueError("stage_labels 与 stage_counts 长度必须相同")

    # ---- 颜色渐变 ----
    c_start = np.array(plt.cm.colors.to_rgb(color_start))
    c_end = np.array(plt.cm.colors.to_rgb(color_end))
    colors = [
        tuple(c_start + (c_end - c_start) * i / (n_stages - 1))
        for i in range(n_stages)
    ]

    max_count = max(stage_counts)
    initial_count = stage_counts[0] if stage_counts[0] > 0 else 1

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(12, max(5, n_stages * 1.0)),
                           constrained_layout=True)

    bar_height = 0.7
    y_positions = list(range(n_stages - 1, -1, -1))  # 从上到下

    for i in range(n_stages):
        y = y_positions[i]
        count = stage_counts[i]
        width = count / max_count  # 相对宽度

        # 梯形：上底 = 当前宽度，下底 = 下一层宽度
        if i < n_stages - 1:
            next_width = stage_counts[i + 1] / max_count
        else:
            next_width = width * 0.8  # 最底层稍收窄

        # 用 fill_betweenx 绘制梯形
        x_left_top = -width / 2
        x_right_top = width / 2
        x_left_bot = -next_width / 2
        x_right_bot = next_width / 2

        xs = [x_left_top, x_right_top, x_right_bot, x_left_bot]
        ys = [y + bar_height / 2, y + bar_height / 2,
              y - bar_height / 2, y - bar_height / 2]

        ax.fill(xs, ys, color=colors[i], alpha=0.85, edgecolor="white",
                linewidth=1.5)

        # ---- 右侧标签 ----
        label_parts = [stage_labels[i]]
        info_parts = []
        if show_abs:
            info_parts.append(f"{count}")
        if show_pct:
            pct = 100.0 * count / initial_count
            info_parts.append(f"{pct:.1f}%")
        info_str = "  ".join(info_parts)

        ax.text(
            width / 2 + 0.03, y,
            f"{label_parts[0]}    {info_str}",
            ha="left", va="center",
            fontsize=plot_config.tick_fontsize,
            fontweight="bold",
        )

        # ---- 左侧淘汰标注（非第一阶段）----
        if i > 0:
            removed = stage_counts[i - 1] - stage_counts[i]
            if stage_counts[i - 1] > 0:
                removal_pct = 100.0 * removed / stage_counts[i - 1]
            else:
                removal_pct = 0.0
            ax.text(
                -width / 2 - 0.03, y,
                f"-{removed} ({removal_pct:.1f}%)",
                ha="right", va="center",
                fontsize=plot_config.tick_fontsize - 1,
                color=plot_config.COLORS[6],
                fontstyle="italic",
            )

    # ---- 样式 ----
    ax.set_xlim(-0.75, 1.5)
    ax.set_ylim(-0.5, n_stages - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(title, fontsize=plot_config.title_fontsize,
                 fontweight="bold", pad=20)

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_constraint_funnel] 已保存: {save_path}")


# ============================================================
#  2. 约束淘汰比例水平条形图
# ============================================================
def plot_constraint_bar(
    stage_labels: List[str],
    stage_counts: List[int],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "constraint_bar.png"),
    title: str = "各阶段窗口存活数",
    color_active: Optional[str] = None,
    color_eliminated: str = "#E5E7EB",
) -> None:
    """绘制水平条形图，展示各约束阶段的窗口存活与淘汰数量。

    每一行由两段组成：左侧彩色段 = 该阶段存活数，
    右侧灰色段 = 被该阶段淘汰的数量（与上一阶段之差）。

    Parameters
    ----------
    stage_labels : list[str]
        各阶段名称。
    stage_counts : list[int]
        各阶段剩余窗口数量。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    color_active : str, optional
        存活段颜色。若为 None 则使用 ``plot_config.COLORS[3]``。
    color_eliminated : str
        淘汰段颜色，默认浅灰。

    Notes
    -----
    - 每行右侧标注存活数 / 淘汰数。
    - 适用于论文中展示约束效率。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    n_stages = len(stage_labels)
    if color_active is None:
        color_active = plot_config.COLORS[3]

    max_count = max(stage_counts) if stage_counts else 1

    fig, ax = plt.subplots(figsize=(12, max(3, n_stages * 0.6 + 1)),
                           constrained_layout=True)

    y_positions = list(range(n_stages - 1, -1, -1))

    for i in range(n_stages):
        y = y_positions[i]
        count = stage_counts[i]

        # 存活段
        ax.barh(y, count, height=0.55, color=color_active,
                alpha=0.85, edgecolor="white", linewidth=0.8)

        # 淘汰段（与上一阶段之差）
        if i > 0:
            eliminated = stage_counts[i - 1] - stage_counts[i]
            if eliminated > 0:
                ax.barh(y, eliminated, height=0.55, left=count,
                        color=color_eliminated, alpha=0.7,
                        edgecolor="#D1D5DB", linewidth=0.5)

        # ---- 右侧标注 ----
        eliminated = (stage_counts[i - 1] - stage_counts[i]) if i > 0 else 0
        pct = (100.0 * count / stage_counts[0]) if stage_counts[0] > 0 else 0
        ax.text(
            max_count * 1.02, y,
            f"{count}  ({pct:.1f}%)  淘汰 {eliminated}",
            ha="left", va="center",
            fontsize=plot_config.tick_fontsize,
        )

    # ---- 样式 ----
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_labels, fontsize=plot_config.tick_fontsize)
    ax.set_xlabel("窗口数量", fontsize=plot_config.label_fontsize)
    ax.set_title(title, fontsize=plot_config.title_fontsize, fontweight="bold")
    ax.set_xlim(0, max_count * 1.8)
    ax.invert_yaxis()

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_constraint_bar] 已保存: {save_path}")


# ============================================================
#  辅助：从 stage4 数据自动构建漏斗阶段
# ============================================================
def build_funnel_from_stage4(
    windows_shoot: list,
    windows_photo: list,
    scheduled_tasks: list,
    all_targets: list,
) -> Tuple[List[str], List[int]]:
    """从 stage4 的窗口和调度结果自动构建漏斗各阶段数据。

    Parameters
    ----------
    windows_shoot : list
        射击可行窗口列表。
    windows_photo : list
        拍照可行窗口列表。
    scheduled_tasks : list
        最终调度结果。
    all_targets : list
        目标点列表。

    Returns
    -------
    stage_labels : list[str]
        各阶段名称。
    stage_counts : list[int]
        各阶段窗口数。

    Notes
    -----
    由于约束检查过程的中间步骤数据不可直接获取，
    此函数基于已有数据推算近似漏斗：
      · 第 1 层 = 全部候选窗口
      · 第 2 层 = 稀疏采样后窗口数（若有）
      · 第 3 层 = 最终调度任务数
      · 第 4 层 = 覆盖的目标数

    若需更精细的漏斗，建议在 ConstraintChecker 或
    调度函数中插入计数器。
    """
    total_candidates = len(windows_shoot) + len(windows_photo)

    # 按目标统计各类型窗口数
    shoot_target_count = len(set(w["target_id"] for w in windows_shoot))
    photo_target_count = len(set(w["target_id"] for w in windows_photo))
    all_window_target_count = len(
        set(w["target_id"] for w in windows_shoot + windows_photo)
    )

    total_scheduled = len(scheduled_tasks)
    sched_shoot = len([t for t in scheduled_tasks if t["task_type"] == "shoot"])
    sched_photo = len([t for t in scheduled_tasks if t["task_type"] == "photo"])

    covered_targets = len(
        set(t["target_id"] for t in scheduled_tasks)
    )
    total_targets = len(all_targets)

    stage_labels = [
        "全部候选窗口",
        "射击候选窗口",
        "拍照候选窗口",
        "最终调度任务",
        "  └ 射击任务",
        "  └ 拍照任务",
        f"覆盖目标 ({covered_targets}/{total_targets})",
    ]
    stage_counts = [
        total_candidates,
        len(windows_shoot),
        len(windows_photo),
        total_scheduled,
        sched_shoot,
        sched_photo,
        covered_targets,
    ]

    return stage_labels, stage_counts


# ============================================================
#  自检
# ============================================================
if __name__ == "__main__":
    # ---- 示例数据演示 ----
    demo_labels = [
        "全部候选窗口",
        "时间冲突剔除",
        "速度约束过滤",
        "距离约束过滤",
        "角度约束过滤",
        "最终调度结果",
    ]
    demo_counts = [520, 380, 290, 210, 175, 85]

    out = Path(PLOT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    plot_constraint_funnel(
        demo_labels, demo_counts,
        save_path=out / "constraint_funnel_demo.png",
    )
    plot_constraint_bar(
        demo_labels, demo_counts,
        save_path=out / "constraint_bar_demo.png",
    )
    print("\n[plot_constraint_analysis] 自检完成。")
