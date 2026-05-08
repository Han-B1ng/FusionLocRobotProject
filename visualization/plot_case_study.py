# file: visualization/plot_case_study.py
# @Description : 局部案例分析可视化模块（多轴联动图）
# 依赖：matplotlib, numpy, pathlib
# 上游：stage3 融合轨迹 + stage4 调度结果 + 约束参数
# 下游：被 main.py 或各 stage 脚本调用

"""
visualization/plot_case_study.py
=================================
提供局部案例分析可视化函数：
  1. plot_case_study — 某一时刻附近的多轴联动图
     （轨迹 + 速度 + 加速度 + 距离 + 视锥）

设计目标：
  选中一个任务执行时刻，在其前后 ±window 秒的窗口内，
  将轨迹、运动状态、目标几何关系以多子图联动方式呈现，
  直观展示该时刻是否满足所有约束条件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Wedge

from config import plot_config
# 应用中文字体配置
plot_config.apply_style()


# ============================================================
#  辅助
# ============================================================
def _ensure_parent(path: Path) -> None:
    """创建保存路径的父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def _compute_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算轨迹各点的航向角（度，0°=正东，逆时针为正）。

    Parameters
    ----------
    x, y : np.ndarray
        轨迹坐标序列。

    Returns
    -------
    heading : np.ndarray
        航向角序列，长度与输入相同。首尾点用邻近差分填充。
    """
    n = len(x)
    heading = np.zeros(n)

    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.degrees(np.arctan2(dy, dx))

    return heading


def _compute_fov_cone(
    robot_x: float,
    robot_y: float,
    heading_deg: float,
    fov_angle: float = 60.0,
    cone_length: float = 15.0,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算视锥（扇形）的边界坐标。

    Parameters
    ----------
    robot_x, robot_y : float
        机器人当前位置。
    heading_deg : float
        航向角（度）。
    fov_angle : float
        视锥张角（度）。
    cone_length : float
        视锥延伸长度（米）。
    n_points : int
        扇形弧上的采样点数。

    Returns
    -------
    cone_x, cone_y : np.ndarray
        视锥封闭多边形的 X / Y 坐标（含起点闭合）。
    """
    half_fov = np.radians(fov_angle / 2.0)
    heading_rad = np.radians(heading_deg)

    angles = np.linspace(
        heading_rad - half_fov,
        heading_rad + half_fov,
        n_points,
    )

    arc_x = robot_x + cone_length * np.cos(angles)
    arc_y = robot_y + cone_length * np.sin(angles)

    cone_x = np.concatenate([[robot_x], arc_x, [robot_x]])
    cone_y = np.concatenate([[robot_y], arc_y, [robot_y]])

    return cone_x, cone_y


# ============================================================
#  主函数
# ============================================================
def plot_case_study(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    acc: np.ndarray,
    target_x: float,
    target_y: float,
    target_name: str = "T1",
    task_type: str = "shoot",
    t_exec: Optional[float] = None,
    window: float = 30.0,
    save_path: Union[str, Path] = "output/figures/case_study.png",
    title: Optional[str] = None,
    fov_angle: float = 60.0,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    speed_limit: Optional[float] = None,
) -> None:
    """绘制某一任务执行时刻附近的多轴联动案例分析图。

    Parameters
    ----------
    t, x, y : np.ndarray
        融合轨迹的时间、X、Y 坐标序列。
    speed : np.ndarray
        合成速率序列（m/s），长度与 t 相同。
    acc : np.ndarray
        合成加速度序列（m/s^2），长度与 t 相同。
    target_x, target_y : float
        目标点坐标。
    target_name : str
        目标名称，用于标题和标注。
    task_type : str
        任务类型，``'shoot'`` 或 ``'photo'``。
    t_exec : float, optional
        任务执行时刻。若为 None 则取轨迹中点时刻。
    window : float
        时间窗口半径（秒），默认 ±30 秒。
    save_path : str 或 Path
        图片保存路径。
    title : str, optional
        总标题。若为 None 则自动生成。
    fov_angle : float
        视锥张角（度），默认 60°。
    dist_min, dist_max : float, optional
        距离约束上下限（米），在距离子图上以水平虚线标注。
    speed_limit : float, optional
        速度约束上限（m/s），在速度子图上以水平虚线标注。

    Notes
    -----
    布局为 3×2 网格：

    ┌──────────────────┬──────────────┐
    │  (a) 轨迹 + 视锥  │ (b) 速度曲线  │
    ├──────────────────┼──────────────┤
    │  (c) 加速度曲线   │ (d) 距离曲线  │
    ├──────────────────┼──────────────┤
    │  (e) 航向角曲线   │ (f) 约束摘要  │
    └──────────────────┴──────────────┘

    - 左上 (a)：2D 轨迹局部放大 + 目标点 + 视锥扇形
    - 右上 (b)：速度随时间变化 + 速度约束线
    - 左中 (c)：加速度随时间变化
    - 右中 (d)：到目标的距离随时间变化 + 距离约束区间
    - 左下 (e)：航向角随时间变化
    - 右下 (f)：文本约束满足摘要
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    # ---- 确定分析窗口 ----
    if t_exec is None:
        t_exec = float(t[len(t) // 2])

    t_lo = t_exec - window
    t_hi = t_exec + window

    mask = (t >= t_lo) & (t <= t_hi)
    t_sel = t[mask]
    x_sel = x[mask]
    y_sel = y[mask]
    speed_sel = speed[mask]
    acc_sel = acc[mask]

    if len(t_sel) == 0:
        print(f"[plot_case_study] 警告：窗口 [{t_lo:.1f}, {t_hi:.1f}] "
              f"内无数据点，跳过绘图")
        return

    # ---- 航向角 ----
    heading = _compute_heading(x, y)
    heading_sel = heading[mask]

    # ---- 到目标距离 ----
    dist_to_target = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
    dist_sel = dist_to_target[mask]

    # ---- 执行时刻的索引和状态 ----
    idx_exec = int(np.argmin(np.abs(t - t_exec)))
    x_exec, y_exec = x[idx_exec], y[idx_exec]
    speed_exec = speed[idx_exec]
    acc_exec = acc[idx_exec]
    heading_exec = heading[idx_exec]
    dist_exec = dist_to_target[idx_exec]

    # ---- 视锥 ----
    cone_x, cone_y = _compute_fov_cone(
        x_exec, y_exec, heading_exec,
        fov_angle=fov_angle,
        cone_length=max(dist_exec * 0.8, 10.0),
    )

    # ---- 标题 ----
    if title is None:
        task_cn = "射击" if task_type == "shoot" else "拍照"
        title = (f"案例分析 — {target_name} {task_cn} "
                 f"@ t={t_exec:.1f}s")

    # ---- 颜色 ----
    c_traj = plot_config.COLORS[3]
    c_exec = plot_config.COLORS[0]
    c_target = plot_config.COLORS[6]
    c_fov = plot_config.COLORS[4] if task_type == "shoot" else plot_config.COLORS[5]
    c_speed = plot_config.COLORS[1]
    c_acc = plot_config.COLORS[2]
    c_dist = plot_config.COLORS[5]

    # ══════════════════════════════════════════════
    #  绘图：3×2 网格
    # ══════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 12))

    # 自定义网格布局：左列宽于右列
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.3, 1],
        hspace=0.35, wspace=0.30,
    )

    # ── (a) 轨迹 + 视锥 ──
    ax_traj = fig.add_subplot(gs[0, 0])

    # 完整轨迹（淡色背景）
    ax_traj.plot(x, y, color=c_traj, linewidth=0.4, alpha=0.25)

    # 局部窗口轨迹（高亮）
    ax_traj.plot(x_sel, y_sel, color=c_traj,
                 linewidth=plot_config.linewidth, alpha=0.9,
                 label="局部轨迹")

    # 目标点
    ax_traj.scatter(
        target_x, target_y, s=150, marker="^",
        color=c_target, edgecolors="black", linewidths=1.0,
        zorder=5, label=f"{target_name}",
    )
    ax_traj.annotate(
        target_name, (target_x, target_y),
        textcoords="offset points", xytext=(8, 8),
        fontsize=plot_config.tick_fontsize,
        fontweight="bold", color=c_target,
    )

    # 执行时刻位置
    ax_traj.scatter(
        x_exec, y_exec, s=120, marker="o",
        color=c_exec, edgecolors="white", linewidths=1.5,
        zorder=6, label="执行位置",
    )

    # 视锥扇形
    ax_traj.fill(
        cone_x, cone_y,
        color=c_fov, alpha=0.15, edgecolor=c_fov,
        linewidth=1.0, linestyle="--", label=f"视锥 ({fov_angle:.0f}°)",
    )

    # 执行位置 → 目标的连线
    ax_traj.plot(
        [x_exec, target_x], [y_exec, target_y],
        color=c_exec, linewidth=0.8, linestyle=":",
        alpha=0.6,
    )

    # 距离标注
    ax_traj.annotate(
        f"d = {dist_exec:.1f} m",
        ((x_exec + target_x) / 2, (y_exec + target_y) / 2),
        textcoords="offset points", xytext=(5, 10),
        fontsize=plot_config.tick_fontsize - 1,
        color=c_exec, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax_traj.set_xlabel("X (m)", fontsize=plot_config.label_fontsize)
    ax_traj.set_ylabel("Y (m)", fontsize=plot_config.label_fontsize)
    ax_traj.set_title("(a) 轨迹与视锥",
                      fontsize=plot_config.title_fontsize, fontweight="bold")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(
        loc="best", fontsize=plot_config.legend_fontsize - 1,
        frameon=plot_config.legend_frameon,
    )

    # ── (b) 速度曲线 ──
    ax_speed = fig.add_subplot(gs[0, 1])

    ax_speed.plot(t_sel, speed_sel, color=c_speed,
                  linewidth=plot_config.linewidth, label="合成速率")

    # 执行时刻标记
    ax_speed.axvline(t_exec, color=c_exec, linestyle="--",
                     linewidth=1.0, alpha=0.7, label=f"t={t_exec:.1f}s")
    ax_speed.plot(t_exec, speed_exec, marker="o", markersize=8,
                  color=c_exec, markeredgecolor="white",
                  markeredgewidth=1.0, zorder=5)

    # 速度约束线
    if speed_limit is not None:
        ax_speed.axhline(speed_limit, color=c_target, linestyle="--",
                         linewidth=1.0, alpha=0.7,
                         label=f"v_max = {speed_limit:.1f} m/s")

    ax_speed.set_xlabel("时间 (s)", fontsize=plot_config.label_fontsize)
    ax_speed.set_ylabel("速率 (m/s)", fontsize=plot_config.label_fontsize)
    ax_speed.set_title("(b) 速度曲线",
                      fontsize=plot_config.title_fontsize, fontweight="bold")
    ax_speed.legend(
        loc="best", fontsize=plot_config.legend_fontsize - 1,
        frameon=plot_config.legend_frameon,
    )

    # ── (c) 加速度曲线 ──
    ax_acc = fig.add_subplot(gs[1, 0])

    ax_acc.plot(t_sel, acc_sel, color=c_acc,
                linewidth=plot_config.linewidth, label="合成加速度")

    ax_acc.axvline(t_exec, color=c_exec, linestyle="--",
                   linewidth=1.0, alpha=0.7)
    ax_acc.plot(t_exec, acc_exec, marker="o", markersize=8,
                color=c_exec, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)

    ax_acc.set_xlabel("时间 (s)", fontsize=plot_config.label_fontsize)
    ax_acc.set_ylabel("加速度 (m/s^2)", fontsize=plot_config.label_fontsize)
    ax_acc.set_title("(c) 加速度曲线",
                     fontsize=plot_config.title_fontsize, fontweight="bold")
    ax_acc.legend(
        loc="best", fontsize=plot_config.legend_fontsize - 1,
        frameon=plot_config.legend_frameon,
    )

    # ── (d) 距离曲线 ──
    ax_dist = fig.add_subplot(gs[1, 1])

    ax_dist.plot(t_sel, dist_sel, color=c_dist,
                 linewidth=plot_config.linewidth, label=f"到 {target_name}")

    ax_dist.axvline(t_exec, color=c_exec, linestyle="--",
                    linewidth=1.0, alpha=0.7)
    ax_dist.plot(t_exec, dist_exec, marker="o", markersize=8,
                 color=c_exec, markeredgecolor="white",
                 markeredgewidth=1.0, zorder=5)

    # 距离约束区间
    if dist_min is not None and dist_max is not None:
        ax_dist.axhspan(dist_min, dist_max, color=c_traj,
                        alpha=0.10, label=f"约束区间 [{dist_min}, {dist_max}] m")
        ax_dist.axhline(dist_min, color=c_traj, linestyle=":",
                        linewidth=0.8, alpha=0.6)
        ax_dist.axhline(dist_max, color=c_traj, linestyle=":",
                        linewidth=0.8, alpha=0.6)
    elif dist_min is not None:
        ax_dist.axhline(dist_min, color=c_traj, linestyle="--",
                        linewidth=1.0, alpha=0.7,
                        label=f"d_min = {dist_min:.1f} m")
    elif dist_max is not None:
        ax_dist.axhline(dist_max, color=c_traj, linestyle="--",
                        linewidth=1.0, alpha=0.7,
                        label=f"d_max = {dist_max:.1f} m")

    ax_dist.set_xlabel("时间 (s)", fontsize=plot_config.label_fontsize)
    ax_dist.set_ylabel("距离 (m)", fontsize=plot_config.label_fontsize)
    ax_dist.set_title("(d) 到目标距离",
                      fontsize=plot_config.title_fontsize, fontweight="bold")
    ax_dist.legend(
        loc="best", fontsize=plot_config.legend_fontsize - 1,
        frameon=plot_config.legend_frameon,
    )

    # ── (e) 航向角曲线 ──
    ax_heading = fig.add_subplot(gs[2, 0])

    # 将航向角展开到连续曲线（处理 ±180° 跳变）
    heading_unwrap = np.unwrap(np.radians(heading_sel))
    heading_unwrap_deg = np.degrees(heading_unwrap)

    ax_heading.plot(t_sel, heading_unwrap_deg, color=c_fov,
                    linewidth=plot_config.linewidth, label="航向角")

    ax_heading.axvline(t_exec, color=c_exec, linestyle="--",
                       linewidth=1.0, alpha=0.7)
    ax_heading.plot(t_exec, heading_unwrap_deg[
        int(np.argmin(np.abs(t_sel - t_exec)))
    ] if len(t_sel) > 0 else 0,
        marker="o", markersize=8, color=c_exec,
        markeredgecolor="white", markeredgewidth=1.0, zorder=5,
    )

    ax_heading.set_xlabel("时间 (s)", fontsize=plot_config.label_fontsize)
    ax_heading.set_ylabel("航向角 (deg)", fontsize=plot_config.label_fontsize)
    ax_heading.set_title("(e) 航向角",
                         fontsize=plot_config.title_fontsize, fontweight="bold")
    ax_heading.legend(
        loc="best", fontsize=plot_config.legend_fontsize - 1,
        frameon=plot_config.legend_frameon,
    )

    # ── (f) 约束满足摘要 ──
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis("off")

    # 计算约束满足情况
    lines = []
    lines.append(f"案例：{target_name}  任务：{'射击' if task_type == 'shoot' else '拍照'}")
    lines.append(f"执行时刻：{t_exec:.2f} s")
    lines.append(f"窗口范围：[{t_lo:.1f}, {t_hi:.1f}] s")
    lines.append("")

    # 速度
    v_ok = True
    v_info = f"  速率 = {speed_exec:.3f} m/s"
    if speed_limit is not None:
        v_ok = speed_exec <= speed_limit
        v_info += f"  (限 {speed_limit:.1f} m/s)"
    lines.append(f"{'[PASS]' if v_ok else '[FAIL]'} 速度约束")
    lines.append(v_info)

    # 加速度
    lines.append(f"  加速度 = {acc_exec:.3f} m/s^2")
    lines.append("")

    # 距离
    d_ok = True
    d_info = f"  距离 = {dist_exec:.2f} m"
    if dist_min is not None and dist_max is not None:
        d_ok = dist_min <= dist_exec <= dist_max
        d_info += f"  (约束 [{dist_min:.1f}, {dist_max:.1f}] m)"
    lines.append(f"{'[PASS]' if d_ok else '[FAIL]'} 距离约束")
    lines.append(d_info)

    # 航向角
    lines.append("")
    lines.append(f"  航向角 = {heading_exec:.1f} deg")

    # 综合判定
    all_pass = v_ok and d_ok
    lines.append("")
    lines.append("=" * 30)
    if all_pass:
        lines.append("  * 所有约束满足")
    else:
        lines.append("  x 存在约束违反")

    text_content = "\n".join(lines)
    ax_text.text(
        0.05, 0.92, text_content,
        transform=ax_text.transAxes, fontsize=9,
        verticalalignment="top",
        fontname=plot_config.CN_FONT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F3F4F6", alpha=0.95),
    )

    ax_text.set_title("(f) 约束满足摘要",
                      fontsize=plot_config.title_fontsize, fontweight="bold",
                      loc="left")

    # ---- 总标题 ----
    fig.suptitle(title, fontsize=plot_config.title_fontsize + 2,
                 fontweight="bold", y=1.01)

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_case_study] 已保存: {save_path}")


# ============================================================
#  自检
# ============================================================
if __name__ == "__main__":
    out = Path("output/figures")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 生成模拟轨迹数据 ----
    np.random.seed(42)
    t_demo = np.linspace(0, 200, 2000)

    # 模拟一段弯曲轨迹
    x_demo = 50 * np.sin(0.02 * t_demo) + 0.01 * t_demo
    y_demo = 30 * np.cos(0.015 * t_demo) + 0.005 * t_demo ** 1.1

    # 模拟速度和加速度
    vx_demo = np.gradient(x_demo, t_demo)
    vy_demo = np.gradient(y_demo, t_demo)
    speed_demo = np.sqrt(vx_demo ** 2 + vy_demo ** 2)

    ax_demo = np.gradient(vx_demo, t_demo)
    ay_demo = np.gradient(vy_demo, t_demo)
    acc_demo = np.sqrt(ax_demo ** 2 + ay_demo ** 2)

    # ---- 调用绘图 ----
    plot_case_study(
        t=t_demo, x=x_demo, y=y_demo,
        speed=speed_demo, acc=acc_demo,
        target_x=40.0, target_y=25.0,
        target_name="T1", task_type="shoot",
        t_exec=100.0, window=40.0,
        save_path=out / "case_study_demo.png",
        fov_angle=60.0,
        dist_min=5.0, dist_max=30.0,
        speed_limit=2.0,
    )

    print("\n[plot_case_study] 自检完成。")
