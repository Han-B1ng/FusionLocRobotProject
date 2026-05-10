# file: visualization/plot_publication.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from config import plot_config, PLOT_DIR
plot_config.apply_style()

_JOURNAL_RC: Dict[str, Any] = {

    "font.family":        "sans-serif",
    "font.sans-serif":    [plot_config.font_cjk, "Microsoft YaHei", "STSong", "Source Han Serif SC", "DejaVu Sans"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.6,
    "lines.linewidth":    0.8,
    "grid.linewidth":     0.3,
    "grid.alpha":         0.25,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.minor.size":   1.5,
    "ytick.minor.size":   1.5,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#CCCCCC",
    "legend.borderpad":   0.4,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}

_C_BLUE   = "#0072B2"   # 蓝
_C_ORANGE = "#E69F00"   # 橙
_C_GREEN  = "#009E73"   # 绿
_C_RED    = "#D55E00"   # 红
_C_GRAY   = "#999999"   # 灰
_C_PURPLE = "#CC79A7"   # 紫

_SENSOR_COLORS: Dict[str, str] = {
    "方式1": _C_BLUE,
    "方式2": _C_ORANGE,
}

_DPI_PUB = 300


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_journal_style() -> None:
    plt.rcParams.update(_JOURNAL_RC)


def _save_multi_format(
    fig: plt.Figure,
    base_path: Path,
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    for fmt in formats:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=_DPI_PUB, bbox_inches="tight", pad_inches=0.02)
        print(f"  [save] {out}")


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


def _compute_cross_corr(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delay_range: Tuple[float, float] = (-2.0, 2.0),
    n_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    if t_end - t_start < 1.0:
        return np.array([]), np.array([])

    dt = 0.1
    t_grid = np.arange(t_start, t_end, dt)

    x1g = np.interp(t_grid, t1, x1)
    y1g = np.interp(t_grid, t1, y1)
    x2g = np.interp(t_grid, t2, x2)
    y2g = np.interp(t_grid, t2, y2)

    delays = np.linspace(delay_range[0], delay_range[1], n_steps)
    scores = np.zeros(len(delays))

    for i, delta in enumerate(delays):
        t_shifted = t_grid - delta
        mask = (t_shifted >= t_grid[0]) & (t_shifted <= t_grid[-1])
        n_overlap = int(np.sum(mask))
        if n_overlap < 10:
            scores[i] = -np.inf
            continue

        x2s = np.interp(t_shifted[mask], t_grid, x2g)
        y2s = np.interp(t_shifted[mask], t_grid, y2g)
        x1s = x1g[mask]
        y1s = y1g[mask]

        def _pearson(a, b):
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
            if denom < 1e-15:
                return 0.0
            return float(np.sum(a_c * b_c) / denom)

        scores[i] = 0.5 * _pearson(x1s, x2s) + 0.5 * _pearson(y1s, y2s)

    return delays, scores


def summary_figure_paper(
    problem_num: int,
    data_dict: Dict[str, Any],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "summary_p{}.png"),
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if "{" in str(save_path) or "%" in str(save_path):
        save_path = Path(str(save_path).format(problem_num=problem_num, p=problem_num))
    else:
        stem = save_path.stem
        save_path = save_path.parent / f"{stem}_p{problem_num}{save_path.suffix}"
    _ensure_parent(save_path)
    base_path = save_path.with_suffix("")

    saved_rc = plt.rcParams.copy()
    _apply_journal_style()

    t1 = np.asarray(data_dict["t1"])
    x1 = np.asarray(data_dict["x1"])
    y1 = np.asarray(data_dict["y1"])
    t2 = np.asarray(data_dict["t2"])
    x2 = np.asarray(data_dict["x2"])
    y2 = np.asarray(data_dict["y2"])
    t_fused = np.asarray(data_dict["t_fused"])
    x_fused = np.asarray(data_dict["x_fused"])
    y_fused = np.asarray(data_dict["y_fused"])

    has_ref = all(k in data_dict for k in ("t_ref", "x_ref", "y_ref"))
    has_error = all(k in data_dict for k in ("error_x", "error_y"))
    has_bias = all(k in data_dict for k in ("bias_x", "bias_y"))
    show_bias = problem_num >= 2 and has_bias

    fig = plt.figure(figsize=(7.2, 5.6))  # 双栏期刊典型宽度
    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.35, wspace=0.30,
        left=0.08, right=0.96, top=0.90, bottom=0.10,
    )

    ax_traj   = fig.add_subplot(gs[0, 0])
    ax_err    = fig.add_subplot(gs[0, 1])
    ax_speed  = fig.add_subplot(gs[1, 0])
    ax_bias   = fig.add_subplot(gs[1, 1])

    ax_traj.scatter(
        x1, y1, s=1.5, c=_C_BLUE, alpha=0.35, linewidths=0,
        label="传感器 1", rasterized=True,
    )
    ax_traj.scatter(
        x2, y2, s=1.5, c=_C_ORANGE, alpha=0.35, linewidths=0,
        label="传感器 2", rasterized=True,
    )
    ax_traj.plot(
        x_fused, y_fused,
        color=_C_GREEN, linewidth=0.8, alpha=0.95,
        label="融合轨迹",
    )
    if has_ref:
        x_ref = np.asarray(data_dict["x_ref"])
        y_ref = np.asarray(data_dict["y_ref"])
        ax_traj.plot(
            x_ref, y_ref,
            color=_C_GRAY, linewidth=0.5, linestyle="--", alpha=0.8,
            label="参考轨迹",
        )

    ax_traj.plot(x_fused[0], y_fused[0], "o", color=_C_GREEN,
                 markersize=4, markeredgecolor="white", markeredgewidth=0.5)
    ax_traj.plot(x_fused[-1], y_fused[-1], "s", color=_C_GREEN,
                 markersize=4, markeredgecolor="white", markeredgewidth=0.5)

    ax_traj.set_xlabel("$x$ (m)")
    ax_traj.set_ylabel("$y$ (m)")
    ax_traj.set_title("(a) 轨迹对比", fontsize=10, fontweight="bold")
    ax_traj.legend(loc="best", fontsize=7, markerscale=3)
    ax_traj.set_aspect("equal", adjustable="datalim")

    if has_error:
        t_err = np.asarray(data_dict.get("t_error", t_fused))
        err_x = np.asarray(data_dict["error_x"])
        err_y = np.asarray(data_dict["error_y"])

        ax_err.plot(t_err, err_x, color=_C_BLUE, linewidth=0.5,
                    alpha=0.8, label=r"$e_x$")
        ax_err.plot(t_err, err_y, color=_C_RED, linewidth=0.5,
                    alpha=0.8, label=r"$e_y$")
        ax_err.axhline(0, color="black", linewidth=0.3, linestyle="--")

        rmse_x = np.sqrt(np.mean(err_x ** 2))
        rmse_y = np.sqrt(np.mean(err_y ** 2))
        stats = (
            f"RMSE$_x$={rmse_x:.3f} m\n"
            f"RMSE$_y$={rmse_y:.3f} m"
        )
        ax_err.text(
            0.97, 0.97, stats,
            transform=ax_err.transAxes, fontsize=7,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    else:
        ax_err.text(0.5, 0.5, "无误差数据", transform=ax_err.transAxes,
                    fontsize=9, ha="center", va="center", color=_C_GRAY)
    ax_err.set_xlabel("$t$ (s)")
    ax_err.set_ylabel("误差 (m)")
    ax_err.set_title("(b) 融合误差", fontsize=10, fontweight="bold")
    ax_err.legend(loc="upper left", fontsize=7)

    has_cc = all(k in data_dict for k in ("cc_delays", "cc_scores"))
    if has_cc:
        cc_d = np.asarray(data_dict["cc_delays"])
        cc_s = np.asarray(data_dict["cc_scores"])
    else:
        cc_d, cc_s = _compute_cross_corr(
            t1, x1, y1,
            data_dict.get("t2_orig", t2), x2, y2,
        )

    if cc_d is not None and cc_s is not None and len(cc_d) > 0:
        ax_speed.plot(cc_d, cc_s, color=_C_BLUE, linewidth=0.6, alpha=0.85)

        best_idx = np.argmax(cc_s)
        ax_speed.axvline(cc_d[best_idx], color=_C_RED, linewidth=0.5,
                         linestyle="--", alpha=0.7)
        ax_speed.plot(cc_d[best_idx], cc_s[best_idx], "o",
                      color=_C_RED, markersize=5, markeredgecolor="white",
                      markeredgewidth=0.5, zorder=5)

        delay_val = data_dict.get("delay", cc_d[best_idx])
        ax_speed.text(
            0.97, 0.97, f"$\\Delta t$ = {delay_val:+.4f} s",
            transform=ax_speed.transAxes, fontsize=7,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    else:
        ax_speed.text(0.5, 0.5, "无互相关数据", transform=ax_speed.transAxes,
                      fontsize=9, ha="center", va="center", color=_C_GRAY)

    ax_speed.set_xlabel("候选时偏 $\\Delta$ (s)")
    ax_speed.set_ylabel("加权相关系数")
    ax_speed.set_title("(c) 时间偏差估计", fontsize=10, fontweight="bold")

    if show_bias:
        t_b = np.asarray(data_dict["t_bias"])
        bx = np.asarray(data_dict["bias_x"])
        by = np.asarray(data_dict["bias_y"])

        ax_bias.plot(t_b, bx, color=_C_BLUE, linewidth=0.6,
                     alpha=0.85, label=r"$\hat{b}_x$")
        ax_bias.plot(t_b, by, color=_C_RED, linewidth=0.6,
                     alpha=0.85, label=r"$\hat{b}_y$")

        bx_true = data_dict.get("bias_true_x", None)
        by_true = data_dict.get("bias_true_y", None)
        if bx_true is not None:
            ax_bias.axhline(
                bx_true, color=_C_BLUE, linewidth=0.4,
                linestyle=":", alpha=0.6,
                label=f"$b_x^*$={bx_true:.2f}",
            )
        if by_true is not None:
            ax_bias.axhline(
                by_true, color=_C_RED, linewidth=0.4,
                linestyle=":", alpha=0.6,
                label=f"$b_y^*$={by_true:.2f}",
            )

        ax_bias.legend(loc="best", fontsize=7, ncol=2)
    else:
        ax_bias.text(
            0.5, 0.5,
            "问题 1：无系统偏差" if problem_num == 1 else "无偏差估计数据",
            transform=ax_bias.transAxes, fontsize=9,
            ha="center", va="center", color=_C_GRAY,
        )

    ax_bias.set_xlabel("$t$ (s)")
    ax_bias.set_ylabel("偏差估计 (m)")
    ax_bias.set_title(
        "(d) 系统偏差收敛" if show_bias else "(d) 系统偏差",
        fontsize=10, fontweight="bold",
    )

    problem_labels = {1: "问题 1：无噪声时间对齐",
                      2: "问题 2：含偏差融合",
                      3: "问题 3：实际数据处理"}
    fig.suptitle(
        problem_labels.get(problem_num, f"问题 {problem_num} 结果"),
        fontsize=12, fontweight="bold", y=0.97,
    )

    _save_multi_format(fig, base_path, formats)
    plt.close(fig)
    plt.rcParams.update(saved_rc)  # 恢复原始样式
    print(f"[summary_figure_paper] 问题 {problem_num} 组合图已保存。")


def task_planning_paper(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    tasks: List[Any],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "task_planning.png"),
    t: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)
    base_path = save_path.with_suffix("")

    saved_rc = plt.rcParams.copy()
    _apply_journal_style()

    parsed: List[Dict[str, Any]] = []
    for task in tasks:
        ttype = str(_extract_task_field(task, "task_type", "type", default="")).lower()
        tid = _extract_task_field(task, "target_id", "target", "id", default="?")
        t_prep_start = _extract_task_field(task, "t_prep_start", default=None)
        t_exec_start = _extract_task_field(task, "t_exec_start", "t_start", default=None)
        t_exec_end = _extract_task_field(task, "t_exec_end", "t_end", default=None)
        prep_dur = _extract_task_field(task, "prep_duration", "prep_time", default=0.0)
        tx = _extract_task_field(task, "x", "pos_x", default=None)
        ty = _extract_task_field(task, "y", "pos_y", default=None)
        t_exec_time = _extract_task_field(task, "t_exec", "time", default=None)

        if t_exec_start is None:
            continue

        if t_prep_start is None:
            t_prep_start = float(t_exec_start) - float(prep_dur)

        if tx is None and t is not None and t_exec_time is not None:
            idx = np.argmin(np.abs(np.asarray(t) - float(t_exec_time)))
            tx = traj_x[idx]
            ty = traj_y[idx]

        is_shoot = "shoot" in ttype or "射击" in ttype

        parsed.append({
            "target_id":    tid,
            "is_shoot":     is_shoot,
            "t_prep_start": float(t_prep_start),
            "t_exec_start": float(t_exec_start),
            "t_exec_end":   float(t_exec_end) if t_exec_end else float(t_exec_start),
            "x":            tx,
            "y":            ty,
        })

    parsed.sort(key=lambda d: d["t_exec_start"])
    n_tasks = len(parsed)

    fig = plt.figure(figsize=(7.2, 4.0))
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.3, 1.0],
        wspace=0.30,
        left=0.06, right=0.97, top=0.88, bottom=0.12,
    )

    ax_traj = fig.add_subplot(gs[0, 0])
    ax_gantt = fig.add_subplot(gs[0, 1])

    ax_traj.plot(
        traj_x, traj_y,
        color=_C_GRAY, linewidth=0.5, alpha=0.6, zorder=1,
    )

    if targets is not None:
        tgt = np.asarray(targets)
        if tgt.ndim == 2 and tgt.shape[1] >= 3:
            tx_plot, ty_plot = tgt[:, 1], tgt[:, 2]
        elif tgt.ndim == 2 and tgt.shape[1] == 2:
            tx_plot, ty_plot = tgt[:, 0], tgt[:, 1]
        else:
            tx_plot, ty_plot = np.array([]), np.array([])

        ax_traj.scatter(
            tx_plot, ty_plot, s=18, marker="D",
            facecolor="none", edgecolor=_C_GRAY, linewidths=0.6,
            alpha=0.6, zorder=2, label="目标点",
        )

    for p in parsed:
        px, py = p["x"], p["y"]
        if px is None or py is None:
            continue

        if p["is_shoot"]:
            marker, color, ms = "^", _C_RED, 6
            tag = f"S{p['target_id']}"
            offset = (5, 5)
        else:
            marker, color, ms = "o", _C_BLUE, 5
            tag = f"P{p['target_id']}"
            offset = (5, -8)

        ax_traj.scatter(
            px, py, marker=marker, s=ms ** 2, c=color,
            edgecolors="white", linewidths=0.4, zorder=5,
        )
        ax_traj.annotate(
            tag, (px, py), fontsize=6, color=color, fontweight="bold",
            xytext=offset, textcoords="offset points",
        )

    ax_traj.set_xlabel("$x$ (m)")
    ax_traj.set_ylabel("$y$ (m)")
    ax_traj.set_title("(a) 任务执行位置", fontsize=10, fontweight="bold")
    ax_traj.set_aspect("equal", adjustable="datalim")

    handles_traj = [
        Line2D([0], [0], color=_C_GRAY, linewidth=0.5, label="融合轨迹"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=_C_RED,
               markersize=6, linestyle="None", label="射击"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_BLUE,
               markersize=5, linestyle="None", label="拍照"),
    ]
    if targets is not None:
        handles_traj.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
                   markeredgecolor=_C_GRAY, markersize=5, linestyle="None",
                   label="目标点")
        )
    ax_traj.legend(handles=handles_traj, loc="best", fontsize=7)

    if n_tasks == 0:
        ax_gantt.text(0.5, 0.5, "无任务", transform=ax_gantt.transAxes,
                      fontsize=10, ha="center", va="center", color=_C_GRAY)
    else:
        bar_h = 0.65

        for i, p in enumerate(parsed):
            y = n_tasks - 1 - i
            color_exec = _C_RED if p["is_shoot"] else _C_BLUE

            prep_w = p["t_exec_start"] - p["t_prep_start"]
            if prep_w > 0:
                ax_gantt.barh(
                    y, prep_w, left=p["t_prep_start"],
                    height=bar_h, color="#E5E7EB", edgecolor="#D1D5DB",
                    linewidth=0.3,
                )

            exec_w = p["t_exec_end"] - p["t_exec_start"]
            if exec_w <= 0:
                exec_w = 0.3
            ax_gantt.barh(
                y, exec_w, left=p["t_exec_start"],
                height=bar_h, color=color_exec, alpha=0.85,
                edgecolor="white", linewidth=0.3,
            )

            ax_gantt.text(
                p["t_exec_start"] + exec_w / 2, y,
                f'{p["t_exec_start"]:.1f}',
                ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold",
            )

        y_labels = [
            f"{'S' if p['is_shoot'] else 'P'}-{p['target_id']}"
            for p in reversed(parsed)
        ]
        ax_gantt.set_yticks(range(n_tasks))
        ax_gantt.set_yticklabels(y_labels, fontsize=7)
        ax_gantt.invert_yaxis()

    ax_gantt.set_xlabel("$t$ (s)")
    ax_gantt.set_title("(b) 任务调度", fontsize=10, fontweight="bold")

    handles_gantt = [
        Rectangle((0, 0), 1, 1, facecolor="#E5E7EB", edgecolor="#D1D5DB",
                  linewidth=0.3, label="准备"),
        Rectangle((0, 0), 1, 1, facecolor=_C_RED, alpha=0.85,
                  label="射击执行"),
        Rectangle((0, 0), 1, 1, facecolor=_C_BLUE, alpha=0.85,
                  label="拍照执行"),
    ]
    ax_gantt.legend(handles=handles_gantt, loc="lower right", fontsize=7)

    fig.suptitle(
        "问题 4：任务规划与优化",
        fontsize=12, fontweight="bold", y=0.97,
    )

    _save_multi_format(fig, base_path, formats)
    plt.close(fig)
    plt.rcParams.update(saved_rc)
    print(f"[task_planning_paper] 问题 4 双栏图已保存。")
