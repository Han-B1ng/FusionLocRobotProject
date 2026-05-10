# file: visualization/plot_sensitivity_analysis.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import plot_config, PLOT_DIR
plot_config.apply_style()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_sensitivity_single(
    param_name: str,
    param_values: np.ndarray,
    metrics: Dict[str, np.ndarray],
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "sensitivity_single.png"),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "指标值",
    baseline_value: Optional[float] = None,
    baseline_label: str = "默认值",
    highlight_best: Dict[str, str] = None,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if title is None:
        title = f"参数敏感性分析 — {param_name}"
    if xlabel is None:
        xlabel = param_name

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("metrics 不能为空")

    if n_metrics <= 2:
        fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

        ax2 = None
        if n_metrics == 2:
            ax2 = ax1.twinx()

        axes = [ax1] + ([ax2] if ax2 is not None else [])
        metric_names = list(metrics.keys())
        colors = [plot_config.COLORS[i % len(plot_config.COLORS)]
                  for i in range(n_metrics)]

        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes[min(idx, len(axes) - 1)]
            ax.plot(
                param_values, values,
                color=colors[idx], linewidth=plot_config.linewidth,
                marker="o", markersize=5, label=name,
            )
            ax.set_ylabel(name, fontsize=plot_config.label_fontsize,
                          color=colors[idx])
            ax.tick_params(axis="y", labelcolor=colors[idx])

            if highlight_best and name in highlight_best:
                direction = highlight_best[name]
                if direction == "max":
                    best_idx = int(np.argmax(values))
                else:
                    best_idx = int(np.argmin(values))
                ax.plot(
                    param_values[best_idx], values[best_idx],
                    marker="o", markersize=12, color=colors[idx],
                    markeredgecolor="black", markeredgewidth=1.5,
                    zorder=5,
                )
                ax.annotate(
                    f"最优\n{values[best_idx]:.2f}",
                    (param_values[best_idx], values[best_idx]),
                    textcoords="offset points", xytext=(12, 5),
                    fontsize=plot_config.tick_fontsize,
                    fontweight="bold", color=colors[idx],
                )

        ax1.set_xlabel(xlabel, fontsize=plot_config.label_fontsize)
        ax1.set_title(title, fontsize=plot_config.title_fontsize,
                      fontweight="bold")

    else:
        n_cols = 2
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes_grid = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            constrained_layout=True,
        )
        axes_flat = np.array(axes_grid).flatten()
        metric_names = list(metrics.keys())
        colors = [plot_config.COLORS[i % len(plot_config.COLORS)]
                  for i in range(n_metrics)]

        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes_flat[idx]
            ax.plot(
                param_values, values,
                color=colors[idx], linewidth=plot_config.linewidth,
                marker="o", markersize=4, label=name,
            )
            ax.set_ylabel(name, fontsize=plot_config.label_fontsize)
            ax.set_title(name, fontsize=plot_config.tick_fontsize,
                         fontweight="bold")

            if highlight_best and name in highlight_best:
                direction = highlight_best[name]
                if direction == "max":
                    best_idx = int(np.argmax(values))
                else:
                    best_idx = int(np.argmin(values))
                ax.plot(
                    param_values[best_idx], values[best_idx],
                    marker="o", markersize=10, color=colors[idx],
                    markeredgecolor="black", markeredgewidth=1.5,
                    zorder=5,
                )

            ax.set_xlabel(xlabel, fontsize=plot_config.tick_fontsize)

        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=plot_config.title_fontsize,
                     fontweight="bold", y=1.02)
        ax1 = axes_flat[0]

    if baseline_value is not None:
        ax1.axvline(
            baseline_value, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7, label=baseline_label,
        )

    ax1.legend(
        loc="best", fontsize=plot_config.legend_fontsize,
        frameon=plot_config.legend_frameon,
    )

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sensitivity_single] 已保存: {save_path}")


def plot_sensitivity_heatmap(
    param1_name: str,
    param1_values: np.ndarray,
    param2_name: str,
    param2_values: np.ndarray,
    metric_matrix: np.ndarray,
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "sensitivity_heatmap.png"),
    title: Optional[str] = None,
    metric_name: str = "指标值",
    cmap_name: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".1f",
    baseline: Optional[Tuple[float, float]] = None,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if title is None:
        title = f"双参数敏感性热力图 — {metric_name}"

    n_rows, n_cols = metric_matrix.shape

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.8),
                                    max(5, n_rows * 0.6)),
                           constrained_layout=True)

    cmap = plt.cm.get_cmap(cmap_name)
    im = ax.imshow(
        metric_matrix, cmap=cmap, aspect="auto",
        origin="lower",
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"{v:.2g}" if isinstance(v, (int, float)) else str(v)
         for v in param1_values],
        fontsize=plot_config.tick_fontsize,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{v:.2g}" if isinstance(v, (int, float)) else str(v)
         for v in param2_values],
        fontsize=plot_config.tick_fontsize,
    )

    ax.set_xlabel(param1_name, fontsize=plot_config.label_fontsize)
    ax.set_ylabel(param2_name, fontsize=plot_config.label_fontsize)
    ax.set_title(title, fontsize=plot_config.title_fontsize, fontweight="bold")

    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                val = metric_matrix[i, j]
                norm_val = (val - metric_matrix.min()) / (
                    metric_matrix.max() - metric_matrix.min() + 1e-12
                )
                text_color = "white" if norm_val > 0.6 else "black"
                ax.text(
                    j, i, f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=plot_config.tick_fontsize - 1,
                    color=text_color, fontweight="bold",
                )

    best_idx = np.unravel_index(np.argmax(metric_matrix), metric_matrix.shape)
    ax.plot(
        best_idx[1], best_idx[0], marker="*", markersize=16,
        color="white", markeredgecolor="black", markeredgewidth=1.0,
        zorder=5,
    )

    if baseline is not None:
        b1, b2 = baseline
        b1_idx = int(np.argmin(np.abs(param1_values - b1)))
        b2_idx = int(np.argmin(np.abs(param2_values - b2)))
        ax.plot(
            b1_idx, b2_idx, marker="+", markersize=14,
            color="white", markeredgewidth=2.0, zorder=5,
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_name, fontsize=plot_config.label_fontsize)
    cbar.ax.tick_params(labelsize=plot_config.tick_fontsize)

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sensitivity_heatmap] 已保存: {save_path}")


def plot_tradeoff_curve(
    x_metric: np.ndarray,
    y_metric: np.ndarray,
    param_values: np.ndarray,
    x_label: str = "指标 X",
    y_label: str = "指标 Y",
    save_path: Union[str, Path] = os.path.join(PLOT_DIR, "tradeoff_curve.png"),
    title: str = "多目标 Trade-off 曲线",
    param_label: str = "参数值",
    pareto_highlight: bool = True,
    baseline_idx: Optional[int] = None,
    annotate_extremes: bool = True,
) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    scatter = ax.scatter(
        x_metric, y_metric,
        c=param_values, cmap="viridis",
        s=80, edgecolors="white", linewidths=0.8,
        zorder=3,
    )

    sorted_idx = np.argsort(param_values)
    ax.plot(
        x_metric[sorted_idx], y_metric[sorted_idx],
        color="gray", linewidth=0.6, alpha=0.5, linestyle="-",
        zorder=2,
    )

    if pareto_highlight:
        pareto_mask = np.zeros(len(x_metric), dtype=bool)
        for i in range(len(x_metric)):
            dominated = False
            for j in range(len(x_metric)):
                if i == j:
                    continue
                if (x_metric[j] >= x_metric[i]
                        and y_metric[j] >= y_metric[i]
                        and (x_metric[j] > x_metric[i]
                             or y_metric[j] > y_metric[i])):
                    dominated = True
                    break
            pareto_mask[i] = not dominated

        if np.any(pareto_mask):
            ax.scatter(
                x_metric[pareto_mask], y_metric[pareto_mask],
                facecolors="none", edgecolors="black",
                linewidths=2.0, s=180, zorder=4,
                label="帕累托前沿",
            )

    if baseline_idx is not None:
        ax.plot(
            x_metric[baseline_idx], y_metric[baseline_idx],
            marker="o", markersize=14,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=2.0, zorder=5, label="默认参数",
        )
        ax.annotate(
            "默认",
            (x_metric[baseline_idx], y_metric[baseline_idx]),
            textcoords="offset points", xytext=(10, 8),
            fontsize=plot_config.tick_fontsize,
            fontweight="bold",
        )

    if annotate_extremes:
        idx_xmax = int(np.argmax(x_metric))
        ax.annotate(
            f"X最优\n{x_metric[idx_xmax]:.2f}",
            (x_metric[idx_xmax], y_metric[idx_xmax]),
            textcoords="offset points", xytext=(12, -8),
            fontsize=plot_config.tick_fontsize - 1,
            fontweight="bold", color=plot_config.COLORS[3],
        )
        idx_ymax = int(np.argmax(y_metric))
        ax.annotate(
            f"Y最优\n{y_metric[idx_ymax]:.2f}",
            (x_metric[idx_ymax], y_metric[idx_ymax]),
            textcoords="offset points", xytext=(12, 8),
            fontsize=plot_config.tick_fontsize - 1,
            fontweight="bold", color=plot_config.COLORS[1],
        )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(param_label, fontsize=plot_config.label_fontsize)
    cbar.ax.tick_params(labelsize=plot_config.tick_fontsize)

    ax.set_xlabel(x_label, fontsize=plot_config.label_fontsize)
    ax.set_ylabel(y_label, fontsize=plot_config.label_fontsize)
    ax.set_title(title, fontsize=plot_config.title_fontsize, fontweight="bold")
    ax.legend(
        loc="best", fontsize=plot_config.legend_fontsize,
        frameon=plot_config.legend_frameon,
    )

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_tradeoff_curve] 已保存: {save_path}")


if __name__ == "__main__":
    out = Path(PLOT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    param_vals = np.arange(0.1, 1.1, 0.1)
    task_counts = np.array([20, 28, 35, 42, 48, 50, 49, 47, 45, 43])
    coverage = np.array([0.40, 0.55, 0.70, 0.85, 0.92, 0.95,
                         0.94, 0.93, 0.91, 0.90])

    plot_sensitivity_single(
        param_name="MIN_TASK_SEP",
        param_values=param_vals,
        metrics={
            "任务总数": task_counts,
            "目标覆盖率": coverage,
        },
        save_path=out / "sensitivity_single_demo.png",
        baseline_value=0.5,
        highlight_best={"任务总数": "max", "目标覆盖率": "max"},
    )

    p1 = np.array([10, 20, 30, 40, 50])
    p2 = np.array([0.5, 1.0, 1.5, 2.0])
    metric = np.array([
        [60, 70, 75, 72, 68],
        [65, 80, 88, 85, 80],
        [55, 75, 90, 87, 82],
        [45, 65, 82, 88, 85],
    ])

    plot_sensitivity_heatmap(
        param1_name="MIN_HEADING_DIFF (°)",
        param1_values=p1,
        param2_name="MIN_TASK_SEP (s)",
        param2_values=p2,
        metric_matrix=metric,
        save_path=out / "sensitivity_heatmap_demo.png",
        metric_name="任务总数",
        baseline=(30, 1.5),
    )

    coverage_arr = np.array([0.50, 0.65, 0.78, 0.85, 0.90,
                             0.93, 0.95, 0.96, 0.97, 0.97])
    time_arr = np.array([5, 12, 25, 45, 80, 130, 200, 300, 420, 580])
    sep_vals = np.array([2.0, 1.5, 1.0, 0.8, 0.6,
                         0.5, 0.4, 0.3, 0.2, 0.1])

    plot_tradeoff_curve(
        x_metric=coverage_arr,
        y_metric=time_arr,
        param_values=sep_vals,
        x_label="目标覆盖率",
        y_label="求解时间 (s)",
        save_path=out / "tradeoff_curve_demo.png",
        title="覆盖率 vs 求解时间 Trade-off",
        param_label="MIN_TASK_SEP (s)",
        baseline_idx=5,
    )

    print("\n[plot_sensitivity_analysis] 自检完成。")
