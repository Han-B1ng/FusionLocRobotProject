# file: visualization/plot_sensitivity_analysis.py
# @Description : 参数敏感性分析可视化模块
# 依赖：matplotlib, numpy, pathlib
# 上游：stage4_problem4.py 的调度函数（ILP / 贪心），config.py 约束参数
# 下游：被 main.py 或 stage4 脚本调用

"""
visualization/plot_sensitivity_analysis.py
===========================================
提供参数敏感性分析可视化函数：
  1. plot_sensitivity_single   — 单参数扫描（横轴参数值，纵轴指标）
  2. plot_sensitivity_heatmap  — 双参数扫描热力图
  3. plot_tradeoff_curve       — 多目标 trade-off 曲线
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import plot_config
# 应用中文字体配置
plot_config.apply_style()


# ============================================================
#  辅助
# ============================================================
def _ensure_parent(path: Path) -> None:
    """创建保存路径的父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
#  1. 单参数扫描折线图
# ============================================================
def plot_sensitivity_single(
    param_name: str,
    param_values: np.ndarray,
    metrics: Dict[str, np.ndarray],
    save_path: Union[str, Path] = "output/figures/sensitivity_single.png",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "指标值",
    baseline_value: Optional[float] = None,
    baseline_label: str = "默认值",
    highlight_best: Dict[str, str] = None,
) -> None:
    """绘制单参数敏感性扫描折线图。

    横轴为参数值，纵轴为多个指标，每条线代表一个指标。

    Parameters
    ----------
    param_name : str
        被扫描的参数名称，用于轴标签和标题。
    param_values : np.ndarray
        参数取值序列。
    metrics : dict[str, np.ndarray]
        指标字典，键为指标名，值为与 param_values 等长的数组。
        示例::

            {
                "任务总数": np.array([30, 42, 45, 43, 40]),
                "目标覆盖率": np.array([0.6, 0.85, 0.92, 0.90, 0.88]),
            }

    save_path : str 或 Path
        图片保存路径。
    title : str, optional
        图表标题。若为 None 则自动生成。
    xlabel : str, optional
        横轴标签。若为 None 则使用 param_name。
    ylabel : str
        纵轴标签。
    baseline_value : float, optional
        默认参数值，在图上以垂直虚线标注。
    baseline_label : str
        基线标注文字。
    highlight_best : dict[str, str], optional
        每个指标的最优方向，``"max"`` 或 ``"min"``。
        若为 None 则不标注最优点。

    Notes
    -----
    - 每条线用不同颜色区分，颜色取自 ``plot_config.COLORS``。
    - 基线位置以灰色垂直虚线标注。
    - 最优点以大号实心圆标注。
    - 多指标时自动使用右侧第二纵轴（共享横轴）。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if title is None:
        title = f"参数敏感性分析 — {param_name}"
    if xlabel is None:
        xlabel = param_name

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("metrics 不能为空")

    # ---- 单指标或双指标用共享轴，多指标用多轴 ----
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

            # 标注最优点
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
        # 多指标：子图网格
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

        # 隐藏多余子图
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # 最上方加总标题
        fig.suptitle(title, fontsize=plot_config.title_fontsize,
                     fontweight="bold", y=1.02)
        ax1 = axes_flat[0]

    # ---- 基线标注 ----
    if baseline_value is not None:
        ax1.axvline(
            baseline_value, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7, label=baseline_label,
        )

    # ---- 图例 ----
    ax1.legend(
        loc="best", fontsize=plot_config.legend_fontsize,
        frameon=plot_config.legend_frameon,
    )

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sensitivity_single] 已保存: {save_path}")


# ============================================================
#  2. 双参数扫描热力图
# ============================================================
def plot_sensitivity_heatmap(
    param1_name: str,
    param1_values: np.ndarray,
    param2_name: str,
    param2_values: np.ndarray,
    metric_matrix: np.ndarray,
    save_path: Union[str, Path] = "output/figures/sensitivity_heatmap.png",
    title: Optional[str] = None,
    metric_name: str = "指标值",
    cmap_name: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".1f",
    baseline: Optional[Tuple[float, float]] = None,
) -> None:
    """绘制双参数扫描热力图。

    横轴为参数 1，纵轴为参数 2，颜色为指标值。

    Parameters
    ----------
    param1_name : str
        横轴参数名。
    param1_values : np.ndarray
        横轴参数取值序列。
    param2_name : str
        纵轴参数名。
    param2_values : np.ndarray
        纵轴参数取值序列。
    metric_matrix : np.ndarray
        shape ``(len(param2_values), len(param1_values))`` 的指标矩阵。
        ``metric_matrix[i, j]`` 对应 ``param2_values[i]`` 和
        ``param1_values[j]`` 的组合。
    save_path : str 或 Path
        图片保存路径。
    title : str, optional
        图表标题。
    metric_name : str
        色标标签。
    cmap_name : str
        matplotlib 色图名称。
    annot : bool
        是否在格子中标注数值。
    fmt : str
        数值格式字符串。
    baseline : tuple[float, float], optional
        默认参数值 ``(param1_default, param2_default)``，
        在图上以白色十字标注。

    Notes
    -----
    - 使用 ``imshow`` + colorbar 展示双参数交互效应。
    - 最优组合以星号高亮标注。
    """
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

    # ---- 坐标轴刻度 ----
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

    # ---- 格子内数值标注 ----
    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                val = metric_matrix[i, j]
                # 根据背景亮度选择文字颜色
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

    # ---- 最优点标注 ----
    best_idx = np.unravel_index(np.argmax(metric_matrix), metric_matrix.shape)
    ax.plot(
        best_idx[1], best_idx[0], marker="*", markersize=16,
        color="white", markeredgecolor="black", markeredgewidth=1.0,
        zorder=5,
    )

    # ---- 基线标注 ----
    if baseline is not None:
        b1, b2 = baseline
        b1_idx = int(np.argmin(np.abs(param1_values - b1)))
        b2_idx = int(np.argmin(np.abs(param2_values - b2)))
        ax.plot(
            b1_idx, b2_idx, marker="+", markersize=14,
            color="white", markeredgewidth=2.0, zorder=5,
        )

    # ---- 色标 ----
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_name, fontsize=plot_config.label_fontsize)
    cbar.ax.tick_params(labelsize=plot_config.tick_fontsize)

    fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sensitivity_heatmap] 已保存: {save_path}")


# ============================================================
#  3. 多目标 trade-off 曲线
# ============================================================
def plot_tradeoff_curve(
    x_metric: np.ndarray,
    y_metric: np.ndarray,
    param_values: np.ndarray,
    x_label: str = "指标 X",
    y_label: str = "指标 Y",
    save_path: Union[str, Path] = "output/figures/tradeoff_curve.png",
    title: str = "多目标 Trade-off 曲线",
    param_label: str = "参数值",
    pareto_highlight: bool = True,
    baseline_idx: Optional[int] = None,
    annotate_extremes: bool = True,
) -> None:
    """绘制多目标 trade-off 曲线（帕累托前沿可视化）。

    横轴和纵轴分别代表两个需要权衡的指标（如覆盖率 vs 计算时间），
    散点颜色或大小编码第三个参数值。

    Parameters
    ----------
    x_metric, y_metric : np.ndarray
        两个指标的取值序列，长度相同。
    param_values : np.ndarray
        对应的参数值序列，用于颜色编码。
    x_label, y_label : str
        轴标签。
    save_path : str 或 Path
        图片保存路径。
    title : str
        图表标题。
    param_label : str
        颜色色标标签。
    pareto_highlight : bool
        是否高亮帕累托前沿点。
    baseline_idx : int, optional
        默认参数对应的索引，在图上以特殊标记标注。
    annotate_extremes : bool
        是否标注两个轴上的极端点。

    Notes
    -----
    - 散点大小固定，颜色由 ``param_values`` 映射。
    - 帕累托前沿点以黑色边框高亮。
    - 基线点以大号白色圆标注。
    """
    save_path = Path(save_path)
    _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # ---- 散点 ----
    scatter = ax.scatter(
        x_metric, y_metric,
        c=param_values, cmap="viridis",
        s=80, edgecolors="white", linewidths=0.8,
        zorder=3,
    )

    # ---- 连线（按参数值排序）----
    sorted_idx = np.argsort(param_values)
    ax.plot(
        x_metric[sorted_idx], y_metric[sorted_idx],
        color="gray", linewidth=0.6, alpha=0.5, linestyle="-",
        zorder=2,
    )

    # ---- 帕累托前沿 ----
    if pareto_highlight:
        # 帕累托：不存在另一个点同时在 x 和 y 上更优
        # 假设两个指标都是越大越好
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

    # ---- 基线标注 ----
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

    # ---- 极端点标注 ----
    if annotate_extremes:
        # x 最大
        idx_xmax = int(np.argmax(x_metric))
        ax.annotate(
            f"X最优\n{x_metric[idx_xmax]:.2f}",
            (x_metric[idx_xmax], y_metric[idx_xmax]),
            textcoords="offset points", xytext=(12, -8),
            fontsize=plot_config.tick_fontsize - 1,
            fontweight="bold", color=plot_config.COLORS[3],
        )
        # y 最大
        idx_ymax = int(np.argmax(y_metric))
        ax.annotate(
            f"Y最优\n{y_metric[idx_ymax]:.2f}",
            (x_metric[idx_ymax], y_metric[idx_ymax]),
            textcoords="offset points", xytext=(12, 8),
            fontsize=plot_config.tick_fontsize - 1,
            fontweight="bold", color=plot_config.COLORS[1],
        )

    # ---- 色标 ----
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(param_label, fontsize=plot_config.label_fontsize)
    cbar.ax.tick_params(labelsize=plot_config.tick_fontsize)

    # ---- 样式 ----
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


# ============================================================
#  自检
# ============================================================
if __name__ == "__main__":
    out = Path("output/figures")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 示例 1：单参数扫描 ----
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

    # ---- 示例 2：双参数热力图 ----
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

    # ---- 示例 3：trade-off 曲线 ----
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
