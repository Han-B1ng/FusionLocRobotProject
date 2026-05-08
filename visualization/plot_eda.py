# file: visualization/plot_eda.py
# @Description : 探索性数据分析（EDA）可视化模块
# 依赖：matplotlib, numpy, pathlib
# 上游：stage0_eda.py 产出的数据字典
# 下游：被 stage0_eda.py 或 main.py 调用，生成 EDA 图表

"""
visualization/plot_eda.py
=========================
提供三类 EDA 可视化函数：
  1. plot_time_series        — X / Y 坐标随时间变化的双子图
  2. plot_sampling_interval_histogram — 采样间隔分布直方图
  3. plot_missing_summary    — 各数据集缺失值柱状图
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
#  全局样式与字体
# ============================================================
# 先应用样式
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

# 再应用中文字体配置（确保不被覆盖）
from config import plot_config
plot_config.apply_style()




# 预设配色池：区分多条曲线
_COLORS: List[str] = [
    "#2563EB",  # 蓝 — 传感器 1
    "#DC2626",  # 红 — 传感器 2
    "#059669",  # 绿
    "#D97706",  # 橙
    "#7C3AED",  # 紫
    "#DB2777",  # 粉
]

_DPI: int = 180


# ============================================================
#  1. 时间序列图（X-t / Y-t 并排子图）
# ============================================================
def plot_time_series(
    t: Union[np.ndarray, Sequence[np.ndarray]],
    x: Union[np.ndarray, Sequence[np.ndarray]],
    y: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Union[str, Sequence[str]],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """绘制 X 和 Y 坐标随时间变化的并排子图。

    支持单条或多条曲线叠加显示。

    Parameters
    ----------
    t : np.ndarray 或 list[np.ndarray]
        时间轴数据。传入单个数组时视为单条曲线；
        传入列表时每个元素对应一条曲线。
    x : np.ndarray 或 list[np.ndarray]
        X 坐标序列，长度/结构与 *t* 一致。
    y : np.ndarray 或 list[np.ndarray]
        Y 坐标序列，长度/结构与 *t* 一致。
    labels : str 或 list[str]
        曲线标签。单条曲线时传字符串，多条时传列表。
    title : str
        图表总标题。
    save_path : str 或 Path
        图片保存路径（含文件名及后缀，如 ``output/figures/eda_ts.png``）。

    Notes
    -----
    - 左子图为 X-t，右子图为 Y-t。
    - 多条曲线从预设配色池中循环取色。
    - 图片以 180 DPI 保存。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 统一包装为列表 ----
    def _ensure_list(arr: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            return [arr]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            # 2D 数组按列拆分
            return [arr[:, i] for i in range(arr.shape[1])]
        return list(arr)

    t_list = _ensure_list(t)
    x_list = _ensure_list(x)
    y_list = _ensure_list(y)

    if isinstance(labels, str):
        label_list = [labels]
    else:
        label_list = list(labels)

    n_series = max(len(t_list), len(x_list), len(y_list))

    # 长度不足时自动补齐标签
    while len(label_list) < n_series:
        label_list.append(f"Series {len(label_list) + 1}")

    # ---- 绘图 ----
    fig, (ax_x, ax_y) = plt.subplots(
        1, 2, figsize=(16, 5), sharex=True, constrained_layout=True
    )

    for i in range(n_series):
        color = _COLORS[i % len(_COLORS)]
        t_i = t_list[i] if i < len(t_list) else t_list[0]
        x_i = x_list[i] if i < len(x_list) else x_list[0]
        y_i = y_list[i] if i < len(y_list) else y_list[0]

        ax_x.plot(
            t_i, x_i,
            color=color, linewidth=0.7, alpha=0.85,
            label=label_list[i],
        )
        ax_y.plot(
            t_i, y_i,
            color=color, linewidth=0.7, alpha=0.85,
            label=label_list[i],
        )

    # ---- 样式 ----
    ax_x.set_xlabel("时间 (s)", fontsize=11)
    ax_x.set_ylabel("X (m)", fontsize=11)
    ax_x.set_title("X 坐标 — 时间序列", fontsize=12, fontweight="bold")
    ax_x.legend(loc="best", fontsize=9, framealpha=0.9)

    ax_y.set_xlabel("时间 (s)", fontsize=11)
    ax_y.set_ylabel("Y (m)", fontsize=11)
    ax_y.set_title("Y 坐标 — 时间序列", fontsize=12, fontweight="bold")
    ax_y.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_time_series] 已保存: {save_path}")


# ============================================================
#  2. 采样间隔直方图
# ============================================================
def plot_sampling_interval_histogram(
    t: np.ndarray,
    save_path: Union[str, Path],
    expected_dt: Optional[float] = None,
    title: str = "采样间隔分布",
) -> None:
    """绘制相邻采样时间间隔的直方图，并标注统计摘要。

    Parameters
    ----------
    t : np.ndarray
        单条时间轴（一维），按时间升序排列。
    save_path : str 或 Path
        图片保存路径。
    expected_dt : float, optional
        理论采样间隔（秒）。若提供则在图上叠加竖线标注，
        便于与实际分布对比。
    title : str, default='采样间隔分布'
        图表标题。

    Notes
    -----
    - 直方图 bins 数量由 Sturges 规则自动确定。
    - 右上角文本框显示均值、标准差、中位数。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dt = np.diff(t)
    # 仅保留正值（排除重复时间戳或异常）
    dt_pos = dt[dt > 0]

    if len(dt_pos) == 0:
        warnings.warn(
            "[plot_sampling_interval_histogram] 无有效正间隔数据，跳过绘图。"
        )
        return

    # ---- 统计量 ----
    dt_mean = float(np.mean(dt_pos))
    dt_std = float(np.std(dt_pos))
    dt_median = float(np.median(dt_pos))
    dt_min = float(np.min(dt_pos))
    dt_max = float(np.max(dt_pos))

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # 直方图
    n_bins = max(30, int(np.ceil(np.log2(len(dt_pos)) + 1)))  # Sturges
    counts, bin_edges, patches = ax.hist(
        dt_pos, bins=n_bins,
        color="#2563EB", edgecolor="white", linewidth=0.5, alpha=0.85,
        label="采样间隔",
    )

    # 理论间隔竖线
    if expected_dt is not None:
        ax.axvline(
            expected_dt, color="#DC2626", linestyle="--", linewidth=1.5,
            label=f"理论间隔 = {expected_dt:.4f} s",
        )

    # 统计信息文本框
    stats_text = (
        f"均值   = {dt_mean:.4f} s\n"
        f"标准差 = {dt_std:.4f} s\n"
        f"中位数 = {dt_median:.4f} s\n"
        f"范围   = [{dt_min:.4f}, {dt_max:.4f}] s"
    )
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        family="monospace",
    )

    # ---- 样式 ----
    ax.set_xlabel("采样间隔 (s)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sampling_interval_histogram] 已保存: {save_path}")


# ============================================================
#  3. 缺失值汇总柱状图
# ============================================================
def plot_missing_summary(
    missing_df: pd.DataFrame,
    save_path: Union[str, Path],
    title: str = "各数据集缺失值统计",
) -> None:
    """用柱状图展示各数据集 / 各列的缺失值数量。

    Parameters
    ----------
    missing_df : pd.DataFrame
        缺失值统计表，行为数据集标识（如 ``附件1 / 方式1``），
        列为字段名（如 ``t``, ``x``, ``y``），值为缺失数量。
        可由 ``df.isnull().sum()`` 按组聚合后 ``pd.DataFrame`` 得到。

        示例::

            missing_df = pd.DataFrame({
                't': [0, 0, 0],
                'x': [2, 0, 5],
                'y': [1, 0, 3],
            }, index=['附件1/方式1', '附件1/方式2', '附件2/方式1'])

    save_path : str 或 Path
        图片保存路径。
    title : str, default='各数据集缺失值统计'
        图表标题。

    Notes
    -----
    - 若 missing_df 全为 0，仍生成图但标注"无缺失"。
    - 使用分组柱状图，每组对应一个数据集，每根柱子对应一列。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if missing_df.empty:
        warnings.warn("[plot_missing_summary] missing_df 为空，跳过绘图。")
        return

    datasets = list(missing_df.index.astype(str))
    columns = list(missing_df.columns.astype(str))
    n_datasets = len(datasets)
    n_cols = len(columns)

    # ---- 绘图 ----
    fig, ax = plt.subplots(
        figsize=(max(8, 1.8 * n_datasets), 5), constrained_layout=True
    )

    x_pos = np.arange(n_datasets)
    bar_width = 0.7 / max(n_cols, 1)

    col_colors = _COLORS[:n_cols] if n_cols <= len(_COLORS) else [
        _COLORS[i % len(_COLORS)] for i in range(n_cols)
    ]

    total_missing = int(missing_df.values.sum())

    for j, col_name in enumerate(columns):
        offsets = x_pos - 0.35 + j * bar_width + bar_width / 2
        values = missing_df[col_name].values.astype(float)
        bars = ax.bar(
            offsets, values,
            width=bar_width * 0.9,
            color=col_colors[j],
            edgecolor="white", linewidth=0.5,
            label=col_name, alpha=0.85,
        )
        # 在柱顶标注数值（仅 > 0 时显示）
        for bar_rect, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + max(total_missing * 0.01, 0.3),
                    f"{int(val)}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    # ---- 样式 ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("缺失记录数", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9, title="字段")

    # 若全部为 0，在图中央标注
    if total_missing == 0:
        ax.text(
            0.5, 0.5, "所有数据集无缺失值",
            transform=ax.transAxes,
            fontsize=14, ha="center", va="center",
            color="#059669", fontweight="bold",
            alpha=0.6,
        )

    # y 轴从 0 开始，整数刻度
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_missing_summary] 已保存: {save_path}")
