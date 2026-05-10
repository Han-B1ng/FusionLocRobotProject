# file: visualization/plot_eda.py


from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

from config import plot_config, PLOT_DIR
plot_config.apply_style()


_COLORS: List[str] = [
    "#2563EB",  # 蓝 — 传感器 1
    "#DC2626",  # 红 — 传感器 2
    "#059669",  # 绿
    "#D97706",  # 橙
    "#7C3AED",  # 紫
    "#DB2777",  # 粉
]

_DPI: int = 180


def plot_time_series(
    t: Union[np.ndarray, Sequence[np.ndarray]],
    x: Union[np.ndarray, Sequence[np.ndarray]],
    y: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Union[str, Sequence[str]],
    title: str,
    save_path: Union[str, Path],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_list(arr: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            return [arr]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
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

    while len(label_list) < n_series:
        label_list.append(f"Series {len(label_list) + 1}")

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


def plot_sampling_interval_histogram(
    t: np.ndarray,
    save_path: Union[str, Path],
    expected_dt: Optional[float] = None,
    title: str = "采样间隔分布",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dt = np.diff(t)
    dt_pos = dt[dt > 0]

    if len(dt_pos) == 0:
        warnings.warn(
            "[plot_sampling_interval_histogram] 无有效正间隔数据，跳过绘图。"
        )
        return

    dt_mean = float(np.mean(dt_pos))
    dt_std = float(np.std(dt_pos))
    dt_median = float(np.median(dt_pos))
    dt_min = float(np.min(dt_pos))
    dt_max = float(np.max(dt_pos))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    n_bins = max(30, int(np.ceil(np.log2(len(dt_pos)) + 1)))  # Sturges
    counts, bin_edges, patches = ax.hist(
        dt_pos, bins=n_bins,
        color="#2563EB", edgecolor="white", linewidth=0.5, alpha=0.85,
        label="采样间隔",
    )

    if expected_dt is not None:
        ax.axvline(
            expected_dt, color="#DC2626", linestyle="--", linewidth=1.5,
            label=f"理论间隔 = {expected_dt:.4f} s",
        )

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

    ax.set_xlabel("采样间隔 (s)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sampling_interval_histogram] 已保存: {save_path}")


def plot_missing_summary(
    missing_df: pd.DataFrame,
    save_path: Union[str, Path],
    title: str = "各数据集缺失值统计",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if missing_df.empty:
        warnings.warn("[plot_missing_summary] missing_df 为空，跳过绘图。")
        return

    datasets = list(missing_df.index.astype(str))
    columns = list(missing_df.columns.astype(str))
    n_datasets = len(datasets)
    n_cols = len(columns)

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
        for bar_rect, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + max(total_missing * 0.01, 0.3),
                    f"{int(val)}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("缺失记录数", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9, title="字段")

    if total_missing == 0:
        ax.text(
            0.5, 0.5, "所有数据集无缺失值",
            transform=ax.transAxes,
            fontsize=14, ha="center", va="center",
            color="#059669", fontweight="bold",
            alpha=0.6,
        )

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_missing_summary] 已保存: {save_path}")
