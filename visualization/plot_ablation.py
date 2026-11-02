# file: visualization/plot_ablation.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import plot_config, PLOT_DIR, TABLE_DIR

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plot_config.apply_style()

_ABLATION_COLORS = ["#DC2626", "#F59E0B", "#2563EB", "#16A34A"]
_BAR_EDGE = "#FFFFFF"
_VALUE_COLOR = "#1F2937"
_DROP_ARROW_COLOR = "#6B7280"

_DPI = 300


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_ablation_bars(
    df: Optional[pd.DataFrame] = None,
    xlsx_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    title: str = "消融实验 RMSE 递进",
    figsize: tuple = (10, 6),
) -> Path:
    if xlsx_path is None:
        xlsx_path = Path(TABLE_DIR) / "ablation.xlsx"
    if save_path is None:
        save_path = Path(PLOT_DIR) / "ablation_rmse.png"
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if df is None:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"消融实验数据文件不存在: {xlsx_path}")
        df = pd.read_excel(xlsx_path, engine="openpyxl")

    configs = df["配置"].tolist()
    rmse_vals = df["RMSE (m)"].values.astype(float)
    n = len(configs)

    short_labels = [
        "基线",
        "+小波去噪",
        "+AR1建模",
        "完整方案",
    ][:n]

    drops_pct = []
    for i in range(1, n):
        pct = (rmse_vals[i - 1] - rmse_vals[i]) / rmse_vals[i - 1] * 100
        drops_pct.append(pct)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    x = np.arange(n)
    colors = _ABLATION_COLORS[:n]

    bars = ax.bar(
        x, rmse_vals,
        width=0.55,
        color=colors,
        edgecolor=_BAR_EDGE,
        linewidth=1.2,
        alpha=0.92,
        zorder=3,
    )

    for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + rmse_vals.max() * 0.02,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=_VALUE_COLOR,
        )

    for i in range(n - 1):
        mid_x = (x[i] + x[i + 1]) / 2
        y_top = max(rmse_vals[i], rmse_vals[i + 1]) + rmse_vals.max() * 0.12
        ax.annotate(
            "",
            xy=(x[i + 1] - 0.15, y_top),
            xytext=(x[i] + 0.15, y_top),
            arrowprops=dict(
                arrowstyle="->", color=_DROP_ARROW_COLOR,
                lw=1.5, connectionstyle="arc3,rad=-0.05",
            ),
        )
        ax.text(
            mid_x, y_top + rmse_vals.max() * 0.02,
            f"↓ {drops_pct[i]:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color=_DROP_ARROW_COLOR, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, rmse_vals.max() * 1.35)
    ax.tick_params(axis="y", labelsize=10)

    footnote_lines = [f"  {sl}：{full}" for sl, full in zip(short_labels, configs)]
    footnote = "\n".join(footnote_lines)
    ax.text(
        0.5, -0.22, footnote,
        transform=ax.transAxes,
        fontsize=7.5, color="#6B7280", ha="center", va="top",
    )

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_bars] 已保存: {save_path}")
    return save_path


def plot_ablation_horizontal(
    df: Optional[pd.DataFrame] = None,
    xlsx_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    title: str = "消融实验 — 各组件贡献",
    figsize: tuple = (10, 5.5),
) -> Path:
    if xlsx_path is None:
        xlsx_path = Path(TABLE_DIR) / "ablation.xlsx"
    if save_path is None:
        save_path = Path(PLOT_DIR) / "ablation_rmse_horizontal.png"
    save_path = Path(save_path)
    _ensure_parent(save_path)

    if df is None:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"消融实验数据文件不存在: {xlsx_path}")
        df = pd.read_excel(xlsx_path, engine="openpyxl")

    configs = df["配置"].tolist()
    rmse_vals = df["RMSE (m)"].values.astype(float)
    n = len(configs)

    short_labels = ["基线", "+小波去噪", "+AR1建模", "完整方案"][:n]

    contributions = []
    for i in range(n):
        if i == 0:
            contributions.append(0.0)
        else:
            contributions.append(rmse_vals[0] - rmse_vals[i])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    y = np.arange(n)[::-1]
    colors = _ABLATION_COLORS[:n]

    bars = ax.barh(
        y, rmse_vals,
        height=0.5,
        color=colors,
        edgecolor=_BAR_EDGE,
        linewidth=1.0,
        alpha=0.92,
        zorder=3,
    )

    for bar, val, contrib in zip(bars, rmse_vals, contributions):
        label = f"  RMSE = {val:.4f}"
        if contrib > 0:
            label += f"  (Δ = -{contrib:.4f})"
        ax.text(
            bar.get_width() + rmse_vals.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left", va="center",
            fontsize=9.5, color=_VALUE_COLOR, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("RMSE (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, rmse_vals.max() * 1.55)
    ax.invert_yaxis()

    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_horizontal] 已保存: {save_path}")
    return save_path


if __name__ == "__main__":
    out = Path(PLOT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    xlsx = Path(TABLE_DIR) / "ablation.xlsx"

    if xlsx.exists():
        df_abl = pd.read_excel(xlsx, engine="openpyxl")
        plot_ablation_bars(df=df_abl)
        plot_ablation_horizontal(df=df_abl)
    else:
        print(f"[plot_ablation] ablation.xlsx 不存在 ({xlsx})，使用示例数据自检。")
        demo_df = pd.DataFrame({
            "配置": [
                "基线（无去噪/无AR1/默认R）",
                "+小波去噪（无AR1/默认R）",
                "+AR1偏差建模（去噪+AR1/默认R）",
                "+自适应R（完整方案）",
            ],
            "RMSE (m)": [2.15, 1.68, 1.24, 0.98],
        })
        plot_ablation_bars(df=demo_df)
        plot_ablation_horizontal(df=demo_df)

    print("\n[plot_ablation] 自检完成。")
