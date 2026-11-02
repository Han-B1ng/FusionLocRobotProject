# file: stage0_eda.py

import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import time_config, data_path, PLOT_DIR, INTERMEDIATE_DIR, ensure_dirs

import matplotlib.pyplot as plt
plot_config = None
try:
    from config import plot_config
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plot_config.apply_style()
except ImportError:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


_COLOR_S1 = "#2563EB"   # 方式1 — 蓝
_COLOR_S2 = "#DC2626"   # 方式2 — 红

SENSOR_SHEETS = {
    "方式1(4Hz)": "方式1",
    "方式2(5Hz)": "方式2",
}

ATTACHMENT_IDS = (1, 2, 3)

_PATH_ATTR = {1: "path1", 2: "path2", 3: "path3"}


def _resolve_file_path(config_path: Path) -> Optional[Path]:
    if config_path.exists():
        return config_path
    for ext in (".xlsx", ".xls", ".csv"):
        alt = config_path.with_suffix(ext)
        if alt.exists():
            warnings.warn(
                f"[resolve] config 指定 '{config_path.name}'，"
                f"实际找到 '{alt.name}'，已自动切换。"
            )
            return alt
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_aliases = {
        "时间(s)": "t", "时间": "t",
        "Time": "t", "time": "t", "t": "t",
        "X坐标(m)": "x", "X坐标": "x",
        "X": "x", "x": "x",
        "Y坐标(m)": "y", "Y坐标": "y",
        "Y": "y", "y": "y",
    }
    df = df.rename(columns=col_aliases)

    if list(df.columns[:3]) == [0, 1, 2]:
        new_cols = ["t", "x", "y"] + list(df.columns[3:])
        df.columns = new_cols

    required = ["t", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"缺少必要列 {missing}，实际列: {list(df.columns)}"
        )

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _read_excel_source(file_path: Path) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for sheet_name, sensor_label in SENSOR_SHEETS.items():
        try:
            df = pd.read_excel(
                file_path, sheet_name=sheet_name, engine="openpyxl"
            )
            df = _normalize_columns(df)
            result[sensor_label] = df
        except ValueError:
            warnings.warn(
                f"[read] {file_path.name} 中未找到 sheet '{sheet_name}'，跳过。"
            )
        except Exception as exc:
            warnings.warn(
                f"[read] {file_path.name} / {sheet_name} 读取失败: {exc}"
            )
    return result


def _read_csv_source(file_path: Path) -> Dict[str, pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        warnings.warn(f"[read] CSV 读取失败 {file_path.name}: {exc}")
        return {}

    sensor_col = None
    for candidate in ("sensor", "传感器", "type", "类型", "source"):
        if candidate in df.columns:
            sensor_col = candidate
            break

    if sensor_col:
        result: Dict[str, pd.DataFrame] = {}
        for sensor_val, group_df in df.groupby(sensor_col):
            result[str(sensor_val)] = _normalize_columns(group_df.copy())
        return result
    else:
        return {"数据": _normalize_columns(df)}


def load_all_data() -> Dict[Tuple[str, str], pd.DataFrame]:
    data_dict: Dict[Tuple[str, str], pd.DataFrame] = {}

    for att_id in ATTACHMENT_IDS:
        att_name = f"附件{att_id}"
        config_path: Path = getattr(data_path, _PATH_ATTR[att_id])

        file_path = _resolve_file_path(config_path)
        if file_path is None:
            warnings.warn(
                f"[load] 文件不存在: {config_path}（及同名 .xlsx/.csv），跳过。"
            )
            continue

        suffix = file_path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            sensor_data = _read_excel_source(file_path)
        else:
            sensor_data = _read_csv_source(file_path)

        for sensor_label, df in sensor_data.items():
            data_dict[(att_name, sensor_label)] = df

    print(f"[load_all_data] 共加载 {len(data_dict)} 个数据集。")
    return data_dict


def check_data_quality(
    data_dict: Dict[Tuple[str, str], pd.DataFrame],
) -> None:
    print("\n" + "=" * 70)
    print("  数据质量检查报告")
    print("=" * 70)

    for (att_name, sensor_label), df in data_dict.items():
        tag = f"{att_name} / {sensor_label}"
        t = df["t"].values
        x = df["x"].values
        y = df["y"].values

        print(f"  记录数: {len(df)}")

        dt = np.diff(t)
        n_non_mono = int(np.sum(dt <= 0))
        if n_non_mono > 0:
            print(f"  ⚠ 时间非单调递增点数: {n_non_mono}")
        else:
            print("  ✓ 时间单调递增")

        dt_pos = dt[dt > 0]
        if len(dt_pos) > 0:
            dt_mean = float(np.mean(dt_pos))
            dt_std = float(np.std(dt_pos))
            print(f"  平均采样间隔: {dt_mean:.4f} s  (std={dt_std:.4f} s)")
        else:
            print("  ⚠ 无法计算采样间隔（数据不足或全部重复）")

        for coord_name, coord_arr in [("X", x), ("Y", y)]:
            diff = np.abs(np.diff(coord_arr))
            if len(diff) < 2:
                continue
            mu = np.mean(diff)
            sigma = np.std(diff)
            threshold = mu + 3.0 * sigma
            outlier_idx = np.where(diff > threshold)[0]
            n_outliers = len(outlier_idx)
            if n_outliers > 0:
                print(
                    f"  ⚠ {coord_name} 方向突跳异常点: {n_outliers} 个"
                    f"  (阈值={threshold:.3f} m)"
                )
            else:
                print(f"  ✓ {coord_name} 方向无明显突跳")

    print("\n" + "=" * 70 + "\n")


def plot_raw_data(
    data_dict: Dict[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    figures_dir = Path(PLOT_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for att_id in ATTACHMENT_IDS:
        att_name = f"附件{att_id}"

        sensor_frames = {
            label: df
            for (aname, label), df in data_dict.items()
            if aname == att_name
        }

        if not sensor_frames:
            warnings.warn(f"[plot] {att_name} 无数据，跳过绘图。")
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for sensor_label, df in sensor_frames.items():
            color = _COLOR_S1 if "1" in sensor_label else _COLOR_S2
            axes[0].plot(
                df["t"], df["x"],
                color=color, linewidth=0.6, alpha=0.85,
                label=sensor_label,
            )
            axes[1].plot(
                df["t"], df["y"],
                color=color, linewidth=0.6, alpha=0.85,
                label=sensor_label,
            )

        axes[0].set_ylabel("X (m)", fontsize=12)
        axes[0].set_title(
            f"{att_name} — 原始轨迹", fontsize=14, fontweight="bold",
        )
        axes[0].legend(loc="upper right", fontsize=10)

        axes[1].set_ylabel("Y (m)", fontsize=12)
        axes[1].set_xlabel("Time (s)", fontsize=12)
        axes[1].legend(loc="upper right", fontsize=10)

        fig.tight_layout()
        save_path = figures_dir / f"{att_name}_raw.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] 已保存: {save_path}")


def save_cleaned_data(
    data_dict: Dict[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = save_path.stat().st_size / 1024
    print(f"[save] 已保存: {save_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    ensure_dirs()
    output_dir = data_path.output_dir
    figures_dir = Path(PLOT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data()

    if not data:
        print("[stage0] 未加载到任何数据，请检查 data/ 目录下的附件文件。")
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("  各数据集基本信息")
    print("=" * 70)
    for (att_name, sensor_label), df in data.items():
        t_start = df["t"].iloc[0]
        t_end = df["t"].iloc[-1]
        n_points = len(df)
        n_missing = int(df.isnull().sum().sum())
        print(
            f"  {att_name} / {sensor_label:8s} | "
            f"时间: [{t_start:.2f}, {t_end:.2f}] s | "
            f"点数: {n_points:>6d} | "
            f"缺失值: {n_missing}"
        )
    print("=" * 70)

    check_data_quality(data)

    plot_raw_data(data, output_dir)

    save_cleaned_data(data, output_dir)

    print("\n[stage0] EDA 全部完成。")
