# file: config.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 全局参数配置，分组管理时间、滤波、对齐、任务约束及路径

"""
╔══════════════════════════════════════════════════════╗
║  多源融合机器人定位项目 —— 全局配置                    ║
╚══════════════════════════════════════════════════════╝

设计原则：
  · 所有配置均使用@dataclass(frozen=True)管理，不可变
  · 矩阵类参数以对角元元组存储，运行时用numpy.diag()还原
  · 分组管理：时间、滤波、对齐、任务约束、数据路径
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
import matplotlib.font_manager as fm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediate")


def ensure_dirs():
    for d in [OUTPUT_DIR, TABLE_DIR, PLOT_DIR, INTERMEDIATE_DIR]:
        os.makedirs(d, exist_ok=True)
fm._load_fontmanager(try_read_cache=False)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
#  中文字体自动检测（解决跨平台乱码）
# ============================================================
import matplotlib
import matplotlib.font_manager as fm

def _setup_chinese_font():
    """自动检测并配置中文字体，兼容 Windows / Linux / macOS。"""
    # 候选字体列表（按优先级）
    candidates = [
        # Windows
        "SimHei", "Microsoft YaHei", "FangSong", "KaiTi",
        # macOS
        "PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
        # Linux
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC",
        "Source Han Sans SC", "Source Han Sans CN",
        "Droid Sans Fallback", "AR PL UMing CN",
    ]

    # 获取系统已安装字体名集合
    installed = {f.name for f in fm.fontManager.ttflist}

    for font_name in candidates:
        if font_name in installed:
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[config] 中文字体已配置: {font_name}")
            return font_name

    # 全部未找到：尝试从系统路径直接搜索 ttf
    import os
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "C:\\Windows\\Fonts",
    ]
    keyword_map = {
        "simhei": "SimHei", "msyh": "Microsoft YaHei",
        "wqy": "WenQuanYi Micro Hei", "noto": "Noto Sans CJK SC",
        "sourcehansans": "Source Han Sans SC",
        "droid": "Droid Sans Fallback",
    }
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith((".ttf", ".ttc", ".otf")):
                    continue
                fl = f.lower()
                for keyword, display_name in keyword_map.items():
                    if keyword in fl:
                        font_path = os.path.join(root, f)
                        fm.fontManager.addfont(font_path)
                        matplotlib.rcParams["font.sans-serif"] = [display_name, "DejaVu Sans"]
                        matplotlib.rcParams["font.family"] = "sans-serif"
                        matplotlib.rcParams["axes.unicode_minus"] = False
                        print(f"[config] 中文字体已从文件加载: {font_path}")
                        return display_name

    print("[config] 警告：未找到中文字体，图表中文可能乱码。")
    print("  建议安装: sudo apt install fonts-wqy-microhei  (Linux)")
    print("  或: pip install matplotlib-font  (备选)")
    return None

_detected_font = _setup_chinese_font()

# ╔══ 时间与频率配置 ══╗
@dataclass(frozen=True)
class TimeConfig:
    """时间与频率配置：传感器采样率、目标输出率、各问题时间区间。"""

    # --- 传感器固有频率 ---
    freq1: float = 4.0          # Hz，传感器 1 采样频率
    freq2: float = 5.0          # Hz，传感器 2 采样频率

    # --- 融合输出目标频率 ---
    target_freq: float = 10.0   # Hz，融合后目标输出频率

    # --- 各问题数据的时间范围 (秒) ---
    t_range_p1: Tuple[float, float] = (221.0, 970.75)
    t_range_p2: Tuple[float, float] = (102.0, 852.0)
    t_range_p3: Tuple[float, float] = (469.0, 1268.83)

    @property
    def dt1(self) -> float:
        """传感器 1 采样周期 (秒)。"""
        return 1.0 / self.freq1

    @property
    def dt2(self) -> float:
        """传感器 2 采样周期 (秒)。"""
        return 1.0 / self.freq2

    @property
    def dt_target(self) -> float:
        """目标输出周期 (秒)。"""
        return 1.0 / self.target_freq


# ╔══ 卡尔曼滤波配置 ══╗
@dataclass(frozen=True)
class FilterConfig:
    """扩展卡尔曼滤波器（EKF）参数配置。

    状态向量：
      · 默认：x = [x, y, vx, vy]ᵀ ∈ ℝ⁴
      · 含偏差估计：x = [x, y, vx, vy, bx, by]ᵀ ∈ ℝ⁶

    矩阵存储：以对角元元组存储，运行时用numpy.diag()还原。

    自适应观测噪声R：
      · adaptive_R=True时，R1/R2从数据残差自动估计
      · fuse_sensors()通过R1_est/R2_est接收外部值
      · 若未传入则回退至R1_fixed/R2_fixed
    """

    # --- 状态维度 ---
    state_dim: int = 6

    # --- 初始状态协方差 P0 对角元 ---
    P0: Tuple[float, ...] = (
        10.0, 10.0,         # 位置初始不确定性 (m²)
        5.0,  5.0,          # 速度初始不确定性 ((m/s)²)
        1.0,  1.0,          # 偏差初始不确定性 (m²)
    )

    # --- 过程噪声协方差 Q 对角元 ---
    Q: Tuple[float, ...] = (
        0.1,  0.1,          # 位置过程噪声 (m²)
        0.5,  0.5,          # 速度过程噪声 ((m/s)²)
        0.01, 0.01,         # 偏差随机游走噪声 (m²)
    )

    # --- 固定观测噪声（默认值 / 自适应模式下的后备值）---
    R1_fixed: Tuple[float, float] = (0.5, 0.5)   # 传感器 1 观测噪声 (m²)
    R2_fixed: Tuple[float, float] = (0.3, 0.3)   # 传感器 2 观测噪声 (m²)

    # --- 是否启用系统偏差估计 ---
    estimate_bias: bool = True

    # --- 是否启用自适应观测噪声估计 ---
    adaptive_R: bool = False

    @property
    def R1(self) -> Tuple[float, float]:
        """传感器 1 观测噪声对角元（兼容旧代码）。"""
        return self.R1_fixed

    @property
    def R2(self) -> Tuple[float, float]:
        """传感器 2 观测噪声对角元（兼容旧代码）。"""
        return self.R2_fixed


# ╔══ 时间对齐配置 ══╗
@dataclass(frozen=True)
class AlignmentConfig:
    """时间对齐参数配置。"""

    corr_window: float = 1.0
    method: str = 'linear'
    delay_range: Tuple[float, float] = (0.8, 1.2)


# ╔══ 任务约束配置（射击与拍照） ══╗
@dataclass(frozen=True)
class TaskConfig:
    """任务约束配置（射击与拍照）。

    注意：所有角度单位为度(°)，计算时需转换为弧度。
    """

    # ── 射击约束 ──
    shoot_d: Tuple[float, float] = (5.0, 30.0)
    shoot_vmax: float = 2.0
    shoot_amax: float = 1.5
    shoot_prep: float = 1.5

    # ── 拍照约束 ──
    photo_d: Tuple[float, float] = (10.0, 40.0)
    photo_vmax: float = 1.5
    photo_amax: float = 1.5
    photo_angle_min: float = 60.0
    photo_prep: float = 0.5


# ╔══ 数据路径配置 ══╗
@dataclass(frozen=True)
class DataPath:
    """数据路径配置：输入文件与输出目录（pathlib.Path）。"""

    data_dir: Path = Path('data')
    output_dir: Path = Path('output')

    file1: str = '附件1.xlsx'
    file2: str = '附件2.xlsx'
    file3: str = '附件3.xlsx'
    file4 = os.path.join(data_dir, "附件4.xlsx")

    @property
    def path1(self) -> Path:
        return self.data_dir / self.file1

    @property
    def path2(self) -> Path:
        return self.data_dir / self.file2

    @property
    def path3(self) -> Path:
        return self.data_dir / self.file3

    @property
    def path4(self) -> Path:
        return self.data_dir / self.file4


# ╔══ 全局配置实例 ══╗
time_config = TimeConfig()
filter_config = FilterConfig()
alignment_config = AlignmentConfig()

# ╔══ 可视化样式配置 ══╗
@dataclass(frozen=True)
class PlotConfig:
    """可视化全局样式配置：配色、字体、线宽、DPI、坐标轴。

    配色方案采用 Nature 色盲友好调色板 (Wong, 2011, Nat Methods)。
    """

    # --- Nature 色盲友好配色 (Wong 2011) ---
    COLORS: Tuple[str, ...] = (
        '#000000',   # 黑      — 原始轨迹 / 基线
        '#E69F00',   # 橙      — 传感器 1
        '#56B4E9',   # 天蓝    — 传感器 2
        '#009E73',   # 蓝绿    — 融合轨迹
        '#F0E442',   # 黄      — 射击窗口
        '#0072B2',   # 蓝      — 拍照窗口
        '#D55E00',   # 朱红    — 冲突 / 丢弃
        '#CC79A7',   # 粉紫    — 备用
    )

    # --- 字体 ---
    font_family: str = 'sans-serif'
    font_cjk: str = 'SimHei'

    # --- 线宽 ---
    linewidth: float = 2.0
    linewidth_thin: float = 1.0
    linewidth_thick: float = 3.0

    # --- 分辨率 ---
    dpi: int = 300
    dpi_high: int = 600

    # --- 坐标轴 ---
    remove_top_right: bool = True
    tick_direction: str = 'in'
    tick_length: float = 4.0
    tick_width: float = 1.0

    # --- 字号 ---
    title_fontsize: float = 14
    label_fontsize: float = 12
    tick_fontsize: float = 10
    legend_fontsize: float = 10
    legend_frameon: bool = False

    @property
    def rcParams(self) -> dict:
        """返回 matplotlib.rcParams 字典，供 apply_style() 使用。"""
        return {
            'font.family': self.font_family,
            'font.sans-serif': [self.font_cjk, self.font_family],
            'axes.unicode_minus': False,
            'axes.linewidth': 1.0,
            'axes.spines.top': not self.remove_top_right,
            'axes.spines.right': not self.remove_top_right,
            'xtick.direction': self.tick_direction,
            'ytick.direction': self.tick_direction,
            'xtick.major.size': self.tick_length,
            'ytick.major.size': self.tick_length,
            'xtick.major.width': self.tick_width,
            'ytick.major.width': self.tick_width,
            'xtick.labelsize': self.tick_fontsize,
            'ytick.labelsize': self.tick_fontsize,
            'legend.fontsize': self.legend_fontsize,
            'legend.frameon': self.legend_frameon,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'lines.linewidth': self.linewidth,
        }

    def apply_style(self) -> None:
        """将样式配置应用到 matplotlib 全局 rcParams。"""
        import matplotlib.pyplot as plt
        plt.rcParams.update(self.rcParams)


plot_config = PlotConfig()

task_config = TaskConfig()
data_path = DataPath()


# ╔══ 自检 ══╗
if __name__ == '__main__':
    import textwrap

    def _print_section(title: str, obj: object) -> None:
        header = f' {title} '
        print(f'\n{"=" * 60}')
        print(f'{header:=^60}')
        print(f'{"=" * 60}')
        for fld_name in obj.__dataclass_fields__:
            value = getattr(obj, fld_name)
            if isinstance(value, Path):
                value = value.as_posix()
            print(f'  {fld_name:<20s} = {value!r}')

    print('多源融合机器人定位 —— 配置总览')
    print(f'  Python frozen dataclass, 共 5 组配置')

    _print_section('TimeConfig', time_config)
    _print_section('FilterConfig', filter_config)
    _print_section('AlignmentConfig', alignment_config)
    _print_section('TaskConfig', task_config)
    _print_section('DataPath', data_path)

    print(f'\n{"─" * 60}')
    print('  派生属性速查：')
    print(f'    传感器1 采样周期 dt1       = {time_config.dt1:.4f} s')
    print(f'    传感器2 采样周期 dt2       = {time_config.dt2:.4f} s')
    print(f'    目标输出周期   dt_target   = {time_config.dt_target:.4f} s')
    print(f'    附件1 完整路径             = {data_path.path1}')
    print(f'    附件2 完整路径             = {data_path.path2}')
    print(f'    附件3 完整路径             = {data_path.path3}')
    print(f'    附件4 完整路径             = {data_path.path4}')
    print(f'{"─" * 60}')
    print('\n配置检查完毕。')

# ============================================================
#  任务约束参数
# ============================================================
class TaskConfig:
    """射击与拍照任务的物理约束参数。"""

    # ---- 射击任务 ----
    SHOOT_PREP_TIME: float = 1.5        # 准备时间 (s)
    SHOOT_DIST_MIN: float = 5.0         # 最小距离 (m)
    SHOOT_DIST_MAX: float = 30.0        # 最大距离 (m)
    SHOOT_SPEED_MAX: float = 2.0        # 速率上限 (m/s)
    SHOOT_ACC_MAX: float = 1.5          # 加速度上限 (m/s²)

    # ---- 拍照任务 ----
    PHOTO_PREP_TIME: float = 0.5        # 准备时间 (s)
    PHOTO_DIST_MIN: float = 10.0        # 最小距离 (m)
    PHOTO_DIST_MAX: float = 40.0        # 最大距离 (m)
    PHOTO_SPEED_MAX: float = 1.0        # 速率上限 (m/s)
    PHOTO_ACC_MAX: float = 1.5          # 加速度上限 (m/s²)
    PHOTO_HEADING_DIFF_MIN: float = 60.0  # 最小航向角差异 (°)
    PHOTO_MAX_PER_TARGET: int = 3       # 每个目标最大拍照次数