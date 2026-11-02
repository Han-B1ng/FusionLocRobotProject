# file: config.py


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

def _setup_chinese_font():
    import os
    import platform

    _SYSTEM = platform.system()
    if _SYSTEM == "Windows":
        font_search_roots = [r"C:\Windows\Fonts"]
        candidates = [
            ("simhei", "SimHei"),
            ("msyh", "Microsoft YaHei"),
            ("simkai", "KaiTi"),
            ("simfang", "FangSong"),
            ("simsun", "SimSun"),
            ("msjh", "Microsoft JhengHei"),
        ]
    elif _SYSTEM == "Darwin":
        font_search_roots = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
        candidates = [
            ("PingFang", "PingFang SC"),
            ("Heiti SC", "Heiti SC"),
            ("STHeiti", "STHeiti"),
            ("Songti SC", "Songti SC"),
        ]
    else:  # Linux
        font_search_roots = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
        candidates = [
            ("wqy-microhei", "WenQuanYi Micro Hei"),
            ("wqy-zenhei", "WenQuanYi Zen Hei"),
            ("NotoSansCJK", "Noto Sans CJK SC"),
            ("NotoSansSC", "Noto Sans SC"),
            ("SourceHanSansSC", "Source Han Sans SC"),
            ("SourceHanSansCN", "Source Han Sans CN"),
            ("DroidSansFallback", "Droid Sans Fallback"),
            ("uming", "AR PL UMing CN"),
        ]

    all_font_files: list = []
    for root_dir in font_search_roots:
        if not os.path.isdir(root_dir):
            continue
        for base, _dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith((".ttf", ".ttc", ".otf")):
                    all_font_files.append((fname.lower(), os.path.join(base, fname)))

    for kw, display_name in candidates:
        for fl, font_path in all_font_files:
            if kw in fl:
                try:
                    fm.fontManager.addfont(font_path)
                    matplotlib.rcParams["font.sans-serif"] = [display_name, "DejaVu Sans"]
                    matplotlib.rcParams["font.family"] = "sans-serif"
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    print(f"[config] 中文字体已从文件加载: {font_path} -> {display_name}")
                    return display_name
                except Exception:
                    continue

    installed = {f.name for f in fm.fontManager.ttflist}
    name_priority = [
        "SimHei", "Microsoft YaHei", "FangSong", "KaiTi", "SimSun",
        "PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC",
        "Source Han Sans SC", "Source Han Sans CN",
        "Droid Sans Fallback", "AR PL UMing CN",
    ]
    for font_name in name_priority:
        if font_name in installed:
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[config] 中文字体已配置 (名称匹配): {font_name}")
            return font_name

    print("[config] 警告：未找到中文字体，图表中文可能乱码。")
    print("  Windows: 检查 C:\\Windows\\Fonts\\simhei.ttf 是否存在")
    print("  Linux:   sudo apt install fonts-wqy-microhei")
    print("  macOS:   系统应自带 PingFang SC")
    return None


_detected_font = _setup_chinese_font()

@dataclass(frozen=True)
class TimeConfig:

    freq1: float = 4.0          # Hz，传感器 1 采样频率
    freq2: float = 5.0          # Hz，传感器 2 采样频率

    target_freq: float = 10.0   # Hz，融合后目标输出频率

    t_range_p1: Tuple[float, float] = (221.0, 970.75)
    t_range_p2: Tuple[float, float] = (102.0, 852.0)
    t_range_p3: Tuple[float, float] = (469.0, 1268.83)

    @property
    def dt1(self) -> float:
        return 1.0 / self.freq1

    @property
    def dt2(self) -> float:
        return 1.0 / self.freq2

    @property
    def dt_target(self) -> float:
        return 1.0 / self.target_freq


@dataclass(frozen=True)
class FilterConfig:

    state_dim: int = 6

    P0: Tuple[float, ...] = (
        10.0, 10.0,         # 位置初始不确定性 (m²)
        5.0,  5.0,          # 速度初始不确定性 ((m/s)²)
        1.0,  1.0,          # 偏差初始不确定性 (m²)
    )

    Q: Tuple[float, ...] = (
        0.1,  0.1,          # 位置过程噪声 (m²)
        0.5,  0.5,          # 速度过程噪声 ((m/s)²)
        0.01, 0.01,         # 偏差随机游走噪声 (m²)
    )

    R1_fixed: Tuple[float, float] = (0.5, 0.5)   # 传感器 1 观测噪声 (m²)
    R2_fixed: Tuple[float, float] = (0.3, 0.3)   # 传感器 2 观测噪声 (m²)

    estimate_bias: bool = True

    adaptive_R: bool = False

    @property
    def R1(self) -> Tuple[float, float]:
        return self.R1_fixed

    @property
    def R2(self) -> Tuple[float, float]:
        return self.R2_fixed


@dataclass(frozen=True)
class AlignmentConfig:

    corr_window: float = 1.0
    method: str = 'linear'
    delay_range: Tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class TaskConfig:

    shoot_d: Tuple[float, float] = (5.0, 30.0)
    shoot_vmax: float = 2.0
    shoot_amax: float = 1.5
    shoot_prep: float = 1.5

    photo_d: Tuple[float, float] = (10.0, 40.0)
    photo_vmax: float = 1.5
    photo_amax: float = 1.5
    photo_angle_min: float = 60.0
    photo_prep: float = 0.5


@dataclass(frozen=True)
class DataPath:

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


time_config = TimeConfig()
filter_config = FilterConfig()
alignment_config = AlignmentConfig()

@dataclass(frozen=True)
class PlotConfig:

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

    font_family: str = 'sans-serif'
    font_cjk: str = _detected_font or 'SimHei'

    @property
    def CN_FONT(self) -> str:
        return self.font_cjk

    linewidth: float = 2.0
    linewidth_thin: float = 1.0
    linewidth_thick: float = 3.0

    dpi: int = 300
    dpi_high: int = 600

    remove_top_right: bool = True
    tick_direction: str = 'in'
    tick_length: float = 4.0
    tick_width: float = 1.0

    title_fontsize: float = 14
    label_fontsize: float = 12
    tick_fontsize: float = 10
    legend_fontsize: float = 10
    legend_frameon: bool = False

    @property
    def rcParams(self) -> dict:
        return {
            'font.family': self.font_family,
            'font.sans-serif': [self.font_cjk, 'DejaVu Sans'],
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
        import matplotlib.pyplot as plt
        plt.rcParams.update(self.rcParams)


plot_config = PlotConfig()

task_config = TaskConfig()
data_path = DataPath()


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

    print('  派生属性速查：')
    print(f'    传感器1 采样周期 dt1       = {time_config.dt1:.4f} s')
    print(f'    传感器2 采样周期 dt2       = {time_config.dt2:.4f} s')
    print(f'    目标输出周期   dt_target   = {time_config.dt_target:.4f} s')
    print(f'    附件1 完整路径             = {data_path.path1}')
    print(f'    附件2 完整路径             = {data_path.path2}')
    print(f'    附件3 完整路径             = {data_path.path3}')
    print(f'    附件4 完整路径             = {data_path.path4}')
    print('\n配置检查完毕。')

class TaskConfig:

    SHOOT_PREP_TIME: float = 1.5        # 准备时间 (s)
    SHOOT_DIST_MIN: float = 5.0         # 最小距离 (m)
    SHOOT_DIST_MAX: float = 30.0        # 最大距离 (m)
    SHOOT_SPEED_MAX: float = 2.0        # 速率上限 (m/s)
    SHOOT_ACC_MAX: float = 1.5          # 加速度上限 (m/s²)

    PHOTO_PREP_TIME: float = 0.5        # 准备时间 (s)
    PHOTO_DIST_MIN: float = 10.0        # 最小距离 (m)
    PHOTO_DIST_MAX: float = 40.0        # 最大距离 (m)
    PHOTO_SPEED_MAX: float = 1.0        # 速率上限 (m/s)
    PHOTO_ACC_MAX: float = 1.5          # 加速度上限 (m/s²)
    PHOTO_HEADING_DIFF_MIN: float = 60.0  # 最小航向角差异 (°)
    PHOTO_MAX_PER_TARGET: int = 3       # 每个目标最大拍照次数
