# @Author : Han_B1ng
# @Time : 2026/5/6 20:45
# @Description :
"""
多源融合机器人定位项目 — 全局配置
=================================
所有配置均使用 dataclass 管理，便于类型检查与 IDE 补全。
矩阵类参数以对角元元组表示，运行时可用 numpy.diag() 转换为完整矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ──────────────────────────────────────────────
#  时间与频率配置
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TimeConfig:
    """传感器采样频率、目标输出频率及各问题数据的时间区间。"""

    # --- 传感器固有频率 ---
    freq1: float = 4.0          # Hz，传感器 1 采样频率
    freq2: float = 5.0          # Hz，传感器 2 采样频率

    # --- 融合输出目标频率 ---
    target_freq: float = 10.0   # Hz，融合后目标输出频率

    # --- 各问题数据的时间范围 (秒) ---
    #     根据附件文件名 / 表头推测的大致有效区间，
    #     格式为 (起始秒, 结束秒)，用于数据截取与对齐。
    t_range_p1: Tuple[float, float] = (221.0, 645.0)    # 问题 1
    t_range_p2: Tuple[float, float] = (102.0, 852.0)    # 问题 2
    t_range_p3: Tuple[float, float] = (469.0, 1002.0)   # 问题 3

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


# ──────────────────────────────────────────────
#  卡尔曼滤波配置
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class FilterConfig:
    """
    扩展卡尔曼滤波器 / 普通卡尔曼滤波器参数。

    状态向量默认定义为 [x, y, vx, vy]（4 维）；
    当 estimate_bias = True 时扩展为 [x, y, vx, vy, bx, by]（6 维），
    其中 (bx, by) 为系统偏差估计量。

    所有矩阵以对角元元组存储，运行时可用 numpy.diag() 还原。
    """

    # --- 状态维度 ---
    state_dim: int = 6                  # 含系统偏差时的状态维数

    # --- 初始状态协方差 P0 对角元 ---
    #     [σ²_x, σ²_y, σ²_vx, σ²_vy, σ²_bx, σ²_by]
    #     单位：位置 m²，速度 (m/s)²，偏差 m²
    P0: Tuple[float, ...] = (
        10.0, 10.0,         # 位置初始不确定性 (m²)
        5.0,  5.0,          # 速度初始不确定性 ((m/s)²)
        1.0,  1.0,          # 偏差初始不确定性 (m²)
    )

    # --- 过程噪声协方差 Q 对角元 ---
    #     单位同 P0
    Q: Tuple[float, ...] = (
        0.1,  0.1,          # 位置过程噪声 (m²)
        0.5,  0.5,          # 速度过程噪声 ((m/s)²)
        0.01, 0.01,         # 偏差随机游走噪声 (m²)
    )

    # --- 传感器 1 观测噪声 R1 ---
    #     观测量为 [x, y]，单位 m²
    R1: Tuple[float, float] = (0.5, 0.5)

    # --- 传感器 2 观测噪声 R2 ---
    #     观测量为 [x, y]，单位 m²
    R2: Tuple[float, float] = (0.3, 0.3)

    # --- 是否启用系统偏差估计 ---
    estimate_bias: bool = True


# ──────────────────────────────────────────────
#  时间对齐配置
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class AlignmentConfig:
    """传感器时间戳对齐参数。"""

    corr_window: float = 2.0            # 秒，互相关搜索滑动窗口长度
    method: str = 'cubic'               # 插值方法：'cubic' | 'linear'
    delay_range: Tuple[float, float] = (-1.0, 1.0)  # 秒，时偏搜索范围


# ──────────────────────────────────────────────
#  任务约束配置（射击 & 拍照）
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TaskConfig:
    """
    机器人执行任务的物理约束参数。
    所有角度单位为 **度 (°)**，计算时请自行转换为弧度。
    """

    # ====== 射击约束 ======
    shoot_d: Tuple[float, float] = (5.0, 30.0)   # m，射击有效距离范围
    shoot_vmax: float = 2.0                       # m/s，射击时最大允许速度
    shoot_amax: float = 1.5                       # m/s²，射击时最大允许加速度
    shoot_prep: float = 1.5                       # s，射击前准备时间

    # ====== 拍照约束 ======
    photo_d: Tuple[float, float] = (10.0, 40.0)  # m，拍照有效距离范围
    photo_vmax: float = 1.5                       # m/s，拍照时最大允许速度
    photo_amax: float = 1.5                       # m/s²，拍照时最大允许加速度
    photo_angle_min: float = 60.0                 # °，相邻拍照角度最小差异
    photo_prep: float = 0.5                       # s，拍照前准备时间


# ──────────────────────────────────────────────
#  数据路径配置
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class DataPath:
    """项目数据文件与输出目录。使用 pathlib.Path 保证跨平台兼容。"""

    data_dir: Path = Path('data')         # 数据根目录
    output_dir: Path = Path('output')     # 输出目录

    # --- 附件文件名（相对于 data_dir）---
    file1: str = '附件1.csv'              # 传感器 1 原始数据
    file2: str = '附件2.csv'              # 传感器 2 原始数据
    file3: str = '附件3.csv'              # 问题 1 / 问题 2 参考轨迹
    file4: str = '附件4.csv'              # 问题 3 参考 / 补充数据

    @property
    def path1(self) -> Path:
        """附件 1 完整路径。"""
        return self.data_dir / self.file1

    @property
    def path2(self) -> Path:
        """附件 2 完整路径。"""
        return self.data_dir / self.file2

    @property
    def path3(self) -> Path:
        """附件 3 完整路径。"""
        return self.data_dir / self.file3

    @property
    def path4(self) -> Path:
        """附件 4 完整路径。"""
        return self.data_dir / self.file4


# ──────────────────────────────────────────────
#  全局配置实例 —— 项目中直接 import 使用
# ──────────────────────────────────────────────
time_config = TimeConfig()
filter_config = FilterConfig()
alignment_config = AlignmentConfig()
task_config = TaskConfig()
data_path = DataPath()


# ──────────────────────────────────────────────
#  自检：直接运行此文件时打印所有配置
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import textwrap

    def _print_section(title: str, obj: object) -> None:
        """格式化打印一个 dataclass 实例的所有字段。"""
        header = f' {title} '
        print(f'\n{"=" * 60}')
        print(f'{header:=^60}')
        print(f'{"=" * 60}')
        for fld_name in obj.__dataclass_fields__:          # type: ignore[attr-defined]
            value = getattr(obj, fld_name)
            # 对 Path 对象显示为 POSIX 风格字符串
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

    # 额外输出便捷属性
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

