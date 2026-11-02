# 多源传感器融合定位与任务规划

<div align="center">

**从原始传感器数据到融合轨迹与任务调度的完整处理流水线**

小波去噪 · 扩展卡尔曼滤波 · 互相关时间对齐 · 贪心调度

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

</div>

---

## 问题描述

两台定位传感器以不同频率工作（方式1: 4 Hz，方式2: 5 Hz），且存在时钟偏差、测量噪声和系统偏差。本项目的目标是将两组传感器数据融合为统一的 10 Hz 轨迹，并基于该轨迹为目标点规划观测任务（射击/拍照）。

本项目按难度递增分为四个子问题：

| 问题 | 输入 | 挑战 | 输出 |
|------|------|------|------|
| **问题1** | 无噪声数据 | 仅需时间对齐 | 10 Hz 融合轨迹 |
| **问题2** | 含噪声与系统偏差 | 去噪、偏差估计、EKF 融合 | 10 Hz 融合轨迹 + 偏差序列 |
| **问题3** | 实际采集数据 | 大尺度时钟偏差、未知噪声特性 | 10 Hz 融合轨迹 + 偏差序列 |
| **问题4** | 问题3 轨迹 + 目标点坐标 | 约束满足、无冲突调度 | 任务执行时刻表 |

---

## 方法论

### 流水线总览

```
原始传感器数据 --> 数据探索与预处理 --> 小波去噪 --> 时间对齐
                                                        |
    [任务时刻表] <-- 贪心调度 <-- 运动状态计算 <-- EKF 融合
```

### 时间对齐

互相关给出两路传感器轨迹之间的粗略时偏估计，再通过有界最小二乘优化（Brent 方法）进行亚采样级精化。针对问题3中的大尺度时钟偏差，采用两阶段策略：先用 MSE 网格搜索进行粗对齐，再用互相关精调。

### 小波去噪

使用离散小波变换（DWT），自动遍历 2 种小波基（db4, sym5） × 2 种阈值策略（Universal, BayesShrink）共 4 种参数组合，以去噪后信号的加速度方差最小化为指标自动选优。噪声水平通过 MAD（中位数绝对偏差）估计。

### 系统偏差估计

在时间对齐后，对位置残差分别采用中位数和 10% 截尾均值两种估计量进行评估。采用基于 MAD 的 3-sigma 准则滤除粗差后再行估计。通过 Wilcoxon 符号秩检验判断偏差是否显著——若不显著，则不做偏差校正。

### 扩展卡尔曼滤波融合

采用 6 维状态向量 `[x, y, vx, vy, bx, by]` 对位置、速度及传感器偏差建模。偏差服从 AR(1) 随机游走。顺序更新以传感器1 为参考（无偏项），传感器2 包含偏差修正项。自适应观测噪声基于残差 MAD 估计，引入偏差补偿项和上界约束以增强鲁棒性。最后进行 RTS 固定区间平滑。

### 任务调度

将运动学约束与融合轨迹做卷积，识别满足约束的可行时间窗口。采用带 MRV（最受约束变量）启发式的贪心调度器分配任务，优先调度可行窗口少的稀有目标以最大化覆盖率。拍照任务强制要求对同一目标的多次拍摄具有 >= 60° 的航向角差异。

#### 任务约束

| 参数 | 射击任务 | 拍照任务 |
|------|----------|----------|
| 距离 | [5, 30] m | [10, 40] m |
| 速率 | <= 2.0 m/s | <= 1.0 m/s |
| 加速度 | <= 1.5 m/s² | <= 1.5 m/s² |
| 准备时间 | 1.5 s | 0.5 s |
| 航向角多样性 | — | >= 60° |

---

## 项目结构

```
FusionLocRobotProject/
├── main.py                          # 主入口：一键运行全流程或选定阶段
├── config.py                        # 全局配置（冻结 dataclass）
├── stage0_eda.py                    # 阶段0：数据探索与预处理
├── stage1_problem1.py               # 阶段1：无噪声时间对齐
├── stage2_problem2.py               # 阶段2：含噪声与系统偏差融合
├── stage3_problem3.py               # 阶段3：实际数据融合
├── stage4_problem4.py               # 阶段4：任务规划与调度
├── sensitivity_analysis.py          # 过程噪声 Q 敏感性分析
├── generate_summary.py              # 多 Sheet 汇总表生成
│
├── core/                            # 核心算法模块
│   ├── time_alignment.py            # 时间对齐主流程
│   ├── interpolation.py             # 重采样与插值
│   ├── sync_estimation.py           # 互相关与最小二乘时偏估计
│   ├── wavelet_utils.py             # 小波去噪与自动参数选择
│   ├── robust_stats.py              # 鲁棒统计与偏差显著性检验
│   ├── kalman_filters.py            # 自适应 EKF + RTS 平滑
│   ├── motion_utils.py              # 速度、加速度、航向角计算
│   ├── constraint_checker.py        # 可行时间窗口搜索（v5 优化版）
│   └── task_scheduler.py            # MRV 启发式贪心调度器
│
├── visualization/                   # 可视化模块
│   ├── plot_eda.py                  # EDA 图表
│   ├── plot_trajectory.py           # 轨迹对比图
│   ├── plot_results.py              # 任务调度图与甘特图
│   ├── plot_case_study.py           # 案例分析图
│   ├── plot_constraint_analysis.py  # 约束漏斗图
│   ├── plot_sensitivity_analysis.py # 敏感性分析图
│   ├── plot_ablation.py             # 消融实验图
│   └── plot_publication.py          # 论文级组合图
│
├── data/                            # 数据目录（放置附件文件）
│   ├── 附件1.xlsx                   # 问题1：无噪声数据
│   ├── 附件2.xlsx                   # 问题2：含噪声与偏差数据
│   ├── 附件3.xlsx                   # 问题3：实际采集数据
│   └── 附件4.xlsx                   # 问题4：目标点坐标
│
├── output/                          # 自动生成的输出
│   ├── intermediate/                # 中间结果（.pkl）
│   ├── plots/                       # 图表（.png）
│   └── tables/                      # 结果表（.xlsx）
│
├── requirements.txt                 # Python 依赖
└── README.md
```

---

## 快速开始

### 环境要求

- Python >= 3.8
- Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

将四个竞赛附件文件放入 `data/` 目录：
`附件1.xlsx`、`附件2.xlsx`、`附件3.xlsx`、`附件4.xlsx`

### 运行

```bash
# 全流程（所有阶段 + 可视化）
python main.py

# 仅运行指定阶段
python main.py --stages 0 1 2

# 跳过可视化加速运行
python main.py --skip-viz

# 单独运行某个阶段
python stage1_problem1.py
```

---

## 配置说明

所有可调参数集中在 `config.py`，以冻结 dataclass 组织：

| 配置类 | 范围 | 关键参数 |
|--------|------|----------|
| `TimeConfig` | 传感器频率与目标输出 | freq1=4 Hz, freq2=5 Hz, target=10 Hz |
| `FilterConfig` | EKF 参数 | 6 维状态，Q/P0/R 对角元 |
| `AlignmentConfig` | 时偏搜索 | delay_range, 相关窗口 |
| `TaskConfig` | 射击与拍照约束 | 距离、速率、加速度限制 |
| `DataPath` | 输入输出路径 | data/, output/ 目录 |
| `PlotConfig` | 可视化样式 | Nature 色盲友好配色，DPI，字体 |

内置跨平台中文字体自动检测（Windows: SimHei, macOS: PingFang SC, Linux: Noto Sans CJK）。

---

## 输出结果

### 数据文件

| 文件 | 内容 |
|------|------|
| `output/tables/Problem1_10Hz.xlsx` | 问题1 融合轨迹 |
| `output/tables/Problem2_10Hz.xlsx` | 问题2 融合轨迹 + 偏差序列 |
| `output/tables/Problem3_10Hz.xlsx` | 问题3 融合轨迹 + 偏差序列 |
| `output/tables/result.xlsx` | 任务执行时刻表 |
| `output/tables/ablation.xlsx` | 消融实验结果 |
| `output/tables/literature_comparison.xlsx` | 文献对比结果 |
| `output/tables/summary.xlsx` | 多 Sheet 汇总报告 |

### 图表

全部保存在 `output/plots/`：轨迹对比图（2D/3D）、偏差估计图、速度曲线图、甘特图、约束漏斗图、敏感性分析热力图，以及论文级组合大图。

---

## 核心设计决策

- **自适应观测噪声** 优于固定 R —— 基于残差 MAD 估计，配合偏差补偿和上界约束，对未知噪声特性具有更好的鲁棒性。
- **两阶段粗对齐** 应对问题3的大尺度时钟偏差 —— MSE 网格搜索粗定位后再用互相关精调。
- **小波参数自动选择** 避免人工调参 —— 4 种组合自动评估，选取加速度方差最小的方案。
- **MRV 调度启发式** 优先调度可行窗口少的稀有目标，相比简单贪心获得更高的覆盖率。
- **偏差显著性检验** 在偏差未通过 Wilcoxon 检验时不进行校正，避免对噪声的过拟合。

---

## 依赖

| 包 | 用途 |
|----|------|
| NumPy, SciPy | 数值计算、优化、信号处理 |
| Pandas | 数据加载与处理 |
| Matplotlib, Seaborn | 可视化 |
| OpenPyXL | Excel 读写 |
| PyWavelets | 离散小波变换 |
| statsmodels | 统计检验 |
| filterpy | 卡尔曼滤波辅助 |
| scikit-learn | 辅助工具 |
| tqdm | 进度条 |

---

## 作者

**Han_B1ng** — 2026年5月