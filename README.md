# 多源传感器融合定位系统

<div align="center">

基于小波去噪、扩展卡尔曼滤波（EKF）与贪心调度的多源传感器数据融合与任务规划方案

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 项目概述

本项目针对两类不同频率的定位传感器（方式1: 4Hz，方式2: 5Hz），实现从**数据预处理**到**任务规划**的完整处理流水线，最终输出统一的 **10Hz 融合轨迹**并生成**任务执行时刻表**。

### ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔧 **时间对齐** | 互相关 + 最小二乘精细对齐，支持大尺度时钟偏差的两阶段搜索 |
| 🌊 **小波去噪** | 自动对比 2 小波基 × 2 阈值策略的最优组合 |
| 📊 **偏差估计** | 中位数 / 截尾均值双方法对比 + 显著性检验 |
| 🎯 **状态估计** | 扩展卡尔曼滤波（EKF），支持自适应观测噪声矩阵 R |
| 📅 **任务规划** | 基于约束的可行窗口搜索 + 贪心调度算法 |
| 📈 **可视化** | 完整的图表输出，包括轨迹图、甘特图、3D图等 |
| 🌏 **中文支持** | 完善的matplotlib中文字体配置，跨平台兼容 |

---

## 📁 项目结构

```
FusionLocRobotProject/
├── main.py                      # 主入口，一键运行全流程
├── config.py                    # 全局配置（路径、参数、约束条件）
├── stage0_eda.py                # 阶段0：数据探索与预处理
├── stage1_problem1.py           # 阶段1：无噪声时间对齐
├── stage2_problem2.py           # 阶段2：含噪声 + 系统偏差融合
├── stage3_problem3.py           # 阶段3：实际数据处理
├── stage4_problem4.py           # 阶段4：任务规划与调度
│
├── core/                        # 核心算法模块
│   ├── time_alignment.py        # 时间对齐算法
│   ├── wavelet_utils.py         # 小波去噪工具
│   ├── robust_stats.py          # 鲁棒统计与偏差检验
│   ├── kalman_filters.py        # 扩展卡尔曼滤波器
│   ├── motion_utils.py          # 运动状态计算（速度/加速度）
│   ├── constraint_checker.py    # 约束条件检查器
│   └── task_scheduler.py        # 贪心任务调度器
│
├── visualization/               # 可视化模块
│   ├── plot_eda.py              # EDA 图表
│   ├── plot_trajectory.py       # 轨迹对比图
│   ├── plot_results.py          # 任务规划图
│   ├── plot_case_study.py       # 案例分析图
│   ├── plot_constraint_analysis.py  # 约束分析图
│   ├── plot_sensitivity_analysis.py # 敏感性分析图
│   └── plot_publication.py      # 论文级组合图
│
├── data/                        # 数据目录（放置附件文件）
│   ├── 附件1.xlsx
│   ├── 附件2.xlsx
│   ├── 附件3.xlsx
│   └── 附件4.xlsx
│
├── output/                      # 输出目录（自动生成）
│   ├── Problem1_10Hz.xlsx
│   ├── Problem2_10Hz.xlsx
│   ├── Problem3_10Hz.xlsx
│   ├── result.xlsx
│   ├── cleaned_data.pkl
│   └── figures/                 # 可视化图表
│
├── requirements.txt             # Python依赖包
├── CHINESE_FONT_FIX.md          # 中文字体配置说明
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- 支持的操作系统：Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install numpy pandas matplotlib openpyxl scipy pywt
```

### 数据准备

将以下数据文件放入 `data/` 目录：

- `附件1.xlsx` - 问题1数据（无噪声）
- `附件2.xlsx` - 问题2数据（含噪声和偏差）
- `附件3.xlsx` - 问题3数据（实际数据）
- `附件4.xlsx` - 问题4目标点坐标

### 运行方式

#### 方式1：一键运行全流程

```bash
python main.py
```

#### 方式2：按阶段运行

```bash
# 运行指定阶段（如阶段 0 和 1）
python main.py --stages 0 1

# 运行单个阶段
python main.py --stages 3

# 跳过可视化
python main.py --skip-viz
```

#### 方式3：单独运行某个阶段

```bash
python stage0_eda.py        # 数据探索
python stage1_problem1.py   # 问题1求解
python stage2_problem2.py   # 问题2求解
python stage3_problem3.py   # 问题3求解
python stage4_problem4.py   # 问题4求解
```

---

## 📊 处理流程

### 阶段 0 — 数据探索与预处理（EDA）

- ✅ 自动识别 Excel（双 sheet）/ CSV 格式
- ✅ 数据质量检查：时间单调性、采样间隔、坐标突跳检测
- ✅ 绘制原始轨迹并保存至 `output/figures/`
- ✅ 序列化为 `cleaned_data.pkl` 供下游使用

**输出：**
- `output/cleaned_data.pkl` - 清洗后的数据
- `output/figures/附件*_raw.png` - 原始轨迹图
- `output/figures/附件*_timeseries.png` - 时间序列图
- `output/figures/附件*_dt_hist.png` - 采样间隔分布

### 阶段 1 — 无噪声时间对齐（问题 1）

**输入：** 附件1（无噪声数据）

**流程：**
1. 加载两组传感器数据
2. 互相关 + 最小二乘估计时间偏差
3. 等权重融合输出 10Hz 轨迹

**输出：**
- `output/Problem1_10Hz.xlsx` - 融合轨迹
- `output/figures/Problem1_trajectory.png` - 轨迹对比图

### 阶段 2 — 含噪声融合（问题 2）

**输入：** 附件2（含噪声和系统偏差）

**流程：**
1. **去噪**：小波去噪参数对比实验（db4/sym5 × universal/bayes = 4种组合），以加速度方差为指标自动选优
2. **时间对齐**：估计时偏
3. **偏差估计**：中位数 vs 截尾均值对比，含 3σ 异常点检测与显著性检验
4. **融合**：EKF 融合，默认 R 与自适应 R 对比，选取残差方差更小的方案

**输出：**
- `output/Problem2_10Hz.xlsx` - 融合轨迹 + 偏差序列
- `output/figures/Problem2_trajectory.png` - 轨迹对比图
- `output/figures/Problem2_bias.png` - 偏差估计图
- `output/figures/Problem2_3D.png` - 3D轨迹图
- `output/ablation.xlsx` - 消融实验结果

### 阶段 3 — 实际数据处理（问题 3）

**输入：** 附件3（实际采集数据）

**流程：**
- 与阶段 2 流程一致，新增：
  - **两阶段粗略时间偏移估计**：轨迹形状匹配，处理大尺度时钟偏差
  - **自适应观测噪声估计改进**：MAD + 偏差补偿 + 上界约束

**输出：**
- `output/Problem3_10Hz.xlsx` - 融合轨迹 + 偏差序列
- `output/figures/Problem3_trajectory.png` - 轨迹对比图
- `output/figures/Problem3_3D.png` - 3D轨迹图
- `output/figures/Problem3_multilayer.png` - 多层融合轨迹图
- `output/literature_comparison.xlsx` - 文献对比结果

### 阶段 4 — 任务规划（问题 4）

**输入：** 
- 问题3输出的融合轨迹
- 附件4的目标点坐标

**流程：**
1. 读取融合轨迹与目标点坐标
2. 计算速度、加速度等运动状态
3. 基于距离 / 速率 / 加速度约束搜索所有可行时间窗口
4. 贪心调度生成无冲突任务序列
5. 输出 Excel 结果表 + 轨迹图 + 甘特图

**约束条件：**
- **射击任务**：距离 [5, 30]m，速度 ≤ 2.0 m/s，加速度 ≤ 1.5 m/s²，准备时间 1.5s
- **拍照任务**：距离 [10, 40]m，速度 ≤ 1.0 m/s，加速度 ≤ 1.5 m/s²，航向角差异 ≥ 60°，准备时间 0.5s

**输出：**
- `output/result.xlsx` - 任务执行时刻表
- `output/figures/Problem4_schedule.png` - 任务调度图
- `output/figures/Problem4_gantt.png` - 甘特图
- `output/figures/constraint_funnel_p4.png` - 约束漏斗图

---

## ⚙️ 配置说明

所有可调参数集中在 [`config.py`](config.py) 中：

### 核心配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `data_path` | 数据文件路径与输出目录 | `data/`, `output/` |
| `time_config` | 目标输出频率、采样间隔等 | 10 Hz |
| `alignment_config` | 时间对齐搜索范围与方法 | [-500, 800]s |
| `filter_config` | EKF 过程噪声 Q、观测噪声 R1/R2 | 见代码 |
| `TaskConfig` | 射击 / 拍照的距离、速率、加速度约束 | 见代码 |

### 修改配置示例

```python
# config.py
from config import filter_config, task_config

# 修改EKF参数
filter_config.Q = (0.2, 0.2, 1.0, 1.0, 0.02, 0.02)

# 修改任务约束
task_config.SHOOT_DIST_MAX = 35.0  # 射击最大距离改为35m
```

---

## 📈 输出结果

### 数据文件

| 文件 | 内容 |
|------|------|
| `Problem1_10Hz.xlsx` | 问题1融合轨迹（Time, X, Y） |
| `Problem2_10Hz.xlsx` | 问题2融合轨迹 + 偏差序列 |
| `Problem3_10Hz.xlsx` | 问题3融合轨迹 + 偏差序列 |
| `result.xlsx` | 问题4任务执行时刻表 |
| `cleaned_data.pkl` | 清洗后的原始数据 |
| `ablation.xlsx` | 消融实验结果 |
| `literature_comparison.xlsx` | 文献对比结果 |

### 可视化图表

所有图表保存在 `output/figures/` 目录：

- **轨迹图**：2D/3D轨迹对比、多层融合轨迹
- **误差图**：融合误差随时间变化
- **速度图**：速度曲线、速度热力图
- **偏差图**：系统偏差估计与收敛
- **任务图**：任务调度甘特图、约束漏斗图
- **分析图**：敏感性分析、案例分析

---

## 🌏 中文字体支持

本项目已完善matplotlib中文字体配置，支持跨平台中文显示。

### 自动字体检测

系统会自动检测并使用以下中文字体：

- **Windows**: SimHei（黑体）、Microsoft YaHei（微软雅黑）
- **macOS**: PingFang SC（苹方）、Heiti SC（黑体）
- **Linux**: WenQuanYi Micro Hei（文泉驿微米黑）、Noto Sans CJK

详见 [`CHINESE_FONT_FIX.md`](CHINESE_FONT_FIX.md)

---

## 📝 技术栈

- **数据处理**: NumPy, Pandas
- **科学计算**: SciPy, PyWavelets
- **可视化**: Matplotlib
- **Excel处理**: OpenPyXL
- **优化求解**: PuLP（可选，用于ILP求解）

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👨‍💻 作者

**Han_B1ng** - 2026.05

---

## 🙏 致谢

感谢以下开源项目：

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/)

---

<div align="center">

**如果这个项目对您有帮助，请给个 ⭐ Star！**

</div>
