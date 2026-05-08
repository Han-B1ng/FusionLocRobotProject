多源传感器融合定位系统
基于小波去噪、扩展卡尔曼滤波（EKF）与贪心调度的多源传感器数据融合与任务规划方案。
项目概述
本项目针对两类不同频率的定位传感器（方式 1: 4Hz，方式 2: 5Hz），实现从数据预处理到任务规划的完整处理流水线，最终输出统一的 10Hz 融合轨迹 并生成任务执行时刻表。
核心能力
表格
能力	说明
时间对齐	互相关 + 最小二乘精细对齐，支持大尺度时钟偏差的两阶段搜索
去噪	小波去噪，自动对比 2 小波基 × 2 阈值策略的最优组合
偏差估计	中位数 / 截尾均值双方法对比 + 显著性检验
状态估计	扩展卡尔曼滤波（EKF），支持自适应观测噪声矩阵 R
任务规划	基于约束的可行窗口搜索 + 贪心调度算法
项目结构
plaintext
├── main.py              # 主入口，一键运行全流程
├── config.py            # 全局配置（路径、参数、约束条件）
├── stage0_eda.py        # 阶段0：数据探索与预处理
├── stage1_problem1.py   # 阶段1：无噪声时间对齐
├── stage2_problem2.py   # 阶段2：含噪声 + 系统偏差融合
├── stage3_problem3.py   # 阶段3：实际数据处理
├── stage4_problem4.py   # 阶段4：任务规划与调度
├── core/                # 核心算法模块
│   ├── time_alignment.py    # 时间对齐算法
│   ├── wavelet_utils.py     # 小波去噪工具
│   ├── robust_stats.py      # 鲁棒统计与偏差检验
│   ├── kalman_filters.py    # 扩展卡尔曼滤波器
│   ├── motion_utils.py      # 运动状态计算（速度/加速度）
│   ├── constraint_checker.py# 约束条件检查器
│     └── task_scheduler.py    # 贪心任务调度器
├── visualization/       # 可视化模块
│   ├── plot_eda.py         # EDA 图表
│   ├── plot_trajectory.py  # 轨迹对比图
│   ├── plot_results.py     # 任务规划图
│   └── plot_publication.py # 论文级组合图
├── data/                # 数据目录（放置附件文件）
│   ├── 附件1.xlsx
│   ├── 附件2.xlsx
│   ├── 附件3.xlsx
│   └── 附件4.xlsx
└── output/              # 输出目录（自动生成）
    ├── Problem1_10Hz.xlsx
    ├── Problem2_10Hz.xlsx
    ├── Problem3_10Hz.xlsx
    ├── result.xlsx
    ├── cleaned_data.pkl
    └── figures/
环境依赖
Python >= 3.8
安装命令：
bash
运行
pip install numpy pandas matplotlib openpyxl scipy pywt
项目同时兼容 .xlsx 和 .csv 格式的数据文件。
快速开始
一键运行全流程
bash
运行
python main.py
按阶段运行
bash
运行
# 运行指定阶段（如阶段 0 和 1）
python main.py --stages 0 1

# 运行单个阶段
python main.py --stages 3

# 跳过可视化
python main.py --skip-viz
单独运行某个阶段
bash
运行
python stage0_eda.py
python stage1_problem1.py
python stage2_problem2.py
python stage3_problem3.py
python stage4_problem4.py
处理流程
阶段 0 — 数据探索与预处理
自动识别 Excel（双 sheet）/ CSV 格式
数据质量检查：时间单调性、采样间隔、坐标突跳检测
绘制原始轨迹并保存
序列化为 cleaned_data.pkl 供下游使用
阶段 1 — 无噪声时间对齐（问题 1）
加载附件 1 的两组传感器数据
互相关 + 最小二乘估计时间偏差
等权重融合输出 10Hz 轨迹
阶段 2 — 含噪声融合（问题 2）
去噪：小波去噪参数对比实验（db4/sym5 × universal/bayes = 4 种组合），以加速度方差为指标自动选优
时间对齐：估计时偏
偏差估计：中位数 vs 截尾均值对比，含 3σ 异常点检测与显著性检验
融合：EKF 融合，默认 R 与自适应 R 对比，选取残差方差更小的方案
阶段 3 — 实际数据处理（问题 3）
与阶段 2 流程一致，新增两阶段粗略时间偏移估计（轨迹形状匹配），处理大尺度时钟偏差
自适应观测噪声估计改进：MAD + 偏差补偿 + 上界约束
阶段 4 — 任务规划（问题 4）
读取融合轨迹与目标点坐标
计算速度、加速度等运动状态
基于距离 / 速率 / 加速度约束搜索所有可行时间窗口
贪心调度生成无冲突任务序列
输出 Excel 结果表 + 轨迹图 + 甘特图
配置说明
所有可调参数集中在 config.py 中，核心配置项：
表格
配置项	说明
data_path	数据文件路径与输出目录
time_config	目标输出频率、采样间隔等
alignment_config	时间对齐搜索范围与方法
filter_config	EKF 过程噪声 Q、观测噪声 R1/R2
TaskConfig	射击 / 拍照的距离、速率、加速度约束
输出结果
表格
文件	内容
Problem1_10Hz.xlsx	问题 1 融合轨迹（Time, X, Y）
Problem2_10Hz.xlsx	问题 2 融合轨迹 + 偏差序列
Problem3_10Hz.xlsx	问题 3 融合轨迹 + 偏差序列
result.xlsx	问题 4 任务执行时刻表
figures/	各阶段可视化图表
作者
Han_B1ng — 2026.05