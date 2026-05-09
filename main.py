# file: main.py
# @Author : Han_B1ng
# @Time : 2026/5/6 20:45
# @Description : 主入口：一键运行从数据预处理到可视化的全流程

"""
╔══════════════════════════════════════════════════════╗
║  多源融合机器人定位项目 —— 主入口                      ║
╚══════════════════════════════════════════════════════╝

使用方法：
  python main.py                  # 运行全部阶段并可视化
  python main.py --stages 0 1     # 仅运行阶段0和1
  python main.py --skip-viz       # 运行全部阶段但跳过可视化
  python main.py --stages 2       # 运行单个阶段
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from matplotlib import pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
# 中文字体配置已由 config.py 统一处理
# 注意：seaborn 样式会覆盖 font.sans-serif，需在 config 导入后重新应用
# ── 环境准备：确保项目根目录在sys.path中 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 导入项目配置 ──
from config import (
    data_path,
    time_config,
    filter_config,
    alignment_config,
    task_config,
    plot_config,
    TABLE_DIR,
    PLOT_DIR,
    INTERMEDIATE_DIR,
    ensure_dirs,
)
plot_config.apply_style()  # seaborn 覆盖了字体，重新应用中文字体

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── 阶段元数据定义 ──
STAGE_NAMES: Dict[int, str] = {
    0: "数据探索与预处理（EDA）",
    1: "问题1——无噪声时间对齐",
    2: "问题2——含噪声与系统偏差融合",
    3: "问题3——实际数据处理",
    4: "问题4——任务规划与优化",
    5: "敏感性分析——Q矩阵参数扫描",
}

# 每个阶段运行前需要检查的输入文件
# 键为阶段编号，值为 (描述, 路径) 列表
STAGE_INPUT_CHECKS: Dict[int, List[tuple]] = {
    0: [
        ("附件1", data_path.path1),
        ("附件2", data_path.path2),
        ("附件3", data_path.path3),
    ],
    1: [
        ("附件1", data_path.path1),
        ("附件2", data_path.path2),
        ("附件3", data_path.path3),
        ("cleaned_data.pkl", Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"),
    ],
    2: [
        ("附件1", data_path.path1),
        ("附件2", data_path.path2),
        ("附件3", data_path.path3),
        ("cleaned_data.pkl", Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"),
    ],
    3: [
        ("附件3", data_path.path3),
        ("cleaned_data.pkl", Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"),
    ],
    4: [
        ("附件4（目标点）", Path(data_path.file4) if isinstance(data_path.file4, str) else data_path.path4),
        ("cleaned_data.pkl", Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"),
    ],
    5: [
        ("附件2", data_path.path2),
        ("cleaned_data.pkl", Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"),
    ],
}

# 阶段模块名映射（用于动态导入）
STAGE_MODULES: Dict[int, str] = {
    0: "stage0_eda",
    1: "stage1_problem1",
    2: "stage2_problem2",
    3: "stage3_problem3",
    4: "stage4_problem4",
    5: "sensitivity_analysis",
}

# 阶段脚本文件名（subprocess 回退时使用）
STAGE_SCRIPTS: Dict[int, str] = {
    0: "stage0_eda.py",
    1: "stage1_problem1.py",
    2: "stage2_problem2.py",
    3: "stage3_problem3.py",
    4: "stage4_problem4.py",
    5: "sensitivity_analysis.py",
}


# ============================================================
#  工具函数
# ============================================================
def _print_banner(text: str, width: int = 70, char: str = "═") -> None:
    """打印居中横幅标题（装饰用）。"""
    padding = max(0, (width - len(text) - 2) // 2)
    line = char * padding + f" {text} " + char * padding
    # 确保长度一致
    if len(line) < width:
        line += char * (width - len(line))
    logger.info("")
    logger.info(line)


def _print_separator(width: int = 70, char: str = "─") -> None:
    """打印水平分隔线。"""
    logger.info(char * width)


def _check_input_files(stage_num: int) -> bool:
    """检查指定阶段所需的输入文件是否全部存在。

    Parameters
    ----------
    stage_num : int
        阶段编号（0~4）。

    Returns
    -------
    bool
        全部存在返回True，否则返回False。
    """
    checks = STAGE_INPUT_CHECKS.get(stage_num, [])
    all_ok = True
    for desc, fpath in checks:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.error(f"[Main] 缺少输入文件：{desc} → {fpath}")
            all_ok = False
    return all_ok


def _elapsed_str(seconds: float) -> str:
    """将秒数格式化为可读时间字符串（如'2m 30.5s'）。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.1f}s"


# ============================================================
#  阶段执行
# ============================================================
def run_stage(stage_num: int) -> bool:
    """运行指定编号的处理阶段。

    执行策略：优先导入模块调用run()，若不存在则回退至subprocess执行脚本。

    Parameters
    ----------
    stage_num : int
        阶段编号（0~4）。

    Returns
    -------
    bool
        成功返回True，失败返回False。

    Notes
    -----
    - 执行前自动校验输入文件
    - 执行前后打印分隔线与耗时统计
    """
    stage_name = STAGE_NAMES.get(stage_num, f"阶段 {stage_num}")

    _print_banner(f"Stage {stage_num}: {stage_name}")
    logger.info(f"[Main] 开始执行阶段{stage_num}——{stage_name}")

    # ---- 输入文件检查 ----
    if not _check_input_files(stage_num):
        logger.error(f"[Main] 阶段{stage_num}输入文件缺失，跳过。")
        return False

    t_start = time.time()

    # ---- 方式一：尝试导入模块并调用 run() ----
    module_name = STAGE_MODULES.get(stage_num)
    if module_name is not None:
        try:
            import importlib
            module = importlib.import_module(module_name)

            if hasattr(module, "run") and callable(getattr(module, "run")):
                logger.info(f"[Main] 通过 {module_name}.run() 执行…")
                module.run()
                elapsed = time.time() - t_start
                logger.info(
                    f"[Main] 阶段{stage_num}完成（耗时：{_elapsed_str(elapsed)}）"
                )
                _print_separator()
                return True
            else:
                logger.info(
                    f"[Main] {module_name} 未提供 run() 函数，"
                    f"回退至subprocess执行…"
                )
        except ImportError as exc:
            logger.warning(
                f"[Main] 无法导入 {module_name}：{exc}，"
                f"回退至subprocess执行…"
            )
        except Exception as exc:
            elapsed = time.time() - t_start
            logger.error(
                f"[Main] 阶段{stage_num}执行异常："
                f"(耗时: {_elapsed_str(elapsed)}): {exc}"
            )
            logger.debug(traceback.format_exc())
            _print_separator()
            return False

    # ---- 方式二：subprocess 回退 ----
    script_name = STAGE_SCRIPTS.get(stage_num)
    if script_name is None:
        logger.error(f"[Main] 阶段{stage_num}无对应脚本定义。")
        return False

    script_path = _PROJECT_ROOT / script_name
    if not script_path.exists():
        logger.error(f"[Main] 脚本文件不存在：{script_path}")
        return False

    logger.info(f"[Main] 通过subprocess执行：{script_name}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        # 输出子进程日志
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info(f"  │ {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning(f"  │ [stderr] {line}")

        if result.returncode != 0:
            elapsed = time.time() - t_start
            logger.error(
                f"[Main] 阶段{stage_num}返回非零退出码："
                f"{result.returncode} (耗时: {_elapsed_str(elapsed)})"
            )
            _print_separator()
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"[Main] 阶段{stage_num}执行超时（>600 s）。")
        _print_separator()
        return False
    except Exception as exc:
        logger.error(f"[Main] 阶段{stage_num} subprocess异常：{exc}")
        logger.debug(traceback.format_exc())
        _print_separator()
        return False

    elapsed = time.time() - t_start
    logger.info(
        f"[Main] 阶段{stage_num}完成（耗时：{_elapsed_str(elapsed)}）"
    )
    _print_separator()
    return True


# ============================================================
#  可视化执行
# ============================================================
def run_visualization() -> bool:
    """运行全部可视化模块，生成论文级图表。

    读取output/目录下的结果文件，调用visualization/子包中的绘图函数，
    图表统一保存至output/plots/。

    Returns
    -------
    bool
        全部成功返回True，否则返回False。

    Notes
    -----
    - 依赖各阶段产出的结果文件（.pkl / .xlsx）
    - 若某个结果文件缺失，仅跳过对应图表，不中断整体流程
    """
    _print_banner("可视化：生成全部图表")
    t_start = time.time()

    # ---- 全局样式应用 ----
    try:
        plot_config.apply_style()
        logger.info("[Viz] 全局绘图样式已应用（plot_config）")
    except Exception as exc:
        logger.warning(f"[Viz] 样式应用失败，使用默认样式：{exc}")

    figures_dir: Path = Path(PLOT_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    # ---- 辅助：安全调用绘图函数 ----
    def _safe_call(func, description: str, **kwargs) -> None:
        nonlocal success_count, fail_count
        try:
            logger.info(f"[Viz] {description}…")
            func(**kwargs)
            success_count += 1
        except FileNotFoundError as exc:
            logger.warning(f"[Viz] {description}——文件缺失，跳过：{exc}")
            fail_count += 1
        except KeyError as exc:
            logger.warning(f"[Viz] {description}——数据字段缺失，跳过：{exc}")
            fail_count += 1
        except Exception as exc:
            logger.warning(f"[Viz] {description}——失败：{exc}")
            logger.debug(traceback.format_exc())
            fail_count += 1

    # ╔══ 1. EDA图表 ══╗
    try:
        import pickle
        pkl_path = Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                cleaned_data = pickle.load(f)

            from visualization.plot_eda import (
                plot_time_series,
                plot_sampling_interval_histogram,
                plot_missing_summary,
            )
            import numpy as np
            import pandas as pd

            # 按附件分组绘图
            att_groups: Dict[str, Dict[str, pd.DataFrame]] = {}
            for (att_name, sensor_label), df in cleaned_data.items():
                att_groups.setdefault(att_name, {})[sensor_label] = df

            for att_name, sensors in att_groups.items():
                # 时间序列
                t_list, x_list, y_list, labels = [], [], [], []
                for sname, sdf in sensors.items():
                    t_list.append(sdf["t"].values)
                    x_list.append(sdf["x"].values)
                    y_list.append(sdf["y"].values)
                    labels.append(sname)

                _safe_call(
                    plot_time_series,
                    f"EDA 时间序列 — {att_name}",
                    t=t_list, x=x_list, y=y_list, labels=labels,
                    title=f"{att_name} 原始时间序列",
                    save_path=figures_dir / f"{att_name}_timeseries.png",
                )

                # 采样间隔直方图
                for sname, sdf in sensors.items():
                    expected_dt = (
                        time_config.dt1 if "1" in sname else time_config.dt2
                    )
                    _safe_call(
                        plot_sampling_interval_histogram,
                        f"EDA 采样间隔 — {att_name}/{sname}",
                        t=sdf["t"].values,
                        expected_dt=expected_dt,
                        save_path=figures_dir / f"{att_name}_{sname}_dt_hist.png",
                        title=f"{att_name} / {sname} 采样间隔分布",
                    )

            # 缺失值汇总
            missing_records = {}
            for (att_name, sensor_label), df in cleaned_data.items():
                key = f"{att_name}/{sensor_label}"
                missing_records[key] = {
                    col: int(df[col].isnull().sum()) for col in df.columns
                }
            if missing_records:
                missing_df = pd.DataFrame(missing_records).T
                _safe_call(
                    plot_missing_summary,
                    "EDA 缺失值汇总",
                    missing_df=missing_df,
                    save_path=figures_dir / "missing_summary.png",
                )
        else:
            logger.warning(
                f"[Viz] cleaned_data.pkl 不存在（{pkl_path}），跳过EDA图表。"
            )
    except Exception as exc:
        logger.warning(f"[Viz] EDA可视化模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 2. 轨迹对比图（问题1/2/3） ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_trajectory import (
            plot_trajectory_comparison,
            plot_error_time_series,
            plot_velocity_profile,
        )

        pkl_path = Path(INTERMEDIATE_DIR) / "cleaned_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                cleaned_data = pickle.load(f)

            # 尝试加载各问题的结果
            result_files = {
                1: Path(INTERMEDIATE_DIR) / "result_problem1.pkl",
                2: Path(INTERMEDIATE_DIR) / "result_problem2.pkl",
                3: Path(INTERMEDIATE_DIR) / "result_problem3.pkl",
            }

            for pnum, rpath in result_files.items():
                if not rpath.exists():
                    logger.info(
                        f"[Viz] 问题{pnum}结果文件不存在（{rpath.name}），"
                        f"跳过轨迹对比图。"
                    )
                    continue

                with open(rpath, "rb") as f:
                    result = pickle.load(f)

                # 轨迹对比（仅问题1保留2D轨迹图，问题2/3使用3D轨迹图）
                if pnum == 1:
                    _safe_call(
                        plot_trajectory_comparison,
                        f"轨迹对比 — 问题 {pnum}",
                        t1=result.get("t1", np.array([])),
                        x1=result.get("x1", np.array([])),
                        y1=result.get("y1", np.array([])),
                        t2=result.get("t2", np.array([])),
                        x2=result.get("x2", np.array([])),
                        y2=result.get("y2", np.array([])),
                        t_fused=result.get("t_fused"),
                        x_fused=result.get("x_fused"),
                        y_fused=result.get("y_fused"),
                        t_ref=result.get("t_ref"),
                        x_ref=result.get("x_ref"),
                        y_ref=result.get("y_ref"),
                        save_path=figures_dir / f"trajectory_p{pnum}.png",
                        title=f"问题{pnum} 轨迹对比",
                    )

                # 误差时序
                if "error_x" in result and "error_y" in result:
                    _safe_call(
                        plot_error_time_series,
                        f"误差曲线 — 问题 {pnum}",
                        t=result.get("t_error", result.get("t_fused", np.array([]))),
                        error_x=result["error_x"],
                        error_y=result["error_y"],
                        save_path=figures_dir / f"error_p{pnum}.png",
                        title=f"问题{pnum} 融合误差",
                    )

                # 速度曲线
                if "speed" in result:
                    _safe_call(
                        plot_velocity_profile,
                        f"速度曲线 — 问题 {pnum}",
                        t=result.get("t_speed", result.get("t_fused", np.array([]))),
                        speed=result["speed"],
                        save_path=figures_dir / f"velocity_p{pnum}.png",
                        title=f"问题{pnum} 速度曲线",
                    )
        else:
            logger.warning("[Viz] cleaned_data.pkl 不存在，跳过轨迹对比图。")
    except Exception as exc:
        logger.warning(f"[Viz] 轨迹可视化模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 2.5 速度热力轨迹图 ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_trajectory import plot_velocity_heatmap_trajectory

        for pnum in (1, 2, 3):
            rpath = Path(INTERMEDIATE_DIR) / f"result_problem{pnum}.pkl"
            if not rpath.exists():
                continue

            with open(rpath, "rb") as f:
                result = pickle.load(f)

            x_fused = result.get("x_fused")
            y_fused = result.get("y_fused")
            speed = result.get("speed")

            if x_fused is None or y_fused is None or speed is None:
                continue

            # 速度序列长度可能与轨迹不一致，截取
            n = min(len(x_fused), len(y_fused), len(speed))
            _safe_call(
                plot_velocity_heatmap_trajectory,
                f"速度热力轨迹 — 问题 {pnum}",
                x=x_fused[:n], y=y_fused[:n], speed=speed[:n],
                save_path=figures_dir / f"velocity_heatmap_p{pnum}.png",
                title=f"问题{pnum} 速度热力轨迹",
            )
    except Exception as exc:
        logger.warning(f"[Viz] 速度热力轨迹模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 3. 任务规划图（问题4） ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_results import (
            plot_tasks_on_trajectory,
            plot_task_gantt,
            plot_heading_diversity,
        )

        result4_path = Path(INTERMEDIATE_DIR) / "result_problem4.pkl"
        if result4_path.exists():
            with open(result4_path, "rb") as f:
                result4 = pickle.load(f)

            traj_x = result4.get("traj_x", np.array([]))
            traj_y = result4.get("traj_y", np.array([]))
            t_traj = result4.get("t_fused", None)
            tasks = result4.get("tasks", [])

            # 加载目标点
            targets = None
            target_path = Path(data_path.file4)
            if not target_path.is_absolute():
                target_path = data_path.data_dir / target_path.name
            if target_path.exists():
                import pandas as pd
                df_targets = pd.read_excel(target_path, engine="openpyxl")
                targets = df_targets.values

            # 轨迹 + 任务标记
            _safe_call(
                plot_tasks_on_trajectory,
                "问题4 任务执行位置",
                traj_x=traj_x, traj_y=traj_y, tasks=tasks,
                save_path=figures_dir / "tasks_trajectory_p4.png",
                t=t_traj, targets=targets,
            )

            # 甘特图
            _safe_call(
                plot_task_gantt,
                "问题4 任务甘特图",
                tasks=tasks,
                save_path=figures_dir / "task_gantt_p4.png",
            )

            # 航向角多样性
            heading_data = result4.get("heading_diversity", {})
            if isinstance(heading_data, dict):
                for tid, headings in heading_data.items():
                    _safe_call(
                        plot_heading_diversity,
                        f"航向角多样性——目标{tid}",
                        target_id=tid,
                        headings=np.asarray(headings),
                        save_path=figures_dir / f"heading_diversity_t{tid}.png",
                    )
        else:
            logger.info(
                f"[Viz] 问题4结果文件不存在（{result4_path.name}），"
                f"跳过任务规划图。"
            )
    except Exception as exc:
        logger.warning(f"[Viz] 任务规划可视化模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 3.5 增强甘特图 + 约束漏斗图 ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_results import plot_task_gantt_enhanced

        result4_path = Path(INTERMEDIATE_DIR) / "result_problem4.pkl"
        if result4_path.exists():
            with open(result4_path, "rb") as f:
                result4 = pickle.load(f)

            tasks = result4.get("tasks", [])
            candidate_windows = result4.get("candidate_windows", None)

            # 增强甘特图
            _safe_call(
                plot_task_gantt_enhanced,
                "问题4 增强甘特图",
                tasks=tasks,
                save_path=figures_dir / "task_gantt_enhanced_p4.png",
                candidate_windows=candidate_windows,
            )
    except Exception as exc:
        logger.warning(f"[Viz] 增强甘特图模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ---- 约束漏斗图 ----
    try:
        import pandas as pd
        from visualization.plot_constraint_analysis import (
            plot_constraint_funnel,
            build_funnel_from_stage4,
        )

        # 优先从 result_problem4.pkl 构建漏斗
        result4_path = Path(INTERMEDIATE_DIR) / "result_problem4.pkl"
        funnel_built = False

        if result4_path.exists():
            with open(result4_path, "rb") as f:
                result4 = pickle.load(f)

            windows_shoot = result4.get("windows_shoot", [])
            windows_photo = result4.get("windows_photo", [])
            tasks = result4.get("tasks", [])
            all_targets = result4.get("all_targets", [])

            if windows_shoot or windows_photo:
                labels, counts = build_funnel_from_stage4(
                    windows_shoot, windows_photo, tasks, all_targets,
                )
                _safe_call(
                    plot_constraint_funnel,
                    "约束漏斗图",
                    stage_labels=labels, stage_counts=counts,
                    save_path=figures_dir / "constraint_funnel_p4.png",
                )
                funnel_built = True

        # 回退：从 constraint_stats.xlsx 读取
        if not funnel_built:
            stats_path = Path(TABLE_DIR) / "constraint_stats.xlsx"
            if stats_path.exists():
                df_stats = pd.read_excel(stats_path, engine="openpyxl")
                stage_map = {}
                for _, row in df_stats.iterrows():
                    stage = str(row.get("阶段", ""))
                    cat = str(row.get("类别", "全部"))
                    count = int(row.get("数量", 0))
                    if cat == "全部":
                        stage_map[stage] = count

                if len(stage_map) >= 2:
                    labels = list(stage_map.keys())
                    counts = list(stage_map.values())
                    _safe_call(
                        plot_constraint_funnel,
                        "约束漏斗图（从xlsx构建）",
                        stage_labels=labels, stage_counts=counts,
                        save_path=figures_dir / "constraint_funnel_p4.png",
                    )
    except Exception as exc:
        logger.warning(f"[Viz] 约束漏斗图模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 4. 论文级组合图 ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_publication import (
            summary_figure_paper,
            task_planning_paper,
        )

        # 问题 1/2/3 组合图
        for pnum in (1, 2, 3):
            rpath = Path(INTERMEDIATE_DIR) / f"result_problem{pnum}.pkl"
            if rpath.exists():
                with open(rpath, "rb") as f:
                    result = pickle.load(f)
                _safe_call(
                    summary_figure_paper,
                    f"论文组合图——问题{pnum}",
                    problem_num=pnum,
                    data_dict=result,
                    save_path=str(figures_dir / "summary_p{}.png").format(pnum),
                )

        # 问题 4 双栏图
        result4_path = Path(INTERMEDIATE_DIR) / "result_problem4.pkl"
        if result4_path.exists():
            with open(result4_path, "rb") as f:
                result4 = pickle.load(f)

            traj_x = result4.get("traj_x", np.array([]))
            traj_y = result4.get("traj_y", np.array([]))
            t_traj = result4.get("t_fused", None)
            tasks = result4.get("tasks", [])

            targets = None
            target_path = Path(data_path.file4)
            if not target_path.is_absolute():
                target_path = data_path.data_dir / target_path.name
            if target_path.exists():
                import pandas as pd
                df_targets = pd.read_excel(target_path, engine="openpyxl")
                targets = df_targets.values

            _safe_call(
                task_planning_paper,
                "论文双栏图——问题4",
                traj_x=traj_x, traj_y=traj_y, tasks=tasks,
                save_path=figures_dir / "task_planning_p4.png",
                t=t_traj, targets=targets,
            )

    except Exception as exc:
        logger.warning(f"[Viz] 论文级图表模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 5. 参数敏感性分析（若有扫描结果） ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_sensitivity_analysis import (
            plot_sensitivity_single,
            plot_sensitivity_heatmap,
            plot_tradeoff_curve,
        )

        # 检查是否有预计算的敏感性分析结果
        sa_path = Path(INTERMEDIATE_DIR) / "sensitivity_results.pkl"
        if sa_path.exists():
            with open(sa_path, "rb") as f:
                sa_data = pickle.load(f)

            # 单参数扫描
            if "single" in sa_data:
                for param_name, pdata in sa_data["single"].items():
                    _safe_call(
                        plot_sensitivity_single,
                        f"单参数敏感性 — {param_name}",
                        param_name=param_name,
                        param_values=pdata["param_values"],
                        metrics=pdata["metrics"],
                        save_path=figures_dir / f"sensitivity_{param_name}.png",
                        baseline_value=pdata.get("baseline"),
                        highlight_best=pdata.get("highlight_best"),
                    )

            # 双参数热力图
            if "heatmap" in sa_data:
                for key, hdata in sa_data["heatmap"].items():
                    _safe_call(
                        plot_sensitivity_heatmap,
                        f"双参数热力图 — {key}",
                        param1_name=hdata["param1_name"],
                        param1_values=hdata["param1_values"],
                        param2_name=hdata["param2_name"],
                        param2_values=hdata["param2_values"],
                        metric_matrix=hdata["metric_matrix"],
                        save_path=figures_dir / f"sensitivity_heatmap_{key}.png",
                        metric_name=hdata.get("metric_name", "指标值"),
                        baseline=hdata.get("baseline"),
                    )

            # Trade-off 曲线
            if "tradeoff" in sa_data:
                for key, tdata in sa_data["tradeoff"].items():
                    _safe_call(
                        plot_tradeoff_curve,
                        f"Trade-off — {key}",
                        x_metric=tdata["x_metric"],
                        y_metric=tdata["y_metric"],
                        param_values=tdata["param_values"],
                        x_label=tdata.get("x_label", "指标 X"),
                        y_label=tdata.get("y_label", "指标 Y"),
                        save_path=figures_dir / f"tradeoff_{key}.png",
                        param_label=tdata.get("param_label", "参数值"),
                        baseline_idx=tdata.get("baseline_idx"),
                    )
        else:
            logger.info(
                "[Viz] sensitivity_results.pkl 不存在，跳过敏感性分析图。"
                "（需先运行参数扫描并保存结果至该文件）"
            )
    except Exception as exc:
        logger.warning(f"[Viz] 敏感性分析模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ╔══ 6. 案例分析（Case Study） ══╗
    try:
        import pickle
        import numpy as np
        from visualization.plot_case_study import plot_case_study
        from config import task_config

        result4_path = Path(INTERMEDIATE_DIR) / "result_problem4.pkl"
        if result4_path.exists():
            with open(result4_path, "rb") as f:
                result4 = pickle.load(f)

            traj_x = result4.get("traj_x", None)
            traj_y = result4.get("traj_y", None)
            t_fused = result4.get("t_fused", None)
            speed = result4.get("speed", None)
            acc = result4.get("acc", None)
            tasks = result4.get("tasks", [])
            all_targets = result4.get("all_targets", [])

            if (traj_x is not None and traj_y is not None
                    and t_fused is not None and speed is not None
                    and acc is not None and len(tasks) > 0):

                id_to_target = {tgt["id"]: tgt for tgt in all_targets}

                # 选前 3 个任务做案例分析
                for task in tasks[:3]:
                    tid = task.get("target_id")
                    tgt = id_to_target.get(tid)
                    if tgt is None:
                        continue

                    task_type = task.get("task_type", "shoot")
                    t_exec = task.get("t_exec", task.get("t_execute"))

                    # 约束参数
                    if task_type == "shoot":
                        dmin, dmax = task_config.SHOOT_DIST_MIN, task_config.SHOOT_DIST_MAX
                        vlim = task_config.SHOOT_SPEED_MAX
                    else:
                        dmin, dmax = task_config.PHOTO_DIST_MIN, task_config.PHOTO_DIST_MAX
                        vlim = task_config.PHOTO_SPEED_MAX

                    _safe_call(
                        plot_case_study,
                        f"案例分析 — {tgt.get('name', f'T{tid}')} {task_type}",
                        t=t_fused, x=traj_x, y=traj_y,
                        speed=speed, acc=acc,
                        target_x=tgt["x"], target_y=tgt["y"],
                        target_name=tgt.get("name", f"T{tid}"),
                        task_type=task_type,
                        t_exec=t_exec,
                        window=30.0,
                        dist_min=dmin, dist_max=dmax,
                        speed_limit=vlim,
                        save_path=figures_dir / f"case_study_{tgt.get('name', f'T{tid}')}_{task_type}.png",
                    )
        else:
            logger.info(
                "[Viz] result_problem4.pkl 不存在，跳过案例分析图。"
            )
    except Exception as exc:
        logger.warning(f"[Viz] 案例分析模块加载失败：{exc}")
        logger.debug(traceback.format_exc())

    # ---- 汇总 ----
    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"[Viz] 可视化完成——成功：{success_count}，失败/跳过：{fail_count}")
    logger.info(f"[Viz] 总耗时：{_elapsed_str(elapsed)}")
    _print_separator()

    return fail_count == 0


# ============================================================
#  主函数
# ============================================================
def main() -> None:
    """主入口：解析命令行参数，按序执行各阶段并生成可视化图表。"""

    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(
        description="多源融合机器人定位项目——一键运行全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py                    # 运行全部阶段并可视化\n"
            "  python main.py --stages 0 1       # 仅运行阶段0和1\n"
            "  python main.py --stages 2 3 4     # 运行阶段2~4\n"
            "  python main.py --stages 5         # 仅运行敏感性分析\n"
            "  python main.py --skip-viz          # 跳过可视化\n"
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["0", "1", "2", "3", "4", "5"],
        default=["0", "1", "2", "3", "4", "5"],
        help="要运行的阶段编号（默认：0 1 2 3 4 5）",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        default=False,
        help="跳过可视化步骤",
    )
    args = parser.parse_args()

    stages_to_run = [int(s) for s in args.stages]
    skip_viz = args.skip_viz

    # ---- 总体计时 ----
    t_total_start = time.time()

    _print_banner("多源融合机器人定位项目——全流程")
    logger.info(f"[Main] 项目根目录：{_PROJECT_ROOT}")
    logger.info(f"[Main] 数据目录：{data_path.data_dir}")
    logger.info(f"[Main] 输出目录：{data_path.output_dir}")
    logger.info(f"[Main] 待执行阶段：{stages_to_run}")
    logger.info(f"[Main] 跳过可视化：{'是' if skip_viz else '否'}")
    _print_separator()

    # ---- 确保输出目录存在 ----
    ensure_dirs()

    # ---- 执行各阶段 ----
    stage_results: Dict[int, bool] = {}
    for stage_num in stages_to_run:
        try:
            ok = run_stage(stage_num)
            stage_results[stage_num] = ok
            if ok:
                logger.info(f"[Main] ✓ 阶段{stage_num}成功完成。")
            else:
                logger.warning(f"[Main] ✗ 阶段{stage_num}失败。")
                # 阶段 0 失败则终止（后续阶段依赖它）
                if stage_num == 0:
                    logger.error(
                        "[Main] 阶段0是基础阶段，失败后无法继续，终止。"
                    )
                    break
        except KeyboardInterrupt:
            logger.warning(f"[Main] 用户中断，阶段{stage_num}未完成。")
            stage_results[stage_num] = False
            break
        except Exception as exc:
            logger.error(
                f"[Main] 阶段{stage_num}未捕获异常：{exc}"
            )
            logger.debug(traceback.format_exc())
            stage_results[stage_num] = False

    # ---- 可视化 ----
    if not skip_viz:
        # 至少有一个阶段成功才执行可视化
        if any(stage_results.values()):
            try:
                run_visualization()
            except KeyboardInterrupt:
                logger.warning("[Main] 用户中断可视化。")
            except Exception as exc:
                logger.error(f"[Main] 可视化异常: {exc}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning("[Main] 所有阶段均失败，跳过可视化。")
    else:
        logger.info("[Main] 已跳过可视化（--skip-viz）。")

    # ---- 最终汇总 ----
    t_total = time.time() - t_total_start
    _print_banner("运行汇总")
    for sn in sorted(stage_results.keys()):
        status = "✓ 成功" if stage_results[sn] else "✗ 失败"
        logger.info(f"  阶段{sn}（{STAGE_NAMES.get(sn, '?')}）：{status}")

    if not skip_viz:
        logger.info("  可视化：已执行")
    else:
        logger.info("  可视化：已跳过")

    logger.info(f"  总耗时：{_elapsed_str(t_total)}")
    _print_separator()

    # 退出码：任何阶段失败则返回 1
    if not all(stage_results.values()):
        sys.exit(1)


# ============================================================
#  运行入口
# ============================================================
if __name__ == "__main__":
    main()