# file: main.py
# @Author : Han_B1ng
# @Time : 2026/5/6 20:45
# @Description : 主入口：一键运行从数据预处理到可视化的全流程

"""
多源融合机器人定位项目 — 主入口
================================
使用方法::

    # 运行全部阶段 + 可视化
    python main.py

    # 仅运行阶段 0 和 1
    python main.py --stages 0 1

    # 运行全部阶段但跳过可视化
    python main.py --skip-viz

    # 运行单个阶段
    python main.py --stages 2
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

# ──────────────────────────────────────────────
#  确保项目根目录在 sys.path 中
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ──────────────────────────────────────────────
#  项目配置
# ──────────────────────────────────────────────
from config import (
    data_path,
    time_config,
    filter_config,
    alignment_config,
    task_config,
)

# ──────────────────────────────────────────────
#  日志设置
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  阶段名称映射
# ──────────────────────────────────────────────
STAGE_NAMES: Dict[int, str] = {
    0: "数据探索与预处理 (EDA)",
    1: "问题1 — 无噪声时间对齐",
    2: "问题2 — 含噪声+系统偏差融合",
    3: "问题3 — 实际数据处理",
    4: "问题4 — 任务规划与优化",
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
        ("cleaned_data.pkl", data_path.output_dir / "cleaned_data.pkl"),
    ],
    2: [
        ("附件1", data_path.path1),
        ("附件2", data_path.path2),
        ("附件3", data_path.path3),
        ("cleaned_data.pkl", data_path.output_dir / "cleaned_data.pkl"),
    ],
    3: [
        ("附件3", data_path.path3),
        ("cleaned_data.pkl", data_path.output_dir / "cleaned_data.pkl"),
    ],
    4: [
        ("附件4（目标点）", Path(data_path.file4) if isinstance(data_path.file4, str) else data_path.path4),
        ("cleaned_data.pkl", data_path.output_dir / "cleaned_data.pkl"),
    ],
}

# 阶段模块名映射（用于动态导入）
STAGE_MODULES: Dict[int, str] = {
    0: "stage0_eda",
    1: "stage1_problem1",
    2: "stage2_problem2",
    3: "stage3_problem3",
    4: "stage4_problem4",
}

# 阶段脚本文件名（subprocess 回退时使用）
STAGE_SCRIPTS: Dict[int, str] = {
    0: "stage0_eda.py",
    1: "stage1_problem1.py",
    2: "stage2_problem2.py",
    3: "stage3_problem3.py",
    4: "stage4_problem4.py",
}


# ============================================================
#  工具函数
# ============================================================
def _print_banner(text: str, width: int = 70, char: str = "═") -> None:
    """打印居中横幅标题。"""
    padding = max(0, (width - len(text) - 2) // 2)
    line = char * padding + f" {text} " + char * padding
    # 确保长度一致
    if len(line) < width:
        line += char * (width - len(line))
    logger.info("")
    logger.info(line)


def _print_separator(width: int = 70, char: str = "─") -> None:
    """打印分隔线。"""
    logger.info(char * width)


def _check_input_files(stage_num: int) -> bool:
    """检查指定阶段所需的输入文件是否存在。

    Parameters
    ----------
    stage_num : int
        阶段编号。

    Returns
    -------
    bool
        所有文件均存在时返回 True，否则返回 False。
    """
    checks = STAGE_INPUT_CHECKS.get(stage_num, [])
    all_ok = True
    for desc, fpath in checks:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.error(f"[Main] 缺少输入文件: {desc} → {fpath}")
            all_ok = False
    return all_ok


def _elapsed_str(seconds: float) -> str:
    """将秒数格式化为可读时间字符串。"""
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

    优先尝试导入对应模块并调用其 ``run()`` 函数；
    若模块无 ``run()`` 函数，则回退到 subprocess 方式执行脚本。

    Parameters
    ----------
    stage_num : int
        阶段编号 (0~4)。

    Returns
    -------
    bool
        成功返回 True，失败返回 False。

    Notes
    -----
    - 执行前自动检查输入文件。
    - 执行前后打印分隔线和耗时。
    """
    stage_name = STAGE_NAMES.get(stage_num, f"阶段 {stage_num}")

    _print_banner(f"Stage {stage_num}: {stage_name}")
    logger.info(f"[Main] 开始执行 Stage {stage_num} — {stage_name}")

    # ---- 输入文件检查 ----
    if not _check_input_files(stage_num):
        logger.error(f"[Main] Stage {stage_num} 输入文件缺失，跳过。")
        return False

    t_start = time.time()

    # ---- 方式一：尝试导入模块并调用 run() ----
    module_name = STAGE_MODULES.get(stage_num)
    if module_name is not None:
        try:
            import importlib
            module = importlib.import_module(module_name)

            if hasattr(module, "run") and callable(getattr(module, "run")):
                logger.info(f"[Main] 通过 {module_name}.run() 执行...")
                module.run()
                elapsed = time.time() - t_start
                logger.info(
                    f"[Main] Stage {stage_num} 完成  "
                    f"(耗时: {_elapsed_str(elapsed)})"
                )
                _print_separator()
                return True
            else:
                logger.info(
                    f"[Main] {module_name} 未提供 run() 函数，"
                    f"回退到 subprocess 执行..."
                )
        except ImportError as exc:
            logger.warning(
                f"[Main] 无法导入 {module_name}: {exc}，"
                f"回退到 subprocess 执行..."
            )
        except Exception as exc:
            elapsed = time.time() - t_start
            logger.error(
                f"[Main] Stage {stage_num} 执行异常 "
                f"(耗时: {_elapsed_str(elapsed)}): {exc}"
            )
            logger.debug(traceback.format_exc())
            _print_separator()
            return False

    # ---- 方式二：subprocess 回退 ----
    script_name = STAGE_SCRIPTS.get(stage_num)
    if script_name is None:
        logger.error(f"[Main] Stage {stage_num} 无对应脚本定义。")
        return False

    script_path = _PROJECT_ROOT / script_name
    if not script_path.exists():
        logger.error(f"[Main] 脚本文件不存在: {script_path}")
        return False

    logger.info(f"[Main] 通过 subprocess 执行: {script_name}")
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
                f"[Main] Stage {stage_num} 返回非零退出码: "
                f"{result.returncode} (耗时: {_elapsed_str(elapsed)})"
            )
            _print_separator()
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"[Main] Stage {stage_num} 执行超时 (>600s)。")
        _print_separator()
        return False
    except Exception as exc:
        logger.error(f"[Main] Stage {stage_num} subprocess 异常: {exc}")
        logger.debug(traceback.format_exc())
        _print_separator()
        return False

    elapsed = time.time() - t_start
    logger.info(
        f"[Main] Stage {stage_num} 完成  (耗时: {_elapsed_str(elapsed)})"
    )
    _print_separator()
    return True


# ============================================================
#  可视化执行
# ============================================================
def run_visualization() -> bool:
    """运行全部可视化模块，生成论文级图表。

    读取 ``output/`` 目录下的结果文件，调用 ``visualization/`` 子包中
    的绘图函数，图表保存至 ``output/figures/``。

    Returns
    -------
    bool
        成功返回 True，失败返回 False。

    Notes
    -----
    - 依赖各阶段产出的结果文件（pickle / xlsx）。
    - 若某个结果文件缺失，仅跳过对应图表，不中断整体流程。
    """
    _print_banner("Visualization: 生成全部图表")
    t_start = time.time()

    figures_dir: Path = data_path.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    # ---- 辅助：安全调用绘图函数 ----
    def _safe_call(func, description: str, **kwargs) -> None:
        nonlocal success_count, fail_count
        try:
            logger.info(f"[Viz] {description}...")
            func(**kwargs)
            success_count += 1
        except FileNotFoundError as exc:
            logger.warning(f"[Viz] {description} — 文件缺失，跳过: {exc}")
            fail_count += 1
        except KeyError as exc:
            logger.warning(f"[Viz] {description} — 数据字段缺失，跳过: {exc}")
            fail_count += 1
        except Exception as exc:
            logger.warning(f"[Viz] {description} — 失败: {exc}")
            logger.debug(traceback.format_exc())
            fail_count += 1

    # ==========================================================
    #  1. EDA 图表
    # ==========================================================
    try:
        import pickle
        pkl_path = data_path.output_dir / "cleaned_data.pkl"
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
                f"[Viz] cleaned_data.pkl 不存在 ({pkl_path})，跳过 EDA 图表。"
            )
    except Exception as exc:
        logger.warning(f"[Viz] EDA 可视化模块加载失败: {exc}")
        logger.debug(traceback.format_exc())

    # ==========================================================
    #  2. 轨迹对比图（问题 1/2/3）
    # ==========================================================
    try:
        import pickle
        import numpy as np
        from visualization.plot_trajectory import (
            plot_trajectory_comparison,
            plot_error_time_series,
            plot_velocity_profile,
        )

        pkl_path = data_path.output_dir / "cleaned_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                cleaned_data = pickle.load(f)

            # 尝试加载各问题的结果
            result_files = {
                1: data_path.output_dir / "result_problem1.pkl",
                2: data_path.output_dir / "result_problem2.pkl",
                3: data_path.output_dir / "result_problem3.pkl",
            }

            for pnum, rpath in result_files.items():
                if not rpath.exists():
                    logger.info(
                        f"[Viz] 问题 {pnum} 结果文件不存在 ({rpath.name})，"
                        f"跳过轨迹对比图。"
                    )
                    continue

                with open(rpath, "rb") as f:
                    result = pickle.load(f)

                # 轨迹对比
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
                    title=f"问题 {pnum} 轨迹对比",
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
                        title=f"问题 {pnum} 融合误差",
                    )

                # 速度曲线
                if "speed" in result:
                    _safe_call(
                        plot_velocity_profile,
                        f"速度曲线 — 问题 {pnum}",
                        t=result.get("t_speed", result.get("t_fused", np.array([]))),
                        speed=result["speed"],
                        save_path=figures_dir / f"velocity_p{pnum}.png",
                        title=f"问题 {pnum} 速度曲线",
                    )
        else:
            logger.warning("[Viz] cleaned_data.pkl 不存在，跳过轨迹对比图。")
    except Exception as exc:
        logger.warning(f"[Viz] 轨迹可视化模块加载失败: {exc}")
        logger.debug(traceback.format_exc())

    # ==========================================================
    #  3. 任务规划图（问题 4）
    # ==========================================================
    try:
        import pickle
        import numpy as np
        from visualization.plot_results import (
            plot_tasks_on_trajectory,
            plot_task_gantt,
            plot_heading_diversity,
        )

        result4_path = data_path.output_dir / "result_problem4.pkl"
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
                        f"航向角多样性 — 目标 {tid}",
                        target_id=tid,
                        headings=np.asarray(headings),
                        save_path=figures_dir / f"heading_diversity_t{tid}.png",
                    )
        else:
            logger.info(
                f"[Viz] 问题 4 结果文件不存在 ({result4_path.name})，"
                f"跳过任务规划图。"
            )
    except Exception as exc:
        logger.warning(f"[Viz] 任务规划可视化模块加载失败: {exc}")
        logger.debug(traceback.format_exc())

    # ==========================================================
    #  4. 论文级组合图
    # ==========================================================
    try:
        import pickle
        import numpy as np
        from visualization.plot_publication import (
            summary_figure_paper,
            task_planning_paper,
        )

        # 问题 1/2/3 组合图
        for pnum in (1, 2, 3):
            rpath = data_path.output_dir / f"result_problem{pnum}.pkl"
            if rpath.exists():
                with open(rpath, "rb") as f:
                    result = pickle.load(f)
                _safe_call(
                    summary_figure_paper,
                    f"论文组合图 — 问题 {pnum}",
                    problem_num=pnum,
                    data_dict=result,
                    save_path=str(figures_dir / "summary_p{}.png").format(pnum),
                )

        # 问题 4 双栏图
        result4_path = data_path.output_dir / "result_problem4.pkl"
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
                "论文双栏图 — 问题 4",
                traj_x=traj_x, traj_y=traj_y, tasks=tasks,
                save_path=figures_dir / "task_planning_p4.png",
                t=t_traj, targets=targets,
            )

    except Exception as exc:
        logger.warning(f"[Viz] 论文级图表模块加载失败: {exc}")
        logger.debug(traceback.format_exc())

    # ---- 汇总 ----
    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"[Viz] 可视化完成 — 成功: {success_count}, 失败/跳过: {fail_count}")
    logger.info(f"[Viz] 总耗时: {_elapsed_str(elapsed)}")
    _print_separator()

    return fail_count == 0


# ============================================================
#  主函数
# ============================================================
def main() -> None:
    """主入口：解析命令行参数，按序执行各阶段并生成图表。"""

    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(
        description="多源融合机器人定位项目 — 一键运行全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py                    # 运行全部阶段 + 可视化\n"
            "  python main.py --stages 0 1       # 仅运行阶段 0 和 1\n"
            "  python main.py --stages 2 3 4     # 运行阶段 2~4\n"
            "  python main.py --skip-viz          # 跳过可视化\n"
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["0", "1", "2", "3", "4"],
        default=["0", "1", "2", "3", "4"],
        help="要运行的阶段编号（默认: 0 1 2 3 4）",
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

    _print_banner("多源融合机器人定位项目 — 全流程")
    logger.info(f"[Main] 项目根目录: {_PROJECT_ROOT}")
    logger.info(f"[Main] 数据目录:   {data_path.data_dir}")
    logger.info(f"[Main] 输出目录:   {data_path.output_dir}")
    logger.info(f"[Main] 待执行阶段: {stages_to_run}")
    logger.info(f"[Main] 跳过可视化: {'是' if skip_viz else '否'}")
    _print_separator()

    # ---- 确保输出目录存在 ----
    data_path.output_dir.mkdir(parents=True, exist_ok=True)
    (data_path.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- 执行各阶段 ----
    stage_results: Dict[int, bool] = {}
    for stage_num in stages_to_run:
        try:
            ok = run_stage(stage_num)
            stage_results[stage_num] = ok
            if ok:
                logger.info(f"[Main] ✓ Stage {stage_num} 成功完成。")
            else:
                logger.warning(f"[Main] ✗ Stage {stage_num} 失败。")
                # 阶段 0 失败则终止（后续阶段依赖它）
                if stage_num == 0:
                    logger.error(
                        "[Main] Stage 0 是基础阶段，失败后无法继续。终止。"
                    )
                    break
        except KeyboardInterrupt:
            logger.warning(f"[Main] 用户中断，Stage {stage_num} 未完成。")
            stage_results[stage_num] = False
            break
        except Exception as exc:
            logger.error(
                f"[Main] Stage {stage_num} 未捕获异常: {exc}"
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
        logger.info("[Main] 已跳过可视化 (--skip-viz)。")

    # ---- 最终汇总 ----
    t_total = time.time() - t_total_start
    _print_banner("运行汇总")
    for sn in sorted(stage_results.keys()):
        status = "✓ 成功" if stage_results[sn] else "✗ 失败"
        logger.info(f"  Stage {sn} ({STAGE_NAMES.get(sn, '?')}): {status}")

    if not skip_viz:
        logger.info("  可视化:   已执行")
    else:
        logger.info("  可视化:   已跳过")

    logger.info(f"  总耗时:   {_elapsed_str(t_total)}")
    _print_separator()

    # 退出码：任何阶段失败则返回 1
    if not all(stage_results.values()):
        sys.exit(1)


# ============================================================
#  运行入口
# ============================================================
if __name__ == "__main__":
    main()
