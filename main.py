# file: main.py


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
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


STAGE_NAMES: Dict[int, str] = {
    0: "数据探索与预处理（EDA）",
    1: "问题1——无噪声时间对齐",
    2: "问题2——含噪声与系统偏差融合",
    3: "问题3——实际数据处理",
    4: "问题4——任务规划与优化",
    5: "敏感性分析——Q矩阵参数扫描",
}

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

STAGE_MODULES: Dict[int, str] = {
    0: "stage0_eda",
    1: "stage1_problem1",
    2: "stage2_problem2",
    3: "stage3_problem3",
    4: "stage4_problem4",
    5: "sensitivity_analysis",
}

STAGE_SCRIPTS: Dict[int, str] = {
    0: "stage0_eda.py",
    1: "stage1_problem1.py",
    2: "stage2_problem2.py",
    3: "stage3_problem3.py",
    4: "stage4_problem4.py",
    5: "sensitivity_analysis.py",
}


def _print_banner(text: str, width: int = 70, char: str = "═") -> None:
    padding = max(0, (width - len(text) - 2) // 2)
    line = char * padding + f" {text} " + char * padding
    if len(line) < width:
        line += char * (width - len(line))
    logger.info("")
    logger.info(line)


def _print_separator(width: int = 70, char: str = "─") -> None:
    logger.info(char * width)


def _check_input_files(stage_num: int) -> bool:
    checks = STAGE_INPUT_CHECKS.get(stage_num, [])
    all_ok = True
    for desc, fpath in checks:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.error(f"[Main] 缺少输入文件：{desc} → {fpath}")
            all_ok = False
    return all_ok


def _elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.1f}s"


def run_stage(stage_num: int) -> bool:
    stage_name = STAGE_NAMES.get(stage_num, f"阶段 {stage_num}")

    _print_banner(f"Stage {stage_num}: {stage_name}")
    logger.info(f"[Main] 开始执行阶段{stage_num}——{stage_name}")

    if not _check_input_files(stage_num):
        logger.error(f"[Main] 阶段{stage_num}输入文件缺失，跳过。")
        return False

    t_start = time.time()

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


def run_visualization() -> bool:
    _print_banner("可视化：生成全部图表")
    t_start = time.time()

    try:
        plot_config.apply_style()
        logger.info("[Viz] 全局绘图样式已应用（plot_config）")
    except Exception as exc:
        logger.warning(f"[Viz] 样式应用失败，使用默认样式：{exc}")

    figures_dir: Path = Path(PLOT_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

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

            att_groups: Dict[str, Dict[str, pd.DataFrame]] = {}
            for (att_name, sensor_label), df in cleaned_data.items():
                att_groups.setdefault(att_name, {})[sensor_label] = df

            for att_name, sensors in att_groups.items():
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

            targets = None
            target_path = Path(data_path.file4)
            if not target_path.is_absolute():
                target_path = data_path.data_dir / target_path.name
            if target_path.exists():
                import pandas as pd
                df_targets = pd.read_excel(target_path, engine="openpyxl")
                targets = df_targets.values

            _safe_call(
                plot_tasks_on_trajectory,
                "问题4 任务执行位置",
                traj_x=traj_x, traj_y=traj_y, tasks=tasks,
                save_path=figures_dir / "tasks_trajectory_p4.png",
                t=t_traj, targets=targets,
            )

            _safe_call(
                plot_task_gantt,
                "问题4 任务甘特图",
                tasks=tasks,
                save_path=figures_dir / "task_gantt_p4.png",
            )

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

    try:
        import pandas as pd
        from visualization.plot_constraint_analysis import (
            plot_constraint_funnel,
            build_funnel_from_stage4,
        )

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

    try:
        import pickle
        import numpy as np
        from visualization.plot_publication import (
            summary_figure_paper,
            task_planning_paper,
        )

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

    try:
        import pickle
        import numpy as np
        from visualization.plot_sensitivity_analysis import (
            plot_sensitivity_single,
            plot_sensitivity_heatmap,
            plot_tradeoff_curve,
        )

        sa_path = Path(INTERMEDIATE_DIR) / "sensitivity_results.pkl"
        if sa_path.exists():
            with open(sa_path, "rb") as f:
                sa_data = pickle.load(f)

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

                for task in tasks[:3]:
                    tid = task.get("target_id")
                    tgt = id_to_target.get(tid)
                    if tgt is None:
                        continue

                    task_type = task.get("task_type", "shoot")
                    t_exec = task.get("t_exec", task.get("t_execute"))

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

    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"[Viz] 可视化完成——成功：{success_count}，失败/跳过：{fail_count}")
    logger.info(f"[Viz] 总耗时：{_elapsed_str(elapsed)}")
    _print_separator()

    return fail_count == 0


def main() -> None:

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

    t_total_start = time.time()

    _print_banner("多源融合机器人定位项目——全流程")
    logger.info(f"[Main] 项目根目录：{_PROJECT_ROOT}")
    logger.info(f"[Main] 数据目录：{data_path.data_dir}")
    logger.info(f"[Main] 输出目录：{data_path.output_dir}")
    logger.info(f"[Main] 待执行阶段：{stages_to_run}")
    logger.info(f"[Main] 跳过可视化：{'是' if skip_viz else '否'}")
    _print_separator()

    ensure_dirs()

    stage_results: Dict[int, bool] = {}
    for stage_num in stages_to_run:
        try:
            ok = run_stage(stage_num)
            stage_results[stage_num] = ok
            if ok:
                logger.info(f"[Main] ✓ 阶段{stage_num}成功完成。")
            else:
                logger.warning(f"[Main] ✗ 阶段{stage_num}失败。")
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

    if not skip_viz:
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

    if not all(stage_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
