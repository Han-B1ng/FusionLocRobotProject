# file: core/task_scheduler.py
# @Description : 射击与拍照任务的优化调度

"""
任务调度模块。

基于 ConstraintChecker 产出的可行时间窗口，对射击和拍照任务
进行贪心调度，使在不冲突的前提下安排尽可能多的任务。

核心约束：
  - 每个目标最多完成一次射击任务
  - 每个目标的拍照次数不超过规定上限（默认 3 次）
  - 同一目标的任意两次拍照方向角差异 ≥ 60°
  - 相邻任务的时间窗口不得重叠（前一个 t_exec ≤ 后一个 t_start）

扩展接口 schedule_tasks_optimize 可替换为遗传算法等全局优化方法。
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from config import TaskConfig


# ============================================================
#  冲突检查
# ============================================================
def _has_conflict(
    scheduled: list,
    t_start: float,
    t_exec: float,
) -> bool:
    """检查新窗口是否与已安排任务存在时间重叠。

    冲突定义：新任务的准备起始时间 < 已安排任务的执行结束时间，
    且已安排任务的准备起始时间 < 新任务的执行结束时间。

    简化约束：相邻任务之间必须满足
        上一个 t_exec ≤ 下一个 t_start
    即按 t_exec 排序后，只需检查与最后一个已安排任务的间隔。

    Parameters
    ----------
    scheduled : list of dict
        已安排任务列表（按 t_exec 升序排列）。
    t_start : float
        新窗口的准备起始时刻 (s)。
    t_exec : float
        新窗口的执行结束时刻 (s)。

    Returns
    -------
    bool
        存在冲突返回 True，否则 False。
    """
    if len(scheduled) == 0:
        return False

    # 取最后一个已安排任务的执行时刻
    last_exec = scheduled[-1]["t_execute"]

    # 新任务必须在上一个任务执行完毕之后才能开始准备
    if t_start < last_exec:
        return True

    return False


# ============================================================
#  贪心调度
# ============================================================
def schedule_tasks_greedy(
    windows: list,
    targets_photo: list | None = None,
    max_photo_per_target: int | None = None,
) -> list:
    """贪心算法调度射击与拍照任务。

    算法流程：
      1. 按 t_exec 升序排列所有可行窗口。
      2. 维护已射击目标集合、已拍照目标角度字典。
      3. 依次遍历窗口：
         a. 射击：目标未射击过 且 无时间冲突 → 安排。
         b. 拍照：目标拍照次数未超限 且 方向角与已有角度差异 ≥ 60°
                  且 无时间冲突 → 安排，记录方向角。
      4. 返回安排结果。

    Parameters
    ----------
    windows : list of dict
        ConstraintChecker 产出的可行窗口列表，每个元素须包含：
          - target_id : 目标 ID
          - task_type : str, 'shoot' 或 'photo'
          - t_start   : 准备起始时刻 (s)
          - t_exec    : 执行结束时刻 (s)
          - distance  : 执行时刻到目标距离 (m)
          - speed     : 执行时刻速率 (m/s)
          - heading   : 方向角 (°)，仅拍照任务
    targets_photo : list of dict, optional
        拍照目标列表，每个元素含 'id' 键。
        用于初始化拍照目标的角度跟踪字典。若为 None，
        则从 windows 中自动提取所有拍照目标 ID。
    max_photo_per_target : int, optional
        每个目标最大拍照次数，默认取 TaskConfig.PHOTO_MAX_PER_TARGET。

    Returns
    -------
    list of dict
        已安排任务列表（按执行时间升序），每个元素包含：
          - target_id   : 目标 ID
          - task_type    : 'shoot' 或 'photo'
          - t_start_prep : 准备起始时刻 (s)
          - t_execute    : 执行结束时刻 (s)
          - distance     : 执行时刻距离 (m)
          - speed        : 执行时刻速率 (m/s)
          - heading      : 方向角 (°)，仅拍照
    """
    if max_photo_per_target is None:
        max_photo_per_target = getattr(
            TaskConfig, "PHOTO_MAX_PER_TARGET", 3
        )

    # ---- 初始化跟踪状态 ----
    shot_targets: set = set()                     # 已完成射击的目标
    photo_headings: dict[int, list[float]] = {}   # {target_id: [heading, ...]}
    photo_counts: dict[int, int] = {}             # {target_id: 已拍次数}

    # 从 targets_photo 初始化角度字典
    if targets_photo is not None:
        for tp in targets_photo:
            tid = tp["id"]
            photo_headings.setdefault(tid, [])
            photo_counts.setdefault(tid, 0)

    # ---- 按 t_exec 排序 ----
    sorted_windows = sorted(windows, key=lambda w: w["t_exec"])

    scheduled: list[dict[str, Any]] = []

    # ---- 逐窗口遍历 ----
    for win in sorted_windows:
        tid = win["target_id"]
        task_type = win["task_type"]
        t_start = win["t_start"]
        t_exec = win["t_exec"]

        # 时间冲突检查
        if _has_conflict(scheduled, t_start, t_exec):
            continue

        # ---------- 射击任务 ----------
        if task_type == "shoot":
            # 每个目标最多射击一次
            if tid in shot_targets:
                continue

            shot_targets.add(tid)
            scheduled.append({
                "target_id": tid,
                "task_type": "shoot",
                "t_start_prep": round(t_start, 4),
                "t_execute": round(t_exec, 4),
                "distance": win.get("distance", None),
                "speed": win.get("speed", None),
            })

        # ---------- 拍照任务 ----------
        elif task_type == "photo":
            # 初始化（若 targets_photo 未提供）
            photo_headings.setdefault(tid, [])
            photo_counts.setdefault(tid, 0)

            # 拍照次数上限
            if photo_counts[tid] >= max_photo_per_target:
                continue

            # 方向角差异检查
            current_heading = win.get("heading", None)
            if current_heading is None:
                continue

            too_close = False
            for prev_h in photo_headings[tid]:
                diff = abs(current_heading - prev_h)
                angular_diff = min(diff, 360.0 - diff)
                if angular_diff < TaskConfig.PHOTO_HEADING_DIFF_MIN:
                    too_close = True
                    break

            if too_close:
                continue

            # 安排拍照
            photo_headings[tid].append(current_heading)
            photo_counts[tid] += 1

            scheduled.append({
                "target_id": tid,
                "task_type": "photo",
                "t_start_prep": round(t_start, 4),
                "t_execute": round(t_exec, 4),
                "distance": win.get("distance", None),
                "speed": win.get("speed", None),
                "heading": round(current_heading, 2),
            })

        else:
            # 未知任务类型，跳过
            continue

    # 按执行时间排序（理论上已有序，保险起见）
    scheduled.sort(key=lambda s: s["t_execute"])

    return scheduled


# ============================================================
#  优化调度（扩展接口）
# ============================================================
def schedule_tasks_optimize(
    windows: list,
    target_groups: dict | None = None,
    method: str = "greedy",
    **kwargs,
) -> list:
    """任务调度的统一入口，支持多种优化策略。

    当前实现：
      - method='greedy'：调用 schedule_tasks_greedy
      - 其他：预留接口，后续可扩展遗传算法、模拟退火等

    Parameters
    ----------
    windows : list of dict
        可行窗口列表，格式同 schedule_tasks_greedy。
    target_groups : dict, optional
        目标分组信息，格式示例：
          {
              'shoot': [{'id': 1, 'x': 500, 'y': 300}, ...],
              'photo': [{'id': 2, 'x': 620, 'y': 450}, ...],
          }
        当前仅用于提取 targets_photo 传递给贪心算法。
    method : str
        优化方法名称，默认 'greedy'。
    **kwargs
        传递给具体调度算法的额外参数。

    Returns
    -------
    list of dict
        已安排任务列表，格式同 schedule_tasks_greedy 返回值。

    Raises
    ------
    ValueError
        method 不是已知的优化方法时抛出。
    """
    known_methods = ("greedy",)

    if method not in known_methods:
        raise ValueError(
            f"未知优化方法: '{method}'，"
            f"当前支持: {known_methods}"
        )

    # 从 target_groups 提取拍照目标
    targets_photo = None
    if target_groups is not None and "photo" in target_groups:
        targets_photo = target_groups["photo"]

    if method == "greedy":
        return schedule_tasks_greedy(
            windows,
            targets_photo=targets_photo,
            **kwargs,
        )

    # 后续扩展点：遗传算法、模拟退火等
    # elif method == "genetic":
    #     return schedule_tasks_genetic(windows, target_groups, **kwargs)

    # 理论上不会到达此处
    return []
