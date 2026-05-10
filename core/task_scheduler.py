# file: core/task_scheduler.py


from __future__ import annotations

import copy
from typing import Any

import numpy as np

from config import TaskConfig


def _has_conflict(
    scheduled: list,
    t_start: float,
    t_exec: float,
) -> bool:
    if len(scheduled) == 0:
        return False

    last_exec = scheduled[-1]["t_execute"]

    if t_start < last_exec:
        return True

    return False


def schedule_tasks_greedy(
    windows: list,
    targets_photo: list | None = None,
    max_photo_per_target: int | None = None,
) -> list:
    if max_photo_per_target is None:
        max_photo_per_target = getattr(
            TaskConfig, "PHOTO_MAX_PER_TARGET", 3
        )

    shot_targets: set = set()                     # 已完成射击的目标
    photo_headings: dict[int, list[float]] = {}   # {target_id: [heading, ...]}
    photo_counts: dict[int, int] = {}             # {target_id: 已拍次数}

    if targets_photo is not None:
        for tp in targets_photo:
            tid = tp["id"]
            photo_headings.setdefault(tid, [])
            photo_counts.setdefault(tid, 0)

    sorted_windows = sorted(windows, key=lambda w: w["t_exec"])

    scheduled: list[dict[str, Any]] = []

    for win in sorted_windows:
        tid = win["target_id"]
        task_type = win["task_type"]
        t_start = win["t_start"]
        t_exec = win["t_exec"]

        if _has_conflict(scheduled, t_start, t_exec):
            continue

        if task_type == "shoot":
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

        elif task_type == "photo":
            photo_headings.setdefault(tid, [])
            photo_counts.setdefault(tid, 0)

            if photo_counts[tid] >= max_photo_per_target:
                continue

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
            continue

    scheduled.sort(key=lambda s: s["t_execute"])

    return scheduled


def schedule_tasks_optimize(
    windows: list,
    target_groups: dict | None = None,
    method: str = "greedy",
    **kwargs,
) -> list:
    known_methods = ("greedy",)

    if method not in known_methods:
        raise ValueError(
            f"未知优化方法: '{method}'，"
            f"当前支持: {known_methods}"
        )

    targets_photo = None
    if target_groups is not None and "photo" in target_groups:
        targets_photo = target_groups["photo"]

    if method == "greedy":
        return schedule_tasks_greedy(
            windows,
            targets_photo=targets_photo,
            **kwargs,
        )


    return []
