# file: core/constraint_checker.py

import numpy as np
from config import TaskConfig

class ConstraintChecker:

    def __init__(self, t, x, y, vx, vy, ax, ay):
        self.t = np.asarray(t, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.vx = np.asarray(vx, dtype=np.float64)
        self.vy = np.asarray(vy, dtype=np.float64)
        self.ax = np.asarray(ax, dtype=np.float64)
        self.ay = np.asarray(ay, dtype=np.float64)

        self.dt = float(np.median(np.diff(self.t)))
        self.n = len(self.t)

        self.speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        self.acc = np.sqrt(self.ax ** 2 + self.ay ** 2)

    def find_all_feasible_windows(self, targets, task_type="shoot",
                                  step_time: float = 0.5):
        if task_type == "shoot":
            prep_time = TaskConfig.SHOOT_PREP_TIME        # 1.5
            dist_min = TaskConfig.SHOOT_DIST_MIN          # 5
            dist_max = TaskConfig.SHOOT_DIST_MAX          # 30
            speed_max = TaskConfig.SHOOT_SPEED_MAX        # 2.0
            acc_max = TaskConfig.SHOOT_ACC_MAX            # 1.5
        elif task_type == "photo":
            prep_time = TaskConfig.PHOTO_PREP_TIME        # 0.5
            dist_min = TaskConfig.PHOTO_DIST_MIN          # 10
            dist_max = TaskConfig.PHOTO_DIST_MAX          # 40
            speed_max = TaskConfig.PHOTO_SPEED_MAX        # 1.0
            acc_max = TaskConfig.PHOTO_ACC_MAX            # 1.5
        else:
            raise ValueError("未知任务类型")

        prep_steps = int(np.round(prep_time / self.dt))
        window_size = prep_steps + 1   # 准备期间采样点数

        step_steps = max(1, int(np.round(step_time / self.dt)))
        raw_windows = []

        for target in targets:
            tid = target["id"]
            tx, ty = target["x"], target["y"]

            dist = np.sqrt((self.x - tx) ** 2 + (self.y - ty) ** 2)

            valid = (
                (dist >= dist_min) & (dist <= dist_max) &
                (self.speed <= speed_max) &
                (self.acc <= acc_max)
            )

            if window_size <= 1:
                valid_starts = np.where(valid)[0]
            else:
                kernel = np.ones(window_size, dtype=np.int32)
                conv = np.convolve(valid.astype(np.int32), kernel, mode="valid")
                valid_starts = np.where(conv == window_size)[0]

            if len(valid_starts) == 0:
                continue

            exec_indices = valid_starts + prep_steps
            mask = exec_indices < self.n
            valid_starts = valid_starts[mask]
            exec_indices = exec_indices[mask]
            if len(valid_starts) == 0:
                continue

            sampled_exec_indices = exec_indices[::step_steps]
            sampled_t_execs = self.t[sampled_exec_indices]
            sampled_t_starts = sampled_t_execs - prep_time
            sampled_distances = dist[sampled_exec_indices]
            sampled_speeds = self.speed[sampled_exec_indices]

            if task_type == "photo":
                dx = tx - self.x[sampled_exec_indices]
                dy = ty - self.y[sampled_exec_indices]
                headings = np.degrees(np.arctan2(dy, dx)) % 360.0
            else:
                headings = None

            for k in range(len(sampled_exec_indices)):
                win = {
                    "target_id": tid,
                    "t_start": round(float(sampled_t_starts[k]), 4),
                    "t_exec": round(float(sampled_t_execs[k]), 4),
                    "distance": round(float(sampled_distances[k]), 4),
                    "speed": round(float(sampled_speeds[k]), 4),
                }
                if headings is not None:
                    win["heading"] = round(float(headings[k]), 2)
                raw_windows.append(win)

        return raw_windows
