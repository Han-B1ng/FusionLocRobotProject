# file: core/constraint_checker.py
# @Description : 射击与拍照任务的时间窗口约束检查（密集采样版，不合并窗口）

import numpy as np
from config import TaskConfig

class ConstraintChecker:
    """轨迹约束检查器（保留所有可行执行时刻，不做窗口合并）。

    严格依据附录约束：
    - 射击准备1.5s内：距离、速率、加速度均需满足限制。
    - 拍照准备0.5s内：距离、速率、加速度均需满足限制。

    输出参数：
        step_time: 执行时刻采样间隔（秒），默认0.5s，避免候选过密。
    """

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
        """返回所有可行执行时刻，每个时刻作为一个独立窗口。

        Parameters
        ----------
        targets : list of dict
            每个元素含 'id','x','y'。
        task_type : str
            'shoot' 或 'photo'。
        step_time : float
            执行时刻的采样间隔（秒），避免过于密集。
            实际采样步长 = int(step_time / self.dt) 并保证 >=1。

        Returns
        -------
        list of dict
            每个字典代表一个可行时刻，包含 target_id, t_start, t_exec,
            distance, speed, heading（仅photo）。
        """
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

        # 计算执行时刻采样步长
        step_steps = max(1, int(np.round(step_time / self.dt)))
        # 窗口采样：每隔 step_steps 取一个执行点
        raw_windows = []

        for target in targets:
            tid = target["id"]
            tx, ty = target["x"], target["y"]

            # 向量化距离
            dist = np.sqrt((self.x - tx) ** 2 + (self.y - ty) ** 2)

            # 全程约束掩码（距离 + 速度 + 加速度）
            valid = (
                (dist >= dist_min) & (dist <= dist_max) &
                (self.speed <= speed_max) &
                (self.acc <= acc_max)
            )

            # 卷积找连续满足准备时间的窗口起点
            if window_size <= 1:
                valid_starts = np.where(valid)[0]
            else:
                kernel = np.ones(window_size, dtype=np.int32)
                conv = np.convolve(valid.astype(np.int32), kernel, mode="valid")
                valid_starts = np.where(conv == window_size)[0]

            if len(valid_starts) == 0:
                continue

            # 执行时刻 = 起点 + 准备步数
            exec_indices = valid_starts + prep_steps
            # 确保不越界
            mask = exec_indices < self.n
            valid_starts = valid_starts[mask]
            exec_indices = exec_indices[mask]
            if len(valid_starts) == 0:
                continue

            # 按 step_steps 间隔下采样执行时刻
            sampled_exec_indices = exec_indices[::step_steps]
            # 对应的 t_start = 执行时刻 - prep_time
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

        # ---------- 不再合并连续窗口，直接返回所有采样点 ----------
        return raw_windows