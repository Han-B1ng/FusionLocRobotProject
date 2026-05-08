# file: core/constraint_checker.py
# @Description : 射击与拍照任务的时间窗口约束检查（严格遵循题目约束）

import numpy as np
from config import TaskConfig


class ConstraintChecker:
    """轨迹约束检查器（向量化 + 窗口合并）。

    严格依据附录约束：
    - 射击准备1.5s内：距离、速率、加速度均需满足限制。
    - 拍照准备0.5s内：距离、速率、加速度均需满足限制。
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

    # ------------------------------------------------------------------ #
    #  向量化搜索所有可行窗口（自动合并连续窗口，保证加速度检查）
    # ------------------------------------------------------------------ #
    def find_all_feasible_windows(self, targets, task_type="shoot"):
        """返回符合条件的窗口列表（已合并连续时间窗口）。

        Parameters
        ----------
        targets : list of dict
            每个元素含 'id','x','y'。
        task_type : str
            'shoot' 或 'photo'。

        Returns
        -------
        list of dict
            窗口信息，包含 target_id, t_start, t_exec, distance, speed, (heading)。
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

            # 卷积找连续满足的窗口
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

            t_starts = self.t[valid_starts]
            t_execs = self.t[exec_indices]
            distances = dist[exec_indices]
            speeds = self.speed[exec_indices]

            if task_type == "photo":
                dx = tx - self.x[exec_indices]
                dy = ty - self.y[exec_indices]
                headings = np.degrees(np.arctan2(dy, dx)) % 360.0
            else:
                headings = None

            for k in range(len(valid_starts)):
                win = {
                    "target_id": tid,
                    "t_start": round(float(t_starts[k]), 4),
                    "t_exec": round(float(t_execs[k]), 4),
                    "distance": round(float(distances[k]), 4),
                    "speed": round(float(speeds[k]), 4),
                }
                if headings is not None:
                    win["heading"] = round(float(headings[k]), 2)
                raw_windows.append(win)

        # ---------- 合并连续窗口 ----------
        if len(raw_windows) == 0:
            return []

        # 按目标分组
        by_target = {}
        for w in raw_windows:
            by_target.setdefault(w["target_id"], []).append(w)

        merged = []
        for tid, wins in by_target.items():
            wins.sort(key=lambda x: x["t_exec"])
            i = 0
            while i < len(wins):
                cluster = [wins[i]]
                j = i + 1
                while j < len(wins) and (wins[j]["t_exec"] - wins[j-1]["t_exec"] <= 2 * self.dt):
                    cluster.append(wins[j])
                    j += 1
                # 代表窗口：距离最近的点（执行时刻）
                best = min(cluster, key=lambda w: w["distance"])
                merged.append(best)
                i = j

        return merged