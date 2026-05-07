# file: core/constraint_checker.py
# @Description : 射击与拍照任务的时间窗口约束检查（向量化版）

"""
约束检查模块（向量化优化）。

对融合轨迹的每个可能时间窗口，检查射击 / 拍照任务的全部物理约束：
  - 距离范围
  - 速率上限
  - 加速度上限

优化策略：
  - 距离、速率、加速度对全部时间点一次性向量化计算
  - 用卷积滑动窗口替代逐时间步 Python 循环
  - 仅对目标数（≤18）做外层循环
"""

import numpy as np

from config import TaskConfig


class ConstraintChecker:
    """轨迹约束检查器（向量化版）。

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        等间距时间序列 (s)。
    x, y : np.ndarray, shape (N,)
        轨迹坐标 (m)。
    vx, vy : np.ndarray, shape (N,)
        速度分量 (m/s)。
    ax, ay : np.ndarray, shape (N,)
        加速度分量 (m/s²)。
    """

    def __init__(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
    ) -> None:
        self.t = np.asarray(t, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.vx = np.asarray(vx, dtype=np.float64)
        self.vy = np.asarray(vy, dtype=np.float64)
        self.ax = np.asarray(ax, dtype=np.float64)
        self.ay = np.asarray(ay, dtype=np.float64)

        self.dt = float(np.median(np.diff(self.t)))
        self.n = len(self.t)

        # 预计算速率与加速度大小
        self.speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        self.acc = np.sqrt(self.ax ** 2 + self.ay ** 2)

    # ------------------------------------------------------------------ #
    #  单窗口检查（保留供外部单独调用）
    # ------------------------------------------------------------------ #
    def _time_to_index(self, time_val: float) -> int:
        idx = int(np.searchsorted(self.t, time_val, side="left"))
        if idx >= self.n:
            idx = self.n - 1
        if idx > 0 and abs(self.t[idx - 1] - time_val) < abs(self.t[idx] - time_val):
            idx -= 1
        return idx

    def _window_indices(self, t_start: float, t_end: float) -> np.ndarray:
        i_start = self._time_to_index(t_start)
        i_end = self._time_to_index(t_end)
        if i_end < i_start:
            i_end = i_start
        return np.arange(i_start, i_end + 1)

    def _distance_at(self, idx, tx, ty):
        return np.sqrt((self.x[idx] - tx) ** 2 + (self.y[idx] - ty) ** 2)

    def _heading_at(self, idx, tx, ty):
        dx = tx - self.x[idx]
        dy = ty - self.y[idx]
        return np.degrees(np.arctan2(dy, dx)) % 360.0

    def check_shooting_window(self, target_x, target_y, t_start) -> bool:
        t_exec = t_start + TaskConfig.SHOOT_PREP_TIME
        if t_start < self.t[0] or t_exec > self.t[-1]:
            return False
        idx = self._window_indices(t_start, t_exec)
        dist = self._distance_at(idx, target_x, target_y)
        if np.any(dist < TaskConfig.SHOOT_DIST_MIN) or np.any(dist > TaskConfig.SHOOT_DIST_MAX):
            return False
        if np.any(self.speed[idx] > TaskConfig.SHOOT_SPEED_MAX):
            return False
        if np.any(self.acc[idx] > TaskConfig.SHOOT_ACC_MAX):
            return False
        return True

    def check_photo_window(self, target_x, target_y, t_start,
                           previous_headings=None) -> bool:
        if previous_headings is None:
            previous_headings = []
        t_exec = t_start + TaskConfig.PHOTO_PREP_TIME
        if t_start < self.t[0] or t_exec > self.t[-1]:
            return False
        idx = self._window_indices(t_start, t_exec)
        dist = self._distance_at(idx, target_x, target_y)
        if np.any(dist < TaskConfig.PHOTO_DIST_MIN) or np.any(dist > TaskConfig.PHOTO_DIST_MAX):
            return False
        if np.any(self.speed[idx] > TaskConfig.PHOTO_SPEED_MAX):
            return False
        if np.any(self.acc[idx] > TaskConfig.PHOTO_ACC_MAX):
            return False
        if len(previous_headings) > 0:
            i_exec = self._time_to_index(t_exec)
            h = float(self._heading_at(np.array([i_exec]), target_x, target_y)[0])
            for ph in previous_headings:
                if min(abs(h - ph), 360 - abs(h - ph)) < TaskConfig.PHOTO_HEADING_DIFF_MIN:
                    return False
        return True

    # ------------------------------------------------------------------ #
    #  向量化搜索所有可行窗口（核心优化）
    # ------------------------------------------------------------------ #
    def find_all_feasible_windows(
        self,
        targets: list,
        task_type: str = "shoot",
    ) -> list:
        """遍历所有目标，向量化收集满足约束的可行窗口。

        优化策略：
          1. 对每个目标一次性计算全部 N 个点的距离（向量化）
          2. 一次性生成布尔掩码：距离+速度+加速度（向量化）
          3. 用 np.convolve 滑动窗口找出所有连续满足的段（向量化）
          4. 用花式索引批量提取窗口属性（向量化）
          5. 仅在构建结果列表时使用 Python 循环（不可避免）

        Parameters
        ----------
        targets : list of dict
            目标列表，每个元素须包含 'id', 'x', 'y' 键。
        task_type : str
            任务类型，'shoot' 或 'photo'。

        Returns
        -------
        list of dict
            可行窗口列表。
        """
        # ---- 选择参数 ----
        if task_type == "shoot":
            prep_time = TaskConfig.SHOOT_PREP_TIME
            dist_min = TaskConfig.SHOOT_DIST_MIN
            dist_max = TaskConfig.SHOOT_DIST_MAX
            speed_max = TaskConfig.SHOOT_SPEED_MAX
            acc_max = TaskConfig.SHOOT_ACC_MAX
        elif task_type == "photo":
            prep_time = TaskConfig.PHOTO_PREP_TIME
            dist_min = TaskConfig.PHOTO_DIST_MIN
            dist_max = TaskConfig.PHOTO_DIST_MAX
            speed_max = TaskConfig.PHOTO_SPEED_MAX
            acc_max = TaskConfig.PHOTO_ACC_MAX
        else:
            raise ValueError(f"未知任务类型: {task_type}，仅支持 'shoot' 或 'photo'")

        # 准备窗口包含的采样点数（含首尾两端）
        # 例：1.5s / 0.1s = 15 步 → 16 个点
        prep_steps = int(np.round(prep_time / self.dt))
        window_size = prep_steps + 1

        feasible = []

        for target in targets:
            tid = target["id"]
            tx = target["x"]
            ty = target["y"]

            # ---- Step 1: 向量化计算全部距离 ----
            dist = np.sqrt((self.x - tx) ** 2 + (self.y - ty) ** 2)

            # ---- Step 2: 向量化生成布尔掩码 ----
            valid = (
                (dist >= dist_min) &
                (dist <= dist_max) &
                (self.speed <= speed_max) &
                (self.acc <= acc_max)
            )

            # ---- Step 3: 卷积滑动窗口找连续满足段 ----
            if window_size <= 1:
                valid_starts = np.where(valid)[0]
            else:
                kernel = np.ones(window_size, dtype=np.int32)
                conv = np.convolve(valid.astype(np.int32), kernel, mode="valid")
                # conv[i] == window_size 表示从 i 起连续 window_size 个点全部 valid
                valid_starts = np.where(conv == window_size)[0]

            if len(valid_starts) == 0:
                continue

            # 执行时刻索引 = 起始索引 + 准备步数
            exec_indices = valid_starts + prep_steps

            # 边界过滤
            mask = exec_indices < self.n
            valid_starts = valid_starts[mask]
            exec_indices = exec_indices[mask]

            if len(valid_starts) == 0:
                continue

            # ---- Step 4: 花式索引批量提取属性 ----
            t_starts = self.t[valid_starts]
            t_execs = self.t[exec_indices]
            distances = dist[exec_indices]
            speeds = self.speed[exec_indices]

            headings = None
            if task_type == "photo":
                dx = tx - self.x[exec_indices]
                dy = ty - self.y[exec_indices]
                headings = np.degrees(np.arctan2(dy, dx)) % 360.0

            # ---- Step 5: 构建结果列表 ----
            for k in range(len(valid_starts)):
                win = {
                    "target_id": tid,
                    "t_start": round(float(t_starts[k]), 4),
                    "t_exec": round(float(t_execs[k]), 4),
                    "distance": round(float(distances[k]), 4),
                    "speed": round(float(speeds[k]), 4),
                }
                if task_type == "photo" and headings is not None:
                    win["heading"] = round(float(headings[k]), 2)
                feasible.append(win)

        return feasible
