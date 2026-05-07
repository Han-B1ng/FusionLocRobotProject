# file: core/constraint_checker.py
# @Description : 射击与拍照任务的时间窗口约束检查

"""
约束检查模块。

对融合轨迹的每个可能时间窗口，检查射击 / 拍照任务的全部物理约束：
  - 距离范围
  - 速率上限
  - 加速度上限
  - （拍照额外）航向角差异

时间步长 dt = t[1] - t[0]，索引定位通过 np.searchsorted 或
整数步数换算完成，确保窗口边界严格落在采样点上。
"""

import numpy as np

from config import TaskConfig


class ConstraintChecker:
    """轨迹约束检查器。

    接收完整轨迹及其运动状态（速度、加速度），
    提供射击窗口、拍照窗口的可行性检查。

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

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #
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

        # 时间步长：取相邻差的中位数以抗微小浮点偏差
        self.dt = float(np.median(np.diff(self.t)))
        self.n = len(self.t)

        # 预计算速率与加速度大小（避免重复计算）
        self.speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        self.acc = np.sqrt(self.ax ** 2 + self.ay ** 2)

    # ------------------------------------------------------------------ #
    #  内部工具：时间 → 最近采样点索引
    # ------------------------------------------------------------------ #
    def _time_to_index(self, time_val: float) -> int:
        """将时间值映射到最近的采样点索引。

        使用 searchsorted 找到插入位置后取最近邻，
        并 clamp 到 [0, n-1]。
        """
        idx = int(np.searchsorted(self.t, time_val, side="left"))
        # 右边界保护
        if idx >= self.n:
            idx = self.n - 1
        # 比较左右两侧哪个更近
        if idx > 0 and abs(self.t[idx - 1] - time_val) < abs(self.t[idx] - time_val):
            idx -= 1
        return idx

    def _window_indices(self, t_start: float, t_end: float) -> np.ndarray:
        """返回 [t_start, t_end] 闭区间内所有采样点的索引数组。"""
        i_start = self._time_to_index(t_start)
        i_end = self._time_to_index(t_end)
        # 保证至少包含 1 个点
        if i_end < i_start:
            i_end = i_start
        return np.arange(i_start, i_end + 1)

    # ------------------------------------------------------------------ #
    #  距离计算（内部复用）
    # ------------------------------------------------------------------ #
    def _distance_at(self, idx: np.ndarray, target_x: float, target_y: float) -> np.ndarray:
        """计算指定索引处到目标点的欧氏距离。"""
        return np.sqrt(
            (self.x[idx] - target_x) ** 2
            + (self.y[idx] - target_y) ** 2
        )

    def _heading_at(self, idx: np.ndarray, target_x: float, target_y: float) -> np.ndarray:
        """计算指定索引处到目标点的方向角 (°)，范围 [0, 360)。"""
        dx = target_x - self.x[idx]
        dy = target_y - self.y[idx]
        return np.degrees(np.arctan2(dy, dx)) % 360.0

    # ------------------------------------------------------------------ #
    #  射击窗口检查
    # ------------------------------------------------------------------ #
    def check_shooting_window(
        self,
        target_x: float,
        target_y: float,
        t_start: float,
    ) -> bool:
        """检查从 t_start 开始的射击时间窗口是否满足全部约束。

        时间线::

            t_start          t_start + 1.5 s
              |---- 准备 ----|---- 执行 ----|
              ↑ 检查起点                    ↑ 检查终点（含）

        窗口内每个采样点须同时满足：
          - 距离 ∈ [SHOOT_DIST_MIN, SHOOT_DIST_MAX]  (默认 [5, 30] m)
          - 速率 ≤ SHOOT_SPEED_MAX                    (默认 2.0 m/s)
          - 加速度 ≤ SHOOT_ACC_MAX                    (默认 1.5 m/s²)

        Parameters
        ----------
        target_x, target_y : float
            目标点坐标 (m)。
        t_start : float
            准备开始时刻 (s)。

        Returns
        -------
        bool
            所有约束均满足时返回 True，否则 False。
        """
        t_exec = t_start + TaskConfig.SHOOT_PREP_TIME  # t_start + 1.5 s

        # 边界检查：窗口须完全在轨迹时间范围内
        if t_start < self.t[0] or t_exec > self.t[-1]:
            return False

        idx = self._window_indices(t_start, t_exec)

        # 距离约束
        dist = self._distance_at(idx, target_x, target_y)
        if np.any(dist < TaskConfig.SHOOT_DIST_MIN):
            return False
        if np.any(dist > TaskConfig.SHOOT_DIST_MAX):
            return False

        # 速率约束
        if np.any(self.speed[idx] > TaskConfig.SHOOT_SPEED_MAX):
            return False

        # 加速度约束
        if np.any(self.acc[idx] > TaskConfig.SHOOT_ACC_MAX):
            return False

        return True

    # ------------------------------------------------------------------ #
    #  拍照窗口检查
    # ------------------------------------------------------------------ #
    def check_photo_window(
        self,
        target_x: float,
        target_y: float,
        t_start: float,
        previous_headings: list | None = None,
    ) -> bool:
        """检查从 t_start 开始的拍照时间窗口是否满足全部约束。

        时间线::

            t_start          t_start + 0.5 s
              |---- 准备 ----|---- 执行 ----|
              ↑ 检查起点                    ↑ 检查终点（含）& 航向检查点

        窗口内每个采样点须同时满足：
          - 距离 ∈ [PHOTO_DIST_MIN, PHOTO_DIST_MAX]  (默认 [10, 40] m)
          - 速率 ≤ PHOTO_SPEED_MAX                    (默认 1.5 m/s)
          - 加速度 ≤ PHOTO_ACC_MAX                    (默认 1.5 m/s²)

        额外约束（仅在执行时刻检查）：
          - 执行时刻航向角与 previous_headings 中任意一个的差值
            ≥ PHOTO_HEADING_DIFF_MIN（默认 60°）

        Parameters
        ----------
        target_x, target_y : float
            目标点坐标 (m)。
        t_start : float
            准备开始时刻 (s)。
        previous_headings : list of float, optional
            已完成拍照的方向角列表 (°)，默认为空列表。

        Returns
        -------
        bool
            所有约束均满足时返回 True，否则 False。
        """
        if previous_headings is None:
            previous_headings = []

        t_exec = t_start + TaskConfig.PHOTO_PREP_TIME  # t_start + 0.5 s

        # 边界检查
        if t_start < self.t[0] or t_exec > self.t[-1]:
            return False

        idx = self._window_indices(t_start, t_exec)

        # 距离约束
        dist = self._distance_at(idx, target_x, target_y)
        if np.any(dist < TaskConfig.PHOTO_DIST_MIN):
            return False
        if np.any(dist > TaskConfig.PHOTO_DIST_MAX):
            return False

        # 速率约束
        if np.any(self.speed[idx] > TaskConfig.PHOTO_SPEED_MAX):
            return False

        # 加速度约束
        if np.any(self.acc[idx] > TaskConfig.PHOTO_ACC_MAX):
            return False

        # 航向角差异约束（仅执行时刻）
        if len(previous_headings) > 0:
            i_exec = self._time_to_index(t_exec)
            current_heading = float(
                self._heading_at(
                    np.array([i_exec]), target_x, target_y
                )[0]
            )
            for ph in previous_headings:
                diff = abs(current_heading - ph)
                # 角度差取最小弧（0~180°）
                angular_diff = min(diff, 360.0 - diff)
                if angular_diff < TaskConfig.PHOTO_HEADING_DIFF_MIN:
                    return False

        return True

    # ------------------------------------------------------------------ #
    #  搜索所有可行窗口
    # ------------------------------------------------------------------ #
    def find_all_feasible_windows(
        self,
        targets: list,
        task_type: str = "shoot",
    ) -> list:
        """遍历所有目标与时间步，收集满足约束的可行窗口。

        遍历方式：从轨迹起始时间到结束时间，以 dt 为步长枚举
        所有可能的 t_start，确保整个准备+执行窗口在轨迹范围内。

        Parameters
        ----------
        targets : list of dict
            目标列表，每个元素须包含 'id', 'x', 'y' 键。
            示例: [{'id': 1, 'x': 500.0, 'y': 300.0}, ...]
        task_type : str
            任务类型，'shoot' 或 'photo'。

        Returns
        -------
        list of dict
            每个可行窗口包含以下字段：
              - target_id : 目标 ID
              - t_start   : 准备开始时刻 (s)
              - t_exec    : 执行时刻 (s)
              - distance  : 执行时刻到目标距离 (m)
              - speed     : 执行时刻速率 (m/s)
              - heading   : 执行时刻方向角 (°)，仅 photo
        """
        # 选择准备时间
        if task_type == "shoot":
            prep_time = TaskConfig.SHOOT_PREP_TIME
        elif task_type == "photo":
            prep_time = TaskConfig.PHOTO_PREP_TIME
        else:
            raise ValueError(f"未知任务类型: {task_type}，仅支持 'shoot' 或 'photo'")

        feasible = []

        for target in targets:
            tid = target["id"]
            tx = target["x"]
            ty = target["y"]

            # t_start 的有效范围：
            #   下界: self.t[0]
            #   上界: self.t[-1] - prep_time（保证执行时刻不超出轨迹）
            t_min = self.t[0]
            t_max = self.t[-1] - prep_time

            if t_min > t_max:
                # 轨迹长度不足一个准备周期
                continue

            # 以 dt 为步长遍历
            # 使用整数步数避免浮点累积误差
            n_steps = int(np.floor((t_max - t_min) / self.dt))

            for step in range(n_steps + 1):
                t_start = t_min + step * self.dt
                t_exec = t_start + prep_time

                # ---------- 射击任务 ----------
                if task_type == "shoot":
                    if not self.check_shooting_window(tx, ty, t_start):
                        continue

                    i_exec = self._time_to_index(t_exec)
                    feasible.append({
                        "target_id": tid,
                        "t_start": round(float(t_start), 4),
                        "t_exec": round(float(t_exec), 4),
                        "distance": round(float(
                            self._distance_at(
                                np.array([i_exec]), tx, ty
                            )[0]
                        ), 4),
                        "speed": round(float(self.speed[i_exec]), 4),
                    })

                # ---------- 拍照任务 ----------
                elif task_type == "photo":
                    # 拍照的航向差异需要已有历史；
                    # 在搜索阶段不传 previous_headings，
                    # 仅做基础物理约束检查（距离、速度、加速度）。
                    # 航向差异约束在排程阶段传入后二次验证。
                    if not self.check_photo_window(tx, ty, t_start):
                        continue

                    i_exec = self._time_to_index(t_exec)
                    heading = float(
                        self._heading_at(
                            np.array([i_exec]), tx, ty
                        )[0]
                    )
                    feasible.append({
                        "target_id": tid,
                        "t_start": round(float(t_start), 4),
                        "t_exec": round(float(t_exec), 4),
                        "distance": round(float(
                            self._distance_at(
                                np.array([i_exec]), tx, ty
                            )[0]
                        ), 4),
                        "speed": round(float(self.speed[i_exec]), 4),
                        "heading": round(heading, 2),
                    })

        return feasible
