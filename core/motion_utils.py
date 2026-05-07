# file: core/motion_utils.py
# @Description : 从融合轨迹计算速度、加速度、方向角与距离

"""
运动学工具模块。

从 10 Hz 融合轨迹 (t, x, y) 计算：
  - 速度分量 (vx, vy) 与速率 (speed)
  - 加速度分量 (ax, ay) 与加速度大小 (acc)
  - 到目标点的航向角 (heading_deg, 0~360°)
  - 到目标点的欧氏距离 (distances)

所有差分均基于等间距时间网格，dt = t[1] - t[0]。
边界点使用前向 / 后向差分，内部点使用中心差分。
"""

import numpy as np


# ============================================================
#  速度
# ============================================================
def compute_velocity(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """从等间距轨迹计算速度分量与速率。

    内部点采用中心差分，首尾边界分别采用前向和后向差分，
    以保证输出长度与输入一致。

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        等间距时间序列 (s)。
    x : np.ndarray, shape (N,)
        X 坐标序列 (m)。
    y : np.ndarray, shape (N,)
        Y 坐标序列 (m)。

    Returns
    -------
    vx : np.ndarray, shape (N,)
        X 方向速度分量 (m/s)。
    vy : np.ndarray, shape (N,)
        Y 方向速度分量 (m/s)。
    speed : np.ndarray, shape (N,)
        瞬时速率，speed = sqrt(vx² + vy²) (m/s)。
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    dt = t[1] - t[0]

    vx = np.empty(n, dtype=np.float64)
    vy = np.empty(n, dtype=np.float64)

    # 前向差分（首点）
    vx[0] = (x[1] - x[0]) / dt
    vy[0] = (y[1] - y[0]) / dt

    # 中心差分（内部点）
    if n > 2:
        vx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        vy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)

    # 后向差分（末点）
    vx[-1] = (x[-1] - x[-2]) / dt
    vy[-1] = (y[-1] - y[-2]) / dt

    speed = np.sqrt(vx ** 2 + vy ** 2)

    return vx, vy, speed


# ============================================================
#  加速度
# ============================================================
def compute_acceleration(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    smooth_window: int = 9,
) -> tuple:
    """从等间距轨迹计算加速度分量与加速度大小。

    先计算速度，对速度做移动平均平滑后再差分，
    避免二阶差分放大残余高频噪声。
    """
    vx, vy, _ = compute_velocity(t, x, y)

    # ---- 新增：对速度做移动平均平滑 ----
    if smooth_window > 1 and len(vx) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # 用 reflect 填充避免边界效应
        vx_smooth = np.convolve(vx, kernel, mode="same")
        vy_smooth = np.convolve(vy, kernel, mode="same")
    else:
        vx_smooth = vx
        vy_smooth = vy

    ax, ay, acc = compute_velocity(t, vx_smooth, vy_smooth)

    return ax, ay, acc



# ============================================================
#  航向角
# ============================================================
def compute_heading_to_target(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    target_y: float,
) -> np.ndarray:
    """计算每个时刻机器人到目标点的方向角。

    方向角定义：从正 X 轴逆时针旋转到目标方向的角度，
    范围 [0°, 360°)。使用 atan2 确定象限后映射到 0~360°。

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        等间距时间序列 (s)。（本函数未直接使用，保留接口一致性）
    x : np.ndarray, shape (N,)
        机器人 X 坐标序列 (m)。
    y : np.ndarray, shape (N,)
        机器人 Y 坐标序列 (m)。
    target_x : float
        目标点 X 坐标 (m)。
    target_y : float
        目标点 Y 坐标 (m)。

    Returns
    -------
    heading_deg : np.ndarray, shape (N,)
        每个时刻到目标的方向角 (°)，范围 [0, 360)。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dx = target_x - x
    dy = target_y - y

    # atan2 返回 [-π, π]，转换为 [0, 2π)
    heading_rad = np.arctan2(dy, dx)
    heading_deg = np.degrees(heading_rad) % 360.0

    return heading_deg


# ============================================================
#  到目标距离
# ============================================================
def compute_distance_to_target(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    target_y: float,
) -> np.ndarray:
    """计算每个时刻机器人到目标点的欧氏距离。

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        机器人 X 坐标序列 (m)。
    y : np.ndarray, shape (N,)
        机器人 Y 坐标序列 (m)。
    target_x : float
        目标点 X 坐标 (m)。
    target_y : float
        目标点 Y 坐标 (m)。

    Returns
    -------
    distances : np.ndarray, shape (N,)
        每个时刻到目标点的欧氏距离 (m)。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    distances = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    return distances
