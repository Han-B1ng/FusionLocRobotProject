# file: core/motion_utils.py


import numpy as np


def compute_velocity(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    dt = t[1] - t[0]

    vx = np.empty(n, dtype=np.float64)
    vy = np.empty(n, dtype=np.float64)

    vx[0] = (x[1] - x[0]) / dt
    vy[0] = (y[1] - y[0]) / dt

    if n > 2:
        vx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        vy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)

    vx[-1] = (x[-1] - x[-2]) / dt
    vy[-1] = (y[-1] - y[-2]) / dt

    speed = np.sqrt(vx ** 2 + vy ** 2)

    return vx, vy, speed


try:
    from scipy.signal import savgol_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_acceleration(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    window_length: int = 9,
    polyorder: int = 3,
) -> tuple:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dt = float(t[1] - t[0])
    n = len(x)

    if HAS_SCIPY and n > polyorder + 2:
        wl_v = min(window_length, n - 1)
        if wl_v % 2 == 0:
            wl_v -= 1
        if wl_v < polyorder + 2:
            wl_v = polyorder + 2 + (1 - (polyorder + 2) % 2)  # 保证奇数且 > polyorder

        vx = savgol_filter(x, wl_v, polyorder, deriv=1, delta=dt)
        vy = savgol_filter(y, wl_v, polyorder, deriv=1, delta=dt)

        wl_a = min(window_length + 2, n - 1)  # 加速度用稍大窗口更稳定
        if wl_a % 2 == 0:
            wl_a -= 1
        if wl_a < polyorder + 2:
            wl_a = wl_v  # 退回速度的窗口

        ax = savgol_filter(vx, wl_a, polyorder, deriv=1, delta=dt)
        ay = savgol_filter(vy, wl_a, polyorder, deriv=1, delta=dt)
    else:
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)

    acc = np.sqrt(ax ** 2 + ay ** 2)
    return ax, ay, acc


def compute_heading_to_target(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    target_y: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dx = target_x - x
    dy = target_y - y

    heading_rad = np.arctan2(dy, dx)
    heading_deg = np.degrees(heading_rad) % 360.0

    return heading_deg


def compute_distance_to_target(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    target_y: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    distances = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    return distances
