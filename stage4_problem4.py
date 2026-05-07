# file: stage4_problem4.py
# @Author : Han_B1ng
# @Time : 2026/5/7
# @Description : 问题4求解：读取轨迹与目标 → 运动状态计算 → 约束检查 → 贪心调度 → 输出结果

"""
╔══════════════════════════════════════════════════════╗
║  阶段 4 — 问题4：任务规划与结果输出                   ║
╚══════════════════════════════════════════════════════╝

问题描述：
  基于问题3输出的10 Hz融合轨迹，对射击和拍照两类任务
  进行可行窗口搜索与贪心调度，输出无冲突的任务时刻表。

求解流程：
  ① 读取10 Hz融合轨迹（Problem3_10Hz.xlsx）
  ② 读取目标点坐标（每个目标可同时尝试射击和拍照）
  ③ 计算速度、加速度等运动状态量
  ④ 实例化约束检查器，搜索全部可行时间窗口
  ⑤ 贪心调度，生成无冲突任务序列
  ⑥ 保存结果至Excel，输出汇总信息

依赖模块：config.py → TaskConfig, data_path
          core.motion_utils, core.constraint_checker, core.task_scheduler
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TaskConfig, data_path
from core.constraint_checker import ConstraintChecker
from core.motion_utils import (
    compute_acceleration,
    compute_velocity,
)
from core.task_scheduler import schedule_tasks_greedy


# ============================================================
#  全局绘图样式
# ============================================================

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


_COLOR_SHOOT = "#DC2626"
_COLOR_PHOTO = "#2563EB"
_COLOR_TRAJ  = "#16A34A"


# ============================================================
#  Step 1: 读取融合轨迹
# ============================================================
def load_trajectory() -> tuple:
    """读取问题3输出的10 Hz融合轨迹。

    Returns
    -------
    t, x, y : np.ndarray, shape (N,)
        时间戳(s)、X坐标(m)、Y坐标(m)。
    """
    traj_path = data_path.output_dir / "Problem3_10Hz.xlsx"

    if not traj_path.exists():
        for ext in (".xlsx", ".xls", ".csv"):
            alt = traj_path.with_suffix(ext)
            if alt.exists():
                traj_path = alt
                break

    print(f"[问题4] 加载轨迹：{traj_path}")

    df = pd.read_excel(traj_path, engine="openpyxl")

    col_map = {
        "Time(s)": "t", "时间(s)": "t", "时间": "t", "t": "t",
        "X(m)": "x", "X坐标(m)": "x", "X坐标": "x", "x": "x",
        "Y(m)": "y", "Y坐标(m)": "y", "Y坐标": "y", "y": "y",
    }
    df = df.rename(columns=col_map)
    df = df[["t", "x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()

    t = df["t"].values.astype(np.float64)
    x = df["x"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)

    print(f"[问题4] 轨迹：{len(t)} 个采样点，"
          f"[{t[0]:.2f}, {t[-1]:.2f}] s，dt={t[1]-t[0]:.4f} s")

    return t, x, y


# ============================================================
#  Step 2: 读取目标点
# ============================================================
def load_targets() -> list:
    """读取附件4的目标点坐标。

    说明：每个目标可同时尝试射击和拍照，不按前缀预分类。

    Returns
    -------
    all_targets : list[dict]
        每个元素包含 {'id': int, 'name': str, 'x': float, 'y': float}。
    """
    file_path = Path(data_path.file4)

    if not file_path.exists():
        search_candidates = [
            file_path,
            Path("data") / file_path.name,
            Path("data") / file_path,
        ]
        found = False
        for candidate in search_candidates:
            if candidate.exists():
                file_path = candidate
                found = True
                break
            for ext in (".xlsx", ".xls", ".csv"):
                alt = candidate.with_suffix(ext)
                if alt.exists():
                    file_path = alt
                    found = True
                    break
            if found:
                break

    if not file_path.exists():
        raise FileNotFoundError(
            f"[问题4] 找不到目标文件，已尝试以下路径：\n"
            f"  - {data_path.file4}\n"
            f"  - data/{Path(data_path.file4).name}\n"
            f"  请检查 config.py 中 file4 的路径配置。"
        )

    print(f"[问题4] 加载目标：{file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")

    col_map = {
        "目标编号": "name", "编号": "name", "编号/名称": "name",
        "类型": "type", "任务类型": "type",
        "X坐标(m)": "x", "X坐标": "x", "X": "x", "x": "x",
        "Y坐标(m)": "y", "Y坐标": "y", "Y": "y", "y": "y",
    }
    df = df.rename(columns=col_map)

    for col in ("x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["x", "y"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    all_targets = []
    for idx, row in df.iterrows():
        name = str(row.get("name", f"T{idx + 1}"))
        all_targets.append({
            "id": idx + 1,
            "name": name,
            "x": float(row["x"]),
            "y": float(row["y"]),
        })

    print(f"[问题4] 目标总数：{len(all_targets)} 个")
    for t in all_targets:
        print(f"    {t['name']:>6s}  ({t['x']:.1f}, {t['y']:.1f})")

    return all_targets


# ============================================================
#  Step 9: 打印汇总
# ============================================================
def print_summary(
    scheduled_tasks: list,
    all_targets: list,
) -> None:
    """打印调度结果汇总信息。

    Parameters
    ----------
    scheduled_tasks : list[dict]
        已安排的任务列表。
    all_targets : list[dict]
        全部目标列表（含'name'字段）。
    """
    shoot_tasks = [t for t in scheduled_tasks if t["task_type"] == "shoot"]
    photo_tasks = [t for t in scheduled_tasks if t["task_type"] == "photo"]

    print("\n" + "=" * 60)
    print("  问题 4 结果汇总")
    print("=" * 60)
    print(f"  总任务数：{len(scheduled_tasks)}")
    print(f"  射击任务：{len(shoot_tasks)} 个")
    print(f"  拍照任务：{len(photo_tasks)} 个")

    if len(scheduled_tasks) > 0:
        t_first = scheduled_tasks[0]["t_start_prep"]
        t_last  = scheduled_tasks[-1]["t_execute"]
        print(f"  时间跨度：[{t_first:.2f}, {t_last:.2f}] s")
        print(f"  总耗时：{t_last - t_first:.2f} s")

    # 拍照目标详情
    if len(photo_tasks) > 0:
        print("\n  拍照目标详情：")
        id_to_name = {t["id"]: t["name"] for t in all_targets}

        photo_by_target: dict = {}
        for pt in photo_tasks:
            tid = pt["target_id"]
            photo_by_target.setdefault(tid, [])
            photo_by_target[tid].append(pt.get("heading", None))

        for tid, headings in photo_by_target.items():
            name = id_to_name.get(tid, f"目标{tid}")
            heading_strs = []
            for h in headings:
                if h is not None:
                    heading_strs.append(f"{h:.1f} deg")
                else:
                    heading_strs.append("N/A")
            print(f"    {name}: {len(headings)} 次, "
                  f"角度=[{', '.join(heading_strs)}]")

    print("=" * 60)


# ============================================================
#  可视化
# ============================================================
def plot_problem4_results(
    t: np.ndarray, x: np.ndarray, y: np.ndarray,
    all_targets: list,
    scheduled_tasks: list,
    output_dir: Path,
) -> None:
    """绘制问题4的结果图（轨迹+目标+甘特图）。"""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # ── 绘制轨迹 ──
    ax.plot(x, y, color=_COLOR_TRAJ, linewidth=0.6, alpha=0.7,
            label="融合轨迹")

    # ── 标注目标点 ──
    for tgt in all_targets:
        ax.scatter(tgt["x"], tgt["y"], s=120, marker="^",
                   color=_COLOR_SHOOT, edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(tgt["name"], (tgt["x"], tgt["y"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color=_COLOR_SHOOT)

    # 已安排任务的执行点连线
    id_to_info = {tgt["id"]: tgt for tgt in all_targets}

    for task in scheduled_tasks:
        tid = task["target_id"]
        tgt = id_to_info.get(tid)
        if tgt is None:
            continue
        i_exec = np.argmin(np.abs(t - task["t_execute"]))
        color = _COLOR_SHOOT if task["task_type"] == "shoot" else _COLOR_PHOTO
        ax.plot([x[i_exec], tgt["x"]], [y[i_exec], tgt["y"]],
                color=color, linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        f"问题4 — 任务调度（共{len(scheduled_tasks)}项任务）",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(figures_dir / "Problem4_schedule.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # ── 甘特图 ──
    if len(scheduled_tasks) > 0:
        fig, ax = plt.subplots(
            figsize=(16, max(4, len(scheduled_tasks) * 0.5))
        )

        y_labels = []
        for i, task in enumerate(scheduled_tasks):
            tid = task["target_id"]
            tgt = id_to_info.get(tid)
            name = tgt["name"] if tgt else f"T{tid}"
            task_label = "射击" if task["task_type"] == "shoot" else "拍照"
            y_labels.append(f"{name} ({task_label})")

            color = (
                _COLOR_SHOOT if task["task_type"] == "shoot" else _COLOR_PHOTO
            )

            ax.barh(
                i, task["t_execute"] - task["t_start_prep"],
                left=task["t_start_prep"],
                height=0.6, color=color, alpha=0.3, edgecolor=color,
            )
            ax.plot(task["t_execute"], i, marker="|", markersize=15,
                    color=color, markeredgewidth=2)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=9, fontfamily="sans-serif")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_title(
            "任务调度 — 甘特图",
            fontsize=14, fontweight="bold",
        )
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(figures_dir / "Problem4_gantt.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"[问题4] 图表已保存至 {figures_dir}")


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    output_dir = data_path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取融合轨迹 ──
    print("[问题4] 数据加载")
    print("=" * 60)
    t, x, y = load_trajectory()

    # ── 2. 读取目标点坐标 ──
    all_targets = load_targets()

    # ── 3. 运动状态计算（v, a）──
    print("\n" + "=" * 60)
    print("  步骤3：运动状态计算")
    print("=" * 60)

    vx, vy, speed = compute_velocity(t, x, y)
    ax, ay, acc = compute_acceleration(t, x, y)

    print(f"  速率：min={np.min(speed):.3f}，max={np.max(speed):.3f}，"
          f"mean={np.mean(speed):.3f} m/s")
    print(f"  加速度：min={np.min(acc):.3f}，max={np.max(acc):.3f}，"
          f"mean={np.mean(acc):.3f} m/s²")

    # ── 4. 实例化约束检查器 ──
    print("\n" + "=" * 60)
    print("  步骤4：约束检查器实例化")
    print("=" * 60)

    checker = ConstraintChecker(t, x, y, vx, vy, ax, ay)
    print(f"  dt = {checker.dt:.4f} s，{checker.n} 个采样点")
    print(f"  射击距离约束：[{TaskConfig.SHOOT_DIST_MIN}, "
          f"{TaskConfig.SHOOT_DIST_MAX}] m")
    print(f"  射击速率上限：{TaskConfig.SHOOT_SPEED_MAX} m/s")
    print(f"  拍照距离约束：[{TaskConfig.PHOTO_DIST_MIN}, "
          f"{TaskConfig.PHOTO_DIST_MAX}] m")
    print(f"  拍照速率上限：{TaskConfig.PHOTO_SPEED_MAX} m/s")

    # ── 5. 搜索可行时间窗口 ──
    print("\n" + "=" * 60)
    print("  步骤5：搜索可行时间窗口")
    print("=" * 60)

    windows_shoot = checker.find_all_feasible_windows(
        all_targets, task_type="shoot",
    )
    for w in windows_shoot:
        w["task_type"] = "shoot"

    windows_photo = checker.find_all_feasible_windows(
        all_targets, task_type="photo",
    )
    for w in windows_photo:
        w["task_type"] = "photo"

    print(f"  射击可行窗口：{len(windows_shoot)} 个")
    print(f"  拍照可行窗口：{len(windows_photo)} 个")

    # ── 按目标统计窗口数 ──
    id_to_name = {tgt["id"]: tgt["name"] for tgt in all_targets}

    for label, windows in [("射击", windows_shoot), ("拍照", windows_photo)]:
        if len(windows) > 0:
            counts = {}
            for w in windows:
                tid = w["target_id"]
                counts[tid] = counts.get(tid, 0) + 1
            print(f"  {label}窗口分布:")
            for tid, cnt in sorted(counts.items()):
                print(f"    {id_to_name.get(tid, f'T{tid}'):>6s}: {cnt} 个窗口")

    # ── 窗口时间分布诊断 ──
    windows_all = windows_shoot + windows_photo

    if len(windows_all) > 0:
        print("\n  窗口时间分布 (前 30 个):")
        for w in sorted(windows_all, key=lambda x: x["t_exec"])[:30]:
            name = id_to_name.get(w["target_id"], f"T{w['target_id']}")
            print(
                f"    {name:>6s} {w['task_type']:6s}  "
                f"t_start={w['t_start']:.2f}  t_exec={w['t_exec']:.2f}  "
                f"dist={w['distance']:.1f}m  speed={w['speed']:.2f}m/s"
            )

    if len(windows_all) == 0:
        print("\n[问题4] 警告：未找到任何可行窗口，无法调度。")
        print("[问题4] 可能原因：")
        print("  1. 目标坐标与轨迹坐标系不匹配（轨迹 X:[{:.0f},{:.0f}], "
              "目标 X:[{:.0f},{:.0f}]）".format(
                  x.min(), x.max(),
                  min(t["x"] for t in all_targets),
                  max(t["x"] for t in all_targets),
              ))
        print("  2. 速率/加速度约束过严（轨迹平均速率 {:.2f} m/s）".format(
            np.mean(speed)))
        df_empty = pd.DataFrame(columns=[
            "序号", "目标编号", "任务", "开始准备时刻(s)", "任务执行时刻(s)"
        ])
        df_empty.to_excel(
            output_dir / "result.xlsx", index=False, engine="openpyxl"
        )
        print(f"[问题4] 空结果已保存至 {output_dir / 'result.xlsx'}")
        exit(0)

    # ── 6. 贪心调度 ──
    print("\n" + "=" * 60)
    print("  步骤6：贪心调度")
    print("=" * 60)

    scheduled_tasks = schedule_tasks_greedy(
        windows_all,
        targets_photo=all_targets,
    )

    print(f"  调度完成：{len(scheduled_tasks)} 个任务")

    # ── 7. 生成结果表 ──
    print("\n" + "=" * 60)
    print("  步骤7：生成结果表")
    print("=" * 60)

    rows = []
    for i, task in enumerate(scheduled_tasks, start=1):
        task_label = "射击" if task["task_type"] == "shoot" else "拍照"
        rows.append({
            "序号": i,
            "目标编号": id_to_name.get(
                task["target_id"], f"T{task['target_id']}"
            ),
            "任务": task_label,
            "开始准备时刻(s)": round(task["t_start_prep"], 2),
            "任务执行时刻(s)": round(task["t_execute"], 2),
        })

    df_result = pd.DataFrame(rows)
    df_result = df_result.sort_values("任务执行时刻(s)").reset_index(drop=True)
    df_result["序号"] = range(1, len(df_result) + 1)

    print(f"\n{df_result.to_string(index=False)}")

    # ── 8. 保存结果 ──
    xlsx_path = output_dir / "result.xlsx"

    template_path = Path(data_path.file4).parent / "result_template.xlsx"
    if template_path.exists():
        try:
            from openpyxl import load_workbook

            wb = load_workbook(template_path)
            ws = wb.active

            for r_idx, row in enumerate(rows, start=2):
                ws.cell(row=r_idx, column=1, value=row["序号"])
                ws.cell(row=r_idx, column=2, value=row["目标编号"])
                ws.cell(row=r_idx, column=3, value=row["任务"])
                ws.cell(row=r_idx, column=4, value=row["开始准备时刻(s)"])
                ws.cell(row=r_idx, column=5, value=row["任务执行时刻(s)"])

            wb.save(xlsx_path)
            print(f"[问题4] 结果已保存至 {xlsx_path}（模板格式）")
        except Exception as e:
            print(f"[问题4] 模板写入失败（{e}），使用默认格式")
            df_result.to_excel(xlsx_path, index=False, engine="openpyxl")
            print(f"[问题4] 结果已保存至 {xlsx_path}")
    else:
        df_result.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"[问题4] 结果已保存至 {xlsx_path}")

    size_kb = xlsx_path.stat().st_size / 1024
    print(f"[问题4] 文件大小：{size_kb:.1f} KB")

    # ── 9. 输出汇总 ──
    print_summary(scheduled_tasks, all_targets)

    # 可视化
    plot_problem4_results(
        t, x, y,
        all_targets,
        scheduled_tasks,
        output_dir,
    )

    print("\n[问题4] 问题4求解完毕。")