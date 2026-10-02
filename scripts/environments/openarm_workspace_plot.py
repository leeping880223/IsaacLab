# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""[A 版] 右手 EE 可達範圍 + 方塊/目標隨機範圍 → matplotlib 圖（不開視窗，存成 PNG）。

圖與報告都用「環境原點（世界）座標系」：桌子、方塊固定不動，改機器人時是機器人和綠色點雲在動。
方塊座標可直接填進 scene.object.init_state.pos；目標要扣掉機器人位置才是 commands.object_pose.ranges。

執行：
    ./isaaclab.sh -p scripts/environments/openarm_workspace_plot.py
    ./isaaclab.sh -p scripts/environments/openarm_workspace_plot.py --samples 400000 --down_angle 45
"""

import argparse
import contextlib
import os

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

import torch

from openarm_workspace_common import (
    box_grid,
    check_coverage,
    get_table_box,
    get_task_boxes,
    print_report,
    sample_right_ee_workspace,
)

TASK_NAME = "IsaacContrib-Lift-Cube-OpenArm-Bimanual"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace_output")


def to_env_frame(result: dict, boxes: dict, grids: dict, table, robot_pos):
    """root 座標系 → 環境原點座標系（機器人沒旋轉，所以只加上機器人位置）。"""
    offset = torch.tensor(robot_pos, dtype=torch.float32, device=result["points"].device)
    result = {**result, "points": result["points"] + offset, "shoulder": result["shoulder"] + offset}
    shift = lambda v: [a + b for a, b in zip(v, robot_pos)]  # noqa: E731
    boxes = {k: (shift(lo), shift(hi)) for k, (lo, hi) in boxes.items()}
    grids = {k: g + offset for k, g in grids.items()}
    if table is not None:
        table = (shift(table[0]), shift(table[1]))
    return result, boxes, grids, table


def plot(result: dict, boxes: dict, grids: dict, reach: dict, args, out_path: str, table=None, robot_pos=(0, 0, 0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pts = result["points"].cpu().numpy()
    # 綠色 = 夾爪朝下且手臂不撞桌子；紅色小點 = 夾爪朝下但會撞桌子（被剔除）
    down = (result["down_mask"] & result["free_mask"]).cpu().numpy()
    hit = (result["down_mask"] & ~result["free_mask"]).cpu().numpy()
    shoulder = result["shoulder"].cpu().numpy()

    # 灰色點太多會蓋掉畫面，隨機抽一部分
    rng = torch.Generator().manual_seed(0)
    all_idx = torch.randperm(len(pts), generator=rng)[:30000].numpy()

    box_colors = {"cube": "tab:orange", "target": "tab:blue"}
    box_labels = {"cube": "cube spawn range", "target": "target (command) range"}
    views = [("top view (x-y)", 0, 1), ("side view (x-z)", 0, 2), ("front view (y-z)", 1, 2)]

    fig = plt.figure(figsize=(18, 13))
    for k, (title, a, b) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1)
        if table is not None:
            # 桌子碰撞體（含桌腳）的包圍盒，灰色斜線
            (tlo, thi) = table
            ax.add_patch(Rectangle((tlo[a], tlo[b]), thi[a] - tlo[a], thi[b] - tlo[b], fc="none", ec="dimgray",
                                   hatch="//", lw=1.5, label="table (bounding box)"))
        ax.scatter(pts[all_idx, a], pts[all_idx, b], s=1, c="lightgray", alpha=0.3, label="EE reachable (any orientation)")
        ax.scatter(pts[down, a], pts[down, b], s=1, c="tab:green", alpha=0.25,
                   label=f"EE reachable, gripper down (<{args.down_angle:.0f} deg), no table hit")
        hit_idx = torch.randperm(int(hit.sum()), generator=rng)[:15000].numpy()
        ax.scatter(pts[hit][hit_idx, a], pts[hit][hit_idx, b], s=1, c="tab:red", alpha=0.15,
                   label="gripper down, but arm hits table (removed)")
        for name, (lo, hi) in boxes.items():
            w, h = max(hi[a] - lo[a], 0.003), max(hi[b] - lo[b], 0.003)
            ax.add_patch(Rectangle((lo[a], lo[b]), w, h, fill=False, lw=2.5, ec=box_colors[name], label=box_labels[name]))
            g = grids[name].cpu().numpy()
            ok = reach[name].cpu().numpy()
            ax.scatter(g[ok, a], g[ok, b], s=12, c=box_colors[name], marker="o", edgecolors="k", linewidths=0.3)
            ax.scatter(g[~ok, a], g[~ok, b], s=30, c="red", marker="x")
        ax.scatter([shoulder[a]], [shoulder[b]], s=120, c="k", marker="*", label="right shoulder")
        ax.scatter([robot_pos[a]], [robot_pos[b]], s=80, c="k", marker="+", label="robot root")
        ax.set_title(title)
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if k == 0:
            leg = ax.legend(loc="upper left", fontsize=8)
            for h in leg.legend_handles:
                if hasattr(h, "set_sizes"):
                    h.set_sizes([30])
                    h.set_alpha(1.0)

    # 3D
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    down_idx = torch.randperm(int(down.sum()), generator=rng)[:15000].numpy()
    dp = pts[down][down_idx]
    ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], s=1, c="tab:green", alpha=0.15)
    for name, (lo, hi) in boxes.items():
        g = grids[name].cpu().numpy()
        ok = reach[name].cpu().numpy()
        ax.scatter(g[ok, 0], g[ok, 1], g[ok, 2], s=10, c=box_colors[name])
        ax.scatter(g[~ok, 0], g[~ok, 1], g[~ok, 2], s=25, c="red", marker="x")
    ax.scatter(*shoulder, s=120, c="k", marker="*")
    ax.set_title("3D (green = gripper down, red x = unreachable grid point)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    summary = "   ".join(f"{name}: {100.0 * reach[name].float().mean():.0f}% reachable" for name in boxes)
    fig.suptitle(f"OpenArm right EE workspace (env origin frame)   |   {summary}   |   tol={args.tol * 100:.1f} cm", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[A] 圖已存到: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000, help="關節隨機取樣次數")
    parser.add_argument("--num_envs", type=int, default=1024, help="一次平行算幾組 FK")
    parser.add_argument("--down_angle", type=float, default=30.0, help="夾爪與垂直向下夾角小於幾度算「朝下」")
    parser.add_argument("--tol", type=float, default=0.03, help="格點離點雲多近算可達 [m]")
    parser.add_argument("--grid_steps", type=int, default=7, help="範圍框每軸取幾個格點")
    parser.add_argument("--link_margin", type=float, default=0.04, help="手臂連桿離桌面至少多高才不算撞 [m]")
    parser.add_argument("--no_table_check", action="store_true", help="不剔除手臂撞桌子的姿勢（舊行為）")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    # 剩下的參數交給 Hydra；要明確傳 overrides，不然 Hydra 會讀 sys.argv 並拒絕本腳本自己的參數
    args, hydra_overrides = parser.parse_known_args()

    env_cfg, _ = resolve_task_config(TASK_NAME, "", overrides=hydra_overrides)
    env_cfg.scene.num_envs = args.num_envs
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        with contextlib.closing(gym.make(TASK_NAME, cfg=env_cfg)) as env:
            env.reset()
            table = get_table_box(env)
            with torch.inference_mode():
                result = sample_right_ee_workspace(
                    env, args.samples, args.down_angle, None if args.no_table_check else table, args.link_margin
                )
                boxes = get_task_boxes(env_cfg)
                cloud = result["points"][result["down_mask"] & result["free_mask"]]
                grids = {k: box_grid(lo, hi, args.grid_steps, cloud.device) for k, (lo, hi) in boxes.items()}
                reach = {k: check_coverage(g, cloud, args.tol) for k, g in grids.items()}
            n_down = int(result["down_mask"].sum())
            n_hit = int((result["down_mask"] & ~result["free_mask"]).sum())
            print(f"\n夾爪朝下的取樣 {n_down} 筆，其中手臂撞桌子被剔除 {n_hit} 筆 ({100.0 * n_hit / max(n_down, 1):.1f}%)")

            # 判定在 root 座標系做完，畫圖/報告前換成環境原點座標系
            robot_pos = list(env_cfg.scene.robot.init_state.pos)
            result, boxes, grids, table = to_env_frame(result, boxes, grids, table, robot_pos)

            print_report(boxes, grids, reach, result["shoulder"], args.tol, args.down_angle, table, frame="env")
            r = env_cfg.commands.object_pose.ranges
            print(f"機器人 init_state.pos = {robot_pos}")
            print(f"目前 commands ranges（相對機器人）: pos_x={r.pos_x} pos_y={r.pos_y} pos_z={r.pos_z}")
            print("目標換算：ranges = 圖上的世界座標 - 機器人 init_state.pos\n")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, "openarm_right_workspace.png")
            plot(result, boxes, grids, reach, args, out_path, table, robot_pos)


if __name__ == "__main__":
    main()
