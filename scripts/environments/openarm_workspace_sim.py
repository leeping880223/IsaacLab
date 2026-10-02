# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""[B 版] 右手 EE 可達範圍 + 方塊/目標隨機範圍 → 直接畫在 Isaac Sim 場景裡（只畫在 env 0）。

顏色：
    綠色小球      EE 可到、夾爪朝下、且手臂不撞桌子的位置
    暗紅小球      夾爪朝下但手臂會撞桌子（被剔除，加 --no_table_check 關閉檢查）
    灰色小球      EE 可到、任意姿態（加 --show_all 才顯示）
    橘色半透明框  方塊生成範圍
    藍色半透明框  目標 (command) 範圍
    青色球 / 紅球 範圍內格點：可達 / 不可達
場景持續跑 zero action：方塊每個 episode 重新隨機、目標 marker（debug_vis）也會一直換位置。

執行：
    ./isaaclab.sh -p scripts/environments/openarm_workspace_sim.py
    ./isaaclab.sh -p scripts/environments/openarm_workspace_sim.py --show_all --down_angle 45
"""

import argparse
import contextlib

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
MAX_CLOUD_MARKERS = 20000


def build_markers():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    def sphere(r, color, opacity=1.0):
        return sim_utils.SphereCfg(
            radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)
        )

    def box(color, opacity):
        return sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)
        )

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RightArmWorkspace",
        markers={
            "down": sphere(0.004, (0.1, 0.8, 0.1)),
            "any": sphere(0.003, (0.6, 0.6, 0.6), 0.5),
            "grid_ok": sphere(0.010, (0.0, 0.8, 0.9)),
            "grid_bad": sphere(0.010, (1.0, 0.0, 0.0)),
            "cube_box": box((1.0, 0.5, 0.0), 0.25),
            "target_box": box((0.1, 0.3, 1.0), 0.2),
            "hit": sphere(0.004, (0.5, 0.0, 0.0), 0.6),
        },
    )
    return VisualizationMarkers(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000, help="關節隨機取樣次數")
    parser.add_argument("--num_envs", type=int, default=64, help="環境數（也是一次平行算幾組 FK）")
    parser.add_argument("--down_angle", type=float, default=30.0, help="夾爪與垂直向下夾角小於幾度算「朝下」")
    parser.add_argument("--tol", type=float, default=0.03, help="格點離點雲多近算可達 [m]")
    parser.add_argument("--grid_steps", type=int, default=7, help="範圍框每軸取幾個格點")
    parser.add_argument("--show_all", action="store_true", help="另外顯示任意姿態可到的點（灰色）")
    parser.add_argument("--link_margin", type=float, default=0.04, help="手臂連桿離桌面至少多高才不算撞 [m]")
    parser.add_argument("--no_table_check", action="store_true", help="不剔除手臂撞桌子的姿勢（舊行為）")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=["kit"])
    # 剩下的參數交給 Hydra；要明確傳 overrides，不然 Hydra 會讀 sys.argv 並拒絕本腳本自己的參數
    args, hydra_overrides = parser.parse_known_args()

    env_cfg, _ = resolve_task_config(TASK_NAME, "", overrides=hydra_overrides)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.object_pose.debug_vis = True
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        with contextlib.closing(gym.make(TASK_NAME, cfg=env_cfg)) as env:
            env.reset()
            unwrapped = env.unwrapped
            robot = unwrapped.scene["robot"]

            table = get_table_box(env)
            with torch.inference_mode():
                result = sample_right_ee_workspace(
                    env, args.samples, args.down_angle, None if args.no_table_check else table, args.link_margin
                )
                boxes = get_task_boxes(env_cfg)
                cloud = result["points"][result["down_mask"] & result["free_mask"]]
                hit_pts = result["points"][result["down_mask"] & ~result["free_mask"]]
                grids = {k: box_grid(lo, hi, args.grid_steps, cloud.device) for k, (lo, hi) in boxes.items()}
                reach = {k: check_coverage(g, cloud, args.tol) for k, g in grids.items()}
            n_down = int(result["down_mask"].sum())
            print(f"\n夾爪朝下的取樣 {n_down} 筆，其中手臂撞桌子被剔除 {len(hit_pts)} 筆"
                  f" ({100.0 * len(hit_pts) / max(n_down, 1):.1f}%)")
            print_report(boxes, grids, reach, result["shoulder"], args.tol, args.down_angle, table)

            # 取樣時改過關節，重置回預設姿態
            env.reset()

            # ---- 組 marker（root 座標系 → env 0 世界座標）----
            from isaaclab.utils.math import quat_apply

            root_pos = robot.data.root_pos_w.torch[0]
            root_quat = robot.data.root_quat_w.torch[0]
            device = root_pos.device

            def to_world(p: torch.Tensor) -> torch.Tensor:
                return root_pos + quat_apply(root_quat.expand(len(p), 4), p)

            pos, scale, idx = [], [], []

            def add(points: torch.Tensor, marker_id: int, s=None):
                pos.append(to_world(points))
                scale.append(torch.ones(len(points), 3, device=device) if s is None else s)
                idx.append(torch.full((len(points),), marker_id, device=device, dtype=torch.long))

            add(cloud[torch.randperm(len(cloud), device=device)[:MAX_CLOUD_MARKERS]], 0)
            if len(hit_pts) > 0:
                add(hit_pts[torch.randperm(len(hit_pts), device=device)[:MAX_CLOUD_MARKERS]], 6)
            if args.show_all:
                pts = result["points"]
                add(pts[torch.randperm(len(pts), device=device)[:MAX_CLOUD_MARKERS]], 1)
            for name in boxes:
                add(grids[name][reach[name]], 2)
                add(grids[name][~reach[name]], 3)
            for marker_id, name in ((4, "cube"), (5, "target")):
                lo, hi = (torch.tensor(v, device=device) for v in boxes[name])
                size = (hi - lo).clamp(min=0.01).unsqueeze(0)
                add(((lo + hi) / 2).unsqueeze(0), marker_id, size)

            markers = build_markers()
            markers.visualize(translations=torch.cat(pos), scales=torch.cat(scale), marker_indices=torch.cat(idx))

            eye = (root_pos + torch.tensor([1.4, -1.4, 1.1], device=device)).tolist()
            target = (root_pos + torch.tensor([0.3, -0.1, 0.3], device=device)).tolist()
            unwrapped.sim.set_camera_view(eye, target)

            print("[B] 已在 env 0 畫出可達範圍，關閉視窗或 Ctrl+C 結束。")
            actions = torch.zeros_like(unwrapped.action_manager.action)
            sim = unwrapped.sim
            cube = unwrapped.scene["object"]
            min_h = env_cfg.rewards.lifting_object.params["minimal_height"]
            step = 0
            while sim.is_headless_or_exist_active_visualizer():
                with torch.inference_mode():
                    env.step(actions)
                step += 1
                if step == 60:
                    # 方塊落到桌上靜止後的中心高度（環境原點座標系）；要低於抬起門檻，否則放著不動就算「抬起」
                    cz = cube.data.root_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]
                    print(f"\n[方塊靜止高度] 中心 z: min={cz.min():.4f} mean={cz.mean():.4f} max={cz.max():.4f} [m]"
                          f"｜抬起門檻 minimal_height={min_h}｜超過門檻的環境: {int((cz > min_h).sum())}/{len(cz)}\n")


if __name__ == "__main__":
    main()
