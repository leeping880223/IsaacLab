# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""第 8 步：訓練前檢查觀測與各項獎勵（不訓練）。

A. 全零動作：手停在初始姿勢、方塊在桌上 → 抬起 / 目標追蹤應為 0，reaching 小於滿分。
B. 把方塊瞬移到目標點 → 抬起 ≈ 權重，兩個目標追蹤接近權重。

印出的獎勵是「每步、已乘權重」的值（reward_manager 的 step reward），env 間取平均。

執行：
    uv run python scripts/environments/openarm_reward_check.py
    uv run python scripts/environments/openarm_reward_check.py --num_envs 32 --steps 100
"""

import argparse
import contextlib

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

import torch

TASK_NAME = "IsaacContrib-Lift-Cube-OpenArm-Bimanual"


def print_rewards(title: str, env):
    rm = env.unwrapped.reward_manager
    print(f"\n--- {title}（每步、已乘權重，env 平均） ---")
    for idx, name in enumerate(rm.active_terms):
        weight = rm.get_term_cfg(name).weight
        v = rm._step_reward[:, idx]
        print(f"  {name:36s} mean={v.mean():+9.4f}  min={v.min():+9.4f}  max={v.max():+9.4f}  (weight={weight})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16, help="環境數")
    parser.add_argument("--steps", type=int, default=50, help="測試 A 跑幾步全零動作")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    # 剩下的參數交給 Hydra；要明確傳 overrides，不然 Hydra 會讀 sys.argv 並拒絕本腳本自己的參數
    args, hydra_overrides = parser.parse_known_args()

    env_cfg, _ = resolve_task_config(TASK_NAME, "", overrides=hydra_overrides)
    env_cfg.scene.num_envs = args.num_envs
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        with contextlib.closing(gym.make(TASK_NAME, cfg=env_cfg)) as env:
            unwrapped = env.unwrapped
            obs, _ = env.reset()
            actions = torch.zeros_like(unwrapped.action_manager.action)

            # ---- 觀測：每一項的維度與 env 0 的數值 ----
            om = unwrapped.observation_manager
            print("\n========== 觀測（policy group，env 0） ==========")
            print(f"動作維度 = {actions.shape[1]}（手臂 + 夾爪）")
            for name, values in om.get_active_iterable_terms(0):
                print(f"  {name:36s} dim={len(values):2d}  {[round(v, 3) for v in values]}")
            print(f"  總維度 = {obs['policy'].shape[1]}")

            with torch.inference_mode():
                # ---- A. 全零動作 ----
                for _ in range(args.steps):
                    env.step(actions)
                robot = unwrapped.scene["robot"]
                cube = unwrapped.scene["object"]
                ee = unwrapped.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
                cube_pos = cube.data.root_pos_w.torch
                cz = cube_pos[:, 2] - unwrapped.scene.env_origins[:, 2]
                print("\n========== A. 全零動作 ==========")
                print(f"TCP 到方塊距離 [m]: mean={(ee - cube_pos).norm(dim=1).mean():.3f}"
                      f"｜方塊中心 z [m]: mean={cz.mean():.4f}")
                print_rewards(f"A. 第 {args.steps} 步", env)

                # ---- B. 方塊瞬移到目標點 ----
                # 目標是 robot root 座標系，換到世界座標（機器人沒旋轉）
                from isaaclab.utils.math import quat_apply

                cmd = unwrapped.command_manager.get_command("object_pose")
                goal_w = robot.data.root_pos_w.torch + quat_apply(robot.data.root_quat_w.torch, cmd[:, :3])
                pose = torch.cat([goal_w, torch.tensor([0.0, 0.0, 0.0, 1.0], device=goal_w.device).expand(len(goal_w), 4)], 1)
                cube.write_root_pose_to_sim_index(root_pose=pose)
                cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(goal_w), 6, device=goal_w.device))
                env.step(actions)
                d = (cube.data.root_pos_w.torch - goal_w).norm(dim=1)
                print("\n========== B. 方塊瞬移到目標點（下一步量測） ==========")
                print(f"方塊到目標距離 [m]: mean={d.mean():.4f}（重力一步會掉一點點）")
                print_rewards("B. 瞬移後", env)
            print()


if __name__ == "__main__":
    main()
