# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Practice controlling OpenArm right-arm joints."""

import argparse
import contextlib

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

import time
import torch

TASK_NAME = "IsaacContrib-Lift-Cube-OpenArm-Bimanual" #在這邊改任務
ACTION_INDEX = 1      # 第幾個joint
MOVE_ACTION = -20 # 第一目標位置
RETURN_ACTION = 3.31 # 第二目標位置
HOLD_STEPS = 100       # 每段維持 100 個環境步

def main():
    """Launch the environment and inspect its action terms."""
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=["kit"])
    args = parser.parse_args()

    # 讀取 lift 任務設定。
    env_cfg, _ = resolve_task_config(TASK_NAME, "")
    env_cfg.scene.num_envs = 1

    # 使用任務設定中的運算裝置，例如 cuda:0。
    args.device = env_cfg.sim.device

    # 啟動模擬並建立環境。
    with launch_simulation(env_cfg, args):
        with contextlib.closing(gym.make(TASK_NAME, cfg=env_cfg)) as env:
            env.reset()

            #gym 會wrapper，所以要unwrapped才看得到
            action_manager = env.unwrapped.action_manager 

            #action 種類
            print(action_manager)
            print("Action 維度:", action_manager.total_action_dim)
            print("Action terms:", action_manager.active_terms)

            #關節順序
            for action_name in action_manager.active_terms:
                term = action_manager.get_term(action_name)

                print(f"\n{action_name}：{term.action_dim} 維")
                print("關節順序：", term.IO_descriptor.joint_names)

            #關節極限
            robot = env.unwrapped.scene["robot"]
            limits = robot.data.soft_joint_pos_limits.torch
            defaults = robot.data.default_joint_pos.torch

            for action_name in action_manager.active_terms:
                term = action_manager.get_term(action_name)
                joint_names = term.IO_descriptor.joint_names

                print(f"\n{action_name}：{term.action_dim} 維")

                for name in joint_names:
                    i = robot.joint_names.index(name)

                    lower = limits[0, i, 0].item()
                    upper = limits[0, i, 1].item()
                    default = defaults[0, i].item()

                    print(
                        f"  {name}: "
                        f"下限={lower:.4f}, "
                        f"上限={upper:.4f}, "
                        f"初始位置={default:.4f}"
                    )
            #設定移動關節與角度
            actions = torch.zeros_like(action_manager.action) 
            with torch.inference_mode():
                actions[:,ACTION_INDEX] = MOVE_ACTION #第一個關節改為0.2 往前

                print(f"action[{ACTION_INDEX}] step1 input={MOVE_ACTION} start")
                for _ in range(HOLD_STEPS): #移動100次
                    env.step(actions)
                    time.sleep(env.unwrapped.step_dt) #每次移動後都等0.02秒

                actions[:,ACTION_INDEX] = RETURN_ACTION #第一個關節改為0 回去

                print(f"action[{ACTION_INDEX}] step2 input={RETURN_ACTION} start")
                for _ in range(HOLD_STEPS): #移動100次
                    env.step(actions)
                    time.sleep(env.unwrapped.step_dt) #每次移動後都等0.02秒

if __name__ == "__main__":
    main()