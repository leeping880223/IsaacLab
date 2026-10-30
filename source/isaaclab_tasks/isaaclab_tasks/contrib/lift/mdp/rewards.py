# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the rigid lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w.torch
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


class object_goal_distance(ManagerTermBase):
    """Reward the agent for tracking the object-to-goal pose using a tanh kernel.

    If ``success_threshold`` is provided in the term params, this also tracks per-episode
    success (sticky binary: object ever within ``success_threshold`` of the commanded goal
    while lifted above ``minimal_height``) and logs the mean across environments under
    ``Metrics/success_rate`` on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1 - torch.tanh(distance / std))

def object_spin_on_table(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Penalize object angular speed while it is still resting on the table."""
    object: RigidObject = env.scene[object_cfg.name]
    on_table = object.data.root_pos_w.torch[:, 2] < minimal_height
    return on_table * torch.linalg.norm(object.data.root_ang_vel_w.torch, dim=-1)

def joint_effort_over_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize drive effort demanded beyond the effort limit, as a fraction of each joint's limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    computed = asset.data.computed_torque.torch[:, asset_cfg.joint_ids].abs()
    limit = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    return torch.sum((computed - limit).clamp(min=0.0) / limit, dim=1)

def joint_axes_aligned(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[int, int]],
    cos_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pairs of joint axes that become nearly collinear (redundant-arm singularity).

    ``joint_pairs`` index into ``asset_cfg.joint_ids``; ``asset_cfg.body_ids`` must hold one body on the chain.
    Each pair contributes 0 below ``cos_threshold`` and 1 when the axes coincide.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0] - 1 if asset.is_fixed_base else asset_cfg.body_ids[0]
    cols = [j + asset.num_base_dofs for j in asset_cfg.joint_ids]
    axes = asset.data.body_link_jacobian_w.torch[:, body, 3:, :][:, :, cols]  # joint axes in world [N, 3, J]
    axes = axes / axes.norm(dim=1, keepdim=True).clamp(min=1e-6)
    penalty = torch.zeros(env.num_envs, device=env.device)
    for a, b in joint_pairs:
        cos = (axes[:, :, a] * axes[:, :, b]).sum(-1).abs()
        penalty += ((cos - cos_threshold) / (1.0 - cos_threshold)).clamp(min=0.0)
    return penalty
