# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the mass and center of mass of every robot body as the simulation uses them.

Also shows the mass authored in the USD (``MassAPI``), so bodies whose mass PhysX filled in
(e.g. a virtual frame without colliders getting a default 1 kg) stand out.

Example:
    uv run --extra isaacsim --no-sync python scripts/environments/print_body_mass.py \
        --task IsaacContrib-Lift-Cube-OpenArm-Bimanual-Official-WristAligned-Margin --filter right --viz none
"""

import argparse
import contextlib

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="Registered task name.")
parser.add_argument("--filter", default="", help="Only print bodies whose name contains this text.")
parser.add_argument("--asset", default="robot", help="Scene entity to inspect.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args, overrides = parser.parse_known_args()

env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
env_cfg.scene.num_envs = 1
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args):
    import omni.usd
    from pxr import Usd, UsdPhysics

    with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
        env.reset()
        scene = env.unwrapped.scene
        asset = scene[args.asset]
        mass = asset.data.body_mass.torch[0]
        com = asset.data.body_com_pos_b.torch[0]

        # mass authored on the rigid-body prims under env_0 (0 means unset: PhysX fills it in)
        stage = omni.usd.get_context().get_stage()
        root_path = asset.cfg.prim_path.replace(scene.env_regex_ns, scene.env_prim_paths[0])
        root = stage.GetPrimAtPath(root_path)
        authored = {}
        if root.IsValid():
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI):
                    value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                    if value is not None:
                        authored[prim.GetName()] = value

        print(f"\n[{args.asset}] {asset.cfg.spawn.usd_path}")
        print(f"  {'body':<30}{'sim mass [kg]':>14}{'USD mass [kg]':>15}   center of mass in body frame [m]")
        total = 0.0
        for i, name in enumerate(asset.body_names):
            if args.filter not in name:
                continue
            total += mass[i].item()
            if name not in authored:
                usd = "not set"
            elif authored[name] == 0.0:
                usd = "0 (unset)"
            else:
                usd = f"{authored[name]:.3f}"
            print(f"  {name:<30}{mass[i].item():14.3f}{usd:>15}   {[round(x, 4) for x in com[i].tolist()]}")
        print(f"  {'total':<30}{total:14.3f}")

        if "object" in scene.keys() and args.asset != "object":
            print(f"\n[object] mass {scene['object'].data.body_mass.torch[0].sum().item():.3f} kg")
