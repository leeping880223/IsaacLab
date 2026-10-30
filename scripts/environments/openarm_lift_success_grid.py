# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map the lift policy's success rate over a grid of object start positions.

Every env gets one grid cell (env id modulo the number of cells) and always resets the object there,
so with 1024 envs and a 9 x 9 grid each cell runs about 12 episodes per round. ``--rounds`` repeats
the episode length that many times and adds up the counts. The target command stays random unless
``--command`` fixes it; ``--object`` tests a single object start position instead of the grid.

Each finished episode is counted as one of:

* lifted   - time out with the object above the lift height
* dropped  - ended early because the object fell off the table
* never    - time out and the object was never lifted
* lost     - time out, lifted at some point but back on the table at the end

Outputs ``<out>.csv`` and ``<out>.png`` (one heat map per outcome, dashed box = training reset range).

Example:
    uv run --extra isaacsim --no-sync python scripts/environments/openarm_lift_success_grid.py \
        --task IsaacContrib-Lift-Cube-OpenArm-Bimanual-Official-WristAligned-ShrinkWrap \
        --checkpoint logs/rsl_rl/openarm_lift/<run>/model_40993.pt --num_envs 1024 --rounds 10 --viz none
"""

import argparse
import contextlib
import csv
import os

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

OUTCOMES = ("lifted", "dropped", "never", "lost")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="Registered task name.")
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs.")
parser.add_argument("--rounds", type=int, default=10, help="Number of episode lengths to run.")
parser.add_argument("--nx", type=int, default=9, help="Grid points along x (back to front).")
parser.add_argument("--ny", type=int, default=9, help="Grid points along y (right to left).")
parser.add_argument(
    "--extend", type=float, default=0.0, help="Extend the grid beyond the training reset range on every side [m]."
)
parser.add_argument(
    "--command", type=float, nargs=3, default=None, help="Fix the target position in the robot base frame [m]."
)
parser.add_argument(
    "--object",
    type=float,
    nargs=2,
    default=None,
    help="Test only this object start position (x, y) in env-local coordinates [m] instead of a grid.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--out",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_output", "lift_success_grid"),
    help="Output path without extension.",
)
add_launcher_args(parser)
parser.set_defaults(device=None)
args, overrides = parser.parse_known_args()

env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
args.device = env_cfg.sim.device

# grid in env-local coordinates, built from the training reset range around the object's init position
x0, y0 = env_cfg.scene.object.init_state.pos[:2]
train_range = env_cfg.events.reset_object_position.params["pose_range"]
x_lo, x_hi = x0 + train_range["x"][0], x0 + train_range["x"][1]
y_lo, y_hi = y0 + train_range["y"][0], y0 + train_range["y"][1]
if args.object is not None:
    args.nx = args.ny = 1
    xs, ys = torch.tensor([args.object[0]]), torch.tensor([args.object[1]])
else:
    xs = torch.linspace(x_lo - args.extend, x_hi + args.extend, args.nx)
    ys = torch.linspace(y_lo - args.extend, y_hi + args.extend, args.ny)
cells = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)  # [nx * ny, 2]
num_cells = len(cells)
if args.num_envs < num_cells:
    print(f"[WARN] {args.num_envs} envs < {num_cells} cells: some cells get no episodes.")


def reset_object_on_grid(env, env_ids, cell_offsets, asset_cfg):
    """Put the object at its env's grid cell (offset from the default root position [m])."""
    asset = env.scene[asset_cfg.name]
    offsets = torch.tensor(cell_offsets, device=asset.device)[env_ids % len(cell_offsets)]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, :2] += offsets
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch[env_ids], env_ids=env_ids)


env_cfg.events.reset_object_position.func = reset_object_on_grid
env_cfg.events.reset_object_position.params = {
    "cell_offsets": (cells - torch.tensor([x0, y0])).tolist(),
    "asset_cfg": env_cfg.events.reset_object_position.params["asset_cfg"],
}
if args.command is not None:
    ranges = env_cfg.commands.object_pose.ranges
    ranges.pos_x, ranges.pos_y, ranges.pos_z = ((v, v) for v in args.command)
lift_height = env_cfg.rewards.lifting_object.params["minimal_height"]

def report(counts):
    """Write the csv and heat maps and print the failure table."""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    episodes = counts.sum(0)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "episodes", *OUTCOMES, *(o + "_rate" for o in OUTCOMES)])
        for c in range(num_cells):
            n = max(episodes[c].item(), 1)
            writer.writerow([
                round(cells[c, 0].item(), 4),
                round(cells[c, 1].item(), 4),
                episodes[c].item(),
                *(counts[k, c].item() for k in range(len(OUTCOMES))),
                *(round(counts[k, c].item() / n, 4) for k in range(len(OUTCOMES))),
            ])

    print(f"\nlift height {lift_height} m, training reset range x [{x_lo:.3f}, {x_hi:.3f}] y [{y_lo:.3f}, {y_hi:.3f}] m")
    fail = 1.0 - counts[0] / episodes.clamp(min=1)
    print("failure rate [%] (rows: x back -> front, columns: y left -> right)")
    print("   x \\ y " + "".join(f"{y:8.3f}" for y in ys.flip(0).tolist()))
    for i, x in enumerate(xs.tolist()):
        row = fail.reshape(args.nx, args.ny)[i].flip(0)
        print(f"{x:8.3f} " + "".join(f"{v * 100:8.1f}" for v in row.tolist()))

    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.patches import Rectangle  # noqa: E402

    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(5 * len(OUTCOMES), 5.2), constrained_layout=True)
    dx = (xs[1] - xs[0]).item() if args.nx > 1 else 0.02
    dy = (ys[1] - ys[0]).item() if args.ny > 1 else 0.02
    extent = [ys[0].item() - dy / 2, ys[-1].item() + dy / 2, xs[0].item() - dx / 2, xs[-1].item() + dx / 2]
    for k, (ax, name) in enumerate(zip(axes, OUTCOMES)):
        rate = (counts[k] / episodes.clamp(min=1)).reshape(args.nx, args.ny).numpy()
        cmap = "Greens" if name == "lifted" else "Reds"
        im = ax.imshow(rate, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(args.nx):
            for j in range(args.ny):
                ax.text(ys[j], xs[i], f"{rate[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
        ax.add_patch(Rectangle((y_lo, x_lo), y_hi - y_lo, x_hi - x_lo, fill=False, ls="--", lw=1.5, ec="black"))
        ax.invert_xaxis()  # +y is the robot's left: show it on the left, as seen from behind the robot
        ax.set_xlabel("object start y [m] (left <-> right)")
        ax.set_ylabel("object start x [m] (back <-> front)")
        ax.set_title(f"{name} [%]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{args.task}\n{os.path.basename(os.path.dirname(args.checkpoint))}/{os.path.basename(args.checkpoint)} | "
        f"{int(episodes.sum())} episodes, {int(episodes.min())}-{int(episodes.max())} per cell"
    )
    fig.savefig(args.out + ".png", dpi=120)
    print(f"\nwrote {args.out}.csv and {args.out}.png")


with launch_simulation(env_cfg, args):
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as base_env:
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        unwrapped = env.unwrapped
        obj = unwrapped.scene["object"]
        terms = unwrapped.termination_manager
        dev = unwrapped.device
        cell_of_env = torch.arange(args.num_envs, device=dev) % num_cells
        counts = torch.zeros(len(OUTCOMES), num_cells, dtype=torch.long, device=dev)
        ever_lifted = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)

        obs = env.get_observations()
        total_steps = args.rounds * unwrapped.max_episode_length
        for step in range(total_steps):
            # done envs are already reset after the step, so judge them from the state before it
            lifted_before = obj.data.root_pos_w.torch[:, 2] > lift_height
            ever_lifted |= lifted_before
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            done = dones.bool()
            if done.any():
                time_out = terms.time_outs & done
                outcome = torch.full((args.num_envs,), -1, dtype=torch.long, device=dev)
                outcome[done & ~time_out] = 1
                outcome[time_out & lifted_before] = 0
                outcome[time_out & ~lifted_before & ever_lifted] = 3
                outcome[time_out & ~ever_lifted] = 2
                for k in range(len(OUTCOMES)):
                    counts[k] += torch.bincount(cell_of_env[outcome == k], minlength=num_cells)
                ever_lifted[done] = False
            if (step + 1) % unwrapped.max_episode_length == 0:
                n = counts.sum().item()
                print(
                    f"[round {(step + 1) // unwrapped.max_episode_length}/{args.rounds}] {n} episodes, "
                    + ", ".join(f"{o} {counts[k].sum().item() / max(n, 1) * 100:.2f}%" for k, o in enumerate(OUTCOMES)),
                    flush=True,
                )

        # report before the app shuts down: the process exits when the context closes
        report(counts.cpu())
