# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL policy in slow motion: same args as `isaaclab play`, plus --speed (e.g. 0.1).

--joint_info N prints joint targets, positions and torques every N steps (0 disables).
--episode_stats 1 prints per-phase saturation statistics over finished episodes (0 disables).
--show_state 1 prints env 0's phase changes and episode outcomes (default: on when num_envs is 1).
--lift_height / --grasp_dist / --goal_dist set the phase thresholds [m].
--watch_stuck N prints the envs whose object is still not lifted at episode step N and points the
camera at the first one (0 disables). Episodes that end not lifted are listed by env id as well.
"""

import sys
import time

import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import common, run_play_cli  # noqa: E402
from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

argv = sys.argv[1:]


def _pop_arg(name, default, cast):
    if name not in argv:
        return default
    i = argv.index(name)
    value = cast(argv[i + 1])
    del argv[i : i + 2]
    return value


speed = _pop_arg("--speed", 0.1, float)
joint_info = _pop_arg("--joint_info", 25, int)
episode_stats = _pop_arg("--episode_stats", 1, int)
show_state = _pop_arg("--show_state", None, int)
lift_height = _pop_arg("--lift_height", None, float)  # None: read from the lifting_object reward
grasp_dist = _pop_arg("--grasp_dist", 0.03, float)
goal_dist = _pop_arg("--goal_dist", 0.05, float)
watch_stuck = _pop_arg("--watch_stuck", 0, int)

_env = []
_stats = []
_legend_printed = []
_orig_create_env = play_rsl_rl.create_isaaclab_env


def _capture_create_env(*args, **kwargs):
    env = _orig_create_env(*args, **kwargs)
    _env.append(env)
    return env


def _print_joint_info(step_count):
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    env = _env[0].unwrapped
    n = env.num_envs
    scope = "env 0" if n == 1 else f"mean of {n} envs"
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if not isinstance(term, JointPositionAction):
            continue
        data = term._asset.data
        ids = term._joint_ids
        lim = data.soft_joint_pos_limits.torch[0, ids]
        lo, hi = lim[:, 0], lim[:, 1]
        raw = term.raw_actions
        target = term.processed_actions
        q = data.joint_pos.torch[:, ids]
        computed = data.computed_torque.torch[:, ids]
        applied = data.applied_torque.torch[:, ids]
        # per-env flags, averaged over envs -> fraction of envs
        t_out = ((target < lo - 1e-3) | (target > hi + 1e-3)).float().mean(0)
        q_out = ((q < lo - 1e-3) | (q > hi + 1e-3)).float().mean(0)
        sat = (computed.abs() > applied.abs() + 1e-3).float().mean(0)
        print(f"[joint_info] step {step_count} | {name} ({scope})")
        if not _legend_printed:
            print(
                "  raw act = policy output (unitless)\n"
                "  target  = raw act * scale + default joint pos [rad], sent to the PD drive\n"
                "  q       = measured joint position [rad]\n"
                "  pd want = kp * (target - q) - kd * joint vel [N*m]\n"
                "  torque  = pd want clamped to the effort limit [N*m]\n"
                "  * = outside [lower, upper]   ! = pd want exceeds effort limit (saturated)\n"
                "  with num_envs > 1: values are means over envs, a mark means at least one env,\n"
                "  and t out% / q out% / sat% give the fraction of envs"
            )
            _legend_printed.append(True)
        header = (
            f"  {'joint':<24}{'lower':>8}{'upper':>8}{'raw act':>9}{'target':>9}{'q':>9}{'pd want':>10}{'torque':>9}"
        )
        if n > 1:
            header += f"{'t out%':>8}{'q out%':>8}{'sat%':>7}"
        print(header)
        for j, jname in enumerate(term._joint_names):
            row = (
                f"  {jname:<24}{lo[j].item():8.3f}{hi[j].item():8.3f}{raw[:, j].mean().item():9.2f}"
                f"{target[:, j].mean().item():8.2f}{'*' if t_out[j] > 0 else ' '}"
                f"{q[:, j].mean().item():8.3f}{'*' if q_out[j] > 0 else ' '}"
                f"{computed[:, j].mean().item():10.1f}{applied[:, j].mean().item():8.2f}{'!' if sat[j] > 0 else ' '}"
            )
            if n > 1:
                row += f"{t_out[j].item() * 100:7.0f}%{q_out[j].item() * 100:7.0f}%{sat[j].item() * 100:6.0f}%"
            print(row)
    # same quantity as mdp.action_l2: sum of squared raw actions over all action dims
    l2 = env.action_manager.action.pow(2).sum(-1)
    l2_mean = l2.mean().item()
    print(
        f"  action_l2 (sum raw^2, all action dims): mean {l2_mean:.1f}"
        + (f"  min {l2.min().item():.1f}  max {l2.max().item():.1f}" if n > 1 else "")
        + f"  -> Episode_Reward/s at weight -1e-2: {-1e-2 * l2_mean:.2f}, -1e-3: {-1e-3 * l2_mean:.2f}"
    )
    sys.stdout.flush()


class _EpisodeStats:
    """Classifies every step into a phase from the object state and commits per-episode joint statistics.

    Phases: reach (object not lifted), lift (lifted, in the gripper, away from the goal),
    hold (lifted, in the gripper, at the goal) and loose (lifted but not in the gripper).
    Only clean episodes (time out while carrying, never dropped back to the table) are counted;
    the other outcomes are only tallied. Environments that reset during a step are judged with
    the state saved one step earlier, because the env resets them inside ``step()``.
    """

    PHASES = ("reach", "lift", "hold", "loose")
    REACH, LIFT, HOLD, LOOSE = range(4)
    DROP_HYSTERESIS = 0.01  # carried object counts as dropped only below lift_height - this [m]

    def __init__(self, env, term):
        import torch

        from isaaclab.utils.math import combine_frame_transforms

        self.torch, self.combine = torch, combine_frame_transforms
        self.env, self.term = env, term
        n, j, p = env.num_envs, len(term._joint_names), len(self.PHASES)
        dev = env.device
        zeros = lambda *shape: torch.zeros(*shape, device=dev)  # noqa: E731
        self.robot, self.obj, self.ee = env.scene["robot"], env.scene["object"], env.scene["ee_frame"]
        self.lift_h = lift_height
        if self.lift_h is None:
            cfg = getattr(env.cfg.rewards, "lifting_object", None)
            self.lift_h = cfg.params.get("minimal_height", 0.04) if cfg is not None else 0.04
        self.has_drop = "object_dropping" in env.termination_manager.active_terms
        self.show = show_state if show_state is not None else int(n == 1)
        # per-env, current episode
        self.cur = {"steps": zeros(n, p), "sat": zeros(n, p, j), "qout": zeros(n, p, j), "l2": zeros(n, p)}
        self.cur_over = zeros(n, j)
        self.first = {"lift": torch.full((n,), -1, device=dev), "hold": torch.full((n,), -1, device=dev)}
        self.min_goal = torch.full((n,), float("inf"), device=dev)
        self.drops = torch.zeros(n, dtype=torch.long, device=dev)
        self.carrying = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_state = torch.zeros(n, dtype=torch.long, device=dev)
        self.ep_index = 0
        # totals over clean episodes
        self.tot = {k: torch.zeros_like(v[0]) for k, v in self.cur.items()}
        self.tot_over = zeros(j)
        self.first_sum = {"lift": 0.0, "hold": 0.0}
        self.first_n = {"lift": 0, "hold": 0}
        self.min_goal_sum = 0.0
        self.counts = {"clean": 0, "regrasp": 0, "not lifted": 0, "dropped": 0}
        # gripper distance while lifted, over all episodes (to check the grasp_dist threshold)
        self.tcp = {"n": 0, "sum": 0.0, "max": 0.0, "above": 0}
        self.printed_at = 0

    def _finish(self, done):
        torch = self.torch
        dropped = done & self.env.termination_manager.get_term("object_dropping") if self.has_drop else done & False
        carrying_end = done & ~dropped & ((self.prev_state == self.LIFT) | (self.prev_state == self.HOLD))
        clean = carrying_end & (self.drops == 0)
        regrasp = carrying_end & (self.drops > 0)
        self.counts["clean"] += int(clean.sum())
        self.counts["regrasp"] += int(regrasp.sum())
        self.counts["dropped"] += int(dropped.sum())
        not_lifted = done & ~dropped & ~carrying_end
        self.counts["not lifted"] += int(not_lifted.sum())
        if dropped.any():
            print(f"[dropped] episode ended with the object off the table in env {dropped.nonzero().flatten().tolist()}", flush=True)
        if not_lifted.any():
            never =(not_lifted & ~self.ever_lifted).nonzero().flatten().tolist()
            lost = (not_lifted & self.ever_lifted).nonzero().flatten().tolist()
            print(
                f"[not_lifted] episode ended: never lifted in env {never}, "
                f"lifted but not in the gripper at the end in env {lost}",
                flush=True,
            )
        if self.show and bool(done[0]):
            outcome = "clean" if clean[0] else "regrasp" if regrasp[0] else "dropped" if dropped[0] else "not lifted"
            print(f"[state] episode {self.ep_index} ended: {outcome} (drops back to table: {int(self.drops[0])})")
            self.ep_index += 1
        for k, v in self.cur.items():
            self.tot[k] += v[clean].sum(0)
            v[done] = 0
        if clean.any():
            self.tot_over = torch.maximum(self.tot_over, self.cur_over[clean].max(0).values)
            self.min_goal_sum += self.min_goal[clean].sum().item()
            for k, first in self.first.items():
                hit = clean & (first >= 0)
                self.first_sum[k] += first[hit].sum().item()
                self.first_n[k] += int(hit.sum())
        self.cur_over[done] = 0
        self.min_goal[done] = float("inf")
        self.drops[done] = 0
        self.carrying[done] = False
        for first in self.first.values():
            first[done] = -1

    def update(self):
        torch, env, term = self.torch, self.env, self.term
        done = env.reset_buf.clone()
        if done.any():
            self._finish(done)
        live = ~done

        # object state
        obj_pos = self.obj.data.root_pos_w.torch
        goal_w, _ = self.combine(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            env.command_manager.get_command("object_pose")[:, :3],
        )
        tcp_d = torch.linalg.norm(obj_pos - self.ee.data.target_pos_w.torch[:, 0], dim=-1)
        goal_d = torch.linalg.norm(obj_pos - goal_w, dim=-1)
        lifted = obj_pos[:, 2] > self.lift_h
        in_hand = tcp_d < grasp_dist
        state = torch.full_like(self.prev_state, self.REACH)
        state[lifted & ~in_hand] = self.LOOSE
        state[lifted & in_hand] = self.LIFT
        state[lifted & in_hand & (goal_d < goal_dist)] = self.HOLD

        # events
        step_idx = env.episode_length_buf
        dropped_back = live & self.carrying & (obj_pos[:, 2] < self.lift_h - self.DROP_HYSTERESIS)
        self.drops += dropped_back.long()
        self.carrying = (self.carrying & ~dropped_back) | (live & lifted & in_hand)
        for k, phase in (("lift", self.LIFT), ("hold", self.HOLD)):
            reached = (state == phase) | (state == self.HOLD) if k == "lift" else state == phase
            new = live & reached & (self.first[k] < 0)
            self.first[k][new] = step_idx[new]
        self.ever_lifted = torch.where(live, self.ever_lifted | lifted, torch.zeros_like(lifted))
        if watch_stuck > 0:
            stuck = live & (step_idx == watch_stuck) & ~self.ever_lifted
            if stuck.any():
                ids = stuck.nonzero().flatten().tolist()
                i = ids[0]
                target = obj_pos[i].tolist()
                eye = (target[0] + 0.8, target[1] + 0.8, target[2] + 0.6)
                env.sim.set_camera_view(eye, tuple(target))
                print(
                    f"[watch_stuck] step {watch_stuck}: object not lifted yet in env {ids}; camera -> env {i} "
                    f"(object at {[round(x, 2) for x in target]})",
                    flush=True,
                )
        self.min_goal = torch.where(live & lifted & in_hand, torch.minimum(self.min_goal, goal_d), self.min_goal)
        sel = live & lifted
        if sel.any():
            d = tcp_d[sel]
            self.tcp["n"] += d.numel()
            self.tcp["sum"] += d.sum().item()
            self.tcp["max"] = max(self.tcp["max"], d.max().item())
            self.tcp["above"] += int((d >= grasp_dist).sum())

        # joint statistics
        onehot = torch.nn.functional.one_hot(state, len(self.PHASES)).float() * live[:, None].float()
        data = term._asset.data
        ids = term._joint_ids
        lim = data.soft_joint_pos_limits.torch[0, ids]
        q = data.joint_pos.torch[:, ids]
        sat = (data.computed_torque.torch[:, ids].abs() > data.applied_torque.torch[:, ids].abs() + 1e-3).float()
        over = torch.maximum(lim[:, 0] - q, q - lim[:, 1]).clamp(min=0.0)
        l2 = env.action_manager.action.pow(2).sum(-1)
        self.cur["steps"] += onehot
        self.cur["sat"] += onehot[:, :, None] * sat[:, None, :]
        self.cur["qout"] += onehot[:, :, None] * (over > 1e-3).float()[:, None, :]
        self.cur["l2"] += onehot * l2[:, None]
        self.cur_over = torch.where(live[:, None], torch.maximum(self.cur_over, over), self.cur_over)

        if self.show and bool(live[0]) and (state[0] != self.prev_state[0] or step_idx[0] == 1):
            old = "start" if step_idx[0] == 1 else self.PHASES[int(self.prev_state[0])]
            print(
                f"[state] episode {self.ep_index} step {int(step_idx[0])} ({float(step_idx[0]) * env.step_dt:.2f} s): "
                f"{old} -> {self.PHASES[int(state[0])]} | object z {obj_pos[0, 2].item():.3f} m, "
                f"to gripper {tcp_d[0].item():.3f} m, to goal {goal_d[0].item():.3f} m",
                flush=True,
            )
        if self.show and bool(dropped_back[0]):
            print(f"[state] episode {self.ep_index} step {int(step_idx[0])}: dropped back to table", flush=True)
        self.prev_state = torch.where(live, state, torch.full_like(state, self.REACH))

        finished = sum(self.counts.values())
        if episode_stats and finished - self.printed_at >= env.num_envs:
            self.printed_at = finished
            self.report()

    def report(self):
        steps = self.tot["steps"].clamp(min=1)
        c = self.counts
        print(
            f"[episode_stats] clean {c['clean']} | regrasp {c['regrasp']} | not lifted {c['not lifted']} | "
            f"dropped {c['dropped']}  (statistics use clean episodes only)"
        )
        print(
            f"  thresholds: lifted z > {self.lift_h:.3f} m, in gripper < {grasp_dist:.3f} m, "
            f"at goal < {goal_dist:.3f} m"
        )
        total = self.tot["steps"].sum().clamp(min=1).item()
        print(
            "  phase time share: "
            + "  ".join(f"{p} {self.tot['steps'][i].item() / total * 100:.0f}%" for i, p in enumerate(self.PHASES))
        )
        dt = self.env.step_dt
        firsts = []
        for k in ("lift", "hold"):
            if self.first_n[k]:
                mean = self.first_sum[k] / self.first_n[k]
                firsts.append(f"first {k} at step {mean:.0f} ({mean * dt:.2f} s, {self.first_n[k]} episodes)")
            else:
                firsts.append(f"first {k}: never")
        print("  " + ", ".join(firsts))
        if c["clean"]:
            print(f"  closest object-to-goal distance per clean episode: mean {self.min_goal_sum / c['clean']:.3f} m")
        if self.tcp["n"]:
            t = self.tcp
            print(
                f"  object-to-gripper distance while lifted: mean {t['sum'] / t['n']:.3f} m, max {t['max']:.3f} m, "
                f"{t['above'] / t['n'] * 100:.1f}% of steps >= grasp_dist"
            )
        print(
            f"  {'joint':<24}"
            + "".join(f"{'sat% ' + p:>12}" for p in self.PHASES[:3])
            + f"{'q out% hold':>13}{'max over [rad]':>16}"
        )
        for j, jname in enumerate(self.term._joint_names):
            sat = "".join(f"{self.tot['sat'][p, j].item() / steps[p].item() * 100:11.1f}%" for p in range(3))
            qout = self.tot["qout"][self.HOLD, j].item() / steps[self.HOLD].item() * 100
            print(f"  {jname:<24}{sat}{qout:12.1f}%{self.tot_over[j].item():16.3f}")
        l2 = [self.tot["l2"][p].item() / steps[p].item() for p in range(3)]
        print(
            "  action_l2 mean: "
            + "  ".join(f"{p} {v:.1f}" for p, v in zip(self.PHASES, l2))
            + f"  -> hold Episode_Reward/s at weight -1e-2: {-1e-2 * l2[self.HOLD]:.2f}"
            + f", -1e-3: {-1e-3 * l2[self.HOLD]:.2f}"
        )
        sys.stdout.flush()


def _update_episode_stats():
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    if not _stats:
        env = _env[0].unwrapped
        terms = [env.action_manager.get_term(n) for n in env.action_manager.active_terms]
        term = next((t for t in terms if isinstance(t, JointPositionAction)), None)
        if term is None:
            return
        _stats.append(_EpisodeStats(env, term))
    _stats[0].update()


_orig_run_playback = common.run_playback


def _slow_run_playback(step, *, dt, args_cli, env_cfg):
    args_cli.real_time = True
    count = [0, time.time()]

    def counted_step():
        step()
        count[0] += 1
        if count[0] % 50 == 0:
            print(
                f"[slow_play] {count[0]} steps, {count[0] / (time.time() - count[1]):.1f} steps/s "
                f"(sim runs {dt:.3f}s per step -> {speed}x)",
                flush=True,
            )
        if joint_info > 0 and count[0] % joint_info == 0 and _env:
            _print_joint_info(count[0])
        if (episode_stats or show_state != 0) and _env:
            _update_episode_stats()

    return _orig_run_playback(counted_step, dt=dt / speed, args_cli=args_cli, env_cfg=env_cfg)


play_rsl_rl.create_isaaclab_env = _capture_create_env
play_rsl_rl.run_playback = _slow_run_playback
sys.exit(run_play_cli(argv))
