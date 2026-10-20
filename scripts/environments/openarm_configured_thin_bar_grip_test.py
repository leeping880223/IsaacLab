# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether both right-hand fingers grip and lift a thin rigid cylinder.

The cylinder is positioned once at the TCP with gravity off during setup. Gravity is
restored after the fingers close; the cylinder is never moved by the script again.
The test uses the current bimanual lift task's robot USD and joint controls.

Run in the Isaac Lab project directory inside Docker::

    uv run --extra isaacsim python scripts/environments/openarm_thin_bar_grip_test.py \
        --num_envs 4 --diameter_mm 7 --mass_g 30 --viz kit --keep_open

Use ``--bar_offset X Y Z`` to adjust the initial cylinder centre in TCP coordinates
[m] if the open fingers do not surround it. A positive/negative gripper action
opens/closes the fingers, respectively. This is a scripted physics test, not RL.
Use ``--gripper_control joint1_only`` to leave joint2 without a drive target or
drive gains, so it can move only through the model's mechanical coupling.
"""

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import quat_apply
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK_NAME = "IsaacContrib-Lift-Cube-OpenArm-Bimanual-Configured"
OUTPUT_DIR = Path(__file__).resolve().parent / "grip_output"
LEFT_FINGER = "follower_r_left_finger"#"openarm_right_left_finger"  # 
RIGHT_FINGER = "follower_r_right_finger"#"openarm_right_right_finger"  # 
NEWTON_TO_GF = 1000.0 / 9.80665


def _term_slice(action_manager, name: str) -> slice:
    """Return the action columns belonging to a named action term."""
    start = 0
    for term_name in action_manager.active_terms:
        width = action_manager.get_term(term_name).action_dim
        if term_name == name:
            return slice(start, start + width)
        start += width
    raise RuntimeError(f"Action term {name!r} was not found: {action_manager.active_terms}")


def _object_normal_force(sensor) -> torch.Tensor:
    """Return each environment's finger-to-cylinder normal force magnitude [N]."""
    matrix = sensor.data.normal_force_matrix_w
    if matrix is None:
        raise RuntimeError("Contact force matrix is missing; check the finger sensor and Object filter paths.")
    return torch.linalg.vector_norm(torch.nan_to_num(matrix.torch[:, 0, 0, :]), dim=-1)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject invalid trial dimensions and durations before launching the simulation."""
    if (
        args.num_envs < 1
        or args.diameter_mm <= 0
        or not math.isfinite(args.mass_g)
        or args.mass_g <= 0
        or args.lift_height <= 0
        or any(value <= 0 for value in (args.settle_s, args.open_s, args.close_s, args.lift_s, args.hold_s))
    ):
        parser.error("num_envs, diameter_mm, mass_g, lift_height and phase durations must be positive")
    if args.physics_hz is not None and args.physics_hz < 1:
        parser.error("physics_hz must be positive")


def main() -> None:
    """Run a scripted close, lift and hold trial in parallel environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel robots (e.g. 4).")
    parser.add_argument("--diameter_mm", type=float, default=1.0, help="Cylinder diameter [mm].")
    parser.add_argument("--mass_g", type=float, default=30.0, help="Cylinder mass [g] (default: 30).")
    parser.add_argument(
        "--gripper_control",
        choices=("both", "joint1_only"),
        default="joint1_only",
        help="Drive both finger joints or only joint1, leaving joint2 passive.",
    )
    parser.add_argument(
        "--bar_offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),  # (0.03, 0.0, -0.01)
        metavar=("X", "Y", "Z"),
        help="Cylinder centre relative to the TCP [m].",
    )
    parser.add_argument(
        "--bar_offset_z_step",
        type=float,
        default=0.0,
        help="Add this TCP-local z offset [m] for each successive environment.",
    )
    parser.add_argument("--lift_height", type=float, default=0.05, help="Requested hand lift [m].")
    parser.add_argument("--settle_s", type=float, default=2.0, help="Time to let the open arm settle [s].")
    parser.add_argument("--open_s", type=float, default=0.2, help="Open-finger setup time [s].")
    parser.add_argument("--close_s", type=float, default=1.5, help="Finger closing time [s].")
    parser.add_argument("--lift_s", type=float, default=2.0, help="Hand lift time [s].")
    parser.add_argument("--hold_s", type=float, default=1.0, help="Hold after lift [s].")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed for repeatable trials.")
    parser.add_argument(
        "--physics_hz",
        type=int,
        default=None,
        help="Optional physics rate; default uses the lift task setting (100 Hz).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Per-step CSV output path (default includes diameter, mass and control mode)."
    )
    parser.add_argument("--fast", action="store_true", help="Run without real-time pacing.")
    parser.add_argument("--keep_open", action="store_true", help="Keep the Kit view running until Ctrl+C.")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=["kit"])
    args, hydra_overrides = parser.parse_known_args()

    _validate_args(args, parser)
    if args.csv is None:
        args.csv = OUTPUT_DIR / (
            f"openarm_configured_cylinder_{args.diameter_mm:g}mm_{args.mass_g:g}g_{args.gripper_control}_grip.csv"
        )

    env_cfg, _ = resolve_task_config(TASK_NAME, "", overrides=hydra_overrides)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.gripper_control == "joint1_only":
        '''
        env_cfg.actions.gripper_action.joint_names = ["openarm_right_finger_joint1"]
        env_cfg.actions.gripper_action.open_command_expr = {"openarm_right_finger_joint1": 0.044}
        env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint1": 0.0}
        '''
        env_cfg.actions.gripper_action.joint_names = ["follower_r_finger_joint1"]
        env_cfg.actions.gripper_action.open_command_expr = {"follower_r_finger_joint1": 0.044}
        env_cfg.actions.gripper_action.close_command_expr = {"follower_r_finger_joint1": 0.0}
    env_cfg.episode_length_s = 3600.0
    env_cfg.terminations.object_dropping = None
    env_cfg.events.reset_object_position = None
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    camera_eye = (3.0, 3.0, 2.0) if args.num_envs > 1 else (1.2, 1.2, 0.9)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=camera_eye, lookat=(0.0, 0.0, 0.1))
    if args.physics_hz is not None:
        env_cfg.sim.dt = 1.0 / args.physics_hz
        env_cfg.decimation = max(1, round(0.02 / env_cfg.sim.dt))
        env_cfg.sim.render_interval = env_cfg.decimation

    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        
        #[圓棍]
        spawn=sim_utils.CylinderCfg(
            radius=args.diameter_mm * 0.0005,
            height=0.06,
            axis="X",
        ###    
        
        #[方棍]
        #spawn=sim_utils.CuboidCfg(
        #    size=(0.06, args.diameter_mm * 0.001, args.diameter_mm * 0.001),
        ###    
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
                PhysxRigidBodyCfg(
                    disable_gravity=False, solver_position_iteration_count=16, solver_velocity_iteration_count=2
                ),
            ],
            mass_props=sim_utils.MassCfg(mass=args.mass_g / 1000.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.7, 0.75), metallic=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    for name, finger_body in (("left_finger_contact", LEFT_FINGER), ("right_finger_contact", RIGHT_FINGER)):
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm/{finger_body}",
                # prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_body}",
                update_period=0.0,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                debug_vis=True,
            ),
        )
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        import carb

        with contextlib.closing(gym.make(TASK_NAME, cfg=env_cfg)) as env:
            unwrapped = env.unwrapped
            try:
                env.reset()
                robot = unwrapped.scene["robot"]
                bar = unwrapped.scene["object"]
                left_sensor = unwrapped.scene["left_finger_contact"]
                right_sensor = unwrapped.scene["right_finger_contact"]
                action_manager = unwrapped.action_manager
                '''
                ###[機器人 : official]
                missing = {LEFT_FINGER, RIGHT_FINGER, "openarm_right_hand"} - set(robot.body_names)
                if missing:
                    raise RuntimeError(f"Robot USD is missing bodies {sorted(missing)}; found {robot.body_names}")

                print(f"[機器人] hand_tcp 是否為 articulation body：{'openarm_right_ee_tcp' in robot.body_names}")

                arm_term = action_manager.get_term("arm_action")
                arm_names = arm_term.IO_descriptor.joint_names
                expected_arm = {f"openarm_right_joint{index}" for index in range(1, 8)}
                if set(arm_names) != expected_arm:
                    raise RuntimeError(f"Expected seven right-arm joints, got {arm_names}")

                arm_ids = [robot.joint_names.index(name) for name in arm_names]
                gripper_joint_id = robot.joint_names.index("openarm_right_finger_joint1")
                follower_joint_id = robot.joint_names.index("openarm_right_finger_joint2")
                if args.gripper_control == "joint1_only":
                    # The task's implicit actuator config also drives joint2. Disable that
                    # drive so changing the action term alone does not bias this comparison.
                    robot.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=[follower_joint_id])
                    robot.write_joint_damping_to_sim_index(damping=0.0, joint_ids=[follower_joint_id])
                hand_idx = robot.body_names.index("openarm_right_hand")
                hand_jacobian_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
                jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in arm_ids]

                arm_slice = _term_slice(action_manager, "arm_action")
                gripper_slice = _term_slice(action_manager, "gripper_action")
                if gripper_slice.stop - gripper_slice.start != 1 or not isinstance(arm_term.cfg.scale, (float, int)):
                    raise RuntimeError("This test expects one binary gripper action and a scalar arm-action scale.")
                ###
                '''
                #[機器人 : configured]
                missing = {LEFT_FINGER, RIGHT_FINGER, "follower_r_hand"} - set(robot.body_names)
                if missing:
                    raise RuntimeError(f"Robot USD is missing bodies {sorted(missing)}; found {robot.body_names}")
                print(f"[機器人] hand_tcp 是否為 articulation body：{'follower_r_hand_tcp' in robot.body_names}")
                arm_term = action_manager.get_term("arm_action")
                arm_names = arm_term.IO_descriptor.joint_names
                expected_arm = {f"follower_r_joint{index}" for index in range(1, 8)}
                if set(arm_names) != expected_arm:
                    raise RuntimeError(f"Expected seven right-arm joints, got {arm_names}")
                arm_ids = [robot.joint_names.index(name) for name in arm_names]
                gripper_joint_id = robot.joint_names.index("follower_r_finger_joint1")
                follower_joint_id = robot.joint_names.index("follower_r_finger_joint2")
                hand_idx = robot.body_names.index("follower_r_hand")
                hand_jacobian_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
                jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in arm_ids]
                arm_slice = _term_slice(action_manager, "arm_action")
                gripper_slice = _term_slice(action_manager, "gripper_action")
                if gripper_slice.stop - gripper_slice.start != 1 or not isinstance(arm_term.cfg.scale, (float, int)):
                    raise RuntimeError("This test expects one binary gripper action and a scalar arm-action scale.")
                
                
                actions = torch.zeros_like(action_manager.action)
                actions[:, gripper_slice] = 1.0  # BinaryJointAction: nonnegative opens, negative closes.
                step_dt = unwrapped.step_dt
                configured_gravity_w = tuple(float(component) for component in unwrapped.sim.cfg.gravity)
                current_gravity_w = configured_gravity_w
                records: list[tuple] = []
                step_count = 0
                closing_peak = torch.zeros(args.num_envs, 2, device=unwrapped.device)
                gravity_peak = torch.zeros_like(closing_peak)
                lift_contact_count = torch.zeros_like(closing_peak)
                close_both_count = torch.zeros(args.num_envs, device=unwrapped.device)
                lift_both_count = torch.zeros_like(close_both_count)

                def step(stage: str, count: int, *, record_force: bool = True) -> None:
                    nonlocal step_count
                    started = time.perf_counter()
                    env.step(actions)
                    step_count += 1
                    gravity_m_s2 = math.sqrt(sum(component**2 for component in current_gravity_w))
                    object_gravity_n = args.mass_g * 0.001 * gravity_m_s2
                    object_gravity_gf = object_gravity_n * NEWTON_TO_GF
                    forces = torch.stack(
                        (_object_normal_force(left_sensor), _object_normal_force(right_sensor)), dim=-1
                    )
                    if stage == "close":
                        closing_peak[:] = torch.maximum(closing_peak, forces)
                        close_both_count[:] += ((forces[:, 0] > 0.1) & (forces[:, 1] > 0.1)).float()
                    elif stage in ("lift", "hold"):
                        gravity_peak[:] = torch.maximum(gravity_peak, forces)
                        if stage == "lift":
                            lift_contact_count[:] += (forces > 0.1).float()
                            lift_both_count[:] += ((forces[:, 0] > 0.1) & (forces[:, 1] > 0.1)).float()
                    if record_force:
                        bar_z = bar.data.root_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]
                        hand_z = robot.data.body_pos_w.torch[:, hand_idx, 2] - unwrapped.scene.env_origins[:, 2]
                        joint_pos = robot.data.joint_pos.torch
                        joint_targets = robot.data.joint_pos_target.torch
                        # ImplicitActuator estimates drive effort from position/velocity error.
                        # It is not a PhysX measurement of the joint's actual force.
                        drive_est = robot.data.applied_torque.torch
                        rows = (
                            torch.stack(
                                (
                                    forces[:, 0],
                                    forces[:, 1],
                                    bar_z,
                                    hand_z,
                                    joint_pos[:, gripper_joint_id],
                                    joint_pos[:, follower_joint_id],
                                    joint_targets[:, gripper_joint_id],
                                    joint_targets[:, follower_joint_id],
                                    drive_est[:, gripper_joint_id],
                                    drive_est[:, follower_joint_id],
                                ),
                                dim=-1,
                            )
                            .cpu()
                            .tolist()
                        )
                        records.extend(
                            (env_id, stage, step_count * step_dt, *row, gravity_m_s2, object_gravity_n, object_gravity_gf)
                            for env_id, row in enumerate(rows)
                        )
                        print(
                            f"{stage:5s} t={step_count * step_dt:5.2f}s env0 "
                            f"g={gravity_m_s2:.2f}m/s² 圓柱重力={object_gravity_n:.3f}N({object_gravity_gf:.1f}gf) "
                            f"左={rows[0][0]:6.2f}N 右={rows[0][1]:6.2f}N "
                            f"圓柱z={rows[0][2]:.3f}m 手z={rows[0][3]:.3f}m "
                            f"關節1/2={rows[0][4]:.4f}/{rows[0][5]:.4f}m "
                            f"驅動力估計1/2={rows[0][8]:.2f}/{rows[0][9]:.2f}N",
                            flush=True,
                        )
                    elif stage == "settle":
                        print(
                            f"{stage:5s} t={step_count * step_dt:5.2f}s "
                            f"g={gravity_m_s2:.2f}m/s² 圓柱重力={object_gravity_n:.3f}N({object_gravity_gf:.1f}gf)",
                            flush=True,
                        )
                    if not args.fast:
                        time.sleep(max(0.0, step_dt - (time.perf_counter() - started)))

                def phase_steps(duration_s: float) -> int:
                    return max(1, math.ceil(duration_s / step_dt))

                print(
                    f"\n[測試] {args.num_envs} 個環境，物理 {1 / unwrapped.sim.cfg.dt:.0f} Hz，"
                    f"控制 {1 / step_dt:.0f} Hz，圓柱長 60 mm、直徑 {args.diameter_mm:g} mm / "
                    f"{args.mass_g:g} g（重量約 {args.mass_g / NEWTON_TO_GF:.3f} N）；"
                    f"夾爪控制={args.gripper_control}"
                )
                print(
                    f"[測試] TCP 局部偏移 {tuple(args.bar_offset)} m；先穩定手臂，"
                    "再在零重力下閉合，最後恢復重力並抬升。"
                )
                with torch.inference_mode():
                    for index in range(phase_steps(args.settle_s)):
                        step("settle", index, record_force=False)

                    # Place the cylinder once, after the robot has reached its open-hand pose.
                    unwrapped.sim.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
                    current_gravity_w = (0.0, 0.0, 0.0)
                    tcp = unwrapped.scene["ee_frame"].data
                    tcp_pos = tcp.target_pos_w.torch[:, 0, :].clone()
                    tcp_quat = tcp.target_quat_w.torch[:, 0, :].clone()
                    offset = (
                        torch.tensor(args.bar_offset, device=unwrapped.device, dtype=tcp_pos.dtype)
                        .expand_as(tcp_pos)
                        .clone()
                    )
                    offset[:, 2] += torch.arange(args.num_envs, device=unwrapped.device) * args.bar_offset_z_step
                    bar_pos = tcp_pos + quat_apply(tcp_quat, offset)
                    bar.write_root_pose_to_sim_index(root_pose=torch.cat((bar_pos, tcp_quat), dim=-1))
                    bar.write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros(args.num_envs, 6, device=unwrapped.device)
                    )
                    approach_axis = torch.tensor((0.0, 0.0, 1.0), device=unwrapped.device).expand_as(tcp_pos)
                    approach_world = quat_apply(tcp_quat, approach_axis)
                    print(
                        f"[擺位] env0 TCP={tcp_pos[0].tolist()}，圓柱中心={bar_pos[0].tolist()}，"
                        f"TCP local +z 的世界方向={approach_world[0].tolist()}"
                    )
                    print(f"[擺位] 各環境 TCP 局部偏移 [m]：{offset.tolist()}")

                    for index in range(phase_steps(args.open_s)):
                        step("open", index)
                    open_forces = torch.stack(
                        (_object_normal_force(left_sensor), _object_normal_force(right_sensor)), dim=-1
                    )
                    if torch.any(open_forces > 0.1):
                        print("[注意] 張開時已有指片接觸圓柱；請在 Kit 檢查是否穿模，必要時調 --bar_offset。")
                    placement_error = torch.linalg.vector_norm(bar.data.root_pos_w.torch - bar_pos, dim=-1)
                    if torch.any(placement_error > 0.01):
                        print("[注意] 圓柱在張開階段已移動超過 1 cm；請檢查與手掌或桌子的碰撞。")

                    actions[:, gripper_slice] = -1.0
                    for index in range(phase_steps(args.close_s)):
                        step("close", index)
                    if torch.any(closing_peak.sum(dim=-1) < 0.1):
                        print("[注意] 有環境在閉合時兩指都沒有碰到圓柱；該環境不能用來判斷指片碰撞 bug。")

                    hand_start = robot.data.body_pos_w.torch[:, hand_idx, :].clone()
                    bar_start = bar.data.root_pos_w.torch.clone()
                    unwrapped.sim.physics_sim_view.set_gravity(carb.Float3(*unwrapped.sim.cfg.gravity))
                    current_gravity_w = configured_gravity_w
                    print("[測試] 已恢復重力，開始抬手；此後不再設定圓柱位置。")

                    ik_cfg = DifferentialIKControllerCfg(
                        command_type="position",
                        use_relative_mode=False,
                        ik_method="dls",
                        ik_params={"lambda_val": 0.05},
                    )
                    ik = DifferentialIKController(ik_cfg, num_envs=args.num_envs, device=unwrapped.device)
                    lift_steps = phase_steps(args.lift_s)
                    for index in range(lift_steps):
                        fraction = (index + 1) / lift_steps
                        target = hand_start.clone()
                        target[:, 2] += args.lift_height * fraction
                        hand_pos = robot.data.body_pos_w.torch[:, hand_idx, :]
                        hand_quat = robot.data.body_quat_w.torch[:, hand_idx, :]
                        jacobian = robot.data.body_link_jacobian_w.torch[:, hand_jacobian_idx, :, jacobian_joint_ids]
                        joint_pos = robot.data.joint_pos.torch[:, arm_ids]
                        ik.set_command(target, ee_quat=hand_quat)
                        desired = ik.compute(hand_pos, hand_quat, jacobian, joint_pos)
                        limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids, :]
                        desired = torch.minimum(torch.maximum(desired, limits[..., 0]), limits[..., 1])
                        defaults = robot.data.default_joint_pos.torch[:, arm_ids]
                        actions[:, arm_slice] = (desired - defaults) / float(arm_term.cfg.scale)
                        step("lift", index)

                    for index in range(phase_steps(args.hold_s)):
                        step("hold", index)

                    hand_end = robot.data.body_pos_w.torch[:, hand_idx, :]
                    bar_end = bar.data.root_pos_w.torch
                    hand_rise = hand_end[:, 2] - hand_start[:, 2]
                    bar_rise = bar_end[:, 2] - bar_start[:, 2]
                    slip = torch.linalg.vector_norm((bar_end - hand_end) - (bar_start - hand_start), dim=-1)
                    contact_rate = lift_contact_count / lift_steps
                    print("\n[結果] 力是指片對圓柱的法向接觸力；接觸率以 0.1 N 為界。")
                    for env_id in range(args.num_envs):
                        print(
                            f"env {env_id}: 閉合峰值 左/右={closing_peak[env_id, 0]:.2f}/"
                            f"{closing_peak[env_id, 1]:.2f} N；重力後峰值 左/右="
                            f"{gravity_peak[env_id, 0]:.2f}/{gravity_peak[env_id, 1]:.2f} N；"
                            f"抬升接觸率 左/右={contact_rate[env_id, 0]:.0%}/"
                            f"{contact_rate[env_id, 1]:.0%}；雙指同時接觸 閉合/抬升="
                            f"{close_both_count[env_id] / phase_steps(args.close_s):.0%}/"
                            f"{lift_both_count[env_id] / lift_steps:.0%}；手/圓柱上升="
                            f"{hand_rise[env_id]:.3f}/{bar_rise[env_id]:.3f} m；相對滑移="
                            f"{slip[env_id]:.3f} m"
                        )
                    print("[判讀] 手沒有升到約 5 cm 時，先查手臂動作；某指全程 0 N 時，先檢查圓柱位置與該指片碰撞殼。")

                args.csv.parent.mkdir(parents=True, exist_ok=True)
                with args.csv.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file, lineterminator="\n")
                    writer.writerow(
                        (
                            "env",
                            "phase",
                            "time_s",
                            "left_normal_N",
                            "right_normal_N",
                            "cylinder_z_m",
                            "hand_z_m",
                            "finger_joint1_m",
                            "finger_joint2_m",
                            "finger_joint1_target_m",
                            "finger_joint2_target_m",
                            "finger_joint1_drive_est_N",
                            "finger_joint2_drive_est_N",
                            "gravity_m_s2",
                            "object_gravity_N",
                            "object_gravity_gf",
                        )
                    )
                    writer.writerows(records)
                print(f"[資料] 已寫入 {args.csv}")

                if args.keep_open:
                    print("[畫面] 繼續模擬並保持 Kit 視窗；在終端按 Ctrl+C 結束。")
                    try:
                        with torch.inference_mode():
                            index = 0
                            while True:
                                step("view", index, record_force=False)
                                index += 1
                    except KeyboardInterrupt:
                        pass
            finally:
                unwrapped.sim.physics_sim_view.set_gravity(carb.Float3(*unwrapped.sim.cfg.gravity))


if __name__ == "__main__":
    main()
