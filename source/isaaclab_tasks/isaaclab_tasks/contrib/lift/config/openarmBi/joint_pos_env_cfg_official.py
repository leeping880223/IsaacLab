# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab_newton.sim.schemas import NewtonArticulationCfg  # 9.開啟自碰撞
from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,  # 9. 開啟自碰撞
    PhysxRigidBodyCfg,
)

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.lift import mdp
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg

from isaaclab_assets.robots.openarm import (
    OPENARM_BI_CFG,  # 1.換雙臂 #官網的有bug #[更正robot]但這邊還是要先匯入 後面做更換
)

import isaaclab.sim as sim_utils #改cube重量

from isaaclab.managers import CurriculumTermCfg as CurrTerm #延遲懲罰

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab.managers import RewardTermCfg as RewTerm #方塊旋轉reward

# The OpenArm is smaller than the Franka, so the goal workspace and the joint-scoped
# terms are re-tuned for it relative to the shared lift base.
OPENARM_LEFT_JOINTS = [
    "openarm_left_joint[1-7]",  # 官網的有bug
    "openarm_left_finger_joint.*",  # 官網的有bug
    # "follower_l_joint[1-7]",       # 右臂 7 軸
    # "follower_l_finger_joint.*",    # 右手兩片手指
]

OPENARM_RIGHT_JOINTS = [
    "openarm_right_joint[1-7]",  # 官網的有bug
    "openarm_right_finger_joint.*",  # 官網的有bug
    # "follower_r_joint[1-7]",       # 右臂 7 軸
    # "follower_r_finger_joint.*",    # 右手兩片手指
]

def spawn_from_usd_with_shrink_wrapped_body(prim_path, cfg, *args, **kwargs):
    """Spawn the robot USD with the body collider shrink-wrapped onto its mesh.

    The default convex decomposition of the body bulges at the left shoulder and overlaps the left link2,
    which the solver pushes on with millions of newtons when self-collisions are enabled.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import PhysxSchema

    prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path):
        collider = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/openarm_body_link/collisions")
        if collider.IsValid():
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(collider).CreateShrinkWrapAttr().Set(True)
    return prim


@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set OpenArm as robot

        self.scene.robot = OPENARM_BI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # 2.換機器人
        # [更正robot]
        """
        self.scene.robot.spawn.usd_path = str(
            Path(__file__).resolve().parents[7]
            / "source/isaaclab_assets/data/Robots/OpenArm/openarm_configured.usd"
        )
        self.scene.robot.articulation_root_prim_path = "/openarm/root_joint"
        self.scene.robot.actuators = {
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["follower_[lr]_joint[1-7]"],
                stiffness=None,
                damping=None,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["follower_[lr]_finger_joint1"],
                stiffness=None,
                damping=None,
            ),
        }
        """

        # 9. 關自碰撞（）
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg)
        ).enabled_self_collisions = False
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, NewtonArticulationCfg)
        ).self_collision_enabled = False
        
        # [更正robot]
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(-0.40, 0.0, -0.25),  # 5-1.機器人向下25cm 後40cm
            joint_pos={  # 5-4. 改初始姿勢以免手肘撞到
                # 5-4 右手：手肘彎、手在桌面上方、夾爪朝下
                "openarm_right_joint1": 0.589,
                "openarm_right_joint2": 1.03,
                "openarm_right_joint3": -0.991,
                "openarm_right_joint4": 1.571,
                "openarm_right_joint5": 1.208,
                "openarm_right_joint6": 0.169,
                "openarm_right_joint7": -1.267,
                "openarm_right_finger_joint.*": 0.044,  # 夾爪先張開
                # 左手：不控制，維持原樣
                "openarm_left_joint.*": 0.0,
                "openarm_left_finger_joint.*": 0.0,
            },
        )



        # Scope the joint-space terms to the OpenArm's own joints.
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_RIGHT_JOINTS)
        }  # 1.單獨右手mdp #官網的有bug : joint_names=OPENARM_RIGHT_JOINTS
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_RIGHT_JOINTS)
        }  # 1.單獨右手mdp #官網的有bug : joint_names=OPENARM_RIGHT_JOINTS
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=OPENARM_RIGHT_JOINTS
        )  # 1.單獨右手mdp

        # Shrink the goal workspace to the smaller arm's reach. #5-3.目標會跟著機器人移動,目標位置往桌角
        self.commands.object_pose.ranges.pos_x = (0.20, 0.30)  # (0.2, 0.4) => (0.20, 0.30)
        self.commands.object_pose.ranges.pos_y = (-0.4, -0.2)  # (-0.2, 0.2) => (-0.4,-0.2)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)  # (0.15, 0.4) ＝> (0.35, 0.45) #>0.04 ok

        # Re-tuned reach weight, and no success tracking on this task.
        self.rewards.reaching_object.weight = 1.1
        # del self.rewards.object_goal_tracking.params["success_threshold"] #紀錄success_threshold

        # Set actions for the specific robot type (OpenArm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint.*",  # 2.手臂改右手
                # "follower_r_joint[1-7]",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            # [更正robot]
            joint_names=["openarm_right_finger_joint.*"],  # 2.抓取改右手
            open_command_expr={"openarm_right_finger_joint.*": 0.044},  # 2.抓取改右手
            close_command_expr={"openarm_right_finger_joint.*": 0.0},  # 2.抓取改右手
            # joint_names=["follower_r_finger_joint1"],#2.抓取改右手
            # open_command_expr={"follower_r_finger_joint1": 0.044},#2.抓取改右手
            # close_command_expr={"follower_r_finger_joint1": 0.0},#2.抓取改右手
        )

        # Set the body name for the end effector
        # [更正robot]
        self.commands.object_pose.body_name = "openarm_right_hand"  # 2.改右手（這是common ee）
        # self.commands.object_pose.body_name = "follower_r_hand" #2.改右手（這是common ee）
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.object_pose.debug_vis = True  # 8.顯示目標位置
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-0.15, -0.25, 0.055], rot=[0, 0, 0, 1]
            ),  # 5-2.Cube往桌角 0.4, 0.0 , 0.055 => -0.15, -0.25 , 0.055 -0.1 , 0.055  , 0.055
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=PhysxRigidBodyCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        # 5-2.方塊隨機範圍：相對上面 init_state.pos 的偏移 [m]，落在桌上右手可達區內（四周留約 5 cm）
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),  # (往後, 往前)
            "y": (-0.15, 0.15),  # (往右, 往左)
            "z": (0.0, 0.0),  # 保持 0，方塊才會坐在桌上
        }

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(  # 2.續取得末端執行器 TCP 的位置與姿態
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",  # [更正robot] #2.機器人的root
            # prim_path="{ENV_REGEX_NS}/Robot/openarm/follower_body_link0", #2.機器人的root
            debug_vis=False,  # 2.座標軸開啟來看一下
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",  # [更正robot] #2.換EE的點
                    # prim_path="{ENV_REGEX_NS}/Robot/openarm/follower_r_hand_tcp",
                    name="end_effector",
                ),
            ],
        )
@configclass
class OpenArmCubeLiftSelfCollisionEnvCfg(OpenArmCubeLiftEnvCfg):
    """跟 OpenArmCubeLiftEnvCfg 相同，但開啟自碰撞（階段 2：從階段 1 接著訓練）。"""
    def __post_init__(self):
        super().__post_init__()
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg)
        ).enabled_self_collisions = True
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, NewtonArticulationCfg)
        ).self_collision_enabled = True
        # 接著訓練時讓懲罰一開始就是最終值（不加這兩行，大約 417 次迭代後懲罰會突然加重）
        self.curriculum.action_rate.params["num_steps"] = 0
        self.curriculum.joint_vel.params["num_steps"] = 0

@configclass
class OpenArmCubeLift30gEnvCfg(OpenArmCubeLiftEnvCfg):
    """跟 OpenArmCubeLiftEnvCfg 相同（無自碰撞），但方塊質量改成 30 g。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.mass_props = sim_utils.MassCfg(mass=0.03)  # 方塊質量 [kg]

###########因為手腕沒有跟方塊對齊,修正後重新訓練
WRIST_ALIGNED_RIGHT_ARM = {  # IK 解：夾爪垂直朝下，閉合方向對齊方塊的面（yaw 0）
    "openarm_right_joint1": 0.538,
    "openarm_right_joint2": 1.008,
    "openarm_right_joint3": -0.94,
    "openarm_right_joint4": 1.631,
    "openarm_right_joint5": 1.204,
    "openarm_right_joint6": -0.502,
    "openarm_right_joint7": -1.274,
}
@configclass
class OpenArmCubeLiftWristAlignedEnvCfg(OpenArmCubeLiftEnvCfg):
    """手腕轉正，從頭訓練（階段 1：無自碰撞）。"""

    def __post_init__(self):
        super().__post_init__()
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(WRIST_ALIGNED_RIGHT_ARM)
        self.scene.robot.init_state = self.scene.robot.init_state.replace(joint_pos=joint_pos)
        # 方塊直接放在桌面上（平放時中心 z ≈ 0.021），不要從空中掉下來
        self.scene.object.init_state = self.scene.object.init_state.replace(pos=(-0.15, -0.25, 0.022))
        #50000步之後懲罰
        self.rewards.object_spin = RewTerm(
            func=mdp.object_spin_on_table, params={"minimal_height": 0.04}, weight=0.0
        )
        self.curriculum.object_spin = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "object_spin", "weight": -0.5, "num_steps": 50000},  
        )

        self.curriculum.action_rate.params["num_steps"] = 50000
        self.curriculum.joint_vel.params["num_steps"] = 50000

@configclass
class OpenArmCubeLiftWristAlignedSelfCollisionEnvCfg(OpenArmCubeLiftWristAlignedEnvCfg):
    """手腕轉正，並開啟自碰撞（階段 2：從階段 1 接著訓練）。"""

    def __post_init__(self):
        super().__post_init__()
        #開自體碰撞
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg)
        ).enabled_self_collisions = True
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, NewtonArticulationCfg)
        ).self_collision_enabled = True
        # 接著訓練時讓懲罰一開始就是最終值
        self.curriculum.action_rate.params["num_steps"] = 0
        self.curriculum.joint_vel.params["num_steps"] = 0
        self.curriculum.object_spin.params["num_steps"] = 0 

# 右手關節極限 [rad]（從模擬讀出的 soft limits）
OPENARM_RIGHT_ARM_LIMITS = {
    "openarm_right_joint1": (-1.396, 3.491),
    "openarm_right_joint2": (-0.175, 3.316),
    "openarm_right_joint3": (-1.571, 1.571),
    "openarm_right_joint4": (0.0, 2.443),
    "openarm_right_joint5": (-1.571, 1.571),
    "openarm_right_joint6": (-0.785, 0.785),
    "openarm_right_joint7": (-1.571, 1.571),
}


@configclass
class OpenArmCubeLiftWristAlignedClipEnvCfg(OpenArmCubeLiftWristAlignedSelfCollisionEnvCfg):
    """目標角度限制在關節極限內，並懲罰動作大小，避免頂極限出力飽和。"""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.clip = OPENARM_RIGHT_ARM_LIMITS  # target 夾在極限內 [rad]
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-2)  # 懲罰動作大小

@configclass
class OpenArmCubeLiftWristAlignedEffortEnvCfg(OpenArmCubeLiftWristAlignedClipEnvCfg):
    """懲罰出力超過極限，以及右手 link1~7 碰到任何東西（自碰撞、桌子、方塊）。"""

    def __post_init__(self):
        super().__post_init__()
        # 右手 link1~7 的接觸感測器（手指、hand 要碰方塊，不加）
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_link[1-7]", history_length=3
        )
        # pd want 超過極限的比例
        self.rewards.effort_over_limit = RewTerm(
            func=mdp.joint_effort_over_limit,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
        )
        # 接觸力 > 1 N 的 link 數
        self.rewards.arm_contact = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("arm_contact", body_names="openarm_right_link.*"), "threshold": 1.0},
        )

@configclass
class OpenArmCubeLiftWristAlignedMarginEnvCfg(OpenArmCubeLiftWristAlignedEffortEnvCfg):
    """所有關節目標角度離極限留 0.2 rad 餘量，避免慣性把關節甩出極限。"""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.clip = {
            name: (lo + 0.2, hi - 0.2) for name, (lo, hi) in OPENARM_RIGHT_ARM_LIMITS.items()
        }  # [rad]

@configclass
class OpenArmCubeLiftWristAlignedSingularEnvCfg(OpenArmCubeLiftWristAlignedMarginEnvCfg):
    """懲罰轉軸接近共線的奇異姿勢：j1-j3（j2≈90°）、j3-j5（j4≈0）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.axes_aligned = RewTerm(
            func=mdp.joint_axes_aligned,
            weight=-1.0,
            params={
                "joint_pairs": [(0, 2), (2, 4)],  # j1-j3（肩）、j3-j5（肘）
                "cos_threshold": 0.966,  # cos 15°：夾角小於 15° 開始罰
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_right_joint.*"], body_names=["openarm_right_hand"]
                ),
            },
        )

@configclass
class OpenArmCubeLiftWristAlignedShrinkWrapEnvCfg(OpenArmCubeLiftWristAlignedSingularEnvCfg):
    """身體碰撞形狀 shrink wrap，避免左肩凸出的碰撞形狀卡住左 link2 讓左手抖動。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.func = spawn_from_usd_with_shrink_wrapped_body