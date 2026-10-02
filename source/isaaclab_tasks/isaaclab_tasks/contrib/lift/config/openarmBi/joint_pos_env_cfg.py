# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.lift import mdp
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg 

from isaaclab_assets.robots.openarm import OPENARM_BI_CFG #1.換雙臂

from isaaclab_newton.sim.schemas import NewtonArticulationCfg # 9.開啟自碰撞
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg #9. 開啟自碰撞
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# The OpenArm is smaller than the Franka, so the goal workspace and the joint-scoped
# terms are re-tuned for it relative to the shared lift base.
OPENARM_LEFT_JOINTS = [
    "openarm_left_joint[1-7]",
    "openarm_left_finger_joint.*",
]

OPENARM_RIGHT_JOINTS = [
    "openarm_right_joint[1-7]",
    "openarm_right_finger_joint.*",
]

@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set OpenArm as robot
        self.scene.robot = OPENARM_BI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") #2.換機器人
        #9. 開啟自碰撞
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg)
        ).enabled_self_collisions = True
        next(
            frag for frag in self.scene.robot.spawn.articulation_props if isinstance(frag, NewtonArticulationCfg)
        ).self_collision_enabled = True
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(-0.40, 0.0, -0.25),  # 5-1.機器人向下25cm 後40cm
            joint_pos={ #5-4. 改初始姿勢以免手肘撞到
                #5-4 右手：手肘彎、手在桌面上方、夾爪朝下
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
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_RIGHT_JOINTS)}#1.單獨右手mdp
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_RIGHT_JOINTS)}#1.單獨右手mdp
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=OPENARM_RIGHT_JOINTS)#1.單獨右手mdp

        # Shrink the goal workspace to the smaller arm's reach. #5-3.目標會跟著機器人移動,目標位置往桌角
        self.commands.object_pose.ranges.pos_x = (0.20, 0.30) #(0.2, 0.4) => (0.20, 0.30) 
        self.commands.object_pose.ranges.pos_y = (-0.4,-0.2) #(-0.2, 0.2) => (-0.4,-0.2) 
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45) #(0.15, 0.4) ＝> (0.35, 0.45) #>0.04 ok

        # Re-tuned reach weight, and no success tracking on this task.
        self.rewards.reaching_object.weight = 1.1
        #del self.rewards.object_goal_tracking.params["success_threshold"] #紀錄success_threshold

        # Set actions for the specific robot type (OpenArm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint.*",#2.手臂改右手
            ],
            scale=0.5,
            use_default_offset=True,
        )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],#2.抓取改右手
            open_command_expr={"openarm_right_finger_joint.*": 0.044},#2.抓取改右手
            close_command_expr={"openarm_right_finger_joint.*": 0.0},#2.抓取改右手
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "openarm_right_hand" #2.改右手（這是common ee）
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.object_pose.debug_vis = True  # 8.顯示目標位置
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, -0.25 , 0.055], rot=[0, 0, 0, 1]), #5-2.Cube往桌角 0.4, 0.0 , 0.055 => -0.15, -0.25 , 0.055 -0.1 , 0.055  , 0.055
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
        self.scene.ee_frame = FrameTransformerCfg( #2.續取得末端執行器 TCP 的位置與姿態
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link", #2.機器人的root
            debug_vis=False, #2.座標軸開啟來看一下
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp", #2.換EE的點
                    name="end_effector",
                ),
            ],
        )
