# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # , RigidObjectCfg # for shelf
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from . import mdp

##
# Pre-defined configs
##

from .franka_cfg import FRANKA_PANDA_CFG


##
# Scene definition
##


@configclass
class IsaacLearningSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG

    # Contact sensor for collision detection TODO(mrahme): is there a better way?
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link(1|2|3|4|5|6|7|hand)",
        force_threshold=1.0
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Shelf TODO(mrahme): enable and randomize
    # shelf = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Shelf",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.5, 0.4, 0.25),
    #         visual_material=materials.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    #         rigid_props=schemas.RigidBodyPropertiesCfg(),
    #         collision_props=schemas.CollisionPropertiesCfg(collision_enabled=True),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    # )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(-0.5, 0.5), pitch=(-0.5, 0.5), yaw=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointEffortActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        target_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # Reaching reward with lower weight
    reaching_target = RewTerm(func=mdp.pose_err, params={"std": 0.05}, weight=2)

    # Action penalty to encourage smooth movements
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # Joint velocity penalty to prevent erratic movements
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Collision
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )

# @configclass


class CurriculumCfg:
    #     """Curriculum terms for the MDP."""

    #     # Stage 1: Focus on reaching
    #     # Start with higher reaching reward, then gradually decrease it
    #     reaching_reward = CurrTerm(
    #         func=mdp.modify_reward_weight,
    #         params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 6000}
    #     )

    #     # Stage 2: Transition to lifting
    #     # Start with lower lifting reward, gradually increase to encourage lifting behavior
    #     lifting_reward = CurrTerm(
    #         func=mdp.modify_reward_weight,
    #         params={"term_name": "lifting_object", "weight": 35.0, "num_steps": 8000}
    #     )

    # Stage 4: Stabilize the policy
    # Gradually increase action penalties to encourage smoother, more stable movements
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    )


##
# Environment configuration
##


@configclass
class IsaacLearningEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: IsaacLearningSceneCfg = IsaacLearningSceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# TODO(mrahme): see https://github.com/MuammerBay/IsaacLab-SO_100/blob/541b84d4c1a41826d3a78c25dfdc5507cbdf3791/source/SO_100/SO_100/tasks/manager_based/so_100/so_100_base_env_cfg.py#L89 for:
# CommandsCfg: targets/goals
# ActionsCfg: actions (e.g. torque or vel)
# ObservationsCfg: measurements
# EventCfg: e.g. when to reset
# RewardsCfg: rewards for policy
# CurriculumCfg: curriculum
# EnvCfg: the final thing that puts everything else together
