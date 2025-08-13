# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.envs.mdp import terminations as mdp_terminations
from isaaclab.envs.mdp.events import reset_joints_by_scale
from isaaclab.managers import (
    CurriculumTermCfg,
    TerminationTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    CommandTermCfg,
    EventTermCfg,
    SceneEntityCfg
)

# Import the local project modules
from . import observations, rewards, terminations, events

##
# MDP Commands
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    goal_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.6), pos_y=(-0.3, 0.3), pos_z=(0.25, 0.8),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


##
# MDP Rewards
##


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ee_pos_tracking_reward = RewardTermCfg(
        func=rewards.ee_pos_tracking_reward, weight=1.0, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_quat_tracking_reward = RewardTermCfg(
        func=rewards.ee_quat_tracking_reward, weight=0.5, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_stay_up_reward = RewardTermCfg(
        func=rewards.ee_stay_up_reward, weight=1.0, params={"weight": 1.0, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    joint_vel_penalty = RewardTermCfg(
        func=rewards.joint_vel_penalty, weight=-0.2, params={"weight": -0.2, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    joint_acc_penalty = RewardTermCfg(
        func=rewards.joint_acc_penalty, weight=-0.01, params={"weight": -0.01, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    ee_twist_penalty = RewardTermCfg(
        func=rewards.ee_twist_penalty, weight=-0.01, params={"weight": -0.1, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    action_smoothness_penalty = RewardTermCfg(
        func=rewards.action_smoothness_penalty, weight=-0.1, params={"weight": -0.1}
    )
    joint_limit: RewardTermCfg = RewardTermCfg(
        func=rewards.joint_limit_margin_penalty,
        weight=-0.5, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[r"panda_joint[1-7]"]), "margin": 0.05}
    )
    # One‑time success bonus when goal reached
    success_reward = RewardTermCfg(
        func=rewards.success_bonus,
        weight=200.0,
        params={"threshold": 0.05}
    )
    # Penalty on any collision event
    collision_penalty = RewardTermCfg(
        func=rewards.collision_penalty,
        weight=1.0,
        params={}
    )


##
# MDP Events (including Terminations)
##

@configclass
class EventCfg:
    """Configuration for events."""

    # Replace your prior reset sequence (and remove freeze/quiesce events).
    zero_vel_reset = EventTermCfg(
        func=events.authoritative_zero_vel_reset,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "robot_cfg": SceneEntityCfg(name="robot", joint_names=[r"panda_joint[1-7]"]),
        },
    )

    reset_robot_joints = EventTermCfg(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # place_goal_and_shelf = EventTermCfg(
    #     func=events.place_goal_and_shelf_on_reset,
    #     mode="reset",
    #     min_step_count_between_reset=0,
    #     params={
    #         # Omit params to use env.goal_path/env.shelf_path and env.*_ranges,
    #         # or pass explicit overrides like:
    #         # "goal_path": "/World/envs/env_.*/Goal",
    #         # "shelf_path": "/World/envs/env_.*/Shelf",
    #         # "goal_ranges": {...},
    #         # "shelf_ranges": {...},
    #     },
    # )


@configclass
class TerminationsCfg:
    """Event terms for the MDP, including termination conditions."""
    time_out = TerminationTermCfg(func=mdp_terminations.time_out, time_out=True)
    franka_joint_limits = TerminationTermCfg(
        func=terminations.franka_joint_limits,

        params={
            "robot_cfg": SceneEntityCfg(name="robot", joint_names=[r"panda_joint[1-7]"])
        }
    )
    franka_self_collision = TerminationTermCfg(
        func=terminations.franka_self_collision,

        params={"sensor_cfg": mdp.SceneEntityCfg("contact_sensor"), "threshold": 0.0, }
    )
    goal_reached = TerminationTermCfg(
        func=terminations.goal_reached,
        params={
            "robot_cfg": SceneEntityCfg(name="robot", body_names=["panda_hand"]),
            "position_threshold": 0.05,
            "orientation_threshold": 0.10,   # radians
        }
    )


##
# MDP Curriculum
##


@configclass
class CurriculumCfg:
    """Step‑based curricula disabled; runner will handle performance updates."""
    pass


##
# MDP Observations
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        ee_pos = ObservationTermCfg(func=observations.ee_pos, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])})
        ee_quat = ObservationTermCfg(func=observations.ee_quat, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])})
        ee_goal_pose = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP Actions
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = JointEffortActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        scale={
            "panda_joint[1-4]*": 87.0,
            "panda_joint[5-7]*": 12.0,
        },
    )
