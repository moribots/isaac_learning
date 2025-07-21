# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.envs.mdp import terminations as mdp_terminations
from isaaclab.managers import (
    CurriculumTermCfg,
    TerminationTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    CommandTermCfg
)

# Import the local project modules
from . import curriculums as curr_terms
from . import observations, rewards, terminations

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
        func=rewards.ee_pos_tracking_reward, weight=1.0, params={"std": 0.1, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_quat_tracking_reward = RewardTermCfg(
        func=rewards.ee_quat_tracking_reward, weight=0.5, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_stay_up_reward = RewardTermCfg(
        func=rewards.ee_stay_up_reward, weight=1.0, params={"weight": 1.0, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    joint_vel_penalty = RewardTermCfg(
        func=rewards.joint_vel_penalty, weight=0.0, params={"weight": 0.0, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    joint_acc_penalty = RewardTermCfg(
        func=rewards.joint_acc_penalty, weight=0.0, params={"weight": 0.0, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    ee_twist_penalty = RewardTermCfg(
        func=rewards.ee_twist_penalty, weight=0.0, params={"weight": 0.0, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    action_smoothness_penalty = RewardTermCfg(
        func=rewards.action_smoothness_penalty, weight=0.0, params={"weight": 0.0}
    )


##
# MDP Events (including Terminations)
##


@configclass
class TerminationsCfg:
    """Event terms for the MDP, including termination conditions."""
    time_out = TerminationTermCfg(func=mdp_terminations.time_out, time_out=True)
    franka_joint_limits = TerminationTermCfg(
        func=terminations.franka_joint_limits,

        params={"robot_cfg": mdp.SceneEntityCfg("robot", joint_names=".*")}
    )
    franka_self_collision = TerminationTermCfg(
        func=terminations.franka_self_collision,

        params={"robot_cfg": mdp.SceneEntityCfg("contact_sensor")}
    )


##
# MDP Curriculum
##


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    ee_pos_reward_schedule = CurriculumTermCfg(
        func=curr_terms.linear_interpolation,
        params={
            "address": "rewards.ee_pos_tracking.weight",
            "start_value": 0.1,
            "end_value": 1.0,
            "start_step": 0,
            "end_step": 25000,
        },
    )
    ee_quat_reward_schedule = CurriculumTermCfg(
        func=curr_terms.linear_interpolation,
        params={
            "address": "rewards.ee_quat_tracking.weight",
            "start_value": 0.05,
            "end_value": 0.5,
            "start_step": 0,
            "end_step": 25000,
        },
    )
    action_smoothness_penalty_schedule = CurriculumTermCfg(
        func=curr_terms.linear_interpolation,
        params={
            "address": "rewards.action_smoothness_penalty.weight",
            "start_value": 0.0,
            "end_value": -0.01,
            "start_step": 0,
            "end_step": 40000,
        },
    )


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

    arm_action = JointEffortActionCfg(asset_name="robot", joint_names=["panda_joint[1-7]"], scale=100.0)
