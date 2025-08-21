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
from . import observations, rewards, terminations, events, curricula

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

# TODO(mrahme): put somewhere

"""
        # sample & set goal
        gx, gy, gz = [self._np_random.uniform(*self.goal_ranges[k]) for k in ("x", "y", "z")]
        gr = random.choice(self.goal_ranges["rpy"])
        self.scene.set_world_pose(self.goal_path,
                                    position=(gx, gy, gz),
                                    orientation=quat_from_euler_xyz(*gr),
                                    env_index=i)
        # sample & set shelf
        sx, sy, sz = [self._np_random.uniform(*self.shelf_ranges[k]) for k in ("x", "y", "z")]
        sr = random.choice(self.shelf_ranges["rpy"])
        self.scene.set_world_pose(self.shelf_path,
                                    position=(sx, sy, sz),
                                    orientation=quat_from_euler_xyz(*sr),
                                    env_index=i)

        robot = env.scene["robot"]
        # 1) root pose (offset by env origins if you use them)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins   # if you use InteractiveScene origins
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        # 2) zero root velocity
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)
        # 3) joint state with zero dq
        q = robot.data.default_joint_pos.clone()
        dq = torch.zeros_like(robot.data.default_joint_vel)
        robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
        # 4) finalize reset (clears internal buffers)
        robot.reset(env_ids=env_ids)

        def _get_extras(self):
            extras = super()._get_extras()
            # log individual reward terms
            for name, val in self.reward_manager.component_reward_terms.items():
                extras[f"Rewards/{name}"] = val.mean()
            # compute success & distance
            robot = self.scene.robots[0]
            idx = robot.body_names.index("panda_hand")
            ee = robot.data.body_pos_w[:, idx]
            tgt = self.scene.ee_goal.data.target_pos_w[:, 0, :]
            dist = torch.norm(ee - tgt, dim=1)
            extras["is_success"] = (dist < self.success_thr).float()
            extras["dist_to_target"] = dist
            return extras

        self.goal_ranges = self.cfg.task.goal_pose_range
        self.shelf_ranges = self.cfg.task.shelf_pose_range
"""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ee_pos_tracking_reward = RewardTermCfg(
        func=rewards.ee_pos_tracking_reward, weight=2.5, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_quat_tracking_reward = RewardTermCfg(
        func=rewards.ee_quat_tracking_reward, weight=0.5, params={"robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    ee_stay_up_reward = RewardTermCfg(
        func=rewards.ee_stay_up_reward, weight=1.0, params={"weight": 1.0, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    joint_vel_penalty = RewardTermCfg(
        func=rewards.joint_vel_penalty, weight=0.5, params={"weight": 0.5, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    joint_acc_penalty = RewardTermCfg(
        func=rewards.joint_acc_penalty, weight=1.0e-6, params={"weight": 1.0e-6, "robot_cfg": mdp.SceneEntityCfg("robot")}
    )
    ee_twist_penalty = RewardTermCfg(
        func=rewards.ee_twist_penalty, weight=0.5, params={"weight": 0.5, "robot_cfg": mdp.SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    action_smoothness_penalty = RewardTermCfg(
        func=rewards.action_smoothness_penalty, weight=0.1, params={"weight": 0.1}
    )
    joint_torque_penalty: RewardTermCfg = RewardTermCfg(
        func=rewards.joint_torque_limit_penalty,
        weight=1.0e-2, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[r"panda_joint[1-7]"]), "margin": 0.05}
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
        weight=100.0,
        params={"sensor_cfg": mdp.SceneEntityCfg("contact_sensor"), "threshold": 0.1, "weight": 100.0}
    )

    # self.k_dist_reward = 2.5
    # self.k_joint_limit_penalty = 5.0
    # self.k_collision_penalty = 20.0
    # self.action_penalty_curriculum = CurriculumConfig(
    #     start_value=1.0e-4, end_value=1.0e-3, start_metric_val=0.0, end_metric_val=0.2)
    # self.accel_penalty_curriculum = CurriculumConfig(
    #     start_value=0.0, end_value=1.0e-6, start_metric_val=0.4, end_metric_val=0.6)
    # self.jerk_penalty_curriculum = CurriculumConfig(
    #     start_value=0.0, end_value=1.0e-12, start_metric_val=0.7, end_metric_val=0.8)
    # self.joint_velocity_penalty_curriculum = CurriculumConfig(
    #     start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95)
    # self.ee_velocity_penalty_curriculum = CurriculumConfig(
    #     start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95)
    # self.upright_bonus_curriculum = CurriculumConfig(
    #     start_value=1.0, end_value=0.0, start_metric_val=0.0, end_metric_val=0.4)
    # self.threshold_curriculum = CurriculumConfig(
    #     start_value=0.05, end_value=0.005, start_metric_val=0.5, end_metric_val=0.84)


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
        func=terminations.franka_self_collision, time_out=False,
        params={"sensor_cfg": mdp.SceneEntityCfg("contact_sensor"), "threshold": 0.0, }
    )
    success = TerminationTermCfg(
        func=terminations.goal_reached, time_out=False,
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
    """Success-rate curricula mapped to your rewards/terminations."""

    # Tighten goal success threshold (termination)
    tighten_goal_pos_threshold = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.success.params.position_threshold",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 0.05, "end_value": 0.005,
                "start_success": 0.50, "end_success": 0.84,
            },
        },
    )

    # Keep success reward threshold in sync
    tighten_success_reward_threshold = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.success_reward.params.threshold",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 0.05, "end_value": 0.005,
                "start_success": 0.50, "end_success": 0.84,
            },
        },
    )

    # Action smoothness weight: 1e-4 → 1e-3 as success rises 0.0 → 0.2
    action_smoothness_weight = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_smoothness_penalty.params.weight",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 1.0e-4, "end_value": 1.0e-3,
                "start_success": 0.0, "end_success": 0.2,
            },
        },
    )

    # Joint acceleration weight: 0 → 1e-6 as success 0.4 → 0.6
    joint_acc_weight = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.joint_acc_penalty.params.weight",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 0.0, "end_value": 1.0e-6,
                "start_success": 0.4, "end_success": 0.6,
            },
        },
    )

    # Joint velocity weight: 0 → 0.5 as success 0.85 → 0.95
    joint_vel_weight = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.joint_vel_penalty.params.weight",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 0.0, "end_value": 0.5,
                "start_success": 0.85, "end_success": 0.95,
            },
        },
    )

    # EE twist weight: 0 → 0.5 as success 0.85 → 0.95
    ee_twist_weight = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_twist_penalty.params.weight",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 0.0, "end_value": 0.5,
                "start_success": 0.85, "end_success": 0.95,
            },
        },
    )

    # Upright bonus weight: 1.0 → 0.0 as success 0.0 → 0.4
    upright_bonus_weight = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_stay_up_reward.params.weight",
            "modify_fn": curricula.success_linear,
            "modify_params": {
                "start_value": 1.0, "end_value": 0.0,
                "start_success": 0.0, "end_success": 0.4,
            },
        },
    )

    # Optional: disable collision penalty at high success with hysteresis
    collision_penalty_toggle = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.collision_penalty.params.weight",
            "modify_fn": curricula.success_toggle,
            "modify_params": {
                "on_value": 100.0, "off_value": 0.0,
                "threshold": 0.90, "hysteresis": 0.02,
                "state_key": "_coll_penalty_on",
            },
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

    arm_action = JointEffortActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        scale={
            "panda_joint[1-4]*": 87.0,
            "panda_joint[5-7]*": 12.0,
        },
    )
