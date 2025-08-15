# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides functions for determining the termination conditions of an episode.
"""

from __future__ import annotations

import torch


from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.terminations import joint_pos_out_of_limit, joint_vel_out_of_limit, joint_effort_out_of_limit, illegal_contact

from ..config.franka.franka_cfg import *

PRINT = False


def franka_joint_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Combined termination: position OR velocity OR torque limit violated."""
    pos = joint_pos_out_of_limit(env, robot_cfg)
    vel = False  # joint_vel_out_of_limit(env, robot_cfg)
    tau = joint_effort_out_of_limit(env, robot_cfg)
    return pos | vel | tau


def franka_self_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,    # your ContactSensor cfg
    threshold: float = 0.0         # any nonzero force → collision
) -> torch.Tensor:
    """
    Per-env self-collision termination: returns True wherever
    the contact sensor reports any force above `threshold`.
    """

    return illegal_contact(env, threshold, sensor_cfg)


def goal_reached(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    position_threshold: float,
    orientation_threshold: float
) -> torch.Tensor:
    """
    Terminate when both:
      • End-effector within `position_threshold` meters of the goal position.
      • And its orientation within `orientation_threshold` radians of the goal orientation.
    """

    # 1) Look up the robot articulation by its config name ("robot")
    robot = env.scene[robot_cfg.name]

    # 2) Extract current EE pose from the Articulation’s data buffers
    ee_idx = robot.body_names.index("panda_hand")
    ee_pos = robot.data.body_pos_w[:, ee_idx]      # (num_envs, 3)
    ee_quat = robot.data.body_quat_w[:, ee_idx, :]  # (num_envs, 4)

    # 3) Fetch the goal pose from the command manager (UniformPoseCommandCfg)
    #    get_command("goal_pose") returns a tensor of shape (num_envs, 7):
    #       [x, y, z, qx, qy, qz, qw]
    goal_cmd = env.command_manager.get_command("goal_pose")
    goal_pos = goal_cmd[..., :3]    # (num_envs, 3)
    goal_quat = goal_cmd[..., 3:]    # (num_envs, 4)

    # 4) Position check
    pos_err = torch.norm(ee_pos - goal_pos, dim=1)    # (num_envs,)
    pos_ok = pos_err < position_threshold            # BoolTensor

    # 5) Orientation check via quaternion dot → angle
    dot = torch.abs((ee_quat * goal_quat).sum(dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    angle_err = 2.0 * torch.acos(dot)                   # (num_envs,)
    ori_ok = angle_err < orientation_threshold       # BoolTensor

    # 6) Both must be true
    return pos_ok & ori_ok
