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

from ..config.franka.franka_cfg import FRANKA_JOINT_LIMITS_MAX, FRANKA_JOINT_LIMITS_MIN


def franka_joint_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminates the episode if the Franka robot's joint positions exceed their limits.

    This function checks if any of the robot's joints have moved beyond the predefined
    minimum and maximum limits. Exceeding these limits is considered a failure condition.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos_full = robot.data.joint_pos       # shape: (num_envs, 9)
    joint_pos = joint_pos_full[:, -7:]     # shape: (num_envs, 7)

    min_limits = FRANKA_JOINT_LIMITS_MIN.to(env.device)
    max_limits = FRANKA_JOINT_LIMITS_MAX.to(env.device)

    # Check for violations in either direction
    lower_limit_violation = torch.any(joint_pos < min_limits, dim=1)
    upper_limit_violation = torch.any(joint_pos > max_limits, dim=1)

    return torch.logical_or(lower_limit_violation, upper_limit_violation)


def franka_self_collision(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Terminate an episode as soon as any self‑collision force is registered.
    Uses the ContactSensor’s net_forces_w buffer.

    Returns:
        torch.BoolTensor of shape (num_envs,), where True indicates a collision.
    """
    # Retrieve the ContactSensor instance by its config name
    sensor = env.scene.sensors["contact_sensor"]
    # net_forces_w has shape (num_envs, num_bodies, 3)
    net_forces = sensor.data.net_forces_w
    # Compute per-body force magnitudes: (num_envs, num_bodies)
    magnitudes = torch.norm(net_forces, dim=-1)
    # A collision occurred in an environment if any body force > 0
    collided = magnitudes > 0.0
    # Return a per-env mask
    return collided.any(dim=-1)


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
