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
    joint_pos = robot.data.joint_pos

    min_limits = FRANKA_JOINT_LIMITS_MIN.to(env.device)
    max_limits = FRANKA_JOINT_LIMITS_MAX.to(env.device)

    # Check for violations in either direction
    lower_limit_violation = torch.any(joint_pos < min_limits, dim=1)
    upper_limit_violation = torch.any(joint_pos > max_limits, dim=1)

    return torch.logical_or(lower_limit_violation, upper_limit_violation)


def franka_self_collision(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminates the episode if the Franka robot has self-collisions.

    This function checks for self-collisions by monitoring the contact forces
    on the robot's collision sensors.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Check for any contact forces on the robot's bodies
    # A more robust check might involve checking contact pairs to ensure it's a self-collision
    # but for now, any significant contact force can be a proxy.
    net_contact_forces = torch.norm(robot.data.net_contact_force_w, dim=-1)
    # The threshold can be tuned based on expected contact forces during normal operation.
    return torch.any(net_contact_forces > 1.0, dim=1)
