# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module defines the reward functions for the reach task.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_inv, quat_mul


def ee_pos_tracking_reward(env: ManagerBasedRLEnv, std: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculates the reward for end-effector position tracking.

    This reward function encourages the robot's end-effector to stay close to the
    target position. The reward is based on an exponential kernel, where the
    distance between the current and target positions is penalized.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_goal = env.scene["ee_goal"]
    ee_body_idx = robot_cfg.body_ids[0]

    # Get current and target end-effector positions
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    target_pos_w = ee_goal.data.target_pos_w[:, 0, :]

    # Calculate the squared distance between current and target positions
    pos_dist_sq = torch.sum((ee_pos_w - target_pos_w) ** 2, dim=1)

    # Reward is based on an exponential kernel of the distance
    return torch.exp(-pos_dist_sq / (std**2))


def ee_quat_tracking_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculates the reward for end-effector orientation tracking.

    This function rewards the robot for aligning its end-effector orientation
    with the target orientation.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_goal = env.scene["ee_goal"]
    ee_body_idx = robot_cfg.body_ids[0]

    # Get current and target end-effector orientations
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    target_quat_w = ee_goal.data.target_quat_w[:, 0, :]

    # Compute the orientation error
    quat_error = quat_mul(target_quat_w, quat_inv(ee_quat_w))
    # The error is the angle part of the quaternion, which is related to the x,y,z components
    # A dot product formulation is common here: (q1 . q2)^2
    dot_product = torch.sum(ee_quat_w * target_quat_w, dim=1)
    return 2 * (dot_product**2) - 1


def ee_stay_up_reward(env: ManagerBasedRLEnv, weight: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Rewards the robot for keeping its end-effector up.

    This is particularly useful in torque control scenarios to counteract gravity.
    The reward is proportional to the Z-coordinate of the end-effector.

    Args:
        env: The environment instance.
        weight: The weight of the reward.
        robot_cfg: The configuration object for the robot.

    Returns:
        A tensor of shape (num_envs,) containing the stay-up reward.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_id = robot_cfg.body_ids[0]
    ee_pos_z = robot.data.body_pos_w[:, ee_body_id, 2]
    return weight * ee_pos_z


def joint_acc_penalty(env: ManagerBasedRLEnv, weight: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes high joint accelerations.

    This function encourages smoother joint movements by applying a penalty
    proportional to the squared joint accelerations.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return weight * torch.sum(torch.square(robot.data.joint_acc), dim=1)


def joint_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalizes large joint velocities.

    This function discourages high-speed joint movements to promote smoother,
    more controlled motions.
    """
    return torch.sum(torch.square(env.scene["robot"].data.joint_vel), dim=1)


def ee_twist_penalty(env: ManagerBasedRLEnv, weight: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes the end-effector's linear and angular velocity (twist).

    This function encourages the end-effector to move smoothly and avoid
    abrupt changes in velocity.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot_cfg.body_ids[0]
    ee_twist = robot.data.body_vel_w[:, ee_body_idx]
    return weight * torch.sum(torch.square(ee_twist), dim=1)


def action_smoothness_penalty(env: ManagerBasedRLEnv, weight: float) -> torch.Tensor:
    """Penalizes large changes in actions between consecutive time steps.

    This function promotes smoother control policies by penalizing abrupt
    changes in the actions.
    """
    actions = env.action_manager.action
    last_actions = env.action_manager.prev_action
    return weight * torch.sum(torch.square(actions - last_actions), dim=1)
