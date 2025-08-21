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
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab.envs.mdp.terminations import joint_vel_out_of_limit

from isaaclab.utils.math import quat_inv, quat_mul
from ..config.franka.franka_cfg import FRANKA_JOINT_TORQUE_LIMITS_MIN, FRANKA_JOINT_TORQUE_LIMITS_MAX


def ee_pos_tracking_reward(
    env: ManagerBasedRLEnv,
    robot_cfg,            # SceneEntityCfg for the robot
    weight: float = 1.0,  # scaling factor
) -> torch.Tensor:
    """
    Continuous position‐tracking reward:
      reward = weight * ||ee_pos - goal_pos||_2
    where goal_pos is fetched from the command manager under "goal_pose".
    """
    # 1. Lookup the robot articulation by its config name ("robot")
    robot = env.scene[robot_cfg.name]

    # 2. Extract end‐effector world position (num_envs, 3)
    ee_index = robot.body_names.index("panda_hand")
    ee_pos = robot.data.body_pos_w[:, ee_index]  # (N,3)

    # 3. Fetch the goal pose command (num_envs, 7): [x,y,z, qx,qy,qz,qw]
    goal_cmd = env.command_manager.get_command("goal_pose")
    goal_pos = goal_cmd[..., :3]                   # (N,3)

    # 4. Compute Euclidean error and scale
    error = torch.norm(ee_pos - goal_pos, dim=1)  # (N,)
    reward = weight * error                       # (N,)

    return reward


def ee_quat_tracking_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward for aligning the end‑effector orientation with the goal orientation.
    Fetches the goal quaternion from the command manager under "goal_pose"
    instead of looking for a nonexistent ee_goal scene entity.
    Returns a tensor of shape (num_envs,).
    """
    # 1. Lookup Articulation
    robot = env.scene[robot_cfg.name]
    ee_index = robot.body_names.index("panda_hand")
    # 2. Current EE quaternion (world frame), shape (num_envs, 4)
    ee_quat_w = robot.data.body_quat_w[:, ee_index, :]
    # 3. Goal quaternion from command manager, shape (num_envs, 7) → slice to (num_envs, 4)
    goal_cmd = env.command_manager.get_command("goal_pose")
    target_quat_w = goal_cmd[..., 3:]  # (num_envs, 4)
    # 4. Quaternion alignment reward: 2*(dot(q, q_goal)^2) - 1
    dot_product = torch.sum(ee_quat_w * target_quat_w, dim=1).abs()
    return 2.0 * (dot_product ** 2) - 1.0


def ee_stay_up_reward(
    env: ManagerBasedRLEnv,
    weight: float,
    robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward proportional to the Z‐coordinate of the end‑effector.
    Encourages the arm to stay raised against gravity.
    """
    robot = env.scene[robot_cfg.name]
    ee_index = robot.body_names.index("panda_hand")
    ee_pos_z = robot.data.body_pos_w[:, ee_index, 2]  # (num_envs,)
    return weight * ee_pos_z


def joint_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.05,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Penalize joints as measured torque approaches the Franka limits.

    Args:
        env: Manager-based RL env.
        asset_cfg: Scene entity for the robot (uses asset_cfg.joint_ids).
        margin: Fractional margin. Penalty starts when |tau|/limit > 1 - margin.
        power: Exponent of the near-limit hinge (2.0 quadratic).

    Returns:
        Tensor [num_envs] of negative penalties.
    """
    # Live handles
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device

    # Select measured torques for controlled joints
    # applied_torque: what was actually applied by sim this step
    # computed_torque: controller-computed before clamping
    tau_appl = asset.data.applied_torque[:, asset_cfg.joint_ids]
    tau_comp = asset.data.computed_torque[:, asset_cfg.joint_ids]
    # Conservative: use the larger magnitude per joint
    tau = torch.where(tau_appl.abs() >= tau_comp.abs(), tau_appl, tau_comp)

    # Broadcast Franka limits to device and select joints
    lim_hi = FRANKA_JOINT_TORQUE_LIMITS_MAX.to(device)[asset_cfg.joint_ids]  # [dof_sel]
    lim_lo = FRANKA_JOINT_TORQUE_LIMITS_MIN.to(device)[asset_cfg.joint_ids]
    lim_sym = torch.max(lim_hi.abs(), lim_lo.abs())  # [dof_sel], symmetric magnitude

    # Utilization and hinge near the limit
    util = tau.abs() / (lim_sym + 1e-6)               # [N, dof_sel]
    thresh = 1.0 - float(margin)
    over = torch.clamp(util - thresh, min=0.0)
    pen = over.pow(power)                              # stronger near the limit

    return -pen.sum(dim=-1)


def joint_acc_penalty(env: ManagerBasedRLEnv, weight: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes high joint accelerations.

    This function encourages smoother joint movements by applying a penalty
    proportional to the squared joint accelerations.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return -weight * torch.sum(torch.square(robot.data.joint_acc), dim=1)


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    weight: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-env penalty: −weight where any joint velocity exceeds limit, else 0. Shape [N]."""
    mask = joint_vel_out_of_limit(env, robot_cfg)  # [N] bool on device
    return (-float(weight)) * mask.to(dtype=torch.float32)


def ee_twist_penalty(env: ManagerBasedRLEnv, weight: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes the end-effector's linear and angular velocity (twist).

    This function encourages the end-effector to move smoothly and avoid
    abrupt changes in velocity.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot_cfg.body_ids[0]
    ee_twist = robot.data.body_vel_w[:, ee_body_idx]
    return -weight * torch.sum(torch.square(ee_twist), dim=1)


def action_smoothness_penalty(env: ManagerBasedRLEnv, weight: float) -> torch.Tensor:
    """Penalizes large changes in actions between consecutive time steps.

    This function promotes smoother control policies by penalizing abrupt
    changes in the actions.
    """
    actions = env.action_manager.action
    last_actions = env.action_manager.prev_action
    return -weight * torch.sum(torch.square(actions - last_actions), dim=1)


def success_bonus(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """
    Returns 1.0 for each environment where the end‑effector is within
    `threshold` meters of the current goal position, else 0.0.

    Fetches the goal position from the command manager under "goal_pose",
    matching IsaacLab 2.2’s pattern for all goal‑driven terms.
    """
    # 1) Lookup the robot articulation
    robot = env.scene["robot"]                    # single Articulation
    ee_idx = robot.body_names.index("panda_hand")  # index of the end‑effector body
    ee_pos = robot.data.body_pos_w[:, ee_idx]     # (num_envs, 3)

    # 2) Fetch the goal pose command (UniformPoseCommandCfg)
    #    Returns a tensor shape (num_envs, 7): [x, y, z, qx, qy, qz, qw]
    goal_cmd = env.command_manager.get_command("goal_pose")
    goal_pos = goal_cmd[..., :3]                  # (num_envs, 3)

    # 3) Compute distance and apply threshold
    dist = torch.norm(ee_pos - goal_pos, dim=1)   # (num_envs,)
    return (dist < threshold).float()             # 1.0 where dist < threshold


def collision_penalty(
    env: ManagerBasedRLEnv,
    weight: float,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-env penalty: −weight where any illegal contact is present, else 0. Shape [N]."""
    mask = illegal_contact(env, threshold, sensor_cfg)  # [N] bool on device
    return (-float(weight)) * mask.to(dtype=torch.float32)
