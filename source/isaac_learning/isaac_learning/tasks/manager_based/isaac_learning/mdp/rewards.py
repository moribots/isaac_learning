# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Reward terms for IsaacLab reach task.

Implements:
- Position tracking: negative L2 distance.
- Orientation reward: quaternion alignment reward in [0, 1].
- Stay-up bonus: positive when end-effector z is high (unchanged).
- Joint velocity penalty: -||dq|| with proximity scaling to goal.
- Joint velocity limit exceed penalty: penalize ReLU(|dq|-dq_lim).
- Joint acceleration penalty: -||ddq||.
- EE twist penalty: -(||v_lin|| + ||v_ang||).
- Action smoothness penalty: -||a_t - a_{t-1}||.
- Joint torque limit penalty: unchanged passthrough.
- Success bonus and collision penalty: unchanged passthrough.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.terminations import illegal_contact


# -------------------------
# Helper utilities
# -------------------------

def _get_robot(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    return env.scene[robot_cfg.name]


def _ee_body_index(robot, body_name: str) -> int:
    # Robust body index lookup
    try:
        return robot.body_names.index(body_name)
    except ValueError:
        # fallback: assume last link is the tool
        return len(robot.body_names) - 1


def _goal_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd = env.command_manager.get_command("goal_pose")  # shape: (N, 7)
    return cmd[..., :3]  # xyz


# -------------------------
# Primary rewards
# -------------------------

def ee_pos_tracking_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
) -> torch.Tensor:
    """
    Negative L2 distance between end-effector and goal.

    Returns: shape (num_envs,)
    """
    robot = _get_robot(env, robot_cfg)
    idx = _ee_body_index(robot, robot_cfg.body_names[0])
    ee_pos = robot.data.body_pos_w[:, idx]  # (N,3)
    tgt_pos = _goal_pos(env)                # (N,3)
    dist = torch.norm(ee_pos - tgt_pos, dim=1)
    return -weight * dist


def ee_quat_tracking_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
) -> torch.Tensor:
    """
    Quaternion alignment reward in [0, 1], 1 at perfect alignment.

    r = (dot(q_ee, q_goal))^2
    """
    robot = _get_robot(env, robot_cfg)
    idx = _ee_body_index(robot, robot_cfg.body_names[0])
    q_ee = robot.data.body_quat_w[:, idx]          # (N,4) wxyz
    cmd = env.command_manager.get_command("goal_pose")  # (N, 7)
    q_goal = cmd[..., 3:]  # quaternion from the command
    # If your command emits [x,y,z,w] instead of [w,x,y,z], reorder once:
    # q_goal = q_goal[:, [3, 0, 1, 2]]
    # Normalize for safety
    q_ee = torch.nn.functional.normalize(q_ee, dim=1)
    q_goal = torch.nn.functional.normalize(q_goal, dim=1)
    # Ambiguity q ~ -q handled by square
    dot = torch.sum(q_ee * q_goal, dim=1).abs()
    return -weight * (dot * dot)


def ee_stay_up_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
    upright_threshold: float = 0.10,
) -> torch.Tensor:
    """
    Upright bonus: binary indicator on EE height.
    Returns 1.0 if ee_z > upright_threshold else 0.0.
    Magnitude set via the reward weight in cfg.
    """
    robot = _get_robot(env, robot_cfg)
    idx = _ee_body_index(robot, robot_cfg.body_names[0])
    z = robot.data.body_pos_w[:, idx, 2]
    return (z > upright_threshold).float() * weight


# -------------------------
# Penalties
# -------------------------


# rewards.py  — replace joint_pos_barrier_penalty with:

def joint_pos_barrier_penalty(env, robot_cfg, margin: float, weight: float):
    robot = env.scene[robot_cfg.name]
    device = robot.data.joint_pos.device

    # Resolve 7-DOF indices robustly
    if getattr(robot_cfg, "joint_ids", None) is None or len(robot_cfg.joint_ids) == 0:
        arm_names = [f"panda_joint{i}" for i in range(1, 8)]
        joint_ids = torch.tensor([robot.joint_names.index(n) for n in arm_names],
                                 device=device, dtype=torch.long)
    else:
        joint_ids = torch.as_tensor(robot_cfg.joint_ids, device=device, dtype=torch.long)

    # Current positions: (N, 7)
    q = robot.data.joint_pos[:, joint_ids]

    # Limits: support (DoF,2) or (N,DoF,2)
    jlim = getattr(robot.data, "joint_pos_limits", None)
    if jlim is not None:
        if jlim.ndim == 2:             # (DoF, 2)
            lo = jlim[joint_ids, 0].unsqueeze(0).expand_as(q)
            hi = jlim[joint_ids, 1].unsqueeze(0).expand_as(q)
        else:                           # (N, DoF, 2)
            lo = jlim[:, joint_ids, 0]
            hi = jlim[:, joint_ids, 1]
    else:
        # Fallback to constants
        from ..config.franka.franka_cfg import (
            FRANKA_JOINT_POSITION_LIMITS_MIN as LMIN,
            FRANKA_JOINT_POSITION_LIMITS_MAX as LMAX,
        )
        lo = LMIN.to(device)[joint_ids].unsqueeze(0).expand_as(q)
        hi = LMAX.to(device)[joint_ids].unsqueeze(0).expand_as(q)

    d_lo = q - lo
    d_hi = hi - q
    pen = torch.relu(margin - d_lo) + torch.relu(margin - d_hi)   # (N,7)
    return -weight * pen.sum(dim=1)                                         # unsigned magnitude


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
    dist_threshold: float = 0.2,
    max_scale: float = 4.0,
) -> torch.Tensor:
    """
    Minimizing joint velocity magnitude penalty with proximity scaling.

    base = ||dq||_2
    scale(dist) = 1                      if dist >= dist_threshold
                = 1 + (max_scale-1)*(1 - dist/dist_threshold)   otherwise
    penalty = - base * scale(dist)
    """
    robot = _get_robot(env, robot_cfg)
    dq = robot.data.joint_vel  # (N,DoF)
    base = torch.norm(dq, dim=1)

    # Proximity scaling based on EE distance to goal
    idx = _ee_body_index(robot, robot_cfg.body_names[0]) if robot_cfg.body_names else 0
    ee = robot.data.body_pos_w[:, idx]
    tgt = _goal_pos(env)
    dist = torch.norm(ee - tgt, dim=1)

    scale = torch.ones_like(dist)
    mask = dist < dist_threshold
    if mask.any():
        # linear ramp up to max_scale as we approach the target
        scale_in = 1.0 + (max_scale - 1.0) * (1.0 - dist[mask] / dist_threshold)
        scale = scale.clone()
        scale[mask] = scale_in

    return -weight * base * scale


def joint_vel_limit_exceed_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
) -> torch.Tensor:
    """
    Penalize only when joint velocity exceeds per-joint limits:
        - sum_j ReLU(|dq_j| - dq_lim_j)

    Uses soft_joint_vel_limits if available on articulation data.
    """
    robot = _get_robot(env, robot_cfg)
    dq = robot.data.joint_vel.abs()
    # Try soft limits first (populated by articulation + actuator cfg)
    dq_lim = getattr(robot.data, "soft_joint_vel_limits", None)
    if dq_lim is None:
        # fallback: try joint_velocity_limits, else no penalty
        dq_lim = getattr(robot.data, "joint_velocity_limits", None)
        if dq_lim is None:
            return torch.zeros(dq.shape[0], device=dq.device)
    over = torch.nn.functional.relu(dq - dq_lim)
    return -weight * over.sum(dim=1)


def joint_acc_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
) -> torch.Tensor:
    """
    Penalize joint acceleration magnitude: -||ddq||_2.

    Relies on articulation finite differencing inside simulation buffers.
    """
    robot = _get_robot(env, robot_cfg)
    ddq = robot.data.joint_acc  # (N,DoF)
    return -weight * torch.norm(ddq, dim=1)


def ee_twist_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    weight: float,
) -> torch.Tensor:
    """
    Penalize end-effector spatial velocity magnitude:
        - ( ||v_lin||_2 + ||v_ang||_2 )
    """
    robot = _get_robot(env, robot_cfg)
    idx = _ee_body_index(robot, robot_cfg.body_names[0])
    v_lin = robot.data.body_lin_vel_w[:, idx]
    v_ang = robot.data.body_ang_vel_w[:, idx]
    return -weight * (torch.norm(v_lin, dim=1) + torch.norm(v_ang, dim=1))


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    weight: float,
) -> torch.Tensor:
    """
    Penalize action delta: -||a_t - a_{t-1}||_2.

    Expects last action buffer to be populated by the wrapper.
    """
    a_t = env.action_manager.action  # (N, A)
    a_prev = env.action_manager.prev_action  # (N, A)
    return -weight * torch.norm(a_t - a_prev, dim=1)

# -------------------------
# Success and Termination Rewards
# -------------------------


def success_bonus(env: ManagerBasedRLEnv, threshold: float, weight: float) -> torch.Tensor:
    """
    Returns 1.0 for each environment where the end-effector is within
    `threshold` meters of the current goal position, else 0.0.
    """
    robot = env.scene["robot"]
    ee_idx = robot.body_names.index("panda_hand")
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    goal_cmd = env.command_manager.get_command("goal_pose")  # (N, 7)
    goal_pos = goal_cmd[..., :3]
    dist = torch.norm(ee_pos - goal_pos, dim=1)
    return weight * (dist < threshold).float()


def collision_penalty(
    env: ManagerBasedRLEnv,
    weight: float,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-env penalty: −weight where any illegal contact is present, else 0."""
    mask = illegal_contact(env, threshold, sensor_cfg)
    return -weight * mask.to(dtype=torch.float32)
