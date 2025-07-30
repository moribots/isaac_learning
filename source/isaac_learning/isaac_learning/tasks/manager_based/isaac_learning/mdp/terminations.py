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
from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab.envs.mdp.observations import joint_pos
from isaaclab.envs.mdp.observations import joint_vel
from isaaclab.envs.mdp.observations import joint_effort

from ..config.franka.franka_cfg import *


def _resolve_arm_joint_ids(env: ManagerBasedRLEnv, robot: Articulation, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the 7 Franka arm joint indices as a LongTensor on env.device."""
    if robot_cfg.joint_ids is not None and len(robot_cfg.joint_ids) == 7:
        return torch.as_tensor(robot_cfg.joint_ids, device=env.device, dtype=torch.long)
    # Fallback: resolve by regex against the articulation's actual joint names
    ids, _ = robot.find_joints([r"panda_joint[1-7]"])
    return torch.as_tensor(ids, device=env.device, dtype=torch.long)


def _print_joint_limit_violations(
    env: ManagerBasedRLEnv,
    joint_pos: torch.Tensor,     # shape [num_envs, 7]
    min_limits: torch.Tensor,    # shape [7,]
    max_limits: torch.Tensor,    # shape [7,]
    name: str                    # e.g. "Pos", "Vel", or "Torque"
) -> None:
    return
    # find which entries violate
    violated_min = (joint_pos < min_limits).nonzero(as_tuple=True)
    violated_max = (joint_pos > max_limits).nonzero(as_tuple=True)

    # for each violation, print an informative line
    for idx in range(violated_min[0].shape[0]):
        e = violated_min[0][idx].item()
        j = violated_min[1][idx].item()
        print(f"  Env {e}: Joint {j} violated {name} MIN limit. "
              f"Value={joint_pos[e, j]:.4f}, Min={min_limits[j]:.4f}")
    for idx in range(violated_max[0].shape[0]):
        e = violated_max[0][idx].item()
        j = violated_max[1][idx].item()
        print(f"  Env {e}: Joint {j} violated {name} MAX limit. "
              f"Value={joint_pos[e, j]:.4f}, Max={max_limits[j]:.4f}")


def franka_joint_pos_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate per env if any Franka arm joint leaves its *soft* position limits."""
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids = _resolve_arm_joint_ids(env, robot, robot_cfg)

    # Current joint positions for the 7 arm joints -> [N, 7]
    q = robot.data.joint_pos[:, arm_ids]

    # Soft position limits from the articulation data -> [N, 7, 2] (low, high)
    q_lims = robot.data.soft_joint_pos_limits[:, arm_ids, :]
    q_min = q_lims[0, :, 0]  # identical across envs; take env 0
    q_max = q_lims[0, :, 1]

    low = q < q_min.view(1, -1)
    high = q > q_max.view(1, -1)
    viol = low | high  # [N, 7]

    if viol.any() and False:
        # Optional: pretty print which joints/envs failed
        names = [robot.data.joint_names[i] for i in arm_ids.tolist()]
        env_ids = torch.nonzero(viol.any(dim=1), as_tuple=False).squeeze(1).tolist()
        for e in env_ids:
            which = torch.nonzero(viol[e], as_tuple=False).squeeze(1).tolist()
            for j in which:
                side = "MIN" if low[e, j] else "MAX"
                print(f"Env {e}: {names[j]} POS {side}  q={q[e, j].item():.4f}  "
                      f"range=[{q_min[j].item():.4f}, {q_max[j].item():.4f}]")
    return viol.any(dim=1)


def franka_joint_vel_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate per env if any Franka arm joint velocity exceeds its *soft* limits (symmetric)."""
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids = _resolve_arm_joint_ids(env, robot, robot_cfg)

    dq = robot.data.joint_vel[:, arm_ids]                 # [N, 7]
    dq_abs = robot.data.soft_joint_vel_limits[0, arm_ids]  # [7]   (use ±dq_abs)

    low = dq < (-dq_abs).view(1, -1)
    high = dq > (dq_abs).view(1, -1)
    viol = low | high

    if viol.any() and False:
        names = [robot.data.joint_names[i] for i in arm_ids.tolist()]
        env_ids = torch.nonzero(viol.any(dim=1), as_tuple=False).squeeze(1).tolist()
        for e in env_ids:
            which = torch.nonzero(viol[e], as_tuple=False).squeeze(1).tolist()
            for j in which:
                side = "MIN" if low[e, j] else "MAX"
                print(f"Env {e}: {names[j]} VEL {side}  dq={dq[e, j].item():.4f}  "
                      f"range=[{-dq_abs[j].item():.4f}, {dq_abs[j].item():.4f}]")
    return viol.any(dim=1)


def _effort_limits_from_actuators(robot: Articulation, env: ManagerBasedRLEnv, arm_ids: torch.Tensor) -> torch.Tensor:
    """Build a [7]-vector of effort limits using joint-name groups (87 Nm for joints 1–4, 12 Nm for 5–7)."""
    tau_abs = torch.zeros((arm_ids.shape[0],), device=env.device, dtype=torch.float32)
    # Resolve which arm_ids belong to joints 1–4 vs 5–7 by name
    names = [robot.data.joint_names[i] for i in arm_ids.tolist()]
    for idx, n in enumerate(names):
        # n like "panda_jointX"; parse X
        try:
            jnum = int(n.split("panda_joint")[1])
        except Exception:
            jnum = None
        tau_abs[idx] = 87.0 if jnum is not None and 1 <= jnum <= 4 else 12.0
    return tau_abs


def franka_joint_torque_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Terminate per env if any Franka arm joint applied torque exceeds limits.
    Note: with ImplicitActuatorCfg, 'applied_torque' may be an approximation.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids = _resolve_arm_joint_ids(env, robot, robot_cfg)

    # Try articulation data first
    tau = getattr(robot.data, "applied_torque", None)
    if tau is not None:
        tau = tau[:, arm_ids]  # [N, 7]
    else:
        # Fallback to observation helper if needed
        tau = joint_effort(env, robot_cfg)[:, arm_ids]

    # Try effort limits from data; if missing/unset, build from actuator groups (87/12)
    tau_abs = getattr(robot.data, "joint_effort_limits", None)
    if tau_abs is not None:
        tau_abs = tau_abs[0, arm_ids].abs()
        if torch.any(tau_abs > 1e6):  # sentinel large values → treat as unset
            tau_abs = _effort_limits_from_actuators(robot, env, arm_ids)
    else:
        tau_abs = _effort_limits_from_actuators(robot, env, arm_ids)

    low = tau < (-tau_abs).view(1, -1)
    high = tau > (tau_abs).view(1, -1)
    viol = low | high

    if viol.any() and False:
        names = [robot.data.joint_names[i] for i in arm_ids.tolist()]
        env_ids = torch.nonzero(viol.any(dim=1), as_tuple=False).squeeze(1).tolist()
        for e in env_ids:
            which = torch.nonzero(viol[e], as_tuple=False).squeeze(1).tolist()
            for j in which:
                side = "MIN" if low[e, j] else "MAX"
                print(f"Env {e}: {names[j]} TAU {side}  τ={tau[e, j].item():.4f}  "
                      f"range=[{-tau_abs[j].item():.4f}, {tau_abs[j].item():.4f}]")
    return viol.any(dim=1)


def franka_joint_limits(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Combined termination: position OR velocity OR torque limit violated."""
    pos = franka_joint_pos_limits(env, robot_cfg)
    vel = franka_joint_vel_limits(env, robot_cfg)
    tau = franka_joint_torque_limits(env, robot_cfg)
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
    # undesired_contacts returns an (N,) tensor of counts per env
    counts: torch.Tensor = undesired_contacts(env, threshold, sensor_cfg)

    # boolean mask of which envs had ≥1 violation
    collided: torch.Tensor = counts > 0

    # optional per-env debug print
    for e, c in enumerate(collided.tolist()):
        if c:
            print(f"Env {e}: self - collision detected({int(counts[e])} contacts above {threshold})")

    # inside franka_self_collision before computing `collided`
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history    # shape [N_envs, hist, N_bodies, 3]
    # take max over history window, then norm to get per-body force
    max_forces = torch.max(
        torch.norm(net_forces[..., :], dim=-1), dim=1
    )[0]  # shape (N_envs, N_bodies)

    for e in range(env.num_envs):
        # find bodies with any non-zero force
        body_idxs = (max_forces[e] > 0.0).nonzero(as_tuple=False).squeeze(-1).tolist()
        if body_idxs:
            mags = [max_forces[e, i].item() for i in body_idxs]
            print(f"Env {e} spurious contacts on bodies {body_idxs} with mags {mags}")

    return collided


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
