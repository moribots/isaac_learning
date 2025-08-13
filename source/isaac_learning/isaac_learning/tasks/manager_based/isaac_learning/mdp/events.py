# SPDX-License-Identifier: BSD-3-Clause
# Canonical zero-velocity reset: root pose -> zero root vel -> joint state (dq=0) -> robot.reset()

from __future__ import annotations

from typing import Any

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import torch
import numpy as np
from isaaclab.utils.math import quat_from_euler_xyz


def authoritative_zero_vel_reset(
    env: Any,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
) -> None:
    """
    Write root pose, zero root velocity, write joint state with zero dq, then call Articulation.reset().
    No targets are set on the reset frame.

    Args:
        env: Manager-based Isaac Lab env (has .scene, .device, .num_envs, .scene.env_origins).
        env_ids: Tensor of env indices to reset. If None, applies to all envs.
        robot_cfg: Scene entity selecting the robot, e.g., SceneEntityCfg(name="robot").
    """
    asset: Articulation = env.scene[robot_cfg.name]

    # Resolve env ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Default root state: [pos(3) | quat(4) | lin(3) | ang(3)]
    default_root = asset.data.default_root_state[env_ids].clone()

    # Place at env origins if provided
    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        origins = env.scene.env_origins[env_ids]
        default_root[:, 0:3] = origins

    # Joint defaults
    q_default = asset.data.default_joint_pos[env_ids].clone()
    dq_zeros = torch.zeros_like(asset.data.default_joint_vel[env_ids])

    # 1) root pose (7)
    root_pose = default_root[:, :7]
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    # 2) zero root velocity (6)
    root_vel_zeros = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=default_root.dtype)
    asset.write_root_velocity_to_sim(root_vel_zeros, env_ids=env_ids)

    # 3) joint state with dq = 0
    asset.write_joint_state_to_sim(q_default, dq_zeros, env_ids=env_ids)

    # 4) finalize reset to clear internal caches
    asset.reset(env_ids=env_ids)


def _sample_xyz(ranges: Dict[str, Tuple[float, float]], rng: np.random.RandomState) -> Tuple[float, float, float]:
    return (
        float(rng.uniform(*ranges["x"])),
        float(rng.uniform(*ranges["y"])),
        float(rng.uniform(*ranges["z"])),
    )


def _sample_rpy(rpy_spec: Iterable[Tuple[float, float, float]], rng: np.random.RandomState) -> Tuple[float, float, float]:
    lst = list(rpy_spec)
    idx = int(rng.randint(0, len(lst)))
    return lst[idx]


def place_goal_and_shelf_on_reset(
    env: Any,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg | None = None,
) -> None:
    """
    Reads:
      - goal_path  := env.scene.ee_goal.prim_path
      - shelf_path := env.scene.shelf.prim_path
      - goal_ranges  := env.task.goal_pose_range with keys x,y,z,rpy
      - shelf_ranges := env.task.shelf_pose_range with keys x,y,z,rpy
    Samples per-env poses and writes them before robot reset.
    """
    rng = getattr(env, "_np_random", None)
    if rng is None:
        rng = np.random.RandomState()

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    goal_path = env.scene.ee_goal.prim_path
    # shelf_path = env.scene.shelf.prim_path
    goal_ranges = env.task.goal_pose_range
    # shelf_ranges = env.task.shelf_pose_range

    for i_t in env_ids.tolist():
        i = int(i_t)

        gx, gy, gz = _sample_xyz(goal_ranges, rng)
        gr, gp, gyaw = _sample_rpy(goal_ranges["rpy"], rng)
        env.scene.set_world_pose(
            goal_path,
            position=(gx, gy, gz),
            orientation=_quat_from_euler_xyz(gr, gp, gyaw),
            env_index=i,
        )

        # sx, sy, sz = _sample_xyz(shelf_ranges, rng)
        # sr, sp, syaw = _sample_rpy(shelf_ranges["rpy"], rng)
        # env.scene.set_world_pose(
        #     shelf_path,
        #     position=(sx, sy, sz),
        #     orientation=_quat_from_euler_xyz(sr, sp, syaw),
        #     env_index=i,
        # )
