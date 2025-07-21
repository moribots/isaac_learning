# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def ee_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in the environment frame.

    This function computes the position of the end-effector, which is defined by the
    first body name in the `robot_cfg`, relative to the environment's origin.
    """
    # extract the robot from the scene
    robot = env.scene[robot_cfg.name]
    # resolve the body index for the end-effector
    ee_body_id = robot_cfg.body_ids[0]
    # obtain the end-effector position in the world frame
    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    # return the position in the environment frame by subtracting the environment origins
    return ee_pos_w - env.scene.env_origins


def ee_quat(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector orientation in the environment frame.

    This function computes the orientation of the end-effector, which is defined by the
    first body name in the `robot_cfg`. The orientation is the same in the world
    and environment frames.
    """
    # extract the robot from the scene
    robot = env.scene[robot_cfg.name]
    # resolve the body index for the end-effector
    ee_body_id = robot_cfg.body_ids[0]
    # return the orientation in the environment frame
    return robot.data.body_quat_w[:, ee_body_id]
