# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the Franka reach environment with joint torque (effort) control.

This configuration inherits from the base ReachEnvCfg and specifies the Franka robot
as the robot asset and sets the action manager to use joint torque (effort) control.
"""

from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.utils import configclass

from ...reach_env_cfg import ReachEnvCfg
from .franka_cfg import FRANKA_PANDA_CFG
import math


@configclass
class FrankaReachJointTorqueEnvCfg(ReachEnvCfg):
    """Configuration for the Franka reach environment with joint torque (effort) control."""

    def __post_init__(self):
        """Post-initialization checks."""
        # Call the parent's post_init method
        super().__post_init__()

        # Scene settings
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # Action settings
        self.actions.arm_action = JointEffortActionCfg(
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale={
                "panda_joint[1-4]*": 87.0,
                "panda_joint[5-7]*": 12.0,
            },
        )


@configclass
class FrankaReachJointTorqueEnvCfg_PLAY(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # Action settings
        self.actions.arm_action = JointEffortActionCfg(
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale={
                "panda_joint[1-4]*": 87.0,
                "panda_joint[5-7]*": 12.0,
            },
        )
