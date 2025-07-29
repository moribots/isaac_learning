# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the Franka reach environment with joint position control.

This configuration inherits from the base ReachEnvCfg and specifies the Franka robot
as the robot asset and sets the action manager to use joint position control.
"""

from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.utils import configclass

from ...reach_env_cfg import ReachEnvCfg
from ...logging_env import FrankaReachEnv  # for more detailed W&B logging
from .franka_cfg import FRANKA_PANDA_CFG
import math


@configclass
class FrankaReachJointTorqueEnvCfg(ReachEnvCfg):
    """Configuration for the Franka reach environment with joint position control."""

    _env_class = FrankaReachEnv

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
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale=100.0
        )


@configclass
class FrankaReachJointTorqueEnvCfg_PLAY(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 3
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
