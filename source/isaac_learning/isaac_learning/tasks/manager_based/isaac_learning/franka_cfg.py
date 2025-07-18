# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script defines the configuration for the Franka robot asset by loading the standard
asset from isaaclab_assets and modifying it with custom properties.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG as FRANKA_PANDA_DEFAULT_CFG  # isort: skip

# Create the final robot configuration by replacing properties of the standard Franka asset
FRANKA_PANDA_CFG = FRANKA_PANDA_DEFAULT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    # Explicitly set activate_contact_sensors to True in the spawn configuration.
    # We copy the usd_path from the default config to avoid having to specify it manually.
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_PANDA_DEFAULT_CFG.spawn.usd_path,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=0.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=1e5,
            damping=1e3,
        ),
    },
)
