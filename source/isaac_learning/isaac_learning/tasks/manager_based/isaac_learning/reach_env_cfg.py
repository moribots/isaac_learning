# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

# Import the new config file that contains all the MDP terms
from .config.franka.franka_cfg import FRANKA_PANDA_CFG
from .mdp.cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    EventCfg,
)

import numpy as np

##
# Scene
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # Contact sensor
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link[1-7]$",
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )


##
# Environment
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka trajectory tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()

    # The `terminations` attribute was missing. It is required by the base environment class.
    terminations: TerminationsCfg = TerminationsCfg()

    # Provide a default value for the nested command configuration.
    commands: CommandsCfg = CommandsCfg(
    )

    # Curriculum settings
    curriculum: CurriculumCfg = CurriculumCfg()

    # Adds one-frame PD hold on reset
    events: EventCfg = EventCfg()

    # Uniform sampling ranges for goal and shelf poses
    goal_pose_range = {
        "x": [0.25, 0.6],
        "y": [-0.3, 0.3],
        "z": [0.25, 0.8],
        "rpy": [(0.0, 0.0, 0.0)]
    }
    shelf_pose_range = {
        "x": [0.4, 0.7],
        "y": [-0.2, 0.2],
        "z": [0.3, 0.6],
        "rpy": [(0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2)]
    }

    def __post_init__(self):
        """Post-initialization checks."""
        self.decimation = 2
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (7.5, 7.5, 7.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
