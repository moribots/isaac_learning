"""
This module provides the configuration for the RSL-RL PPO agent.
It includes settings for the policy, algorithm, and runner.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg


@configclass
class RslRlPpoAgentCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the RSL-RL PPO agent."""

    seed = 42
    max_iterations = 2501
    num_steps_per_env = 24
    save_interval = 200
    experiment_name = "franka_trajectory_tracking"
    run_name = ""
    logger = "wandb"
    wandb_entity = "moribots"  # Explicitly set the W&B entity

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
