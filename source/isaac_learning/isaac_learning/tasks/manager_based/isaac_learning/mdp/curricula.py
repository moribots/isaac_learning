# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Success-rate–driven curriculum helpers for IsaacLab 2.2 Manager-based tasks.

Use with:
    CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.success_reward.params.threshold",  # example
            "modify_fn": success_linear,       # or success_toggle
            "modify_params": { ... }           # kwargs below
        },
    )

Notes:
- We return mdp.modify_term_cfg.NO_CHANGE when no write is desired.
- Success rate is taken from RecorderManager counters (per-env rolling success).
- Functions may return either a scalar (applied to all envs) or a tensor per env.
"""
from __future__ import annotations
from typing import Any, Optional

import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp

_EPS = 1e-9


def _get_success_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns per-env success rate in [0, 1]. Falls back to zeros if not available.

    We read RecorderManager stats if present; else we approximate by using a decayed
    moving-average of the 'success' termination flag stored in the env's extras.
    """
    device = env.device if hasattr(env, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preferred: recorder manager rolling success
    if hasattr(env, "recorder") and env.recorder is not None:
        # recorder exposes 'episode_success_rate' shaped (num_envs,)
        rate = getattr(env.recorder, "episode_success_rate", None)
        if isinstance(rate, torch.Tensor) and rate.numel() == env.num_envs:
            return rate[env_ids].to(device)

    # Fallback: extras flag averaged via a tiny EMA stored on env
    if not hasattr(env, "_success_rate_ema"):
        env._success_rate_ema = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
    # read last-step success flags if present
    flag = 0.0
    if isinstance(env.extras, dict):
        # common keys used in this project
        if "is_success" in env.extras and isinstance(env.extras["is_success"], torch.Tensor):
            s = env.extras["is_success"].to(device).float()
            flag = s
        elif "Episode_Termination/success" in env.extras and isinstance(env.extras["Episode_Termination/success"], torch.Tensor):
            flag = env.extras["Episode_Termination/success"].to(device).float()
    if not torch.is_tensor(flag):
        flag = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
    # EMA update with long memory
    env._success_rate_ema = 0.995 * env._success_rate_ema + 0.005 * flag
    return env._success_rate_ema[env_ids]


def _near_goal(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pos_thresh_scale: float = 1.0) -> torch.Tensor:
    """
    Boolean mask per env for 'near goal'.

    We use the current termination success thresholds if present:
      - position_threshold (meters)
      - orientation_threshold (radians)

    Args:
        pos_thresh_scale: multiply the current position threshold by this factor
                          to define "near". e.g., 1.0 means same as success.
    """
    device = env.device if hasattr(env, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Current thresholds from term cfg if available
    pos_thr = 0.05
    ori_thr = 0.10
    try:
        pos_thr = float(env.termination_manager.cfg.success.params["position_threshold"])
    except Exception:
        pass
    try:
        ori_thr = float(env.termination_manager.cfg.success.params["orientation_threshold"])
    except Exception:
        pass
    pos_thr = pos_thr * float(pos_thresh_scale)

    # Acquire current pose error from extras if provided by rewards
    pos_err = env.extras.get("Metrics/goal_pose/position_error", None) if isinstance(env.extras, dict) else None
    ori_err = env.extras.get("Metrics/goal_pose/orientation_error", None) if isinstance(env.extras, dict) else None
    if isinstance(pos_err, torch.Tensor) and isinstance(ori_err, torch.Tensor):
        pos_ok = (pos_err.to(device).float()[env_ids] <= pos_thr)
        ori_ok = (ori_err.to(device).float()[env_ids] <= ori_thr)
        return pos_ok & ori_ok

    # Fallback: treat none as far
    return torch.zeros((len(env_ids),), dtype=torch.bool, device=device)


def throttle(every_n_steps: int):
    """
    Decorator to restrict curriculum writes to every N common steps.
    """
    def _wrap(fn):
        def _inner(env: ManagerBasedRLEnv, env_ids: torch.Tensor, *args, **kwargs):
            step = getattr(env, "common_step_counter", 0)
            if (step % max(1, int(every_n_steps))) != 0:
                return mdp.modify_term_cfg.NO_CHANGE
            return fn(env, env_ids, *args, **kwargs)
        return _inner
    return _wrap


@throttle(every_n_steps=128)
def success_linear(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_value: float,
    end_value: float,
    start_success: float,
    end_success: float,
    clamp: bool = True,
) -> torch.Tensor | float:
    """
    Linearly interpolate value vs success rate.

    Returns per-env tensor V = lerp(start_value, end_value, t), where
      t = clamp((S - start_success) / (end_success - start_success), 0, 1)
    and S is the per-env success rate in [0,1].
    """
    S = _get_success_rate(env, env_ids)
    denom = max(_EPS, float(end_success - start_success))
    t = (S - float(start_success)) / denom
    if clamp:
        t = torch.clamp(t, 0.0, 1.0)
    return float(start_value) + (float(end_value) - float(start_value)) * t


@throttle(every_n_steps=128)
def success_toggle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    on_value: float,
    off_value: float,
    threshold: float,
    hysteresis: float = 0.0,
    state_key: str = "_curriculum_toggle_state",
    near_goal_only: bool = False,
    near_goal_scale: float = 1.0,
) -> torch.Tensor | float:
    """
    ON/OFF based on success rate with optional hysteresis and 'near goal' gating.

    - If near_goal_only=True, we only turn ON for envs currently near goal
      under the scaled termination thresholds.
    - Hysteresis: once ON, stays ON until success < (threshold - hysteresis).
    """
    device = env.device if hasattr(env, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = _get_success_rate(env, env_ids)

    # persistent per-env state
    if not hasattr(env, state_key):
        setattr(env, state_key, torch.zeros((env.num_envs,), dtype=torch.bool, device=device))
    state: torch.Tensor = getattr(env, state_key)

    on_mask = S >= float(threshold)
    off_mask = S < float(max(0.0, threshold - hysteresis))
    # apply near-goal gating only to the ON transition
    if near_goal_only:
        on_mask &= _near_goal(env, env_ids, pos_thresh_scale=near_goal_scale)

    # update state
    state = state.clone()
    state[env_ids] |= on_mask
    state[env_ids] &= ~off_mask
    setattr(env, state_key, state)

    return torch.where(state[env_ids], torch.as_tensor(on_value, device=device), torch.as_tensor(off_value, device=device))
