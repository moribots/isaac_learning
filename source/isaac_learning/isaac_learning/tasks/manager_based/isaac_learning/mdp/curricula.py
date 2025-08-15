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
- No external scheduler. The modify_fn returns mdp.modify_term_cfg.NO_CHANGE to skip writes.  # see docs
- Success rate uses exported episode counters from RecorderManager.
"""
from __future__ import annotations
from typing import Any, Optional

import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp

_EPS = 1e-9


def _success_rate(env: ManagerBasedRLEnv) -> float:
    """
    Global success rate across envs based on exported episode counts.
    """
    rec = env.recorder_manager
    succ = rec.exported_successful_episode_count
    fail = rec.exported_failed_episode_count
    total = succ + fail
    return float(succ / total) if total > 0 else 0.0


def _throttle(env: ManagerBasedRLEnv, key: str, min_step_delta: int) -> bool:
    """
    Return True if enough steps elapsed since last update under `key`.
    """
    if min_step_delta <= 0:
        return True
    step = int(getattr(env, "common_step_counter", 0))
    last = getattr(env, key, None)
    if last is None or step - int(last) >= int(min_step_delta):
        setattr(env, key, step)
        return True
    return False


def success_linear(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    old_value: Any,
    *,
    start_value: float,
    end_value: float,
    start_success: float,
    end_success: float,
    clip: bool = True,
    # update gating
    min_step_delta: int = 0,
    state_key: Optional[str] = None,
) -> Any:
    """
    Linearly interpolate a parameter as success improves.

    When success <= start_success -> start_value.
    When success >= end_success -> end_value.
    Else linear interpolation. Returns NO_CHANGE on tiny deltas or if throttled.

    Args:
        env: IsaacLab environment.
        env_ids: Unused. Required by signature.
        old_value: Current parameter value at `address`.
        start_value, end_value: Range of parameter.
        start_success, end_success: Range of success ∈ [0,1] mapped to values.
        clip: Clamp success into [start_success, end_success].
        min_step_delta: Minimum env steps between writes. 0 disables.
        state_key: Optional throttle state name. Defaults to function name.
    """
    throttle_key = (state_key or "success_linear") + "_last_step"
    if not _throttle(env, throttle_key, min_step_delta):
        return mdp.modify_term_cfg.NO_CHANGE

    s = _success_rate(env)
    if clip and end_success > start_success:
        s = min(max(s, start_success), end_success)
    denom = max(_EPS, abs(end_success - start_success))
    t = max(0.0, min(1.0, (s - start_success) / denom))
    new_value = (1.0 - t) * float(start_value) + t * float(end_value)

    if isinstance(old_value, (int, float)) and abs(float(old_value) - new_value) < 1e-6:
        return mdp.modify_term_cfg.NO_CHANGE
    return new_value


def success_toggle(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    old_value: Any,
    *,
    on_value: float | int,
    off_value: float | int,
    threshold: float,
    hysteresis: float = 0.02,
    state_key: Optional[str] = None,
    min_step_delta: int = 0,
) -> Any:
    """
    Select on/off value based on success with hysteresis. Throttled by steps.

    Args:
        env: IsaacLab environment.
        env_ids: Unused.
        old_value: Current parameter value at `address`.
        on_value, off_value: Returned values.
        threshold: Central threshold in [0,1].
        hysteresis: Half-width of band to prevent flapping.
        state_key: Base name for internal state on env.
        min_step_delta: Min steps between potential writes.
    """
    throttle_key = (state_key or "success_toggle") + "_last_step"
    if not _throttle(env, throttle_key, min_step_delta):
        return mdp.modify_term_cfg.NO_CHANGE

    s = _success_rate(env)
    key = (state_key or "_curriculum_toggle_state")
    state = getattr(env, key, None)

    lo = max(0.0, threshold - hysteresis)
    hi = min(1.0, threshold + hysteresis)

    if state is None:
        state = s >= threshold
    else:
        state = bool(state)
        state = (s >= lo) if state else (s >= hi)

    setattr(env, key, bool(state))
    new_value = on_value if state else off_value

    if isinstance(old_value, (int, float)) and abs(float(old_value) - float(new_value)) < 1e-12:
        return mdp.modify_term_cfg.NO_CHANGE
    return new_value
