# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script defines curriculum functions that can be used to modify the environment's parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def linear_interpolation(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
    address: str,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly interpolates a parameter over a number of steps.

    This function can be used as a `modify_fn` with the `modify_term_cfg` curriculum term.
    It changes a specified environment parameter from a `start_value` to an `end_value`
    over a range of steps defined by `start_step` and `end_step`.

    Args:
        env: The learning environment.
        start_value: The starting value of the parameter.
        end_value: The final value of the parameter.
        start_step: The step count at which the interpolation begins.
        end_step: The step count at which the interpolation ends.

    Returns:
        The new interpolated value for the parameter.
    """
    current_step = env.common_step_counter
    if current_step < start_step:
        return start_value
    if current_step > end_step:
        return end_value
    # Compute the interpolation factor (alpha)
    alpha = (current_step - start_step) / (end_step - start_step)
    # Return the interpolated value
    return (1 - alpha) * start_value + alpha * end_value
