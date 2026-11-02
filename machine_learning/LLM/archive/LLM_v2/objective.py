"""Objective functions aligned to the RL v3 comparison target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

try:
    from machine_learning.RL.two_opt_chain_rl_v4.rewards import (
        MIXED_WEIGHTED,
        compute_final_correction,
        incremental_step_reward,
        single_step_reward,
    )
except ImportError:
    from ...RL.two_opt_chain_rl_v4.rewards import (
        MIXED_WEIGHTED,
        compute_final_correction,
        incremental_step_reward,
        single_step_reward,
    )


DEFAULT_OBJECTIVE_VERSION = "rl_v4_episode_utility_v1"
DEFAULT_REWARD_TYPE = MIXED_WEIGHTED
DEFAULT_RUNTIME_WEIGHT = 0.1
DEFAULT_TWO_QUBIT_WEIGHT = 1.0
DEFAULT_DEPTH_WEIGHT = 0.1
DEFAULT_FINAL_REWARD_WEIGHT = 0.5


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for deterministic chain scoring."""

    reward_type: str = DEFAULT_REWARD_TYPE
    runtime_weight: float = DEFAULT_RUNTIME_WEIGHT
    two_qubit_weight: float = DEFAULT_TWO_QUBIT_WEIGHT
    depth_weight: float = DEFAULT_DEPTH_WEIGHT
    final_reward_weight: float = DEFAULT_FINAL_REWARD_WEIGHT
    objective_version: str = DEFAULT_OBJECTIVE_VERSION


def _safe_float(value: Any) -> float:
    """Convert a value to a finite float."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def score_original_circuit(_: dict[str, Any], config: ObjectiveConfig) -> float:
    """Return the deterministic score for stopping immediately."""
    del config
    return 0.0


def score_single_chain(
    row: dict[str, Any],
    config: ObjectiveConfig,
) -> float:
    """Score a one-step chain using the same structure as the RL episode utility."""
    step_reward = single_step_reward(
        row,
        reward_type=config.reward_type,
        runtime_weight=config.runtime_weight,
        two_qubit_weight=config.two_qubit_weight,
        depth_weight=config.depth_weight,
    )
    final_reward = compute_final_correction(
        row,
        reward_type=config.reward_type,
        final_reward_weight=config.final_reward_weight,
        two_qubit_weight=config.two_qubit_weight,
        depth_weight=config.depth_weight,
    )
    return _safe_float(step_reward) + _safe_float(final_reward)


def score_two_step_chain(
    final_row: dict[str, Any],
    first_row: dict[str, Any],
    config: ObjectiveConfig,
) -> float:
    """Score a two-step chain using the same structure as the RL episode utility."""
    first_step_reward = single_step_reward(
        first_row,
        reward_type=config.reward_type,
        runtime_weight=config.runtime_weight,
        two_qubit_weight=config.two_qubit_weight,
        depth_weight=config.depth_weight,
    )
    second_step_reward = incremental_step_reward(
        final_row,
        first_row,
        reward_type=config.reward_type,
        runtime_weight=config.runtime_weight,
        two_qubit_weight=config.two_qubit_weight,
        depth_weight=config.depth_weight,
    )
    final_reward = compute_final_correction(
        final_row,
        reward_type=config.reward_type,
        final_reward_weight=config.final_reward_weight,
        two_qubit_weight=config.two_qubit_weight,
        depth_weight=config.depth_weight,
    )
    return (
        _safe_float(first_step_reward)
        + _safe_float(second_step_reward)
        + _safe_float(final_reward)
    )


def objective_metadata(config: ObjectiveConfig) -> dict[str, Any]:
    """Return objective metadata for experiment logging."""
    return {
        "objective_version": config.objective_version,
        "reward_type": config.reward_type,
        "runtime_weight": float(config.runtime_weight),
        "two_qubit_weight": float(config.two_qubit_weight),
        "depth_weight": float(config.depth_weight),
        "final_reward_weight": float(config.final_reward_weight),
    }
