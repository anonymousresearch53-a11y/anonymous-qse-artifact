"""Editable reward functions for optimizer-chain reinforcement learning.

EDITABLE BY THE RESEARCH AGENT.

Training rewards are intentionally separate from the locked evaluator.  An
experiment can therefore explore depth-, gate-, runtime-, sparse-, or
multi-objective rewards without changing the benchmark used to promote models.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


QUALITY_METRIC_NAMES = (
    "two_qubit_gates",
    "depth",
    "total_gates",
    "two_qubit_depth",
)


def relative_improvement(previous: float, current: float) -> float:
    """Return the fractional reduction from ``previous`` to ``current``.

    Positive values indicate improvement and negative values indicate
    degradation.  A zero denominator yields zero because no meaningful
    percentage reduction exists.
    """

    if previous == 0:
        return 0.0
    return (float(previous) - float(current)) / float(previous)


def metric_improvements(
    reference_metrics: Mapping[str, float],
    current_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Calculate relative improvement for every supported quality metric."""

    return {
        name: relative_improvement(
            float(reference_metrics[name]), float(current_metrics[name])
        )
        for name in QUALITY_METRIC_NAMES
    }


def normalized_runtime(runtime_seconds: float, runtime_scale: float) -> float:
    """Normalize runtime with the locked training-set 99th percentile.

    The logarithm keeps very slow optimizer runs from overwhelming all quality
    feedback while preserving a monotonic cost signal.
    """

    denominator = math.log1p(max(float(runtime_scale), 1e-9))
    return math.log1p(max(float(runtime_seconds), 0.0)) / denominator


def _weighted_quality(
    improvements: Mapping[str, float],
    weights: Mapping[str, float],
    degradation_multiplier: float,
) -> float:
    """Combine selected metric improvements using normalized absolute weights."""

    active = {
        metric: float(weight)
        for metric, weight in weights.items()
        if metric in improvements and float(weight) != 0.0
    }
    if not active:
        return 0.0

    denominator = sum(abs(weight) for weight in active.values())
    total = 0.0
    for metric, weight in active.items():
        value = float(improvements[metric])
        if value < 0:
            value *= float(degradation_multiplier)
        total += weight * value
    return total / denominator


def compute_reward(
    *,
    previous_metrics: Mapping[str, float],
    current_metrics: Mapping[str, float],
    initial_metrics: Mapping[str, float],
    step_runtime_seconds: float,
    total_runtime_seconds: float,
    runtime_scale: float,
    step_index: int,
    terminal: bool,
    failed: bool,
    invalid_action: bool,
    config: Mapping[str, Any],
) -> float:
    """Compute the editable reward for one environment transition.

    Args:
        previous_metrics: Circuit metrics immediately before the action.
        current_metrics: Circuit metrics after the action.
        initial_metrics: Metrics at the beginning of the episode.
        step_runtime_seconds: Runtime attributable to the current action.
        total_runtime_seconds: Cumulative runtime for the selected chain.
        runtime_scale: Training-derived runtime normalization constant.
        step_index: Zero-based optimizer step being rewarded.
        terminal: Whether the episode ends after this transition.
        failed: Whether the selected optimizer/chain has an error flag.
        invalid_action: Whether the action violates the environment contract.
        config: Reward declaration from ``experiment_config.py``.

    Returns:
        The scalar training reward.

    Raises:
        ValueError: If an unknown reward family or metric reference is used.
    """

    if invalid_action:
        return float(config.get("invalid_action_penalty", -1.0))
    if failed:
        return float(config.get("failure_penalty", -1.0))

    family = str(config.get("family", "weighted_dense")).lower()
    reference_name = str(config.get("reference", "previous")).lower()
    if reference_name == "previous":
        reference = previous_metrics
    elif reference_name == "initial":
        reference = initial_metrics
    else:
        raise ValueError(f"Unknown reward reference: {reference_name!r}")

    # Sparse rewards intentionally postpone quality credit until the final
    # decision, but may still charge a per-step cost while the episode unfolds.
    is_sparse = family in {"weighted_terminal", "sparse_terminal"}
    if is_sparse and not terminal:
        quality = 0.0
    else:
        if is_sparse:
            reference = initial_metrics
        improvements = metric_improvements(reference, current_metrics)
        quality = _weighted_quality(
            improvements,
            config.get("metric_weights", {}),
            float(config.get("degradation_multiplier", 1.0)),
        )

    if family not in {
        "weighted_dense",
        "dense",
        "weighted_terminal",
        "sparse_terminal",
    }:
        raise ValueError(f"Unknown reward family: {family!r}")

    runtime_seconds = total_runtime_seconds if is_sparse and terminal else step_runtime_seconds
    runtime_cost = float(config.get("runtime_weight", 0.0)) * normalized_runtime(
        runtime_seconds, runtime_scale
    )
    reward = (
        quality
        - runtime_cost
        - float(config.get("step_penalty", 0.0))
        + (float(config.get("terminal_bonus", 0.0)) if terminal else 0.0)
    )

    clip = config.get("reward_clip")
    if clip is not None:
        limit = abs(float(clip))
        reward = max(-limit, min(limit, reward))
    return float(reward)
