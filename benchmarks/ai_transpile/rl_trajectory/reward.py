"""Reward functions for RL-based quantum circuit optimization.

This module provides various reward computation functions consistent with
the design in rl_orchestrator.py. Multiple reward variants are provided
for offline experimentation with different reward shaping strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..transpilers import CircuitMetrics


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    Attributes:
        alpha: Coefficient for metric improvement reward
        beta: Coefficient for time cost penalty
        gamma: Existential tax per step (constant penalty)
        depth_weight: Weight for depth in multi-objective reward
        two_qubit_weight: Weight for two-qubit gates in multi-objective reward
    """

    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.01
    depth_weight: float = 0.3
    two_qubit_weight: float = 0.7


def compute_improvement_only_reward(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    config: RewardConfig | None = None,
) -> float:
    """Compute reward based purely on metric improvement.

    R = alpha * (M_{t-1} - M_t) / M_{t-1}

    where M is the two-qubit gate count.

    Args:
        prev_metrics: Metrics before optimization
        new_metrics: Metrics after optimization
        config: Reward configuration

    Returns:
        Improvement-only reward
    """
    cfg = config or RewardConfig()

    if prev_metrics.two_qubit_gates == 0:
        return 0.0

    relative_improvement = (
        prev_metrics.two_qubit_gates - new_metrics.two_qubit_gates
    ) / prev_metrics.two_qubit_gates

    return cfg.alpha * relative_improvement


def compute_efficiency_reward(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    time_cost: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute time-normalized efficiency reward.

    R_t = alpha * (M_{t-1} - M_t) / M_{t-1} - beta * time_cost - gamma

    This is the primary reward function from rl_orchestrator.py.

    Args:
        prev_metrics: Metrics before optimization
        new_metrics: Metrics after optimization
        time_cost: Time spent on optimization in seconds
        config: Reward configuration

    Returns:
        Efficiency reward
    """
    cfg = config or RewardConfig()

    if prev_metrics.two_qubit_gates == 0:
        relative_improvement = 0.0
    else:
        relative_improvement = (
            prev_metrics.two_qubit_gates - new_metrics.two_qubit_gates
        ) / prev_metrics.two_qubit_gates

    return cfg.alpha * relative_improvement - cfg.beta * time_cost - cfg.gamma


def compute_multi_objective_reward(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    time_cost: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute multi-objective reward combining depth and two-qubit gate reduction.

    R = w_2q * improvement_2q + w_d * improvement_depth - beta * time_cost - gamma

    where:
        - improvement_2q = (prev_2q - new_2q) / prev_2q
        - improvement_depth = (prev_depth - new_depth) / prev_depth

    Args:
        prev_metrics: Metrics before optimization
        new_metrics: Metrics after optimization
        time_cost: Time spent on optimization in seconds
        config: Reward configuration

    Returns:
        Multi-objective reward
    """
    cfg = config or RewardConfig()

    # Two-qubit gate improvement
    if prev_metrics.two_qubit_gates == 0:
        improvement_2q = 0.0
    else:
        improvement_2q = (
            prev_metrics.two_qubit_gates - new_metrics.two_qubit_gates
        ) / prev_metrics.two_qubit_gates

    # Depth improvement
    if prev_metrics.depth == 0:
        improvement_depth = 0.0
    else:
        improvement_depth = (prev_metrics.depth - new_metrics.depth) / prev_metrics.depth

    # Weighted combination
    improvement = (
        cfg.two_qubit_weight * improvement_2q + cfg.depth_weight * improvement_depth
    )

    return cfg.alpha * improvement - cfg.beta * time_cost - cfg.gamma


def compute_sparse_final_reward(
    initial_metrics: CircuitMetrics,
    final_metrics: CircuitMetrics,
    is_final_step: bool,
    config: RewardConfig | None = None,
) -> float:
    """Compute sparse reward given only at the final step.

    R = alpha * (M_0 - M_T) / M_0 if final step, else 0

    This reward variant gives all credit at the end of the episode,
    which can be useful for learning long-horizon optimization strategies.

    Args:
        initial_metrics: Metrics at episode start
        final_metrics: Metrics at current step
        is_final_step: Whether this is the last step in the episode
        config: Reward configuration

    Returns:
        Sparse reward (0 unless final step)
    """
    if not is_final_step:
        return 0.0

    cfg = config or RewardConfig()

    if initial_metrics.two_qubit_gates == 0:
        return 0.0

    total_improvement = (
        initial_metrics.two_qubit_gates - final_metrics.two_qubit_gates
    ) / initial_metrics.two_qubit_gates

    return cfg.alpha * total_improvement


def compute_efficiency_normalized_reward(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    time_cost: float,
    time_budget: float = 300.0,
    config: RewardConfig | None = None,
) -> float:
    """Compute efficiency reward with time cost normalized by the episode budget.

    R_t = alpha * (M_{t-1} - M_t) / M_{t-1} - beta * (time_cost / time_budget) - gamma

    Unlike the raw efficiency reward, the time penalty is dimensionless (0–1 range),
    keeping it on the same scale as the improvement term.

    Args:
        prev_metrics: Metrics before optimization
        new_metrics: Metrics after optimization
        time_cost: Time spent on optimization in seconds
        time_budget: Total episode time budget in seconds (default: 300)
        config: Reward configuration

    Returns:
        Normalized-efficiency reward
    """
    cfg = config or RewardConfig()

    if prev_metrics.two_qubit_gates == 0:
        relative_improvement = 0.0
    else:
        relative_improvement = (
            prev_metrics.two_qubit_gates - new_metrics.two_qubit_gates
        ) / prev_metrics.two_qubit_gates

    normalized_time = time_cost / max(time_budget, 1e-6)
    return cfg.alpha * relative_improvement - cfg.beta * normalized_time - cfg.gamma


def compute_category_relative_reward(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    category: str,
    category_baselines: dict[str, float],
    time_cost: float,
    config: RewardConfig | None = None,
    optimizer_name: str = "",
) -> float:
    """Reward relative to category average, reducing tket dominance.

    R = alpha * (raw_improvement - category_baseline) - beta * time_cost - gamma

    A positive reward only if the optimizer beats the average for this category.
    When optimizer_name is provided, looks up a finer-grained baseline keyed as
    "{category}:{optimizer_name}" before falling back to the per-category baseline.

    Args:
        prev_metrics: Metrics before optimization
        new_metrics: Metrics after optimization
        category: Circuit category (e.g. "qft", "feynman")
        category_baselines: Mapping from category name (or "category:optimizer") to avg relative improvement
        time_cost: Time spent on optimization in seconds
        config: Reward configuration
        optimizer_name: Optimizer name for finer-grained baseline lookup (optional)

    Returns:
        Category-relative reward
    """
    cfg = config or RewardConfig()

    if prev_metrics.two_qubit_gates == 0:
        raw_improvement = 0.0
    else:
        raw_improvement = (
            prev_metrics.two_qubit_gates - new_metrics.two_qubit_gates
        ) / prev_metrics.two_qubit_gates

    catopt_key = f"{category}:{optimizer_name}" if optimizer_name else ""
    baseline = category_baselines.get(catopt_key, category_baselines.get(category, 0.0))
    relative = raw_improvement - baseline

    return cfg.alpha * relative - cfg.beta * time_cost - cfg.gamma


@dataclass
class RewardSet:
    """Collection of all reward variants for a single step."""

    improvement_only: float
    efficiency: float
    multi_objective: float
    sparse_final: float
    category_relative: float = 0.0
    efficiency_normalized: float = 0.0


def compute_all_rewards(
    prev_metrics: CircuitMetrics,
    new_metrics: CircuitMetrics,
    time_cost: float,
    initial_metrics: CircuitMetrics | None = None,
    is_final_step: bool = False,
    config: RewardConfig | None = None,
    category: str = "unknown",
    category_baselines: dict[str, float] | None = None,
    time_budget: float = 300.0,
    optimizer_name: str = "",
) -> RewardSet:
    """Compute all reward variants for a step.

    Args:
        prev_metrics: Metrics before this step
        new_metrics: Metrics after this step
        time_cost: Time spent on optimization in seconds
        initial_metrics: Metrics at episode start (for sparse reward)
        is_final_step: Whether this is the last step
        config: Reward configuration
        category: Circuit category (for category-relative reward)
        category_baselines: Per-category average improvements (for category-relative reward)

    Returns:
        RewardSet with all reward variants
    """
    cfg = config or RewardConfig()

    return RewardSet(
        improvement_only=compute_improvement_only_reward(prev_metrics, new_metrics, cfg),
        efficiency=compute_efficiency_reward(prev_metrics, new_metrics, time_cost, cfg),
        multi_objective=compute_multi_objective_reward(
            prev_metrics, new_metrics, time_cost, cfg
        ),
        sparse_final=compute_sparse_final_reward(
            initial_metrics or prev_metrics,
            new_metrics,
            is_final_step,
            cfg,
        ),
        category_relative=compute_category_relative_reward(
            prev_metrics, new_metrics, category, category_baselines or {}, time_cost, cfg, optimizer_name
        ),
        efficiency_normalized=compute_efficiency_normalized_reward(
            prev_metrics, new_metrics, time_cost, time_budget, cfg
        ),
    )


def get_default_config() -> RewardConfig:
    """Get the default reward configuration.

    Returns:
        Default RewardConfig
    """
    return RewardConfig()


def compute_improvement_percentage(
    initial_value: int,
    final_value: int,
) -> float:
    """Compute percentage improvement for a metric.

    Args:
        initial_value: Initial metric value
        final_value: Final metric value

    Returns:
        Percentage improvement (positive means reduction)
    """
    if initial_value == 0:
        return 0.0
    return 100.0 * (initial_value - final_value) / initial_value


def summarize_trajectory_rewards(
    step_rewards: list[RewardSet],
) -> dict[str, float]:
    """Summarize rewards across a trajectory.

    Args:
        step_rewards: List of RewardSet for each step

    Returns:
        Dictionary with summary statistics
    """
    if not step_rewards:
        return {
            "total_improvement_only": 0.0,
            "total_efficiency": 0.0,
            "total_multi_objective": 0.0,
            "total_sparse_final": 0.0,
            "total_category_relative": 0.0,
            "total_efficiency_normalized": 0.0,
            "mean_improvement_only": 0.0,
            "mean_efficiency": 0.0,
            "mean_multi_objective": 0.0,
            "mean_category_relative": 0.0,
            "mean_efficiency_normalized": 0.0,
        }

    total_improvement_only = sum(r.improvement_only for r in step_rewards)
    total_efficiency = sum(r.efficiency for r in step_rewards)
    total_multi_objective = sum(r.multi_objective for r in step_rewards)
    total_sparse_final = step_rewards[-1].sparse_final if step_rewards else 0.0
    total_category_relative = sum(r.category_relative for r in step_rewards)
    total_efficiency_normalized = sum(r.efficiency_normalized for r in step_rewards)

    n = len(step_rewards)
    return {
        "total_improvement_only": total_improvement_only,
        "total_efficiency": total_efficiency,
        "total_multi_objective": total_multi_objective,
        "total_sparse_final": total_sparse_final,
        "total_category_relative": total_category_relative,
        "total_efficiency_normalized": total_efficiency_normalized,
        "mean_improvement_only": total_improvement_only / n,
        "mean_efficiency": total_efficiency / n,
        "mean_multi_objective": total_multi_objective / n,
        "mean_category_relative": total_category_relative / n,
        "mean_efficiency_normalized": total_efficiency_normalized / n,
    }
