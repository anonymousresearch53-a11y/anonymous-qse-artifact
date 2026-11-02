"""Editable experiment declaration for autonomous optimizer-chain research.

EDITABLE BY THE RESEARCH AGENT.

This module deliberately keeps the current experiment in ordinary Python data
structures.  The agent may tune these values or change :mod:`train`,
:mod:`reward`, and :mod:`optimizer_chain_env` when a hypothesis requires more
than configuration changes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# A short, falsifiable statement is stored with every run.  The future
# ``program.md`` instructions should require the agent to update this before
# launching an experiment.
HYPOTHESIS = (
    "A narrower 64-by-64 SiLU policy and value network will reduce overfitting "
    "and seed variance while preserving the new anchor's smooth features, "
    "improving the balanced validation track."
)


# ---------------------------------------------------------------------------
# Algorithm and model architecture
# ---------------------------------------------------------------------------

ALGORITHM = "ppo"  # Supported in v1: "ppo", "dqn", and "a2c".

ARCHITECTURE: dict[str, Any] = {
    # PPO/A2C use separate policy and value networks.  DQN uses ``q_network``.
    "policy_network": [64, 64],
    "value_network": [64, 64],
    "q_network": [128, 128],
    "activation": "silu",  # relu, tanh, elu, leaky_relu, or silu
}

ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "ppo": {
        "learning_rate": 1e-3,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 64,
        "gamma": 1.0,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": False,
    },
}


# ---------------------------------------------------------------------------
# Observation and environment behavior
# ---------------------------------------------------------------------------

OBSERVATION_CONFIG: dict[str, Any] = {
    "metrics": [
        "num_qubits",
        "depth",
        "two_qubit_gates",
        "total_gates",
        "two_qubit_depth",
        "gate_density",
        "two_qubit_ratio",
    ],
    "include_categories": True,
    "include_previous_optimizer": True,
    "include_step_features": True,
    "log_scale_count_metrics": False,
    "metric_clip": 5.0,
}

ENVIRONMENT_CONFIG: dict[str, Any] = {
    "shuffle_each_epoch": True,
    "invalid_stop_penalty": -1.0,
    # The source table contains one- and two-optimizer chains, so two is the
    # largest horizon that can be evaluated without synthesizing new outcomes.
    "max_optimizer_steps": 2,
}


# ---------------------------------------------------------------------------
# Editable training reward
# ---------------------------------------------------------------------------

REWARD_CONFIG: dict[str, Any] = {
    "family": "weighted_terminal",
    "metric_weights": {
        "two_qubit_gates": 1.0,
        "depth": 1.0,
        "total_gates": 1.0,
        "two_qubit_depth": 1.0,
    },
    "reference": "initial",  # "previous" for dense credit or "initial".
    "runtime_weight": 0.10,
    "step_penalty": 0.0,
    "terminal_bonus": 0.0,
    "failure_penalty": -1.0,
    "invalid_action_penalty": -1.0,
    "degradation_multiplier": 1.0,
    "reward_clip": None,
}


# The exact reward signature receives its own champion.  The primary objective
# is also used when comparing models within a broader reward family.
PRIMARY_OBJECTIVE: dict[str, Any] = {
    "metrics": [
        "two_qubit_gates",
        "depth",
        "total_gates",
        "two_qubit_depth",
    ],
    "runtime_weight": 0.10,
}


def get_experiment_config() -> dict[str, Any]:
    """Return a detached, serializable copy of the active experiment.

    Returns:
        A dictionary safe for mutation by callers and for JSON serialization.

    Raises:
        ValueError: If the selected algorithm is unsupported or the primary
            objective does not name at least one quality metric.
    """

    algorithm = ALGORITHM.lower()
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"Unsupported algorithm {ALGORITHM!r}; choose ppo, dqn, or a2c."
        )
    if not PRIMARY_OBJECTIVE.get("metrics"):
        raise ValueError("PRIMARY_OBJECTIVE must contain at least one metric.")

    return deepcopy(
        {
            "hypothesis": HYPOTHESIS,
            "algorithm": algorithm,
            "architecture": ARCHITECTURE,
            "algorithm_config": ALGORITHM_CONFIGS[algorithm],
            "observation": OBSERVATION_CONFIG,
            "environment": ENVIRONMENT_CONFIG,
            "reward": REWARD_CONFIG,
            "primary_objective": PRIMARY_OBJECTIVE,
        }
    )
