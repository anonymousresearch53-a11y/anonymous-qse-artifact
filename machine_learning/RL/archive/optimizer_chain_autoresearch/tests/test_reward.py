"""Tests for editable reward calculations and metric support."""

from __future__ import annotations

import pytest

from machine_learning.RL.optimizer_chain_autoresearch.reward import (
    compute_reward,
    metric_improvements,
)


INITIAL = {
    "two_qubit_gates": 100.0,
    "depth": 200.0,
    "total_gates": 400.0,
    "two_qubit_depth": 80.0,
}
CURRENT = {
    "two_qubit_gates": 80.0,
    "depth": 150.0,
    "total_gates": 300.0,
    "two_qubit_depth": 60.0,
}


def _config(**overrides):
    config = {
        "family": "weighted_dense",
        "metric_weights": {
            "two_qubit_gates": 1.0,
            "depth": 1.0,
            "total_gates": 1.0,
            "two_qubit_depth": 1.0,
        },
        "reference": "previous",
        "runtime_weight": 0.0,
        "step_penalty": 0.0,
        "terminal_bonus": 0.0,
        "failure_penalty": -2.0,
        "invalid_action_penalty": -3.0,
        "degradation_multiplier": 1.0,
        "reward_clip": None,
    }
    config.update(overrides)
    return config


def test_all_quality_metrics_are_percentage_based():
    """Each supported quality dimension is normalized by its own initial size."""

    improvements = metric_improvements(INITIAL, CURRENT)
    assert improvements == pytest.approx(
        {
            "two_qubit_gates": 0.20,
            "depth": 0.25,
            "total_gates": 0.25,
            "two_qubit_depth": 0.25,
        }
    )


def test_dense_and_sparse_rewards():
    """Sparse quality is withheld until terminal while dense quality is immediate."""

    kwargs = dict(
        previous_metrics=INITIAL,
        current_metrics=CURRENT,
        initial_metrics=INITIAL,
        step_runtime_seconds=1.0,
        total_runtime_seconds=1.0,
        runtime_scale=100.0,
        step_index=0,
        failed=False,
        invalid_action=False,
    )
    dense = compute_reward(
        **kwargs, terminal=False, config=_config(family="weighted_dense")
    )
    sparse_nonterminal = compute_reward(
        **kwargs, terminal=False, config=_config(family="weighted_terminal")
    )
    sparse_terminal = compute_reward(
        **kwargs, terminal=True, config=_config(family="weighted_terminal")
    )
    assert dense == pytest.approx(0.2375)
    assert sparse_nonterminal == 0.0
    assert sparse_terminal == pytest.approx(0.2375)


def test_failure_and_invalid_action_penalties_take_precedence():
    """Broken actions cannot earn quality credit from malformed result rows."""

    common = dict(
        previous_metrics=INITIAL,
        current_metrics=CURRENT,
        initial_metrics=INITIAL,
        step_runtime_seconds=0.0,
        total_runtime_seconds=0.0,
        runtime_scale=100.0,
        step_index=0,
        terminal=True,
        config=_config(),
    )
    assert compute_reward(**common, failed=True, invalid_action=False) == -2.0
    assert compute_reward(**common, failed=False, invalid_action=True) == -3.0
