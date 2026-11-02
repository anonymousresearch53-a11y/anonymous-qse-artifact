"""Tests for duplicate hashes, champions, and Pareto promotion."""

from __future__ import annotations

from copy import deepcopy

from machine_learning.RL.optimizer_chain_autoresearch.archive import (
    dominates,
    experiment_fingerprint,
    promote_finalist,
)
from machine_learning.RL.optimizer_chain_autoresearch.evaluate import objective_tracks


def _aggregate(score: float):
    objectives = {track_id: score for track_id in objective_tracks()}
    per_seed = {
        str(seed): {track_id: score for track_id in objectives}
        for seed in range(42, 47)
    }
    return {
        "seeds": list(range(42, 47)),
        "objective_scores": objectives,
        "per_seed_objectives": per_seed,
        "pareto_vector": {
            "two_qubit_gates": score,
            "depth": score,
            "total_gates": score,
            "two_qubit_depth": score,
            "runtime_normalized": 1.0 - score,
            "failure_rate": 0.0,
        },
    }


def test_fingerprint_changes_with_compute_contract(experiment_config):
    """Stage, seeds, and budget are part of exact duplicate detection."""

    sources = {"train.py": "same"}
    first = experiment_fingerprint(
        editable_sources=sources,
        config=experiment_config,
        stage="screen",
        seeds=[42],
        timesteps=10,
    )
    second = experiment_fingerprint(
        editable_sources=sources,
        config=experiment_config,
        stage="screen",
        seeds=[42],
        timesteps=11,
    )
    assert first != second


def test_pareto_dominance_uses_quality_runtime_and_failures():
    """Higher quality and lower costs must all be no worse to dominate."""

    left = {
        "two_qubit_gates": 0.2,
        "depth": 0.2,
        "total_gates": 0.2,
        "two_qubit_depth": 0.2,
        "runtime_normalized": 0.5,
        "failure_rate": 0.0,
    }
    right = dict(left, depth=0.1, runtime_normalized=0.6)
    assert dominates(left, right)
    assert not dominates(right, left)


def test_first_finalist_populates_all_champions(experiment_config):
    """A five-seed first model owns 75 tracks and enters Pareto storage."""

    champions = {
        "objective_tracks": {},
        "reward_families": {},
        "reward_signatures": {},
    }
    pareto = []
    promotion = promote_finalist(
        experiment_id="baseline",
        aggregate=_aggregate(0.1),
        model_paths=["one.zip", "two.zip"],
        config=deepcopy(experiment_config),
        champions=champions,
        pareto_entries=pareto,
    )
    assert promotion["retained"]
    assert len(promotion["won_objective_tracks"]) == 75
    assert len(champions["objective_tracks"]) == 75
    assert pareto[0]["experiment_id"] == "baseline"


def test_challenger_requires_seed_majority(experiment_config):
    """Aggregate score alone cannot replace an incumbent without 3/5 seed wins."""

    champions = {
        "objective_tracks": {},
        "reward_families": {},
        "reward_signatures": {},
    }
    pareto = []
    promote_finalist(
        experiment_id="baseline",
        aggregate=_aggregate(0.1),
        model_paths=["baseline.zip"],
        config=experiment_config,
        champions=champions,
        pareto_entries=pareto,
    )
    challenger = _aggregate(0.2)
    # Lose three seeds while retaining an artificially high aggregate field.
    for seed in ("42", "43", "44"):
        for track_id in challenger["per_seed_objectives"][seed]:
            challenger["per_seed_objectives"][seed][track_id] = 0.0
    promote_finalist(
        experiment_id="challenger",
        aggregate=challenger,
        model_paths=["challenger.zip"],
        config=experiment_config,
        champions=champions,
        pareto_entries=pareto,
    )
    owners = {
        entry["experiment_id"] for entry in champions["objective_tracks"].values()
    }
    assert owners == {"baseline"}
