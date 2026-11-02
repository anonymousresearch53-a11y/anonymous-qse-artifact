"""Tests for locked percentage evaluation, objectives, and baselines."""

from __future__ import annotations

import pandas as pd
import pytest

from machine_learning.RL.optimizer_chain_autoresearch.evaluate import (
    compute_baseline_scores,
    objective_tracks,
    score_objectives,
    selected_rows_to_outcomes,
    summarize_metrics,
)


def test_exactly_75_objective_tracks():
    """Fifteen quality subsets times five runtime settings equals 75."""

    tracks = objective_tracks()
    assert len(tracks) == 75
    assert "quality:two_qubit_gates|runtime:0.00" in tracks
    assert (
        "quality:two_qubit_gates+depth+total_gates+two_qubit_depth|runtime:0.35"
        in tracks
    )


def test_macro_percentage_prevents_large_circuit_dominance(data_bundle):
    """Each circuit contributes one percentage, not a raw gate-count total."""

    rows = data_bundle.validation.groupby("original_circuit").head(1).head(2).copy()
    rows["final_two_qubit_gates"] = rows["final_two_qubit_gates"].astype(float)
    rows["selected_chain"] = rows["chain_name"]
    rows["invalid_action"] = False
    rows["failed"] = False
    rows.loc[rows.index[0], "final_two_qubit_gates"] = (
        rows.loc[rows.index[0], "initial_two_qubit_gates"] * 0.5
    )
    rows.loc[rows.index[1], "final_two_qubit_gates"] = (
        rows.loc[rows.index[1], "initial_two_qubit_gates"] * 0.9
    )
    outcomes = selected_rows_to_outcomes(rows, data_bundle.statistics)
    assert outcomes["raw_two_qubit_gates"].mean() == pytest.approx(0.30)
    summary = summarize_metrics(outcomes)
    assert summary["quality_metrics"]["two_qubit_gates"]["raw_mean"] == pytest.approx(
        0.30
    )
    assert len(score_objectives(outcomes)) == 75


def test_failure_has_zero_quality_and_locked_penalty(data_bundle):
    """A failed source row cannot retain a positive metric outcome."""

    row = data_bundle.validation.iloc[[0]].copy()
    row["selected_chain"] = row["chain_name"]
    row["invalid_action"] = False
    row["failed"] = True
    row["final_two_qubit_gates"] = 0
    outcomes = selected_rows_to_outcomes(row, data_bundle.statistics)
    assert outcomes.iloc[0]["raw_two_qubit_gates"] == 0.0
    track = "quality:two_qubit_gates|runtime:0.00"
    assert score_objectives(outcomes)[track] == -1.0


def test_baselines_exist_for_every_track(data_bundle):
    """Every objective receives all six independently recomputed baselines."""

    baselines = compute_baseline_scores(
        train_frame=data_bundle.train,
        evaluation_frame=data_bundle.validation,
        statistics=data_bundle.statistics,
    )
    assert len(baselines) == 75
    expected = {
        "no_optimization",
        "random_expected",
        "tket__wisq_bqskit",
        "best_fixed_single",
        "best_fixed_chain",
        "validation_oracle",
    }
    assert all(set(values) == expected for values in baselines.values())
