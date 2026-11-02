"""Training-set profile helpers for candidate-chain prompting and routing."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from machine_learning.LLM.llm_v3.data import ORIGINAL_CHAIN_NAME
except ImportError:
    from .data import ORIGINAL_CHAIN_NAME


def build_chain_profiles(train_scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate behavior across the training split."""
    if train_scored.empty:
        raise ValueError("train_scored must be non-empty.")

    grouped = (
        train_scored.groupby(["chain_id", "chain_name", "length"], as_index=False)
        .agg(
            avg_objective_score=("objective_score", "mean"),
            avg_two_qubit_gate_reduction_pct=(
                "total_two_qubit_gate_reduction_pct",
                "mean",
            ),
            avg_depth_reduction_pct=("total_depth_reduction_pct", "mean"),
            avg_runtime_seconds=("total_runtime_seconds", "mean"),
            avg_runtime_score=("total_runtime_score", "mean"),
            top_1_rate=("objective_score", lambda values: 0.0),
            circuit_count=("objective_score", "count"),
        )
    )

    top_rows = (
        train_scored.sort_values(
            ["original_circuit", "objective_score", "chain_id"],
            ascending=[True, False, True],
        )
        .groupby("original_circuit", as_index=False)
        .first()[["original_circuit", "chain_id"]]
    )
    top_counts = top_rows["chain_id"].value_counts().to_dict()
    total_circuits = max(1, int(top_rows["original_circuit"].nunique()))
    grouped["top_1_rate"] = grouped["chain_id"].map(
        lambda chain_id: float(top_counts.get(chain_id, 0)) / total_circuits
    )

    grouped = grouped.sort_values(
        ["avg_objective_score", "avg_two_qubit_gate_reduction_pct", "chain_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    grouped["objective_rank"] = grouped.index + 1
    return grouped


def chain_profile_lookup(chain_profiles: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return chain profiles keyed by stable chain id."""
    if chain_profiles.empty:
        return {}
    return {
        str(row["chain_id"]): row.to_dict()
        for _, row in chain_profiles.iterrows()
    }


def default_optimizer_docs(chain_profiles: pd.DataFrame) -> dict[str, str]:
    """Build prompt-facing single-optimizer descriptions from train averages."""
    single_rows = chain_profiles[
        (chain_profiles["length"] == 1)
        & (chain_profiles["chain_name"].astype(str) != ORIGINAL_CHAIN_NAME)
    ].copy()
    if single_rows.empty:
        return {}

    docs: dict[str, str] = {}
    for _, row in single_rows.iterrows():
        optimizer_name = str(row["chain_name"])
        docs[optimizer_name] = (
            f"Train avg objective {float(row['avg_objective_score']):.3f}; "
            f"avg 2Q reduction {float(row['avg_two_qubit_gate_reduction_pct']):.1f}%; "
            f"avg depth reduction {float(row['avg_depth_reduction_pct']):.1f}%; "
            f"avg runtime {float(row['avg_runtime_seconds']):.2f}s; "
            f"best overall on {100.0 * float(row['top_1_rate']):.1f}% of train circuits."
        )
    return docs


def top_chain_ids_by_average(
    chain_profiles: pd.DataFrame,
    limit: int,
    length: int | None = None,
) -> list[str]:
    """Return the top train-average chain ids in stable order."""
    rows = chain_profiles.copy()
    if length is not None:
        rows = rows[rows["length"] == int(length)].copy()
    rows = rows.sort_values(
        ["avg_objective_score", "avg_two_qubit_gate_reduction_pct", "chain_id"],
        ascending=[False, False, True],
    )
    return rows["chain_id"].astype(str).head(int(limit)).tolist()
