"""Deterministic evaluation helpers for LLM-based chain selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    from machine_learning.LLM.LLM_v2.data import build_candidate_chains
    from machine_learning.LLM.LLM_v2.objective import (
        ObjectiveConfig,
        score_single_chain,
        score_two_step_chain,
    )
except ImportError:
    from .data import build_candidate_chains
    from .objective import (
        ObjectiveConfig,
        score_single_chain,
        score_two_step_chain,
    )


@dataclass
class CircuitEvaluation:
    """Evaluation record for one selected chain on one circuit."""

    circuit_id: str
    selected_chain_id: str
    selected_chain_name: str
    selected_chain: list[str]
    objective_score: float
    oracle_best_chain_id: str
    oracle_best_chain_name: str
    oracle_best_score: float
    regret: float
    total_two_qubit_gate_reduction_pct: float
    total_depth_reduction_pct: float
    total_runtime_seconds: float
    total_runtime_score: float
    win_vs_best_single_optimizer: bool
    win_vs_global_best_average_chain: bool


def score_circuit_candidates(
    circuit_rows: pd.DataFrame,
    config: ObjectiveConfig,
) -> pd.DataFrame:
    """Score all valid candidate chains for a single original circuit."""
    row_lookup = {
        str(row["chain_name"]): row.to_dict()
        for _, row in circuit_rows.iterrows()
    }

    scored_rows: list[dict[str, Any]] = []

    for candidate in build_candidate_chains():
        chain_name = candidate["chain_name"]
        candidate_row = row_lookup.get(chain_name)
        if candidate_row is None:
            continue

        if candidate["length"] == 1:
            objective_score = score_single_chain(candidate_row, config)
        else:
            first_chain_name = candidate["chain"][0]
            first_row = row_lookup.get(first_chain_name)
            if first_row is None:
                continue
            objective_score = score_two_step_chain(
                final_row=candidate_row,
                first_row=first_row,
                config=config,
            )

        scored_rows.append(
            {
                "chain_id": candidate["chain_id"],
                "chain_name": chain_name,
                "chain": list(candidate["chain"]),
                "length": int(candidate["length"]),
                "objective_score": float(objective_score),
                "total_two_qubit_gate_reduction_pct": float(
                    candidate_row["total_two_qubit_gate_reduction_pct"]
                ),
                "total_depth_reduction_pct": float(
                    candidate_row["total_depth_reduction_pct"]
                ),
                "total_runtime_seconds": float(candidate_row["total_runtime_seconds"]),
                "total_runtime_score": float(
                    candidate_row.get("total_runtime_score", 0.0)
                ),
            }
        )

    scored_frame = pd.DataFrame(scored_rows)
    if scored_frame.empty:
        raise ValueError("No valid candidate rows were scored for this circuit.")

    scored_frame = scored_frame.sort_values(
        by=[
            "objective_score",
            "total_two_qubit_gate_reduction_pct",
            "chain_id",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return scored_frame


def evaluate_selection(
    circuit_id: str,
    scored_candidates: pd.DataFrame,
    selected_chain_id: str,
    global_best_average_chain_id: str,
) -> CircuitEvaluation:
    """Evaluate one selected chain against oracle and baseline references."""
    selected_rows = scored_candidates[
        scored_candidates["chain_id"] == selected_chain_id
    ]
    if selected_rows.empty:
        raise KeyError(
            f"Selected chain id {selected_chain_id} was not found for circuit {circuit_id}"
        )
    selected = selected_rows.iloc[0]

    oracle = scored_candidates.iloc[0]

    single_step_rows = scored_candidates[scored_candidates["length"] == 1]
    if single_step_rows.empty:
        raise ValueError(
            "No single-step candidate rows were available for baseline comparison."
        )
    best_single = single_step_rows.iloc[0]

    global_rows = scored_candidates[
        scored_candidates["chain_id"] == global_best_average_chain_id
    ]
    if global_rows.empty:
        raise KeyError(
            f"Global best average chain id {global_best_average_chain_id} "
            f"was not found for circuit {circuit_id}"
        )
    global_best = global_rows.iloc[0]

    return CircuitEvaluation(
        circuit_id=str(circuit_id),
        selected_chain_id=str(selected["chain_id"]),
        selected_chain_name=str(selected["chain_name"]),
        selected_chain=list(selected["chain"]),
        objective_score=float(selected["objective_score"]),
        oracle_best_chain_id=str(oracle["chain_id"]),
        oracle_best_chain_name=str(oracle["chain_name"]),
        oracle_best_score=float(oracle["objective_score"]),
        regret=float(oracle["objective_score"] - selected["objective_score"]),
        total_two_qubit_gate_reduction_pct=float(
            selected["total_two_qubit_gate_reduction_pct"]
        ),
        total_depth_reduction_pct=float(selected["total_depth_reduction_pct"]),
        total_runtime_seconds=float(selected["total_runtime_seconds"]),
        total_runtime_score=float(selected["total_runtime_score"]),
        win_vs_best_single_optimizer=bool(
            float(selected["objective_score"]) >= float(best_single["objective_score"])
        ),
        win_vs_global_best_average_chain=bool(
            float(selected["objective_score"]) >= float(global_best["objective_score"])
        ),
    )
