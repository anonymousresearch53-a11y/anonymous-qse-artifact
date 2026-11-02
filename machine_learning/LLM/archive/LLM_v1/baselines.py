"""Baseline chain selectors for v1 benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import random

import pandas as pd


@dataclass
class BaselineSelection:
    """Selected chain plus metadata for a baseline."""

    method_name: str
    selected_chain_id: str
    selected_chain_name: str
    selected_chain: list[str]
    metadata: dict[str, Any]


def random_valid_chain(scored_candidates: pd.DataFrame, seed: int) -> BaselineSelection:
    """Pick one valid chain uniformly at random."""
    rng = random.Random(int(seed))
    selected = scored_candidates.iloc[rng.randrange(len(scored_candidates))]
    return BaselineSelection(
        method_name="random_valid_chain",
        selected_chain_id=str(selected["chain_id"]),
        selected_chain_name=str(selected["chain_name"]),
        selected_chain=list(selected["chain"]),
        metadata={"seed": int(seed)},
    )


def best_single_optimizer(scored_candidates: pd.DataFrame) -> BaselineSelection:
    """Pick the best single-step chain for one circuit."""
    selected = scored_candidates[scored_candidates["length"] == 1].iloc[0]
    return BaselineSelection(
        method_name="best_single_optimizer",
        selected_chain_id=str(selected["chain_id"]),
        selected_chain_name=str(selected["chain_name"]),
        selected_chain=list(selected["chain"]),
        metadata={},
    )


def global_best_average_chain(train_scored_frame: pd.DataFrame) -> BaselineSelection:
    """Pick the chain with the best average training objective."""
    average_scores = (
        train_scored_frame.groupby(["chain_id", "chain_name"], as_index=False)
        .agg(
            average_objective_score=("objective_score", "mean"),
            chain=("chain", "first"),
        )
        .sort_values(
            ["average_objective_score", "chain_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    selected = average_scores.iloc[0]
    return BaselineSelection(
        method_name="global_best_average_chain",
        selected_chain_id=str(selected["chain_id"]),
        selected_chain_name=str(selected["chain_name"]),
        selected_chain=list(selected["chain"]),
        metadata={"average_objective_score": float(selected["average_objective_score"])},
    )

