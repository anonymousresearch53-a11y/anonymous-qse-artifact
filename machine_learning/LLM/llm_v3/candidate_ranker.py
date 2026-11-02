"""Learned candidate ranking for llm_v3 shortlist generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from machine_learning.LLM.llm_v3.data import (
        ORIGINAL_CHAIN_ID,
        OPTIMIZER_NAMES,
        build_candidate_chains,
    )
    from machine_learning.LLM.llm_v3.features import (
        build_query_feature_frame,
        build_query_feature_row,
        feature_subset_columns,
    )
except ImportError:
    from .data import (
        ORIGINAL_CHAIN_ID,
        OPTIMIZER_NAMES,
        build_candidate_chains,
    )
    from .features import (
        build_query_feature_frame,
        build_query_feature_row,
        feature_subset_columns,
    )


DEFAULT_RANKER_VERSION = "rf_candidate_ranker_v1"


@dataclass
class CandidateRankerResult:
    """Ranking output for one query circuit."""

    ranked_chain_ids: list[str]
    chain_scores: dict[str, float]


class LearnedCandidateRanker:
    """Train a circuit-conditioned scorer for candidate chains."""

    def __init__(
        self,
        feature_subset: str,
        candidate_profile_lookup: dict[str, dict[str, Any]],
        random_state: int = 0,
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
    ) -> None:
        self.feature_subset = str(feature_subset)
        self.query_feature_columns = feature_subset_columns(self.feature_subset)
        self.candidate_profile_lookup = {
            str(chain_id): dict(profile)
            for chain_id, profile in candidate_profile_lookup.items()
        }
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.version = DEFAULT_RANKER_VERSION
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._feature_columns: list[str] | None = None
        self._candidate_feature_frame = self._build_candidate_feature_frame()
        self._is_fitted = False
        self.training_rows = 0
        self.training_circuits = 0

    def _build_candidate_feature_frame(self) -> pd.DataFrame:
        """Build static candidate-side features for every chain."""
        rows: list[dict[str, Any]] = []
        for candidate in build_candidate_chains():
            row: dict[str, Any] = {
                "chain_id": str(candidate["chain_id"]),
                "chain_id_token": str(candidate["chain_id"]),
                "length": int(candidate["length"]),
                "is_original": int(candidate["chain_id"] == ORIGINAL_CHAIN_ID),
                "same_optimizer_twice": int(
                    len(candidate["chain"]) == 2 and candidate["chain"][0] == candidate["chain"][1]
                ),
            }
            first_optimizer = candidate["chain"][0] if len(candidate["chain"]) >= 1 else "__none__"
            second_optimizer = candidate["chain"][1] if len(candidate["chain"]) >= 2 else "__none__"
            row["first_optimizer"] = first_optimizer
            row["second_optimizer"] = second_optimizer

            for optimizer in OPTIMIZER_NAMES:
                row[f"contains_{optimizer}"] = int(optimizer in candidate["chain"])
                row[f"first_is_{optimizer}"] = int(first_optimizer == optimizer)
                row[f"second_is_{optimizer}"] = int(second_optimizer == optimizer)

            profile = self.candidate_profile_lookup.get(str(candidate["chain_id"]), {})
            row["profile_avg_objective_score"] = float(profile.get("avg_objective_score", 0.0))
            row["profile_avg_two_qubit_gate_reduction_pct"] = float(
                profile.get("avg_two_qubit_gate_reduction_pct", 0.0)
            )
            row["profile_avg_depth_reduction_pct"] = float(
                profile.get("avg_depth_reduction_pct", 0.0)
            )
            row["profile_avg_runtime_seconds"] = float(
                profile.get("avg_runtime_seconds", 0.0)
            )
            row["profile_avg_runtime_score"] = float(
                profile.get("avg_runtime_score", 0.0)
            )
            row["profile_top_1_rate"] = float(profile.get("top_1_rate", 0.0))
            row["profile_objective_rank"] = float(profile.get("objective_rank", 999.0))
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame = pd.get_dummies(
            frame,
            columns=["chain_id_token", "first_optimizer", "second_optimizer"],
            dtype=float,
        )
        return frame

    def _build_feature_frame(
        self,
        query_frame: pd.DataFrame,
        scored_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Cross query-side features with static candidate features."""
        query_side = query_frame[["original_circuit", *self.query_feature_columns]].copy()
        query_side["__merge_key"] = 1
        candidate_side = self._candidate_feature_frame.copy()
        candidate_side["__merge_key"] = 1
        full_frame = query_side.merge(candidate_side, on="__merge_key", how="inner").drop(
            columns="__merge_key"
        )
        if scored_frame is not None:
            full_frame = full_frame.merge(
                scored_frame[
                    [
                        "original_circuit",
                        "chain_id",
                        "objective_score",
                    ]
                ],
                on=["original_circuit", "chain_id"],
                how="inner",
            )
        return full_frame

    def fit(
        self,
        train_frame: pd.DataFrame,
        train_scored: pd.DataFrame,
    ) -> None:
        """Fit the learned ranker on train circuits and scored actions."""
        train_representatives = build_query_feature_frame(
            train_frame,
            feature_subset=self.feature_subset,
        )
        training_frame = self._build_feature_frame(
            query_frame=train_representatives,
            scored_frame=train_scored,
        )
        if training_frame.empty:
            raise ValueError("Training frame for the candidate ranker was empty.")

        X = training_frame.drop(columns=["original_circuit", "chain_id", "objective_score"])
        y = training_frame["objective_score"].astype(float)
        self._feature_columns = list(X.columns)
        self.model.fit(X, y)
        self._is_fitted = True
        self.training_rows = int(len(training_frame))
        self.training_circuits = int(train_representatives["original_circuit"].nunique())

    def score_query(self, query_row: dict[str, Any]) -> CandidateRankerResult:
        """Score all candidate chains for one query circuit."""
        if not self._is_fitted or self._feature_columns is None:
            raise RuntimeError("LearnedCandidateRanker must be fit before scoring.")

        query_frame = pd.DataFrame(
            [
                build_query_feature_row(
                    query_row,
                    feature_subset=self.feature_subset,
                )
            ]
        )
        feature_frame = self._build_feature_frame(query_frame=query_frame)
        X = feature_frame.drop(columns=["original_circuit", "chain_id"])
        X = X.reindex(columns=self._feature_columns, fill_value=0.0)
        scores = self.model.predict(X)

        scored = pd.DataFrame(
            {
                "chain_id": feature_frame["chain_id"].astype(str),
                "predicted_score": scores.astype(float),
            }
        ).sort_values(["predicted_score", "chain_id"], ascending=[False, True])

        return CandidateRankerResult(
            ranked_chain_ids=scored["chain_id"].tolist(),
            chain_scores={
                str(row["chain_id"]): float(row["predicted_score"])
                for _, row in scored.iterrows()
            },
        )

    def metadata(self) -> dict[str, Any]:
        """Return training metadata for experiment logging."""
        return {
            "shortlist_ranker_version": self.version,
            "shortlist_ranker_model": "RandomForestRegressor",
            "shortlist_ranker_feature_subset": self.feature_subset,
            "shortlist_ranker_query_feature_columns": list(self.query_feature_columns),
            "shortlist_ranker_training_rows": int(self.training_rows),
            "shortlist_ranker_training_circuits": int(self.training_circuits),
            "shortlist_ranker_n_estimators": int(self.n_estimators),
            "shortlist_ranker_min_samples_leaf": int(self.min_samples_leaf),
        }
