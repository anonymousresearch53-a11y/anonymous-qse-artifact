"""Learned optimize-vs-original gate for llm_v3."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from machine_learning.LLM.llm_v3.data import ORIGINAL_CHAIN_ID
    from machine_learning.LLM.llm_v3.features import (
        build_query_feature_frame,
        build_query_feature_row,
        feature_subset_columns,
    )
except ImportError:
    from .data import ORIGINAL_CHAIN_ID
    from .features import (
        build_query_feature_frame,
        build_query_feature_row,
        feature_subset_columns,
    )


DEFAULT_GATE_VERSION = "rf_optimize_gate_v1"


@dataclass
class GateDecision:
    """Gate output for one circuit."""

    recommend_optimization: bool
    probability_original: float
    probability_optimize: float
    recommendation_label: str


class OptimizationGate:
    """Predict whether the model should optimize or keep the original circuit."""

    def __init__(
        self,
        feature_subset: str,
        random_state: int = 0,
        n_estimators: int = 250,
        min_samples_leaf: int = 2,
    ) -> None:
        self.feature_subset = str(feature_subset)
        self.query_feature_columns = feature_subset_columns(self.feature_subset)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.version = DEFAULT_GATE_VERSION
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._feature_columns: list[str] | None = None
        self.training_rows = 0
        self.training_circuits = 0

    def _build_query_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build the feature frame used by the classifier."""
        return build_query_feature_frame(
            frame,
            feature_subset=self.feature_subset,
        )

    def fit(self, train_frame: pd.DataFrame, train_scored: pd.DataFrame) -> None:
        """Fit the gate from train circuits."""
        query_frame = self._build_query_frame(train_frame)
        oracle_rows = (
            train_scored.sort_values(
                ["original_circuit", "objective_score", "chain_id"],
                ascending=[True, False, True],
            )
            .groupby("original_circuit", as_index=False)
            .first()[["original_circuit", "chain_id"]]
        )
        oracle_rows["should_optimize"] = (
            oracle_rows["chain_id"].astype(str) != ORIGINAL_CHAIN_ID
        ).astype(int)
        full = query_frame.merge(oracle_rows, on="original_circuit", how="inner")
        X = full.drop(columns=["original_circuit", "chain_id", "should_optimize"])
        y = full["should_optimize"].astype(int)
        self._feature_columns = list(X.columns)
        self.model.fit(X, y)
        self.training_rows = int(len(full))
        self.training_circuits = int(full["original_circuit"].nunique())

    def predict_query(self, query_row: dict[str, float]) -> GateDecision:
        """Return gate probabilities for one query circuit."""
        if self._feature_columns is None:
            raise RuntimeError("OptimizationGate must be fit before prediction.")

        query_frame = pd.DataFrame(
            [
                build_query_feature_row(
                    query_row,
                    feature_subset=self.feature_subset,
                )
            ]
        )
        X = query_frame.drop(columns=["original_circuit"]).reindex(
            columns=self._feature_columns,
            fill_value=0.0,
        )
        probabilities = self.model.predict_proba(X)[0]
        class_index = {
            int(class_value): idx for idx, class_value in enumerate(self.model.classes_)
        }
        probability_optimize = (
            float(probabilities[class_index[1]]) if 1 in class_index else 0.0
        )
        probability_original = (
            float(probabilities[class_index[0]]) if 0 in class_index else 0.0
        )
        if 1 not in class_index:
            probability_original = 1.0
        if 0 not in class_index:
            probability_optimize = 1.0
        recommend_optimization = probability_optimize >= probability_original
        return GateDecision(
            recommend_optimization=recommend_optimization,
            probability_original=probability_original,
            probability_optimize=probability_optimize,
            recommendation_label="optimize" if recommend_optimization else "original",
        )

    def metadata(self) -> dict[str, float | int | str | list[str]]:
        """Return gate training metadata."""
        return {
            "gate_version": self.version,
            "gate_model": "RandomForestClassifier",
            "gate_feature_subset": self.feature_subset,
            "gate_query_feature_columns": list(self.query_feature_columns),
            "gate_training_rows": int(self.training_rows),
            "gate_training_circuits": int(self.training_circuits),
            "gate_n_estimators": int(self.n_estimators),
            "gate_min_samples_leaf": int(self.min_samples_leaf),
        }
