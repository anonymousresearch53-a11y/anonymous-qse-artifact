"""Train-only nearest-neighbor retrieval for similar circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from machine_learning.LLM.llm_v3.features import (
        build_query_feature_row,
        build_query_feature_frame,
        extract_query_features,
        retrieval_feature_columns,
    )
except ImportError:
    from .features import (
        build_query_feature_row,
        build_query_feature_frame,
        extract_query_features,
        retrieval_feature_columns,
    )


DEFAULT_RETRIEVAL_K = 3
DEFAULT_TOP_OUTCOMES = 3
DEFAULT_RETRIEVAL_METHOD = "standardized_euclidean_knn"


@dataclass
class RetrievedNeighbor:
    """Compact retrieved neighbor summary for prompting and logging."""

    circuit_id: str
    distance: float
    neighbor_feature_summary: dict[str, float]
    best_chain_id: str
    best_chain_name: str
    best_chain_score: float
    best_non_original_chain_id: str
    best_non_original_chain_name: str
    best_non_original_chain_score: float
    original_score: float
    best_vs_original_margin: float
    top_outcomes: list[dict[str, Any]]


class TrainingCircuitRetriever:
    """Nearest-neighbor retriever restricted to training-split circuits only."""

    def __init__(
        self,
        train_frame: pd.DataFrame,
        feature_subset: str,
        score_lookup_fn: Callable[[str], pd.DataFrame],
        k: int = DEFAULT_RETRIEVAL_K,
        top_outcomes: int = DEFAULT_TOP_OUTCOMES,
    ) -> None:
        if train_frame.empty:
            raise ValueError("Training frame is empty.")
        if int(k) <= 0:
            raise ValueError("k must be a positive integer.")
        if int(top_outcomes) <= 0:
            raise ValueError("top_outcomes must be a positive integer.")

        self.train_frame = train_frame.copy()
        self.feature_subset = str(feature_subset)
        self.feature_columns = retrieval_feature_columns(self.feature_subset)
        self.k = int(k)
        self.top_outcomes = int(top_outcomes)
        self.retrieval_method = DEFAULT_RETRIEVAL_METHOD
        self._score_lookup_fn = score_lookup_fn
        self.search_pool_size = min(max(self.k * 6, self.k), len(train_frame))

        self.train_representatives = build_query_feature_frame(
            self.train_frame,
            feature_subset=self.feature_subset,
        )
        if self.train_representatives.empty:
            raise ValueError("No circuit representatives could be built from training data.")

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in self.train_representatives.columns
        ]
        if missing_columns:
            raise KeyError(
                "Training representatives are missing retrieval columns: "
                f"{missing_columns}"
            )

        self.train_representatives = self.train_representatives.sort_values(
            "original_circuit"
        ).reset_index(drop=True)

        self.scaler = StandardScaler()
        train_matrix = self._frame_to_feature_matrix(self.train_representatives)
        self.scaled_train_matrix = self.scaler.fit_transform(train_matrix)

        self.index = NearestNeighbors(
            n_neighbors=self.search_pool_size,
            metric="euclidean",
        )
        self.index.fit(self.scaled_train_matrix)

    def _frame_to_feature_matrix(self, frame: pd.DataFrame):
        """Convert a feature frame into a clean numeric matrix."""
        return (
            frame[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _query_row_to_matrix(self, query_row: dict[str, Any]):
        """Convert one query row into a single-row numeric matrix."""
        query_frame = pd.DataFrame(
            [
                build_query_feature_row(
                    query_row,
                    feature_subset=self.feature_subset,
                )
            ]
        )
        return (
            query_frame[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _build_top_outcomes(self, scored_rows: pd.DataFrame) -> list[dict[str, Any]]:
        """Build a compact top-outcome evidence payload for one retrieved circuit."""
        if scored_rows.empty:
            raise ValueError("Scored rows were empty for a retrieved circuit.")

        top_rows = scored_rows.head(self.top_outcomes)
        outcomes: list[dict[str, Any]] = []

        for _, row in top_rows.iterrows():
            outcomes.append(
                {
                    "chain_id": str(row["chain_id"]),
                    "chain_name": str(row["chain_name"]),
                    "chain": list(row["chain"]),
                    "score": float(row["objective_score"]),
                    "two_qubit_reduction_pct": float(
                        row["total_two_qubit_gate_reduction_pct"]
                    ),
                    "depth_reduction_pct": float(row["total_depth_reduction_pct"]),
                    "runtime_seconds": float(row["total_runtime_seconds"]),
                    "runtime_score": float(row.get("total_runtime_score", 0.0)),
                }
            )

        return outcomes

    def retrieve(self, query_row: dict[str, Any]) -> list[RetrievedNeighbor]:
        """Return the nearest training circuits for one query circuit."""
        query_matrix = self._query_row_to_matrix(query_row)
        scaled_query = self.scaler.transform(query_matrix)

        distances, indices = self.index.kneighbors(scaled_query)

        raw_neighbors: list[tuple[float, str, pd.Series]] = []
        for distance, index in zip(distances[0], indices[0], strict=True):
            representative = self.train_representatives.iloc[int(index)]
            circuit_id = str(representative["original_circuit"])
            raw_neighbors.append((float(distance), circuit_id, representative))

        raw_neighbors.sort(key=lambda item: (item[0], item[1]))

        neighbors: list[RetrievedNeighbor] = []
        for distance, circuit_id, representative in raw_neighbors[: self.k]:
            scored_rows = self._score_lookup_fn(circuit_id)
            if scored_rows.empty:
                raise ValueError(
                    f"Score lookup returned no rows for retrieved circuit {circuit_id}."
                )

            best_row = scored_rows.iloc[0]
            original_row = scored_rows[scored_rows["length"] == 0].iloc[0]
            best_non_original_rows = scored_rows[scored_rows["length"] > 0]
            best_non_original = (
                best_non_original_rows.iloc[0]
                if not best_non_original_rows.empty
                else original_row
            )
            neighbors.append(
                RetrievedNeighbor(
                    circuit_id=circuit_id,
                    distance=distance,
                    neighbor_feature_summary=extract_query_features(
                        representative.to_dict(),
                        feature_subset=self.feature_subset,
                    ),
                    best_chain_id=str(best_row["chain_id"]),
                    best_chain_name=str(best_row["chain_name"]),
                    best_chain_score=float(best_row["objective_score"]),
                    best_non_original_chain_id=str(best_non_original["chain_id"]),
                    best_non_original_chain_name=str(best_non_original["chain_name"]),
                    best_non_original_chain_score=float(best_non_original["objective_score"]),
                    original_score=float(original_row["objective_score"]),
                    best_vs_original_margin=float(
                        best_row["objective_score"] - original_row["objective_score"]
                    ),
                    top_outcomes=self._build_top_outcomes(scored_rows),
                )
            )

        return neighbors

    def metadata(self) -> dict[str, Any]:
        """Return retrieval metadata for experiment logging."""
        return {
            "retrieval_method": self.retrieval_method,
            "retrieval_k": int(self.k),
            "retrieval_search_pool_size": int(self.search_pool_size),
            "retrieval_top_outcomes": int(self.top_outcomes),
            "retrieval_feature_subset": self.feature_subset,
            "retrieval_feature_columns": list(self.feature_columns),
            "retrieval_feature_count": len(self.feature_columns),
            "train_circuit_count": int(len(self.train_representatives)),
        }
