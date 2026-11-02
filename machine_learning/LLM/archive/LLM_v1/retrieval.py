"""Numeric nearest-neighbor retrieval over training circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import candidate_rows_for_circuit, circuit_representatives
from .features import extract_query_features, retrieval_feature_columns


DEFAULT_RETRIEVAL_K = 5
DEFAULT_TOP_OUTCOMES = 3
DEFAULT_RETRIEVAL_METHOD = "standardized_numeric_knn"


@dataclass
class RetrievedNeighbor:
    """Compact retrieved neighbor summary for prompt use and logging."""

    circuit_id: str
    distance: float
    query_feature_summary: dict[str, Any]
    best_chain_id: str
    best_chain_name: str
    best_chain_score: float
    top_outcomes: list[dict[str, Any]]


class TrainingCircuitRetriever:
    """Feature-space nearest-neighbor retriever restricted to the training split."""

    def __init__(
        self,
        train_frame: pd.DataFrame,
        feature_subset: str,
        score_lookup_fn,
        k: int = DEFAULT_RETRIEVAL_K,
    ) -> None:
        self.train_frame = train_frame.copy()
        self.feature_subset = str(feature_subset)
        self.feature_columns = retrieval_feature_columns(self.feature_subset)
        self.k = int(k)
        self.retrieval_method = DEFAULT_RETRIEVAL_METHOD
        self._score_lookup_fn = score_lookup_fn

        self.train_representatives = circuit_representatives(self.train_frame)
        self.scaler = StandardScaler()
        train_matrix = (
            self.train_representatives[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        self.scaled_train_matrix = self.scaler.fit_transform(train_matrix)
        self.index = NearestNeighbors(n_neighbors=min(self.k, len(self.train_representatives)))
        self.index.fit(self.scaled_train_matrix)

    def retrieve(self, query_row: dict[str, Any]) -> list[RetrievedNeighbor]:
        """Return top-k similar circuits from the training split only."""
        query_frame = pd.DataFrame([{column: query_row.get(column) for column in self.feature_columns}])
        query_matrix = query_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scaled_query = self.scaler.transform(query_matrix)
        distances, indices = self.index.kneighbors(scaled_query)

        neighbors: list[RetrievedNeighbor] = []
        for distance, index in zip(distances[0], indices[0], strict=True):
            representative = self.train_representatives.iloc[int(index)]
            circuit_id = str(representative["original_circuit"])
            score_rows = self._score_lookup_fn(circuit_id)
            best_row = score_rows.iloc[0]
            top_outcomes = []
            for _, row in score_rows.head(DEFAULT_TOP_OUTCOMES).iterrows():
                top_outcomes.append(
                    {
                        "chain_id": row["chain_id"],
                        "chain_name": row["chain_name"],
                        "chain": row["chain"],
                        "score": float(row["objective_score"]),
                        "two_qubit_reduction_pct": float(row["total_two_qubit_gate_reduction_pct"]),
                        "depth_reduction_pct": float(row["total_depth_reduction_pct"]),
                        "runtime_seconds": float(row["total_runtime_seconds"]),
                        "runtime_score": float(row.get("total_runtime_score", 0.0)),
                    }
                )
            neighbors.append(
                RetrievedNeighbor(
                    circuit_id=circuit_id,
                    distance=float(distance),
                    query_feature_summary=extract_query_features(
                        representative.to_dict(),
                        feature_subset=self.feature_subset,
                    ),
                    best_chain_id=str(best_row["chain_id"]),
                    best_chain_name=str(best_row["chain_name"]),
                    best_chain_score=float(best_row["objective_score"]),
                    top_outcomes=top_outcomes,
                )
            )
        return neighbors

    def metadata(self) -> dict[str, Any]:
        """Return retrieval metadata for logging."""
        return {
            "retrieval_method": self.retrieval_method,
            "retrieval_k": int(self.k),
            "retrieval_feature_columns": list(self.feature_columns),
            "retrieval_feature_count": len(self.feature_columns),
        }

