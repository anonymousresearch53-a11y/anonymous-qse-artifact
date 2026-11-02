"""Feature subset definitions for prompt construction and retrieval."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from machine_learning.LLM.llm_v3.data import circuit_representatives
except ImportError:
    from .data import circuit_representatives


FEATURE_SUBSET_VERSION = "v1"
DEFAULT_FEATURE_SUBSET = "full_structural"


BASIC_SIZE_FEATURES = [
    "initial_num_qubits",
    "initial_depth",
    "initial_two_qubit_gates",
    "initial_total_gates",
    "initial_two_qubit_depth",
    "initial_gate_density",
    "initial_two_qubit_ratio",
]


SIZE_PLUS_GATE_COMPOSITION_FEATURES = BASIC_SIZE_FEATURES + [
    "initial_rz_gate_fraction",
    "initial_sx_gate_fraction",
    "initial_x_gate_fraction",
    "initial_cx_gate_fraction",
    "initial_two_qubit_gate_fraction",
]


FULL_STRUCTURAL_FEATURES = SIZE_PLUS_GATE_COMPOSITION_FEATURES + [
    "initial_num_interaction_edges",
    "initial_avg_interaction_degree",
    "initial_max_interaction_degree",
    "initial_active_interaction_qubits",
    "initial_active_interaction_qubit_fraction",
    "initial_depth_per_qubit",
    "initial_gates_per_qubit",
    "initial_two_qubit_gates_per_qubit",
]

RL_V3_DYNAMIC_PROGRESS_FEATURES = [
    "depth_reduction_so_far",
    "depth_reduction_so_far_fraction",
    "two_qubit_gate_reduction_so_far",
    "two_qubit_gate_reduction_so_far_fraction",
]

# RL v3 plus_all_features at the initial decision point:
# initial-state structural features plus the progress features, which are all
# zero before any optimizer has been applied.
RL_V3_ALL_FEATURES = FULL_STRUCTURAL_FEATURES + RL_V3_DYNAMIC_PROGRESS_FEATURES


FEATURE_SUBSETS = {
    "basic_size": BASIC_SIZE_FEATURES,
    "size_plus_gate_composition": SIZE_PLUS_GATE_COMPOSITION_FEATURES,
    "full_structural": FULL_STRUCTURAL_FEATURES,
    "rl_v3_all_features": RL_V3_ALL_FEATURES,
}

FEATURE_SUBSET_ALIASES = {
    "all_features": "rl_v3_all_features",
    "plus_all_features": "rl_v3_all_features",
}


def feature_subset_columns(name: str = DEFAULT_FEATURE_SUBSET) -> list[str]:
    """Return the ordered column list for a named feature subset."""
    normalized = FEATURE_SUBSET_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if normalized not in FEATURE_SUBSETS:
        raise ValueError(
            f"Unknown feature subset '{name}'. "
            f"Expected one of: {sorted(FEATURE_SUBSETS)}"
        )
    return list(FEATURE_SUBSETS[normalized])


def feature_subset_version(name: str = DEFAULT_FEATURE_SUBSET) -> str:
    """Return a version tag for logging and reproducibility."""
    del name
    return FEATURE_SUBSET_VERSION


def extract_query_features(
    row: dict[str, Any],
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> dict[str, float]:
    """Extract the prompt-facing features for one query circuit.

    This is intentionally restricted to the initial/original circuit-state
    columns so we never leak post-optimization information into the prompt.
    """
    query_features: dict[str, float] = {}

    for column in feature_subset_columns(feature_subset):
        if column in RL_V3_DYNAMIC_PROGRESS_FEATURES:
            query_features[column] = 0.0
            continue
        value = row.get(column, 0.0)
        try:
            query_features[column] = float(value)
        except (TypeError, ValueError):
            query_features[column] = 0.0

    return query_features


def build_query_feature_row(
    row: dict[str, Any],
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> dict[str, Any]:
    """Build one sanitized query row for predictive components."""
    circuit_id = row.get("original_circuit")
    if circuit_id is None:
        raise KeyError("Query row is missing original_circuit.")
    return {
        "original_circuit": str(circuit_id),
        **extract_query_features(row, feature_subset=feature_subset),
    }


def build_query_feature_frame(
    frame: pd.DataFrame,
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> pd.DataFrame:
    """Build one sanitized query-feature row per original circuit."""
    representatives = circuit_representatives(frame)
    rows = [
        build_query_feature_row(row.to_dict(), feature_subset=feature_subset)
        for _, row in representatives.iterrows()
    ]
    columns = ["original_circuit", *feature_subset_columns(feature_subset)]
    return pd.DataFrame(rows, columns=columns)


def retrieval_feature_columns(
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> list[str]:
    """Return the numeric columns allowed for retrieval."""
    return feature_subset_columns(feature_subset)


def available_feature_subsets() -> list[str]:
    """Return the supported subset names in stable order."""
    return sorted(FEATURE_SUBSETS)
