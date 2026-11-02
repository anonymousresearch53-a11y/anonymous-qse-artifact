"""Feature subset definitions for prompt construction and retrieval."""

from __future__ import annotations

from typing import Any


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


FEATURE_SUBSETS = {
    "basic_size": BASIC_SIZE_FEATURES,
    "size_plus_gate_composition": SIZE_PLUS_GATE_COMPOSITION_FEATURES,
    "full_structural": FULL_STRUCTURAL_FEATURES,
}


def feature_subset_columns(name: str = DEFAULT_FEATURE_SUBSET) -> list[str]:
    """Return the ordered column list for a named feature subset."""
    normalized = str(name).strip().lower()
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
        value = row.get(column, 0.0)
        try:
            query_features[column] = float(value)
        except (TypeError, ValueError):
            query_features[column] = 0.0

    return query_features


def retrieval_feature_columns(
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> list[str]:
    """Return the numeric columns allowed for retrieval."""
    return feature_subset_columns(feature_subset)


def available_feature_subsets() -> list[str]:
    """Return the supported subset names in stable order."""
    return sorted(FEATURE_SUBSETS)
