"""Prompt-facing feature subset definitions."""

from __future__ import annotations

from typing import Any


FEATURE_SUBSET_VERSION = "v1"
INITIAL_PROMPT_FEATURES = [
    "initial_num_qubits",
    "initial_depth",
    "initial_two_qubit_gates",
    "initial_total_gates",
    "initial_two_qubit_depth",
    "initial_gate_density",
    "initial_two_qubit_ratio",
    "initial_num_interaction_edges",
    "initial_avg_interaction_degree",
    "initial_max_interaction_degree",
    "initial_active_interaction_qubits",
    "initial_active_interaction_qubit_fraction",
    "initial_rz_gate_fraction",
    "initial_sx_gate_fraction",
    "initial_x_gate_fraction",
    "initial_cx_gate_fraction",
    "initial_two_qubit_gate_fraction",
    "initial_depth_per_qubit",
    "initial_gates_per_qubit",
    "initial_two_qubit_gates_per_qubit",
]
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
FULL_STRUCTURAL_FEATURES = INITIAL_PROMPT_FEATURES

FEATURE_SUBSETS = {
    "basic_size": BASIC_SIZE_FEATURES,
    "size_plus_gate_composition": SIZE_PLUS_GATE_COMPOSITION_FEATURES,
    "full_structural": FULL_STRUCTURAL_FEATURES,
}
DEFAULT_FEATURE_SUBSET = "full_structural"


def feature_subset_columns(name: str = DEFAULT_FEATURE_SUBSET) -> list[str]:
    """Return one named prompt feature subset."""
    normalized = str(name).strip()
    if normalized not in FEATURE_SUBSETS:
        raise ValueError(
            f"Unsupported feature subset: {name}. "
            f"Supported values: {sorted(FEATURE_SUBSETS)}"
        )
    return list(FEATURE_SUBSETS[normalized])


def feature_subset_version(name: str = DEFAULT_FEATURE_SUBSET) -> str:
    """Return the logged version token for one feature subset."""
    return f"{FEATURE_SUBSET_VERSION}:{name}"


def extract_query_features(
    row: dict[str, Any],
    feature_subset: str = DEFAULT_FEATURE_SUBSET,
) -> dict[str, float | str]:
    """Extract the exact query feature view shown in prompts."""
    extracted: dict[str, float | str] = {}
    for column in feature_subset_columns(feature_subset):
        extracted[column] = row.get(column)
    return extracted


def retrieval_feature_columns(feature_subset: str = DEFAULT_FEATURE_SUBSET) -> list[str]:
    """Return the numeric features used by nearest-neighbor retrieval."""
    return feature_subset_columns(feature_subset)

