"""Named ablation presets for llm_v3 experiment runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentPreset:
    """Resolved prompt/routing configuration for one named experiment."""

    name: str
    prompt_mode: str
    include_candidate_profiles: bool
    include_optimizer_docs: bool
    include_retrieval: bool
    include_gate: bool
    include_ranker: bool
    shortlist_size: int | None
    include_query_circuit_id: bool
    include_neighbor_identity: bool


PRESET_CLEAN_ALL_CANDIDATES = "clean_all_candidates"
PRESET_PROFILES_ONLY_ALL_CANDIDATES = "profiles_only_all_candidates"
PRESET_RETRIEVAL_ONLY_ALL_CANDIDATES = "retrieval_only_all_candidates"
PRESET_GATE_ONLY_ALL_CANDIDATES = "gate_only_all_candidates"
PRESET_SHORTLIST_ONLY_TOP10 = "shortlist_only_top10"
PRESET_RETRIEVAL_PLUS_GATE_ALL_CANDIDATES = "retrieval_plus_gate_all_candidates"
PRESET_ALL_PROMPT_EXTRAS_ALL_CANDIDATES = "all_prompt_extras_all_candidates"
PRESET_FULL_SYSTEM = "full_system"


_PRESETS: dict[str, ExperimentPreset] = {
    PRESET_CLEAN_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_CLEAN_ALL_CANDIDATES,
        prompt_mode="no_retrieval",
        include_candidate_profiles=False,
        include_optimizer_docs=False,
        include_retrieval=False,
        include_gate=False,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_PROFILES_ONLY_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_PROFILES_ONLY_ALL_CANDIDATES,
        prompt_mode="no_retrieval",
        include_candidate_profiles=True,
        include_optimizer_docs=False,
        include_retrieval=False,
        include_gate=False,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_RETRIEVAL_ONLY_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_RETRIEVAL_ONLY_ALL_CANDIDATES,
        prompt_mode="similar_circuits_only",
        include_candidate_profiles=False,
        include_optimizer_docs=False,
        include_retrieval=True,
        include_gate=False,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_GATE_ONLY_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_GATE_ONLY_ALL_CANDIDATES,
        prompt_mode="no_retrieval",
        include_candidate_profiles=False,
        include_optimizer_docs=False,
        include_retrieval=False,
        include_gate=True,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_SHORTLIST_ONLY_TOP10: ExperimentPreset(
        name=PRESET_SHORTLIST_ONLY_TOP10,
        prompt_mode="no_retrieval",
        include_candidate_profiles=False,
        include_optimizer_docs=False,
        include_retrieval=False,
        include_gate=False,
        include_ranker=True,
        shortlist_size=10,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_RETRIEVAL_PLUS_GATE_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_RETRIEVAL_PLUS_GATE_ALL_CANDIDATES,
        prompt_mode="similar_circuits_only",
        include_candidate_profiles=False,
        include_optimizer_docs=False,
        include_retrieval=True,
        include_gate=True,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_ALL_PROMPT_EXTRAS_ALL_CANDIDATES: ExperimentPreset(
        name=PRESET_ALL_PROMPT_EXTRAS_ALL_CANDIDATES,
        prompt_mode="full_evidence",
        include_candidate_profiles=True,
        include_optimizer_docs=True,
        include_retrieval=True,
        include_gate=True,
        include_ranker=False,
        shortlist_size=None,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
    PRESET_FULL_SYSTEM: ExperimentPreset(
        name=PRESET_FULL_SYSTEM,
        prompt_mode="full_evidence",
        include_candidate_profiles=True,
        include_optimizer_docs=True,
        include_retrieval=True,
        include_gate=True,
        include_ranker=True,
        shortlist_size=10,
        include_query_circuit_id=False,
        include_neighbor_identity=False,
    ),
}


def available_experiment_presets() -> list[str]:
    """Return the supported named presets in stable order."""
    return list(_PRESETS)


def resolve_experiment_preset(name: str) -> ExperimentPreset:
    """Return the preset config for one supported preset name."""
    normalized = str(name).strip().lower()
    if normalized not in _PRESETS:
        raise ValueError(
            f"Unknown experiment preset '{name}'. "
            f"Expected one of: {available_experiment_presets()}"
        )
    return _PRESETS[normalized]
