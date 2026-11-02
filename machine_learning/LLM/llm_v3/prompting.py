"""Prompt construction helpers for the LLM orchestration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from machine_learning.LLM.llm_v3.objective import ObjectiveConfig
    from machine_learning.LLM.llm_v3.retrieval import RetrievedNeighbor
except ImportError:
    from .objective import ObjectiveConfig
    from .retrieval import RetrievedNeighbor


PROMPT_VERSION = "v2_shortlisted_profiles"
DEFAULT_PROMPT_MODE = "full_evidence"
ALLOWED_PROMPT_MODES = {
    "full_evidence",
    "docs_only",
    "similar_circuits_only",
    "no_retrieval",
}


@dataclass(frozen=True)
class BuiltPrompt:
    """Fully rendered prompt package plus structured sections for logging."""

    system_prompt: str
    user_prompt: str
    prompt_version: str
    prompt_mode: str
    prompt_sections: dict[str, Any]


def validate_prompt_mode(prompt_mode: str) -> str:
    """Normalize and validate a prompt mode name."""
    normalized = str(prompt_mode).strip().lower()
    if normalized not in ALLOWED_PROMPT_MODES:
        raise ValueError(
            f"Unsupported prompt mode '{prompt_mode}'. "
            f"Expected one of: {sorted(ALLOWED_PROMPT_MODES)}"
        )
    return normalized


def build_system_prompt() -> str:
    """Return the stable system instruction for chain selection."""
    return (
        "You are selecting a quantum-circuit optimizer chain. "
        "Choose exactly one candidate chain from the provided list. "
        "Optimize the provided objective using only the given evidence. "
        "Prefer the no-optimization action when the evidence does not justify an optimizer chain. "
        "Do not invent optimizers, chains, metrics, or evidence. "
        "Return a structured response that selects one valid chain."
    )


def format_objective_section(objective_config: ObjectiveConfig) -> str:
    """Describe the fixed optimization objective in plain language."""
    return (
        "Objective:\n"
        "Maximize the deterministic episode-utility score used for RL comparison.\n"
        "Higher two-qubit reduction is better. Higher depth reduction is better. "
        "Lower runtime is better. The final optimized result also receives an outcome correction.\n"
        f"reward_type={objective_config.reward_type}, "
        f"2q_weight={float(objective_config.two_qubit_weight):.3f}, "
        f"depth_weight={float(objective_config.depth_weight):.3f}, "
        f"runtime_weight={float(objective_config.runtime_weight):.3f}, "
        f"final_weight={float(objective_config.final_reward_weight):.3f}"
    )


def format_candidate_chains_section(
    candidate_chains: list[dict[str, Any]],
    candidate_profile_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the valid candidate chain list in stable order."""
    lines = ["Valid Candidate Chains:"]
    for candidate in candidate_chains:
        chain_id = str(candidate["chain_id"])
        chain_label = "No optimization" if not candidate["chain"] else " -> ".join(candidate["chain"])
        profile = None if candidate_profile_lookup is None else candidate_profile_lookup.get(chain_id)
        if profile is None:
            lines.append(f"{chain_id}: {chain_label}")
            continue
        lines.append(
            f"{chain_id}: {chain_label} | "
            f"score={float(profile['avg_objective_score']):.3f} | "
            f"2q={float(profile['avg_two_qubit_gate_reduction_pct']):.1f}% | "
            f"depth={float(profile['avg_depth_reduction_pct']):.1f}% | "
            f"rt={float(profile['avg_runtime_seconds']):.1f}s | "
            f"top1={100.0 * float(profile['top_1_rate']):.1f}%"
        )
    return "\n".join(lines)


def format_query_circuit_section(
    query_circuit_id: str,
    query_features: dict[str, float],
    include_circuit_id: bool = True,
) -> str:
    """Render the query circuit id and approved initial-state features."""
    lines = ["Query Circuit:"]
    if include_circuit_id:
        lines.append(f"circuit_id: {query_circuit_id}")
    lines.append("initial_state_features:")
    for feature_name, feature_value in query_features.items():
        lines.append(f"{feature_name}: {float(feature_value):.4f}")
    return "\n".join(lines)


def format_optimizer_docs_section(
    optimizer_docs: dict[str, str],
) -> str:
    """Render single-optimizer train priors."""
    lines = ["Single-Optimizer Train Priors:"]
    for optimizer_name in sorted(optimizer_docs):
        doc_text = str(optimizer_docs[optimizer_name]).strip()
        lines.append(f"{optimizer_name}: {doc_text}")
    return "\n".join(lines)


def format_retrieved_neighbors_section(
    retrieved_neighbors: list[RetrievedNeighbor],
    include_neighbor_identity: bool = True,
) -> str:
    """Render similar-circuit evidence from the training split."""
    lines = ["Similar Training Circuits:"]
    if not retrieved_neighbors:
        lines.append("No similar-circuit evidence was retrieved.")
        return "\n".join(lines)

    all_neighbors_original = all(
        neighbor.best_chain_name == "__original__" for neighbor in retrieved_neighbors
    )
    strong_non_original_neighbors = [
        neighbor for neighbor in retrieved_neighbors if neighbor.best_vs_original_margin > 0.25
    ]
    if all_neighbors_original:
        lines.append("All retrieved neighbors preferred No optimization over every optimizer chain.")
    elif strong_non_original_neighbors:
        lines.append(
            "Strong non-original neighbor wins exist for: "
            + ", ".join(
                f"{neighbor.circuit_id} ({neighbor.best_chain_name}, margin={neighbor.best_vs_original_margin:.3f})"
                for neighbor in strong_non_original_neighbors[:3]
            )
        )
    for neighbor_index, neighbor in enumerate(retrieved_neighbors, start=1):
        identity_prefix = (
            f"circuit_id: {neighbor.circuit_id}, "
            if include_neighbor_identity
            else f"neighbor_index: {neighbor_index}, "
        )
        lines.append(
            identity_prefix
            + f"distance: {float(neighbor.distance):.4f}, "
            f"best_chain_id: {neighbor.best_chain_id}, "
            f"best_chain_name: {neighbor.best_chain_name}, "
            f"best_chain_score: {float(neighbor.best_chain_score):.3f}, "
            f"best_vs_original_margin: {float(neighbor.best_vs_original_margin):.3f}"
        )
        lines.append(
            "counterfactuals: "
            f"original_score={float(neighbor.original_score):.3f} | "
            f"best_non_original={neighbor.best_non_original_chain_id} "
            f"({neighbor.best_non_original_chain_name}) score={float(neighbor.best_non_original_chain_score):.3f}"
        )
        lines.append("neighbor_feature_summary:")
        for feature_name, feature_value in neighbor.neighbor_feature_summary.items():
            lines.append(f"{feature_name}: {float(feature_value):.4f}")
        lines.append("top_outcomes:")
        for outcome in neighbor.top_outcomes:
            lines.append(
                " | ".join(
                    [
                        f"chain_id: {outcome['chain_id']}",
                        f"chain_name: {outcome['chain_name']}",
                        f"chain: {outcome['chain']}",
                        f"score: {float(outcome['score']):.3f}",
                        "two_qubit_reduction_pct: "
                        f"{float(outcome['two_qubit_reduction_pct']):.1f}",
                        f"depth_reduction_pct: {float(outcome['depth_reduction_pct']):.1f}",
                        f"runtime_seconds: {float(outcome['runtime_seconds']):.2f}",
                        f"runtime_score: {float(outcome['runtime_score']):.3f}",
                    ]
                )
            )
    return "\n".join(lines)


def format_gate_section(gate_decision: dict[str, Any] | None) -> str:
    """Render the learned optimize-vs-original gate output."""
    if gate_decision is None:
        return ""
    return (
        "Optimize-vs-Original Gate:\n"
        f"recommendation: {gate_decision['recommendation_label']}\n"
        f"probability_optimize: {float(gate_decision['probability_optimize']):.3f}\n"
        f"probability_original: {float(gate_decision['probability_original']):.3f}\n"
        "Treat this as a learned prior, not as a guaranteed answer."
    )


def build_user_prompt(
    query_circuit_id: str,
    query_features: dict[str, float],
    candidate_chains: list[dict[str, Any]],
    objective_config: ObjectiveConfig,
    optimizer_docs: dict[str, str],
    candidate_profile_lookup: dict[str, dict[str, Any]] | None,
    gate_decision: dict[str, Any] | None,
    retrieved_neighbors: list[RetrievedNeighbor] | None,
    prompt_mode: str,
    include_query_circuit_id: bool = True,
    include_neighbor_identity: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Assemble the user prompt and structured prompt-section payload."""
    normalized_mode = validate_prompt_mode(prompt_mode)

    sections: dict[str, Any] = {
        "task_overview": (
            "Choose the single best optimizer chain for the query circuit "
            "from the provided valid candidate list. You may choose the explicit "
            "no-optimization action if it looks safest or strongest."
        ),
        "objective": format_objective_section(objective_config),
        "candidate_chains": format_candidate_chains_section(
            candidate_chains,
            candidate_profile_lookup=candidate_profile_lookup,
        ),
        "query_circuit": format_query_circuit_section(
            query_circuit_id=query_circuit_id,
            query_features=query_features,
            include_circuit_id=include_query_circuit_id,
        ),
    }
    if gate_decision is not None:
        sections["gate_prior"] = format_gate_section(gate_decision)

    if normalized_mode in {"full_evidence", "docs_only"}:
        sections["optimizer_docs"] = format_optimizer_docs_section(optimizer_docs)

    if normalized_mode in {"full_evidence", "similar_circuits_only"}:
        sections["retrieved_neighbors"] = format_retrieved_neighbors_section(
            retrieved_neighbors or [],
            include_neighbor_identity=include_neighbor_identity,
        )

    sections["final_instruction"] = (
        "Choose exactly one candidate chain from the provided list. "
        "Reference the selected chain by its provided chain_id. "
        "Do not invent any optimizer or chain. "
        "Prefer the candidate with the strongest evidence-supported objective tradeoff, "
        "not just the highest expected 2Q reduction."
    )

    ordered_section_names = [
        "task_overview",
        "objective",
        "candidate_chains",
        "query_circuit",
        "gate_prior",
        "optimizer_docs",
        "retrieved_neighbors",
        "final_instruction",
    ]

    prompt_parts = []
    for section_name in ordered_section_names:
        if section_name in sections:
            prompt_parts.append(sections[section_name])

    return "\n\n".join(prompt_parts), sections


def build_prompt(
    query_circuit_id: str,
    query_features: dict[str, float],
    candidate_chains: list[dict[str, Any]],
    objective_config: ObjectiveConfig,
    optimizer_docs: dict[str, str],
    candidate_profile_lookup: dict[str, dict[str, Any]] | None = None,
    gate_decision: dict[str, Any] | None = None,
    retrieved_neighbors: list[RetrievedNeighbor] | None = None,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    include_query_circuit_id: bool = True,
    include_neighbor_identity: bool = True,
) -> BuiltPrompt:
    """Build the full prompt package for one circuit decision."""
    normalized_mode = validate_prompt_mode(prompt_mode)
    system_prompt = build_system_prompt()
    user_prompt, prompt_sections = build_user_prompt(
        query_circuit_id=query_circuit_id,
        query_features=query_features,
        candidate_chains=candidate_chains,
        objective_config=objective_config,
        optimizer_docs=optimizer_docs,
        candidate_profile_lookup=candidate_profile_lookup,
        gate_decision=gate_decision,
        retrieved_neighbors=retrieved_neighbors,
        prompt_mode=normalized_mode,
        include_query_circuit_id=include_query_circuit_id,
        include_neighbor_identity=include_neighbor_identity,
    )
    return BuiltPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_version=PROMPT_VERSION,
        prompt_mode=normalized_mode,
        prompt_sections=prompt_sections,
    )
