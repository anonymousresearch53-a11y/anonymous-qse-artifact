"""Prompt construction helpers for the LLM orchestration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from machine_learning.LLM.LLM_v2.objective import ObjectiveConfig
    from machine_learning.LLM.LLM_v2.retrieval import RetrievedNeighbor
except ImportError:
    from .objective import ObjectiveConfig
    from .retrieval import RetrievedNeighbor


PROMPT_VERSION = "v1"
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
        "Do not invent optimizers, chains, metrics, or evidence. "
        "Return a structured response that selects one valid chain."
    )


def format_objective_section(objective_config: ObjectiveConfig) -> str:
    """Describe the fixed optimization objective in plain language."""
    return (
        "Objective:\n"
        "Maximize a fixed score that rewards stronger two-qubit-gate reduction, "
        "rewards depth reduction, penalizes runtime cost, and includes a final "
        "outcome correction based on the overall optimized result.\n"
        f"reward_type: {objective_config.reward_type}\n"
        f"runtime_weight: {float(objective_config.runtime_weight):.6f}\n"
        f"two_qubit_weight: {float(objective_config.two_qubit_weight):.6f}\n"
        f"depth_weight: {float(objective_config.depth_weight):.6f}\n"
        f"final_reward_weight: {float(objective_config.final_reward_weight):.6f}"
    )


def format_candidate_chains_section(
    candidate_chains: list[dict[str, Any]],
) -> str:
    """Render the valid candidate chain list in stable order."""
    lines = ["Valid Candidate Chains:"]
    for candidate in candidate_chains:
        lines.append(f"{candidate['chain_id']}: {candidate['chain']}")
    return "\n".join(lines)


def format_query_circuit_section(
    query_circuit_id: str,
    query_features: dict[str, float],
) -> str:
    """Render the query circuit id and approved initial-state features."""
    lines = [
        "Query Circuit:",
        f"circuit_id: {query_circuit_id}",
        "initial_state_features:",
    ]
    for feature_name, feature_value in query_features.items():
        lines.append(f"{feature_name}: {float(feature_value):.6f}")
    return "\n".join(lines)


def format_optimizer_docs_section(
    optimizer_docs: dict[str, str],
) -> str:
    """Render the canonical optimizer descriptions."""
    lines = ["Optimizer Reference Notes:"]
    for optimizer_name in sorted(optimizer_docs):
        doc_text = str(optimizer_docs[optimizer_name]).strip()
        lines.append(f"{optimizer_name}: {doc_text}")
    return "\n".join(lines)


def format_retrieved_neighbors_section(
    retrieved_neighbors: list[RetrievedNeighbor],
) -> str:
    """Render similar-circuit evidence from the training split."""
    lines = ["Similar Training Circuits:"]
    for neighbor in retrieved_neighbors:
        lines.append(
            f"circuit_id: {neighbor.circuit_id}, "
            f"distance: {float(neighbor.distance):.6f}, "
            f"best_chain_id: {neighbor.best_chain_id}, "
            f"best_chain_name: {neighbor.best_chain_name}, "
            f"best_chain_score: {float(neighbor.best_chain_score):.6f}"
        )
        lines.append("neighbor_feature_summary:")
        for feature_name, feature_value in neighbor.neighbor_feature_summary.items():
            lines.append(f"{feature_name}: {float(feature_value):.6f}")
        lines.append("top_outcomes:")
        for outcome in neighbor.top_outcomes:
            lines.append(
                " | ".join(
                    [
                        f"chain_id: {outcome['chain_id']}",
                        f"chain_name: {outcome['chain_name']}",
                        f"chain: {outcome['chain']}",
                        f"score: {float(outcome['score']):.6f}",
                        "two_qubit_reduction_pct: "
                        f"{float(outcome['two_qubit_reduction_pct']):.6f}",
                        f"depth_reduction_pct: {float(outcome['depth_reduction_pct']):.6f}",
                        f"runtime_seconds: {float(outcome['runtime_seconds']):.6f}",
                        f"runtime_score: {float(outcome['runtime_score']):.6f}",
                    ]
                )
            )
    return "\n".join(lines)


def build_user_prompt(
    query_circuit_id: str,
    query_features: dict[str, float],
    candidate_chains: list[dict[str, Any]],
    objective_config: ObjectiveConfig,
    optimizer_docs: dict[str, str],
    retrieved_neighbors: list[RetrievedNeighbor] | None,
    prompt_mode: str,
) -> tuple[str, dict[str, Any]]:
    """Assemble the user prompt and structured prompt-section payload."""
    normalized_mode = validate_prompt_mode(prompt_mode)

    sections: dict[str, Any] = {
        "task_overview": (
            "Choose the single best optimizer chain for the query circuit "
            "from the provided valid candidate list."
        ),
        "objective": format_objective_section(objective_config),
        "candidate_chains": format_candidate_chains_section(candidate_chains),
        "query_circuit": format_query_circuit_section(
            query_circuit_id=query_circuit_id,
            query_features=query_features,
        ),
    }

    if normalized_mode in {"full_evidence", "docs_only"}:
        sections["optimizer_docs"] = format_optimizer_docs_section(optimizer_docs)

    if normalized_mode in {"full_evidence", "similar_circuits_only"}:
        sections["retrieved_neighbors"] = format_retrieved_neighbors_section(
            retrieved_neighbors or []
        )

    sections["final_instruction"] = (
        "Choose exactly one candidate chain from the provided list. "
        "Reference the selected chain by its provided chain_id. "
        "Do not invent any optimizer or chain."
    )

    ordered_section_names = [
        "task_overview",
        "objective",
        "candidate_chains",
        "query_circuit",
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
    retrieved_neighbors: list[RetrievedNeighbor] | None = None,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
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
        retrieved_neighbors=retrieved_neighbors,
        prompt_mode=normalized_mode,
    )
    return BuiltPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_version=PROMPT_VERSION,
        prompt_mode=normalized_mode,
        prompt_sections=prompt_sections,
    )
