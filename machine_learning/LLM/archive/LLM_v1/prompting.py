"""Prompt assembly for one-shot chain selection."""

from __future__ import annotations

from typing import Any

from .data import build_candidate_chains
from .features import DEFAULT_FEATURE_SUBSET, extract_query_features, feature_subset_version
from .schemas import chain_selection_json_schema


PROMPT_VERSION = "v1"
DEFAULT_PROMPT_MODE = "full_evidence"
PROMPT_MODES = {
    "no_retrieval",
    "docs_only",
    "similar_circuits_only",
    "full_evidence",
}


def load_optimizer_docs(doc_dir) -> dict[str, str]:
    """Load the stable optimizer documentation bundle."""
    docs = {}
    for path in sorted(doc_dir.glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8").strip()
    return docs


def candidate_chain_lines() -> list[str]:
    """Return candidate chains formatted for prompt inclusion."""
    lines = []
    for candidate in build_candidate_chains():
        chain_text = " -> ".join(candidate["chain"])
        lines.append(
            f"- {candidate['chain_id']}: {chain_text} (chain_name={candidate['chain_name']})"
        )
    return lines


def _format_mapping(title: str, mapping: dict[str, Any]) -> list[str]:
    lines = [title]
    for key, value in mapping.items():
        lines.append(f"- {key}: {value}")
    return lines


def build_prompt(
    query_row: dict[str, Any],
    feature_subset: str,
    prompt_mode: str,
    optimizer_docs: dict[str, str],
    retrieved_neighbors: list[Any],
    objective_description: str,
) -> str:
    """Build one deterministic prompt string."""
    normalized_mode = str(prompt_mode).strip().lower()
    if normalized_mode not in PROMPT_MODES:
        raise ValueError(
            f"Unsupported prompt mode: {prompt_mode}. "
            f"Supported values: {sorted(PROMPT_MODES)}"
        )

    query_features = extract_query_features(query_row, feature_subset=feature_subset)
    lines = [
        "Task: choose exactly one valid optimizer chain from the provided list.",
        "Goal: maximize the fixed scientific objective described below.",
        "Rules:",
        "- Use only the provided candidate chains.",
        "- Do not invent optimizer names or chains.",
        "- Return only JSON with the required keys.",
        "",
        "Fixed objective:",
        objective_description,
        "",
        "Candidate chains:",
        *candidate_chain_lines(),
        "",
        "Current circuit features:",
        *_format_mapping(
            f"(feature_subset={feature_subset_version(feature_subset)})",
            query_features,
        ),
        "",
    ]

    if normalized_mode in {"docs_only", "full_evidence"}:
        lines.append("Optimizer documentation:")
        for optimizer_name, text in optimizer_docs.items():
            lines.append(f"- {optimizer_name}: {text}")
        lines.append("")

    if normalized_mode in {"similar_circuits_only", "full_evidence"}:
        lines.append("Retrieved similar training circuits:")
        if not retrieved_neighbors:
            lines.append("- none")
        for neighbor in retrieved_neighbors:
            lines.append(
                f"- {neighbor.circuit_id}: distance={neighbor.distance:.4f}, "
                f"best_chain={neighbor.best_chain_name}, "
                f"best_score={neighbor.best_chain_score:.4f}"
            )
            lines.append("  Top outcomes:")
            for outcome in neighbor.top_outcomes:
                lines.append(
                    "  "
                    f"- {outcome['chain_id']} ({outcome['chain_name']}): "
                    f"score={outcome['score']:.4f}, "
                    f"2Q={outcome['two_qubit_reduction_pct']:.2f}%, "
                    f"depth={outcome['depth_reduction_pct']:.2f}%, "
                    f"runtime={outcome['runtime_seconds']:.2f}s"
                )
        lines.append("")

    lines.append("Return JSON only using this schema:")
    lines.append(str(chain_selection_json_schema()))
    return "\n".join(lines)

