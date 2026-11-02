"""Prediction orchestration for the LLM chain selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .baselines import BaselineSelection
from .features import DEFAULT_FEATURE_SUBSET
from .llm_client import LLMClient, LLMResponse
from .prompting import DEFAULT_PROMPT_MODE, build_prompt
from .schemas import (
    ChainSelection,
    chain_selection_json_schema,
    parse_chain_selection_json,
    validate_chain_selection,
)


@dataclass
class PredictionResult:
    """End-to-end prediction outcome for one circuit."""

    selection: ChainSelection
    raw_response: LLMResponse
    prompt_text: str
    prompt_mode: str
    used_fallback: bool
    fallback_reason: str | None
    fallback_method: str | None
    retry_count: int


def _fallback_selection(
    fallback: BaselineSelection,
    reason: str,
) -> ChainSelection:
    return ChainSelection(
        selected_chain_id=fallback.selected_chain_id,
        selected_chain=list(fallback.selected_chain),
        predicted_tradeoff_summary=f"Fallback selection via {fallback.method_name}.",
        reasoning_summary=reason,
        confidence=0.0,
        evidence_used=[fallback.method_name],
    )


def predict_chain_for_circuit(
    query_row: dict[str, Any],
    feature_subset: str,
    prompt_mode: str,
    optimizer_docs: dict[str, str],
    retrieved_neighbors: list[Any],
    objective_description: str,
    llm_client: LLMClient,
    candidate_chain_lookup: dict[str, dict[str, Any]],
    fallback_selection: BaselineSelection,
) -> PredictionResult:
    """Run prompt assembly, provider call, validation, and deterministic fallback."""
    prompt_text = build_prompt(
        query_row=query_row,
        feature_subset=feature_subset,
        prompt_mode=prompt_mode,
        optimizer_docs=optimizer_docs,
        retrieved_neighbors=retrieved_neighbors,
        objective_description=objective_description,
    )

    attempts = 0
    last_response: LLMResponse | None = None
    last_error: str | None = None
    while attempts < 2:
        attempts += 1
        response = llm_client.generate_chain_selection(
            prompt_text,
            schema=chain_selection_json_schema(),
        )
        last_response = response
        try:
            parsed = parse_chain_selection_json(response.raw_text)
            selection = validate_chain_selection(parsed, candidate_chain_lookup)
            return PredictionResult(
                selection=selection,
                raw_response=response,
                prompt_text=prompt_text,
                prompt_mode=prompt_mode,
                used_fallback=False,
                fallback_reason=None,
                fallback_method=None,
                retry_count=attempts - 1,
            )
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            if attempts >= 2:
                break
            prompt_text = (
                prompt_text
                + "\n\nValidation reminder: your previous answer was invalid. "
                + f"Return valid JSON only. Last validation error: {last_error}"
            )

    if last_response is None:
        raise RuntimeError("No provider response was captured before fallback.")

    selection = _fallback_selection(
        fallback_selection,
        reason=f"Schema validation failed after retry. Last error: {last_error}",
    )
    return PredictionResult(
        selection=selection,
        raw_response=last_response,
        prompt_text=prompt_text,
        prompt_mode=prompt_mode,
        used_fallback=True,
        fallback_reason=last_error,
        fallback_method=fallback_selection.method_name,
        retry_count=max(0, attempts - 1),
    )
