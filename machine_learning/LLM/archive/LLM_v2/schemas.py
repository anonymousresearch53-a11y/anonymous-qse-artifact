"""Structured response schema and validation for chain selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESPONSE_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class ChainSelectionResponse:
    """Validated chain-selection response from the LLM."""

    selected_chain_id: str
    selected_chain: list[str]
    predicted_tradeoff_summary: str
    reasoning_summary: str
    confidence: float
    evidence_used: list[str]


def chain_selection_json_schema() -> dict[str, Any]:
    """Return the JSON schema sent to the LLM provider."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_chain_id",
            "selected_chain",
            "predicted_tradeoff_summary",
            "reasoning_summary",
            "confidence",
            "evidence_used",
        ],
        "properties": {
            "selected_chain_id": {
                "type": "string",
                "minLength": 1,
            },
            "selected_chain": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "predicted_tradeoff_summary": {
                "type": "string",
                "minLength": 1,
            },
            "reasoning_summary": {
                "type": "string",
                "minLength": 1,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "evidence_used": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
    }


def validate_chain_selection_response(
    parsed_response: dict[str, Any],
    candidate_lookup: dict[str, dict[str, Any]],
) -> ChainSelectionResponse:
    """Validate a parsed LLM response against the response contract."""
    if not isinstance(parsed_response, dict):
        raise TypeError("Parsed response must be a dictionary.")

    required_fields = [
        "selected_chain_id",
        "selected_chain",
        "predicted_tradeoff_summary",
        "reasoning_summary",
        "confidence",
        "evidence_used",
    ]
    missing_fields = [
        field_name for field_name in required_fields if field_name not in parsed_response
    ]
    if missing_fields:
        raise ValueError(f"Parsed response is missing required fields: {missing_fields}")

    selected_chain_id = str(parsed_response["selected_chain_id"]).strip()
    if not selected_chain_id:
        raise ValueError("selected_chain_id must be a non-empty string.")

    if selected_chain_id not in candidate_lookup:
        raise ValueError(f"Unknown selected_chain_id: {selected_chain_id}")

    raw_selected_chain = parsed_response["selected_chain"]
    if not isinstance(raw_selected_chain, list) or not raw_selected_chain:
        raise ValueError("selected_chain must be a non-empty list of strings.")

    selected_chain = [str(item).strip() for item in raw_selected_chain]
    if any(not item for item in selected_chain):
        raise ValueError("selected_chain contains an empty optimizer name.")

    expected_chain = list(candidate_lookup[selected_chain_id]["chain"])
    if selected_chain != expected_chain:
        raise ValueError(
            "selected_chain does not match the expected chain for "
            f"{selected_chain_id}. Expected {expected_chain}, got {selected_chain}"
        )

    predicted_tradeoff_summary = str(
        parsed_response["predicted_tradeoff_summary"]
    ).strip()
    if not predicted_tradeoff_summary:
        raise ValueError("predicted_tradeoff_summary must be non-empty.")

    reasoning_summary = str(parsed_response["reasoning_summary"]).strip()
    if not reasoning_summary:
        raise ValueError("reasoning_summary must be non-empty.")

    try:
        confidence = float(parsed_response["confidence"])
    except (TypeError, ValueError) as error:
        raise ValueError("confidence must be a numeric value.") from error

    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0 inclusive.")

    raw_evidence_used = parsed_response["evidence_used"]
    if not isinstance(raw_evidence_used, list) or not raw_evidence_used:
        raise ValueError("evidence_used must be a non-empty list of strings.")

    evidence_used = [str(item).strip() for item in raw_evidence_used]
    if any(not item for item in evidence_used):
        raise ValueError("evidence_used contains an empty item.")

    return ChainSelectionResponse(
        selected_chain_id=selected_chain_id,
        selected_chain=selected_chain,
        predicted_tradeoff_summary=predicted_tradeoff_summary,
        reasoning_summary=reasoning_summary,
        confidence=confidence,
        evidence_used=evidence_used,
    )
