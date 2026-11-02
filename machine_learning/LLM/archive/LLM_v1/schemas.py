"""Schema validation helpers for LLM outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import json


OUTPUT_SCHEMA_VERSION = "v1"


@dataclass
class ChainSelection:
    """Validated one-shot chain selection."""

    selected_chain_id: str
    selected_chain: list[str]
    predicted_tradeoff_summary: str
    reasoning_summary: str
    confidence: float
    evidence_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def chain_selection_json_schema() -> dict[str, Any]:
    """Return the response schema shown to providers and prompts."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_chain_id": {"type": "string"},
            "selected_chain": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "predicted_tradeoff_summary": {"type": "string"},
            "reasoning_summary": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence_used": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "selected_chain_id",
            "selected_chain",
            "predicted_tradeoff_summary",
            "reasoning_summary",
            "confidence",
            "evidence_used",
        ],
    }


def parse_chain_selection_json(raw_text: str) -> dict[str, Any]:
    """Parse a raw JSON string into a dictionary."""
    return json.loads(raw_text)


def validate_chain_selection(
    payload: dict[str, Any],
    candidate_chain_lookup: dict[str, dict[str, Any]],
) -> ChainSelection:
    """Validate model output against the fixed candidate set."""
    required = {
        "selected_chain_id",
        "selected_chain",
        "predicted_tradeoff_summary",
        "reasoning_summary",
        "confidence",
        "evidence_used",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    chain_id = str(payload["selected_chain_id"])
    if chain_id not in candidate_chain_lookup:
        raise ValueError(f"Unknown chain id: {chain_id}")

    selected_chain = [str(item) for item in payload["selected_chain"]]
    candidate_chain = list(candidate_chain_lookup[chain_id]["chain"])
    if selected_chain != candidate_chain:
        raise ValueError(
            f"Selected chain does not match chain id {chain_id}: "
            f"{selected_chain} != {candidate_chain}"
        )

    confidence = float(payload["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    evidence_used = [str(item) for item in payload["evidence_used"]]
    return ChainSelection(
        selected_chain_id=chain_id,
        selected_chain=selected_chain,
        predicted_tradeoff_summary=str(payload["predicted_tradeoff_summary"]),
        reasoning_summary=str(payload["reasoning_summary"]),
        confidence=confidence,
        evidence_used=evidence_used,
    )

