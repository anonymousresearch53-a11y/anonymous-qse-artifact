"""Provider boundary for structured LLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import json
import os
import time

from .schemas import chain_selection_json_schema


@dataclass
class LLMResponse:
    """Raw provider response plus metadata."""

    provider_name: str
    model_name: str
    raw_text: str
    latency_seconds: float
    estimated_cost_usd: float | None
    response_metadata: dict[str, Any]


class LLMClient(Protocol):
    """Minimal provider interface used by the predictor."""

    provider_name: str
    model_name: str

    def generate_chain_selection(
        self,
        prompt: str,
        schema: dict[str, Any],
    ) -> LLMResponse:
        """Generate one structured selection response."""


class MockLLMClient:
    """Deterministic mock provider for local testing."""

    provider_name = "mock"

    def __init__(self, model_name: str = "mock-deterministic-v1") -> None:
        self.model_name = model_name

    def generate_chain_selection(
        self,
        prompt: str,
        schema: dict[str, Any],
    ) -> LLMResponse:
        del prompt, schema
        started = time.perf_counter()
        payload = {
            "selected_chain_id": "chain_01",
            "selected_chain": ["qiskit_standard"],
            "predicted_tradeoff_summary": "Mock provider selected a stable baseline chain.",
            "reasoning_summary": "This deterministic response is intended for local integration tests.",
            "confidence": 0.5,
            "evidence_used": ["mock_provider"],
        }
        latency = time.perf_counter() - started
        return LLMResponse(
            provider_name=self.provider_name,
            model_name=self.model_name,
            raw_text=json.dumps(payload, sort_keys=True),
            latency_seconds=latency,
            estimated_cost_usd=0.0,
            response_metadata={"mode": "mock"},
        )


class OpenAILLMClient:
    """OpenAI-first provider with JSON-only responses and local validation."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")

        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "The openai package is required for OpenAILLMClient."
            ) from error

        self._client = OpenAI(api_key=api_key)

    def generate_chain_selection(
        self,
        prompt: str,
        schema: dict[str, Any],
    ) -> LLMResponse:
        del schema
        started = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful scientific assistant. "
                        "Return only valid JSON matching the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        latency = time.perf_counter() - started
        raw_text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        metadata = {}
        if usage is not None:
            metadata = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return LLMResponse(
            provider_name=self.provider_name,
            model_name=self.model_name,
            raw_text=raw_text,
            latency_seconds=latency,
            estimated_cost_usd=None,
            response_metadata=metadata,
        )


def build_llm_client(
    provider: str,
    model_name: str,
    api_key: str | None = None,
) -> LLMClient:
    """Instantiate one supported provider."""
    normalized = str(provider).strip().lower()
    if normalized == "mock":
        return MockLLMClient(model_name=model_name)
    if normalized == "openai":
        return OpenAILLMClient(model_name=model_name, api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def provider_defaults() -> dict[str, Any]:
    """Return provider defaults for run logging."""
    return {
        "provider_name": "openai",
        "default_model_name": "gpt-4o-mini",
        "response_schema_version": "v1",
        "schema": chain_selection_json_schema(),
    }

