"""OpenAI-first structured LLM client for chain-selection experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import json
import os
import time

try:
    from machine_learning.LLM.LLM_v2.prompting import BuiltPrompt
except ImportError:
    from .prompting import BuiltPrompt


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_NAME = "gpt-5.4-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 800
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for one structured LLM provider client."""

    provider: str = DEFAULT_PROVIDER
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass
class LLMCallResult:
    """Complete result of one model call, including raw and parsed outputs."""

    provider: str
    model_name: str
    system_prompt: str
    user_prompt: str
    raw_response_text: str
    parsed_response: dict[str, Any] | None
    success: bool
    error_message: str | None
    latency_seconds: float
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


class StructuredLLMClient(Protocol):
    """Minimal interface for structured prediction calls."""

    config: LLMClientConfig

    def generate_structured(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> LLMCallResult:
        """Send one structured-output request and return the result."""


class OpenAIStructuredClient:
    """OpenAI implementation for structured JSON-schema responses."""

    def __init__(
        self,
        config: LLMClientConfig,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAIStructuredClient."
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "The openai package is required to use OpenAIStructuredClient."
            ) from error

        self._client = OpenAI(
            api_key=resolved_api_key,
            timeout=float(self.config.timeout_seconds),
        )

    def generate_structured(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> LLMCallResult:
        """Call OpenAI with a system prompt, user prompt, and strict JSON schema."""
        started = time.perf_counter()

        try:
            response = self._client.responses.create(
                model=self.config.model_name,
                temperature=float(self.config.temperature),
                max_output_tokens=int(self.config.max_output_tokens),
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": built_prompt.system_prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": built_prompt.user_prompt,
                            }
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": response_schema,
                        "strict": True,
                    }
                },
            )
            latency = time.perf_counter() - started

            raw_response_text = getattr(response, "output_text", "") or ""
            parsed_response = None
            error_message = None
            success = False

            try:
                parsed_response = json.loads(raw_response_text)
                success = True
            except json.JSONDecodeError as error:
                error_message = f"JSON parse failed: {error}"

            usage = self._extract_usage(response)
            finish_reason = self._extract_finish_reason(response)

            return LLMCallResult(
                provider=self.config.provider,
                model_name=self.config.model_name,
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                raw_response_text=raw_response_text,
                parsed_response=parsed_response,
                success=success,
                error_message=error_message,
                latency_seconds=latency,
                finish_reason=finish_reason,
                usage=usage,
                response_metadata={
                    "prompt_version": built_prompt.prompt_version,
                    "prompt_mode": built_prompt.prompt_mode,
                    "schema_name": schema_name,
                    "response_id": getattr(response, "id", None),
                },
            )

        except Exception as error:
            latency = time.perf_counter() - started
            return LLMCallResult(
                provider=self.config.provider,
                model_name=self.config.model_name,
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                raw_response_text="",
                parsed_response=None,
                success=False,
                error_message=str(error),
                latency_seconds=latency,
                finish_reason=None,
                usage={},
                response_metadata={
                    "prompt_version": built_prompt.prompt_version,
                    "prompt_mode": built_prompt.prompt_mode,
                    "schema_name": schema_name,
                },
            )

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        """Pull usage metadata into a plain dict if available."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        return {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _extract_finish_reason(self, response: Any) -> str | None:
        """Extract a finish reason when present."""
        output = getattr(response, "output", None)
        if not output:
            return None

        first_item = output[0]
        status = getattr(first_item, "status", None)
        if status is not None:
            return str(status)

        return None


def build_llm_client(
    config: LLMClientConfig,
    api_key: str | None = None,
) -> StructuredLLMClient:
    """Instantiate the configured structured LLM client."""
    provider = str(config.provider).strip().lower()
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {config.provider}")
    return OpenAIStructuredClient(config=config, api_key=api_key)
