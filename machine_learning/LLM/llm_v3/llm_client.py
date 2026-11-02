"""Structured LLM clients for chain-selection experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import hashlib
import json
import os
import time

try:
    from machine_learning.LLM.llm_v3.prompting import BuiltPrompt
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
    cache_dir: str | None = None
    use_cache: bool = True


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
    from_cache: bool = False


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


class BaseStructuredClient:
    """Shared helpers for concrete structured clients."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self.cache_dir = None
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir).expanduser().resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        payload = {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": float(self.config.temperature),
            "max_output_tokens": int(self.config.max_output_tokens),
            "schema_name": schema_name,
            "response_schema": response_schema,
            "system_prompt": built_prompt.system_prompt,
            "user_prompt": built_prompt.user_prompt,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def _load_cached_result(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> LLMCallResult | None:
        if not self.config.use_cache:
            return None
        cache_key = self._cache_key(built_prompt, response_schema, schema_name)
        cache_path = self._cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text())
        result = LLMCallResult(**payload)
        result.from_cache = True
        result.response_metadata = dict(result.response_metadata)
        result.response_metadata["cache_key"] = cache_key
        return result

    def _save_cached_result(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
        result: LLMCallResult,
    ) -> None:
        if not self.config.use_cache:
            return
        cache_key = self._cache_key(built_prompt, response_schema, schema_name)
        cache_path = self._cache_path(cache_key)
        if cache_path is None:
            return
        payload = asdict(result)
        payload["from_cache"] = False
        cache_path.write_text(json.dumps(payload, indent=2))


class OpenAIStructuredClient(BaseStructuredClient):
    """OpenAI implementation for structured JSON-schema responses."""

    def __init__(
        self,
        config: LLMClientConfig,
        api_key: str | None = None,
    ) -> None:
        super().__init__(config)
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIStructuredClient.")

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
        cached_result = self._load_cached_result(
            built_prompt=built_prompt,
            response_schema=response_schema,
            schema_name=schema_name,
        )
        if cached_result is not None:
            return cached_result

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

            result = LLMCallResult(
                provider=self.config.provider,
                model_name=self.config.model_name,
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                raw_response_text=raw_response_text,
                parsed_response=parsed_response,
                success=success,
                error_message=error_message,
                latency_seconds=latency,
                finish_reason=self._extract_finish_reason(response),
                usage=self._extract_usage(response),
                response_metadata={
                    "prompt_version": built_prompt.prompt_version,
                    "prompt_mode": built_prompt.prompt_mode,
                    "schema_name": schema_name,
                    "response_id": getattr(response, "id", None),
                },
                from_cache=False,
            )
            self._save_cached_result(
                built_prompt=built_prompt,
                response_schema=response_schema,
                schema_name=schema_name,
                result=result,
            )
            return result

        except Exception as error:
            return LLMCallResult(
                provider=self.config.provider,
                model_name=self.config.model_name,
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                raw_response_text="",
                parsed_response=None,
                success=False,
                error_message=str(error),
                latency_seconds=time.perf_counter() - started,
                finish_reason=None,
                usage={},
                response_metadata={
                    "prompt_version": built_prompt.prompt_version,
                    "prompt_mode": built_prompt.prompt_mode,
                    "schema_name": schema_name,
                },
                from_cache=False,
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


class MockStructuredClient(BaseStructuredClient):
    """Local structured client for smoke tests without network access."""

    def generate_structured(
        self,
        built_prompt: BuiltPrompt,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> LLMCallResult:
        """Return a deterministic valid response based on the first shown candidate."""
        del response_schema, schema_name
        started = time.perf_counter()
        candidate_section = str(
            built_prompt.prompt_sections.get("candidate_chains", "")
        ).splitlines()
        selected_chain_id = "chain_00"
        selected_chain: list[str] = []
        for line in candidate_section:
            stripped = line.strip()
            if not stripped.startswith("chain_"):
                continue
            chain_id, remainder = stripped.split(":", 1)
            selected_chain_id = chain_id.strip()
            chain_label = remainder.split("|", 1)[0].strip()
            if chain_label.lower() == "no optimization":
                selected_chain = []
            else:
                selected_chain = [part.strip() for part in chain_label.split("->")]
            break

        parsed_response = {
            "selected_chain_id": selected_chain_id,
            "selected_chain": selected_chain,
            "predicted_tradeoff_summary": "Mock client selected the first available candidate.",
            "reasoning_summary": "This deterministic local client is only for pipeline smoke tests.",
            "confidence": 0.99,
            "evidence_used": ["mock_provider"],
        }
        raw_response_text = json.dumps(parsed_response)
        return LLMCallResult(
            provider="mock",
            model_name=self.config.model_name,
            system_prompt=built_prompt.system_prompt,
            user_prompt=built_prompt.user_prompt,
            raw_response_text=raw_response_text,
            parsed_response=parsed_response,
            success=True,
            error_message=None,
            latency_seconds=time.perf_counter() - started,
            finish_reason="mock_completed",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            response_metadata={
                "prompt_version": built_prompt.prompt_version,
                "prompt_mode": built_prompt.prompt_mode,
                "mock_strategy": "first_presented_candidate",
            },
            from_cache=False,
        )


def build_llm_client(
    config: LLMClientConfig,
    api_key: str | None = None,
) -> StructuredLLMClient:
    """Instantiate the configured structured LLM client."""
    provider = str(config.provider).strip().lower()
    if provider == "openai":
        return OpenAIStructuredClient(config=config, api_key=api_key)
    if provider == "mock":
        return MockStructuredClient(config=config)
    raise ValueError(f"Unsupported provider: {config.provider}")
