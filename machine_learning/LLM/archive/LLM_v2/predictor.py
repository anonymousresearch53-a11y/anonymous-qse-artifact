"""Single-circuit prediction pipeline for LLM-based chain selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from machine_learning.LLM.LLM_v2.data import candidate_chain_lookup
    from machine_learning.LLM.LLM_v2.features import extract_query_features
    from machine_learning.LLM.LLM_v2.llm_client import LLMCallResult, StructuredLLMClient
    from machine_learning.LLM.LLM_v2.objective import ObjectiveConfig
    from machine_learning.LLM.LLM_v2.prompting import BuiltPrompt, build_prompt
    from machine_learning.LLM.LLM_v2.retrieval import (
        RetrievedNeighbor,
        TrainingCircuitRetriever,
    )
    from machine_learning.LLM.LLM_v2.schemas import (
        ChainSelectionResponse,
        chain_selection_json_schema,
        validate_chain_selection_response,
    )
except ImportError:
    from .data import candidate_chain_lookup
    from .features import extract_query_features
    from .llm_client import LLMCallResult, StructuredLLMClient
    from .objective import ObjectiveConfig
    from .prompting import BuiltPrompt, build_prompt
    from .retrieval import RetrievedNeighbor, TrainingCircuitRetriever
    from .schemas import (
        ChainSelectionResponse,
        chain_selection_json_schema,
        validate_chain_selection_response,
    )


DEFAULT_SCHEMA_NAME = "chain_selection_v1"


@dataclass(frozen=True)
class PredictorConfig:
    """Configuration for one predictor instance."""

    feature_subset: str = "full_structural"
    prompt_mode: str = "full_evidence"
    schema_name: str = DEFAULT_SCHEMA_NAME
    enable_retry_on_failure: bool = True
    fallback_chain_id: str = "chain_01"


@dataclass
class PredictionResult:
    """Full prediction artifact for one query circuit."""

    circuit_id: str
    query_features: dict[str, float]
    retrieved_neighbors: list[RetrievedNeighbor]
    built_prompt: BuiltPrompt
    llm_call_results: list[LLMCallResult]
    validated_response: ChainSelectionResponse | None
    final_selected_chain_id: str
    final_selected_chain: list[str]
    used_fallback: bool
    failure_reason: str | None
    attempt_count: int


class ChainSelectionPredictor:
    """Orchestrates retrieval, prompting, model call, validation, and fallback."""

    def __init__(
        self,
        retriever: TrainingCircuitRetriever,
        llm_client: StructuredLLMClient,
        optimizer_docs: dict[str, str],
        objective_config: ObjectiveConfig,
        candidate_lookup: dict[str, dict[str, Any]] | None = None,
        config: PredictorConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.optimizer_docs = dict(optimizer_docs)
        self.objective_config = objective_config
        self.candidate_lookup = candidate_lookup or candidate_chain_lookup()
        self.config = config or PredictorConfig()

    def predict(self, query_row: dict[str, Any]) -> PredictionResult:
        """Run the full prediction pipeline for one query circuit."""
        circuit_id = str(query_row["original_circuit"])
        query_features = extract_query_features(
            query_row,
            feature_subset=self.config.feature_subset,
        )

        retrieved_neighbors = self._maybe_retrieve_neighbors(query_row)
        built_prompt = build_prompt(
            query_circuit_id=circuit_id,
            query_features=query_features,
            candidate_chains=list(self.candidate_lookup.values()),
            objective_config=self.objective_config,
            optimizer_docs=self.optimizer_docs,
            retrieved_neighbors=retrieved_neighbors,
            prompt_mode=self.config.prompt_mode,
        )

        validated_response, failure_reason, llm_call_results = (
            self._validate_with_optional_retry(built_prompt)
        )

        if validated_response is not None:
            return PredictionResult(
                circuit_id=circuit_id,
                query_features=query_features,
                retrieved_neighbors=retrieved_neighbors,
                built_prompt=built_prompt,
                llm_call_results=llm_call_results,
                validated_response=validated_response,
                final_selected_chain_id=validated_response.selected_chain_id,
                final_selected_chain=list(validated_response.selected_chain),
                used_fallback=False,
                failure_reason=None,
                attempt_count=len(llm_call_results),
            )

        fallback_chain = self.candidate_lookup[self.config.fallback_chain_id]["chain"]
        return PredictionResult(
            circuit_id=circuit_id,
            query_features=query_features,
            retrieved_neighbors=retrieved_neighbors,
            built_prompt=built_prompt,
            llm_call_results=llm_call_results,
            validated_response=None,
            final_selected_chain_id=self.config.fallback_chain_id,
            final_selected_chain=list(fallback_chain),
            used_fallback=True,
            failure_reason=failure_reason,
            attempt_count=len(llm_call_results),
        )

    def _maybe_retrieve_neighbors(
        self,
        query_row: dict[str, Any],
    ) -> list[RetrievedNeighbor]:
        """Retrieve neighbors only when the chosen prompt mode uses them."""
        if self.config.prompt_mode in {"docs_only", "no_retrieval"}:
            return []
        return self.retriever.retrieve(query_row)

    def _call_model(self, built_prompt: BuiltPrompt) -> LLMCallResult:
        """Send one structured request to the LLM client."""
        return self.llm_client.generate_structured(
            built_prompt=built_prompt,
            response_schema=chain_selection_json_schema(),
            schema_name=self.config.schema_name,
        )

    def _validate_with_optional_retry(
        self,
        built_prompt: BuiltPrompt,
    ) -> tuple[ChainSelectionResponse | None, str | None, list[LLMCallResult]]:
        """Validate the model result and optionally retry once on failure."""
        llm_call_results: list[LLMCallResult] = []

        initial_result = self._call_model(built_prompt)
        llm_call_results.append(initial_result)
        validated_response, failure_reason = self._validate_result(initial_result)
        if validated_response is not None:
            return validated_response, None, llm_call_results

        if not self.config.enable_retry_on_failure:
            return None, failure_reason, llm_call_results

        retry_prompt = self._build_retry_prompt(
            built_prompt,
            failure_reason or "unknown",
        )
        retry_result = self._call_model(retry_prompt)
        llm_call_results.append(retry_result)
        retry_validated_response, retry_failure_reason = self._validate_result(
            retry_result
        )

        if retry_validated_response is not None:
            return retry_validated_response, None, llm_call_results

        combined_failure_reason = (
            f"initial_failure={failure_reason}; retry_failure={retry_failure_reason}"
        )
        return None, combined_failure_reason, llm_call_results

    def _validate_result(
        self,
        llm_call_result: LLMCallResult,
    ) -> tuple[ChainSelectionResponse | None, str | None]:
        """Validate one LLM call result."""
        if not llm_call_result.success:
            return None, llm_call_result.error_message or "llm_call_failed"

        if llm_call_result.parsed_response is None:
            return None, "parsed_response_missing"

        try:
            validated = validate_chain_selection_response(
                llm_call_result.parsed_response,
                self.candidate_lookup,
            )
            return validated, None
        except Exception as error:
            return None, str(error)

    def _build_retry_prompt(
        self,
        built_prompt: BuiltPrompt,
        failure_reason: str,
    ) -> BuiltPrompt:
        """Append a strict validation reminder for one retry attempt."""
        retry_user_prompt = (
            f"{built_prompt.user_prompt}\n\n"
            "Previous response was invalid.\n"
            f"Validation failure: {failure_reason}\n"
            "Return a valid response that exactly matches the required schema "
            "and choose one provided chain_id with the correct matching chain."
        )
        return BuiltPrompt(
            system_prompt=built_prompt.system_prompt,
            user_prompt=retry_user_prompt,
            prompt_version=built_prompt.prompt_version,
            prompt_mode=built_prompt.prompt_mode,
            prompt_sections=dict(built_prompt.prompt_sections),
        )
