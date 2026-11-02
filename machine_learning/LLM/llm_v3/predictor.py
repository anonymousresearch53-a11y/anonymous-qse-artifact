"""Single-circuit prediction pipeline for LLM-based chain selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from machine_learning.LLM.llm_v3.candidate_ranker import LearnedCandidateRanker
    from machine_learning.LLM.llm_v3.data import (
        ORIGINAL_CHAIN_ID,
        candidate_chain_lookup,
    )
    from machine_learning.LLM.llm_v3.features import extract_query_features
    from machine_learning.LLM.llm_v3.llm_client import (
        LLMCallResult,
        StructuredLLMClient,
    )
    from machine_learning.LLM.llm_v3.optimization_gate import GateDecision, OptimizationGate
    from machine_learning.LLM.llm_v3.objective import ObjectiveConfig
    from machine_learning.LLM.llm_v3.prompting import BuiltPrompt, build_prompt
    from machine_learning.LLM.llm_v3.retrieval import (
        RetrievedNeighbor,
        TrainingCircuitRetriever,
    )
    from machine_learning.LLM.llm_v3.schemas import (
        ChainSelectionResponse,
        chain_selection_json_schema,
        validate_chain_selection_response,
    )
except ImportError:
    from .candidate_ranker import LearnedCandidateRanker
    from .data import ORIGINAL_CHAIN_ID, candidate_chain_lookup
    from .features import extract_query_features
    from .llm_client import LLMCallResult, StructuredLLMClient
    from .optimization_gate import GateDecision, OptimizationGate
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
    fallback_chain_id: str = ORIGINAL_CHAIN_ID
    min_confidence: float = 0.35
    shortlist_size: int | None = 10
    include_original_in_shortlist: bool = True
    include_query_circuit_id: bool = True
    include_neighbor_identity: bool = True


@dataclass
class PredictionResult:
    """Full prediction artifact for one query circuit."""

    circuit_id: str
    query_features: dict[str, float]
    retrieved_neighbors: list[RetrievedNeighbor]
    shortlisted_candidates: list[dict[str, Any]]
    candidate_rank_scores: dict[str, float]
    gate_decision: dict[str, Any] | None
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
        candidate_profile_lookup: dict[str, dict[str, Any]] | None = None,
        candidate_ranker: LearnedCandidateRanker | None = None,
        optimization_gate: OptimizationGate | None = None,
        global_best_average_chain_id: str | None = None,
        config: PredictorConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.optimizer_docs = dict(optimizer_docs)
        self.objective_config = objective_config
        self.candidate_lookup = candidate_lookup or candidate_chain_lookup()
        self.candidate_profile_lookup = candidate_profile_lookup or {}
        self.candidate_ranker = candidate_ranker
        self.optimization_gate = optimization_gate
        self.global_best_average_chain_id = global_best_average_chain_id
        self.config = config or PredictorConfig()
        if self.config.fallback_chain_id not in self.candidate_lookup:
            raise KeyError(
                f"Unknown fallback_chain_id: {self.config.fallback_chain_id}"
            )

    def predict(self, query_row: dict[str, Any]) -> PredictionResult:
        """Run the full prediction pipeline for one query circuit."""
        circuit_id = str(query_row["original_circuit"])
        query_features = extract_query_features(
            query_row,
            feature_subset=self.config.feature_subset,
        )

        retrieved_neighbors = self._maybe_retrieve_neighbors(query_row)
        gate_decision = self._predict_gate(query_row)
        shortlisted_candidates, candidate_rank_scores = self._build_shortlist(
            query_row,
            gate_decision,
            retrieved_neighbors,
        )
        built_prompt = build_prompt(
            query_circuit_id=circuit_id,
            query_features=query_features,
            candidate_chains=shortlisted_candidates,
            objective_config=self.objective_config,
            optimizer_docs=self.optimizer_docs,
            candidate_profile_lookup=self.candidate_profile_lookup,
            gate_decision=gate_decision,
            retrieved_neighbors=retrieved_neighbors,
            prompt_mode=self.config.prompt_mode,
            include_query_circuit_id=self.config.include_query_circuit_id,
            include_neighbor_identity=self.config.include_neighbor_identity,
        )
        shortlist_lookup = {
            str(candidate["chain_id"]): candidate
            for candidate in shortlisted_candidates
        }

        validated_response, failure_reason, llm_call_results = (
            self._validate_with_optional_retry(
                built_prompt,
                shortlist_lookup,
            )
        )

        if validated_response is not None:
            return PredictionResult(
                circuit_id=circuit_id,
                query_features=query_features,
                retrieved_neighbors=retrieved_neighbors,
                shortlisted_candidates=shortlisted_candidates,
                candidate_rank_scores=candidate_rank_scores,
                gate_decision=gate_decision,
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
            shortlisted_candidates=shortlisted_candidates,
            candidate_rank_scores=candidate_rank_scores,
            gate_decision=gate_decision,
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

    def _build_shortlist(
        self,
        query_row: dict[str, Any],
        gate_decision: dict[str, Any] | None,
        retrieved_neighbors: list[RetrievedNeighbor],
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Rank candidates using a learned ranker, or fall back to heuristics."""
        ranked_candidates = list(self.candidate_lookup.values())
        shortlist_size = self.config.shortlist_size
        if shortlist_size is None or shortlist_size >= len(ranked_candidates):
            ordered = self._sort_candidates_for_prompt(ranked_candidates)
            return ordered, {
                str(candidate["chain_id"]): float(
                    self.candidate_profile_lookup.get(candidate["chain_id"], {}).get(
                        "avg_objective_score",
                        0.0,
                    )
                )
                for candidate in ordered
            }

        if self.candidate_ranker is not None:
            ranked_result = self.candidate_ranker.score_query(query_row)
            selected_ids = self._protected_shortlist_ids(gate_decision)
            for chain_id in ranked_result.ranked_chain_ids:
                if chain_id not in selected_ids:
                    selected_ids.append(chain_id)
                if len(selected_ids) >= shortlist_size:
                    break
            shortlisted = [self.candidate_lookup[chain_id] for chain_id in selected_ids]
            return self._sort_candidates_for_prompt(shortlisted), ranked_result.chain_scores

        candidate_scores: dict[str, float] = {
            chain_id: 0.0 for chain_id in self.candidate_lookup
        }

        for chain_id, profile in self.candidate_profile_lookup.items():
            candidate_scores[chain_id] += float(profile.get("avg_objective_score", 0.0))
            candidate_scores[chain_id] += 0.25 * float(profile.get("top_1_rate", 0.0))

        for neighbor in retrieved_neighbors:
            for rank_index, outcome in enumerate(neighbor.top_outcomes):
                bonus = max(0.0, 1.0 - 0.15 * rank_index)
                candidate_scores[str(outcome["chain_id"])] = (
                    candidate_scores.get(str(outcome["chain_id"]), 0.0) + bonus
                )
            candidate_scores[str(neighbor.best_chain_id)] = (
                candidate_scores.get(str(neighbor.best_chain_id), 0.0) + 0.5
            )

        ranked_ids = sorted(
            self.candidate_lookup,
            key=lambda chain_id: (
                candidate_scores.get(chain_id, 0.0),
                float(
                    self.candidate_profile_lookup.get(chain_id, {}).get(
                        "avg_two_qubit_gate_reduction_pct",
                        0.0,
                    )
                ),
                chain_id,
            ),
            reverse=True,
        )

        selected_ids = self._protected_shortlist_ids(gate_decision)

        for chain_id in ranked_ids:
            if chain_id not in selected_ids:
                selected_ids.append(chain_id)
            if len(selected_ids) >= shortlist_size:
                break

        shortlisted = [self.candidate_lookup[chain_id] for chain_id in selected_ids]
        return self._sort_candidates_for_prompt(shortlisted), candidate_scores

    def _protected_shortlist_ids(
        self,
        gate_decision: dict[str, Any] | None,
    ) -> list[str]:
        """Return chains that must always remain available to the LLM."""
        selected_ids: list[str] = []
        protected_ids: list[str | None] = []
        if gate_decision and gate_decision.get("recommendation_label") == "optimize":
            protected_ids.extend(
                [
                    self.global_best_average_chain_id,
                    self.config.fallback_chain_id,
                ]
            )
        else:
            protected_ids.extend(
                [
                    self.config.fallback_chain_id,
                    self.global_best_average_chain_id,
                ]
            )
        if self.config.include_original_in_shortlist:
            protected_ids.append(ORIGINAL_CHAIN_ID)
        for chain_id in protected_ids:
            if chain_id and chain_id in self.candidate_lookup and chain_id not in selected_ids:
                selected_ids.append(chain_id)
        return selected_ids

    def _predict_gate(self, query_row: dict[str, Any]) -> dict[str, Any] | None:
        """Return gate probabilities when a gate is configured."""
        if self.optimization_gate is None:
            return None
        decision: GateDecision = self.optimization_gate.predict_query(query_row)
        return {
            "recommend_optimization": bool(decision.recommend_optimization),
            "probability_original": float(decision.probability_original),
            "probability_optimize": float(decision.probability_optimize),
            "recommendation_label": str(decision.recommendation_label),
        }

    def _sort_candidates_for_prompt(
        self,
        candidate_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep prompt ordering stable and profile-aware."""
        return sorted(
            candidate_rows,
            key=lambda candidate: (
                candidate["length"] == 0,
                -float(
                    self.candidate_profile_lookup.get(candidate["chain_id"], {}).get(
                        "avg_objective_score",
                        -1e9,
                    )
                ),
                candidate["chain_id"],
            ),
        )

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
        shortlist_lookup: dict[str, dict[str, Any]],
    ) -> tuple[ChainSelectionResponse | None, str | None, list[LLMCallResult]]:
        """Validate the model result and optionally retry once on failure."""
        llm_call_results: list[LLMCallResult] = []

        initial_result = self._call_model(built_prompt)
        llm_call_results.append(initial_result)
        validated_response, failure_reason = self._validate_result(
            initial_result,
            shortlist_lookup,
        )
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
            retry_result,
            shortlist_lookup,
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
        shortlist_lookup: dict[str, dict[str, Any]],
    ) -> tuple[ChainSelectionResponse | None, str | None]:
        """Validate one LLM call result."""
        if not llm_call_result.success:
            return None, llm_call_result.error_message or "llm_call_failed"

        if llm_call_result.parsed_response is None:
            return None, "parsed_response_missing"

        try:
            validated = validate_chain_selection_response(
                llm_call_result.parsed_response,
                shortlist_lookup,
            )
        except Exception as error:
            return None, str(error)

        if float(validated.confidence) < float(self.config.min_confidence):
            return None, (
                "confidence_below_threshold: "
                f"{float(validated.confidence):.3f} < {float(self.config.min_confidence):.3f}"
            )
        return validated, None

    def _build_retry_prompt(
        self,
        built_prompt: BuiltPrompt,
        failure_reason: str,
    ) -> BuiltPrompt:
        """Append a strict validation reminder for one retry attempt."""
        retry_user_prompt = built_prompt.user_prompt
        retry_system_prompt = (
            f"{built_prompt.system_prompt}\n"
            "Your previous response failed validation because of formatting or schema compliance. "
            "Fix the format, but do not cite the validation issue as evidence. "
            "Only use circuit features, candidate priors, and similar-circuit evidence.\n"
            f"Validation note for formatting only: {failure_reason}"
        )
        return BuiltPrompt(
            system_prompt=retry_system_prompt,
            user_prompt=retry_user_prompt,
            prompt_version=built_prompt.prompt_version,
            prompt_mode=built_prompt.prompt_mode,
            prompt_sections=dict(built_prompt.prompt_sections),
        )
