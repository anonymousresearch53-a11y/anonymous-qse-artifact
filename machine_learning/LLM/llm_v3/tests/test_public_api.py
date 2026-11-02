from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.LLM.llm_v3 import objective
from machine_learning.LLM.llm_v3.data import (
    ORIGINAL_CHAIN_ID,
    build_candidate_chains,
    candidate_chain_lookup,
)
from machine_learning.LLM.llm_v3.features import (
    BASIC_SIZE_FEATURES,
    build_query_feature_frame,
    extract_query_features,
)
from machine_learning.LLM.llm_v3.llm_client import LLMCallResult
from machine_learning.LLM.llm_v3.objective import ObjectiveConfig
from machine_learning.LLM.llm_v3.predictor import ChainSelectionPredictor, PredictorConfig
from machine_learning.LLM.llm_v3.prompting import build_prompt
from machine_learning.LLM.llm_v3.retrieval import RetrievedNeighbor, TrainingCircuitRetriever
from machine_learning.LLM.llm_v3.schemas import (
    ChainSelectionResponse,
    chain_selection_json_schema,
    validate_chain_selection_response,
)


def _base_feature_row(
    circuit_id: str,
    chain_name: str,
    *,
    depth: float,
    two_qubit: float,
    qubits: float = 4.0,
) -> dict[str, object]:
    return {
        "original_circuit": circuit_id,
        "chain_name": chain_name,
        "initial_num_qubits": qubits,
        "initial_depth": depth,
        "initial_two_qubit_gates": two_qubit,
        "initial_total_gates": 12.0,
        "initial_two_qubit_depth": max(two_qubit - 1.0, 0.0),
        "initial_gate_density": 0.5,
        "initial_two_qubit_ratio": 0.25,
        "depth_reduction_so_far": 99.0,
        "depth_reduction_so_far_fraction": 0.75,
        "two_qubit_gate_reduction_so_far": 88.0,
        "two_qubit_gate_reduction_so_far_fraction": 0.5,
    }


def _candidate_lookup() -> dict[str, dict[str, object]]:
    return {
        "chain_00": {
            "chain_id": "chain_00",
            "chain": [],
            "chain_name": "__original__",
            "length": 0,
        },
        "chain_01": {
            "chain_id": "chain_01",
            "chain": ["opt_a"],
            "chain_name": "opt_a",
            "length": 1,
        },
        "chain_02": {
            "chain_id": "chain_02",
            "chain": ["opt_b"],
            "chain_name": "opt_b",
            "length": 1,
        },
    }


def _candidate_profiles() -> dict[str, dict[str, float]]:
    return {
        "chain_00": {
            "avg_objective_score": 0.0,
            "avg_two_qubit_gate_reduction_pct": 0.0,
            "avg_depth_reduction_pct": 0.0,
            "avg_runtime_seconds": 0.0,
            "top_1_rate": 0.0,
        },
        "chain_01": {
            "avg_objective_score": 1.0,
            "avg_two_qubit_gate_reduction_pct": 10.0,
            "avg_depth_reduction_pct": 5.0,
            "avg_runtime_seconds": 0.4,
            "top_1_rate": 0.5,
        },
        "chain_02": {
            "avg_objective_score": 0.8,
            "avg_two_qubit_gate_reduction_pct": 8.0,
            "avg_depth_reduction_pct": 4.0,
            "avg_runtime_seconds": 0.6,
            "top_1_rate": 0.2,
        },
    }


@dataclass
class _FakeRetriever:
    neighbors: list[RetrievedNeighbor]

    def retrieve(self, query_row: dict[str, object]) -> list[RetrievedNeighbor]:
        del query_row
        return list(self.neighbors)


class _FakeClient:
    def __init__(self, results: list[LLMCallResult]) -> None:
        self.results = list(results)
        self.prompts = []

    def generate_structured(self, built_prompt, response_schema, schema_name):
        self.prompts.append(
            {
                "built_prompt": built_prompt,
                "response_schema": response_schema,
                "schema_name": schema_name,
            }
        )
        if not self.results:
            raise AssertionError("No fake LLM results remaining.")
        return self.results.pop(0)


def _llm_result(
    parsed_response: dict[str, object] | None,
    *,
    success: bool = True,
    error_message: str | None = None,
) -> LLMCallResult:
    return LLMCallResult(
        provider="mock",
        model_name="mock-model",
        system_prompt="system",
        user_prompt="user",
        raw_response_text="{}",
        parsed_response=parsed_response,
        success=success,
        error_message=error_message,
        latency_seconds=0.01,
        finish_reason="stop",
    )


def test_build_candidate_chains_and_lookup_cover_original_single_and_two_step() -> None:
    candidates = build_candidate_chains(["a", "b"])

    assert len(candidates) == 7
    assert candidates[0]["chain_id"] == ORIGINAL_CHAIN_ID
    assert candidates[0]["chain"] == []
    assert {tuple(candidate["chain"]) for candidate in candidates} == {
        (),
        ("a",),
        ("b",),
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    }

    lookup = candidate_chain_lookup(["a", "b"])
    assert lookup["chain_01"]["chain"] == ["a"]
    assert lookup["chain_06"]["chain"] == ["b", "b"]


@pytest.mark.parametrize("subset_name", ["plus_all_features", "rl_v3_all_features"])
def test_extract_query_features_normalizes_aliases_and_zeroes_progress(subset_name: str) -> None:
    row = _base_feature_row("circ_1", "__original__", depth=10.0, two_qubit=3.0)

    features = extract_query_features(row, feature_subset=subset_name)

    assert features["initial_depth"] == 10.0
    assert features["depth_reduction_so_far"] == 0.0
    assert features["two_qubit_gate_reduction_so_far_fraction"] == 0.0


def test_build_query_feature_frame_uses_one_representative_per_circuit() -> None:
    frame = pd.DataFrame(
        [
            _base_feature_row("circ_b", "z_chain", depth=20.0, two_qubit=4.0),
            _base_feature_row("circ_a", "b_chain", depth=10.0, two_qubit=2.0),
            _base_feature_row("circ_a", "a_chain", depth=11.0, two_qubit=3.0),
        ]
    )

    result = build_query_feature_frame(frame, feature_subset="basic_size")

    assert result["original_circuit"].tolist() == ["circ_a", "circ_b"]
    assert result.shape == (2, 1 + len(BASIC_SIZE_FEATURES))
    assert result.loc[result["original_circuit"] == "circ_a", "initial_depth"].item() == 11.0


def test_chain_selection_json_schema_requires_expected_fields() -> None:
    schema = chain_selection_json_schema()

    assert schema["type"] == "object"
    assert "selected_chain_id" in schema["required"]
    assert schema["properties"]["selected_chain"]["maxItems"] == 2


def test_validate_chain_selection_response_accepts_valid_payload() -> None:
    response = validate_chain_selection_response(
        {
            "selected_chain_id": "chain_01",
            "selected_chain": ["opt_a"],
            "predicted_tradeoff_summary": "Best compromise.",
            "reasoning_summary": "Train evidence favors opt_a.",
            "confidence": 0.7,
            "evidence_used": ["candidate_priors", "retrieved_neighbors"],
        },
        _candidate_lookup(),
    )

    assert isinstance(response, ChainSelectionResponse)
    assert response.selected_chain_id == "chain_01"
    assert response.selected_chain == ["opt_a"]


def test_validate_chain_selection_response_rejects_chain_mismatch() -> None:
    with pytest.raises(ValueError, match="selected_chain does not match"):
        validate_chain_selection_response(
            {
                "selected_chain_id": "chain_01",
                "selected_chain": ["opt_b"],
                "predicted_tradeoff_summary": "Mismatch.",
                "reasoning_summary": "Mismatch.",
                "confidence": 0.7,
                "evidence_used": ["candidate_priors"],
            },
            _candidate_lookup(),
        )


def test_build_prompt_includes_only_sections_for_selected_mode() -> None:
    neighbor = RetrievedNeighbor(
        circuit_id="train_1",
        distance=0.1,
        neighbor_feature_summary={"initial_depth": 8.0},
        best_chain_id="chain_01",
        best_chain_name="opt_a",
        best_chain_score=1.5,
        best_non_original_chain_id="chain_01",
        best_non_original_chain_name="opt_a",
        best_non_original_chain_score=1.5,
        original_score=0.2,
        best_vs_original_margin=1.3,
        top_outcomes=[
            {
                "chain_id": "chain_01",
                "chain_name": "opt_a",
                "chain": ["opt_a"],
                "score": 1.5,
                "two_qubit_reduction_pct": 10.0,
                "depth_reduction_pct": 5.0,
                "runtime_seconds": 0.4,
                "runtime_score": 0.1,
            }
        ],
    )

    prompt = build_prompt(
        query_circuit_id="query_1",
        query_features={"initial_depth": 10.0},
        candidate_chains=list(_candidate_lookup().values()),
        objective_config=ObjectiveConfig(),
        optimizer_docs={"opt_a": "Good for shallow circuits."},
        candidate_profile_lookup=_candidate_profiles(),
        retrieved_neighbors=[neighbor],
        prompt_mode="docs_only",
    )

    assert prompt.prompt_mode == "docs_only"
    assert "optimizer_docs" in prompt.prompt_sections
    assert "retrieved_neighbors" not in prompt.prompt_sections
    assert "Single-Optimizer Train Priors" in prompt.user_prompt


def test_score_functions_delegate_and_sum_component_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(objective, "single_step_reward", lambda *args, **kwargs: 1.25)
    monkeypatch.setattr(objective, "incremental_step_reward", lambda *args, **kwargs: 2.5)
    monkeypatch.setattr(objective, "compute_final_correction", lambda *args, **kwargs: 0.75)

    config = ObjectiveConfig()

    assert objective.score_original_circuit({}, config) == 0.0
    assert objective.score_single_chain({}, config) == pytest.approx(2.0)
    assert objective.score_two_step_chain({}, {}, config) == pytest.approx(4.5)


def test_training_circuit_retriever_returns_nearest_neighbors_and_metadata() -> None:
    train_frame = pd.DataFrame(
        [
            _base_feature_row("circ_a", "__original__", depth=10.0, two_qubit=2.0),
            _base_feature_row("circ_b", "__original__", depth=25.0, two_qubit=7.0),
        ]
    )

    score_rows = {
        "circ_a": pd.DataFrame(
            [
                {
                    "chain_id": "chain_01",
                    "chain_name": "opt_a",
                    "chain": ["opt_a"],
                    "length": 1,
                    "objective_score": 1.2,
                    "total_two_qubit_gate_reduction_pct": 12.0,
                    "total_depth_reduction_pct": 5.0,
                    "total_runtime_seconds": 0.5,
                    "total_runtime_score": 0.1,
                },
                {
                    "chain_id": "chain_00",
                    "chain_name": "__original__",
                    "chain": [],
                    "length": 0,
                    "objective_score": 0.2,
                    "total_two_qubit_gate_reduction_pct": 0.0,
                    "total_depth_reduction_pct": 0.0,
                    "total_runtime_seconds": 0.0,
                    "total_runtime_score": 0.0,
                },
            ]
        ),
        "circ_b": pd.DataFrame(
            [
                {
                    "chain_id": "chain_00",
                    "chain_name": "__original__",
                    "chain": [],
                    "length": 0,
                    "objective_score": 0.4,
                    "total_two_qubit_gate_reduction_pct": 0.0,
                    "total_depth_reduction_pct": 0.0,
                    "total_runtime_seconds": 0.0,
                    "total_runtime_score": 0.0,
                }
            ]
        ),
    }

    retriever = TrainingCircuitRetriever(
        train_frame=train_frame,
        feature_subset="basic_size",
        score_lookup_fn=lambda circuit_id: score_rows[circuit_id],
        k=1,
        top_outcomes=1,
    )

    neighbors = retriever.retrieve(_base_feature_row("query", "__original__", depth=11.0, two_qubit=2.0))

    assert len(neighbors) == 1
    assert neighbors[0].circuit_id == "circ_a"
    assert neighbors[0].best_chain_id == "chain_01"
    assert neighbors[0].top_outcomes[0]["chain_id"] == "chain_01"
    assert retriever.metadata()["retrieval_k"] == 1


def test_predictor_returns_model_selection_when_response_matches_shortlist() -> None:
    client = _FakeClient(
        [
            _llm_result(
                {
                    "selected_chain_id": "chain_01",
                    "selected_chain": ["opt_a"],
                    "predicted_tradeoff_summary": "Best tradeoff.",
                    "reasoning_summary": "Evidence favors opt_a.",
                    "confidence": 0.9,
                    "evidence_used": ["candidate_priors"],
                }
            )
        ]
    )

    predictor = ChainSelectionPredictor(
        retriever=_FakeRetriever([]),
        llm_client=client,
        optimizer_docs={"opt_a": "Good default."},
        objective_config=ObjectiveConfig(),
        candidate_lookup=_candidate_lookup(),
        candidate_profile_lookup=_candidate_profiles(),
        config=PredictorConfig(prompt_mode="no_retrieval", shortlist_size=None),
    )

    result = predictor.predict(_base_feature_row("query_1", "__original__", depth=10.0, two_qubit=2.0))

    assert result.used_fallback is False
    assert result.final_selected_chain_id == "chain_01"
    assert result.validated_response is not None
    assert client.prompts[0]["schema_name"] == "chain_selection_v1"


def test_predictor_falls_back_when_model_selects_chain_outside_shortlist() -> None:
    client = _FakeClient(
        [
            _llm_result(
                {
                    "selected_chain_id": "chain_02",
                    "selected_chain": ["opt_b"],
                    "predicted_tradeoff_summary": "Out-of-shortlist answer.",
                    "reasoning_summary": "Should fail validation.",
                    "confidence": 0.95,
                    "evidence_used": ["candidate_priors"],
                }
            )
        ]
    )

    predictor = ChainSelectionPredictor(
        retriever=_FakeRetriever([]),
        llm_client=client,
        optimizer_docs={"opt_a": "Good default."},
        objective_config=ObjectiveConfig(),
        candidate_lookup=_candidate_lookup(),
        candidate_profile_lookup=_candidate_profiles(),
        config=PredictorConfig(
            prompt_mode="no_retrieval",
            shortlist_size=2,
            enable_retry_on_failure=False,
            include_original_in_shortlist=False,
            fallback_chain_id="chain_00",
        ),
    )

    result = predictor.predict(_base_feature_row("query_1", "__original__", depth=10.0, two_qubit=2.0))

    assert result.used_fallback is True
    assert result.final_selected_chain_id == "chain_00"
    assert result.failure_reason is not None
    assert "Unknown selected_chain_id" in result.failure_reason
