"""Run one real LLM prediction for one circuit and print performance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from machine_learning.LLM.LLM_v2.data import (
        candidate_rows_for_circuit,
        circuit_representatives,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from machine_learning.LLM.LLM_v2.evaluate import (
        evaluate_selection,
        score_circuit_candidates,
    )
    from machine_learning.LLM.LLM_v2.llm_client import LLMClientConfig, build_llm_client
    from machine_learning.LLM.LLM_v2.objective import ObjectiveConfig
    from machine_learning.LLM.LLM_v2.predictor import ChainSelectionPredictor, PredictorConfig
    from machine_learning.LLM.LLM_v2.retrieval import TrainingCircuitRetriever
except ImportError:
    from .data import (
        candidate_rows_for_circuit,
        circuit_representatives,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from .evaluate import evaluate_selection, score_circuit_candidates
    from .llm_client import LLMClientConfig, build_llm_client
    from .objective import ObjectiveConfig
    from .predictor import ChainSelectionPredictor, PredictorConfig
    from .retrieval import TrainingCircuitRetriever


def _load_env_file(repo_root: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_optimizer_docs() -> dict[str, str]:
    """Return temporary local optimizer descriptions for prompting."""
    return {
        "qiskit_standard": "Standard Qiskit transpilation baseline.",
        "qiskit_ai": "Model-guided Qiskit optimization pipeline.",
        "tket": "TKET-based optimization pipeline.",
        "wisq_rules": "Rule-based WISQ optimization variant.",
        "wisq_bqskit": "WISQ followed by BQSKit-oriented optimization.",
    }


def _build_training_score_cache(
    train_df: pd.DataFrame,
    objective_config: ObjectiveConfig,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Score all training circuits once and return a cache plus best-average chain id."""
    score_cache: dict[str, pd.DataFrame] = {}
    chain_score_totals: dict[str, float] = {}
    chain_score_counts: dict[str, int] = {}

    for circuit_id in sorted(train_df["original_circuit"].unique().tolist()):
        circuit_rows = candidate_rows_for_circuit(train_df, circuit_id)
        scored_rows = score_circuit_candidates(circuit_rows, objective_config)
        score_cache[circuit_id] = scored_rows

        for _, row in scored_rows.iterrows():
            chain_id = str(row["chain_id"])
            chain_score_totals[chain_id] = chain_score_totals.get(chain_id, 0.0) + float(
                row["objective_score"]
            )
            chain_score_counts[chain_id] = chain_score_counts.get(chain_id, 0) + 1

    average_scores = {
        chain_id: chain_score_totals[chain_id] / chain_score_counts[chain_id]
        for chain_id in chain_score_totals
        if chain_score_counts[chain_id] > 0
    }
    global_best_average_chain_id = max(
        average_scores,
        key=lambda chain_id: (average_scores[chain_id], chain_id),
    )
    return score_cache, global_best_average_chain_id


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run one LLM-based chain selection and print its performance."
    )
    parser.add_argument("--circuit-id", required=True, help="Original circuit id to test.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test", "train"],
        help="Dataset split containing the query circuit.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.4-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--feature-subset",
        default="full_structural",
        help="Feature subset name used for prompt and retrieval.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="full_evidence",
        choices=["full_evidence", "docs_only", "similar_circuits_only", "no_retrieval"],
        help="Prompt evidence mode.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of retrieved neighbors.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum response tokens.",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Print the full raw JSON model response.",
    )
    return parser


def main() -> None:
    """Run one real prediction and print the result plus measured performance."""
    args = _build_argument_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    _load_env_file(repo_root)

    bundle = load_dataset_bundle()
    train_df = split_frame_for_name(bundle, "train")
    eval_df = split_frame_for_name(bundle, args.split)

    objective_config = ObjectiveConfig()
    score_cache, global_best_average_chain_id = _build_training_score_cache(
        train_df,
        objective_config,
    )

    def score_lookup_fn(circuit_id: str) -> pd.DataFrame:
        return score_cache[circuit_id]

    retriever = TrainingCircuitRetriever(
        train_frame=train_df,
        feature_subset=args.feature_subset,
        score_lookup_fn=score_lookup_fn,
        k=args.k,
        top_outcomes=3,
    )

    llm_client = build_llm_client(
        LLMClientConfig(
            model_name=args.model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    )

    predictor = ChainSelectionPredictor(
        retriever=retriever,
        llm_client=llm_client,
        optimizer_docs=_default_optimizer_docs(),
        objective_config=objective_config,
        config=PredictorConfig(
            feature_subset=args.feature_subset,
            prompt_mode=args.prompt_mode,
            enable_retry_on_failure=True,
            fallback_chain_id="chain_01",
        ),
    )

    representatives = circuit_representatives(eval_df)
    matching_rows = representatives[representatives["original_circuit"] == args.circuit_id]
    if matching_rows.empty:
        raise KeyError(
            f"Circuit {args.circuit_id} was not found in the {args.split} split."
        )
    query_row = matching_rows.iloc[0].to_dict()

    prediction = predictor.predict(query_row)

    scored_candidates = score_circuit_candidates(
        candidate_rows_for_circuit(eval_df, args.circuit_id),
        objective_config,
    )
    evaluation = evaluate_selection(
        circuit_id=args.circuit_id,
        scored_candidates=scored_candidates,
        selected_chain_id=prediction.final_selected_chain_id,
        global_best_average_chain_id=global_best_average_chain_id,
    )

    print(f"circuit_id: {prediction.circuit_id}")
    print(f"split: {args.split}")
    print(f"model_name: {args.model_name}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"feature_subset: {args.feature_subset}")
    print(f"selected_chain_id: {prediction.final_selected_chain_id}")
    print(f"selected_chain: {prediction.final_selected_chain}")
    print(f"used_fallback: {prediction.used_fallback}")
    print(f"attempt_count: {prediction.attempt_count}")
    print(f"failure_reason: {prediction.failure_reason}")
    print(f"neighbor_count: {len(prediction.retrieved_neighbors)}")

    if prediction.validated_response is not None:
        print(f"confidence: {prediction.validated_response.confidence}")
        print(
            "predicted_tradeoff_summary: "
            f"{prediction.validated_response.predicted_tradeoff_summary}"
        )
        print(f"reasoning_summary: {prediction.validated_response.reasoning_summary}")
        print(f"evidence_used: {prediction.validated_response.evidence_used}")

    print(f"objective_score: {evaluation.objective_score:.6f}")
    print(f"oracle_best_chain_id: {evaluation.oracle_best_chain_id}")
    print(f"oracle_best_chain_name: {evaluation.oracle_best_chain_name}")
    print(f"oracle_best_score: {evaluation.oracle_best_score:.6f}")
    print(f"regret: {evaluation.regret:.6f}")
    print(
        "two_qubit_gate_reduction_pct: "
        f"{evaluation.total_two_qubit_gate_reduction_pct:.6f}"
    )
    print(f"depth_reduction_pct: {evaluation.total_depth_reduction_pct:.6f}")
    print(f"runtime_seconds: {evaluation.total_runtime_seconds:.6f}")
    print(f"runtime_score: {evaluation.total_runtime_score:.6f}")
    print(f"win_vs_best_single_optimizer: {evaluation.win_vs_best_single_optimizer}")
    print(
        "win_vs_global_best_average_chain: "
        f"{evaluation.win_vs_global_best_average_chain}"
    )

    if args.show_raw_response:
        print("raw_response_text:")
        print(prediction.llm_call_results[-1].raw_response_text)


if __name__ == "__main__":
    main()
