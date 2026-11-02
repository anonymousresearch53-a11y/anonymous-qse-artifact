"""Run one LLM prediction for one circuit and print its measured performance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from machine_learning.LLM.llm_v3.candidate_ranker import LearnedCandidateRanker
    from machine_learning.LLM.llm_v3.data import (
        ORIGINAL_CHAIN_ID,
        candidate_rows_for_circuit,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from machine_learning.LLM.llm_v3.experiment_presets import (
        PRESET_CLEAN_ALL_CANDIDATES,
        available_experiment_presets,
        resolve_experiment_preset,
    )
    from machine_learning.LLM.llm_v3.evaluate import (
        evaluate_selection,
        score_circuit_candidates,
    )
    from machine_learning.LLM.llm_v3.llm_client import (
        LLMClientConfig,
        build_llm_client,
    )
    from machine_learning.LLM.llm_v3.optimization_gate import OptimizationGate
    from machine_learning.LLM.llm_v3.objective import ObjectiveConfig
    from machine_learning.LLM.llm_v3.features import build_query_feature_frame
    from machine_learning.LLM.llm_v3.predictor import (
        ChainSelectionPredictor,
        PredictorConfig,
    )
    from machine_learning.LLM.llm_v3.profiles import (
        build_chain_profiles,
        chain_profile_lookup,
        default_optimizer_docs,
    )
    from machine_learning.LLM.llm_v3.retrieval import TrainingCircuitRetriever
    from machine_learning.LLM.llm_v3.run_experiment import build_training_score_cache
except ImportError:
    from .candidate_ranker import LearnedCandidateRanker
    from .data import (
        ORIGINAL_CHAIN_ID,
        candidate_rows_for_circuit,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from .experiment_presets import (
        PRESET_CLEAN_ALL_CANDIDATES,
        available_experiment_presets,
        resolve_experiment_preset,
    )
    from .evaluate import evaluate_selection, score_circuit_candidates
    from .features import build_query_feature_frame
    from .llm_client import LLMClientConfig, build_llm_client
    from .optimization_gate import OptimizationGate
    from .objective import ObjectiveConfig
    from .predictor import ChainSelectionPredictor, PredictorConfig
    from .profiles import (
        build_chain_profiles,
        chain_profile_lookup,
        default_optimizer_docs,
    )
    from .retrieval import TrainingCircuitRetriever
    from .run_experiment import build_training_score_cache


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
    parser.add_argument("--provider", default="openai", choices=["openai", "mock"])
    parser.add_argument("--model-name", default="gpt-5.4-mini", help="Provider model name.")
    parser.add_argument("--feature-subset", default="full_structural")
    parser.add_argument(
        "--strict-rl-v3-features-only",
        action="store_true",
        help=(
            "Restrict every predictive component to the RL v3 all_features-style numeric "
            "query features only in the learned components."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        default="full_evidence",
        choices=["full_evidence", "docs_only", "similar_circuits_only", "no_retrieval"],
    )
    parser.add_argument(
        "--clean-prompt-baseline",
        action="store_true",
        help=(
            "Show all candidates and strip prompt extras: no candidate profile stats, "
            "no query circuit id, no optimize-vs-original gate, no optimizer docs, "
            "and no retrieval evidence."
        ),
    )
    parser.add_argument(
        "--experiment-preset",
        default=None,
        choices=available_experiment_presets(),
        help=(
            "Named ablation preset that overrides prompt/routing extras with a reproducible "
            "combination."
        ),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retrieval-top-outcomes", type=int, default=4)
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--fallback-chain-id", default=ORIGINAL_CHAIN_ID)
    parser.add_argument("--reward-type", default=ObjectiveConfig().reward_type)
    parser.add_argument("--runtime-weight", type=float, default=ObjectiveConfig().runtime_weight)
    parser.add_argument(
        "--two-qubit-weight",
        type=float,
        default=ObjectiveConfig().two_qubit_weight,
    )
    parser.add_argument("--depth-weight", type=float, default=ObjectiveConfig().depth_weight)
    parser.add_argument(
        "--final-reward-weight",
        type=float,
        default=ObjectiveConfig().final_reward_weight,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Print the full raw JSON model response.",
    )
    return parser


def main() -> None:
    """Run one prediction and print the result plus measured performance."""
    args = _build_argument_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    _load_env_file(repo_root)

    bundle = load_dataset_bundle()
    train_df = split_frame_for_name(bundle, "train")
    eval_df = split_frame_for_name(bundle, args.split)
    strict_rl_v3_features_only = bool(args.strict_rl_v3_features_only)
    feature_subset = (
        "rl_v3_all_features" if strict_rl_v3_features_only else args.feature_subset
    )
    preset_name = args.experiment_preset
    if preset_name is None and bool(args.clean_prompt_baseline):
        preset_name = PRESET_CLEAN_ALL_CANDIDATES
    preset = None if preset_name is None else resolve_experiment_preset(preset_name)
    predictor_prompt_mode = args.prompt_mode if preset is None else preset.prompt_mode

    objective_config = ObjectiveConfig(
        reward_type=args.reward_type,
        runtime_weight=float(args.runtime_weight),
        two_qubit_weight=float(args.two_qubit_weight),
        depth_weight=float(args.depth_weight),
        final_reward_weight=float(args.final_reward_weight),
    )
    score_cache, train_scored, global_best_average_chain_id = build_training_score_cache(
        train_df,
        objective_config,
    )
    chain_profiles = build_chain_profiles(train_scored)
    profile_lookup = chain_profile_lookup(chain_profiles)
    candidate_ranker = None
    if preset is None or preset.include_ranker:
        candidate_ranker = LearnedCandidateRanker(
            feature_subset=feature_subset,
            candidate_profile_lookup=profile_lookup,
        )
        candidate_ranker.fit(train_frame=train_df, train_scored=train_scored)
    optimization_gate = None
    if preset is None or preset.include_gate:
        optimization_gate = OptimizationGate(
            feature_subset=feature_subset,
        )
        optimization_gate.fit(train_frame=train_df, train_scored=train_scored)

    def score_lookup_fn(circuit_id: str) -> pd.DataFrame:
        return score_cache[circuit_id]

    retriever = TrainingCircuitRetriever(
        train_frame=train_df,
        feature_subset=feature_subset,
        score_lookup_fn=score_lookup_fn,
        k=args.k,
        top_outcomes=args.retrieval_top_outcomes,
    )

    llm_client = build_llm_client(
        LLMClientConfig(
            provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            cache_dir=args.cache_dir,
            use_cache=not args.disable_cache,
        )
    )

    predictor = ChainSelectionPredictor(
        retriever=retriever,
        llm_client=llm_client,
        optimizer_docs=(
            default_optimizer_docs(chain_profiles)
            if preset is None or preset.include_optimizer_docs
            else {}
        ),
        objective_config=objective_config,
        candidate_profile_lookup=(
            profile_lookup
            if preset is None or preset.include_candidate_profiles
            else {}
        ),
        candidate_ranker=candidate_ranker,
        optimization_gate=optimization_gate,
        global_best_average_chain_id=global_best_average_chain_id,
        config=PredictorConfig(
            feature_subset=feature_subset,
            prompt_mode=predictor_prompt_mode,
            enable_retry_on_failure=True,
            fallback_chain_id=args.fallback_chain_id,
            min_confidence=float(args.min_confidence),
            shortlist_size=(
                preset.shortlist_size
                if preset is not None
                else int(args.shortlist_size) if args.shortlist_size else None
            ),
            include_query_circuit_id=(
                preset.include_query_circuit_id
                if preset is not None
                else not strict_rl_v3_features_only
            ),
            include_neighbor_identity=(
                preset.include_neighbor_identity
                if preset is not None
                else not strict_rl_v3_features_only
            ),
        ),
    )

    representatives = build_query_feature_frame(
        eval_df,
        feature_subset=feature_subset,
    )
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
    print(f"provider: {args.provider}")
    print(f"model_name: {args.model_name}")
    print(f"prompt_mode: {predictor_prompt_mode}")
    print(f"feature_subset: {feature_subset}")
    print(f"strict_rl_v3_features_only: {strict_rl_v3_features_only}")
    print(f"experiment_preset: {preset_name}")
    print(f"selected_chain_id: {prediction.final_selected_chain_id}")
    print(f"selected_chain: {prediction.final_selected_chain}")
    print(f"shortlist_ids: {[item['chain_id'] for item in prediction.shortlisted_candidates]}")
    print(f"candidate_rank_scores: {prediction.candidate_rank_scores}")
    print(f"gate_decision: {prediction.gate_decision}")
    print(f"used_fallback: {prediction.used_fallback}")
    print(f"attempt_count: {prediction.attempt_count}")
    print(f"failure_reason: {prediction.failure_reason}")
    print(f"neighbor_count: {len(prediction.retrieved_neighbors)}")
    neighbor_summaries = []
    for neighbor in prediction.retrieved_neighbors:
        neighbor_summaries.append(
            (
                neighbor.circuit_id,
                round(neighbor.distance, 4),
                neighbor.best_chain_id,
                round(neighbor.best_vs_original_margin, 3),
            )
        )
    print(
        "neighbor_summaries: "
        f"{neighbor_summaries}"
    )

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
    shortlist_ids = [item["chain_id"] for item in prediction.shortlisted_candidates]
    print(f"oracle_in_shortlist: {evaluation.oracle_best_chain_id in shortlist_ids}")
    if evaluation.oracle_best_chain_id in shortlist_ids:
        print(
            "oracle_shortlist_rank: "
            f"{shortlist_ids.index(evaluation.oracle_best_chain_id) + 1}"
        )
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

    if prediction.llm_call_results:
        final_result = prediction.llm_call_results[-1]
        print(f"from_cache: {final_result.from_cache}")
        print(f"llm_latency_seconds: {final_result.latency_seconds:.4f}")

    if args.show_raw_response and prediction.llm_call_results:
        print("\nraw_response_text:")
        print(prediction.llm_call_results[-1].raw_response_text)


if __name__ == "__main__":
    main()
