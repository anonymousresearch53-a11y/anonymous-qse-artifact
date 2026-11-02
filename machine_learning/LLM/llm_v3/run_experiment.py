"""Run a reproducible LLM chain-selection experiment over one dataset split."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import argparse
import json
import os
import re
import sys
import time

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from machine_learning.LLM.llm_v3.candidate_ranker import LearnedCandidateRanker
    from machine_learning.LLM.llm_v3.data import (
        ORIGINAL_CHAIN_ID,
        candidate_rows_for_circuit,
        dataset_metadata,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from machine_learning.LLM.llm_v3.evaluate import (
        CircuitEvaluation,
        evaluate_selection,
        score_circuit_candidates,
    )
    from machine_learning.LLM.llm_v3.experiment_presets import (
        PRESET_CLEAN_ALL_CANDIDATES,
        available_experiment_presets,
        resolve_experiment_preset,
    )
    from machine_learning.LLM.llm_v3.llm_client import (
        LLMClientConfig,
        build_llm_client,
    )
    from machine_learning.LLM.llm_v3.optimization_gate import OptimizationGate
    from machine_learning.LLM.llm_v3.objective import (
        ObjectiveConfig,
        objective_metadata,
    )
    from machine_learning.LLM.llm_v3.features import build_query_feature_frame
    from machine_learning.LLM.llm_v3.predictor import (
        ChainSelectionPredictor,
        PredictorConfig,
    )
    from machine_learning.LLM.llm_v3.profiles import (
        build_chain_profiles,
        chain_profile_lookup,
        default_optimizer_docs,
        top_chain_ids_by_average,
    )
    from machine_learning.LLM.llm_v3.prompting import PROMPT_VERSION
    from machine_learning.LLM.llm_v3.retrieval import TrainingCircuitRetriever
    from machine_learning.LLM.llm_v3.schemas import RESPONSE_SCHEMA_VERSION
except ImportError:
    from .candidate_ranker import LearnedCandidateRanker
    from .data import (
        ORIGINAL_CHAIN_ID,
        candidate_rows_for_circuit,
        dataset_metadata,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from .evaluate import CircuitEvaluation, evaluate_selection, score_circuit_candidates
    from .experiment_presets import (
        PRESET_CLEAN_ALL_CANDIDATES,
        available_experiment_presets,
        resolve_experiment_preset,
    )
    from .features import build_query_feature_frame
    from .llm_client import LLMClientConfig, build_llm_client
    from .optimization_gate import OptimizationGate
    from .objective import ObjectiveConfig, objective_metadata
    from .predictor import ChainSelectionPredictor, PredictorConfig
    from .profiles import (
        build_chain_profiles,
        chain_profile_lookup,
        default_optimizer_docs,
        top_chain_ids_by_average,
    )
    from .prompting import PROMPT_VERSION
    from .retrieval import TrainingCircuitRetriever
    from .schemas import RESPONSE_SCHEMA_VERSION


DEFAULT_ARTIFACT_ROOT = Path("machine_learning/LLM/llm_v3/artifacts")
DEFAULT_CACHE_ROOT = Path("machine_learning/LLM/llm_v3/cache")


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


def _safe_filename(value: str) -> str:
    """Convert a circuit id into a safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unknown_circuit"


def build_training_score_cache(
    train_df: pd.DataFrame,
    objective_config: ObjectiveConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Score every training circuit once and find the global best-average chain."""
    score_cache: dict[str, pd.DataFrame] = {}
    scored_frames: list[pd.DataFrame] = []

    for circuit_id in sorted(train_df["original_circuit"].unique().tolist()):
        circuit_rows = candidate_rows_for_circuit(train_df, circuit_id)
        scored_rows = score_circuit_candidates(circuit_rows, objective_config)
        scored_rows = scored_rows.copy()
        scored_rows["original_circuit"] = circuit_id
        score_cache[circuit_id] = scored_rows
        scored_frames.append(scored_rows)

    train_scored = pd.concat(scored_frames, ignore_index=True)
    chain_means = (
        train_scored.groupby(["chain_id", "chain_name"], as_index=False)["objective_score"]
        .mean()
        .sort_values(["objective_score", "chain_id"], ascending=[False, True])
    )
    global_best_average_chain_id = str(chain_means.iloc[0]["chain_id"])
    return score_cache, train_scored, global_best_average_chain_id


def serialize_prediction_and_evaluation(
    prediction,
    evaluation: CircuitEvaluation,
) -> dict[str, Any]:
    """Build a JSON-serializable per-circuit artifact payload."""
    validated_response = None
    if prediction.validated_response is not None:
        validated_response = asdict(prediction.validated_response)

    llm_call_results = []
    for result in prediction.llm_call_results:
        llm_call_results.append(asdict(result))

    return {
        "circuit_id": prediction.circuit_id,
        "query_features": prediction.query_features,
        "retrieved_neighbors": [asdict(neighbor) for neighbor in prediction.retrieved_neighbors],
        "shortlisted_candidates": prediction.shortlisted_candidates,
        "candidate_rank_scores": prediction.candidate_rank_scores,
        "gate_decision": prediction.gate_decision,
        "built_prompt": {
            "system_prompt": prediction.built_prompt.system_prompt,
            "user_prompt": prediction.built_prompt.user_prompt,
            "prompt_version": prediction.built_prompt.prompt_version,
            "prompt_mode": prediction.built_prompt.prompt_mode,
            "prompt_sections": prediction.built_prompt.prompt_sections,
        },
        "llm_call_results": llm_call_results,
        "validated_response": validated_response,
        "final_selected_chain_id": prediction.final_selected_chain_id,
        "final_selected_chain": prediction.final_selected_chain,
        "used_fallback": prediction.used_fallback,
        "failure_reason": prediction.failure_reason,
        "attempt_count": prediction.attempt_count,
        "evaluation": asdict(evaluation),
    }


def _aggregate_call_metrics(llm_call_results: list[Any]) -> dict[str, Any]:
    """Aggregate call-level metadata into flat per-circuit metrics."""
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    cache_hits = 0
    total_latency_seconds = 0.0

    for result in llm_call_results:
        usage = result.usage or {}
        total_input_tokens += int(usage.get("input_tokens") or 0)
        total_output_tokens += int(usage.get("output_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        cache_hits += int(bool(result.from_cache))
        total_latency_seconds += float(result.latency_seconds)

    return {
        "api_call_count": int(len(llm_call_results)),
        "cache_hit_count": int(cache_hits),
        "cache_hit_rate": float(cache_hits / len(llm_call_results)) if llm_call_results else 0.0,
        "total_latency_seconds": float(total_latency_seconds),
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_tokens": int(total_tokens),
    }


def build_prediction_row(
    prediction,
    evaluation: CircuitEvaluation,
    scored_candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Build one flat CSV row for later analysis."""
    call_metrics = _aggregate_call_metrics(prediction.llm_call_results)
    shortlist_ids = [candidate["chain_id"] for candidate in prediction.shortlisted_candidates]
    oracle_chain_id = str(evaluation.oracle_best_chain_id)
    oracle_in_shortlist = oracle_chain_id in shortlist_ids
    oracle_shortlist_rank = (
        shortlist_ids.index(oracle_chain_id) + 1 if oracle_in_shortlist else None
    )
    selected_shortlist_rank = (
        shortlist_ids.index(prediction.final_selected_chain_id) + 1
        if prediction.final_selected_chain_id in shortlist_ids
        else None
    )
    oracle_rank_scores = prediction.candidate_rank_scores or {}
    gate_decision = prediction.gate_decision or {}
    neighbor_distances = [float(neighbor.distance) for neighbor in prediction.retrieved_neighbors]
    neighbor_best_ids = {str(neighbor.best_chain_id) for neighbor in prediction.retrieved_neighbors}
    selected_is_original = int(prediction.final_selected_chain_id == ORIGINAL_CHAIN_ID)
    oracle_row = scored_candidates.iloc[0]
    return {
        "circuit_id": prediction.circuit_id,
        "selected_chain_id": prediction.final_selected_chain_id,
        "selected_chain": "__".join(prediction.final_selected_chain)
        if prediction.final_selected_chain
        else "__original__",
        "selected_is_original": selected_is_original,
        "used_fallback": prediction.used_fallback,
        "attempt_count": prediction.attempt_count,
        "failure_reason": prediction.failure_reason,
        "confidence": None
        if prediction.validated_response is None
        else prediction.validated_response.confidence,
        "shortlist_size": len(prediction.shortlisted_candidates),
        "shortlisted_chain_ids": "|".join(shortlist_ids),
        "selected_shortlist_rank": selected_shortlist_rank,
        "oracle_in_shortlist": oracle_in_shortlist,
        "oracle_shortlist_rank": oracle_shortlist_rank,
        "oracle_ranker_score": oracle_rank_scores.get(oracle_chain_id),
        "selected_ranker_score": oracle_rank_scores.get(prediction.final_selected_chain_id),
        "gate_recommendation": gate_decision.get("recommendation_label"),
        "gate_probability_original": gate_decision.get("probability_original"),
        "gate_probability_optimize": gate_decision.get("probability_optimize"),
        "selected_matches_any_neighbor_best": prediction.final_selected_chain_id in neighbor_best_ids,
        "nearest_neighbor_distance": min(neighbor_distances) if neighbor_distances else None,
        "mean_neighbor_distance": (
            sum(neighbor_distances) / len(neighbor_distances) if neighbor_distances else None
        ),
        "objective_score": evaluation.objective_score,
        "oracle_best_chain_id": evaluation.oracle_best_chain_id,
        "oracle_best_score": evaluation.oracle_best_score,
        "oracle_best_chain_name": evaluation.oracle_best_chain_name,
        "oracle_is_original": int(oracle_chain_id == ORIGINAL_CHAIN_ID),
        "oracle_total_two_qubit_gate_reduction_pct": float(
            oracle_row["total_two_qubit_gate_reduction_pct"]
        ),
        "oracle_total_depth_reduction_pct": float(oracle_row["total_depth_reduction_pct"]),
        "oracle_total_runtime_seconds": float(oracle_row["total_runtime_seconds"]),
        "regret": evaluation.regret,
        "two_qubit_gate_reduction_pct": evaluation.total_two_qubit_gate_reduction_pct,
        "depth_reduction_pct": evaluation.total_depth_reduction_pct,
        "runtime_seconds": evaluation.total_runtime_seconds,
        "runtime_score": evaluation.total_runtime_score,
        "win_vs_best_single_optimizer": evaluation.win_vs_best_single_optimizer,
        "win_vs_global_best_average_chain": evaluation.win_vs_global_best_average_chain,
        **call_metrics,
    }


def compute_summary(prediction_rows: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate experiment metrics."""
    if prediction_rows.empty:
        raise ValueError("No prediction rows were available for summary.")

    confidence = pd.to_numeric(prediction_rows["confidence"], errors="coerce")
    regret = pd.to_numeric(prediction_rows["regret"], errors="coerce")
    return {
        "num_circuits": int(len(prediction_rows)),
        "mean_objective_score": float(prediction_rows["objective_score"].mean()),
        "mean_regret": float(regret.mean()),
        "p90_regret": float(regret.quantile(0.90)),
        "p95_regret": float(regret.quantile(0.95)),
        "max_regret": float(regret.max()),
        "mean_two_qubit_gate_reduction_pct": float(
            prediction_rows["two_qubit_gate_reduction_pct"].mean()
        ),
        "mean_depth_reduction_pct": float(
            prediction_rows["depth_reduction_pct"].mean()
        ),
        "mean_runtime_seconds": float(prediction_rows["runtime_seconds"].mean()),
        "mean_runtime_score": float(prediction_rows["runtime_score"].mean()),
        "win_rate_vs_best_single_optimizer": float(
            prediction_rows["win_vs_best_single_optimizer"].mean()
        ),
        "win_rate_vs_global_best_average_chain": float(
            prediction_rows["win_vs_global_best_average_chain"].mean()
        ),
        "fallback_rate": float(prediction_rows["used_fallback"].mean()),
        "mean_attempt_count": float(prediction_rows["attempt_count"].mean()),
        "mean_confidence": float(confidence.mean()) if confidence.notna().any() else None,
        "mean_shortlist_size": float(prediction_rows["shortlist_size"].mean()),
        "original_selection_rate": float(
            pd.to_numeric(prediction_rows["selected_is_original"], errors="coerce").mean()
        ),
        "shortlist_oracle_recall": float(
            pd.to_numeric(prediction_rows["oracle_in_shortlist"], errors="coerce").mean()
        ),
        "mean_oracle_shortlist_rank": float(
            pd.to_numeric(prediction_rows["oracle_shortlist_rank"], errors="coerce").mean()
        ),
        "neighbor_best_match_rate": float(
            pd.to_numeric(
                prediction_rows["selected_matches_any_neighbor_best"],
                errors="coerce",
            ).mean()
        ),
        "gate_original_rate": float(
            pd.to_numeric(
                prediction_rows["gate_probability_original"],
                errors="coerce",
            ).mean()
        ),
        "gate_optimize_rate": float(
            pd.to_numeric(
                prediction_rows["gate_probability_optimize"],
                errors="coerce",
            ).mean()
        ),
        "mean_latency_seconds": float(
            pd.to_numeric(prediction_rows["total_latency_seconds"], errors="coerce").mean()
        ),
        "total_api_calls": int(
            pd.to_numeric(prediction_rows["api_call_count"], errors="coerce").sum()
        ),
        "cache_hit_rate": float(
            pd.to_numeric(prediction_rows["cache_hit_count"], errors="coerce").sum()
            / max(
                1,
                pd.to_numeric(prediction_rows["api_call_count"], errors="coerce").sum(),
            )
        ),
        "total_input_tokens": int(
            pd.to_numeric(prediction_rows["total_input_tokens"], errors="coerce").sum()
        ),
        "total_output_tokens": int(
            pd.to_numeric(prediction_rows["total_output_tokens"], errors="coerce").sum()
        ),
        "total_tokens": int(
            pd.to_numeric(prediction_rows["total_tokens"], errors="coerce").sum()
        ),
    }


def ensure_run_directories(run_dir: Path) -> tuple[Path, Path]:
    """Create the run directory and per-circuit artifact directory."""
    circuit_dir = run_dir / "circuits"
    run_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, circuit_dir


def _format_duration(total_seconds: float) -> str:
    """Format elapsed seconds as a compact clock string."""
    total_seconds = max(0.0, float(total_seconds))
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run an LLM-based chain-selection experiment over one split."
    )
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--provider", default="openai", choices=["openai", "mock"])
    parser.add_argument("--model-name", default="gpt-5.4-mini")
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
    parser.add_argument("--feature-subset", default="full_structural")
    parser.add_argument(
        "--strict-rl-v3-features-only",
        action="store_true",
        help=(
            "Restrict every predictive component to the RL v3 all_features-style numeric "
            "query features only in the learned components."
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
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick test runs.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit run output directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional explicit cache directory.")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a progress update after every N evaluated circuits.",
    )
    return parser


def main() -> None:
    """Run the full experiment and write local artifacts."""
    args = build_argument_parser().parse_args()
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

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if preset is not None:
        run_name = (
            f"{timestamp}_{args.split}_{args.provider}_{args.model_name}_"
            f"{preset.name}"
        )
    else:
        run_name = (
            f"{timestamp}_{args.split}_{args.provider}_{args.model_name}_"
            f"{args.prompt_mode}_{feature_subset}"
        )
    run_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ARTIFACT_ROOT / run_name
    run_dir, circuit_dir = ensure_run_directories(run_dir)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else DEFAULT_CACHE_ROOT / f"{args.provider}_{args.model_name}"
    )
    partial_manifest = {
        "run_name": run_name,
        "created_at_utc": timestamp,
        "status": "running",
        "split": args.split,
        "provider": args.provider,
        "model_name": args.model_name,
        "prompt_mode": predictor_prompt_mode,
        "feature_subset": feature_subset,
        "strict_rl_v3_features_only": strict_rl_v3_features_only,
        "experiment_preset": None if preset is None else preset.name,
        "clean_prompt_baseline": preset_name == PRESET_CLEAN_ALL_CANDIDATES,
        "dataset": dataset_metadata(bundle),
        "objective": objective_metadata(objective_config),
        "global_best_average_chain_id": global_best_average_chain_id,
    }
    (run_dir / "manifest.json").write_text(json.dumps(partial_manifest, indent=2))

    llm_client = build_llm_client(
        LLMClientConfig(
            provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            cache_dir=str(cache_dir),
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
    if args.limit is not None:
        representatives = representatives.head(args.limit).copy()
    total_circuits = int(len(representatives))
    progress_every = max(1, int(args.progress_every))

    prediction_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    fallback_count = 0
    cache_hit_count = 0

    print(
        "Starting llm_v3 evaluation | "
        f"split={args.split} | provider={args.provider} | model={args.model_name} | "
        f"circuits={total_circuits} | feature_subset={feature_subset} | "
        f"strict_rl_v3_features_only={strict_rl_v3_features_only} | run_dir={run_dir}",
        flush=True,
    )

    for index, (_, row) in enumerate(representatives.iterrows(), start=1):
        circuit_started = time.perf_counter()
        query_row = row.to_dict()
        circuit_id = str(query_row["original_circuit"])

        prediction = predictor.predict(query_row)
        scored_candidates = score_circuit_candidates(
            candidate_rows_for_circuit(eval_df, circuit_id),
            objective_config,
        )
        evaluation = evaluate_selection(
            circuit_id=circuit_id,
            scored_candidates=scored_candidates,
            selected_chain_id=prediction.final_selected_chain_id,
            global_best_average_chain_id=global_best_average_chain_id,
        )

        artifact_payload = serialize_prediction_and_evaluation(prediction, evaluation)
        artifact_path = circuit_dir / f"{_safe_filename(circuit_id)}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2))
        prediction_row = build_prediction_row(
            prediction,
            evaluation,
            scored_candidates=scored_candidates,
        )
        prediction_rows.append(prediction_row)

        fallback_count += int(bool(prediction.used_fallback))
        cache_hit_count += int(prediction_row["cache_hit_count"])

        if index % progress_every == 0 or index == total_circuits:
            elapsed = time.perf_counter() - run_started
            circuit_elapsed = time.perf_counter() - circuit_started
            pct_complete = 100.0 * index / max(1, total_circuits)
            running_objective = sum(
                float(row["objective_score"]) for row in prediction_rows
            ) / len(prediction_rows)
            running_regret = sum(
                float(row["regret"]) for row in prediction_rows
            ) / len(prediction_rows)
            selected_chain = prediction_row["selected_chain"]
            if not selected_chain:
                selected_chain = "__original__"
            print(
                f"[{index}/{total_circuits} | {pct_complete:5.1f}%] "
                f"circuit={circuit_id} | selected={selected_chain} | "
                f"objective={float(prediction_row['objective_score']):.4f} | "
                f"regret={float(prediction_row['regret']):.4f} | "
                f"fallback={bool(prediction.used_fallback)} | "
                f"oracle_in_shortlist={bool(prediction_row['oracle_in_shortlist'])} | "
                f"cache_hits={int(prediction_row['cache_hit_count'])}/{int(prediction_row['api_call_count'])} | "
                f"circuit_time={_format_duration(circuit_elapsed)} | "
                f"elapsed={_format_duration(elapsed)} | "
                f"running_mean_objective={running_objective:.4f} | "
                f"running_mean_regret={running_regret:.4f}",
                flush=True,
            )

    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
    chain_profiles.to_csv(run_dir / "train_chain_profiles.csv", index=False)

    summary = compute_summary(prediction_frame)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "run_name": run_name,
        "created_at_utc": timestamp,
        "split": args.split,
        "provider": args.provider,
        "model_name": args.model_name,
        "prompt_mode": predictor_prompt_mode,
        "feature_subset": feature_subset,
        "retrieval_k": args.k,
        "retrieval_top_outcomes": args.retrieval_top_outcomes,
        "shortlist_size": args.shortlist_size,
        "min_confidence": args.min_confidence,
        "fallback_chain_id": args.fallback_chain_id,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "cache_dir": str(cache_dir),
        "cache_enabled": not args.disable_cache,
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "dataset": dataset_metadata(bundle),
        "objective": objective_metadata(objective_config),
        "experiment_preset": None if preset is None else preset.name,
        "clean_prompt_baseline": preset_name == PRESET_CLEAN_ALL_CANDIDATES,
        "retriever": retriever.metadata(),
        "shortlist_ranker": None if candidate_ranker is None else candidate_ranker.metadata(),
        "optimization_gate": None if optimization_gate is None else optimization_gate.metadata(),
        "status": "completed",
        "num_circuits": int(len(prediction_frame)),
        "global_best_average_chain_id": global_best_average_chain_id,
        "top_train_average_chain_ids": top_chain_ids_by_average(chain_profiles, limit=10),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"run_dir: {run_dir}")
    print(f"num_circuits: {len(prediction_frame)}")
    print(f"mean_objective_score: {summary['mean_objective_score']:.6f}")
    print(f"mean_regret: {summary['mean_regret']:.6f}")
    print(f"p95_regret: {summary['p95_regret']:.6f}")
    print(f"fallback_rate: {summary['fallback_rate']:.6f}")
    print(f"original_selection_rate: {summary['original_selection_rate']:.6f}")
    print(f"shortlist_oracle_recall: {summary['shortlist_oracle_recall']:.6f}")
    print(f"total_cache_hits: {cache_hit_count}")
    print(f"elapsed: {_format_duration(time.perf_counter() - run_started)}")


if __name__ == "__main__":
    main()
