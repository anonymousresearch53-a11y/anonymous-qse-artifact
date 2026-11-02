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

import pandas as pd

try:
    from machine_learning.LLM.LLM_v2.data import (
        dataset_metadata,
        candidate_rows_for_circuit,
        circuit_representatives,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from machine_learning.LLM.LLM_v2.evaluate import (
        CircuitEvaluation,
        evaluate_selection,
        score_circuit_candidates,
    )
    from machine_learning.LLM.LLM_v2.llm_client import LLMClientConfig, build_llm_client
    from machine_learning.LLM.LLM_v2.objective import ObjectiveConfig, objective_metadata
    from machine_learning.LLM.LLM_v2.predictor import (
        ChainSelectionPredictor,
        PredictorConfig,
    )
    from machine_learning.LLM.LLM_v2.prompting import PROMPT_VERSION
    from machine_learning.LLM.LLM_v2.retrieval import TrainingCircuitRetriever
    from machine_learning.LLM.LLM_v2.schemas import RESPONSE_SCHEMA_VERSION
except ImportError:
    from .data import (
        dataset_metadata,
        candidate_rows_for_circuit,
        circuit_representatives,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from .evaluate import CircuitEvaluation, evaluate_selection, score_circuit_candidates
    from .llm_client import LLMClientConfig, build_llm_client
    from .objective import ObjectiveConfig, objective_metadata
    from .predictor import ChainSelectionPredictor, PredictorConfig
    from .prompting import PROMPT_VERSION
    from .retrieval import TrainingCircuitRetriever
    from .schemas import RESPONSE_SCHEMA_VERSION


DEFAULT_ARTIFACT_ROOT = Path("machine_learning/LLM/LLM_v2/artifacts")


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
) -> tuple[dict[str, pd.DataFrame], str]:
    """Score every training circuit once and find the global best-average chain."""
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


def default_optimizer_docs() -> dict[str, str]:
    """Return temporary local optimizer descriptions."""
    return {
        "qiskit_standard": "Standard Qiskit transpilation baseline.",
        "qiskit_ai": "Model-guided Qiskit optimization pipeline.",
        "tket": "TKET-based optimization pipeline.",
        "wisq_rules": "Rule-based WISQ optimization variant.",
        "wisq_bqskit": "WISQ followed by BQSKit-oriented optimization.",
    }


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
        llm_call_results.append(
            {
                "provider": result.provider,
                "model_name": result.model_name,
                "raw_response_text": result.raw_response_text,
                "parsed_response": result.parsed_response,
                "success": result.success,
                "error_message": result.error_message,
                "latency_seconds": result.latency_seconds,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "response_metadata": result.response_metadata,
            }
        )

    retrieved_neighbors = []
    for neighbor in prediction.retrieved_neighbors:
        retrieved_neighbors.append(asdict(neighbor))

    return {
        "circuit_id": prediction.circuit_id,
        "query_features": prediction.query_features,
        "retrieved_neighbors": retrieved_neighbors,
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


def build_prediction_row(
    prediction,
    evaluation: CircuitEvaluation,
) -> dict[str, Any]:
    """Build one flat CSV row for later analysis."""
    return {
        "circuit_id": prediction.circuit_id,
        "selected_chain_id": prediction.final_selected_chain_id,
        "selected_chain": "__".join(prediction.final_selected_chain),
        "used_fallback": prediction.used_fallback,
        "attempt_count": prediction.attempt_count,
        "failure_reason": prediction.failure_reason,
        "confidence": None
        if prediction.validated_response is None
        else prediction.validated_response.confidence,
        "objective_score": evaluation.objective_score,
        "oracle_best_chain_id": evaluation.oracle_best_chain_id,
        "oracle_best_score": evaluation.oracle_best_score,
        "regret": evaluation.regret,
        "two_qubit_gate_reduction_pct": evaluation.total_two_qubit_gate_reduction_pct,
        "depth_reduction_pct": evaluation.total_depth_reduction_pct,
        "runtime_seconds": evaluation.total_runtime_seconds,
        "runtime_score": evaluation.total_runtime_score,
        "win_vs_best_single_optimizer": evaluation.win_vs_best_single_optimizer,
        "win_vs_global_best_average_chain": evaluation.win_vs_global_best_average_chain,
    }


def compute_summary(prediction_rows: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate experiment metrics."""
    if prediction_rows.empty:
        raise ValueError("No prediction rows were available for summary.")

    return {
        "num_circuits": int(len(prediction_rows)),
        "mean_objective_score": float(prediction_rows["objective_score"].mean()),
        "mean_regret": float(prediction_rows["regret"].mean()),
        "mean_two_qubit_gate_reduction_pct": float(
            prediction_rows["two_qubit_gate_reduction_pct"].mean()
        ),
        "mean_depth_reduction_pct": float(
            prediction_rows["depth_reduction_pct"].mean()
        ),
        "mean_runtime_seconds": float(prediction_rows["runtime_seconds"].mean()),
        "win_rate_vs_best_single_optimizer": float(
            prediction_rows["win_vs_best_single_optimizer"].mean()
        ),
        "win_rate_vs_global_best_average_chain": float(
            prediction_rows["win_vs_global_best_average_chain"].mean()
        ),
        "fallback_rate": float(prediction_rows["used_fallback"].mean()),
        "mean_attempt_count": float(prediction_rows["attempt_count"].mean()),
    }


def ensure_run_directories(run_dir: Path) -> tuple[Path, Path]:
    """Create the run directory and per-circuit artifact directory."""
    circuit_dir = run_dir / "circuits"
    run_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, circuit_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run an LLM-based chain-selection experiment over one split."
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.4-mini",
    )
    parser.add_argument(
        "--prompt-mode",
        default="full_evidence",
        choices=["full_evidence", "docs_only", "similar_circuits_only", "no_retrieval"],
    )
    parser.add_argument(
        "--feature-subset",
        default="full_structural",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick test runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit run output directory.",
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
    objective_config = ObjectiveConfig()

    score_cache, global_best_average_chain_id = build_training_score_cache(
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
        optimizer_docs=default_optimizer_docs(),
        objective_config=objective_config,
        config=PredictorConfig(
            feature_subset=args.feature_subset,
            prompt_mode=args.prompt_mode,
            enable_retry_on_failure=True,
            fallback_chain_id="chain_01",
        ),
    )

    representatives = circuit_representatives(eval_df)
    if args.limit is not None:
        representatives = representatives.head(args.limit).copy()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"{timestamp}_{args.split}_{args.model_name}_{args.prompt_mode}_{args.feature_subset}"
    )
    run_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ARTIFACT_ROOT / run_name
    run_dir, circuit_dir = ensure_run_directories(run_dir)

    prediction_rows: list[dict[str, Any]] = []

    for _, row in representatives.iterrows():
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

        prediction_rows.append(build_prediction_row(prediction, evaluation))

    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False)

    summary = compute_summary(prediction_frame)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "run_name": run_name,
        "created_at_utc": timestamp,
        "split": args.split,
        "model_name": args.model_name,
        "prompt_mode": args.prompt_mode,
        "feature_subset": args.feature_subset,
        "retrieval_k": args.k,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "dataset": dataset_metadata(bundle),
        "objective": objective_metadata(objective_config),
        "num_circuits": int(len(prediction_frame)),
        "global_best_average_chain_id": global_best_average_chain_id,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"run_dir: {run_dir}")
    print(f"num_circuits: {len(prediction_frame)}")
    print(f"mean_objective_score: {summary['mean_objective_score']:.6f}")
    print(f"mean_regret: {summary['mean_regret']:.6f}")


if __name__ == "__main__":
    main()
