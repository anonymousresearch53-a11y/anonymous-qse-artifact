"""CLI entrypoint for reproducible LLM chain-selection experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import best_single_optimizer, global_best_average_chain, random_valid_chain
from .data import (
    build_candidate_chains,
    candidate_rows_for_circuit,
    circuit_representatives,
    dataset_metadata,
    load_dataset_bundle,
    split_frame_for_name,
    to_pretty_json,
    write_resolved_split_ids,
)
from .evaluate import evaluate_selection, score_circuit_candidates
from .features import DEFAULT_FEATURE_SUBSET
from .llm_client import build_llm_client, provider_defaults
from .objective import ObjectiveConfig, objective_metadata
from .predictor import predict_chain_for_circuit
from .prompting import DEFAULT_PROMPT_MODE, PROMPT_VERSION, load_optimizer_docs
from .retrieval import DEFAULT_RETRIEVAL_K, TrainingCircuitRetriever
from .schemas import ChainSelection


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = provider_defaults()["default_model_name"]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts"
DEFAULT_DOC_DIR = Path(__file__).resolve().parent / "optimizer_docs"


def _objective_description(config: ObjectiveConfig) -> str:
    return (
        "Use the same v3 RL-style episode utility as the decision objective. "
        "For one-step chains, score = first-step local reward + final correction. "
        "For two-step chains, score = first-step local reward + second-step local reward "
        "+ final correction. "
        f"reward_type={config.reward_type}, "
        f"runtime_weight={config.runtime_weight}, "
        f"two_qubit_weight={config.two_qubit_weight}, "
        f"depth_weight={config.depth_weight}, "
        f"final_reward_weight={config.final_reward_weight}."
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_metrics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(evaluations)
    return {
        "circuits_evaluated": int(len(frame)),
        "mean_objective_score": float(frame["objective_score"].mean()),
        "mean_two_qubit_reduction_pct": float(
            frame["total_two_qubit_gate_reduction_pct"].mean()
        ),
        "mean_depth_reduction_pct": float(frame["total_depth_reduction_pct"].mean()),
        "mean_runtime_seconds": float(frame["total_runtime_seconds"].mean()),
        "mean_runtime_score": float(frame["total_runtime_score"].mean()),
        "mean_oracle_regret": float(frame["regret"].mean()),
        "win_rate_vs_best_single_optimizer": float(
            frame["win_vs_best_single_optimizer"].mean()
        ),
        "win_rate_vs_global_best_average_chain": float(
            frame["win_vs_global_best_average_chain"].mean()
        ),
        "fallback_rate": float(frame["used_fallback"].mean()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the experiment CLI."""
    parser = argparse.ArgumentParser(
        description="Run a reproducible LLM chain-selection experiment.",
    )
    parser.add_argument(
        "--selector",
        default="llm",
        choices=[
            "llm",
            "random_valid_chain",
            "best_single_optimizer",
            "global_best_average_chain",
        ],
        help="Decision mechanism to evaluate.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "mock"],
        help="LLM provider backend to use.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Evaluation split to run.",
    )
    parser.add_argument(
        "--prompt-mode",
        default=DEFAULT_PROMPT_MODE,
        choices=[
            "no_retrieval",
            "docs_only",
            "similar_circuits_only",
            "full_evidence",
        ],
        help="Prompt evidence ablation to run.",
    )
    parser.add_argument(
        "--feature-subset",
        default=DEFAULT_FEATURE_SUBSET,
        choices=[
            "basic_size",
            "size_plus_gate_composition",
            "full_structural",
        ],
        help="Prompt feature subset preset.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Number of nearest training circuits to retrieve.",
    )
    parser.add_argument(
        "--limit-circuits",
        type=int,
        default=None,
        help="Optional limit for smoke-test runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for experiment artifacts.",
    )
    parser.add_argument(
        "--runtime-weight",
        type=float,
        default=ObjectiveConfig().runtime_weight,
    )
    parser.add_argument(
        "--two-qubit-weight",
        type=float,
        default=ObjectiveConfig().two_qubit_weight,
    )
    parser.add_argument(
        "--depth-weight",
        type=float,
        default=ObjectiveConfig().depth_weight,
    )
    parser.add_argument(
        "--final-reward-weight",
        type=float,
        default=ObjectiveConfig().final_reward_weight,
    )
    return parser


def main() -> None:
    """Run one experiment and write local artifacts."""
    args = build_arg_parser().parse_args()
    objective_config = ObjectiveConfig(
        runtime_weight=args.runtime_weight,
        two_qubit_weight=args.two_qubit_weight,
        depth_weight=args.depth_weight,
        final_reward_weight=args.final_reward_weight,
    )

    bundle = load_dataset_bundle()
    train_frame = bundle.train_df.copy()
    eval_frame = split_frame_for_name(bundle, args.split)
    if args.limit_circuits is not None:
        limited_circuits = sorted(eval_frame["original_circuit"].unique())[: args.limit_circuits]
        eval_frame = eval_frame[eval_frame["original_circuit"].isin(limited_circuits)].copy()

    train_representatives = circuit_representatives(train_frame)
    eval_representatives = circuit_representatives(eval_frame)
    train_score_cache: dict[str, pd.DataFrame] = {}
    eval_score_cache: dict[str, pd.DataFrame] = {}

    def train_scores(circuit_id: str) -> pd.DataFrame:
        if circuit_id not in train_score_cache:
            train_score_cache[circuit_id] = score_circuit_candidates(
                candidate_rows_for_circuit(train_frame, circuit_id),
                objective_config,
            )
        return train_score_cache[circuit_id]

    def eval_scores(circuit_id: str) -> pd.DataFrame:
        if circuit_id not in eval_score_cache:
            eval_score_cache[circuit_id] = score_circuit_candidates(
                candidate_rows_for_circuit(eval_frame, circuit_id),
                objective_config,
            )
        return eval_score_cache[circuit_id]

    global_baseline = global_best_average_chain(
        pd.concat([train_scores(circuit_id) for circuit_id in train_representatives["original_circuit"]])
    )
    retriever = TrainingCircuitRetriever(
        train_frame=train_frame,
        feature_subset=args.feature_subset,
        score_lookup_fn=train_scores,
        k=args.retrieval_k,
    )
    optimizer_docs = load_optimizer_docs(DEFAULT_DOC_DIR)
    client = build_llm_client(args.provider, args.model_name) if args.selector == "llm" else None
    candidate_chain_lookup = {
        candidate["chain_id"]: candidate for candidate in build_candidate_chains()
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.expanduser().resolve() / run_id
    objective_description = _objective_description(objective_config)

    per_circuit_records: list[dict[str, Any]] = []
    prediction_jsonl_rows: list[dict[str, Any]] = []

    for _, representative in eval_representatives.iterrows():
        circuit_id = str(representative["original_circuit"])
        scored_candidates = eval_scores(circuit_id)
        fallback = best_single_optimizer(scored_candidates)
        retrieved_neighbors = []

        if args.selector == "llm":
            if args.prompt_mode in {"similar_circuits_only", "full_evidence"}:
                retrieved_neighbors = retriever.retrieve(representative.to_dict())

            prediction = predict_chain_for_circuit(
                query_row=representative.to_dict(),
                feature_subset=args.feature_subset,
                prompt_mode=args.prompt_mode,
                optimizer_docs=optimizer_docs,
                retrieved_neighbors=retrieved_neighbors,
                objective_description=objective_description,
                llm_client=client,
                candidate_chain_lookup=candidate_chain_lookup,
                fallback_selection=fallback,
            )
        elif args.selector == "random_valid_chain":
            baseline = random_valid_chain(scored_candidates, seed=bundle.split_seed)
            prediction = None
        elif args.selector == "best_single_optimizer":
            baseline = best_single_optimizer(scored_candidates)
            prediction = None
        elif args.selector == "global_best_average_chain":
            baseline = global_baseline
            prediction = None
        else:
            raise ValueError(f"Unsupported selector: {args.selector}")

        if prediction is None:
            selection = ChainSelection(
                selected_chain_id=baseline.selected_chain_id,
                selected_chain=list(baseline.selected_chain),
                predicted_tradeoff_summary=f"Baseline selection via {baseline.method_name}.",
                reasoning_summary=f"Baseline metadata: {baseline.metadata}",
                confidence=1.0,
                evidence_used=[baseline.method_name],
            )
            selected_chain_id = selection.selected_chain_id
            used_fallback = False
            fallback_method = None
            retry_count = 0
            latency_seconds = 0.0
            estimated_cost_usd = 0.0
            provider_name = "baseline"
            model_name = args.selector
            raw_model_response = ""
            provider_metadata = baseline.metadata
            prompt_text = ""
        else:
            selection = prediction.selection
            selected_chain_id = selection.selected_chain_id
            used_fallback = bool(prediction.used_fallback)
            fallback_method = prediction.fallback_method
            retry_count = int(prediction.retry_count)
            latency_seconds = float(prediction.raw_response.latency_seconds)
            estimated_cost_usd = prediction.raw_response.estimated_cost_usd
            provider_name = client.provider_name
            model_name = client.model_name
            raw_model_response = prediction.raw_response.raw_text
            provider_metadata = prediction.raw_response.response_metadata
            prompt_text = prediction.prompt_text

        evaluation = evaluate_selection(
            circuit_id=circuit_id,
            scored_candidates=scored_candidates,
            selected_chain_id=selected_chain_id,
            global_best_average_chain_id=global_baseline.selected_chain_id,
        )
        circuit_record = {
            **asdict(evaluation),
            "selector": args.selector,
            "provider_name": provider_name,
            "model_name": model_name,
            "prompt_mode": args.prompt_mode,
            "feature_subset": args.feature_subset,
            "retrieval_k": int(args.retrieval_k),
            "used_fallback": used_fallback,
            "fallback_method": fallback_method,
            "retry_count": retry_count,
            "latency_seconds": latency_seconds,
            "estimated_cost_usd": estimated_cost_usd,
        }
        per_circuit_records.append(circuit_record)
        prediction_jsonl_rows.append(
            {
                "circuit_id": circuit_id,
                "query_features_shown": {
                    key: representative[key]
                    for key in representative.index
                    if key in retriever.feature_columns
                },
                "query_features_initial_state": {
                    key: representative[key]
                    for key in representative.index
                    if key.startswith("initial_")
                },
                "retrieved_neighbor_ids": [neighbor.circuit_id for neighbor in retrieved_neighbors],
                "retrieved_neighbors": [
                    {
                        "circuit_id": neighbor.circuit_id,
                        "distance": neighbor.distance,
                        "best_chain_id": neighbor.best_chain_id,
                        "best_chain_name": neighbor.best_chain_name,
                        "best_chain_score": neighbor.best_chain_score,
                        "top_outcomes": neighbor.top_outcomes,
                    }
                    for neighbor in retrieved_neighbors
                ],
                "prompt_text": prompt_text,
                "raw_model_response": raw_model_response,
                "parsed_selection": selection.to_dict(),
                "provider_metadata": provider_metadata,
                "evaluation": asdict(evaluation),
            }
        )

    aggregate_metrics = _aggregate_metrics(per_circuit_records)
    run_manifest = {
        "run_id": run_id,
        "created_at_utc": run_id,
        "selector": args.selector,
        "provider_name": client.provider_name if client else "baseline",
        "model_name": client.model_name if client else args.selector,
        "prompt_version": PROMPT_VERSION,
        "prompt_mode": args.prompt_mode,
        "feature_subset": args.feature_subset,
        "retrieval_metadata": retriever.metadata(),
        "dataset_metadata": dataset_metadata(bundle),
        "resolved_split_ids": write_resolved_split_ids(bundle),
        "objective_metadata": objective_metadata(objective_config),
        "global_best_average_chain": {
            "chain_id": global_baseline.selected_chain_id,
            "chain_name": global_baseline.selected_chain_name,
            "chain": global_baseline.selected_chain,
            "metadata": global_baseline.metadata,
        },
        "aggregate_metrics": aggregate_metrics,
    }

    _write_json(run_dir / "run_manifest.json", run_manifest)
    _write_json(run_dir / "optimizer_docs.json", optimizer_docs)
    _write_csv(run_dir / "per_circuit_summary.csv", per_circuit_records)
    _write_jsonl(run_dir / "per_circuit_predictions.jsonl", prediction_jsonl_rows)

    print(f"Wrote experiment artifacts to {run_dir}")
    print(json.dumps(aggregate_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
