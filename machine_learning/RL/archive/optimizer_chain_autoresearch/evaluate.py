"""Locked multi-objective evaluator for optimizer-chain policies.

READ-ONLY DURING AUTONOMOUS RESEARCH.

The evaluator decodes policy actions independently and looks up their raw
outcomes in the master CSV.  It does not use the editable training reward or
trust an editable environment's ``selected_chain`` metadata.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .data_setup import (
        OPTIMIZER_NAMES,
        QUALITY_METRICS,
        DataBundle,
        ensure_data_setup,
    )
    from .optimizer_chain_env import OptimizerChainEnv
    from .train import load_model
except ImportError:  # pragma: no cover - supports ``python evaluate.py``
    from data_setup import (
        OPTIMIZER_NAMES,
        QUALITY_METRICS,
        DataBundle,
        ensure_data_setup,
    )
    from optimizer_chain_env import OptimizerChainEnv
    from train import load_model


QUALITY_METRIC_ORDER = tuple(QUALITY_METRICS)
RUNTIME_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.35)
LOCKED_FAILURE_PENALTY = 1.0
WISQ_BASELINE_CHAIN = "tket__wisq_bqskit"


@dataclass
class EvaluationResult:
    """A serializable summary plus the per-circuit selection table."""

    summary: dict[str, Any]
    selected: pd.DataFrame


def all_metric_subsets() -> list[tuple[str, ...]]:
    """Return all 15 non-empty subsets in deterministic order."""

    return [
        combination
        for size in range(1, len(QUALITY_METRIC_ORDER) + 1)
        for combination in itertools.combinations(QUALITY_METRIC_ORDER, size)
    ]


def objective_track_id(metrics: tuple[str, ...], runtime_weight: float) -> str:
    """Create a stable identifier for one locked objective track."""

    metric_part = "+".join(metrics)
    return f"quality:{metric_part}|runtime:{runtime_weight:.2f}"


def objective_tracks() -> dict[str, dict[str, Any]]:
    """Return all 75 fixed quality/runtime objective declarations."""

    return {
        objective_track_id(metrics, runtime_weight): {
            "metrics": list(metrics),
            "runtime_weight": runtime_weight,
        }
        for metrics in all_metric_subsets()
        for runtime_weight in RUNTIME_WEIGHTS
    }


def normalized_runtime(runtime_seconds: pd.Series, runtime_scale: float) -> pd.Series:
    """Apply the locked monotonic log-runtime normalization."""

    denominator = math.log1p(max(float(runtime_scale), 1e-9))
    return np.log1p(runtime_seconds.clip(lower=0.0)) / denominator


def _synthetic_noop_row(
    initial_row: pd.Series,
    *,
    invalid_action: bool,
) -> dict[str, Any]:
    """Create an unchanged outcome for no-op or invalid policy decisions."""

    record = initial_row.to_dict()
    for _, (initial_column, final_column) in QUALITY_METRICS.items():
        record[final_column] = record[initial_column]
    record["final_num_qubits"] = record["initial_num_qubits"]
    record["total_runtime_seconds"] = 0.0
    record["error"] = bool(invalid_action)
    record["selected_chain"] = None
    record["invalid_action"] = bool(invalid_action)
    record["failed"] = bool(invalid_action)
    return record


def selected_rows_to_outcomes(
    selected_rows: pd.DataFrame,
    statistics: dict[str, Any],
) -> pd.DataFrame:
    """Convert selected raw rows into circuit-normalized evaluation outcomes."""

    outcomes = selected_rows.copy()
    outcomes["failed"] = outcomes["failed"].astype(bool)
    outcomes["invalid_action"] = outcomes["invalid_action"].astype(bool)

    for metric, (initial_column, final_column) in QUALITY_METRICS.items():
        denominator = outcomes[initial_column].replace(0, np.nan)
        raw = (
            (outcomes[initial_column] - outcomes[final_column]) / denominator
        ).fillna(0.0)
        # Failed executions receive no quality credit even if a malformed source
        # row happens to contain changed final metrics.
        raw = raw.mask(outcomes["failed"], 0.0)
        outcomes[f"raw_{metric}"] = raw.astype(float)
        bounds = statistics["winsor_bounds"][metric]
        outcomes[f"score_{metric}"] = raw.clip(
            lower=float(bounds["lower"]), upper=float(bounds["upper"])
        )

    outcomes["runtime_normalized"] = normalized_runtime(
        outcomes["total_runtime_seconds"].astype(float),
        statistics["runtime_99_seconds"],
    )
    return outcomes


def _row_contribution(
    outcomes: pd.DataFrame,
    track: dict[str, Any],
) -> pd.Series:
    """Return each circuit's scalar contribution to one objective track."""

    quality_columns = [f"score_{metric}" for metric in track["metrics"]]
    quality = outcomes[quality_columns].mean(axis=1)
    return (
        quality
        - float(track["runtime_weight"]) * outcomes["runtime_normalized"]
        - LOCKED_FAILURE_PENALTY * outcomes["failed"].astype(float)
    )


def score_objectives(outcomes: pd.DataFrame) -> dict[str, float]:
    """Score one outcome per circuit on all 75 objective tracks."""

    return {
        track_id: float(_row_contribution(outcomes, track).mean())
        for track_id, track in objective_tracks().items()
    }


def summarize_metrics(outcomes: pd.DataFrame) -> dict[str, Any]:
    """Report winsorized and uncapped macro statistics for every metric."""

    metrics: dict[str, Any] = {}
    for metric in QUALITY_METRIC_ORDER:
        raw = outcomes[f"raw_{metric}"].astype(float)
        capped = outcomes[f"score_{metric}"].astype(float)
        metrics[metric] = {
            "winsorized_mean": float(capped.mean()),
            "raw_mean": float(raw.mean()),
            "median": float(raw.median()),
            "q25": float(raw.quantile(0.25)),
            "q75": float(raw.quantile(0.75)),
            "improvement_rate": float((raw > 0).mean()),
            "degradation_rate": float((raw < 0).mean()),
        }
    return {
        "quality_metrics": metrics,
        "runtime_normalized_mean": float(outcomes["runtime_normalized"].mean()),
        "runtime_seconds_mean": float(outcomes["total_runtime_seconds"].mean()),
        "failure_rate": float(outcomes["failed"].mean()),
        "invalid_action_rate": float(outcomes["invalid_action"].mean()),
        "circuits": int(len(outcomes)),
    }


def pareto_vector(outcomes: pd.DataFrame) -> dict[str, float]:
    """Return the six-dimensional vector used for Pareto retention."""

    vector = {
        metric: float(outcomes[f"score_{metric}"].mean())
        for metric in QUALITY_METRIC_ORDER
    }
    vector["runtime_normalized"] = float(outcomes["runtime_normalized"].mean())
    vector["failure_rate"] = float(outcomes["failed"].mean())
    return vector


def _lookup_chain_row(
    frame: pd.DataFrame, circuit: str, chain_name: str
) -> pd.Series:
    """Return the unique raw outcome row for a circuit and chain."""

    matches = frame[
        (frame["original_circuit"] == circuit)
        & (frame["chain_name"] == chain_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for circuit={circuit!r}, chain={chain_name!r}; "
            f"found {len(matches)}."
        )
    return matches.iloc[0].copy()


def decode_policy_selections(
    *,
    model: Any,
    frame: pd.DataFrame,
    bundle: DataBundle,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run a deterministic policy and independently reconstruct chosen chains."""

    env_config = dict(config["environment"])
    env_config["shuffle_each_epoch"] = False
    env = OptimizerChainEnv(
        frame,
        bundle.statistics["optimizer_names"],
        bundle.statistics["category_names"],
        bundle.statistics["metric_scale"],
        bundle.statistics["runtime_99_seconds"],
        config["reward"],
        config["observation"],
        **env_config,
    )

    selected: list[dict[str, Any]] = []
    stop_action = len(OPTIMIZER_NAMES)
    for circuit in sorted(frame["original_circuit"].unique()):
        initial_row = frame[frame["original_circuit"] == circuit].iloc[0]
        observation, _ = env.reset(options={"circuit": circuit})
        first_action, _ = model.predict(observation, deterministic=True)
        first_action = int(np.asarray(first_action).item())
        actions = [first_action]

        if first_action < 0 or first_action > stop_action or first_action == stop_action:
            record = _synthetic_noop_row(initial_row, invalid_action=True)
            record["actions"] = json.dumps(actions)
            selected.append(record)
            continue

        first_name = OPTIMIZER_NAMES[first_action]
        first_row = _lookup_chain_row(frame, circuit, first_name)
        second_observation, _, terminated, truncated, _ = env.step(first_action)
        selected_row = first_row

        if not bool(first_row["error"]) and not (terminated or truncated):
            second_action, _ = model.predict(
                second_observation, deterministic=True
            )
            second_action = int(np.asarray(second_action).item())
            actions.append(second_action)
            if second_action < 0 or second_action > stop_action:
                record = _synthetic_noop_row(initial_row, invalid_action=True)
                record["actions"] = json.dumps(actions)
                selected.append(record)
                continue
            if second_action != stop_action:
                chain_name = f"{first_name}__{OPTIMIZER_NAMES[second_action]}"
                selected_row = _lookup_chain_row(frame, circuit, chain_name)

        record = selected_row.to_dict()
        record["selected_chain"] = str(selected_row["chain_name"])
        record["invalid_action"] = False
        record["failed"] = bool(selected_row["error"])
        record["actions"] = json.dumps(actions)
        selected.append(record)

    env.close()
    return pd.DataFrame(selected)


def _all_raw_outcomes(
    frame: pd.DataFrame,
    statistics: dict[str, Any],
) -> pd.DataFrame:
    """Convert every available chain row into an evaluator outcome."""

    rows = frame.copy()
    rows["selected_chain"] = rows["chain_name"]
    rows["invalid_action"] = False
    rows["failed"] = rows["error"].astype(bool)
    return selected_rows_to_outcomes(rows, statistics)


def _fixed_chain_outcomes(
    all_outcomes: pd.DataFrame, chain_name: str
) -> pd.DataFrame:
    """Select the same chain for every circuit."""

    selected = all_outcomes[all_outcomes["chain_name"] == chain_name].copy()
    if selected["original_circuit"].nunique() != all_outcomes[
        "original_circuit"
    ].nunique():
        raise ValueError(f"Fixed chain {chain_name!r} is not available everywhere.")
    return selected


def _no_optimization_outcomes(
    frame: pd.DataFrame, statistics: dict[str, Any]
) -> pd.DataFrame:
    """Construct the unchanged no-optimization baseline."""

    rows = [
        _synthetic_noop_row(group.iloc[0], invalid_action=False)
        for _, group in frame.groupby("original_circuit", sort=True)
    ]
    result = pd.DataFrame(rows)
    result["failed"] = False
    return selected_rows_to_outcomes(result, statistics)


def compute_baseline_scores(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    statistics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Recompute all baselines independently for every objective track."""

    train_all = _all_raw_outcomes(train_frame, statistics)
    eval_all = _all_raw_outcomes(evaluation_frame, statistics)
    no_optimization = _no_optimization_outcomes(evaluation_frame, statistics)
    wisq = _fixed_chain_outcomes(eval_all, WISQ_BASELINE_CHAIN)
    single_names = [name for name in OPTIMIZER_NAMES]

    results: dict[str, dict[str, Any]] = {}
    for track_id, track in objective_tracks().items():
        no_score = float(_row_contribution(no_optimization, track).mean())
        wisq_score = float(_row_contribution(wisq, track).mean())

        # Random expected performance is averaged within each circuit before
        # macro-averaging, so circuits with more rows can never gain more weight.
        eval_all_contribution = _row_contribution(eval_all, track)
        random_score = float(
            eval_all.assign(_contribution=eval_all_contribution)
            .groupby("original_circuit")["_contribution"]
            .mean()
            .mean()
        )

        train_contribution = _row_contribution(train_all, track)
        train_scored = train_all.assign(_contribution=train_contribution)
        chain_means = train_scored.groupby("chain_name")["_contribution"].mean()
        best_fixed_chain = str(chain_means.idxmax())
        best_single_chain = str(chain_means.loc[single_names].idxmax())

        best_fixed_score = float(
            _row_contribution(
                _fixed_chain_outcomes(eval_all, best_fixed_chain), track
            ).mean()
        )
        best_single_score = float(
            _row_contribution(
                _fixed_chain_outcomes(eval_all, best_single_chain), track
            ).mean()
        )

        oracle_score = float(
            eval_all.assign(_contribution=eval_all_contribution)
            .groupby("original_circuit")["_contribution"]
            .max()
            .mean()
        )
        results[track_id] = {
            "no_optimization": {"score": no_score},
            "random_expected": {"score": random_score},
            "tket__wisq_bqskit": {"score": wisq_score},
            "best_fixed_single": {
                "score": best_single_score,
                "chain": best_single_chain,
            },
            "best_fixed_chain": {
                "score": best_fixed_score,
                "chain": best_fixed_chain,
            },
            "validation_oracle": {"score": oracle_score},
        }
    return results


def evaluate_model(
    *,
    model_path: Path,
    config: dict[str, Any],
    bundle: DataBundle,
    split: str = "validation",
    device: str = "auto",
    include_baselines: bool = True,
) -> EvaluationResult:
    """Evaluate one saved model against locked raw-outcome objectives."""

    if split not in {"validation", "test"}:
        raise ValueError("Locked model evaluation supports validation or test only.")
    frame = bundle.validation if split == "validation" else bundle.test

    # The model receives observations from the experiment's environment, but
    # selected chains and outcomes are reconstructed by this locked module.
    env_config = dict(config["environment"])
    env_config["shuffle_each_epoch"] = False
    load_env = OptimizerChainEnv(
        frame,
        bundle.statistics["optimizer_names"],
        bundle.statistics["category_names"],
        bundle.statistics["metric_scale"],
        bundle.statistics["runtime_99_seconds"],
        config["reward"],
        config["observation"],
        **env_config,
    )
    model = load_model(
        config["algorithm"], model_path, env=load_env, device=device
    )
    selected_rows = decode_policy_selections(
        model=model, frame=frame, bundle=bundle, config=config
    )
    outcomes = selected_rows_to_outcomes(selected_rows, bundle.statistics)

    summary: dict[str, Any] = {
        "split": split,
        "algorithm": config["algorithm"],
        "objective_scores": score_objectives(outcomes),
        "metric_summary": summarize_metrics(outcomes),
        "pareto_vector": pareto_vector(outcomes),
        "objective_track_count": len(objective_tracks()),
    }
    if include_baselines:
        summary["baselines"] = compute_baseline_scores(
            train_frame=bundle.train,
            evaluation_frame=frame,
            statistics=bundle.statistics,
        )
    load_env.close()
    return EvaluationResult(summary=summary, selected=outcomes)


def main() -> None:
    """Evaluate an explicitly supplied model, including opt-in test evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_model(
        model_path=args.model,
        config=config,
        bundle=ensure_data_setup(),
        split=args.split,
        device=args.device,
    )
    payload = result.summary
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.selected.to_csv(args.output.with_suffix(".selected.csv"), index=False)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
