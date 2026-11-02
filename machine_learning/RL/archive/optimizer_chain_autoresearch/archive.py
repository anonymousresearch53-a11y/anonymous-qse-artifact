"""Fixed experiment aggregation, champion, and Pareto archive utilities.

READ-ONLY DURING AUTONOMOUS RESEARCH.

Keeping registry logic separate from the command-line runner makes promotion
rules straightforward to test.  All persisted records are JSON-compatible and
reference model artifacts by path rather than duplicating ZIP files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .data_setup import QUALITY_METRICS
    from .evaluate import RUNTIME_WEIGHTS, objective_track_id
except ImportError:  # pragma: no cover
    from data_setup import QUALITY_METRICS
    from evaluate import RUNTIME_WEIGHTS, objective_track_id


QUALITY_ORDER = tuple(QUALITY_METRICS)


def utc_now() -> str:
    """Return a stable ISO-8601 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def experiment_fingerprint(
    *,
    editable_sources: dict[str, str],
    config: dict[str, Any],
    stage: str,
    seeds: list[int],
    timesteps: int,
) -> str:
    """Hash all inputs that make an experiment execution semantically exact."""

    payload = {
        "editable_sources": editable_sources,
        "config": config,
        "stage": stage,
        "seeds": sorted(seeds),
        "timesteps": int(timesteps),
    }
    return sha256_text(canonical_json(payload))


def reward_signature(config: dict[str, Any]) -> str:
    """Return a stable signature for the exact editable reward declaration."""

    return sha256_text(canonical_json(config["reward"]))[:16]


def reward_family(config: dict[str, Any]) -> str:
    """Return the broad reward family name used for family champions."""

    return str(config["reward"].get("family", "unknown"))


def primary_track_id(config: dict[str, Any]) -> str:
    """Resolve and validate the configured primary locked objective track."""

    requested = set(config["primary_objective"]["metrics"])
    unknown = requested - set(QUALITY_ORDER)
    if unknown or not requested:
        raise ValueError(f"Invalid primary-objective metrics: {sorted(unknown)}")
    metrics = tuple(metric for metric in QUALITY_ORDER if metric in requested)
    runtime_weight = float(config["primary_objective"].get("runtime_weight", 0.0))
    if runtime_weight not in RUNTIME_WEIGHTS:
        raise ValueError(
            f"Primary runtime weight must be one of {RUNTIME_WEIGHTS}; "
            f"found {runtime_weight}."
        )
    return objective_track_id(metrics, runtime_weight)


def _mean(values: list[float]) -> float:
    """Return a native float mean for JSON serialization."""

    return float(np.mean(values))


def aggregate_evaluations(
    seed_evaluations: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Macro-average evaluation summaries across independent training seeds."""

    if not seed_evaluations:
        raise ValueError("Cannot aggregate an empty set of seed evaluations.")
    ordered = sorted(seed_evaluations)
    summaries = [seed_evaluations[seed] for seed in ordered]

    objective_ids = summaries[0]["objective_scores"].keys()
    objective_scores = {
        track_id: _mean(
            [summary["objective_scores"][track_id] for summary in summaries]
        )
        for track_id in objective_ids
    }
    per_seed_objectives = {
        str(seed): seed_evaluations[seed]["objective_scores"] for seed in ordered
    }

    pareto_keys = summaries[0]["pareto_vector"].keys()
    aggregate_pareto = {
        key: _mean([summary["pareto_vector"][key] for summary in summaries])
        for key in pareto_keys
    }

    metric_summary: dict[str, Any] = {"quality_metrics": {}}
    for metric in QUALITY_ORDER:
        fields = summaries[0]["metric_summary"]["quality_metrics"][metric].keys()
        metric_summary["quality_metrics"][metric] = {
            field: _mean(
                [
                    summary["metric_summary"]["quality_metrics"][metric][field]
                    for summary in summaries
                ]
            )
            for field in fields
        }
    for field in (
        "runtime_normalized_mean",
        "runtime_seconds_mean",
        "failure_rate",
        "invalid_action_rate",
    ):
        metric_summary[field] = _mean(
            [summary["metric_summary"][field] for summary in summaries]
        )
    metric_summary["circuits"] = int(summaries[0]["metric_summary"]["circuits"])

    result = {
        "seeds": ordered,
        "objective_scores": objective_scores,
        "per_seed_objectives": per_seed_objectives,
        "pareto_vector": aggregate_pareto,
        "metric_summary": metric_summary,
        "objective_track_count": len(objective_scores),
    }
    if "baselines" in summaries[0]:
        result["baselines"] = summaries[0]["baselines"]
    return result


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``.

    Quality dimensions are maximized; runtime and failure rate are minimized.
    At least one dimension must improve strictly.
    """

    maximize = QUALITY_ORDER
    minimize = ("runtime_normalized", "failure_rate")
    no_worse = all(left[key] >= right[key] for key in maximize) and all(
        left[key] <= right[key] for key in minimize
    )
    strictly_better = any(left[key] > right[key] for key in maximize) or any(
        left[key] < right[key] for key in minimize
    )
    return no_worse and strictly_better


def candidate_is_pareto_eligible(
    vector: dict[str, float], pareto_entries: list[dict[str, Any]]
) -> bool:
    """Return whether no current Pareto entry dominates the candidate."""

    return not any(dominates(entry["vector"], vector) for entry in pareto_entries)


def is_promising(
    aggregate: dict[str, Any],
    champions: dict[str, Any],
    pareto_entries: list[dict[str, Any]],
) -> bool:
    """Screen candidates for further seeds using objectives or Pareto potential."""

    objective_champions = champions.get("objective_tracks", {})
    if not objective_champions:
        return True
    if any(
        aggregate["objective_scores"][track_id]
        > float(incumbent.get("score", float("-inf")))
        for track_id, incumbent in objective_champions.items()
        if track_id in aggregate["objective_scores"]
    ):
        return True
    return candidate_is_pareto_eligible(
        aggregate["pareto_vector"], pareto_entries
    )


def _wins_seed_majority(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    track_id: str,
) -> bool:
    """Require aggregate improvement and wins on at least three common seeds."""

    if candidate["objective_scores"][track_id] <= float(incumbent["score"]):
        return False
    candidate_by_seed = candidate["per_seed_objectives"]
    incumbent_by_seed = incumbent.get("per_seed_scores", {})
    common = sorted(set(candidate_by_seed) & set(incumbent_by_seed))
    if len(common) < 5:
        return False
    wins = sum(
        candidate_by_seed[seed][track_id] > incumbent_by_seed[seed]
        for seed in common
    )
    return wins >= 3


def _champion_record(
    *,
    experiment_id: str,
    aggregate: dict[str, Any],
    track_id: str,
    model_paths: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build one reusable champion reference."""

    return {
        "experiment_id": experiment_id,
        "score": aggregate["objective_scores"][track_id],
        "track_id": track_id,
        "per_seed_scores": {
            seed: values[track_id]
            for seed, values in aggregate["per_seed_objectives"].items()
        },
        "model_paths": model_paths,
        "algorithm": config["algorithm"],
        "reward_family": reward_family(config),
        "reward_signature": reward_signature(config),
        "updated_at": utc_now(),
    }


def promote_finalist(
    *,
    experiment_id: str,
    aggregate: dict[str, Any],
    model_paths: list[str],
    config: dict[str, Any],
    champions: dict[str, Any],
    pareto_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Promote a five-seed finalist into objective/reward/Pareto registries."""

    if len(aggregate["seeds"]) < 5:
        raise ValueError("Champion promotion requires five seeded evaluations.")

    champions.setdefault("objective_tracks", {})
    champions.setdefault("reward_families", {})
    champions.setdefault("reward_signatures", {})
    won_tracks: list[str] = []

    for track_id in sorted(aggregate["objective_scores"]):
        incumbent = champions["objective_tracks"].get(track_id)
        if incumbent is None or _wins_seed_majority(
            aggregate, incumbent, track_id
        ):
            champions["objective_tracks"][track_id] = _champion_record(
                experiment_id=experiment_id,
                aggregate=aggregate,
                track_id=track_id,
                model_paths=model_paths,
                config=config,
            )
            won_tracks.append(track_id)

    primary = primary_track_id(config)
    family = reward_family(config)
    signature = reward_signature(config)
    # A reward family can legitimately target different locked objectives.
    # Include that comparison track in the key so unlike scores are never
    # treated as interchangeable.
    for registry_name, key in (
        ("reward_families", f"{family}|{primary}"),
        ("reward_signatures", f"{signature}|{primary}"),
    ):
        incumbent = champions[registry_name].get(key)
        if incumbent is None or _wins_seed_majority(aggregate, incumbent, primary):
            champions[registry_name][key] = _champion_record(
                experiment_id=experiment_id,
                aggregate=aggregate,
                track_id=primary,
                model_paths=model_paths,
                config=config,
            )

    vector = aggregate["pareto_vector"]
    dominated_by_existing = any(
        dominates(entry["vector"], vector) for entry in pareto_entries
    )
    pareto_added = not dominated_by_existing
    if pareto_added:
        pareto_entries[:] = [
            entry
            for entry in pareto_entries
            if not dominates(vector, entry["vector"])
            and entry["experiment_id"] != experiment_id
        ]
        pareto_entries.append(
            {
                "experiment_id": experiment_id,
                "vector": vector,
                "model_paths": model_paths,
                "algorithm": config["algorithm"],
                "reward_family": family,
                "reward_signature": signature,
                "updated_at": utc_now(),
            }
        )
        pareto_entries.sort(key=lambda entry: entry["experiment_id"])

    champions["updated_at"] = utc_now()
    return {
        "won_objective_tracks": won_tracks,
        "pareto_added": pareto_added,
        "retained": bool(won_tracks or pareto_added),
    }


def retained_experiment_ids(
    champions: dict[str, Any], pareto_entries: list[dict[str, Any]]
) -> set[str]:
    """Collect every experiment still referenced by a live archive entry."""

    retained = {entry["experiment_id"] for entry in pareto_entries}
    for registry_name in (
        "objective_tracks",
        "reward_families",
        "reward_signatures",
    ):
        retained.update(
            entry["experiment_id"]
            for entry in champions.get(registry_name, {}).values()
        )
    return retained


def ownership_summary(champions: dict[str, Any]) -> dict[str, int]:
    """Count objective-track titles owned by each experiment."""

    return dict(
        Counter(
            entry["experiment_id"]
            for entry in champions.get("objective_tracks", {}).values()
        )
    )


def read_json(path: Path, default: Any) -> Any:
    """Read JSON or return a caller-provided default when absent."""

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a JSON registry to avoid partial writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    """Append one complete JSON record to an experiment ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
