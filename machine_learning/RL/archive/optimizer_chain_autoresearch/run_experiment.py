"""Fixed autonomous experiment runner and research-memory manager.

READ-ONLY DURING AUTONOMOUS RESEARCH.

This is the normal entry point for candidate experiments.  It snapshots the
editable source, rejects exact duplicates, runs tiered seeds, invokes the
locked evaluator, updates champion/Pareto registries, and writes a readable
research report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .archive import (
        aggregate_evaluations,
        append_jsonl,
        canonical_json,
        experiment_fingerprint,
        is_promising,
        ownership_summary,
        primary_track_id,
        promote_finalist,
        read_json,
        retained_experiment_ids,
        reward_family,
        reward_signature,
        sha256_text,
        utc_now,
        write_json_atomic,
    )
    from .data_setup import DataBundle, ensure_data_setup
    from .evaluate import evaluate_model
    from .experiment_config import get_experiment_config
    from .train import train_one_seed
except ImportError:  # pragma: no cover - supports ``python run_experiment.py``
    from archive import (
        aggregate_evaluations,
        append_jsonl,
        canonical_json,
        experiment_fingerprint,
        is_promising,
        ownership_summary,
        primary_track_id,
        promote_finalist,
        read_json,
        retained_experiment_ids,
        reward_family,
        reward_signature,
        sha256_text,
        utc_now,
        write_json_atomic,
    )
    from data_setup import DataBundle, ensure_data_setup
    from evaluate import evaluate_model
    from experiment_config import get_experiment_config
    from train import train_one_seed


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PACKAGE_DIR / "runs"
DEFAULT_REGISTRY_DIR = PACKAGE_DIR
EDITABLE_FILENAMES = (
    "experiment_config.py",
    "reward.py",
    "optimizer_chain_env.py",
    "train.py",
)

STAGE_DEFAULTS = {
    "screen": {"seeds": [42], "timesteps": 10_000},
    "confirm": {"seeds": [42, 43, 44], "timesteps": 30_000},
    "finalize": {"seeds": [42, 43, 44, 45, 46], "timesteps": 30_000},
}

RESULT_COLUMNS = (
    "timestamp",
    "experiment_id",
    "stage",
    "algorithm",
    "seeds",
    "timesteps",
    "primary_track",
    "primary_score",
    "status",
    "reward_family",
    "reward_signature",
    "fingerprint",
    "hypothesis",
)


def _slug(value: str, limit: int = 36) -> str:
    """Create a short filesystem-safe label."""

    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "experiment")[:limit].rstrip("-")


def editable_source_text(package_dir: Path = PACKAGE_DIR) -> dict[str, str]:
    """Read all agent-editable source files for hashing and snapshots."""

    return {
        filename: (package_dir / filename).read_text(encoding="utf-8")
        for filename in EDITABLE_FILENAMES
    }


def _new_experiment_id(config: dict[str, Any], sources: dict[str, str]) -> str:
    """Generate a chronological ID with a source/config digest suffix."""

    timestamp = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
    timestamp = timestamp.split(".")[0]
    digest = sha256_text(canonical_json({"config": config, "sources": sources}))[:8]
    return f"{timestamp}_{_slug(config['hypothesis'])}_{digest}"


def _registry_paths(registry_dir: Path) -> dict[str, Path]:
    """Return all persistent research-memory paths."""

    return {
        "events": registry_dir / "experiments.jsonl",
        "results": registry_dir / "results.tsv",
        "champions": registry_dir / "champions.json",
        "pareto": registry_dir / "pareto.json",
        "report": registry_dir / "research_report.md",
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read the append-only experiment ledger."""

    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_not_duplicate(
    fingerprint: str,
    events_path: Path,
    *,
    allow_duplicate: bool,
) -> None:
    """Reject exact repeated work before spending training compute."""

    if allow_duplicate:
        return
    for event in _read_events(events_path):
        if event.get("fingerprint") == fingerprint:
            raise RuntimeError(
                "Exact duplicate experiment rejected. Existing record: "
                f"{event.get('experiment_id')} stage={event.get('stage')}"
            )


def _snapshot_sources(
    experiment_dir: Path,
    sources: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Persist the exact editable code and normalized config used by a run."""

    snapshot_dir = experiment_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in sources.items():
        (snapshot_dir / filename).write_text(content, encoding="utf-8")
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (experiment_dir / "hypothesis.md").write_text(
        f"# Hypothesis\n\n{config['hypothesis'].strip()}\n", encoding="utf-8"
    )


def _append_results_tsv(path: Path, event: dict[str, Any]) -> None:
    """Append a compact human-readable row to ``results.tsv``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        if needs_header:
            writer.writeheader()
        writer.writerow({column: event.get(column, "") for column in RESULT_COLUMNS})


def _run_seed(
    *,
    seed: int,
    timesteps: int,
    stage_dir: Path,
    config: dict[str, Any],
    bundle: DataBundle,
    device: str,
    include_baselines: bool,
) -> tuple[dict[str, Any], str]:
    """Train and evaluate one seed, preserving all per-seed artifacts."""

    seed_dir = stage_dir / f"seed_{seed}"
    training = train_one_seed(
        config=config,
        bundle=bundle,
        output_dir=seed_dir,
        seed=seed,
        total_timesteps=timesteps,
        device=device,
    )
    model_path = Path(training["model_path"])
    evaluation = evaluate_model(
        model_path=model_path,
        config=config,
        bundle=bundle,
        split="validation",
        device=device,
        include_baselines=include_baselines,
    )
    evaluation_path = seed_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation.selected.to_csv(seed_dir / "selected_chains.csv", index=False)
    return evaluation.summary, str(model_path)


def _run_stage(
    *,
    experiment_id: str,
    stage: str,
    seeds: list[int],
    timesteps: int,
    experiment_dir: Path,
    config: dict[str, Any],
    bundle: DataBundle,
    device: str,
    include_baselines: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Execute all seeds for one research stage."""

    stage_dir = experiment_dir / stage
    seed_evaluations: dict[int, dict[str, Any]] = {}
    model_paths: list[str] = []
    for index, seed in enumerate(seeds):
        print(
            f"[{experiment_id}] {stage}: seed={seed}, timesteps={timesteps}",
            flush=True,
        )
        summary, model_path = _run_seed(
            seed=seed,
            timesteps=timesteps,
            stage_dir=stage_dir,
            config=config,
            bundle=bundle,
            device=device,
            include_baselines=include_baselines and index == 0,
        )
        seed_evaluations[seed] = summary
        model_paths.append(model_path)

    aggregate = aggregate_evaluations(seed_evaluations)
    (stage_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate, model_paths


def _combine_final_evaluations(
    experiment_dir: Path,
    confirm_aggregate: dict[str, Any],
    final_seed_summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Combine 30k-step confirm seeds 42–44 with final seeds 45–46."""

    seed_evaluations: dict[int, dict[str, Any]] = {}
    for seed in (42, 43, 44):
        path = experiment_dir / "confirm" / f"seed_{seed}" / "evaluation.json"
        seed_evaluations[seed] = json.loads(path.read_text(encoding="utf-8"))
    seed_evaluations.update(final_seed_summaries)
    aggregate = aggregate_evaluations(seed_evaluations)
    # Preserve the baseline bundle calculated during confirmation.
    if "baselines" in confirm_aggregate:
        aggregate["baselines"] = confirm_aggregate["baselines"]
    return aggregate


def _event_record(
    *,
    experiment_id: str,
    stage: str,
    config: dict[str, Any],
    seeds: list[int],
    timesteps: int,
    fingerprint: str,
    aggregate: dict[str, Any],
    status: str,
    model_paths: list[str],
    promotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete append-only experiment-stage event."""

    primary = primary_track_id(config)
    return {
        "timestamp": utc_now(),
        "experiment_id": experiment_id,
        "stage": stage,
        "algorithm": config["algorithm"],
        "seeds": ",".join(map(str, seeds)),
        "timesteps": timesteps,
        "primary_track": primary,
        "primary_score": aggregate["objective_scores"][primary],
        "status": status,
        "reward_family": reward_family(config),
        "reward_signature": reward_signature(config),
        "fingerprint": fingerprint,
        "hypothesis": config["hypothesis"],
        "model_paths": model_paths,
        "aggregate": aggregate,
        "promotion": promotion,
    }


def _write_report(
    *,
    path: Path,
    champions: dict[str, Any],
    pareto: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Regenerate a concise Markdown report from canonical registries."""

    ownership = ownership_summary(champions)
    lines = [
        "# Optimizer-Chain Autoresearch Report",
        "",
        f"Last updated: {utc_now()}",
        "",
        "## Objective champions",
        "",
        f"Locked objective tracks: {len(champions.get('objective_tracks', {}))}/75",
        "",
    ]
    if ownership:
        lines.extend(
            f"- `{experiment_id}` owns {count} objective track(s)."
            for experiment_id, count in sorted(
                ownership.items(), key=lambda item: (-item[1], item[0])
            )
        )
    else:
        lines.append("- No five-seed champions have been promoted yet.")

    lines.extend(["", "## Reward champions", ""])
    lines.append(
        f"- Reward-family champions: {len(champions.get('reward_families', {}))}"
    )
    lines.append(
        f"- Exact reward-signature champions: "
        f"{len(champions.get('reward_signatures', {}))}"
    )
    lines.extend(["", "## Pareto archive", ""])
    if pareto:
        for entry in pareto:
            vector = entry["vector"]
            lines.append(
                f"- `{entry['experiment_id']}`: "
                f"2Q={vector['two_qubit_gates']:.4f}, "
                f"depth={vector['depth']:.4f}, "
                f"gates={vector['total_gates']:.4f}, "
                f"2Q-depth={vector['two_qubit_depth']:.4f}, "
                f"runtime={vector['runtime_normalized']:.4f}, "
                f"failures={vector['failure_rate']:.4f}"
            )
    else:
        lines.append("- No finalized Pareto entries yet.")

    lines.extend(["", "## Recent experiments", ""])
    for event in events[-20:][::-1]:
        lines.append(
            f"- `{event['experiment_id']}` / {event['stage']}: "
            f"**{event['status']}**, {event['algorithm'].upper()}, "
            f"primary score={float(event['primary_score']):.6f} — "
            f"{event['hypothesis']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delete_model_paths(paths: list[str]) -> None:
    """Remove unretained model ZIPs while preserving metadata and evaluations."""

    for value in paths:
        path = Path(value)
        if path.exists():
            path.unlink()


def _cleanup_displaced_models(
    *,
    runs_dir: Path,
    champions: dict[str, Any],
    pareto: list[dict[str, Any]],
) -> None:
    """Delete ZIPs for experiments no longer referenced by any archive."""

    retained = retained_experiment_ids(champions, pareto)
    if not runs_dir.exists():
        return
    for experiment_dir in runs_dir.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name in retained:
            continue
        for model_path in experiment_dir.glob("**/model.zip"):
            model_path.unlink()


def execute(
    *,
    stage: str,
    label: str | None = None,
    device: str = "auto",
    timesteps_override: int | None = None,
    seeds_override: list[int] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    allow_duplicate: bool = False,
) -> dict[str, Any]:
    """Execute one fixed stage or the automatic screen/confirm/finalize ladder."""

    if stage not in {"screen", "confirm", "finalize", "auto"}:
        raise ValueError(f"Unknown stage: {stage!r}")
    config = get_experiment_config()
    if label:
        config["hypothesis"] = f"{label}: {config['hypothesis']}"
    primary_track_id(config)  # Validate before spending compute.
    bundle = ensure_data_setup()
    sources = editable_source_text()
    paths = _registry_paths(registry_dir)
    champions = read_json(
        paths["champions"],
        {
            "objective_tracks": {},
            "reward_families": {},
            "reward_signatures": {},
        },
    )
    pareto = read_json(paths["pareto"], [])
    events = _read_events(paths["events"])

    # Reject the first requested stage before creating a run directory or
    # spending any compute.  Later automatic stages are checked immediately
    # before they begin.
    first_stage = "screen" if stage == "auto" else stage
    first_defaults = STAGE_DEFAULTS[first_stage]
    first_seeds = seeds_override or list(first_defaults["seeds"])
    first_timesteps = timesteps_override or int(first_defaults["timesteps"])
    first_fingerprint = experiment_fingerprint(
        editable_sources=sources,
        config=config,
        stage=first_stage,
        seeds=list(first_seeds),
        timesteps=int(first_timesteps),
    )
    _assert_not_duplicate(
        first_fingerprint, paths["events"], allow_duplicate=allow_duplicate
    )

    experiment_id = _new_experiment_id(config, sources)
    experiment_dir = runs_dir / experiment_id
    if experiment_dir.exists():
        # This is mainly relevant to deliberate ``--allow-duplicate`` runs
        # launched within the same second.
        suffix = 2
        while (runs_dir / f"{experiment_id}_{suffix}").exists():
            suffix += 1
        experiment_id = f"{experiment_id}_{suffix}"
        experiment_dir = runs_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    _snapshot_sources(experiment_dir, sources, config)

    def record(
        stage_name: str,
        seeds: list[int],
        timesteps: int,
        aggregate: dict[str, Any],
        model_paths: list[str],
        status: str,
        promotion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fingerprint = experiment_fingerprint(
            editable_sources=sources,
            config=config,
            stage=stage_name,
            seeds=seeds,
            timesteps=timesteps,
        )
        event = _event_record(
            experiment_id=experiment_id,
            stage=stage_name,
            config=config,
            seeds=seeds,
            timesteps=timesteps,
            fingerprint=fingerprint,
            aggregate=aggregate,
            status=status,
            model_paths=model_paths,
            promotion=promotion,
        )
        append_jsonl(paths["events"], event)
        _append_results_tsv(paths["results"], event)
        events.append(event)
        return event

    try:
        if stage != "auto":
            defaults = STAGE_DEFAULTS[stage]
            seeds = seeds_override or list(defaults["seeds"])
            timesteps = timesteps_override or int(defaults["timesteps"])
            aggregate, model_paths = _run_stage(
                experiment_id=experiment_id,
                stage=stage,
                seeds=seeds,
                timesteps=timesteps,
                experiment_dir=experiment_dir,
                config=config,
                bundle=bundle,
                device=device,
                include_baselines=True,
            )
            if stage == "finalize" and len(seeds) >= 5:
                promotion = promote_finalist(
                    experiment_id=experiment_id,
                    aggregate=aggregate,
                    model_paths=model_paths,
                    config=config,
                    champions=champions,
                    pareto_entries=pareto,
                )
                status = "retained" if promotion["retained"] else "discarded"
                write_json_atomic(paths["champions"], champions)
                write_json_atomic(paths["pareto"], pareto)
                if not promotion["retained"]:
                    _delete_model_paths(model_paths)
                event = record(
                    stage, seeds, timesteps, aggregate, model_paths, status, promotion
                )
            else:
                promising = is_promising(aggregate, champions, pareto)
                status = "promising" if promising else "discarded"
                event = record(
                    stage, seeds, timesteps, aggregate, model_paths, status
                )
                _delete_model_paths(model_paths)
        else:
            screen_timesteps = (
                timesteps_override or STAGE_DEFAULTS["screen"]["timesteps"]
            )
            screen_seeds = seeds_override or STAGE_DEFAULTS["screen"]["seeds"]
            screen_aggregate, screen_models = _run_stage(
                experiment_id=experiment_id,
                stage="screen",
                seeds=list(screen_seeds),
                timesteps=int(screen_timesteps),
                experiment_dir=experiment_dir,
                config=config,
                bundle=bundle,
                device=device,
                include_baselines=True,
            )
            screen_promising = is_promising(screen_aggregate, champions, pareto)
            record(
                "screen",
                list(screen_seeds),
                int(screen_timesteps),
                screen_aggregate,
                screen_models,
                "promising" if screen_promising else "discarded",
            )
            _delete_model_paths(screen_models)
            if not screen_promising:
                event = events[-1]
            else:
                confirm_timesteps = (
                    timesteps_override or STAGE_DEFAULTS["confirm"]["timesteps"]
                )
                confirm_seeds = (
                    seeds_override
                    if seeds_override and len(seeds_override) >= 3
                    else STAGE_DEFAULTS["confirm"]["seeds"]
                )
                confirm_fingerprint = experiment_fingerprint(
                    editable_sources=sources,
                    config=config,
                    stage="confirm",
                    seeds=list(confirm_seeds),
                    timesteps=int(confirm_timesteps),
                )
                _assert_not_duplicate(
                    confirm_fingerprint,
                    paths["events"],
                    allow_duplicate=allow_duplicate,
                )
                confirm_aggregate, confirm_models = _run_stage(
                    experiment_id=experiment_id,
                    stage="confirm",
                    seeds=list(confirm_seeds),
                    timesteps=int(confirm_timesteps),
                    experiment_dir=experiment_dir,
                    config=config,
                    bundle=bundle,
                    device=device,
                    include_baselines=True,
                )
                confirm_promising = is_promising(
                    confirm_aggregate, champions, pareto
                )
                record(
                    "confirm",
                    list(confirm_seeds),
                    int(confirm_timesteps),
                    confirm_aggregate,
                    confirm_models,
                    "promising" if confirm_promising else "discarded",
                )
                if not confirm_promising:
                    _delete_model_paths(confirm_models)
                    event = events[-1]
                else:
                    # In normal auto mode, confirmation already supplies seeds
                    # 42–44 at the final 30k budget.  Only seeds 45–46 are new.
                    if list(confirm_seeds) != [42, 43, 44]:
                        raise ValueError(
                            "Auto finalization requires standard confirm seeds "
                            "42,43,44; use --stage finalize for custom seeds."
                        )
                    final_timesteps = (
                        timesteps_override
                        or STAGE_DEFAULTS["finalize"]["timesteps"]
                    )
                    final_fingerprint = experiment_fingerprint(
                        editable_sources=sources,
                        config=config,
                        stage="finalize",
                        seeds=[42, 43, 44, 45, 46],
                        timesteps=int(final_timesteps),
                    )
                    _assert_not_duplicate(
                        final_fingerprint,
                        paths["events"],
                        allow_duplicate=allow_duplicate,
                    )
                    final_seed_summaries: dict[int, dict[str, Any]] = {}
                    final_models: list[str] = []
                    final_dir = experiment_dir / "finalize"
                    for index, seed in enumerate((45, 46)):
                        summary, model_path = _run_seed(
                            seed=seed,
                            timesteps=int(final_timesteps),
                            stage_dir=final_dir,
                            config=config,
                            bundle=bundle,
                            device=device,
                            include_baselines=False,
                        )
                        final_seed_summaries[seed] = summary
                        final_models.append(model_path)
                    final_aggregate = _combine_final_evaluations(
                        experiment_dir, confirm_aggregate, final_seed_summaries
                    )
                    all_models = [*confirm_models, *final_models]
                    promotion = promote_finalist(
                        experiment_id=experiment_id,
                        aggregate=final_aggregate,
                        model_paths=all_models,
                        config=config,
                        champions=champions,
                        pareto_entries=pareto,
                    )
                    status = "retained" if promotion["retained"] else "discarded"
                    write_json_atomic(paths["champions"], champions)
                    write_json_atomic(paths["pareto"], pareto)
                    if not promotion["retained"]:
                        _delete_model_paths(all_models)
                    event = record(
                        "finalize",
                        [42, 43, 44, 45, 46],
                        int(final_timesteps),
                        final_aggregate,
                        all_models,
                        status,
                        promotion,
                    )

        _cleanup_displaced_models(
            runs_dir=runs_dir, champions=champions, pareto=pareto
        )
        _write_report(
            path=paths["report"],
            champions=champions,
            pareto=pareto,
            events=events,
        )
        return event
    except Exception:
        # Leave the run directory and logs intact for diagnosis.  A compact
        # crash marker makes incomplete attempts discoverable even if no event
        # could safely be appended.
        (experiment_dir / "CRASHED.txt").write_text(
            "Experiment terminated with an exception. Inspect seed logs and the "
            "caller traceback.\n",
            encoding="utf-8",
        )
        raise


def _parse_seeds(value: str | None) -> list[int] | None:
    """Parse comma-separated CLI seeds."""

    if value is None:
        return None
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def main() -> None:
    """Run one autonomous experiment stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["screen", "confirm", "finalize", "auto"],
        default="auto",
    )
    parser.add_argument("--label")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    event = execute(
        stage=args.stage,
        label=args.label,
        device=args.device,
        timesteps_override=args.timesteps,
        seeds_override=_parse_seeds(args.seeds),
        runs_dir=args.runs_dir,
        registry_dir=args.registry_dir,
        allow_duplicate=args.allow_duplicate,
    )
    print(json.dumps(event, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
