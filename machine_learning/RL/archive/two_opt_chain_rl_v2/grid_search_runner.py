"""Run a broad v2 RL sweep across algorithms, rewards, and reward weights.

This script is intentionally simple:
- it reuses train.py for the actual training/evaluation flow,
- it always enables the existing per-run hyperparameter grid search,
- and it writes one summary row per completed run to a CSV.

The terminal output is designed to stay readable during long sweeps by showing
outer sweep progress, remaining runs, and elapsed/estimated time.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    from .evaluation import summarize_selected_frame
    from .hyperparameter_search import expand_search_grid
    from .rewards import SUPPORTED_REWARD_TYPES
    from .train import (
        ALGORITHMS,
        DEFAULT_SEARCH_GRIDS,
        DEPTH_WEIGHT,
        FINAL_REWARD_WEIGHT,
        GRID_SEARCH_TIMESTEPS,
        RUNTIME_WEIGHT,
        TOTAL_TIMESTEPS,
        TWO_QUBIT_WEIGHT,
        resolved_search_grid,
        train_model,
    )
except ImportError:
    from evaluation import summarize_selected_frame
    from hyperparameter_search import expand_search_grid
    from rewards import SUPPORTED_REWARD_TYPES
    from train import (
        ALGORITHMS,
        DEFAULT_SEARCH_GRIDS,
        DEPTH_WEIGHT,
        FINAL_REWARD_WEIGHT,
        GRID_SEARCH_TIMESTEPS,
        RUNTIME_WEIGHT,
        TOTAL_TIMESTEPS,
        TWO_QUBIT_WEIGHT,
        resolved_search_grid,
        train_model,
    )


DEFAULT_RUNTIME_WEIGHTS = [0.0, RUNTIME_WEIGHT, 0.2]
DEFAULT_TWO_QUBIT_WEIGHTS = [TWO_QUBIT_WEIGHT]
DEFAULT_DEPTH_WEIGHTS = [DEPTH_WEIGHT]
DEFAULT_FINAL_REWARD_WEIGHTS = [0.25, FINAL_REWARD_WEIGHT, 1.0]
DEFAULT_ALGORITHMS = sorted(ALGORITHMS.keys())
DEFAULT_REWARDS = list(SUPPORTED_REWARD_TYPES)
SWEEP_RESULTS_PATH = Path(__file__).resolve().parent / "grid_search_runs.csv"

SWEEP_LOG_COLUMNS = [
    "sweep_id",
    "run_index",
    "total_runs",
    "status",
    "algorithm_name",
    "reward_type",
    "runtime_weight",
    "two_qubit_weight",
    "depth_weight",
    "final_reward_weight",
    "total_timesteps",
    "grid_search_timesteps",
    "grid_search_candidates",
    "elapsed_seconds",
    "model_path",
    "best_config",
    "validation_utility",
    "test_utility",
    "validation_avg_two_qubit_gate_reduction_pct",
    "test_avg_two_qubit_gate_reduction_pct",
    "validation_avg_depth_reduction_pct",
    "test_avg_depth_reduction_pct",
    "validation_avg_runtime_seconds",
    "test_avg_runtime_seconds",
    "validation_avg_runtime_score",
    "test_avg_runtime_score",
    "error_message",
]


def parse_csv_list(value, cast_fn=str):
    """Turn a comma-separated CLI string into a clean Python list."""
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return [cast_fn(item) for item in items]


def build_arg_parser():
    """Create the command-line interface for the full experiment sweep."""
    parser = argparse.ArgumentParser(
        description="Run the full v2 RL grid-search sweep.",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms to include.",
    )
    parser.add_argument(
        "--reward-types",
        default=",".join(DEFAULT_REWARDS),
        help="Comma-separated reward types to include.",
    )
    parser.add_argument(
        "--runtime-weights",
        default=",".join(str(value) for value in DEFAULT_RUNTIME_WEIGHTS),
        help="Comma-separated runtime weights to test.",
    )
    parser.add_argument(
        "--two-qubit-weights",
        default=",".join(str(value) for value in DEFAULT_TWO_QUBIT_WEIGHTS),
        help="Comma-separated two-qubit weights to test.",
    )
    parser.add_argument(
        "--depth-weights",
        default=",".join(str(value) for value in DEFAULT_DEPTH_WEIGHTS),
        help="Comma-separated depth weights to test.",
    )
    parser.add_argument(
        "--final-reward-weights",
        default=",".join(str(value) for value in DEFAULT_FINAL_REWARD_WEIGHTS),
        help="Comma-separated final reward weights to test.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Training timesteps for each final model.",
    )
    parser.add_argument(
        "--grid-search-timesteps",
        type=int,
        default=GRID_SEARCH_TIMESTEPS,
        help="Training timesteps for each hyperparameter-search candidate.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of outer sweep runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep without training anything.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SWEEP_RESULTS_PATH,
        help="Where to append sweep summary rows.",
    )
    return parser


def validate_choices(algorithms, reward_types):
    """Catch unsupported algorithm/reward names before training starts."""
    invalid_algorithms = sorted(set(algorithms) - set(ALGORITHMS.keys()))
    if invalid_algorithms:
        raise ValueError(f"Unsupported algorithms: {invalid_algorithms}")

    invalid_rewards = sorted(set(reward_types) - set(SUPPORTED_REWARD_TYPES))
    if invalid_rewards:
        raise ValueError(f"Unsupported reward types: {invalid_rewards}")


def build_run_plan(
    algorithms,
    reward_types,
    runtime_weights,
    two_qubit_weights,
    depth_weights,
    final_reward_weights,
):
    """Create the full outer sweep plan."""
    return [
        {
            "algorithm_name": algorithm_name,
            "reward_type": reward_type,
            "runtime_weight": float(runtime_weight),
            "two_qubit_weight": float(two_qubit_weight),
            "depth_weight": float(depth_weight),
            "final_reward_weight": float(final_reward_weight),
        }
        for (
            algorithm_name,
            reward_type,
            runtime_weight,
            two_qubit_weight,
            depth_weight,
            final_reward_weight,
        ) in product(
            algorithms,
            reward_types,
            runtime_weights,
            two_qubit_weights,
            depth_weights,
            final_reward_weights,
        )
    ]


def search_candidate_count(algorithm_name):
    """Return how many inner hyperparameter candidates this algorithm will test."""
    search_grid = resolved_search_grid(
        algorithm_name.lower(),
        search_grid=DEFAULT_SEARCH_GRIDS.get(algorithm_name.lower(), {}),
    )
    if not search_grid:
        return 0
    return len(list(expand_search_grid(search_grid)))


def format_seconds(seconds):
    """Format elapsed time in a readable H:MM:SS style."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def progress_bar(completed, total, width=28):
    """Render a small ASCII progress bar for terminal output."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, math.floor((completed / total) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def append_sweep_row(csv_path, row):
    """Append one sweep result row to disk immediately."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def normalize_float_key(value):
    """Convert numeric-like values into a stable float for run matching."""
    return round(float(value), 10)


def run_identity_key(run, total_timesteps, grid_search_timesteps):
    """Build the outer-sweep identity used to detect duplicate configurations."""
    return (
        str(run["algorithm_name"]).lower(),
        str(run["reward_type"]),
        normalize_float_key(run["runtime_weight"]),
        normalize_float_key(run["two_qubit_weight"]),
        normalize_float_key(run["depth_weight"]),
        normalize_float_key(run["final_reward_weight"]),
        int(total_timesteps),
        int(grid_search_timesteps),
    )


def load_existing_run_keys(csv_path):
    """Read the existing sweep CSV and return the set of already-recorded run keys."""
    if not csv_path.exists():
        return set()

    existing_keys = set()
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                existing_keys.add(
                    (
                        str(row["algorithm_name"]).lower(),
                        str(row["reward_type"]),
                        normalize_float_key(row["runtime_weight"]),
                        normalize_float_key(
                            row.get("two_qubit_weight", TWO_QUBIT_WEIGHT)
                        ),
                        normalize_float_key(row.get("depth_weight", DEPTH_WEIGHT)),
                        normalize_float_key(row["final_reward_weight"]),
                        int(float(row["total_timesteps"])),
                        int(float(row["grid_search_timesteps"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return existing_keys


def summarize_run(results):
    """Extract the validation/test summary values from one train_model result."""
    validation_summary = summarize_selected_frame(results["validation_selected"])
    test_summary = summarize_selected_frame(results["test_selected"])
    return validation_summary, test_summary


def print_plan_summary(run_plan, total_inner_candidates, total_timesteps, grid_search_timesteps):
    """Print a clear overview before the sweep starts."""
    print("=" * 100)
    print("V2 RL SWEEP PLAN")
    print("=" * 100)
    print(f"Outer runs: {len(run_plan)}")
    print(f"Total inner hyperparameter candidates per full sweep pass: {total_inner_candidates}")
    print(f"Final-model timesteps per run: {total_timesteps}")
    print(f"Grid-search timesteps per candidate: {grid_search_timesteps}")
    print("-" * 100)


def run_full_sweep(
    algorithms,
    reward_types,
    runtime_weights,
    two_qubit_weights,
    depth_weights,
    final_reward_weights,
    total_timesteps,
    grid_search_timesteps,
    output_csv,
    max_runs=None,
    dry_run=False,
):
    """Run the outer sweep and keep going even if one configuration fails."""
    validate_choices(algorithms, reward_types)

    full_run_plan = build_run_plan(
        algorithms,
        reward_types,
        runtime_weights,
        two_qubit_weights,
        depth_weights,
        final_reward_weights,
    )
    if max_runs is not None:
        full_run_plan = full_run_plan[:max_runs]

    existing_run_keys = load_existing_run_keys(output_csv)
    skipped_runs = []
    run_plan = []
    for run in full_run_plan:
        identity_key = run_identity_key(
            run,
            total_timesteps=total_timesteps,
            grid_search_timesteps=grid_search_timesteps,
        )
        if identity_key in existing_run_keys:
            skipped_runs.append(run)
            continue
        run_plan.append(run)

    inner_counts = {
        algorithm_name: search_candidate_count(algorithm_name)
        for algorithm_name in algorithms
    }
    total_inner_candidates = sum(
        inner_counts[run["algorithm_name"]] for run in run_plan
    )
    print_plan_summary(
        run_plan,
        total_inner_candidates,
        total_timesteps,
        grid_search_timesteps,
    )
    print(f"Already in CSV (skipped): {len(skipped_runs)}")
    print(f"Remaining to run: {len(run_plan)}")
    if skipped_runs:
        print("-" * 100)
        print("Skipping configurations already present in the results CSV:")
        for index, run in enumerate(skipped_runs, start=1):
            print(
                f"{index:>3}. {run['algorithm_name']:>3} | "
                f"{run['reward_type']:<16} | "
                f"runtime_weight={run['runtime_weight']:<4} | "
                f"two_qubit_weight={run['two_qubit_weight']:<4} | "
                f"depth_weight={run['depth_weight']:<4} | "
                f"final_reward_weight={run['final_reward_weight']}"
            )
        print("-" * 100)

    for index, run in enumerate(run_plan, start=1):
        print(
            f"{index:>3}. {run['algorithm_name']:>3} | "
            f"{run['reward_type']:<16} | "
            f"runtime_weight={run['runtime_weight']:<4} | "
            f"two_qubit_weight={run['two_qubit_weight']:<4} | "
            f"depth_weight={run['depth_weight']:<4} | "
            f"final_reward_weight={run['final_reward_weight']}"
        )
    print("=" * 100)

    if dry_run:
        print("Dry run only. No models were trained.")
        return []

    if not run_plan:
        print("No new runs remain after checking the existing results CSV.")
        return []

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_start = time.time()
    rows = []

    for run_index, run in enumerate(run_plan, start=1):
        completed = run_index - 1
        remaining = len(run_plan) - completed
        elapsed = time.time() - sweep_start
        avg_seconds = elapsed / completed if completed else 0.0
        eta_seconds = avg_seconds * (len(run_plan) - completed) if completed else 0.0

        print()
        print("=" * 100)
        print(
            f"Sweep {progress_bar(completed, len(run_plan))} "
            f"{completed}/{len(run_plan)} done | {remaining} left"
        )
        print(
            f"Now running {run_index}/{len(run_plan)}: "
            f"{run['algorithm_name']} | {run['reward_type']} | "
            f"runtime_weight={run['runtime_weight']} | "
            f"two_qubit_weight={run['two_qubit_weight']} | "
            f"depth_weight={run['depth_weight']} | "
            f"final_reward_weight={run['final_reward_weight']}"
        )
        print(
            f"Inner hyperparameter candidates: {inner_counts[run['algorithm_name']]} | "
            f"Elapsed: {format_seconds(elapsed)} | "
            f"ETA: {format_seconds(eta_seconds)}"
        )
        print("=" * 100)

        run_start = time.time()
        try:
            results = train_model(
                algorithm_name=run["algorithm_name"],
                reward_type=run["reward_type"],
                runtime_weight=run["runtime_weight"],
                two_qubit_weight=run["two_qubit_weight"],
                depth_weight=run["depth_weight"],
                final_reward_weight=run["final_reward_weight"],
                total_timesteps=total_timesteps,
                grid_search_timesteps=grid_search_timesteps,
                run_search=True,
            )
            validation_summary, test_summary = summarize_run(results)
            elapsed_seconds = time.time() - run_start

            row = {
                "sweep_id": sweep_id,
                "run_index": run_index,
                "total_runs": len(run_plan),
                "status": "success",
                "algorithm_name": run["algorithm_name"],
                "reward_type": run["reward_type"],
                "runtime_weight": run["runtime_weight"],
                "two_qubit_weight": run["two_qubit_weight"],
                "depth_weight": run["depth_weight"],
                "final_reward_weight": run["final_reward_weight"],
                "total_timesteps": int(total_timesteps),
                "grid_search_timesteps": int(grid_search_timesteps),
                "grid_search_candidates": inner_counts[run["algorithm_name"]],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "model_path": str(results["model_path"]),
                "best_config": json.dumps(results["config"], sort_keys=True),
                "validation_utility": validation_summary["utility"],
                "test_utility": test_summary["utility"],
                "validation_avg_two_qubit_gate_reduction_pct": validation_summary[
                    "avg_two_qubit_gate_reduction_pct"
                ],
                "test_avg_two_qubit_gate_reduction_pct": test_summary[
                    "avg_two_qubit_gate_reduction_pct"
                ],
                "validation_avg_depth_reduction_pct": validation_summary[
                    "avg_depth_reduction_pct"
                ],
                "test_avg_depth_reduction_pct": test_summary[
                    "avg_depth_reduction_pct"
                ],
                "validation_avg_runtime_seconds": validation_summary[
                    "avg_runtime_seconds"
                ],
                "test_avg_runtime_seconds": test_summary["avg_runtime_seconds"],
                "validation_avg_runtime_score": validation_summary[
                    "avg_runtime_score"
                ],
                "test_avg_runtime_score": test_summary["avg_runtime_score"],
                "error_message": "",
            }

            print(
                f"Finished in {format_seconds(elapsed_seconds)} | "
                f"val utility={validation_summary['utility']:.4f} | "
                f"test utility={test_summary['utility']:.4f}"
            )
        except Exception as exc:
            elapsed_seconds = time.time() - run_start
            row = {
                "sweep_id": sweep_id,
                "run_index": run_index,
                "total_runs": len(run_plan),
                "status": "failed",
                "algorithm_name": run["algorithm_name"],
                "reward_type": run["reward_type"],
                "runtime_weight": run["runtime_weight"],
                "two_qubit_weight": run["two_qubit_weight"],
                "depth_weight": run["depth_weight"],
                "final_reward_weight": run["final_reward_weight"],
                "total_timesteps": int(total_timesteps),
                "grid_search_timesteps": int(grid_search_timesteps),
                "grid_search_candidates": inner_counts[run["algorithm_name"]],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "model_path": "",
                "best_config": "",
                "validation_utility": "",
                "test_utility": "",
                "validation_avg_two_qubit_gate_reduction_pct": "",
                "test_avg_two_qubit_gate_reduction_pct": "",
                "validation_avg_depth_reduction_pct": "",
                "test_avg_depth_reduction_pct": "",
                "validation_avg_runtime_seconds": "",
                "test_avg_runtime_seconds": "",
                "validation_avg_runtime_score": "",
                "test_avg_runtime_score": "",
                "error_message": str(exc),
            }
            print(f"Run failed after {format_seconds(elapsed_seconds)}")
            print(f"Error: {exc}")

        append_sweep_row(output_csv, row)
        rows.append(row)

    total_elapsed = time.time() - sweep_start
    successes = sum(1 for row in rows if row["status"] == "success")
    failures = len(rows) - successes

    print()
    print("=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)
    print(f"Total runs: {len(rows)}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Elapsed: {format_seconds(total_elapsed)}")
    print(f"Results CSV: {output_csv}")
    print("=" * 100)

    return rows


def main():
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    algorithms = [value.lower() for value in parse_csv_list(args.algorithms, str)]
    reward_types = parse_csv_list(args.reward_types, str)
    runtime_weights = parse_csv_list(args.runtime_weights, float)
    two_qubit_weights = parse_csv_list(args.two_qubit_weights, float)
    depth_weights = parse_csv_list(args.depth_weights, float)
    final_reward_weights = parse_csv_list(args.final_reward_weights, float)

    run_full_sweep(
        algorithms=algorithms,
        reward_types=reward_types,
        runtime_weights=runtime_weights,
        two_qubit_weights=two_qubit_weights,
        depth_weights=depth_weights,
        final_reward_weights=final_reward_weights,
        total_timesteps=args.total_timesteps,
        grid_search_timesteps=args.grid_search_timesteps,
        output_csv=args.output_csv,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
