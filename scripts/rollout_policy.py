#!/usr/bin/env python3
"""Collect conservative online rollouts from a trained RL policy."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from benchmarks.ai_transpile.rl_training.checkpointing import load_checkpoint  # noqa: E402
from benchmarks.ai_transpile.rl_training.factory import create_trainer  # noqa: E402
from benchmarks.ai_transpile.rl_training.online import (  # noqa: E402
    record_rollout,
    rollout_policy,
    summarize_rollouts,
)
from benchmarks.ai_transpile.rl_trajectory.database import TrajectoryDatabase  # noqa: E402


def _filter_circuits(circuits, circuit_kind: str):  # noqa: ANN001, ANN202
    if circuit_kind == "all":
        return circuits
    want_artifact = circuit_kind == "artifact"
    return [c for c in circuits if c.name.startswith("artifact_") == want_artifact]


def _ordered_action_names(source_db: TrajectoryDatabase) -> list[str]:
    """Return optimizer names in database ID order."""
    conn = source_db._get_connection()
    rows = conn.execute("SELECT name FROM optimizers ORDER BY id").fetchall()
    return [str(row["name"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect conservative online rollouts")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint directory")
    parser.add_argument("--database", type=Path, default=None, help="Source trajectory database")
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("data/trajectories_online.db"),
        help="Target database for collected online trajectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--circuits",
        type=str,
        choices=["all", "original", "artifact"],
        default="original",
        help="Which circuit kind to roll out on",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of circuits to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for circuit order and exploration")
    parser.add_argument("--device", type=str, default="auto", help="Evaluation device")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum rollout steps per circuit")
    parser.add_argument("--time-budget", type=float, default=None, help="Per-circuit time budget in seconds")
    parser.add_argument(
        "--exploration-rate", type=float, default=None,
        help="Probability of exploring uncertain states",
    )
    parser.add_argument(
        "--uncertainty-threshold", type=float, default=None,
        help="Minimum uncertainty required before exploration",
    )
    parser.add_argument(
        "--mc-dropout-passes", type=int, default=4,
        help="Number of stochastic forward passes for uncertainty estimation",
    )
    parser.add_argument(
        "--degradation-threshold", type=float, default=None,
        help="Stop a rollout early if one step degrades 2Q count beyond this fraction",
    )
    parser.add_argument("--save-intermediates", action="store_true", help="Persist intermediate QASM files")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/online_rollout_artifacts"),
        help="Directory for optional rollout artifacts",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    trainer_state, config, norm_stats, metadata = load_checkpoint(args.checkpoint, device="cpu")
    if args.device != "auto":
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    db_path = Path(args.database) if args.database else Path(config.database_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    trainer = create_trainer(config, device_str)
    trainer.load_state_dict(trainer_state)

    source_db = TrajectoryDatabase(db_path)
    target_db = TrajectoryDatabase(args.output_db)
    action_names = _ordered_action_names(source_db)

    circuits = _filter_circuits(source_db.list_circuits(), args.circuits)
    circuits = [c for c in circuits if c.qasm_path]
    rng = random.Random(args.seed)
    rng.shuffle(circuits)
    if args.limit is not None:
        circuits = circuits[:args.limit]

    if not circuits:
        print(f"No circuits found for kind={args.circuits}")
        sys.exit(1)

    max_steps = args.max_steps if args.max_steps is not None else config.rollout_max_steps
    time_budget = args.time_budget if args.time_budget is not None else config.rollout_time_budget
    exploration_rate = args.exploration_rate if args.exploration_rate is not None else config.exploration_rate
    uncertainty_threshold = (
        args.uncertainty_threshold if args.uncertainty_threshold is not None else config.uncertainty_threshold
    )
    degradation_threshold = (
        args.degradation_threshold if args.degradation_threshold is not None else config.degradation_threshold
    )

    print(f"Collecting rollouts on {len(circuits)} circuits ({args.circuits})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source DB:   {db_path}")
    print(f"Target DB:   {args.output_db}")
    print(
        f"Settings: steps={max_steps}, time_budget={time_budget}, "
        f"exploration={exploration_rate}, uncertainty={uncertainty_threshold}"
    )

    per_circuit = []
    inserted = 0
    for circuit in circuits:
        rollout = rollout_policy(
            trainer=trainer,
            circuit_record=circuit,
            action_names=action_names,
            norm_stats=norm_stats,
            max_steps=max_steps,
            time_budget=time_budget,
            degradation_threshold=degradation_threshold,
            exploration_rate=exploration_rate,
            uncertainty_threshold=uncertainty_threshold,
            mc_dropout_passes=args.mc_dropout_passes,
            output_root=args.artifacts_dir,
            save_intermediates=args.save_intermediates,
            rng=rng,
        )
        per_circuit.append(rollout)

        if rollout.get("success") and rollout.get("num_steps", 0) > 0:
            trajectory_id = record_rollout(
                target_db,
                circuit,
                rollout,
                action_names=action_names,
                metadata={
                    "checkpoint": str(args.checkpoint),
                    "seed": args.seed,
                    "base_training_metadata": metadata,
                },
            )
            if trajectory_id > 0:
                inserted += 1

    source_db.close()
    target_db.close()

    summary = summarize_rollouts(per_circuit)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "source_database": str(db_path),
            "output_database": str(args.output_db),
            "rollouts_inserted": inserted,
            "circuit_kind": args.circuits,
            "settings": {
                "max_steps": max_steps,
                "time_budget": time_budget,
                "exploration_rate": exploration_rate,
                "uncertainty_threshold": uncertainty_threshold,
                "mc_dropout_passes": args.mc_dropout_passes,
                "degradation_threshold": degradation_threshold,
            },
        }
    )

    print(f"Inserted {inserted} new trajectories")
    print(
        f"Executed {summary['num_executed']}/{summary['num_circuits']} rollouts, "
        f"mean 2Q improvement {summary['mean_2q_improvement'] * 100:.1f}%"
    )

    if args.output is not None:
        from benchmarks.ai_transpile.rl_training.evaluation import save_evaluation_results

        save_evaluation_results(summary, args.output)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
