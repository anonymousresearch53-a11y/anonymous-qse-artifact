#!/usr/bin/env python3
"""Evaluate a trained offline RL policy against baselines.

Usage:
    python scripts/evaluate_policy.py --checkpoint data/rl_checkpoints/bc_reward_improvement_only_12345/
    python scripts/evaluate_policy.py --checkpoint data/rl_checkpoints/cql_*/ --output results.json

    # Online evaluation (runs live optimizer calls):
    python scripts/evaluate_policy.py --checkpoint data/rl_checkpoints/cql_*/ --online
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from benchmarks.ai_transpile.rl_training.checkpointing import load_checkpoint  # noqa: E402
from benchmarks.ai_transpile.rl_training.dataset import (  # noqa: E402
    OfflineRLDataset,
    filter_dataset_by_circuit_kind,
    get_circuit_metadata,
    split_dataset,
)
from benchmarks.ai_transpile.rl_training.evaluation import (  # noqa: E402
    compute_baselines,
    evaluate_best_action_oracle,
    evaluate_policy,
    generate_comparison_table,
    save_evaluation_results,
)
from benchmarks.ai_transpile.rl_training.factory import create_trainer  # noqa: E402
from benchmarks.ai_transpile.rl_training.online import (  # noqa: E402
    rollout_policy,
    summarize_rollouts,
)


def _run_online_evaluation(
    trainer,  # noqa: ANN001
    eval_ds,  # noqa: ANN001
    db_path: Path,
    action_names: list[str],
    norm_stats,  # noqa: ANN001
    rollout_steps: int = 1,
    time_budget: float = 300.0,
    degradation_threshold: float | None = None,
    max_circuits: int | None = None,
) -> dict:
    """Run live multi-step rollouts to measure actual 2Q gate reduction."""
    from benchmarks.ai_transpile.rl_trajectory.database import TrajectoryDatabase

    if not hasattr(eval_ds, "circuit_ids") or eval_ds.circuit_ids is None:
        print("Warning: circuit_ids not available; skipping online evaluation.")
        return {"error": "circuit_ids not available"}

    circuit_ids = eval_ds.circuit_ids.tolist()
    seen: set[int] = set()
    ordered_circuit_ids: list[int] = []
    for cid in circuit_ids:
        cid_int = int(cid)
        if cid_int not in seen:
            seen.add(cid_int)
            ordered_circuit_ids.append(cid_int)

    if max_circuits is not None:
        ordered_circuit_ids = ordered_circuit_ids[:max_circuits]

    print(
        f"\nOnline evaluation: running live rollouts on {len(ordered_circuit_ids)} circuits "
        f"(max_steps={rollout_steps})..."
    )

    db = TrajectoryDatabase(db_path)
    per_circuit: list[dict] = []
    for cid in ordered_circuit_ids:
        circuit_record = db.get_circuit_by_id(cid)
        if circuit_record is None or circuit_record.qasm_path is None:
            print(f"  Circuit {cid}: no QASM path, skipping")
            per_circuit.append({
                "circuit_name": circuit_record.name if circuit_record is not None else f"circuit_{cid}",
                "circuit_id": cid,
                "success": False,
                "error": "missing_qasm_path" if circuit_record is not None else "missing_circuit_record",
            })
            continue

        rollout = rollout_policy(
            trainer=trainer,
            circuit_record=circuit_record,
            action_names=action_names,
            norm_stats=norm_stats,
            max_steps=rollout_steps,
            time_budget=time_budget,
            degradation_threshold=degradation_threshold,
        )
        per_circuit.append(rollout)

        if rollout.get("success"):
            action_str = " -> ".join(rollout["optimizers"]) if rollout["optimizers"] else "(none)"
            print(
                f"  {circuit_record.name}: {action_str} → "
                f"{rollout['initial_2q']}→{rollout['final_2q']} 2Q "
                f"({rollout['improvement_pct']:.1f}% reduction)"
            )
        else:
            print(f"  {circuit_record.name}: {rollout.get('error', 'execution failed')}")

    db.close()
    return summarize_rollouts(per_circuit)


def _print_online_results(results: dict) -> None:
    """Print a formatted summary of online evaluation results."""
    print("\n" + "=" * 60)
    print("Online Evaluation Results (live optimizer calls)")
    print("=" * 60)
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    print(f"  Circuits evaluated:   {results['num_executed']}/{results['num_circuits']}")
    print(f"  Success rate:         {results['success_rate']:.1%}")
    print(f"  Mean 2Q improvement:  {results['mean_2q_improvement'] * 100:.1f}%")
    print(f"  Max  2Q improvement:  {results['max_2q_improvement'] * 100:.1f}%")
    print(f"  Mean steps used:      {results.get('mean_num_steps', 0.0):.2f}")
    for kind, metrics in results.get("by_circuit_kind", {}).items():
        print(
            f"  {kind:>8s}: executed={metrics['num_executed']:3d} "
            f"mean_2Q={metrics['mean_2q_improvement'] * 100:6.1f}% "
            f"success={metrics['success_rate']:.1%}"
        )
    print("=" * 60)


def _offline_metrics_for_dataset(trainer, dataset, action_names):  # noqa: ANN001, ANN202
    """Compute offline metrics bundle for a dataset subset."""
    policy_metrics = evaluate_policy(trainer, dataset)
    baselines = compute_baselines(dataset)
    oracle_metrics = None
    if hasattr(dataset, "circuit_ids") and dataset.circuit_ids is not None:
        oracle_metrics = evaluate_best_action_oracle(trainer, dataset)
    table = generate_comparison_table(policy_metrics, baselines, action_names, oracle_metrics)
    return {
        "policy_metrics": policy_metrics,
        "baselines": baselines,
        "oracle_metrics": oracle_metrics,
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained offline RL policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Override database path from config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "train", "all"],
        default="test",
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--circuits",
        type=str,
        choices=["all", "original", "artifact"],
        default="all",
        help="Restrict evaluation to original benchmark circuits or artifact circuits",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Run online evaluation: roll out the policy live via chain_executor "
            "and report actual 2Q gate reduction."
        ),
    )
    parser.add_argument(
        "--online-max-circuits",
        type=int,
        default=None,
        help="Limit online evaluation to N circuits (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="Maximum number of live actions to roll out per circuit",
    )
    parser.add_argument(
        "--report-by-circuit-kind",
        action="store_true",
        help="When evaluating all circuits, also report offline metrics split by original vs artifact circuits",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    trainer_state, config, norm_stats, metadata = load_checkpoint(
        args.checkpoint, device="cpu"
    )

    # Resolve device
    if args.device != "auto":
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Algorithm: {config.algorithm}")
    print(f"Device:    {device_str}")

    # Override database if specified
    db_path = Path(args.database) if args.database else Path(config.database_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Load dataset with checkpoint's normalization stats
    print(f"Loading dataset from {db_path}...")
    full_dataset = OfflineRLDataset.from_database(db_path, config, norm_stats=norm_stats)
    print(f"Loaded {len(full_dataset)} transitions")

    # Create trainer and load weights
    trainer = create_trainer(config, device_str)
    trainer.load_state_dict(trainer_state)

    # Select evaluation split
    if args.split == "all":
        eval_ds = full_dataset
    else:
        train_ds, val_ds, test_ds = split_dataset(full_dataset, db_path, config)
        split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
        eval_ds = split_map[args.split]

    if args.circuits != "all":
        eval_ds = filter_dataset_by_circuit_kind(eval_ds, db_path, args.circuits)

    if len(eval_ds) == 0:
        print(f"No samples available for split={args.split}, circuits={args.circuits}")
        sys.exit(1)

    print(f"Evaluating on {args.split} split ({len(eval_ds)} samples)...")
    if hasattr(eval_ds, "circuit_ids") and eval_ds.circuit_ids is not None:
        circuit_meta = get_circuit_metadata(
            db_path,
            {int(cid) for cid in eval_ds.circuit_ids.tolist()},
        )
        kind_counts = {"original": 0, "artifact": 0}
        for meta in circuit_meta.values():
            kind_counts[meta["kind"]] += 1
        print(
            f"Circuits in split: {len(circuit_meta)} "
            f"(original={kind_counts['original']}, artifact={kind_counts['artifact']})"
        )

    # Evaluate offline
    offline_results = _offline_metrics_for_dataset(trainer, eval_ds, full_dataset.action_names)
    print(offline_results["table"])

    by_kind_report = None
    if args.report_by_circuit_kind and args.circuits == "all":
        by_kind_report = {}
        for circuit_kind in ("original", "artifact"):
            subset = filter_dataset_by_circuit_kind(eval_ds, db_path, circuit_kind)
            if len(subset) == 0:
                continue
            bundle = _offline_metrics_for_dataset(trainer, subset, full_dataset.action_names)
            by_kind_report[circuit_kind] = {
                "policy_metrics": bundle["policy_metrics"],
                "baselines": bundle["baselines"],
                "oracle_metrics": bundle["oracle_metrics"],
            }
            print(f"\nOffline metrics for {circuit_kind} circuits:")
            print(bundle["table"])

    # Online evaluation
    online_results = None
    if args.online:
        online_results = _run_online_evaluation(
            trainer=trainer,
            eval_ds=eval_ds,
            db_path=db_path,
            action_names=full_dataset.action_names,
            norm_stats=norm_stats,
            rollout_steps=args.rollout_steps,
            time_budget=config.rollout_time_budget,
            degradation_threshold=config.degradation_threshold,
            max_circuits=args.online_max_circuits,
        )
        _print_online_results(online_results)

    # Save results
    if args.output:
        results = {
            "checkpoint": str(args.checkpoint),
            "algorithm": config.algorithm,
            "split": args.split,
            "circuit_kind": args.circuits,
            "policy_metrics": offline_results["policy_metrics"],
            "baselines": offline_results["baselines"],
            "oracle_metrics": offline_results["oracle_metrics"],
            "training_metadata": metadata,
        }
        if by_kind_report is not None:
            results["by_circuit_kind"] = by_kind_report
        if online_results is not None:
            results["online_evaluation"] = online_results
        save_evaluation_results(results, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
