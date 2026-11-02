#!/usr/bin/env python3
"""Fine-tune an offline RL checkpoint with collected online trajectories."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from benchmarks.ai_transpile.rl_training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from benchmarks.ai_transpile.rl_training.dataset import (  # noqa: E402
    OfflineRLDataset,
    concat_datasets,
    filter_dataset_by_circuit_kind,
    make_dataloader,
    split_dataset,
)
from benchmarks.ai_transpile.rl_training.evaluation import (  # noqa: E402
    compute_baselines,
    evaluate_best_action_oracle,
    evaluate_policy,
    generate_comparison_table,
)
from benchmarks.ai_transpile.rl_training.factory import create_trainer  # noqa: E402


def _align_action_order(
    dataset: OfflineRLDataset,
    reference_action_names: list[str],
) -> OfflineRLDataset:
    """Remap a dataset's action indices to match a reference action ordering."""
    if dataset.action_names == reference_action_names:
        return dataset

    index_by_name = {name: idx for idx, name in enumerate(reference_action_names)}
    remapped_actions = []
    for action_idx in dataset.actions.tolist():
        action_name = dataset.action_names[int(action_idx)]
        if action_name not in index_by_name:
            raise ValueError(f"Unknown action in online dataset: {action_name}")
        remapped_actions.append(index_by_name[action_name])

    return OfflineRLDataset(
        observations=dataset.observations.numpy(),
        actions=torch.tensor(remapped_actions, dtype=torch.long).numpy(),
        rewards=dataset.rewards.numpy(),
        next_observations=dataset.next_observations.numpy(),
        terminals=dataset.terminals.numpy(),
        norm_stats=dataset.norm_stats,
        action_map={i + 1: i for i in range(len(reference_action_names))},
        action_names=reference_action_names,
        circuit_ids=dataset.circuit_ids.numpy() if dataset.circuit_ids is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a checkpoint with online trajectories")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base checkpoint directory")
    parser.add_argument("--offline-db", type=Path, default=None, help="Offline training database")
    parser.add_argument("--online-db", type=Path, required=True, help="Online trajectory database")
    parser.add_argument("--output-dir", type=Path, default=None, help="Checkpoint output directory")
    parser.add_argument("--num-epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    parser.add_argument("--online-mix-weight", type=int, default=None, help="Oversampling multiplier for online data")
    parser.add_argument(
        "--eval-circuits",
        type=str,
        choices=["all", "original", "artifact"],
        default="original",
        help="Circuit kind to use for validation/test reporting",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not args.online_db.exists():
        print(f"Online DB not found: {args.online_db}")
        sys.exit(1)

    trainer_state, config, norm_stats, metadata = load_checkpoint(args.checkpoint, device="cpu")
    offline_db = Path(args.offline_db) if args.offline_db else Path(config.database_path)
    if not offline_db.exists():
        print(f"Offline DB not found: {offline_db}")
        sys.exit(1)

    if config.algorithm == "dt":
        print("Decision Transformer fine-tuning is not implemented in this mixed replay CLI yet.")
        sys.exit(1)

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.device != "auto":
        config.device = args.device

    mix_weight = args.online_mix_weight if args.online_mix_weight is not None else config.online_mix_weight
    device_str = config.resolve_device()

    print(f"Loading offline dataset from {offline_db}...")
    offline_full = OfflineRLDataset.from_database(offline_db, config, norm_stats=norm_stats)
    train_offline, val_offline, test_offline = split_dataset(offline_full, offline_db, config)

    print(f"Loading online dataset from {args.online_db}...")
    online_full = OfflineRLDataset.from_database(args.online_db, config, norm_stats=norm_stats)
    online_full = _align_action_order(online_full, offline_full.action_names)

    train_dataset = concat_datasets([train_offline, online_full], repeat_factors=[1, max(mix_weight, 1)])
    val_dataset = filter_dataset_by_circuit_kind(val_offline, offline_db, args.eval_circuits)
    test_dataset = filter_dataset_by_circuit_kind(test_offline, offline_db, args.eval_circuits)

    if len(val_dataset) == 0 or len(test_dataset) == 0:
        print(f"No validation/test data available for eval_circuits={args.eval_circuits}")
        sys.exit(1)

    train_loader = make_dataloader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_dataloader(val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)

    trainer = create_trainer(config, device_str)
    trainer.load_state_dict(trainer_state)

    print(
        f"Fine-tuning {config.algorithm} on mixed replay: "
        f"offline_train={len(train_offline)}, online={len(online_full)}, mix_weight={mix_weight}"
    )
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, args.num_epochs)
    elapsed = time.time() - start_time

    test_metrics = evaluate_policy(trainer, test_dataset)
    baselines = compute_baselines(test_dataset)
    oracle_metrics = evaluate_best_action_oracle(trainer, test_dataset)
    table = generate_comparison_table(test_metrics, baselines, offline_full.action_names, oracle_metrics)
    print(table)

    timestamp = int(time.time())
    checkpoint_name = f"{config.algorithm}_online_ft_{timestamp}"
    checkpoint_dir = Path(config.output_dir) / checkpoint_name
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer.state_dict(),
        config=config,
        norm_stats=norm_stats,
        metadata={
            "base_checkpoint": str(args.checkpoint),
            "offline_db": str(offline_db),
            "online_db": str(args.online_db),
            "online_mix_weight": mix_weight,
            "elapsed_seconds": elapsed,
            "num_epochs": args.num_epochs,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "eval_circuit_kind": args.eval_circuits,
            "initial_training_metadata": metadata,
            "final_metrics": {k: v[-1] for k, v in history.items() if v},
            "test_metrics": test_metrics,
            "oracle_metrics": oracle_metrics,
        },
    )
    print(f"Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
