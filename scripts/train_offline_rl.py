#!/usr/bin/env python3
"""Main training CLI for offline RL on quantum circuit optimization data.

Usage:
    python scripts/train_offline_rl.py --config configs/bc_default.yaml
    python scripts/train_offline_rl.py --algorithm bc --num-epochs 50 --seed 42
    python scripts/train_offline_rl.py --config configs/cql_default.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from benchmarks.ai_transpile.rl_training.checkpointing import save_checkpoint  # noqa: E402
from benchmarks.ai_transpile.rl_training.config import TrainingConfig  # noqa: E402
from benchmarks.ai_transpile.rl_training.dataset import (  # noqa: E402
    DTOfflineDataset,
    OfflineRLDataset,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train offline RL policy for quantum circuit optimization"
    )

    # Config file (overrides all other args)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )

    # Override individual settings
    parser.add_argument("--algorithm", type=str, choices=["bc", "cql", "iql", "dt"])
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reward-type", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dims", nargs="+", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--cql-alpha", type=float)
    parser.add_argument("--iql-tau", type=float)
    parser.add_argument("--iql-beta", type=float)
    parser.add_argument("--no-eval", action="store_true", help="Skip final evaluation")

    args = parser.parse_args()

    # Load config from file or defaults
    if args.config is not None:
        config = TrainingConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig()

    # Apply CLI overrides
    overrides = {
        "algorithm": args.algorithm,
        "database_path": str(args.database) if args.database else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "reward_type": args.reward_type,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": args.hidden_dims,
        "seed": args.seed,
        "device": args.device,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "cql_alpha": args.cql_alpha,
        "iql_tau": args.iql_tau,
        "iql_beta": args.iql_beta,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(config, key, val)

    # Set seed
    torch.manual_seed(config.seed)

    # Print config
    device_str = config.resolve_device()
    print(f"Algorithm:    {config.algorithm}")
    print(f"Database:     {config.database_path}")
    print(f"Reward type:  {config.reward_type}")
    print(f"Device:       {device_str}")
    print(f"Hidden dims:  {config.hidden_dims}")
    print(f"Batch size:   {config.batch_size}")
    print(f"Epochs:       {config.num_epochs}")
    print(f"LR:           {config.learning_rate}")
    print(f"Seed:         {config.seed}")
    print()

    # Load data
    print("Loading dataset...")
    db_path = Path(config.database_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    if config.algorithm == "dt":
        # Decision Transformer uses episode-grouped dataset
        full_dataset = DTOfflineDataset.from_database(db_path, config, max_ep_len=config.max_ep_len)
        print(f"Loaded {len(full_dataset)} episodes (max_ep_len={config.max_ep_len})")
        print(f"Action names: {full_dataset.action_names}")

        actual_action_dim = len(full_dataset.action_names)
        if actual_action_dim != config.action_dim:
            print(f"Updating action_dim from {config.action_dim} to {actual_action_dim}")
            config.action_dim = actual_action_dim

        # Simple random split for DT (episodes are independent)
        rng = __import__("numpy").random.RandomState(config.seed)
        n = len(full_dataset)
        indices = rng.permutation(n)
        n_val = max(1, int(n * config.val_fraction))
        n_test = max(1, int(n * config.test_fraction))
        val_idx = indices[:n_val].tolist()
        test_idx = indices[n_val:n_val + n_test].tolist()
        train_idx = indices[n_val + n_test:].tolist()
        from torch.utils.data import Subset
        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)
        test_ds = Subset(full_dataset, test_idx)
        print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    else:
        full_dataset = OfflineRLDataset.from_database(db_path, config)
        print(f"Loaded {len(full_dataset)} transitions, {config.action_dim} actions")
        print(f"Action names: {full_dataset.action_names}")

        actual_action_dim = len(full_dataset.action_names)
        if actual_action_dim != config.action_dim:
            print(f"Updating action_dim from {config.action_dim} to {actual_action_dim}")
            config.action_dim = actual_action_dim

        train_ds, val_ds, test_ds = split_dataset(full_dataset, db_path, config)
        print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = make_dataloader(train_ds, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_dataloader(val_ds, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Create trainer
    print(f"\nCreating {config.algorithm.upper()} trainer...")
    trainer = create_trainer(config)

    # Train
    print(f"Training for {config.num_epochs} epochs...\n")
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, config.num_epochs)
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed:.1f}s")

    # Print final metrics
    if "train/loss" in history:
        final_loss = history["train/loss"][-1]
        print(f"Final train loss: {final_loss:.4f}")
    if "train/accuracy" in history:
        final_acc = history["train/accuracy"][-1]
        print(f"Final train accuracy: {final_acc:.4f}")
    if "val/action_accuracy" in history:
        final_val_acc = history["val/action_accuracy"][-1]
        print(f"Final val accuracy: {final_val_acc:.4f}")

    # Save checkpoint
    timestamp = int(time.time())
    checkpoint_name = f"{config.algorithm}_{config.reward_type}_{timestamp}"
    checkpoint_dir = Path(config.output_dir) / checkpoint_name

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer.state_dict(),
        config=config,
        norm_stats=full_dataset.norm_stats,
        metadata={
            "elapsed_seconds": elapsed,
            "num_epochs": config.num_epochs,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "final_metrics": {k: v[-1] for k, v in history.items() if v},
        },
    )
    print(f"\nCheckpoint saved to {checkpoint_dir}")

    # Final evaluation (skipped for DT since test_ds is a Subset)
    if not args.no_eval and config.algorithm != "dt":
        assert isinstance(test_ds, OfflineRLDataset)
        print("\nEvaluating on test set...")
        test_metrics = evaluate_policy(trainer, test_ds)
        baselines = compute_baselines(test_ds)

        # Oracle evaluation (requires circuit_ids)
        oracle_metrics = None
        if hasattr(test_ds, "circuit_ids") and test_ds.circuit_ids is not None:
            oracle_metrics = evaluate_best_action_oracle(trainer, test_ds)

        table = generate_comparison_table(test_metrics, baselines, full_dataset.action_names, oracle_metrics)
        print(table)

    print("\nDone.")


if __name__ == "__main__":
    main()
