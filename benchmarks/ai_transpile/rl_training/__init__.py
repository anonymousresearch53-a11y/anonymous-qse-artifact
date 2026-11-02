"""Offline RL Training Infrastructure for Quantum Circuit Optimization.

This package provides training algorithms that consume the trajectory database
(collected by rl_trajectory) to produce trained policies for optimizer selection.

Algorithms:
- Behavioral Cloning (BC): Supervised classification on expert actions
- Conservative Q-Learning (CQL): Offline RL with conservative regularization
- Implicit Q-Learning (IQL): Offline RL without OOD action queries

Example usage:

    from benchmarks.ai_transpile.rl_training import (
        TrainingConfig,
        OfflineRLDataset,
        BehavioralCloning,
    )

    config = TrainingConfig.from_yaml("configs/bc_default.yaml")
    dataset = OfflineRLDataset.from_database("data/trajectories.db", config)
    trainer = BehavioralCloning(config)
    trainer.train(dataset.train_loader(), dataset.val_loader(), num_epochs=100)
"""

from .algorithms.behavioral_cloning import BehavioralCloning
from .algorithms.cql import CQL
from .algorithms.iql import IQL
from .checkpointing import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .dataset import (
    OfflineRLDataset,
    concat_datasets,
    filter_dataset_by_circuit_kind,
    make_dataloader,
    split_dataset,
    subset_dataset,
)
from .evaluation import (
    compute_baselines,
    evaluate_policy,
    generate_comparison_table,
    save_evaluation_results,
)
from .factory import create_trainer
from .networks import PolicyNetwork, QNetwork, ValueNetwork
from .normalization import NormalizationStats, compute_normalization_stats
from .online import (
    record_rollout,
    rollout_policy,
    select_action_with_uncertainty,
    summarize_rollouts,
)

__all__ = [
    # Config
    "TrainingConfig",
    # Data
    "OfflineRLDataset",
    "subset_dataset",
    "concat_datasets",
    "filter_dataset_by_circuit_kind",
    "make_dataloader",
    "split_dataset",
    "NormalizationStats",
    "compute_normalization_stats",
    # Networks
    "QNetwork",
    "ValueNetwork",
    "PolicyNetwork",
    # Algorithms
    "BehavioralCloning",
    "CQL",
    "IQL",
    # Evaluation
    "evaluate_policy",
    "compute_baselines",
    "generate_comparison_table",
    "save_evaluation_results",
    "create_trainer",
    "rollout_policy",
    "select_action_with_uncertainty",
    "summarize_rollouts",
    "record_rollout",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
]
