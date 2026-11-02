"""Training configuration for offline RL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Configuration for offline RL training.

    Attributes:
        algorithm: Training algorithm ("bc", "cql", "iql")
        reward_type: Which reward column to use from the database
        database_path: Path to the trajectory database
        output_dir: Directory for checkpoints and logs

        state_dim: Dimension of the state vector
        action_dim: Number of discrete actions (optimizers)
        hidden_dims: Hidden layer sizes for MLP networks

        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        weight_decay: L2 regularization coefficient
        dropout: Dropout rate for networks
        grad_clip: Maximum gradient norm (0 to disable)

        gamma: Discount factor for RL algorithms
        cql_alpha: Conservative weight for CQL
        cql_target_update_rate: Soft update rate for CQL target network
        iql_tau: Expectile parameter for IQL value function
        iql_beta: Temperature for IQL advantage-weighted regression

        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        split_by_circuit: Split by circuit ID to prevent data leakage

        seed: Random seed for reproducibility
        device: Device string ("auto", "cpu", "cuda", "cuda:0", etc.)
        log_interval: Steps between logging
        eval_interval: Epochs between evaluation
        num_workers: DataLoader workers
    """

    # Algorithm
    algorithm: str = "bc"
    reward_type: str = "reward_improvement_only"
    database_path: str = "data/trajectories_step2.db"
    output_dir: str = "data/rl_checkpoints"

    # Architecture
    state_dim: int = 26
    action_dim: int = 5
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])

    # Optimization
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    weight_decay: float = 1e-4
    dropout: float = 0.1
    grad_clip: float = 1.0

    # RL hyperparameters
    gamma: float = 0.99
    cql_alpha: float = 1.0
    cql_target_update_rate: float = 0.005
    iql_tau: float = 0.7
    iql_beta: float = 3.0

    # Data splits
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_by_circuit: bool = True

    # Reward clipping
    reward_clip: float | None = None

    # Runtime
    seed: int = 42
    device: str = "auto"
    log_interval: int = 10
    eval_interval: int = 5
    num_workers: int = 0

    # Conservative online pilot
    rollout_time_budget: float = 300.0
    rollout_max_steps: int = 3
    exploration_rate: float = 0.05
    uncertainty_threshold: float = 0.1
    degradation_threshold: float = 0.25
    online_mix_weight: int = 4
    eval_circuit_kind: str = "all"


    # Decision Transformer specific (unused by other algorithms)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_ep_len: int = 3
    target_return: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
        }

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Create from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
