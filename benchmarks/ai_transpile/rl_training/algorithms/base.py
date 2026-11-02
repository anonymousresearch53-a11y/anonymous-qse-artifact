"""Abstract base class for offline RL training algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import TrainingConfig


class OfflineTrainer(ABC):
    """Base class for offline RL trainers.

    Subclasses must implement:
    - _build_networks(): Initialize networks and optimizers
    - train_step(batch): Perform one gradient step, return metrics dict
    - select_action(state): Pick an action given a state tensor

    The training loop, logging, and checkpointing are handled by the base class.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.resolve_device())
        self._step = 0
        self._epoch = 0
        self._writer = None
        self._build_networks()

    @abstractmethod
    def _build_networks(self) -> None:
        """Initialize neural networks and optimizers. Called by __init__."""

    @abstractmethod
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Dictionary with keys: observations, actions, rewards,
                   next_observations, terminals (all tensors on device)

        Returns:
            Dictionary of metric names to scalar values for logging
        """

    @abstractmethod
    def select_action(self, state: torch.Tensor) -> int:
        """Select an action given a (normalized) state.

        Args:
            state: State tensor of shape (state_dim,) on device

        Returns:
            Action index (0-indexed)
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a checkpoint state dict."""

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Override config.num_epochs if provided

        Returns:
            Dictionary mapping metric names to per-epoch values
        """
        epochs = num_epochs or self.config.num_epochs
        history: dict[str, list[float]] = {}
        writer = self._get_writer()

        for epoch in range(epochs):
            self._epoch = epoch
            epoch_metrics = self._train_epoch(train_loader)

            # Record train metrics
            for key, val in epoch_metrics.items():
                history.setdefault(f"train/{key}", []).append(val)
                if writer is not None:
                    writer.add_scalar(f"train/{key}", val, epoch)

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate(val_loader)
                for key, val in val_metrics.items():
                    history.setdefault(f"val/{key}", []).append(val)
                    if writer is not None:
                        writer.add_scalar(f"val/{key}", val, epoch)

        if writer is not None:
            writer.flush()

        return history

    def _train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch, returning average metrics."""
        self._set_train_mode()
        epoch_metrics: dict[str, list[float]] = {}

        for batch in train_loader:
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            metrics = self.train_step(batch_device)
            self._step += 1

            for key, val in metrics.items():
                epoch_metrics.setdefault(key, []).append(val)

        # Average over batches
        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation, returning average metrics."""
        self._set_eval_mode()
        val_metrics: dict[str, list[float]] = {}

        for batch in val_loader:
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            # Compute action accuracy
            obs = batch_device["observations"]
            true_actions = batch_device["actions"]

            # Get predicted actions
            predicted = torch.tensor(
                [self.select_action(obs[i]) for i in range(obs.shape[0])],
                device=self.device,
            )
            accuracy = (predicted == true_actions).float().mean().item()
            val_metrics.setdefault("action_accuracy", []).append(accuracy)

        self._set_train_mode()
        return {k: sum(v) / len(v) for k, v in val_metrics.items()}

    def _set_train_mode(self) -> None:
        """Set all networks to training mode. Override if needed."""

    def _set_eval_mode(self) -> None:
        """Set all networks to eval mode. Override if needed."""

    def _get_writer(self) -> Any:
        """Get TensorBoard writer, creating if needed."""
        if self._writer is not None:
            return self._writer

        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(self.config.output_dir) / "logs" / f"{self.config.algorithm}_{int(time.time())}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(log_dir))
        except ImportError:
            self._writer = None

        return self._writer
