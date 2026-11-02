"""Model checkpointing: save/load models with normalization stats and config.

A checkpoint is a directory containing:
- model.pt: Algorithm state dict (networks, optimizers, etc.)
- config.yaml: Training configuration
- normalization.json: Feature normalization statistics
- metadata.json: Training metadata (epoch, step, metrics)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import TrainingConfig
from .normalization import NormalizationStats


def save_checkpoint(
    checkpoint_dir: Path | str,
    trainer_state: dict[str, Any],
    config: TrainingConfig,
    norm_stats: NormalizationStats,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a complete training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint files
        trainer_state: State dict from trainer.state_dict()
        config: Training configuration
        norm_stats: Normalization statistics
        metadata: Optional training metadata (epoch, metrics, etc.)

    Returns:
        Path to the checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(trainer_state, checkpoint_dir / "model.pt")

    # Save config
    config.to_yaml(checkpoint_dir / "config.yaml")

    # Save normalization stats
    norm_stats.save(checkpoint_dir / "normalization.json")

    # Save metadata
    if metadata is not None:
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    _update_latest_symlinks(checkpoint_dir, config.algorithm)

    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path | str,
    device: str = "cpu",
) -> tuple[dict[str, Any], TrainingConfig, NormalizationStats, dict[str, Any]]:
    """Load a complete training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load tensors onto

    Returns:
        (trainer_state, config, norm_stats, metadata) tuple
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load model state
    trainer_state = torch.load(
        checkpoint_dir / "model.pt",
        map_location=device,
        weights_only=True,
    )

    # Load config
    config = TrainingConfig.from_yaml(checkpoint_dir / "config.yaml")

    # Load normalization stats
    norm_stats = NormalizationStats.load(checkpoint_dir / "normalization.json")

    # Load metadata
    metadata: dict[str, Any] = {}
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return trainer_state, config, norm_stats, metadata


def _update_latest_symlinks(checkpoint_dir: Path, algorithm: str) -> None:
    """Update convenience symlinks for the most recent checkpoint."""
    parent = checkpoint_dir.parent
    symlink_names = ["latest", f"{algorithm}_latest"]

    for symlink_name in symlink_names:
        symlink_path = parent / symlink_name
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(checkpoint_dir.name)
        except OSError:
            # Symlink creation is a convenience only; ignore unsupported filesystems.
            continue
