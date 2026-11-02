"""Factory helpers for offline RL trainers."""

from __future__ import annotations

from .config import TrainingConfig


def create_trainer(config: TrainingConfig, device_str: str | None = None):  # noqa: ANN202
    """Create a trainer instance from a training config."""
    from .algorithms.behavioral_cloning import BehavioralCloning
    from .algorithms.cql import CQL
    from .algorithms.decision_transformer import DecisionTransformer
    from .algorithms.iql import IQL

    trainers = {
        "bc": BehavioralCloning,
        "cql": CQL,
        "iql": IQL,
        "dt": DecisionTransformer,
    }
    trainer_cls = trainers.get(config.algorithm)
    if trainer_cls is None:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    if device_str is not None:
        config.device = device_str
    return trainer_cls(config)
