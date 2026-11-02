"""Normalization statistics computation and persistence.

Computes mean/std from training data and saves/loads as JSON,
replacing the hardcoded defaults in rl_trajectory/state.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NormalizationStats:
    """Feature-wise normalization statistics.

    Attributes:
        means: Per-feature means, shape (state_dim,)
        stds: Per-feature standard deviations, shape (state_dim,)
        count: Number of samples used to compute statistics
    """

    means: np.ndarray
    stds: np.ndarray
    count: int

    def normalize(self, states: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            states: Raw state array, shape (N, state_dim) or (state_dim,)

        Returns:
            Normalized states with same shape
        """
        safe_stds = np.maximum(self.stds, 1e-8)
        return (states - self.means) / safe_stds

    def save(self, path: Path | str) -> None:
        """Save statistics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "count": self.count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> NormalizationStats:
        """Load statistics from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            means=np.array(data["means"], dtype=np.float32),
            stds=np.array(data["stds"], dtype=np.float32),
            count=data["count"],
        )


def compute_normalization_stats(observations: np.ndarray) -> NormalizationStats:
    """Compute normalization statistics from training observations.

    Args:
        observations: Array of shape (N, state_dim) containing raw state vectors

    Returns:
        NormalizationStats with means and stds computed from data
    """
    if observations.ndim != 2 or observations.shape[0] == 0:
        msg = f"Expected 2D array with at least 1 row, got shape {observations.shape}"
        raise ValueError(msg)

    means = observations.mean(axis=0).astype(np.float32)
    stds = observations.std(axis=0).astype(np.float32)

    return NormalizationStats(
        means=means,
        stds=stds,
        count=observations.shape[0],
    )
