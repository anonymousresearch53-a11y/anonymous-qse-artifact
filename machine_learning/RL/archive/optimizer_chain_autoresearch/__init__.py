"""Autonomous research tools for optimizer-chain reinforcement learning.

The package is intentionally self-contained.  It uses the author's existing
optimizer-chain dataset and does not import the separate RL framework elsewhere
in this repository.
"""

from .data_setup import QUALITY_METRICS, DataBundle, ensure_data_setup

__all__ = ["QUALITY_METRICS", "DataBundle", "ensure_data_setup"]
