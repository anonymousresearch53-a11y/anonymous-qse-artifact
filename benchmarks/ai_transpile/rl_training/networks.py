"""Neural network architectures for offline RL.

Three MLP variants sharing the same backbone structure:
- QNetwork: Outputs Q-values per action (for CQL/IQL)
- ValueNetwork: Outputs scalar V(s) (for IQL)
- PolicyNetwork: Outputs action logits (for BC/IQL)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build an MLP with ReLU activations and optional dropout."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-value network: maps states to Q-values for each action.

    Output shape: (batch_size, action_dim)
    """

    def __init__(
        self,
        state_dim: int = 22,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim, action_dim, hidden_dims, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            states: (batch_size, state_dim)

        Returns:
            Q-values: (batch_size, action_dim)
        """
        return self.net(states)


class ValueNetwork(nn.Module):
    """State value network: maps states to scalar V(s).

    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        state_dim: int = 22,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim, 1, hidden_dims, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            states: (batch_size, state_dim)

        Returns:
            Values: (batch_size, 1)
        """
        return self.net(states)


class PolicyNetwork(nn.Module):
    """Policy network: maps states to action logits.

    Output shape: (batch_size, action_dim) — unnormalized log-probabilities
    """

    def __init__(
        self,
        state_dim: int = 22,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim, action_dim, hidden_dims, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            states: (batch_size, state_dim)

        Returns:
            Logits: (batch_size, action_dim)
        """
        return self.net(states)

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        """Select greedy actions.

        Args:
            states: (batch_size, state_dim)

        Returns:
            Actions: (batch_size,)
        """
        logits = self.forward(states)
        return logits.argmax(dim=-1)

    def get_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities over actions.

        Args:
            states: (batch_size, state_dim)

        Returns:
            Log-probs: (batch_size, action_dim)
        """
        logits = self.forward(states)
        return torch.log_softmax(logits, dim=-1)
