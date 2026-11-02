"""Behavioral Cloning: supervised classification on expert actions.

Treats offline RL as a supervised learning problem: given state s,
predict the action a that was taken (by the exhaustive grid search).

This is the "imitation learning warmup" from the paper and provides
a strong baseline for the small-data regime (~1,500 transitions).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..networks import PolicyNetwork
from .base import OfflineTrainer


class BehavioralCloning(OfflineTrainer):
    """Behavioral Cloning trainer.

    Minimizes cross-entropy loss between the policy network's predicted
    action distribution and the actual actions from the dataset.
    """

    policy: PolicyNetwork
    optimizer: torch.optim.Optimizer

    def _build_networks(self) -> None:
        self.policy = PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._criterion = nn.CrossEntropyLoss()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]

        logits = self.policy(states)
        loss = self._criterion(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            accuracy = (predicted == actions).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> int:
        self.policy.eval()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.policy(state)
        action = logits.argmax(dim=-1).item()
        self.policy.train()
        return int(action)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state.get("step", 0)
        self._epoch = state.get("epoch", 0)

    def _set_train_mode(self) -> None:
        self.policy.train()

    def _set_eval_mode(self) -> None:
        self.policy.eval()
