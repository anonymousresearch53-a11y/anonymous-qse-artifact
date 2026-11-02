"""Conservative Q-Learning (CQL) for discrete actions.

CQL adds a conservative regularization term that penalizes Q-values for
out-of-distribution actions, making it well-suited for offline RL where
we cannot collect new data to correct overestimated Q-values.

Loss = TD_error + alpha * (logsumexp(Q(s, .)) - Q(s, a_data))

Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement
Learning", NeurIPS 2020.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import QNetwork
from .base import OfflineTrainer


class CQL(OfflineTrainer):
    """Conservative Q-Learning trainer for discrete actions."""

    q_network: QNetwork
    target_network: QNetwork
    optimizer: torch.optim.Optimizer

    def _build_networks(self) -> None:
        self.q_network = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.q_network)
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]

        # Current Q-values for taken actions
        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (double DQN style: use online net for action selection)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_a = next_q.gather(1, next_actions).squeeze(1)
            target = rewards + self.config.gamma * (1 - terminals) * next_q_a

        # TD loss
        td_loss = F.mse_loss(q_a, target)

        # CQL conservative regularization
        # Penalize high Q-values for all actions, reward Q-values for data actions
        logsumexp_q = torch.logsumexp(q_values, dim=1).mean()
        data_q = q_a.mean()
        cql_loss = logsumexp_q - data_q

        # Total loss
        loss = td_loss + self.config.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target()

        # Metrics
        with torch.no_grad():
            predicted = q_values.argmax(dim=-1)
            accuracy = (predicted == actions).float().mean().item()

        return {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "mean_q": q_a.mean().item(),
            "accuracy": accuracy,
        }

    def _soft_update_target(self) -> None:
        """Polyak averaging update of target network."""
        tau = self.config.cql_target_update_rate
        for tp, sp in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> int:
        self.q_network.eval()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        q_values = self.q_network(state)
        action = q_values.argmax(dim=-1).item()
        self.q_network.train()
        return int(action)

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state.get("step", 0)
        self._epoch = state.get("epoch", 0)

    def _set_train_mode(self) -> None:
        self.q_network.train()

    def _set_eval_mode(self) -> None:
        self.q_network.eval()
