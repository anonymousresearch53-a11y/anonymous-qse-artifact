"""Implicit Q-Learning (IQL) for discrete actions.

IQL avoids querying out-of-distribution actions entirely by:
1. Learning V(s) with expectile regression against Q(s,a) for data actions
2. Learning Q(s,a) with standard Bellman backup using V(s') (not max_a Q)
3. Extracting policy via advantage-weighted regression

This makes IQL particularly well-suited for small offline datasets where
OOD actions are a serious concern.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit
Q-Learning", ICLR 2022.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import PolicyNetwork, QNetwork, ValueNetwork
from .base import OfflineTrainer


def _expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 loss for expectile regression.

    When tau > 0.5, the loss penalizes under-predictions more heavily,
    causing V(s) to track the upper expectile of Q(s,a).
    """
    diff = target - pred
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


class IQL(OfflineTrainer):
    """Implicit Q-Learning trainer for discrete actions."""

    q_network: QNetwork
    target_q_network: QNetwork
    value_network: ValueNetwork
    policy: PolicyNetwork
    q_optimizer: torch.optim.Optimizer
    v_optimizer: torch.optim.Optimizer
    policy_optimizer: torch.optim.Optimizer

    def _build_networks(self) -> None:
        self.q_network = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.target_q_network = copy.deepcopy(self.q_network)
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        self.value_network = ValueNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.policy = PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.v_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]

        # --- Value function update (expectile regression) ---
        with torch.no_grad():
            target_q = self.target_q_network(states)
            target_q_a = target_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        v = self.value_network(states).squeeze(1)
        v_loss = _expectile_loss(v, target_q_a, self.config.iql_tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.grad_clip)
        self.v_optimizer.step()

        # --- Q-function update (Bellman backup with V, not max Q) ---
        with torch.no_grad():
            next_v = self.value_network(next_states).squeeze(1)
            q_target = rewards + self.config.gamma * (1 - terminals) * next_v

        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_a, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.q_optimizer.step()

        # --- Policy update (advantage-weighted regression) ---
        with torch.no_grad():
            # Advantage = Q(s,a) - V(s) using target Q
            advantage = target_q_a - v.detach()
            # Exp-normalize advantages with temperature
            weights = torch.exp(self.config.iql_beta * advantage)
            weights = weights.clamp(max=100.0)  # Prevent overflow

        log_probs = self.policy.get_log_probs(states)
        policy_log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(weights * policy_log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.policy_optimizer.step()

        # Soft update target Q
        self._soft_update_target()

        # Metrics
        with torch.no_grad():
            predicted = self.policy(states).argmax(dim=-1)
            accuracy = (predicted == actions).float().mean().item()

        return {
            "loss": (v_loss + q_loss + policy_loss).item(),
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_q": q_a.mean().item(),
            "mean_v": v.mean().item(),
            "mean_advantage": advantage.mean().item(),
            "accuracy": accuracy,
        }

    def _soft_update_target(self) -> None:
        tau = self.config.cql_target_update_rate  # Reuse same parameter
        for tp, sp in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

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
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "policy": self.policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q_network.load_state_dict(state["q_network"])
        self.target_q_network.load_state_dict(state["target_q_network"])
        self.value_network.load_state_dict(state["value_network"])
        self.policy.load_state_dict(state["policy"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self._step = state.get("step", 0)
        self._epoch = state.get("epoch", 0)

    def _set_train_mode(self) -> None:
        self.q_network.train()
        self.value_network.train()
        self.policy.train()

    def _set_eval_mode(self) -> None:
        self.q_network.eval()
        self.value_network.eval()
        self.policy.eval()
