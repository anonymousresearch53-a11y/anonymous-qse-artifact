"""Decision Transformer for offline RL on quantum circuit optimization.

Implements Decision Transformer (Chen et al., NeurIPS 2021) conditioned on
return-to-go (RTG).  Given a desired future return, the model produces a
sequence of actions to achieve it.

Architecture (small — matches available ~32k transition dataset):
    Input per step: [RTG, state (state_dim), action (one-hot action_dim)]
    Positional encoding: step index (0..max_ep_len-1)
    GPT-style causal Transformer: n_layers × n_heads, d_model
    Output: action logits for the *current* step (given RTG and state)

Sequence layout per episode (length T ≤ max_ep_len):
    tokens = [R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{T-1}, s_{T-1}, a_{T-1}]

At training time we predict a_t from (R_t, s_t); at inference time we pass
the desired return as R_0 and let the model generate actions autoregressively.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OfflineTrainer


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) / self.scale
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN with residual connections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecisionTransformerNetwork(nn.Module):
    """Decision Transformer network.

    Embeds (RTG, state, action) triples into a common d_model space,
    processes with a causal Transformer, and predicts action logits
    from each state token.
    """

    def __init__(
        self,
        state_dim: int = 26,
        action_dim: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_ep_len: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_ep_len = max_ep_len

        # Token embeddings
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Embedding(action_dim, d_model)

        # Positional embedding over episode steps (0..max_ep_len-1)
        self.pos_embed = nn.Embedding(max_ep_len, d_model)
        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Action prediction head (applied to state tokens)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        rtgs: torch.Tensor,        # (B, T, 1)
        states: torch.Tensor,      # (B, T, state_dim)
        actions: torch.Tensor,     # (B, T) — previous actions (last token predicted)
        timesteps: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            rtgs: Return-to-go values, shape (B, T, 1)
            states: Normalized state vectors, shape (B, T, state_dim)
            actions: Action indices, shape (B, T)
            timesteps: Step indices, shape (B, T)

        Returns:
            Action logits for each state token, shape (B, T, action_dim)
        """
        B, T, _ = rtgs.shape
        pos = self.pos_embed(timesteps)  # (B, T, d_model)

        # Embed each modality and add positional info
        rtg_emb = self.embed_ln(self.embed_rtg(rtgs) + pos)         # (B, T, d_model)
        state_emb = self.embed_ln(self.embed_state(states) + pos)    # (B, T, d_model)
        action_emb = self.embed_ln(self.embed_action(actions) + pos) # (B, T, d_model)

        # Interleave: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        # Each step contributes 3 tokens in order: RTG, state, action
        seq = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, T, 3, d_model)
        seq = seq.reshape(B, 3 * T, self.d_model)                    # (B, 3T, d_model)
        seq = self.embed_dropout(seq)

        # Transformer
        for block in self.blocks:
            seq = block(seq)
        seq = self.ln_f(seq)

        # Extract state tokens (indices 1, 4, 7, ... → position 3t+1)
        state_indices = torch.arange(1, 3 * T, 3, device=seq.device)
        state_tokens = seq[:, state_indices, :]  # (B, T, d_model)

        return self.action_head(state_tokens)  # (B, T, action_dim)

class DecisionTransformer(OfflineTrainer):
    """Decision Transformer trainer.

    Trains the DT network to predict actions conditioned on RTG and state
    using cross-entropy loss over all non-padded steps in each episode.

    Expects batches with keys: rtgs, states, actions, timesteps, masks.
    Use DTOfflineDataset.from_database() to produce these batches.
    """

    network: DecisionTransformerNetwork
    optimizer: torch.optim.Optimizer

    def _build_networks(self) -> None:
        d_model = getattr(self.config, "d_model", 64)
        n_heads = getattr(self.config, "n_heads", 4)
        n_layers = getattr(self.config, "n_layers", 2)
        max_ep_len = getattr(self.config, "max_ep_len", 3)

        self.network = DecisionTransformerNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_ep_len=max_ep_len,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train one step on an episode batch.

        Expects batch keys: rtgs (B,T,1), states (B,T,state_dim),
        actions (B,T), timesteps (B,T), masks (B,T).
        """
        rtgs = batch["rtgs"].to(self.device)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        masks = batch["masks"].to(self.device)

        logits = self.network(rtgs, states, actions, timesteps)  # (B, T, action_dim)

        B, T, A = logits.shape
        # Flatten and apply mask
        logits_flat = logits.reshape(B * T, A)
        actions_flat = actions.reshape(B * T)
        masks_flat = masks.reshape(B * T)

        # Only compute loss on non-padded steps
        valid = masks_flat
        if valid.sum() == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        loss = F.cross_entropy(logits_flat[valid], actions_flat[valid])

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            predicted = logits_flat[valid].argmax(dim=-1)
            accuracy = (predicted == actions_flat[valid]).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> int:
        """Select an action given a single normalized state.

        For compatibility with the base class evaluation loop (which passes
        individual state vectors), this creates a single-step episode with
        the configured target_return and predicts the action.
        """
        self.network.eval()
        target_return = getattr(self.config, "target_return", 0.3)

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        state_seq = state.unsqueeze(1)  # (1, 1, state_dim)
        rtg_seq = torch.tensor([[[target_return]]], dtype=torch.float32, device=self.device)
        action_seq = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        timestep_seq = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        logits = self.network(rtg_seq, state_seq, action_seq, timestep_seq)
        action = int(logits[0, 0].argmax().item())
        self.network.train()
        return action

    def state_dict(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state.get("step", 0)
        self._epoch = state.get("epoch", 0)

    @torch.no_grad()
    def _validate(self, val_loader) -> dict[str, float]:  # type: ignore[override]
        """Validate using DT batch keys (states/actions/masks) instead of observations."""
        self.network.eval()
        all_accuracies: list[float] = []

        for batch in val_loader:
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            rtgs = batch_device["rtgs"]
            states = batch_device["states"]
            actions = batch_device["actions"]
            timesteps = batch_device["timesteps"]
            masks = batch_device["masks"]

            logits = self.network(rtgs, states, actions, timesteps)  # (B, T, A)
            B, T, A = logits.shape
            logits_flat = logits.reshape(B * T, A)
            actions_flat = actions.reshape(B * T)
            masks_flat = masks.reshape(B * T)

            valid = masks_flat
            if valid.sum() > 0:
                predicted = logits_flat[valid].argmax(dim=-1)
                acc = (predicted == actions_flat[valid]).float().mean().item()
                all_accuracies.append(acc)

        self.network.train()
        mean_acc = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        return {"action_accuracy": mean_acc}

    def _set_train_mode(self) -> None:
        self.network.train()

    def _set_eval_mode(self) -> None:
        self.network.eval()
