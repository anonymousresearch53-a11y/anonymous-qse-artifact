"""Offline policy evaluation and baseline comparisons.

Provides metrics for evaluating trained policies without online rollouts:
- Action agreement with dataset
- Policy entropy (confidence)
- Average Q-values
- Comparison with baseline policies (random, greedy, best-single)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithms.base import OfflineTrainer
from .dataset import OfflineRLDataset


def evaluate_policy(
    trainer: OfflineTrainer,
    dataset: OfflineRLDataset,
) -> dict[str, float]:
    """Evaluate a trained policy on a dataset.

    Args:
        trainer: Trained offline RL trainer
        dataset: Dataset to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    device = trainer.device
    observations = dataset.observations.to(device)
    true_actions = dataset.actions.numpy()

    # Get predicted actions
    predicted_actions = []
    for i in range(len(dataset)):
        action = trainer.select_action(observations[i])
        predicted_actions.append(action)
    predicted_actions = np.array(predicted_actions)

    # Action agreement (accuracy)
    agreement = (predicted_actions == true_actions).mean()

    # Per-action accuracy
    action_dim = max(true_actions.max(), predicted_actions.max()) + 1
    per_action_acc = {}
    for a in range(action_dim):
        mask = true_actions == a
        if mask.sum() > 0:
            per_action_acc[f"accuracy_action_{a}"] = float((predicted_actions[mask] == a).mean())

    # Action distribution of policy
    pred_counts = Counter(predicted_actions.tolist())
    total = len(predicted_actions)
    pred_dist = {f"policy_freq_action_{a}": pred_counts.get(a, 0) / total for a in range(action_dim)}

    # Policy entropy (via softmax if policy network available)
    entropy = _compute_policy_entropy(trainer, observations)

    metrics = {
        "action_agreement": float(agreement),
        "policy_entropy": entropy,
        "num_samples": len(dataset),
        **per_action_acc,
        **pred_dist,
    }

    return metrics


def evaluate_best_action_oracle(
    trainer: OfflineTrainer,
    dataset: OfflineRLDataset,
) -> dict[str, float]:
    """Evaluate policy against the best-action oracle per circuit.

    For grid-search data where every optimizer is tried on every circuit,
    this compares the policy's chosen action against the action that achieved
    the highest reward for each circuit (the oracle best action).

    Args:
        trainer: Trained offline RL trainer
        dataset: Dataset with circuit_ids populated

    Returns:
        Dictionary of oracle evaluation metrics
    """
    if dataset.circuit_ids is None:
        return {"error": -1.0}

    device = trainer.device
    observations = dataset.observations.to(device)
    actions = dataset.actions.numpy()
    rewards = dataset.rewards.numpy()
    circuit_ids = dataset.circuit_ids.numpy()

    # Group indices by circuit_id
    circuit_indices: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(circuit_ids):
        circuit_indices[int(cid)].append(i)

    # For each circuit, find the oracle best action (highest reward)
    oracle_action_per_circuit: dict[int, int] = {}
    for cid, indices in circuit_indices.items():
        best_idx = max(indices, key=lambda i: rewards[i])
        oracle_action_per_circuit[cid] = int(actions[best_idx])

    # Evaluate: for each circuit, check if policy picks the oracle best action
    correct = 0
    total_circuits = len(circuit_indices)
    for cid, indices in circuit_indices.items():
        # Use the first observation for this circuit to query the policy
        obs = observations[indices[0]]
        predicted = trainer.select_action(obs)
        if predicted == oracle_action_per_circuit[cid]:
            correct += 1

    oracle_agreement = correct / total_circuits if total_circuits > 0 else 0.0

    # Best-single oracle: always pick the single most-winning action overall
    action_win_counts: Counter = Counter()
    for cid, best_action in oracle_action_per_circuit.items():
        action_win_counts[best_action] += 1
    best_single_action = action_win_counts.most_common(1)[0][0] if action_win_counts else 0
    best_single_oracle = action_win_counts[best_single_action] / total_circuits if total_circuits > 0 else 0.0

    return {
        "oracle_agreement": float(oracle_agreement),
        "oracle_baseline": 1.0,  # by definition, oracle always picks best
        "best_single_oracle": float(best_single_oracle),
        "best_single_oracle_action": int(best_single_action),
        "num_circuits": total_circuits,
        "random_oracle_expected": 1.0 / max(len(dataset.action_names), 1),
    }


def compute_baselines(
    dataset: OfflineRLDataset,
) -> dict[str, dict[str, float]]:
    """Compute baseline policy metrics for comparison.

    Baselines:
    - random: Uniform random action selection
    - greedy: Always pick the most common action in the dataset
    - best_per_state: Oracle that picks the best action per state (upper bound)

    Args:
        dataset: Evaluation dataset

    Returns:
        Dictionary mapping baseline name to metrics
    """
    true_actions = dataset.actions.numpy()
    rewards = dataset.rewards.numpy()
    n = len(dataset)
    action_dim = true_actions.max() + 1

    baselines: dict[str, dict[str, float]] = {}

    # Random baseline
    rng = np.random.RandomState(42)
    random_actions = rng.randint(0, action_dim, size=n)
    random_agreement = (random_actions == true_actions).mean()
    baselines["random"] = {
        "action_agreement": float(random_agreement),
        "expected_agreement": 1.0 / action_dim,
    }

    # Greedy baseline (always pick most common action)
    action_counts = Counter(true_actions.tolist())
    most_common_action = action_counts.most_common(1)[0][0]
    greedy_agreement = (true_actions == most_common_action).mean()
    baselines["greedy_most_common"] = {
        "action_agreement": float(greedy_agreement),
        "chosen_action": int(most_common_action),
    }

    # Best-single: pick the action with highest average reward
    action_rewards: dict[int, list[float]] = {}
    for a, r in zip(true_actions, rewards):
        action_rewards.setdefault(int(a), []).append(float(r))
    best_action = max(action_rewards, key=lambda a: np.mean(action_rewards[a]))
    best_single_agreement = (true_actions == best_action).mean()
    baselines["best_single"] = {
        "action_agreement": float(best_single_agreement),
        "chosen_action": int(best_action),
        "mean_reward": float(np.mean(action_rewards[best_action])),
    }

    # Oracle baseline: for each circuit, pick the action with the highest reward
    if dataset.circuit_ids is not None:
        cids = dataset.circuit_ids.numpy()
        circuit_indices: dict[int, list[int]] = defaultdict(list)
        for i, cid in enumerate(cids):
            circuit_indices[int(cid)].append(i)

        # For each circuit, find the best action (highest reward)
        oracle_matches = 0
        for cid, indices in circuit_indices.items():
            best_idx = max(indices, key=lambda i: rewards[i])
            best_action_for_circuit = true_actions[best_idx]
            # Count how many dataset rows for this circuit match the oracle
            for i in indices:
                if true_actions[i] == best_action_for_circuit:
                    oracle_matches += 1

        oracle_agreement = oracle_matches / n if n > 0 else 0.0
        baselines["oracle_best_per_circuit"] = {
            "action_agreement": float(oracle_agreement),
            "num_circuits": len(circuit_indices),
        }

    # Action distribution in dataset
    for a in range(action_dim):
        for baseline in baselines.values():
            baseline[f"data_freq_action_{a}"] = float(action_counts.get(a, 0) / n)

    return baselines


def generate_comparison_table(
    policy_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
    action_names: list[str] | None = None,
    oracle_metrics: dict[str, float] | None = None,
) -> str:
    """Generate a formatted comparison table.

    Args:
        policy_metrics: Metrics from evaluate_policy()
        baseline_metrics: Metrics from compute_baselines()
        action_names: Optional optimizer names for display
        oracle_metrics: Optional metrics from evaluate_best_action_oracle()

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Policy Evaluation Results")
    lines.append("=" * 60)

    lines.append("\nTrained Policy:")
    lines.append(f"  Action Agreement: {policy_metrics['action_agreement']:.4f}")
    lines.append(f"  Policy Entropy:   {policy_metrics['policy_entropy']:.4f}")
    lines.append(f"  Samples:          {policy_metrics['num_samples']}")

    lines.append("\nBaselines:")
    for name, metrics in baseline_metrics.items():
        agreement = metrics.get("action_agreement", 0)
        lines.append(f"  {name:25s} agreement={agreement:.4f}")

    # Oracle evaluation section
    if oracle_metrics is not None and "oracle_agreement" in oracle_metrics:
        lines.append("\nOracle Evaluation (best action per circuit):")
        lines.append(f"  Policy oracle agreement:  {oracle_metrics['oracle_agreement']:.4f}")
        lines.append(f"  Best-single oracle:       {oracle_metrics['best_single_oracle']:.4f}"
                      f"  (action={int(oracle_metrics.get('best_single_oracle_action', -1))})")
        lines.append(f"  Random expected:          {oracle_metrics['random_oracle_expected']:.4f}")
        lines.append(f"  Circuits evaluated:       {int(oracle_metrics.get('num_circuits', 0))}")

    # Action distribution comparison
    n_actions = sum(1 for k in policy_metrics if k.startswith("policy_freq_action_"))
    if n_actions > 0:
        lines.append("\nAction Distribution:")
        lines.append(f"  {'Action':<20s} {'Data':>8s} {'Policy':>8s}")
        lines.append(f"  {'-' * 36}")
        for a in range(n_actions):
            name = action_names[a] if action_names and a < len(action_names) else f"action_{a}"
            data_freq = next(iter(baseline_metrics.values())).get(f"data_freq_action_{a}", 0)
            policy_freq = policy_metrics.get(f"policy_freq_action_{a}", 0)
            lines.append(f"  {name:<20s} {data_freq:>8.3f} {policy_freq:>8.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_evaluation_results(
    results: dict[str, Any],
    path: Path | str,
) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _compute_policy_entropy(
    trainer: OfflineTrainer,
    observations: torch.Tensor,
) -> float:
    """Compute average entropy of the policy's action distribution."""
    # Try to get log-probs from the policy network
    policy = getattr(trainer, "policy", None)
    if policy is None:
        # For CQL, use Q-values as logits
        q_net = getattr(trainer, "q_network", None)
        if q_net is None:
            return 0.0
        policy = q_net

    with torch.no_grad():
        logits = policy(observations)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    return float(entropy)
