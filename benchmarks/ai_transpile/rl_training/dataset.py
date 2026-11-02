"""Dataset loading and splitting for offline RL training.

Loads trajectory data from the SQLite database, remaps optimizer IDs to
0-indexed action indices, splits by circuit ID to prevent data leakage,
and applies normalization.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..rl_trajectory.state import get_category_encoding
from .config import TrainingConfig
from .normalization import NormalizationStats, compute_normalization_stats


def _encode_category_for_state(category_name: str) -> list[float]:
    """Build the current category one-hot vector from the authoritative text label."""
    return get_category_encoding(category_name)


class OfflineRLDataset(Dataset):
    """PyTorch Dataset for offline RL training from trajectory data.

    Attributes:
        observations: Normalized state vectors, shape (N, state_dim)
        actions: 0-indexed action indices, shape (N,)
        rewards: Reward values, shape (N,)
        next_observations: Normalized next-state vectors, shape (N, state_dim)
        terminals: Done flags, shape (N,)
        norm_stats: Normalization statistics used
        action_map: Mapping from DB optimizer_id to 0-indexed action
        action_names: List of optimizer names indexed by action index
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        norm_stats: NormalizationStats,
        action_map: dict[int, int],
        action_names: list[str],
        circuit_ids: np.ndarray | None = None,
    ):
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.next_observations = torch.as_tensor(next_observations, dtype=torch.float32)
        self.terminals = torch.as_tensor(terminals, dtype=torch.float32)
        self.norm_stats = norm_stats
        self.action_map = action_map
        self.action_names = action_names
        self.circuit_ids = torch.as_tensor(circuit_ids, dtype=torch.long) if circuit_ids is not None else None

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
        }

    @classmethod
    def from_database(
        cls,
        db_path: Path | str,
        config: TrainingConfig,
        norm_stats: NormalizationStats | None = None,
    ) -> OfflineRLDataset:
        """Load the full dataset from a trajectory database.

        Args:
            db_path: Path to the SQLite database
            config: Training configuration
            norm_stats: Pre-computed normalization stats (computed from data if None)

        Returns:
            OfflineRLDataset with all transitions
        """
        db_path = Path(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Build optimizer ID → 0-indexed action mapping
        action_map, action_names = _build_action_map(conn)

        # Load trajectory steps
        observations, actions, rewards, next_observations, terminals = _load_trajectory_steps(
            conn, config.reward_type, action_map
        )
        if config.reward_clip is not None:
            rewards = np.clip(rewards, -config.reward_clip, config.reward_clip)
        circuit_ids_list = _get_step_circuit_ids(conn)
        conn.close()

        circuit_ids = np.array(circuit_ids_list, dtype=np.int64) if circuit_ids_list else None

        if observations.shape[0] == 0:
            msg = f"No trajectory steps found in {db_path}"
            raise ValueError(msg)

        # Compute or apply normalization
        if norm_stats is None:
            norm_stats = compute_normalization_stats(observations)

        norm_obs = norm_stats.normalize(observations)
        norm_next_obs = norm_stats.normalize(next_observations)

        return cls(
            observations=norm_obs,
            actions=actions,
            rewards=rewards,
            next_observations=norm_next_obs,
            terminals=terminals,
            norm_stats=norm_stats,
            action_map=action_map,
            action_names=action_names,
            circuit_ids=circuit_ids,
        )

    def numpy_dict(self) -> dict[str, np.ndarray]:
        """Export dataset tensors as numpy arrays."""
        data = {
            "observations": self.observations.numpy(),
            "actions": self.actions.numpy(),
            "rewards": self.rewards.numpy(),
            "next_observations": self.next_observations.numpy(),
            "terminals": self.terminals.numpy(),
        }
        if self.circuit_ids is not None:
            data["circuit_ids"] = self.circuit_ids.numpy()
        return data


def split_dataset(
    dataset: OfflineRLDataset,
    db_path: Path | str,
    config: TrainingConfig,
) -> tuple[OfflineRLDataset, OfflineRLDataset, OfflineRLDataset]:
    """Split dataset into train/val/test sets.

    When split_by_circuit is True, splits by circuit ID so that all
    transitions from a given circuit appear in the same split. This
    prevents data leakage from seeing the same circuit in train and val.

    Args:
        dataset: Full dataset to split
        db_path: Path to the database (for circuit ID lookup)
        config: Training configuration with split fractions

    Returns:
        (train_dataset, val_dataset, test_dataset) tuple
    """
    rng = np.random.RandomState(config.seed)
    n = len(dataset)

    if config.split_by_circuit:
        # Get circuit IDs for each trajectory step
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        circuit_ids = _get_step_circuit_ids(conn)
        conn.close()

        if len(circuit_ids) != n:
            # Fallback to random split if circuit IDs don't match
            return _random_split(dataset, config, rng)

        # Group indices by circuit
        circuit_to_indices: dict[int, list[int]] = {}
        for i, cid in enumerate(circuit_ids):
            circuit_to_indices.setdefault(cid, []).append(i)

        # Shuffle circuits, then assign to splits
        unique_circuits = list(circuit_to_indices.keys())
        rng.shuffle(unique_circuits)

        n_val = max(1, int(len(unique_circuits) * config.val_fraction))
        n_test = max(1, int(len(unique_circuits) * config.test_fraction))

        val_circuits = set(unique_circuits[:n_val])
        test_circuits = set(unique_circuits[n_val:n_val + n_test])
        train_circuits = set(unique_circuits[n_val + n_test:])

        train_idx = [i for c in train_circuits for i in circuit_to_indices[c]]
        val_idx = [i for c in val_circuits for i in circuit_to_indices[c]]
        test_idx = [i for c in test_circuits for i in circuit_to_indices[c]]
    else:
        indices = rng.permutation(n)
        n_val = max(1, int(n * config.val_fraction))
        n_test = max(1, int(n * config.test_fraction))
        val_idx = indices[:n_val].tolist()
        test_idx = indices[n_val:n_val + n_test].tolist()
        train_idx = indices[n_val + n_test:].tolist()

    return (
        _subset(dataset, train_idx),
        _subset(dataset, val_idx),
        _subset(dataset, test_idx),
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a dataset.

    Args:
        dataset: The dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def subset_dataset(dataset: OfflineRLDataset, indices: list[int]) -> OfflineRLDataset:
    """Create a subset of the dataset with the given indices."""
    return _subset(dataset, indices)


def concat_datasets(
    datasets: list[OfflineRLDataset],
    *,
    repeat_factors: list[int] | None = None,
) -> OfflineRLDataset:
    """Concatenate multiple offline RL datasets.

    Args:
        datasets: Datasets to concatenate
        repeat_factors: Optional repeat multiplier per dataset

    Returns:
        Combined dataset
    """
    if not datasets:
        raise ValueError("At least one dataset is required")

    non_empty = [ds for ds in datasets if len(ds) > 0]
    if not non_empty:
        raise ValueError("Cannot concatenate only empty datasets")

    if repeat_factors is None:
        repeat_factors = [1] * len(datasets)
    if len(repeat_factors) != len(datasets):
        raise ValueError("repeat_factors must match datasets length")

    reference = non_empty[0]
    arrays: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminals": [],
    }
    circuit_parts: list[np.ndarray] = []
    include_circuit_ids = all(ds.circuit_ids is not None for ds in non_empty)

    for ds, repeat in zip(datasets, repeat_factors):
        if len(ds) == 0 or repeat <= 0:
            continue
        if ds.action_names != reference.action_names:
            raise ValueError("All datasets must share the same action ordering")

        data = ds.numpy_dict()
        for key in arrays:
            arrays[key].append(np.repeat(data[key], repeat, axis=0))
        if include_circuit_ids:
            circuit_parts.append(np.repeat(data["circuit_ids"], repeat, axis=0))

    return OfflineRLDataset(
        observations=np.concatenate(arrays["observations"], axis=0),
        actions=np.concatenate(arrays["actions"], axis=0),
        rewards=np.concatenate(arrays["rewards"], axis=0),
        next_observations=np.concatenate(arrays["next_observations"], axis=0),
        terminals=np.concatenate(arrays["terminals"], axis=0),
        norm_stats=reference.norm_stats,
        action_map=reference.action_map,
        action_names=reference.action_names,
        circuit_ids=np.concatenate(circuit_parts, axis=0) if circuit_parts else None,
    )


def filter_dataset_by_circuit_kind(
    dataset: OfflineRLDataset,
    db_path: Path | str,
    circuit_kind: str,
) -> OfflineRLDataset:
    """Filter a dataset to original or artifact circuits."""
    if circuit_kind == "all":
        return dataset
    if dataset.circuit_ids is None:
        raise ValueError("Dataset does not contain circuit_ids for filtering")

    allowed = get_circuit_ids_by_kind(db_path, circuit_kind)
    indices = [
        idx for idx, cid in enumerate(dataset.circuit_ids.tolist())
        if int(cid) in allowed
    ]
    return subset_dataset(dataset, indices)


def get_circuit_ids_by_kind(
    db_path: Path | str,
    circuit_kind: str,
) -> set[int]:
    """Return the circuit IDs belonging to a given circuit kind."""
    if circuit_kind not in {"all", "original", "artifact"}:
        raise ValueError(f"Unknown circuit_kind: {circuit_kind}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if circuit_kind == "all":
            rows = conn.execute("SELECT id FROM circuits").fetchall()
        elif circuit_kind == "original":
            rows = conn.execute("SELECT id FROM circuits WHERE name NOT LIKE 'artifact_%'").fetchall()
        else:
            rows = conn.execute("SELECT id FROM circuits WHERE name LIKE 'artifact_%'").fetchall()
    finally:
        conn.close()
    return {int(row["id"]) for row in rows}


def get_circuit_metadata(
    db_path: Path | str,
    circuit_ids: set[int] | None = None,
) -> dict[int, dict[str, str]]:
    """Load circuit metadata keyed by circuit ID."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if circuit_ids:
            placeholders = ",".join("?" for _ in circuit_ids)
            rows = conn.execute(
                f"SELECT id, name, category FROM circuits WHERE id IN ({placeholders})",
                tuple(sorted(circuit_ids)),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id, name, category FROM circuits").fetchall()
    finally:
        conn.close()

    return {
        int(row["id"]): {
            "name": str(row["name"]),
            "category": str(row["category"]),
            "kind": "artifact" if str(row["name"]).startswith("artifact_") else "original",
        }
        for row in rows
    }


# --- Private helpers ---


def _build_action_map(conn: sqlite3.Connection) -> tuple[dict[int, int], list[str]]:
    """Build mapping from DB optimizer_id (1-indexed) to 0-indexed actions.

    Returns:
        (action_map, action_names) where action_map maps DB IDs to indices
        and action_names[i] is the optimizer name for action i
    """
    rows = conn.execute("SELECT id, name FROM optimizers ORDER BY id").fetchall()
    action_map: dict[int, int] = {}
    action_names: list[str] = []
    for idx, row in enumerate(rows):
        action_map[row["id"]] = idx
        action_names.append(row["name"])
    return action_map, action_names


def _load_trajectory_steps(
    conn: sqlite3.Connection,
    reward_type: str,
    action_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all trajectory steps from the database.

    Returns:
        (observations, actions, rewards, next_observations, terminals)
    """
    rows = conn.execute(
        f"""
        SELECT
            ts.state_depth, ts.state_two_qubit_gates, ts.state_two_qubit_depth,
            ts.state_total_gates, ts.state_num_qubits, ts.state_gate_density,
            ts.state_two_qubit_ratio, ts.state_steps_taken,
            ts.state_time_budget_remaining, ts.state_category_json,
            c.category,
            ts.next_state_depth, ts.next_state_two_qubit_gates,
            ts.next_state_two_qubit_depth, ts.next_state_total_gates,
            ts.state_num_qubits as next_num_qubits,
            ts.next_state_gate_density, ts.next_state_two_qubit_ratio,
            ts.next_state_steps_taken, ts.next_state_time_budget_remaining,
            ts.optimizer_id, ts.{reward_type}, ts.done
        FROM trajectory_steps ts
        JOIN trajectories t ON ts.trajectory_id = t.id
        JOIN circuits c ON t.circuit_id = c.id
        ORDER BY ts.trajectory_id, ts.step_index
        """  # noqa: S608
    ).fetchall()

    if not rows:
        return (
            np.array([], dtype=np.float32).reshape(0, 26),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(0, 26),
            np.array([], dtype=np.bool_),
        )

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    for row in rows:
        category = _encode_category_for_state(row["category"])

        s_2q = row["state_two_qubit_gates"]
        s_2qd = row["state_two_qubit_depth"]
        s_nq = row["state_num_qubits"]
        state = [
            row["state_depth"], s_2q,
            s_2qd, row["state_total_gates"],
            s_nq, row["state_gate_density"],
            row["state_two_qubit_ratio"], row["state_steps_taken"],
            row["state_time_budget_remaining"],
            # Enriched features (Step 1)
            float(np.log1p(s_2q)),
            float((s_2q - s_2qd) / max(s_2q, 1)),
            float(0.0 if s_nq <= 5 else 0.5 if s_nq <= 12 else 1.0),
        ] + category

        n_2q = row["next_state_two_qubit_gates"]
        n_2qd = row["next_state_two_qubit_depth"]
        n_nq = row["next_num_qubits"]
        next_state = [
            row["next_state_depth"], n_2q,
            n_2qd, row["next_state_total_gates"],
            n_nq, row["next_state_gate_density"],
            row["next_state_two_qubit_ratio"], row["next_state_steps_taken"],
            row["next_state_time_budget_remaining"],
            # Enriched features (Step 1)
            float(np.log1p(n_2q)),
            float((n_2q - n_2qd) / max(n_2q, 1)),
            float(0.0 if n_nq <= 5 else 0.5 if n_nq <= 12 else 1.0),
        ] + category

        db_action = row["optimizer_id"]
        action_idx = action_map.get(db_action, 0)

        observations.append(state)
        actions.append(action_idx)
        rewards.append(row[reward_type])
        next_observations.append(next_state)
        terminals.append(bool(row["done"]))

    return (
        np.array(observations, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_observations, dtype=np.float32),
        np.array(terminals, dtype=np.bool_),
    )


def _get_step_circuit_ids(conn: sqlite3.Connection) -> list[int]:
    """Get circuit_id for each trajectory step (in trajectory/step order)."""
    rows = conn.execute(
        """
        SELECT t.circuit_id
        FROM trajectory_steps ts
        JOIN trajectories t ON ts.trajectory_id = t.id
        ORDER BY ts.trajectory_id, ts.step_index
        """
    ).fetchall()
    return [row["circuit_id"] for row in rows]


def _random_split(
    dataset: OfflineRLDataset,
    config: TrainingConfig,
    rng: np.random.RandomState,
) -> tuple[OfflineRLDataset, OfflineRLDataset, OfflineRLDataset]:
    """Fallback random split."""
    n = len(dataset)
    indices = rng.permutation(n)
    n_val = max(1, int(n * config.val_fraction))
    n_test = max(1, int(n * config.test_fraction))
    val_idx = indices[:n_val].tolist()
    test_idx = indices[n_val:n_val + n_test].tolist()
    train_idx = indices[n_val + n_test:].tolist()
    return _subset(dataset, train_idx), _subset(dataset, val_idx), _subset(dataset, test_idx)


def _subset(dataset: OfflineRLDataset, indices: list[int]) -> OfflineRLDataset:
    """Create a subset of the dataset with given indices."""
    idx = torch.tensor(indices, dtype=torch.long)
    return OfflineRLDataset(
        observations=dataset.observations[idx].numpy(),
        actions=dataset.actions[idx].numpy(),
        rewards=dataset.rewards[idx].numpy(),
        next_observations=dataset.next_observations[idx].numpy(),
        terminals=dataset.terminals[idx].numpy(),
        norm_stats=dataset.norm_stats,
        action_map=dataset.action_map,
        action_names=dataset.action_names,
        circuit_ids=dataset.circuit_ids[idx].numpy() if dataset.circuit_ids is not None else None,
    )


class DTOfflineDataset(Dataset):
    """Dataset for Decision Transformer training.

    Groups trajectory steps by trajectory_id and computes return-to-go (RTG)
    per step: R_t = sum(r_t, r_{t+1}, ..., r_T).

    Each episode is represented as a sequence of (RTG, state, action) triples,
    padded with zeros to max_ep_len.

    Attributes:
        rtgs: Return-to-go per step, shape (N_episodes, max_ep_len, 1)
        states: Normalized state vectors, shape (N_episodes, max_ep_len, state_dim)
        actions: Action indices, shape (N_episodes, max_ep_len)
        timesteps: Step indices 0..T, shape (N_episodes, max_ep_len)
        masks: Padding mask (1=real, 0=padded), shape (N_episodes, max_ep_len)
        action_names: Optimizer names indexed by action index
        norm_stats: Normalization statistics
    """

    def __init__(
        self,
        rtgs: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
        masks: np.ndarray,
        norm_stats: NormalizationStats,
        action_map: dict[int, int],
        action_names: list[str],
    ):
        self.rtgs = torch.as_tensor(rtgs, dtype=torch.float32)
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.timesteps = torch.as_tensor(timesteps, dtype=torch.long)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.norm_stats = norm_stats
        self.action_map = action_map
        self.action_names = action_names

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "rtgs": self.rtgs[idx],
            "states": self.states[idx],
            "actions": self.actions[idx],
            "timesteps": self.timesteps[idx],
            "masks": self.masks[idx],
        }

    @classmethod
    def from_database(
        cls,
        db_path: Path | str,
        config: TrainingConfig,
        norm_stats: NormalizationStats | None = None,
        max_ep_len: int = 3,
    ) -> DTOfflineDataset:
        """Load and group trajectory steps by trajectory ID for DT training.

        Args:
            db_path: Path to the SQLite database
            config: Training configuration
            norm_stats: Pre-computed normalization stats (computed from data if None)
            max_ep_len: Maximum episode length (for padding)

        Returns:
            DTOfflineDataset with padded episode sequences
        """
        db_path = Path(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        action_map, action_names = _build_action_map(conn)
        reward_type = config.reward_type

        # Load all steps grouped by trajectory
        rows = conn.execute(
            f"""
            SELECT
                ts.trajectory_id, ts.step_index,
                ts.state_depth, ts.state_two_qubit_gates, ts.state_two_qubit_depth,
                ts.state_total_gates, ts.state_num_qubits, ts.state_gate_density,
                ts.state_two_qubit_ratio, ts.state_steps_taken,
                ts.state_time_budget_remaining, ts.state_category_json,
                c.category,
                ts.next_state_two_qubit_gates, ts.next_state_two_qubit_depth,
                ts.optimizer_id, ts.{reward_type}, ts.done
            FROM trajectory_steps ts
            JOIN trajectories t ON ts.trajectory_id = t.id
            JOIN circuits c ON t.circuit_id = c.id
            ORDER BY ts.trajectory_id, ts.step_index
            """  # noqa: S608
        ).fetchall()
        conn.close()

        # Group by trajectory_id
        trajectories: dict[int, list] = {}
        for row in rows:
            tid = row["trajectory_id"]
            trajectories.setdefault(tid, []).append(row)

        # Build raw observations for normalization
        all_obs: list[list[float]] = []
        for steps in trajectories.values():
            for row in steps:
                cat = _encode_category_for_state(row["category"])
                s_2q = row["state_two_qubit_gates"]
                s_2qd = row["state_two_qubit_depth"]
                s_nq = row["state_num_qubits"]
                obs = [
                    row["state_depth"], s_2q, s_2qd, row["state_total_gates"],
                    s_nq, row["state_gate_density"], row["state_two_qubit_ratio"],
                    row["state_steps_taken"], row["state_time_budget_remaining"],
                    float(np.log1p(s_2q)),
                    float((s_2q - s_2qd) / max(s_2q, 1)),
                    float(0.0 if s_nq <= 5 else 0.5 if s_nq <= 12 else 1.0),
                ] + cat
                all_obs.append(obs)

        if not all_obs:
            msg = f"No trajectory steps found in {db_path}"
            raise ValueError(msg)

        raw_obs = np.array(all_obs, dtype=np.float32)
        if norm_stats is None:
            norm_stats = compute_normalization_stats(raw_obs)

        state_dim = raw_obs.shape[1]
        n_eps = len(trajectories)

        rtgs_arr = np.zeros((n_eps, max_ep_len, 1), dtype=np.float32)
        states_arr = np.zeros((n_eps, max_ep_len, state_dim), dtype=np.float32)
        actions_arr = np.zeros((n_eps, max_ep_len), dtype=np.int64)
        timesteps_arr = np.zeros((n_eps, max_ep_len), dtype=np.int64)
        masks_arr = np.zeros((n_eps, max_ep_len), dtype=np.float32)

        for ep_idx, steps in enumerate(trajectories.values()):
            ep_len = min(len(steps), max_ep_len)
            # Collect rewards and compute RTG
            ep_rewards = [float(s[reward_type]) for s in steps[:max_ep_len]]
            rtg = sum(ep_rewards)
            for t in range(ep_len):
                row = steps[t]
                cat = _encode_category_for_state(row["category"])
                s_2q = row["state_two_qubit_gates"]
                s_2qd = row["state_two_qubit_depth"]
                s_nq = row["state_num_qubits"]
                raw_state = np.array([
                    row["state_depth"], s_2q, s_2qd, row["state_total_gates"],
                    s_nq, row["state_gate_density"], row["state_two_qubit_ratio"],
                    row["state_steps_taken"], row["state_time_budget_remaining"],
                    float(np.log1p(s_2q)),
                    float((s_2q - s_2qd) / max(s_2q, 1)),
                    float(0.0 if s_nq <= 5 else 0.5 if s_nq <= 12 else 1.0),
                ] + cat, dtype=np.float32)

                rtgs_arr[ep_idx, t, 0] = rtg
                states_arr[ep_idx, t] = norm_stats.normalize(raw_state)
                actions_arr[ep_idx, t] = action_map.get(row["optimizer_id"], 0)
                timesteps_arr[ep_idx, t] = t
                masks_arr[ep_idx, t] = 1.0
                rtg -= ep_rewards[t]

        return cls(
            rtgs=rtgs_arr,
            states=states_arr,
            actions=actions_arr,
            timesteps=timesteps_arr,
            masks=masks_arr.astype(bool),
            norm_stats=norm_stats,
            action_map=action_map,
            action_names=action_names,
        )
