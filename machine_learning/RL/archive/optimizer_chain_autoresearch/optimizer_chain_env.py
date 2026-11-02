"""Editable Gymnasium environment for selecting optimizer chains.

EDITABLE BY THE RESEARCH AGENT.

The environment is derived from the author's original
``machine_learning/RL/chain_optimizer/optimizer_chain_env.py``.  Its principal
change is that rewards are calculated at transition time by :mod:`reward`
instead of being precomputed in the source dataframe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

try:  # Package import during tests; script import during standalone execution.
    from .reward import compute_reward
except ImportError:  # pragma: no cover - exercised by command-line execution
    from reward import compute_reward


DEFAULT_METRIC_NAMES = (
    "num_qubits",
    "depth",
    "two_qubit_gates",
    "total_gates",
    "two_qubit_depth",
    "gate_density",
    "two_qubit_ratio",
)

COUNT_METRICS = {
    "num_qubits",
    "depth",
    "two_qubit_gates",
    "total_gates",
    "two_qubit_depth",
}


class OptimizerChainEnv(gym.Env[np.ndarray, int]):
    """A table-backed environment for one- and two-optimizer decisions.

    Each episode corresponds to one original circuit.  The first action chooses
    a single optimizer.  At the second decision the policy may stop or append
    another optimizer.  Outcomes are looked up from the complete 30-row table
    rather than executing expensive transpilers during RL training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        optimizer_names: Sequence[str],
        category_names: Sequence[str],
        metric_scale: Mapping[str, float],
        runtime_scale: float,
        reward_config: Mapping[str, Any],
        observation_config: Mapping[str, Any] | None = None,
        *,
        shuffle_each_epoch: bool = True,
        invalid_stop_penalty: float = -1.0,
        max_optimizer_steps: int = 2,
    ) -> None:
        """Initialize the table-backed environment.

        Raises:
            ValueError: If the configured horizon exceeds the two-step source
                data, an observation metric is unknown, or no circuits exist.
        """

        super().__init__()
        if max_optimizer_steps not in {1, 2}:
            raise ValueError("The source dataset supports one or two optimizer steps.")
        if frame.empty:
            raise ValueError("OptimizerChainEnv requires at least one circuit.")

        self.df = frame.copy()
        self.optimizer_names = list(optimizer_names)
        self.category_names = list(category_names)
        self.metric_scale = {key: float(value) for key, value in metric_scale.items()}
        self.runtime_scale = float(runtime_scale)
        self.reward_config = dict(reward_config)
        self.observation_config = dict(observation_config or {})
        self.shuffle_each_epoch = bool(shuffle_each_epoch)
        self.invalid_stop_penalty = float(invalid_stop_penalty)
        self.max_optimizer_steps = int(max_optimizer_steps)

        self.metric_names = list(
            self.observation_config.get("metrics", DEFAULT_METRIC_NAMES)
        )
        unknown_metrics = set(self.metric_names) - set(DEFAULT_METRIC_NAMES)
        if unknown_metrics:
            raise ValueError(f"Unknown observation metrics: {sorted(unknown_metrics)}")

        self.stop_action = len(self.optimizer_names)
        self.action_space = spaces.Discrete(len(self.optimizer_names) + 1)
        self.previous_optimizer_names = ["none", *self.optimizer_names]

        observation_size = len(self.metric_names)
        if self.observation_config.get("include_categories", True):
            observation_size += len(self.category_names)
        if self.observation_config.get("include_previous_optimizer", True):
            observation_size += len(self.previous_optimizer_names)
        if self.observation_config.get("include_step_features", True):
            observation_size += 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.circuits = np.array(sorted(self.df["original_circuit"].unique()))
        self.row_lookup = {
            (str(row.original_circuit), str(row.chain_name)): row
            for row in self.df.itertuples(index=False)
        }
        self.initial_row = {
            str(circuit): rows.iloc[0]
            for circuit, rows in self.df.groupby("original_circuit", sort=True)
        }

        self.epoch_order: np.ndarray | None = None
        self.epoch_index = 0
        self.current_circuit = ""
        self.current_category = ""
        self.current_metrics: dict[str, float] = {}
        self.initial_metrics: dict[str, float] = {}
        self.previous_optimizer = "none"
        self.first_optimizer: str | None = None
        self.step_index = 0
        self.total_runtime_seconds = 0.0

    # ------------------------------------------------------------------
    # Metric and observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_from_initial_row(row: pd.Series) -> dict[str, float]:
        """Extract initial metrics from one dataframe row."""

        return {
            "num_qubits": float(row["initial_num_qubits"]),
            "depth": float(row["initial_depth"]),
            "two_qubit_gates": float(row["initial_two_qubit_gates"]),
            "total_gates": float(row["initial_total_gates"]),
            "two_qubit_depth": float(row["initial_two_qubit_depth"]),
            "gate_density": float(row["initial_gate_density"]),
            "two_qubit_ratio": float(row["initial_two_qubit_ratio"]),
        }

    @staticmethod
    def _metrics_from_result_row(row: Mapping[str, Any]) -> dict[str, float]:
        """Extract final metrics and derive ratios from a result row."""

        total_gates = float(row["final_total_gates"])
        num_qubits = float(row["final_num_qubits"])
        two_qubit_gates = float(row["final_two_qubit_gates"])
        return {
            "num_qubits": num_qubits,
            "depth": float(row["final_depth"]),
            "two_qubit_gates": two_qubit_gates,
            "total_gates": total_gates,
            "two_qubit_depth": float(row["final_two_qubit_depth"]),
            "gate_density": 0.0 if num_qubits == 0 else total_gates / num_qubits,
            "two_qubit_ratio": (
                0.0 if total_gates == 0 else two_qubit_gates / total_gates
            ),
        }

    def _scaled_metric(self, name: str) -> float:
        """Scale one observation metric using training-only statistics."""

        value = float(self.current_metrics[name])
        scale = max(float(self.metric_scale.get(name, 1.0)), 1e-9)
        if (
            self.observation_config.get("log_scale_count_metrics", False)
            and name in COUNT_METRICS
        ):
            value = math.log1p(max(value, 0.0))
            scale = math.log1p(max(scale, 0.0))
        value /= scale
        clip = float(self.observation_config.get("metric_clip", 5.0))
        return float(np.clip(value, -clip, clip))

    def _observation(self) -> np.ndarray:
        """Build the current policy observation."""

        values = [self._scaled_metric(name) for name in self.metric_names]
        if self.observation_config.get("include_categories", True):
            values.extend(
                1.0 if self.current_category == category else 0.0
                for category in self.category_names
            )
        if self.observation_config.get("include_previous_optimizer", True):
            values.extend(
                1.0 if self.previous_optimizer == name else 0.0
                for name in self.previous_optimizer_names
            )
        if self.observation_config.get("include_step_features", True):
            denominator = max(self.max_optimizer_steps, 1)
            values.extend(
                [
                    self.step_index / denominator,
                    (self.max_optimizer_steps - self.step_index) / denominator,
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _next_circuit(self) -> str:
        """Return the next circuit in the deterministic epoch stream."""

        if self.epoch_order is None or self.epoch_index >= len(self.epoch_order):
            self.epoch_order = self.circuits.copy()
            if self.shuffle_each_epoch:
                self.np_random.shuffle(self.epoch_order)
            self.epoch_index = 0
        circuit = str(self.epoch_order[self.epoch_index])
        self.epoch_index += 1
        return circuit

    def _lookup_row(self, chain_name: str) -> Any:
        """Return the source row for the current circuit and chain."""

        try:
            return self.row_lookup[(self.current_circuit, chain_name)]
        except KeyError as exc:
            raise KeyError(
                f"Missing chain {chain_name!r} for circuit {self.current_circuit!r}."
            ) from exc

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode on a requested or automatically selected circuit."""

        super().reset(seed=seed)
        circuit = (
            str(options["circuit"])
            if options and "circuit" in options
            else self._next_circuit()
        )
        if circuit not in self.initial_row:
            raise ValueError(f"Unknown circuit requested in reset: {circuit!r}")

        row = self.initial_row[circuit]
        self.current_circuit = circuit
        self.current_category = str(row["original_category"])
        self.current_metrics = self._metrics_from_initial_row(row)
        self.initial_metrics = dict(self.current_metrics)
        self.previous_optimizer = "none"
        self.first_optimizer = None
        self.step_index = 0
        self.total_runtime_seconds = 0.0
        return self._observation(), {"circuit": circuit}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one optimizer or STOP action and return the resulting transition."""

        action = int(action)
        if not self.action_space.contains(action):
            reward = compute_reward(
                previous_metrics=self.current_metrics,
                current_metrics=self.current_metrics,
                initial_metrics=self.initial_metrics,
                step_runtime_seconds=0.0,
                total_runtime_seconds=self.total_runtime_seconds,
                runtime_scale=self.runtime_scale,
                step_index=self.step_index,
                terminal=True,
                failed=False,
                invalid_action=True,
                config=self.reward_config,
            )
            return self._observation(), reward, True, False, {
                "invalid_action": True,
                "failed": False,
                "selected_chain": None,
                "circuit": self.current_circuit,
            }

        # STOP is invalid before any optimizer has been selected.
        if self.step_index == 0 and action == self.stop_action:
            config = dict(self.reward_config)
            config["invalid_action_penalty"] = self.invalid_stop_penalty
            reward = compute_reward(
                previous_metrics=self.current_metrics,
                current_metrics=self.current_metrics,
                initial_metrics=self.initial_metrics,
                step_runtime_seconds=0.0,
                total_runtime_seconds=0.0,
                runtime_scale=self.runtime_scale,
                step_index=0,
                terminal=True,
                failed=False,
                invalid_action=True,
                config=config,
            )
            return self._observation(), reward, True, False, {
                "invalid_action": True,
                "failed": False,
                "selected_chain": None,
                "circuit": self.current_circuit,
            }

        # STOP after the first optimizer allows sparse terminal rewards to score
        # the final single-optimizer outcome.
        if action == self.stop_action:
            reward = compute_reward(
                previous_metrics=self.current_metrics,
                current_metrics=self.current_metrics,
                initial_metrics=self.initial_metrics,
                step_runtime_seconds=0.0,
                total_runtime_seconds=self.total_runtime_seconds,
                runtime_scale=self.runtime_scale,
                step_index=self.step_index,
                terminal=True,
                failed=False,
                invalid_action=False,
                config=self.reward_config,
            )
            return self._observation(), reward, True, False, {
                "invalid_action": False,
                "failed": False,
                "selected_chain": self.first_optimizer,
                "circuit": self.current_circuit,
            }

        optimizer = self.optimizer_names[action]
        previous_metrics = dict(self.current_metrics)

        if self.step_index == 0:
            chain_name = optimizer
            row = self._lookup_row(chain_name)
            row_dict = row._asdict()
            cumulative_runtime = float(row_dict["total_runtime_seconds"])
            step_runtime = cumulative_runtime
            self.first_optimizer = optimizer
        else:
            if self.first_optimizer is None:
                raise RuntimeError("Second optimizer requested before the first was set.")
            chain_name = f"{self.first_optimizer}__{optimizer}"
            row = self._lookup_row(chain_name)
            row_dict = row._asdict()
            cumulative_runtime = float(row_dict["total_runtime_seconds"])
            step_runtime = max(cumulative_runtime - self.total_runtime_seconds, 0.0)

        failed = bool(row_dict["error"])
        if not failed:
            self.current_metrics = self._metrics_from_result_row(row_dict)
        self.previous_optimizer = optimizer
        self.total_runtime_seconds = cumulative_runtime
        self.step_index += 1

        terminal = failed or self.step_index >= self.max_optimizer_steps
        reward = compute_reward(
            previous_metrics=previous_metrics,
            current_metrics=self.current_metrics,
            initial_metrics=self.initial_metrics,
            step_runtime_seconds=step_runtime,
            total_runtime_seconds=self.total_runtime_seconds,
            runtime_scale=self.runtime_scale,
            step_index=self.step_index - 1,
            terminal=terminal,
            failed=failed,
            invalid_action=False,
            config=self.reward_config,
        )
        return self._observation(), reward, terminal, False, {
            "invalid_action": False,
            "failed": failed,
            "selected_chain": chain_name,
            "circuit": self.current_circuit,
            "step_runtime_seconds": step_runtime,
            "total_runtime_seconds": self.total_runtime_seconds,
        }
