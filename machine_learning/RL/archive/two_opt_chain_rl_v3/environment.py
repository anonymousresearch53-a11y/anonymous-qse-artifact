"""Gymnasium environment for the two-optimizer chain problem.

This is intentionally close to the v1 behavior so experiments stay comparable.
The main difference in v2 is code organization, not environment logic.
"""

import gymnasium as gym
import numpy as np

from gymnasium import spaces

try:
    from .rewards import (
        CURRENT_REWARD,
        compute_final_correction,
        incremental_step_reward,
        single_step_reward,
    )
except ImportError:
    from rewards import (
        CURRENT_REWARD,
        compute_final_correction,
        incremental_step_reward,
        single_step_reward,
    )


# These are the normalized numeric features exposed to the policy.
METRIC_NAMES = [
    "num_qubits",
    "depth",
    "two_qubit_gates",
    "total_gates",
    "two_qubit_depth",
    "gate_density",
    "two_qubit_ratio",
]
OBS_CLIP_MAX = 5.0


class OptimizerChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame,
        optimizer_names,
        category_names,
        metric_scale,
        observation_feature_scale,
        static_feature_columns=None,
        dynamic_feature_columns=None,
        include_category=True,
        shuffle_each_epoch=True,
        invalid_stop_penalty=-1.0,
        reward_type=CURRENT_REWARD,
        runtime_weight=0.1,
        two_qubit_weight=0.5,
        depth_weight=0.5,
        final_reward_weight=0.5,
    ):
        super().__init__()

        # Keep a copy so callers can safely reuse their input frames.
        self.df = frame.copy()
        self.optimizer_names = list(optimizer_names)
        self.category_names = list(category_names)
        self.metric_scale = dict(metric_scale)
        self.observation_feature_scale = dict(observation_feature_scale)
        self.static_feature_columns = list(static_feature_columns or [])
        self.dynamic_feature_columns = list(dynamic_feature_columns or [])
        self.include_category = bool(include_category)
        self.shuffle_each_epoch = shuffle_each_epoch
        self.invalid_stop_penalty = float(invalid_stop_penalty)
        self.reward_type = reward_type
        self.runtime_weight = float(runtime_weight)
        self.two_qubit_weight = float(two_qubit_weight)
        self.depth_weight = float(depth_weight)
        self.final_reward_weight = float(final_reward_weight)

        # The extra action is "stop now".
        self.stop_action = len(self.optimizer_names)
        self.action_space = spaces.Discrete(len(self.optimizer_names) + 1)

        # Observation = normalized metrics + category one-hot +
        # previous optimizer one-hot + step-position features.
        self.prev_optimizer_names = ["none"] + self.optimizer_names
        obs_dim = (
            len(METRIC_NAMES)
            + len(self.static_feature_columns)
            + len(self.dynamic_feature_columns)
            + (len(self.category_names) if self.include_category else 0)
            + len(self.prev_optimizer_names)
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Precompute lookups once so each step is cheap.
        self.circuits = np.array(sorted(self.df["original_circuit"].unique()))
        self.circuit_to_rows = {
            circuit: rows.copy()
            for circuit, rows in self.df.groupby("original_circuit")
        }
        self.row_lookup = {
            (row.original_circuit, row.chain_name): row
            for row in self.df.itertuples(index=False)
        }
        self.initial_row = {
            circuit: rows.iloc[0]
            for circuit, rows in self.circuit_to_rows.items()
        }

        self.epoch_order = None
        self.epoch_idx = 0
        self.current_circuit = None
        self.current_category = None
        self.current_metrics = None
        self.current_static_features = None
        self.current_dynamic_features = None
        self.previous_optimizer = "none"
        self.first_optimizer = None
        self.step_index = 0

    def _next_circuit(self):
        """Return the next circuit, reshuffling between epochs if requested."""
        if self.epoch_order is None or self.epoch_idx >= len(self.epoch_order):
            self.epoch_order = self.circuits.copy()
            if self.shuffle_each_epoch:
                self.np_random.shuffle(self.epoch_order)
            self.epoch_idx = 0
        circuit = self.epoch_order[self.epoch_idx]
        self.epoch_idx += 1
        return circuit

    def _metrics_from_initial_row(self, row):
        """Read the initial-state metrics for a circuit."""
        return {
            "num_qubits": row["initial_num_qubits"],
            "depth": row["initial_depth"],
            "two_qubit_gates": row["initial_two_qubit_gates"],
            "total_gates": row["initial_total_gates"],
            "two_qubit_depth": row["initial_two_qubit_depth"],
            "gate_density": row["initial_gate_density"],
            "two_qubit_ratio": row["initial_two_qubit_ratio"],
        }

    def _metrics_from_final_row(self, row):
        """Read post-optimization metrics and recompute ratio-style features."""
        total_gates = float(row["final_total_gates"])
        num_qubits = float(row["final_num_qubits"])
        return {
            "num_qubits": row["final_num_qubits"],
            "depth": row["final_depth"],
            "two_qubit_gates": row["final_two_qubit_gates"],
            "total_gates": row["final_total_gates"],
            "two_qubit_depth": row["final_two_qubit_depth"],
            "gate_density": 0.0 if num_qubits == 0 else total_gates / num_qubits,
            "two_qubit_ratio": (
                0.0 if total_gates == 0 else row["final_two_qubit_gates"] / total_gates
            ),
        }

    def _normalized_feature_value(self, value, scale_key):
        scale = float(self.observation_feature_scale.get(scale_key, 1.0))
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        return float(np.clip(value / scale, -OBS_CLIP_MAX, OBS_CLIP_MAX))

    def _row_value(self, row, column, default=0.0):
        if isinstance(row, dict):
            return row.get(column, default)
        if hasattr(row, "get"):
            return row.get(column, default)
        return getattr(row, column, default)

    def _safe_float(self, value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value):
            return default
        return value

    def _initial_ratio_feature_fallback(self, row, column):
        num_qubits = self._safe_float(self._row_value(row, "initial_num_qubits"))
        total_gates = self._safe_float(self._row_value(row, "initial_total_gates"))

        if column == "depth_per_qubit":
            numerator = self._safe_float(self._row_value(row, "initial_depth"))
            return 0.0 if num_qubits == 0 else numerator / num_qubits
        if column == "gates_per_qubit":
            return 0.0 if num_qubits == 0 else total_gates / num_qubits
        if column == "two_qubit_gates_per_qubit":
            numerator = self._safe_float(
                self._row_value(row, "initial_two_qubit_gates")
            )
            return 0.0 if num_qubits == 0 else numerator / num_qubits
        if column == "two_qubit_gate_fraction":
            numerator = self._safe_float(
                self._row_value(row, "initial_two_qubit_gates")
            )
            return 0.0 if total_gates == 0 else numerator / total_gates
        return 0.0

    def _static_features_from_row(self, row, state):
        features = {}
        for column in self.static_feature_columns:
            source_column = f"initial_{column}" if state == "initial" else column
            value = self._row_value(row, source_column, None)
            if value is None and state == "initial":
                value = self._initial_ratio_feature_fallback(row, column)
            features[column] = self._safe_float(value)
        return features

    def _dynamic_features_from_row(self, row):
        return {
            column: self._safe_float(self._row_value(row, column, 0.0))
            for column in self.dynamic_feature_columns
        }

    def _zero_dynamic_features(self):
        return {column: 0.0 for column in self.dynamic_feature_columns}

    def _obs(self):
        """Build the policy observation vector for the current state."""
        # Clip normalized metrics so very large circuits do not dominate scale.
        metrics = [
            min(float(self.current_metrics[name]) / self.metric_scale[name], OBS_CLIP_MAX)
            for name in METRIC_NAMES
        ]
        static_features = [
            self._normalized_feature_value(
                self.current_static_features.get(name, 0.0),
                name,
            )
            for name in self.static_feature_columns
        ]
        dynamic_features = [
            self._normalized_feature_value(
                self.current_dynamic_features.get(name, 0.0),
                name,
            )
            for name in self.dynamic_feature_columns
        ]

        category = (
            [
                1.0 if self.current_category == category_name else 0.0
                for category_name in self.category_names
            ]
            if self.include_category
            else []
        )

        previous_optimizer = [
            1.0 if self.previous_optimizer == optimizer_name else 0.0
            for optimizer_name in self.prev_optimizer_names
        ]

        # Two simple position features are enough for this fixed two-step setup.
        step_features = [
            float(self.step_index),
            float(1 - self.step_index),
        ]
        return np.array(
            metrics
            + static_features
            + dynamic_features
            + category
            + previous_optimizer
            + step_features,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """Start a new episode on a chosen or next circuit."""
        super().reset(seed=seed)
        if options and "circuit" in options:
            circuit = options["circuit"]
        else:
            circuit = self._next_circuit()

        row = self.initial_row[circuit]
        self.current_circuit = circuit
        self.current_category = row["original_category"]
        self.current_metrics = self._metrics_from_initial_row(row)
        self.current_static_features = self._static_features_from_row(
            row,
            state="initial",
        )
        self.current_dynamic_features = self._zero_dynamic_features()
        self.previous_optimizer = "none"
        self.first_optimizer = None
        self.step_index = 0
        return self._obs(), {"circuit": self.current_circuit}

    def _chain_row(self, chain_name):
        """Fetch one specific chain result for the active circuit."""
        return self.row_lookup[(self.current_circuit, chain_name)]

    def _has_chain_row(self, chain_name):
        """Return whether the requested chain exists for the active circuit."""
        return (self.current_circuit, chain_name) in self.row_lookup

    def _final_correction(self, row):
        """Compute any terminal reward correction for the selected final row."""
        return float(
            compute_final_correction(
                row,
                reward_type=self.reward_type,
                final_reward_weight=self.final_reward_weight,
                two_qubit_weight=self.two_qubit_weight,
                depth_weight=self.depth_weight,
            )
        )

    def step(self, action):
        """Apply the first optimizer, stop, or finish the second step.

        Step 0:
        - choose first optimizer, or
        - stop immediately and keep the original circuit.

        Step 1:
        - choose stop to keep the single optimizer result, or
        - choose a second optimizer and receive the incremental reward gain.
        """
        action = int(action)

        if self.step_index == 0:
            if action == self.stop_action:
                info = {
                    "invalid_action": False,
                    "selected_chain": "__original__",
                    "circuit": self.current_circuit,
                }
                return self._obs(), 0.0, True, False, info

            # First action chooses a single optimizer result and receives
            # that row's own step-local reward.
            optimizer = self.optimizer_names[action]
            if not self._has_chain_row(optimizer):
                info = {
                    "invalid_action": True,
                    "selected_chain": None,
                    "circuit": self.current_circuit,
                }
                return self._obs(), self.invalid_stop_penalty, True, False, info

            row = self._chain_row(optimizer)
            reward = float(
                single_step_reward(
                    row,
                    reward_type=self.reward_type,
                    runtime_weight=self.runtime_weight,
                    two_qubit_weight=self.two_qubit_weight,
                    depth_weight=self.depth_weight,
                )
            )

            self.first_optimizer = optimizer
            self.previous_optimizer = optimizer
            row_dict = row._asdict()
            self.current_metrics = self._metrics_from_final_row(row_dict)
            self.current_static_features = self._static_features_from_row(
                row_dict,
                state="final",
            )
            self.current_dynamic_features = self._dynamic_features_from_row(row_dict)
            self.step_index = 1
            info = {
                "invalid_action": False,
                "selected_chain": optimizer,
                "circuit": self.current_circuit,
            }
            return self._obs(), reward, False, False, info

        if action == self.stop_action:
            # Stop after one optimizer and keep the reward already earned.
            single_row = self._chain_row(self.first_optimizer)
            info = {
                "invalid_action": False,
                "selected_chain": self.first_optimizer,
                "circuit": self.current_circuit,
            }
            return self._obs(), self._final_correction(single_row), True, False, info

        # Second action selects the second optimizer in the chain.
        second_optimizer = self.optimizer_names[action]
        single_row = self._chain_row(self.first_optimizer)
        chain_name = f"{self.first_optimizer}__{second_optimizer}"
        if not self._has_chain_row(chain_name):
            info = {
                "invalid_action": True,
                "selected_chain": self.first_optimizer,
                "circuit": self.current_circuit,
            }
            return self._obs(), self.invalid_stop_penalty, True, False, info

        chain_row = self._chain_row(chain_name)

        # Reward at step 2 is the current row's own step-local reward.
        # The current-step reduction percentage was already measured against
        # the parent circuit state in preprocessing.
        reward = float(
            incremental_step_reward(
                chain_row,
                single_row,
                reward_type=self.reward_type,
                runtime_weight=self.runtime_weight,
                two_qubit_weight=self.two_qubit_weight,
                depth_weight=self.depth_weight,
            )
        )
        reward += self._final_correction(chain_row)

        chain_row_dict = chain_row._asdict()
        self.current_metrics = self._metrics_from_final_row(chain_row_dict)
        self.current_static_features = self._static_features_from_row(
            chain_row_dict,
            state="final",
        )
        self.current_dynamic_features = self._dynamic_features_from_row(chain_row_dict)
        self.previous_optimizer = second_optimizer
        self.step_index = 2
        info = {
            "invalid_action": False,
            "selected_chain": chain_name,
            "circuit": self.current_circuit,
        }
        return self._obs(), reward, True, False, info
