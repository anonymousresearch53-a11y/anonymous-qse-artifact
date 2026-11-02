import gymnasium as gym
import numpy as np

from gymnasium import spaces


METRIC_NAMES = [
    "num_qubits",
    "depth",
    "two_qubit_gates",
    "total_gates",
    "two_qubit_depth",
    "gate_density",
    "two_qubit_ratio",
]


class OptimizerChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame,
        optimizer_names,
        category_names,
        metric_scale,
        shuffle_each_epoch=True,
        invalid_stop_penalty=-1.0,
    ):
        super().__init__()
        self.df = frame.copy()
        self.optimizer_names = list(optimizer_names)
        self.category_names = list(category_names)
        self.metric_scale = dict(metric_scale)
        self.shuffle_each_epoch = shuffle_each_epoch
        self.invalid_stop_penalty = float(invalid_stop_penalty)

        self.stop_action = len(self.optimizer_names)
        self.action_space = spaces.Discrete(len(self.optimizer_names) + 1)

        self.prev_optimizer_names = ["none"] + self.optimizer_names
        obs_dim = (
            len(METRIC_NAMES)
            + len(self.category_names)
            + len(self.prev_optimizer_names)
            + 2
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

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
        self.previous_optimizer = "none"
        self.first_optimizer = None
        self.step_index = 0

    def _next_circuit(self):
        if self.epoch_order is None or self.epoch_idx >= len(self.epoch_order):
            self.epoch_order = self.circuits.copy()
            if self.shuffle_each_epoch:
                self.np_random.shuffle(self.epoch_order)
            self.epoch_idx = 0
        circuit = self.epoch_order[self.epoch_idx]
        self.epoch_idx += 1
        return circuit

    def _metrics_from_initial_row(self, row):
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

    def _obs(self):
        metrics = [
            min(float(self.current_metrics[name]) / self.metric_scale[name], 5.0)
            for name in METRIC_NAMES
        ]
        category = [
            1.0 if self.current_category == category_name else 0.0
            for category_name in self.category_names
        ]
        previous_optimizer = [
            1.0 if self.previous_optimizer == optimizer_name else 0.0
            for optimizer_name in self.prev_optimizer_names
        ]
        step_features = [
            float(self.step_index),
            float(1 - self.step_index),
        ]
        return np.array(
            metrics + category + previous_optimizer + step_features,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "circuit" in options:
            circuit = options["circuit"]
        else:
            circuit = self._next_circuit()

        row = self.initial_row[circuit]
        self.current_circuit = circuit
        self.current_category = row["original_category"]
        self.current_metrics = self._metrics_from_initial_row(row)
        self.previous_optimizer = "none"
        self.first_optimizer = None
        self.step_index = 0
        return self._obs(), {"circuit": self.current_circuit}

    def _chain_row(self, chain_name):
        return self.row_lookup[(self.current_circuit, chain_name)]

    def step(self, action):
        action = int(action)

        if self.step_index == 0:
            if action == self.stop_action:
                info = {
                    "invalid_action": True,
                    "selected_chain": None,
                    "circuit": self.current_circuit,
                }
                return self._obs(), self.invalid_stop_penalty, True, False, info

            optimizer = self.optimizer_names[action]
            row = self._chain_row(optimizer)
            reward = float(row.utility_reward)

            self.first_optimizer = optimizer
            self.previous_optimizer = optimizer
            self.current_metrics = self._metrics_from_final_row(row._asdict())
            self.step_index = 1
            info = {
                "invalid_action": False,
                "selected_chain": optimizer,
                "circuit": self.current_circuit,
            }
            return self._obs(), reward, False, False, info

        if action == self.stop_action:
            info = {
                "invalid_action": False,
                "selected_chain": self.first_optimizer,
                "circuit": self.current_circuit,
            }
            return self._obs(), 0.0, True, False, info

        second_optimizer = self.optimizer_names[action]
        single_row = self._chain_row(self.first_optimizer)
        chain_name = f"{self.first_optimizer}__{second_optimizer}"
        chain_row = self._chain_row(chain_name)
        reward = float(chain_row.utility_reward - single_row.utility_reward)

        self.previous_optimizer = second_optimizer
        self.step_index = 2
        info = {
            "invalid_action": False,
            "selected_chain": chain_name,
            "circuit": self.current_circuit,
        }
        return self._obs(), reward, True, False, info
