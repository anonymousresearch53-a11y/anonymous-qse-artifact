"""RL state representation for quantum circuit optimization.

This module provides state extraction and feature computation for
the RL-based optimizer orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit

from ..transpilers import CircuitMetrics, analyze_circuit

# Known circuit categories for one-hot encoding
CATEGORIES: list[str] = [
    "qft",
    "qaoa",
    "clifford",
    "qv",
    "bigint",
    "dtc",
    "feynman",
    "square-heisenberg",
    "qasmbench-small",
    "qasmbench-medium",
    "qasmbench-large",
    "guoq_ibmnew",  # GUOQ IBMN gate set benchmarks
    "local",  # locally defined circuits
    "unknown",
]

CATEGORY_TO_INDEX: dict[str, int] = {cat: i for i, cat in enumerate(CATEGORIES)}


def get_category_encoding(category: str) -> list[float]:
    """Get one-hot encoding for a circuit category.

    Args:
        category: Category name

    Returns:
        One-hot encoded vector as list of floats
    """
    encoding = [0.0] * len(CATEGORIES)
    idx = CATEGORY_TO_INDEX.get(category, CATEGORY_TO_INDEX["unknown"])
    encoding[idx] = 1.0
    return encoding


@dataclass
class RLState:
    """State representation for the RL environment.

    This extends the existing OptimizationState design from rl_orchestrator.py
    with additional derived features useful for learning.

    Attributes:
        depth: Circuit depth
        two_qubit_gates: Number of two-qubit gates
        two_qubit_depth: Depth counting only two-qubit gates
        total_gates: Total number of gates
        num_qubits: Number of qubits in the circuit
        gate_density: Ratio of total_gates to num_qubits
        two_qubit_ratio: Ratio of two_qubit_gates to total_gates
        steps_taken: Number of optimization steps taken so far
        time_budget_remaining: Remaining time budget in seconds
        category_encoding: One-hot encoding of circuit category
    """

    # From CircuitMetrics
    depth: int
    two_qubit_gates: int
    two_qubit_depth: int
    total_gates: int

    # Circuit features
    num_qubits: int
    gate_density: float
    two_qubit_ratio: float

    # Episode context
    steps_taken: int
    time_budget_remaining: float

    # Category encoding
    category_encoding: list[float] = field(default_factory=lambda: [0.0] * len(CATEGORIES))

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        metrics: CircuitMetrics | None = None,
        category: str = "unknown",
        steps_taken: int = 0,
        time_budget_remaining: float = 300.0,
    ) -> "RLState":
        """Create RLState from a QuantumCircuit.

        Args:
            circuit: The quantum circuit
            metrics: Pre-computed metrics (computed if not provided)
            category: Circuit category for encoding
            steps_taken: Number of optimization steps taken
            time_budget_remaining: Remaining time budget

        Returns:
            RLState instance
        """
        if metrics is None:
            metrics = analyze_circuit(circuit)

        num_qubits = circuit.num_qubits
        gate_density = metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        two_qubit_ratio = (
            metrics.two_qubit_gates / metrics.total_gates
            if metrics.total_gates > 0
            else 0.0
        )

        return cls(
            depth=metrics.depth,
            two_qubit_gates=metrics.two_qubit_gates,
            two_qubit_depth=metrics.two_qubit_depth,
            total_gates=metrics.total_gates,
            num_qubits=num_qubits,
            gate_density=gate_density,
            two_qubit_ratio=two_qubit_ratio,
            steps_taken=steps_taken,
            time_budget_remaining=time_budget_remaining,
            category_encoding=get_category_encoding(category),
        )

    @classmethod
    def from_metrics(
        cls,
        metrics: CircuitMetrics,
        num_qubits: int,
        category: str = "unknown",
        steps_taken: int = 0,
        time_budget_remaining: float = 300.0,
    ) -> "RLState":
        """Create RLState from CircuitMetrics.

        Args:
            metrics: Circuit metrics
            num_qubits: Number of qubits
            category: Circuit category for encoding
            steps_taken: Number of optimization steps taken
            time_budget_remaining: Remaining time budget

        Returns:
            RLState instance
        """
        gate_density = metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        two_qubit_ratio = (
            metrics.two_qubit_gates / metrics.total_gates
            if metrics.total_gates > 0
            else 0.0
        )

        return cls(
            depth=metrics.depth,
            two_qubit_gates=metrics.two_qubit_gates,
            two_qubit_depth=metrics.two_qubit_depth,
            total_gates=metrics.total_gates,
            num_qubits=num_qubits,
            gate_density=gate_density,
            two_qubit_ratio=two_qubit_ratio,
            steps_taken=steps_taken,
            time_budget_remaining=time_budget_remaining,
            category_encoding=get_category_encoding(category),
        )

    def to_vector(self) -> np.ndarray:
        """Convert state to a numpy vector for RL model input.

        Returns:
            1D numpy array of shape (state_dim,)
        """
        return np.array(
            [
                float(self.depth),
                float(self.two_qubit_gates),
                float(self.two_qubit_depth),
                float(self.total_gates),
                float(self.num_qubits),
                self.gate_density,
                self.two_qubit_ratio,
                float(self.steps_taken),
                self.time_budget_remaining,
                # Log-scale gate count (captures 5-order-of-magnitude range)
                float(np.log1p(self.two_qubit_gates)),
                # Parallelism: fraction of 2Q gates that are parallelizable
                float((self.two_qubit_gates - self.two_qubit_depth) / max(self.two_qubit_gates, 1)),
                # Size bucket: 0=small(≤5q), 0.5=medium(6-12q), 1=large(>12q)
                float(0.0 if self.num_qubits <= 5 else 0.5 if self.num_qubits <= 12 else 1.0),
            ]
            + self.category_encoding,
            dtype=np.float32,
        )

    @staticmethod
    def state_dim() -> int:
        """Get the dimension of the state vector.

        Returns:
            State vector dimension
        """
        # 12 base features (9 original + 3 enriched) + category encoding
        return 12 + len(CATEGORIES)

    def with_updated_metrics(
        self,
        new_metrics: CircuitMetrics,
        time_spent: float,
    ) -> "RLState":
        """Create a new state with updated metrics after an optimization step.

        Args:
            new_metrics: New circuit metrics after optimization
            time_spent: Time spent on the optimization step

        Returns:
            New RLState with updated values
        """
        gate_density = (
            new_metrics.total_gates / self.num_qubits if self.num_qubits > 0 else 0.0
        )
        two_qubit_ratio = (
            new_metrics.two_qubit_gates / new_metrics.total_gates
            if new_metrics.total_gates > 0
            else 0.0
        )

        return RLState(
            depth=new_metrics.depth,
            two_qubit_gates=new_metrics.two_qubit_gates,
            two_qubit_depth=new_metrics.two_qubit_depth,
            total_gates=new_metrics.total_gates,
            num_qubits=self.num_qubits,
            gate_density=gate_density,
            two_qubit_ratio=two_qubit_ratio,
            steps_taken=self.steps_taken + 1,
            time_budget_remaining=max(0.0, self.time_budget_remaining - time_spent),
            category_encoding=self.category_encoding.copy(),
        )


def compute_circuit_features(
    circuit: QuantumCircuit,
    metrics: CircuitMetrics | None = None,
) -> dict[str, float]:
    """Compute derived features for a circuit.

    Args:
        circuit: The quantum circuit
        metrics: Pre-computed metrics (computed if not provided)

    Returns:
        Dictionary of feature names to values
    """
    if metrics is None:
        metrics = analyze_circuit(circuit)

    num_qubits = circuit.num_qubits

    return {
        "depth": float(metrics.depth),
        "two_qubit_gates": float(metrics.two_qubit_gates),
        "two_qubit_depth": float(metrics.two_qubit_depth),
        "total_gates": float(metrics.total_gates),
        "num_qubits": float(num_qubits),
        "gate_density": metrics.total_gates / num_qubits if num_qubits > 0 else 0.0,
        "two_qubit_ratio": (
            metrics.two_qubit_gates / metrics.total_gates
            if metrics.total_gates > 0
            else 0.0
        ),
    }


def normalize_state(
    state: np.ndarray,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize state features using z-score normalization.

    Args:
        state: Raw state vector
        means: Feature means (if None, uses defaults)
        stds: Feature standard deviations (if None, uses defaults)

    Returns:
        Normalized state vector
    """
    # Default normalization statistics (can be updated from data)
    if means is None:
        means = np.array(
            [
                50.0,   # depth
                20.0,   # two_qubit_gates
                15.0,   # two_qubit_depth
                100.0,  # total_gates
                10.0,   # num_qubits
                10.0,   # gate_density
                0.2,    # two_qubit_ratio
                1.5,    # steps_taken
                150.0,  # time_budget_remaining
                2.0,    # log1p(two_qubit_gates)
                0.5,    # parallelism ratio
                0.5,    # size bucket
            ]
            + [0.1] * len(CATEGORIES),  # category encoding (roughly uniform)
            dtype=np.float32,
        )

    if stds is None:
        stds = np.array(
            [
                50.0,   # depth
                30.0,   # two_qubit_gates
                20.0,   # two_qubit_depth
                100.0,  # total_gates
                8.0,    # num_qubits
                10.0,   # gate_density
                0.2,    # two_qubit_ratio
                1.0,    # steps_taken
                100.0,  # time_budget_remaining
                1.5,    # log1p(two_qubit_gates)
                0.3,    # parallelism ratio
                0.4,    # size bucket
            ]
            + [0.3] * len(CATEGORIES),  # category encoding
            dtype=np.float32,
        )

    # Avoid division by zero
    stds = np.maximum(stds, 1e-8)

    return (state - means) / stds
