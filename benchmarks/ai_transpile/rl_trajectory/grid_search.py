"""Grid search runner for exhaustive optimizer evaluation.

This module provides infrastructure to run exhaustive grid search across
all circuits and optimizer combinations, recording trajectories for
offline RL training.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

from ..chain_executor import ChainResult, ChainStep, execute_chain
from .database import (
    CircuitRecord,
    OptimizerRecord,
    TrajectoryDatabase,
    TrajectoryStepRecord,
)
from .reward import RewardConfig, compute_all_rewards, compute_improvement_percentage
from .state import RLState

# Available optimizer configurations
OPTIMIZER_CONFIGS: dict[str, dict[str, Any]] = {
    "wisq_rules": {
        "runner_type": "wisq",
        "options": {
            "approx_epsilon": 0,
            "opt_timeout": 120},  # Rules only (no BQSKit)
        "description": "WISQ with rules-only optimization (no resynthesis)",
    },
    "wisq_bqskit": {
        "runner_type": "wisq",
        "options": {"approx_epsilon": 1e-10},  # Enable BQSKit resynthesis
        "description": "WISQ with BQSKit resynthesis",
    },
    "tket": {
        "runner_type": "tket",
        "options": {"gate_set": "IBMN"},
        "description": "TKET FullPeepholeOptimise with IBMN gate set",
    },
    "qiskit_ai": {
        "runner_type": "qiskit_ai",
        "options": {"optimization_levels": [3], "iterations_per_level": 1},
        "description": "Qiskit AI transpiler at optimization level 3",
    },
    "qiskit_standard": {
        "runner_type": "qiskit_standard",
        "options": {"optimization_levels": [3]},
        "description": "Standard Qiskit transpiler at optimization level 3",
    },
}

HARDWARE_OPTIMIZER_CONFIGS: dict[str, dict[str, Any]] = {
    "qiskit_standard_hardware": {
        "runner_type": "qiskit_standard_hardware",
        "options": {"optimization_levels": [3]},
        "description": "Standard Qiskit transpilation mapped to full Fake Torino",
    },
    "qiskit_ai_hardware": {
        "runner_type": "qiskit_ai_hardware",
        "options": {
            "optimization_levels": [1],
            "iterations_per_level": 1,
            "layout_mode": "optimize",
            "ai_optimization_level": 3,
        },
        "description": "Qiskit AI optimization and routing on full Fake Torino",
    },
    "tket_hardware": {
        "runner_type": "tket_hardware",
        "options": {"gate_set": "IBMN"},
        "description": "TKET optimization and routing on full Fake Torino",
    },
    "wisq_rules_hardware": {
        "runner_type": "wisq_hardware",
        "options": {
            "approx_epsilon": 0,
            "opt_timeout": 120,
            "optimization_levels": [3],
        },
        "description": "WISQ rules with mapping preservation on full Fake Torino",
    },
    "wisq_bqskit_hardware": {
        "runner_type": "wisq_hardware",
        "options": {
            "approx_epsilon": 1e-10,
            "optimization_levels": [3],
        },
        "description": "WISQ+BQSKit with mapping preservation on full Fake Torino",
    },
}

ALL_OPTIMIZER_CONFIGS = {**OPTIMIZER_CONFIGS, **HARDWARE_OPTIMIZER_CONFIGS}


@dataclass
class GridSearchConfig:
    """Configuration for grid search experiments.

    Attributes:
        circuit_sources: List of sources to include ("benchpress", "local")
        categories: Circuit categories to include (None = all)
        optimizers: List of optimizer names to use
        max_chain_length: Maximum number of steps in optimization chains
        enable_chain_search: Whether to search over chains (vs single optimizers)
        time_budget: Time budget per episode in seconds
        max_qubits: Maximum number of qubits for circuits
        database_path: Path to the trajectory database
        reward_config: Configuration for reward computation
    """

    circuit_sources: list[str] = field(default_factory=lambda: ["benchpress", "local"])
    categories: list[str] | None = None
    optimizers: list[str] = field(
        default_factory=lambda: list(OPTIMIZER_CONFIGS.keys())
    )
    max_chain_length: int = 3
    enable_chain_search: bool = True
    time_budget: float = 300.0
    max_qubits: int = 20
    database_path: Path = field(default_factory=lambda: Path("data/trajectories.db"))
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class GridSearchProgress:
    """Progress information for grid search."""

    total_circuits: int
    completed_circuits: int
    total_combinations: int
    completed_combinations: int
    current_circuit: str
    current_combination: str
    elapsed_seconds: float


@dataclass
class GridSearchReport:
    """Report summarizing grid search results."""

    total_circuits: int
    total_trajectories: int
    total_steps: int
    total_duration_seconds: float
    failures: list[dict[str, Any]]
    best_by_category: dict[str, dict[str, Any]]


def generate_optimizer_combinations(
    optimizers: Sequence[str],
    max_length: int = 3,
    include_single: bool = True,
) -> list[list[str]]:
    """Generate all optimizer combinations up to max_length.

    Args:
        optimizers: List of optimizer names
        max_length: Maximum chain length
        include_single: Whether to include single-optimizer combinations

    Returns:
        List of optimizer sequences (chains)
    """
    combinations: list[list[str]] = []

    start_length = 1 if include_single else 2
    for length in range(start_length, max_length + 1):
        # Generate all permutations with repetition
        for combo in itertools.product(optimizers, repeat=length):
            combinations.append(list(combo))

    return combinations


def _create_chain_name(optimizer_sequence: Sequence[str]) -> str:
    """Create a unique chain name from optimizer sequence."""
    return "_then_".join(optimizer_sequence)


class GridSearchRunner:
    """Runner for exhaustive grid search over circuits and optimizers."""

    def __init__(
        self,
        config: GridSearchConfig,
        progress_callback: Callable[[GridSearchProgress], None] | None = None,
    ):
        """Initialize the grid search runner.

        Args:
            config: Grid search configuration
            progress_callback: Callback for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback
        self.db = TrajectoryDatabase(config.database_path)
        self._ensure_optimizers_registered()

    def _ensure_optimizers_registered(self) -> None:
        """Ensure all configured optimizers are in the database."""
        for name in self.config.optimizers:
            if name not in OPTIMIZER_CONFIGS:
                raise ValueError(f"Unknown optimizer: {name}")

            opt_config = OPTIMIZER_CONFIGS[name]
            optimizer = OptimizerRecord(
                id=None,
                name=name,
                runner_type=opt_config["runner_type"],
                options=opt_config["options"],
                description=opt_config.get("description"),
            )
            self.db.get_or_create_optimizer(optimizer)

    def _load_circuit(self, circuit_record: CircuitRecord) -> QuantumCircuit | None:
        """Load a circuit from its QASM path."""
        if circuit_record.qasm_path is None:
            return None
        try:
            return qasm2.load(
                circuit_record.qasm_path,
                custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
            )
        except Exception:
            return None

    def run_chain(
        self,
        circuit_record: CircuitRecord,
        optimizer_sequence: Sequence[str],
        output_dir: Path | None = None,
    ) -> ChainResult | None:
        """Run an optimization chain and return results.

        Args:
            circuit_record: Circuit to optimize
            optimizer_sequence: Sequence of optimizer names
            output_dir: Directory to save intermediate results

        Returns:
            ChainResult or None if failed
        """
        circuit = self._load_circuit(circuit_record)
        if circuit is None:
            return None

        steps = []
        for opt_name in optimizer_sequence:
            opt_config = OPTIMIZER_CONFIGS[opt_name]
            steps.append(
                ChainStep(
                    runner_type=opt_config["runner_type"],
                    options=opt_config["options"],
                    name=opt_name,
                )
            )

        chain_name = _create_chain_name(optimizer_sequence)
        out_dir = output_dir or Path(f"/tmp/grid_search/{circuit_record.name}/{chain_name}")

        try:
            return execute_chain(
                circuit,
                steps=steps,
                chain_name=chain_name,
                output_dir=out_dir,
                save_intermediates=False,
            )
        except Exception:
            return None

    def record_trajectory(
        self,
        circuit_record: CircuitRecord,
        chain_result: ChainResult,
        optimizer_sequence: Sequence[str],
    ) -> int:
        """Record a trajectory (chain result) to the database.

        Args:
            circuit_record: Source circuit
            chain_result: Results from chain execution
            optimizer_sequence: Sequence of optimizer names used

        Returns:
            Trajectory ID
        """
        circuit_id = circuit_record.id
        if circuit_id is None:
            raise ValueError("Circuit must have an ID")

        chain_name = _create_chain_name(optimizer_sequence)

        # Check if trajectory already exists
        if self.db.trajectory_exists(circuit_id, chain_name):
            return -1  # Skip duplicate

        # Compute improvement
        improvement = compute_improvement_percentage(
            chain_result.initial_metrics.two_qubit_gates,
            chain_result.final_metrics.two_qubit_gates,
        )

        # Compute total reward
        total_reward = 0.0
        step_rewards = []
        prev_metrics = chain_result.initial_metrics
        initial_metrics = chain_result.initial_metrics

        for i, step_result in enumerate(chain_result.step_results):
            is_final = i == len(chain_result.step_results) - 1
            rewards = compute_all_rewards(
                prev_metrics=prev_metrics,
                new_metrics=step_result.output_metrics,
                time_cost=step_result.duration_seconds,
                initial_metrics=initial_metrics,
                is_final_step=is_final,
                config=self.config.reward_config,
            )
            step_rewards.append(rewards)
            total_reward += rewards.efficiency
            prev_metrics = step_result.output_metrics

        # Insert trajectory
        trajectory_id = self.db.insert_trajectory(
            circuit_id=circuit_id,
            chain_name=chain_name,
            num_steps=len(chain_result.step_results),
            initial_depth=chain_result.initial_metrics.depth,
            initial_two_qubit_gates=chain_result.initial_metrics.two_qubit_gates,
            initial_two_qubit_depth=chain_result.initial_metrics.two_qubit_depth,
            initial_total_gates=chain_result.initial_metrics.total_gates,
            final_depth=chain_result.final_metrics.depth,
            final_two_qubit_gates=chain_result.final_metrics.two_qubit_gates,
            final_two_qubit_depth=chain_result.final_metrics.two_qubit_depth,
            final_total_gates=chain_result.final_metrics.total_gates,
            total_duration_seconds=chain_result.total_duration_seconds,
            total_reward=total_reward,
            improvement_percentage=improvement,
            metadata={"optimizer_sequence": list(optimizer_sequence)},
        )

        # Insert trajectory steps
        prev_metrics = chain_result.initial_metrics
        time_remaining = self.config.time_budget

        for i, step_result in enumerate(chain_result.step_results):
            opt_name = optimizer_sequence[i]
            optimizer = self.db.get_optimizer_by_name(opt_name)
            if optimizer is None:
                continue

            is_final = i == len(chain_result.step_results) - 1
            rewards = step_rewards[i]

            # Build state before action
            state = RLState.from_metrics(
                metrics=prev_metrics,
                num_qubits=circuit_record.num_qubits,
                category=circuit_record.category,
                steps_taken=i,
                time_budget_remaining=time_remaining,
            )

            # Build state after action
            new_time_remaining = time_remaining - step_result.duration_seconds
            next_state = RLState.from_metrics(
                metrics=step_result.output_metrics,
                num_qubits=circuit_record.num_qubits,
                category=circuit_record.category,
                steps_taken=i + 1,
                time_budget_remaining=new_time_remaining,
            )

            step_record = TrajectoryStepRecord(
                trajectory_id=trajectory_id,
                step_index=i,
                optimizer_id=optimizer.id or 0,
                # State s
                state_depth=state.depth,
                state_two_qubit_gates=state.two_qubit_gates,
                state_two_qubit_depth=state.two_qubit_depth,
                state_total_gates=state.total_gates,
                state_num_qubits=state.num_qubits,
                state_gate_density=state.gate_density,
                state_two_qubit_ratio=state.two_qubit_ratio,
                state_steps_taken=state.steps_taken,
                state_time_budget_remaining=state.time_budget_remaining,
                state_category=state.category_encoding,
                # Next state s'
                next_state_depth=next_state.depth,
                next_state_two_qubit_gates=next_state.two_qubit_gates,
                next_state_two_qubit_depth=next_state.two_qubit_depth,
                next_state_total_gates=next_state.total_gates,
                next_state_gate_density=next_state.gate_density,
                next_state_two_qubit_ratio=next_state.two_qubit_ratio,
                next_state_steps_taken=next_state.steps_taken,
                next_state_time_budget_remaining=next_state.time_budget_remaining,
                # Rewards
                reward_improvement_only=rewards.improvement_only,
                reward_efficiency=rewards.efficiency,
                reward_multi_objective=rewards.multi_objective,
                reward_sparse_final=rewards.sparse_final,
                # Episode info
                done=is_final,
                duration_seconds=step_result.duration_seconds,
            )

            self.db.insert_trajectory_step(step_record)

            # Update for next iteration
            prev_metrics = step_result.output_metrics
            time_remaining = new_time_remaining

        return trajectory_id

    def run_exhaustive_search(
        self,
        resume: bool = True,
    ) -> GridSearchReport:
        """Run exhaustive grid search across all circuits and optimizer combinations.

        Args:
            resume: Skip already-recorded trajectories

        Returns:
            GridSearchReport with results summary
        """
        start_time = time.perf_counter()
        failures: list[dict[str, Any]] = []
        best_by_category: dict[str, dict[str, Any]] = {}

        # Get circuits from database
        circuits = self.db.list_circuits(max_qubits=self.config.max_qubits)
        if self.config.categories:
            circuits = [c for c in circuits if c.category in self.config.categories]

        if not circuits:
            return GridSearchReport(
                total_circuits=0,
                total_trajectories=0,
                total_steps=0,
                total_duration_seconds=0.0,
                failures=[],
                best_by_category={},
            )

        # Generate optimizer combinations
        if self.config.enable_chain_search:
            combinations = generate_optimizer_combinations(
                self.config.optimizers,
                max_length=self.config.max_chain_length,
            )
        else:
            combinations = [[opt] for opt in self.config.optimizers]

        total_circuits = len(circuits)
        total_combinations = len(combinations)
        completed_circuits = 0
        completed_combinations = 0
        total_trajectories = 0
        total_steps = 0

        for circuit in circuits:
            circuit_id = circuit.id
            if circuit_id is None:
                continue

            for combo in combinations:
                chain_name = _create_chain_name(combo)

                # Update progress
                if self.progress_callback:
                    elapsed = time.perf_counter() - start_time
                    self.progress_callback(
                        GridSearchProgress(
                            total_circuits=total_circuits,
                            completed_circuits=completed_circuits,
                            total_combinations=total_combinations * total_circuits,
                            completed_combinations=completed_combinations,
                            current_circuit=circuit.name,
                            current_combination=chain_name,
                            elapsed_seconds=elapsed,
                        )
                    )

                # Skip if already exists
                if resume and self.db.trajectory_exists(circuit_id, chain_name):
                    completed_combinations += 1
                    continue

                # Run chain
                chain_result = self.run_chain(circuit, combo)
                if chain_result is None:
                    failures.append(
                        {
                            "circuit": circuit.name,
                            "chain": chain_name,
                            "error": "Chain execution failed",
                        }
                    )
                    completed_combinations += 1
                    continue

                # Record trajectory
                trajectory_id = self.record_trajectory(circuit, chain_result, combo)
                if trajectory_id > 0:
                    total_trajectories += 1
                    total_steps += len(chain_result.step_results)

                    # Track best by category
                    improvement = compute_improvement_percentage(
                        chain_result.initial_metrics.two_qubit_gates,
                        chain_result.final_metrics.two_qubit_gates,
                    )

                    if circuit.category not in best_by_category:
                        best_by_category[circuit.category] = {
                            "chain": chain_name,
                            "circuit": circuit.name,
                            "improvement": improvement,
                        }
                    elif improvement > best_by_category[circuit.category]["improvement"]:
                        best_by_category[circuit.category] = {
                            "chain": chain_name,
                            "circuit": circuit.name,
                            "improvement": improvement,
                        }

                completed_combinations += 1

            completed_circuits += 1

        total_duration = time.perf_counter() - start_time

        return GridSearchReport(
            total_circuits=total_circuits,
            total_trajectories=total_trajectories,
            total_steps=total_steps,
            total_duration_seconds=total_duration,
            failures=failures,
            best_by_category=best_by_category,
        )

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    def __enter__(self) -> "GridSearchRunner":
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()


def run_quick_grid_search(
    database_path: Path | str,
    categories: Sequence[str] | None = None,
    optimizers: Sequence[str] | None = None,
    max_chain_length: int = 2,
    max_qubits: int = 10,
    progress_callback: Callable[[GridSearchProgress], None] | None = None,
) -> GridSearchReport:
    """Run a quick grid search with default settings.

    This is a convenience function for testing and quick experiments.

    Args:
        database_path: Path to trajectory database
        categories: Categories to include (None = all)
        optimizers: Optimizers to use (None = all)
        max_chain_length: Maximum chain length
        max_qubits: Maximum qubit count
        progress_callback: Progress callback

    Returns:
        GridSearchReport
    """
    config = GridSearchConfig(
        categories=list(categories) if categories else None,
        optimizers=list(optimizers) if optimizers else list(OPTIMIZER_CONFIGS.keys()),
        max_chain_length=max_chain_length,
        max_qubits=max_qubits,
        database_path=Path(database_path),
    )

    with GridSearchRunner(config, progress_callback=progress_callback) as runner:
        return runner.run_exhaustive_search()
