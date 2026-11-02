"""RL-based optimization orchestration (prototype).

This module implements a basic RL environment for learning to orchestrate
quantum circuit optimization techniques. It demonstrates the feasibility of
the approach described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
from qiskit.circuit import QuantumCircuit

from .runtime_profiler import Timer
from .transpilers import CircuitMetrics, analyze_circuit


class OptimizationAction(Enum):
    """Available optimization actions."""

    RULES_ONLY = "rules_only"  # Fast rule-based optimization
    RESYNTHESIS = "resynthesis"  # Slower unitary resynthesis
    RL_SYNTHESIS = "rl_synthesis"  # RL-based synthesis
    TKET_OPTIMIZE = "tket_optimize"  # TKET optimization
    END_EPISODE = "end_episode"  # Terminate optimization


@dataclass
class OptimizationState:
    """State representation for the RL environment."""

    circuit_metrics: CircuitMetrics
    time_budget_remaining: float
    steps_taken: int
    max_steps: int
    previous_metric: float


@dataclass
class OptimizationConfig:
    """Configuration for RL orchestration."""

    time_budget: float = 300.0  # Maximum time in seconds
    max_steps: int = 10  # Maximum optimization steps
    target_metric: str = "two_qubit_gates"  # Metric to optimize
    alpha: float = 1.0  # Reward coefficient for improvement
    beta: float = 0.1  # Penalty coefficient for time cost
    gamma: float = 0.01  # Existential tax per step


class OptimizerInterface(Protocol):
    """Protocol for optimization technique implementations."""

    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply optimization to a circuit."""
        ...

    def estimate_cost(self, circuit: QuantumCircuit) -> float:
        """Estimate runtime cost in seconds."""
        ...


@dataclass
class MockOptimizer:
    """Mock optimizer for testing (replaces real optimizers)."""

    name: str
    improvement_range: tuple[float, float] = (0.05, 0.15)
    cost_range: tuple[float, float] = (1.0, 5.0)

    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Mock optimization (returns original circuit for prototype)."""
        return circuit

    def estimate_cost(self, circuit: QuantumCircuit) -> float:
        """Estimate cost (random for prototype)."""
        return float(np.random.uniform(*self.cost_range))


class OptimizationEnvironment:
    """Gym-style environment for RL-based optimization orchestration."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        config: OptimizationConfig | None = None,
        optimizers: dict[OptimizationAction, OptimizerInterface] | None = None,
    ):
        """Initialize the environment.

        Args:
            circuit: Initial quantum circuit
            config: Orchestration configuration
            optimizers: Dictionary mapping actions to optimizer implementations
        """
        self.initial_circuit = circuit
        self.config = config or OptimizationConfig()
        self.current_circuit = circuit
        self.current_metrics = analyze_circuit(circuit)

        # Use mock optimizers if none provided
        self.optimizers = optimizers or {
            OptimizationAction.RULES_ONLY: MockOptimizer("rules", (0.05, 0.10), (0.5, 2.0)),
            OptimizationAction.RESYNTHESIS: MockOptimizer("resynth", (0.10, 0.20), (5.0, 15.0)),
            OptimizationAction.RL_SYNTHESIS: MockOptimizer("rl_synth", (0.08, 0.18), (2.0, 8.0)),
            OptimizationAction.TKET_OPTIMIZE: MockOptimizer("tket", (0.06, 0.12), (1.0, 4.0)),
        }

        self.state = OptimizationState(
            circuit_metrics=self.current_metrics,
            time_budget_remaining=self.config.time_budget,
            steps_taken=0,
            max_steps=self.config.max_steps,
            previous_metric=float(getattr(self.current_metrics, self.config.target_metric)),
        )

        self.done = False
        self.total_reward = 0.0
        self.episode_history: list[dict[str, Any]] = []

    def reset(self) -> OptimizationState:
        """Reset the environment to initial state.

        Returns:
            Initial state
        """
        self.current_circuit = self.initial_circuit
        self.current_metrics = analyze_circuit(self.initial_circuit)
        self.state = OptimizationState(
            circuit_metrics=self.current_metrics,
            time_budget_remaining=self.config.time_budget,
            steps_taken=0,
            max_steps=self.config.max_steps,
            previous_metric=float(getattr(self.current_metrics, self.config.target_metric)),
        )
        self.done = False
        self.total_reward = 0.0
        self.episode_history = []

        return self.state

    def step(self, action: OptimizationAction) -> tuple[OptimizationState, float, bool, dict[str, Any]]:
        """Execute one optimization step.

        Args:
            action: Optimization action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Handle end episode action
        if action == OptimizationAction.END_EPISODE:
            self.done = True
            return self.state, 0.0, True, {"action": "end_episode"}

        # Apply optimization
        optimizer = self.optimizers.get(action)
        if optimizer is None:
            raise ValueError(f"No optimizer registered for action {action}")

        timer = Timer()
        timer.start()
        optimized_circuit = optimizer.optimize(self.current_circuit)
        timer.stop()

        time_cost = timer.elapsed

        # Update circuit and metrics
        self.current_circuit = optimized_circuit
        new_metrics = analyze_circuit(optimized_circuit)
        current_metric_value = float(getattr(new_metrics, self.config.target_metric))

        # Compute reward using the paper's formula:
        # R_t = α * (M_{t-1} - M_t) / M_{t-1} - β * TimeCost - γ
        if self.state.previous_metric > 0:
            relative_improvement = (self.state.previous_metric - current_metric_value) / self.state.previous_metric
        else:
            relative_improvement = 0.0

        reward = (
            self.config.alpha * relative_improvement - self.config.beta * time_cost - self.config.gamma
        )

        # Update state
        self.state = OptimizationState(
            circuit_metrics=new_metrics,
            time_budget_remaining=self.state.time_budget_remaining - time_cost,
            steps_taken=self.state.steps_taken + 1,
            max_steps=self.config.max_steps,
            previous_metric=current_metric_value,
        )

        # Check termination conditions
        self.done = (
            self.state.time_budget_remaining <= 0
            or self.state.steps_taken >= self.config.max_steps
        )

        self.total_reward += reward

        # Record history
        info = {
            "action": action.value,
            "reward": reward,
            "time_cost": time_cost,
            "improvement": relative_improvement,
            "metric_value": current_metric_value,
            "steps": self.state.steps_taken,
        }
        self.episode_history.append(info)

        return self.state, reward, self.done, info

    def get_available_actions(self) -> list[OptimizationAction]:
        """Get list of actions available in current state.

        Returns:
            List of available actions
        """
        # All actions available unless time budget exhausted
        actions = [
            OptimizationAction.RULES_ONLY,
            OptimizationAction.RESYNTHESIS,
            OptimizationAction.RL_SYNTHESIS,
            OptimizationAction.TKET_OPTIMIZE,
            OptimizationAction.END_EPISODE,
        ]

        # Filter based on time budget
        available = []
        for action in actions:
            if action == OptimizationAction.END_EPISODE:
                available.append(action)
            else:
                optimizer = self.optimizers.get(action)
                if optimizer:
                    estimated_cost = optimizer.estimate_cost(self.current_circuit)
                    if estimated_cost <= self.state.time_budget_remaining:
                        available.append(action)

        return available


class RandomPolicy:
    """Random baseline policy."""

    def __init__(self, seed: int | None = None):
        """Initialize random policy.

        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

    def select_action(
        self, state: OptimizationState, available_actions: list[OptimizationAction]
    ) -> OptimizationAction:
        """Select random action.

        Args:
            state: Current state
            available_actions: List of available actions

        Returns:
            Selected action
        """
        return np.random.choice(available_actions)


class GreedyPolicy:
    """Greedy policy: always picks fastest optimizer."""

    def select_action(
        self, state: OptimizationState, available_actions: list[OptimizationAction]
    ) -> OptimizationAction:
        """Select fastest available action.

        Args:
            state: Current state
            available_actions: List of available actions

        Returns:
            Selected action
        """
        # Prioritize rules-only (fastest)
        if OptimizationAction.RULES_ONLY in available_actions:
            return OptimizationAction.RULES_ONLY
        # Then TKET
        if OptimizationAction.TKET_OPTIMIZE in available_actions:
            return OptimizationAction.TKET_OPTIMIZE
        # Then RL synthesis
        if OptimizationAction.RL_SYNTHESIS in available_actions:
            return OptimizationAction.RL_SYNTHESIS
        # Then resynthesis
        if OptimizationAction.RESYNTHESIS in available_actions:
            return OptimizationAction.RESYNTHESIS
        # Finally end
        return OptimizationAction.END_EPISODE


class FixedSchedulePolicy:
    """Fixed schedule: IBM → Rules → Resynthesis."""

    def __init__(self):
        """Initialize fixed schedule policy."""
        self.schedule = [
            OptimizationAction.RULES_ONLY,
            OptimizationAction.RESYNTHESIS,
            OptimizationAction.END_EPISODE,
        ]
        self.step = 0

    def select_action(
        self, state: OptimizationState, available_actions: list[OptimizationAction]
    ) -> OptimizationAction:
        """Follow fixed schedule.

        Args:
            state: Current state
            available_actions: List of available actions

        Returns:
            Selected action
        """
        if self.step < len(self.schedule):
            action = self.schedule[self.step]
            self.step += 1
            if action in available_actions:
                return action

        return OptimizationAction.END_EPISODE

    def reset(self) -> None:
        """Reset policy for new episode."""
        self.step = 0


def evaluate_policy(
    policy: RandomPolicy | GreedyPolicy | FixedSchedulePolicy,
    circuit: QuantumCircuit,
    config: OptimizationConfig | None = None,
    num_episodes: int = 10,
) -> dict[str, Any]:
    """Evaluate a policy on a circuit.

    Args:
        policy: Policy to evaluate
        circuit: Initial circuit
        config: Environment configuration
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation results
    """
    results = {
        "total_rewards": [],
        "final_metrics": [],
        "steps_taken": [],
        "time_used": [],
    }

    for episode in range(num_episodes):
        env = OptimizationEnvironment(circuit, config)
        state = env.reset()

        if hasattr(policy, "reset"):
            policy.reset()  # type: ignore[attr-defined]

        while not env.done:
            available_actions = env.get_available_actions()
            action = policy.select_action(state, available_actions)
            state, reward, done, info = env.step(action)

        results["total_rewards"].append(env.total_reward)
        target_metric = config.target_metric if config else "two_qubit_gates"
        results["final_metrics"].append(getattr(state.circuit_metrics, target_metric))
        results["steps_taken"].append(state.steps_taken)
        results["time_used"].append(env.config.time_budget - state.time_budget_remaining)

    # Compute statistics
    results["mean_reward"] = float(np.mean(results["total_rewards"]))
    results["std_reward"] = float(np.std(results["total_rewards"]))
    results["mean_final_metric"] = float(np.mean(results["final_metrics"]))
    results["mean_steps"] = float(np.mean(results["steps_taken"]))

    return results

