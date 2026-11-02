"""Chain executor for sequential quantum circuit optimization.

This module provides infrastructure to run multiple quantum circuit optimizers
in sequence (chains) and track intermediate results at each step.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

from .hardware_aware_transpilers import (
    load_fake_torino_target,
    run_tket_hardware,
    run_wisq_hardware,
    transpile_with_qiskit_ai_hardware,
    transpile_with_qiskit_standard_hardware,
)
from .transpilers import (
    CircuitMetrics,
    QiskitAIRunnerConfig,
    QiskitStandardConfig,
    TKETConfig,
    TranspiledCircuit,
    VOQCConfig,
    WisqConfig,
    analyze_circuit,
    run_tket,
    run_voqc,
    run_wisq_opt,
    transpile_with_qiskit_ai,
    transpile_with_qiskit_standard,
)


@dataclass(frozen=True)
class ChainStep:
    """Represents a single optimization step in a chain.

    Attributes:
        runner_type: Type of optimizer to use ("wisq", "tket", "qiskit_ai",
                     "qiskit_standard", "voqc")
        options: Runner-specific configuration options
        name: Optional name for this step (defaults to runner_type)
    """

    runner_type: str
    options: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None

    @property
    def step_name(self) -> str:
        """Get the name for this step."""
        return self.name if self.name else self.runner_type


@dataclass
class StepResult:
    """Result of executing a single chain step.

    Attributes:
        step: The chain step that was executed
        step_index: Index of this step in the chain (0-based)
        input_metrics: Metrics of the circuit before this step
        output_metrics: Metrics of the circuit after this step
        transpiled: The TranspiledCircuit result from the optimizer
        duration_seconds: Time taken for this step
        artifact_path: Path to the saved QASM file for this step's output
    """

    step: ChainStep
    step_index: int
    input_metrics: CircuitMetrics
    output_metrics: CircuitMetrics
    transpiled: TranspiledCircuit
    duration_seconds: float
    artifact_path: Path | None = None


@dataclass
class ChainResult:
    """Result of executing a complete optimization chain.

    Attributes:
        chain_name: Name identifying this chain
        steps: List of chain steps that were executed
        step_results: Results from each step in order
        initial_circuit: The original input circuit
        initial_metrics: Metrics of the original circuit
        final_circuit: The circuit after all optimizations
        final_metrics: Metrics of the final circuit
        total_duration_seconds: Total time for the entire chain
        metadata: Additional metadata about the chain execution
    """

    chain_name: str
    steps: Sequence[ChainStep]
    step_results: list[StepResult]
    initial_circuit: QuantumCircuit
    initial_metrics: CircuitMetrics
    final_circuit: QuantumCircuit
    final_metrics: CircuitMetrics
    total_duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def intermediate_circuits(self) -> list[QuantumCircuit]:
        """Get all intermediate circuits (after each step)."""
        return [sr.transpiled.circuit for sr in self.step_results]

    @property
    def intermediate_metrics(self) -> list[CircuitMetrics]:
        """Get metrics after each step."""
        return [sr.output_metrics for sr in self.step_results]

    def improvement_percentage(self, metric: str = "two_qubit_gates") -> float:
        """Calculate total improvement percentage for a metric.

        Args:
            metric: Name of the metric to compare

        Returns:
            Percentage improvement (positive means reduction)
        """
        initial = getattr(self.initial_metrics, metric)
        final = getattr(self.final_metrics, metric)
        if initial == 0:
            return 0.0
        return 100.0 * (initial - final) / initial

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary for serialization."""
        return {
            "chain_name": self.chain_name,
            "steps": [
                {"runner_type": s.runner_type, "options": dict(s.options), "name": s.step_name}
                for s in self.steps
            ],
            "initial_metrics": {
                "depth": self.initial_metrics.depth,
                "two_qubit_gates": self.initial_metrics.two_qubit_gates,
                "two_qubit_depth": self.initial_metrics.two_qubit_depth,
                "total_gates": self.initial_metrics.total_gates,
            },
            "final_metrics": {
                "depth": self.final_metrics.depth,
                "two_qubit_gates": self.final_metrics.two_qubit_gates,
                "two_qubit_depth": self.final_metrics.two_qubit_depth,
                "total_gates": self.final_metrics.total_gates,
            },
            "step_results": [
                {
                    "step_name": sr.step.step_name,
                    "step_index": sr.step_index,
                    "input_metrics": {
                        "depth": sr.input_metrics.depth,
                        "two_qubit_gates": sr.input_metrics.two_qubit_gates,
                        "two_qubit_depth": sr.input_metrics.two_qubit_depth,
                        "total_gates": sr.input_metrics.total_gates,
                    },
                    "output_metrics": {
                        "depth": sr.output_metrics.depth,
                        "two_qubit_gates": sr.output_metrics.two_qubit_gates,
                        "two_qubit_depth": sr.output_metrics.two_qubit_depth,
                        "total_gates": sr.output_metrics.total_gates,
                    },
                    "duration_seconds": sr.duration_seconds,
                    "artifact_path": str(sr.artifact_path) if sr.artifact_path else None,
                }
                for sr in self.step_results
            ],
            "total_duration_seconds": self.total_duration_seconds,
            "metadata": self.metadata,
        }


def _execute_wisq_step(
    circuit_path: Path,
    options: Mapping[str, Any],
    output_dir: Path,
    step_name: str,
) -> TranspiledCircuit:
    """Execute a WISQ optimization step.

    Args:
        circuit_path: Path to the input QASM file
        options: WISQ configuration options
        output_dir: Directory to save output files
        step_name: Name for this step (used in output filename)

    Returns:
        TranspiledCircuit result
    """
    config = WisqConfig(
        target_gateset=str(options.get("target_gateset", "IBMN")),
        objective=str(options.get("optimization_objective", "TWO_Q")),
        timeout_seconds=int(options.get("opt_timeout", 600)),
        approximation_epsilon=float(options.get("approx_epsilon", 1e-10)),
        output_dir=output_dir,
        job_info=step_name,
        advanced_args=options.get("advanced_args"),
    )
    return run_wisq_opt(circuit_path, config=config)


def _execute_tket_step(
    circuit_path: Path,
    options: Mapping[str, Any],
    output_dir: Path,
    step_name: str,
) -> TranspiledCircuit:
    """Execute a TKET optimization step.

    Args:
        circuit_path: Path to the input QASM file
        options: TKET configuration options
        output_dir: Directory to save output files
        step_name: Name for this step (used in output filename)

    Returns:
        TranspiledCircuit result
    """
    config = TKETConfig(
        gate_set=str(options.get("gate_set", "IBMN")),
        output_dir=output_dir,
        job_info=step_name,
    )
    return run_tket(circuit_path, config=config)


def _execute_voqc_step(
    circuit_path: Path,
    options: Mapping[str, Any],
    output_dir: Path,
    step_name: str,
) -> TranspiledCircuit:
    """Execute a VOQC optimization step.

    Args:
        circuit_path: Path to the input QASM file
        options: VOQC configuration options
        output_dir: Directory to save output files
        step_name: Name for this step (used in output filename)

    Returns:
        TranspiledCircuit result
    """
    config = VOQCConfig(
        optimization_method=str(options.get("optimization_method", "nam")),
        output_dir=output_dir,
        job_info=step_name,
    )
    return run_voqc(circuit_path, config=config)


def _execute_qiskit_ai_step(
    circuit: QuantumCircuit,
    options: Mapping[str, Any],
) -> TranspiledCircuit:
    """Execute a Qiskit AI optimization step.

    Args:
        circuit: Input quantum circuit
        options: Qiskit AI configuration options

    Returns:
        TranspiledCircuit result (best variant)
    """
    opt_levels = options.get("optimization_levels", (3,))
    if isinstance(opt_levels, int):
        opt_levels = (opt_levels,)

    config = QiskitAIRunnerConfig(
        optimization_levels=tuple(opt_levels),
        iterations_per_level=int(options.get("iterations_per_level", 1)),
        layout_mode=str(options.get("layout_mode", "optimize")),
    )
    results = transpile_with_qiskit_ai(circuit.copy(), config=config)

    # Return the best result (lowest 2-qubit gate count)
    best = min(results, key=lambda r: r.metrics.two_qubit_gates)
    return best


def _execute_qiskit_standard_step(
    circuit: QuantumCircuit,
    options: Mapping[str, Any],
) -> TranspiledCircuit:
    """Execute a standard Qiskit transpiler step.

    Args:
        circuit: Input quantum circuit
        options: Qiskit standard configuration options

    Returns:
        TranspiledCircuit result (best variant)
    """
    opt_levels = options.get("optimization_levels", (3,))
    if isinstance(opt_levels, int):
        opt_levels = (opt_levels,)

    config = QiskitStandardConfig(
        optimization_levels=tuple(opt_levels),
    )
    results = transpile_with_qiskit_standard(circuit.copy(), config=config)

    # Return the best result (lowest 2-qubit gate count)
    best = min(results, key=lambda r: r.metrics.two_qubit_gates)
    return best


def _execute_hardware_step(
    circuit: QuantumCircuit,
    step: ChainStep,
    output_dir: Path,
    step_name: str,
) -> TranspiledCircuit:
    options = step.options
    hardware = load_fake_torino_target()
    opt_levels = options.get("optimization_levels", (3,))
    if isinstance(opt_levels, int):
        opt_levels = (opt_levels,)

    if step.runner_type == "qiskit_standard_hardware":
        return transpile_with_qiskit_standard_hardware(
            circuit.copy(),
            hardware,
            optimization_level=int(tuple(opt_levels)[0]),
            seed_transpiler=int(options.get("seed_transpiler", 0)),
        )

    if step.runner_type == "qiskit_ai_hardware":
        return transpile_with_qiskit_ai_hardware(
            circuit.copy(),
            hardware,
            optimization_level=int(tuple(opt_levels)[0]),
            ai_optimization_level=int(options.get("ai_optimization_level", 3)),
            layout_mode=str(options.get("layout_mode", "optimize")),
            seed_transpiler=int(options.get("seed_transpiler", 0)),
        )

    temp_path = _save_circuit_to_temp(circuit)
    try:
        if step.runner_type == "tket_hardware":
            return run_tket_hardware(
                temp_path,
                hardware,
                output_dir=output_dir,
                job_info=step_name,
                gate_set=str(options.get("gate_set", "IBMN")),
            )
        if step.runner_type == "wisq_hardware":
            return run_wisq_hardware(
                temp_path,
                hardware,
                output_dir=output_dir,
                job_info=step_name,
                approximation_epsilon=float(options.get("approx_epsilon", 1e-10)),
                timeout_seconds=int(options.get("opt_timeout", 600)),
                objective=str(options.get("optimization_objective", "TWO_Q")),
                advanced_args=options.get("advanced_args"),
                optimization_level=int(tuple(opt_levels)[0]),
                seed_transpiler=int(options.get("seed_transpiler", 0)),
            )
    finally:
        temp_path.unlink(missing_ok=True)

    raise ValueError(f"Unknown hardware-aware runner type: {step.runner_type}")


def _save_circuit_to_temp(circuit: QuantumCircuit, suffix: str = ".qasm") -> Path:
    """Save a circuit to a temporary QASM file.

    Args:
        circuit: Circuit to save
        suffix: File suffix

    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        # Decompose swap gates for compatibility
        decomposed = circuit.decompose(gates_to_decompose=["swap"])
        f.write(qasm2.dumps(decomposed))
        return Path(f.name)


def execute_chain(
    circuit: QuantumCircuit | Path,
    steps: Sequence[ChainStep],
    chain_name: str = "chain",
    output_dir: Path | None = None,
    save_intermediates: bool = True,
) -> ChainResult:
    """Execute a chain of optimizers sequentially.

    Args:
        circuit: Input circuit (QuantumCircuit or path to QASM file)
        steps: Sequence of optimization steps to apply
        chain_name: Name for this chain execution
        output_dir: Directory to save intermediate results (default: temp dir)
        save_intermediates: Whether to save intermediate QASM files

    Returns:
        ChainResult with all intermediate and final results

    Raises:
        ValueError: If steps is empty or circuit is invalid
    """
    if not steps:
        raise ValueError("Chain must have at least one step")

    # Load circuit if path provided
    if isinstance(circuit, Path):
        initial_circuit = qasm2.loads(circuit.read_text(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    else:
        initial_circuit = circuit

    initial_metrics = analyze_circuit(initial_circuit)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="chain_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track execution
    step_results: list[StepResult] = []
    current_circuit = initial_circuit
    current_metrics = initial_metrics
    total_start = time.perf_counter()

    for idx, step in enumerate(steps):
        step_name = f"{chain_name}_step{idx}_{step.step_name}"
        step_output_dir = output_dir / f"step_{idx}_{step.step_name}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        input_metrics = current_metrics
        step_start = time.perf_counter()

        # Execute based on runner type
        if step.runner_type in ("wisq", "wisq_rules", "wisq_bqskit"):
            # File-based runner - need to save circuit first
            temp_path = _save_circuit_to_temp(current_circuit)
            try:
                result = _execute_wisq_step(
                    temp_path, step.options, step_output_dir, step_name
                )
            finally:
                temp_path.unlink(missing_ok=True)

        elif step.runner_type == "tket":
            temp_path = _save_circuit_to_temp(current_circuit)
            try:
                result = _execute_tket_step(
                    temp_path, step.options, step_output_dir, step_name
                )
            finally:
                temp_path.unlink(missing_ok=True)

        elif step.runner_type == "voqc":
            temp_path = _save_circuit_to_temp(current_circuit)
            try:
                result = _execute_voqc_step(
                    temp_path, step.options, step_output_dir, step_name
                )
            finally:
                temp_path.unlink(missing_ok=True)

        elif step.runner_type == "qiskit_ai":
            result = _execute_qiskit_ai_step(current_circuit, step.options)

        elif step.runner_type == "qiskit_standard":
            result = _execute_qiskit_standard_step(current_circuit, step.options)

        elif step.runner_type in {
            "qiskit_standard_hardware",
            "qiskit_ai_hardware",
            "tket_hardware",
            "wisq_hardware",
        }:
            result = _execute_hardware_step(
                current_circuit,
                step,
                step_output_dir,
                step_name,
            )

        else:
            raise ValueError(f"Unknown runner type: {step.runner_type}")

        step_duration = time.perf_counter() - step_start

        # Save intermediate if requested
        artifact_path: Path | None = None
        if save_intermediates:
            artifact_path = step_output_dir / f"{step_name}.qasm"
            decomposed = result.circuit.decompose(gates_to_decompose=["swap"])
            artifact_path.write_text(qasm2.dumps(decomposed))

        # Record step result
        step_result = StepResult(
            step=step,
            step_index=idx,
            input_metrics=input_metrics,
            output_metrics=result.metrics,
            transpiled=result,
            duration_seconds=step_duration,
            artifact_path=artifact_path,
        )
        step_results.append(step_result)

        # Update for next iteration
        current_circuit = result.circuit
        current_metrics = result.metrics

    total_duration = time.perf_counter() - total_start

    return ChainResult(
        chain_name=chain_name,
        steps=list(steps),
        step_results=step_results,
        initial_circuit=initial_circuit,
        initial_metrics=initial_metrics,
        final_circuit=current_circuit,
        final_metrics=current_metrics,
        total_duration_seconds=total_duration,
        metadata={
            "output_dir": str(output_dir),
            "save_intermediates": save_intermediates,
        },
    )


def create_chain_from_config(
    steps_config: Sequence[Mapping[str, Any]],
) -> list[ChainStep]:
    """Create a list of ChainStep objects from configuration dictionaries.

    Args:
        steps_config: List of step configuration dictionaries, each with
                      'type' (required) and optional 'name' and other options

    Returns:
        List of ChainStep objects

    Example:
        >>> config = [
        ...     {"type": "wisq", "approx_epsilon": 0},
        ...     {"type": "tket", "gate_set": "IBMN"},
        ... ]
        >>> steps = create_chain_from_config(config)
    """
    steps: list[ChainStep] = []
    for step_cfg in steps_config:
        runner_type = step_cfg.get("type")
        if not runner_type:
            raise ValueError("Each step must have a 'type' field")

        name = step_cfg.get("name")
        options = {k: v for k, v in step_cfg.items() if k not in ("type", "name")}

        steps.append(ChainStep(runner_type=runner_type, options=options, name=name))

    return steps
