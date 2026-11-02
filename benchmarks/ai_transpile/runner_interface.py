"""
Abstract runner interface for circuit optimization transpilers.

This module provides a common interface for all circuit transpilers/optimizers,
making it easier to add new transpilers and ensure consistent behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from qiskit.circuit import QuantumCircuit

from .transpilers import TranspiledCircuit, analyze_circuit


@dataclass(frozen=True)
class RunnerConfig:
    """Base configuration for all runners."""

    output_dir: Path = field(default_factory=lambda: Path("reports/runner_output"))
    job_info: str = "runner"

    def output_file_for(self, circuit_path: Path) -> Path:
        """Generate output file path for a circuit."""
        file_name = f"{circuit_path.stem}_{self.job_info}.qasm"
        return self.output_dir / file_name


class CircuitRunner(ABC):
    """
    Abstract base class for circuit transpilers/optimizers.

    All runner implementations should inherit from this class and implement
    the required methods to ensure consistent behavior across different
    optimization backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the runner (e.g., 'qiskit_ai', 'wisq', 'tket')."""
        ...

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit | Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """
        Run the optimizer on a circuit.

        Args:
            circuit: Either a QuantumCircuit object or a Path to a QASM file.
            config: Optional runner-specific configuration.

        Returns:
            List of TranspiledCircuit results (may contain multiple variants).

        Raises:
            ImportError: If required dependencies are not available.
            ValueError: If the circuit is invalid.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this runner is available (all dependencies installed).

        Returns:
            True if the runner can be used, False otherwise.
        """
        ...

    def get_availability_error(self) -> str | None:
        """
        Get a human-readable error message if the runner is not available.

        Returns:
            Error message explaining why the runner is unavailable, or None if available.
        """
        return None if self.is_available() else f"{self.name} runner is not available."

    @staticmethod
    def _analyze_and_create_result(
        circuit: QuantumCircuit,
        optimizer: str,
        label: str,
        artifact_path: Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TranspiledCircuit:
        """
        Helper to create a TranspiledCircuit with computed metrics.

        Args:
            circuit: The transpiled/optimized circuit.
            optimizer: Name of the optimizer that produced this result.
            label: Label for this specific variant.
            artifact_path: Optional path to saved artifact.
            metadata: Optional additional metadata.

        Returns:
            TranspiledCircuit with computed metrics.
        """
        return TranspiledCircuit(
            optimizer=optimizer,
            label=label,
            circuit=circuit,
            metrics=analyze_circuit(circuit),
            artifact_path=artifact_path,
            metadata=metadata or {},
        )


class FileBasedRunner(CircuitRunner):
    """
    Base class for runners that operate on QASM files.

    Some optimizers (like WISQ, external TKET) work with file paths rather than
    QuantumCircuit objects. This base class provides common functionality for
    file-based runners.
    """

    def run(
        self,
        circuit: QuantumCircuit | Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """
        Run the optimizer on a circuit.

        If a QuantumCircuit is provided, it will be saved to a temporary file first.

        Args:
            circuit: Either a QuantumCircuit object or a Path to a QASM file.
            config: Optional runner-specific configuration.

        Returns:
            List of TranspiledCircuit results.
        """
        if isinstance(circuit, QuantumCircuit):
            return self._run_from_circuit(circuit, config)
        return self._run_from_path(circuit, config)

    @abstractmethod
    def _run_from_path(
        self,
        circuit_path: Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """Run optimization on a circuit from a QASM file path."""
        ...

    def _run_from_circuit(
        self,
        circuit: QuantumCircuit,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """
        Run optimization on a QuantumCircuit by saving it to a temp file.

        Default implementation saves to a temporary file. Override if a more
        efficient method is available.
        """
        import tempfile

        from qiskit import qasm2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
            f.write(qasm2.dumps(circuit))
            temp_path = Path(f.name)

        try:
            return self._run_from_path(temp_path, config)
        finally:
            temp_path.unlink(missing_ok=True)


class InMemoryRunner(CircuitRunner):
    """
    Base class for runners that operate on QuantumCircuit objects in memory.

    Some optimizers (like Qiskit transpiler) work directly with QuantumCircuit
    objects. This base class provides common functionality for in-memory runners.
    """

    def run(
        self,
        circuit: QuantumCircuit | Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """
        Run the optimizer on a circuit.

        If a Path is provided, the QASM file will be loaded first.

        Args:
            circuit: Either a QuantumCircuit object or a Path to a QASM file.
            config: Optional runner-specific configuration.

        Returns:
            List of TranspiledCircuit results.
        """
        if isinstance(circuit, Path):
            return self._run_from_path(circuit, config)
        return self._run_from_circuit(circuit, config)

    @abstractmethod
    def _run_from_circuit(
        self,
        circuit: QuantumCircuit,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """Run optimization on a QuantumCircuit in memory."""
        ...

    def _run_from_path(
        self,
        circuit_path: Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        """
        Run optimization by loading a circuit from a QASM file.

        Default implementation loads the file. Override if needed.
        """
        from qiskit import qasm2

        circuit = qasm2.loads(circuit_path.read_text())
        return self._run_from_circuit(circuit, config)

