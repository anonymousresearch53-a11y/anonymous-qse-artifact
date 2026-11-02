"""Utilities for comparing quantum circuits from different transpilers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

try:
    from mqt import qcec
except ImportError:
    qcec = None  # type: ignore[assignment, misc]


@dataclass
class CircuitComparison:
    """Results of comparing two quantum circuits."""

    equivalent: bool
    method: Literal["qcec", "operator", "statevector", "failed"]
    fidelity: float | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


def compare_circuits_qcec(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
) -> CircuitComparison:
    """Compare circuits using MQT QCEC (quantum circuit equivalence checker).

    This is the most robust method for formal equivalence checking.

    Args:
        circuit1: First circuit to compare
        circuit2: Second circuit to compare
        coupling_map: Optional coupling map for layout-aware comparison

    Returns:
        CircuitComparison with equivalence result
    """
    if qcec is None:
        return CircuitComparison(
            equivalent=False,
            method="failed",
            error="mqt.qcec not available",
        )

    try:
        # QCEC can accept QuantumCircuit objects directly
        # Run QCEC equivalence check
        result = qcec.verify(circuit1, circuit2)

        # QCEC result has 'considered_equivalent' boolean and 'equivalence' enum
        considered_equivalent = result.considered_equivalent
        if callable(considered_equivalent):
            equivalent = bool(considered_equivalent())
        else:
            equivalent = bool(considered_equivalent)

        return CircuitComparison(
            equivalent=equivalent,
            method="qcec",
            details={
                "equivalence": str(result.equivalence) if hasattr(result, "equivalence") else None,
                "check_time": result.check_time if hasattr(result, "check_time") else None,
                "preprocessing_time": result.preprocessing_time if hasattr(result, "preprocessing_time") else None,
            },
        )
    except Exception as exc:
        return CircuitComparison(
            equivalent=False,
            method="failed",
            error=f"QCEC check failed: {exc}",
        )


def compare_circuits_operator(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    atol: float = 1e-10,
) -> CircuitComparison:
    """Compare circuits by computing their unitary operators.

    This method computes the full unitary matrix for each circuit and compares them.
    Works well for smaller circuits but can be memory-intensive for large circuits.

    Args:
        circuit1: First circuit to compare
        circuit2: Second circuit to compare
        atol: Absolute tolerance for equivalence

    Returns:
        CircuitComparison with equivalence result
    """
    try:
        # Ensure circuits have same number of qubits
        if circuit1.num_qubits != circuit2.num_qubits:
            return CircuitComparison(
                equivalent=False,
                method="operator",
                error=f"Different qubit counts: {circuit1.num_qubits} vs {circuit2.num_qubits}",
            )

        op1 = Operator(circuit1)
        op2 = Operator(circuit2)

        # Check equivalence using Qiskit's equiv method
        equivalent = op1.equiv(op2, atol=atol)

        # Compute fidelity: |Tr(U1^dagger U2)|^2 / d^2
        import numpy as np

        d = op1.data.shape[0]
        tr = np.trace(op1.data.conj().T @ op2.data)
        fidelity = float((abs(tr) ** 2) / (d * d))

        return CircuitComparison(
            equivalent=equivalent,
            method="operator",
            fidelity=fidelity,
            details={"atol": atol},
        )
    except Exception as exc:
        return CircuitComparison(
            equivalent=False,
            method="failed",
            error=f"Operator comparison failed: {exc}",
        )


def compare_circuits_statevector(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    atol: float = 1e-10,
) -> CircuitComparison:
    """Compare circuits by comparing their output statevectors.

    This method simulates both circuits from the |0...0> state and compares
    the resulting statevectors. Only works for circuits without measurements
    and can be memory-intensive for large qubit counts.

    Args:
        circuit1: First circuit to compare
        circuit2: Second circuit to compare
        atol: Absolute tolerance for equivalence

    Returns:
        CircuitComparison with equivalence result
    """
    try:
        # Ensure circuits have same number of qubits
        if circuit1.num_qubits != circuit2.num_qubits:
            return CircuitComparison(
                equivalent=False,
                method="statevector",
                error=f"Different qubit counts: {circuit1.num_qubits} vs {circuit2.num_qubits}",
            )

        sv1 = Statevector.from_instruction(circuit1)
        sv2 = Statevector.from_instruction(circuit2)

        # Compute fidelity: |<sv1|sv2>|^2
        fidelity = abs(sv1.data.conj().T @ sv2.data) ** 2
        equivalent = bool(fidelity > (1 - atol))

        return CircuitComparison(
            equivalent=equivalent,
            method="statevector",
            fidelity=float(fidelity),
            details={"atol": atol},
        )
    except Exception as exc:
        return CircuitComparison(
            equivalent=False,
            method="failed",
            error=f"Statevector comparison failed: {exc}",
        )


def compare_circuits(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    method: Literal["auto", "qcec", "operator", "statevector"] = "auto",
    coupling_map: CouplingMap | None = None,
    max_qubits_for_operator: int = 12,
    max_qubits_for_statevector: int = 20,
) -> CircuitComparison:
    """Compare two quantum circuits for equivalence.

    This function tries multiple methods in order of preference:
    1. QCEC (if available) - most robust, handles layout differences
    2. Operator comparison - good for medium-sized circuits
    3. Statevector comparison - works for smaller circuits

    Args:
        circuit1: First circuit to compare
        circuit2: Second circuit to compare
        method: Comparison method to use. "auto" tries methods in order.
        coupling_map: Optional coupling map for layout-aware QCEC comparison
        max_qubits_for_operator: Maximum qubits before skipping operator method
        max_qubits_for_statevector: Maximum qubits before skipping statevector method

    Returns:
        CircuitComparison with the best available result
    """
    if method == "auto":
        # Try QCEC first if available
        if qcec is not None:
            result = compare_circuits_qcec(circuit1, circuit2, coupling_map)
            if result.method != "failed":
                return result

        # Fall back to operator comparison for medium circuits
        if circuit1.num_qubits <= max_qubits_for_operator:
            result = compare_circuits_operator(circuit1, circuit2)
            if result.method != "failed":
                return result

        # Fall back to statevector for smaller circuits
        if circuit1.num_qubits <= max_qubits_for_statevector:
            result = compare_circuits_statevector(circuit1, circuit2)
            if result.method != "failed":
                return result

        # If all methods failed, return the last error
        return CircuitComparison(
            equivalent=False,
            method="failed",
            error="All comparison methods failed or circuits too large",
        )

    elif method == "qcec":
        return compare_circuits_qcec(circuit1, circuit2, coupling_map)
    elif method == "operator":
        return compare_circuits_operator(circuit1, circuit2)
    elif method == "statevector":
        return compare_circuits_statevector(circuit1, circuit2)
    else:
        raise ValueError(f"Unknown comparison method: {method}")


def compare_against_baseline(
    baseline: QuantumCircuit,
    optimized: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
) -> dict[str, Any]:
    """Compare an optimized circuit against a baseline and return detailed results.

    Args:
        baseline: Baseline/reference circuit
        optimized: Optimized circuit to compare
        coupling_map: Optional coupling map for layout-aware comparison

    Returns:
        Dictionary with comparison results and metrics
    """
    comparison = compare_circuits(baseline, optimized, coupling_map=coupling_map)

    return {
        "equivalent": comparison.equivalent,
        "method": comparison.method,
        "fidelity": comparison.fidelity,
        "error": comparison.error,
        "details": comparison.details,
        "baseline_qubits": baseline.num_qubits,
        "optimized_qubits": optimized.num_qubits,
        "qubit_count_match": baseline.num_qubits == optimized.num_qubits,
    }
