from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit, qasm2

from benchmarks.ai_transpile.hardware_aware_transpilers import (
    IBM_AI_AVAILABLE,
    PYTKET_AVAILABLE,
    load_fake_torino_target,
    run_tket_hardware,
    transpile_with_qiskit_ai_hardware,
    transpile_with_qiskit_standard_hardware,
    validate_hardware_circuit,
)


def test_standard_optimizer_uses_full_fake_torino() -> None:
    hardware = load_fake_torino_target()
    assert hardware.num_qubits == 133
    circuit = QuantumCircuit(5)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.cx(4, 2)

    result = transpile_with_qiskit_standard_hardware(
        circuit,
        hardware,
        optimization_level=1,
    )

    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.circuit.num_qubits == hardware.num_qubits
    assert result.metadata["equivalence_verified"] is True
    assert result.metadata["hardware_aware"] is True
    assert result.metadata["backend_name"] == "fake_torino"


def test_full_device_equivalence_is_skipped_for_large_active_circuits() -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(11)
    circuit.h(0)
    for index in range(10):
        circuit.cx(index, index + 1)

    result = transpile_with_qiskit_standard_hardware(
        circuit,
        hardware,
        optimization_level=1,
    )

    assert result.metadata["equivalence_status"] == "skipped"
    assert result.metadata["equivalence_skipped"] is True
    assert result.metadata["equivalence_failed"] is False


def test_full_device_measurements_are_retained_and_skipped_for_exact_check() -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure(range(3), range(3))

    result = transpile_with_qiskit_standard_hardware(
        circuit,
        hardware,
        optimization_level=1,
    )

    assert result.circuit.num_clbits == 3
    assert result.metadata["equivalence_status"] == "skipped"
    assert result.metadata["equivalence_skipped"] is True
    assert result.metadata["equivalence_failed"] is False


@pytest.mark.skipif(not IBM_AI_AVAILABLE, reason="qiskit-ibm-transpiler is unavailable")
def test_ai_optimizer_preserves_equivalence_and_topology() -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(5)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.cx(4, 2)

    result = transpile_with_qiskit_ai_hardware(
        circuit,
        hardware,
        optimization_level=1,
    )

    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.metadata["equivalence_status"] in {"verified", "skipped"}
    assert result.metadata["routing_method"] == "ibm_ai_hybrid"


@pytest.mark.skipif(not IBM_AI_AVAILABLE, reason="qiskit-ibm-transpiler is unavailable")
def test_ai_optimizer_can_use_full_fake_topology() -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.cx(2, 1)

    result = transpile_with_qiskit_ai_hardware(
        circuit,
        hardware,
        optimization_level=1,
    )

    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.metadata["full_device"] is True
    assert len(result.metadata["layout"]["input_logical_to_physical"]) >= circuit.num_qubits


@pytest.mark.skipif(not PYTKET_AVAILABLE, reason="isolated TKET environment is unavailable")
def test_tket_optimizer_preserves_placement_metadata(tmp_path) -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(5)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.cx(4, 2)
    input_path = tmp_path / "input.qasm"
    input_path.write_text(qasm2.dumps(circuit), encoding="utf-8")

    result = run_tket_hardware(
        input_path,
        hardware,
        output_dir=tmp_path,
        job_info="test",
    )

    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.metadata["equivalence_status"] in {"verified", "skipped"}
    assert result.artifact_path is not None and result.artifact_path.exists()
    assert result.metadata["routing_method"] == "tket_graph_placement"


@pytest.mark.skipif(not PYTKET_AVAILABLE, reason="isolated TKET environment is unavailable")
def test_tket_optimizer_can_use_full_fake_topology(tmp_path) -> None:
    hardware = load_fake_torino_target()
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.cx(2, 1)
    input_path = tmp_path / "input.qasm"
    input_path.write_text(qasm2.dumps(circuit), encoding="utf-8")

    result = run_tket_hardware(
        input_path,
        hardware,
        output_dir=tmp_path,
        job_info="full_device",
    )

    assert result.circuit.num_qubits == hardware.num_qubits
    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.metadata["full_device"] is True
    assert len(result.metadata["layout"]["input_logical_to_physical"]) == hardware.num_qubits


@pytest.mark.skipif(not PYTKET_AVAILABLE, reason="isolated TKET environment is unavailable")
def test_tket_routes_after_peephole_changes(tmp_path) -> None:
    circuit_path = Path(__file__).resolve().parents[1] / "data/original_circuits/mqt/full_adder_alg_4.qasm"
    hardware = load_fake_torino_target()

    result = run_tket_hardware(
        circuit_path, hardware, output_dir=tmp_path, job_info="full_adder"
    )

    assert validate_hardware_circuit(result.circuit, hardware) == []
    assert result.metadata["equivalence_status"] in {"verified", "skipped"}
