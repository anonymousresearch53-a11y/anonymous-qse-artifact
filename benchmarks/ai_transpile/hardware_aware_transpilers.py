"""The optimizers in transpilers.py, compiled for an IBM backend."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager, Target
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.transpiler.passes import BasisTranslator

from .transpilers import (
    PYTKET_AVAILABLE,
    TKET_GATE_SET_MAP,
    TKET_OPTYPE_MAP,
    TranspiledCircuit,
    WisqConfig,
    analyze_circuit,
    run_wisq_opt,
)

try:
    from qiskit_ibm_transpiler import generate_ai_pass_manager
    IBM_AI_AVAILABLE = True
except ImportError:
    IBM_AI_AVAILABLE = False


@dataclass(frozen=True)
class HardwareTarget:
    backend_name: str
    source: str
    backend_version: str | None
    basis_gates: tuple[str, ...]
    coupling_map: CouplingMap
    physical_qubits: tuple[int, ...]
    target: Target
    fingerprint: str

    @property
    def num_qubits(self) -> int:
        return len(self.physical_qubits)


def load_fake_torino_target() -> HardwareTarget:
    """Return Fake Torino's complete target without rebuilding its topology."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    coupling_map = backend.target.build_coupling_map()
    if coupling_map is None:
        raise ValueError("Fake Torino has no coupling map")

    basis = tuple(
        name for name in ("id", "rz", "sx", "x", "u", "p", "rx", "ry", "h", "cx", "cz", "ecr")
        if name in backend.target.operation_names
    )
    physical_qubits = tuple(range(backend.num_qubits))
    fingerprint_data = json.dumps(
        [str(backend.name), backend.backend_version, basis,
         sorted(coupling_map.get_edges()), physical_qubits]
    )
    return HardwareTarget(
        backend_name=str(backend.name),
        source="fake",
        backend_version=str(backend.backend_version),
        basis_gates=basis,
        coupling_map=coupling_map,
        physical_qubits=physical_qubits,
        target=backend.target,
        fingerprint=hashlib.sha256(fingerprint_data.encode()).hexdigest(),
    )


def validate_hardware_circuit(
    circuit: QuantumCircuit, hardware: HardwareTarget, *, raise_on_error: bool = True
) -> list[str]:
    errors = []
    if circuit.num_qubits > hardware.num_qubits:
        errors.append("Circuit is wider than the target")
    edges = set(hardware.coupling_map.get_edges())
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in hardware.basis_gates and name not in {"barrier", "measure", "reset"}:
            errors.append(f"Unsupported gate: {name}")
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if len(qubits) == 2 and name != "barrier" and qubits not in edges:
            errors.append(f"Illegal edge: {qubits}")
        if len(qubits) > 2 and name != "barrier":
            errors.append(f"Unsupported multi-qubit gate: {name}")
    if errors and raise_on_error:
        raise ValueError("; ".join(errors))
    return errors


def circuits_are_equivalent(
    original: QuantumCircuit, optimized: QuantumCircuit, *, max_qubits: int = 10
) -> bool | None:
    if original.num_qubits != optimized.num_qubits or original.num_qubits > max_qubits:
        return None
    if original.num_clbits or optimized.num_clbits:
        return None
    return bool(Operator.from_circuit(original).equiv(Operator.from_circuit(optimized)))


def layout_metadata(circuit: QuantumCircuit) -> dict[str, list[int]]:
    """Return the logical-to-physical mappings needed to interpret a mapped circuit."""
    layout = circuit.layout
    if layout is None:
        return {}
    return {
        "input_logical_to_physical": [int(index) for index in layout.initial_index_layout()],
        "output_logical_to_physical": [int(index) for index in layout.final_index_layout()],
    }


def _verify_full_device_equivalence(
    original: QuantumCircuit,
    compiled: QuantumCircuit,
    *,
    max_active_qubits: int = 10,
) -> tuple[str, str | None]:
    if original.num_clbits or compiled.num_clbits:
        return "skipped", (
            "Exact full-device equivalence is unavailable for circuits with "
            "classical bits or measurements"
        )
    if compiled.layout is None:
        raise ValueError("Full-device equivalence requires a transpiler layout")

    initial = compiled.layout.initial_index_layout()[:original.num_qubits]
    final = compiled.layout.final_index_layout()[:original.num_qubits]
    active = sorted({
        compiled.find_bit(qubit).index
        for instruction in compiled.data
        for qubit in instruction.qubits
    } | set(initial) | set(final))
    if len(active) > max_active_qubits:
        return "skipped", (
            "Full-device equivalence exceeds the exact-check limit: "
            f"{len(active)} active qubits > {max_active_qubits}"
        )

    compact_index = {physical: index for index, physical in enumerate(active)}
    compact = QuantumCircuit(len(active))
    for instruction in compiled.data:
        compact.append(
            instruction.operation,
            [compact_index[compiled.find_bit(qubit).index] for qubit in instruction.qubits],
        )

    expected = QuantumCircuit(len(active))
    for instruction in original.data:
        expected.append(
            instruction.operation,
            [compact_index[initial[original.find_bit(qubit).index]] for qubit in instruction.qubits],
        )

    occupants = {compact_index[initial[index]]: index for index in range(original.num_qubits)}
    for logical, destination_physical in enumerate(final):
        destination = compact_index[destination_physical]
        source = next(position for position, value in occupants.items() if value == logical)
        if source != destination:
            expected.swap(source, destination)
            occupants[source], occupants[destination] = occupants[destination], occupants[source]

    if not circuits_are_equivalent(expected, compact, max_qubits=max_active_qubits):
        return "failed", "Full-device transpilation changed the circuit operation"
    return "verified", None


def _equivalence_status(
    original: QuantumCircuit,
    compiled: QuantumCircuit,
) -> tuple[str, str | None]:
    return _verify_full_device_equivalence(original, compiled)


def _equivalence_metadata(status: str, error: str | None) -> dict[str, object]:
    return {
        "equivalence_status": status,
        "equivalence_verified": status == "verified",
        "equivalence_skipped": status == "skipped",
        "equivalence_failed": status == "failed",
        "equivalence_error": error,
    }


def _metadata(hardware: HardwareTarget, **extra: object) -> dict[str, object]:
    return {
        "hardware_aware": True,
        "backend_name": hardware.backend_name,
        "hardware_source": hardware.source,
        "physical_qubits": list(hardware.physical_qubits),
        "basis_gates": list(hardware.basis_gates),
        "coupling_map": list(hardware.coupling_map.get_edges()),
        "target_fingerprint": hardware.fingerprint,
        **extra,
    }


def transpile_with_qiskit_standard_hardware(
    circuit: QuantumCircuit,
    hardware: HardwareTarget,
    *,
    optimization_level: int = 3,
    seed_transpiler: int = 0,
) -> TranspiledCircuit:
    start = time.perf_counter()
    optimized = transpile(
        circuit, target=hardware.target, optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    validate_hardware_circuit(optimized, hardware)
    equivalence_status, equivalence_error = _equivalence_status(circuit, optimized)
    return TranspiledCircuit(
        optimizer="qiskit_standard_hardware",
        label=f"qiskit_standard_{hardware.backend_name}",
        circuit=optimized,
        metrics=analyze_circuit(optimized),
        metadata=_metadata(
            hardware,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            full_device=True,
            **_equivalence_metadata(equivalence_status, equivalence_error),
            layout=layout_metadata(optimized),
            duration_seconds=time.perf_counter() - start,
        ),
    )


def transpile_with_qiskit_ai_hardware(
    circuit: QuantumCircuit,
    hardware: HardwareTarget,
    *,
    optimization_level: int = 1,
    ai_optimization_level: int = 3,
    layout_mode: str = "optimize",
    seed_transpiler: int = 0,
) -> TranspiledCircuit:
    """Compile with IBM's coordinated standard-plus-AI pass manager on Fake Torino."""
    if not IBM_AI_AVAILABLE:
        raise ImportError("qiskit-ibm-transpiler with AI pass-manager support is unavailable")

    start = time.perf_counter()
    pass_manager = generate_ai_pass_manager(
        optimization_level=optimization_level,
        ai_optimization_level=ai_optimization_level,
        coupling_map=hardware.coupling_map,
        ai_layout_mode=layout_mode,
        qiskit_transpile_options={
            "basis_gates": list(hardware.basis_gates),
            "seed_transpiler": seed_transpiler,
        },
    )
    optimized = pass_manager.run(circuit.copy())

    validate_hardware_circuit(optimized, hardware)
    equivalence_status, equivalence_error = _equivalence_status(circuit, optimized)

    return TranspiledCircuit(
        optimizer="qiskit_ai_hardware",
        label=f"qiskit_ai_{hardware.backend_name}",
        circuit=optimized,
        metrics=analyze_circuit(optimized),
        metadata=_metadata(
            hardware,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            layout_mode=layout_mode,
            full_device=True,
            **_equivalence_metadata(equivalence_status, equivalence_error),
            routing_method="ibm_ai_hybrid",
            implementation="generate_ai_pass_manager",
            ai_optimization_level=ai_optimization_level,
            layout=layout_metadata(optimized),
            duration_seconds=time.perf_counter() - start,
        ),
    )


def _write_qasm(circuit: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qasm2.dumps(circuit), encoding="utf-8")


def _tket_hardware_script(
    circuit_qasm: str,
    edges: Sequence[tuple[int, int]],
    gate_names: Sequence[str],
) -> str:
    return f'''
import json
from pytket.architecture import Architecture
from pytket.circuit import Node, OpType
from pytket.passes import AutoRebase, DecomposeBoxes, FullPeepholeOptimise, RemoveRedundancies, RoutingPass, SequencePass
from pytket.placement import GraphPlacement
from pytket.predicates import CompilationUnit
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

circuit = circuit_from_qasm_str({circuit_qasm!r})
architecture = Architecture({list(edges)!r})
DecomposeBoxes().apply(circuit)
placement = GraphPlacement(architecture)
placement_map = placement.get_placement_map(circuit)
used = {{node.index[0] for node in placement_map.values() if node.reg_name != "unplaced"}}
available = iter(sorted(set(range(circuit.n_qubits)) - used))
for qubit, node in placement_map.items():
    if node.reg_name == "unplaced":
        placement_map[qubit] = Node(next(available))
initial_layout = {{qubit.index[0]: node.index[0] for qubit, node in placement_map.items()}}
placement.place_with_map(circuit, placement_map)
routing_unit = CompilationUnit(circuit)
RoutingPass(architecture).apply(routing_unit)
gate_types = {{getattr(OpType, name) for name in {list(gate_names)!r}}}
SequencePass([
    FullPeepholeOptimise(allow_swaps=False),
    RoutingPass(architecture),
    RemoveRedundancies(),
    AutoRebase(gate_types),
]).apply(routing_unit)
circuit = routing_unit.circuit
output_layout = {{
    virtual: routing_unit.final_map[Node(physical)].index[0]
    for virtual, physical in initial_layout.items()
}}
circuit.replace_implicit_wire_swaps()
RoutingPass(architecture).apply(circuit)
AutoRebase(gate_types).apply(circuit)
print(json.dumps({{
    "qasm": circuit_to_qasm_str(circuit),
    "initial_layout": initial_layout,
    "output_layout": output_layout,
}}))
'''


def run_tket_hardware(
    circuit_path: Path,
    hardware: HardwareTarget,
    *,
    output_dir: Path,
    job_info: str,
    gate_set: str = "IBMN",
) -> TranspiledCircuit:
    if not PYTKET_AVAILABLE:
        raise ImportError("The isolated TKET environment is unavailable")
    from benchmarks.tket_runner import run_tket_script

    circuit = qasm2.load(circuit_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    source_circuit = QuantumCircuit(hardware.num_qubits, circuit.num_clbits)
    source_circuit.compose(
        circuit,
        qubits=list(range(circuit.num_qubits)),
        clbits=list(range(circuit.num_clbits)),
        inplace=True,
    )
    source_qasm = qasm2.dumps(source_circuit)
    gate_names = TKET_GATE_SET_MAP.get(gate_set, TKET_GATE_SET_MAP["IBMN"])
    gate_types = [TKET_OPTYPE_MAP[name] for name in gate_names]
    edges = sorted({tuple(sorted(edge)) for edge in hardware.coupling_map.get_edges()})
    start = time.perf_counter()
    script = _tket_hardware_script(source_qasm, edges, gate_types)
    response = json.loads(run_tket_script(script).stdout.strip())
    mapped = QuantumCircuit.from_qasm_str(response["qasm"])
    initial_layout = {
        int(virtual): int(physical)
        for virtual, physical in response["initial_layout"].items()
    }
    output_layout = {
        int(virtual): int(physical)
        for virtual, physical in response["output_layout"].items()
    }
    virtual_at_physical = {physical: virtual for virtual, physical in initial_layout.items()}
    final_layout = Layout({
        mapped.qubits[physical]: output_layout[virtual]
        for physical, virtual in virtual_at_physical.items()
    })
    mapped._layout = TranspileLayout(
        initial_layout=Layout({
            source_circuit.qubits[int(virtual)]: int(physical)
            for virtual, physical in response["initial_layout"].items()
        }),
        input_qubit_mapping={qubit: index for index, qubit in enumerate(source_circuit.qubits)},
        final_layout=final_layout,
        _input_qubit_count=source_circuit.num_qubits,
        _output_qubit_list=list(mapped.qubits),
    )
    tket_layout = {
        "input_logical_to_physical": [initial_layout[index] for index in range(source_circuit.num_qubits)],
        "output_logical_to_physical": [output_layout[index] for index in range(source_circuit.num_qubits)],
    }
    optimized = PassManager([
        BasisTranslator(SessionEquivalenceLibrary, list(hardware.basis_gates)),
    ]).run(mapped)
    optimized._layout = mapped.layout
    validate_hardware_circuit(optimized, hardware)
    equivalence_status, equivalence_error = _equivalence_status(circuit, optimized)
    output_path = output_dir / f"{circuit_path.stem}_{job_info}.qasm"
    _write_qasm(optimized, output_path)
    return TranspiledCircuit(
        optimizer="tket_hardware", label=f"tket_{hardware.backend_name}",
        circuit=optimized, metrics=analyze_circuit(optimized), artifact_path=output_path,
        metadata=_metadata(hardware, gate_set=gate_set,
                           routing_method="tket_graph_placement",
                           full_device=True,
                           **_equivalence_metadata(equivalence_status, equivalence_error),
                           layout=tket_layout,
                           duration_seconds=time.perf_counter() - start),
    )


def run_wisq_hardware(
    circuit_path: Path,
    hardware: HardwareTarget,
    *,
    output_dir: Path,
    job_info: str,
    approximation_epsilon: float,
    timeout_seconds: int,
    objective: str = "TWO_Q",
    advanced_args: Mapping[str, str | None] | None = None,
    optimization_level: int = 3,
    seed_transpiler: int = 0,
) -> TranspiledCircuit:
    circuit = qasm2.load(circuit_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        logical = run_wisq_opt(
            circuit_path,
            WisqConfig(
                target_gateset="IBMN", objective=objective,
                timeout_seconds=timeout_seconds,
                approximation_epsilon=approximation_epsilon,
                output_dir=Path(temp_dir), job_info=job_info,
                advanced_args=advanced_args,
            ),
        ).circuit
    optimized = transpile(
        logical, target=hardware.target, optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    validate_hardware_circuit(optimized, hardware)
    output_path = output_dir / f"{circuit_path.stem}_{job_info}_hardware.qasm"
    _write_qasm(optimized, output_path)
    optimizer = "wisq_bqskit_hardware" if approximation_epsilon > 0 else "wisq_rules_hardware"
    return TranspiledCircuit(
        optimizer=optimizer, label=f"{optimizer}_{hardware.backend_name}",
        circuit=optimized, metrics=analyze_circuit(optimized), artifact_path=output_path,
        metadata=_metadata(hardware, approximation_epsilon=approximation_epsilon,
                           layout=layout_metadata(optimized),
                           duration_seconds=time.perf_counter() - start),
    )


def choose_best_result(results: Sequence[TranspiledCircuit]) -> TranspiledCircuit:
    if not results:
        raise ValueError("Optimizer returned no variants")
    return min(results, key=lambda result: (
        result.metrics.two_qubit_gates,
        result.metrics.two_qubit_depth,
        result.metrics.depth,
    ))
