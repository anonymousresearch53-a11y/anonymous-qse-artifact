from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT, EfficientSU2


@dataclass(frozen=True)
class CircuitMetrics:
    depth: int
    two_qubit_gates: int
    two_qubit_depth: int
    total_gates: int


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    description: str
    tags: List[str]
    generator: Callable[[], QuantumCircuit]


@dataclass(frozen=True)
class CircuitRecord:
    name: str
    file: str
    description: str
    tags: List[str]
    num_qubits: int
    metrics: CircuitMetrics


EXPORT_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "ai_transpile"
QASM_DIR = EXPORT_ROOT / "qasm"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
QASM_DIR.mkdir(parents=True, exist_ok=True)


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for _, qargs, _ in circuit.data if len(qargs) == 2)


def compute_two_qubit_depth(circuit: QuantumCircuit) -> int:
    swap_free = circuit.decompose(gates_to_decompose=["swap"])
    return swap_free.depth(lambda instruction: len(instruction.qubits) >= 2)


def build_metrics(circuit: QuantumCircuit) -> CircuitMetrics:
    return CircuitMetrics(
        depth=circuit.depth(),
        two_qubit_gates=count_two_qubit_gates(circuit),
        two_qubit_depth=compute_two_qubit_depth(circuit),
        total_gates=circuit.size(),
    )


def bind_numeric_parameters(circuit: QuantumCircuit) -> QuantumCircuit:
    if not circuit.parameters:
        return circuit
    bindings = {
        param: 0.1 * (index + 1)
        for index, param in enumerate(sorted(circuit.parameters, key=lambda item: item.name))
    }
    return circuit.assign_parameters(bindings, inplace=False)


def export_circuit(spec: CircuitSpec) -> CircuitRecord:
    circuit = spec.generator()
    circuit = bind_numeric_parameters(circuit)
    qasm_text = qasm2.dumps(circuit).replace("cp(", "cu1(")
    qasm_path = QASM_DIR / f"{spec.name}.qasm"
    qasm_path.write_text(qasm_text)

    metrics = build_metrics(circuit)
    return CircuitRecord(
        name=spec.name,
        file=str(qasm_path.relative_to(EXPORT_ROOT)),
        description=spec.description,
        tags=spec.tags,
        num_qubits=circuit.num_qubits,
        metrics=metrics,
    )


def main() -> None:
    specs: List[CircuitSpec] = [
        CircuitSpec(
            name="qft_8",
            description="QFT without swaps on 8 qubits (AI transpiler baseline).",
            tags=["qft", "ai_transpile_notebook"],
            generator=lambda: QFT(8, do_swaps=False).decompose(),
        ),
        CircuitSpec(
            name="efficient_su2_12",
            description="EfficientSU2 (full entanglement, 1 rep) on 12 qubits.",
            tags=["efficient_su2", "ai_transpile_notebook"],
            generator=lambda: EfficientSU2(12, entanglement="full", reps=1).decompose(),
        ),
    ]

    records: List[CircuitRecord] = [export_circuit(spec) for spec in specs]

    metadata: Dict[str, object] = {
        "source_notebook": "notebooks/ai_transpile_variants.ipynb",
        "description": "Reusable circuits distilled from the AI transpiler comparison notebook.",
        "circuits": [asdict(record) for record in records],
    }

    metadata_path = EXPORT_ROOT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Exported {len(records)} circuits to {EXPORT_ROOT}")


if __name__ == "__main__":
    main()

