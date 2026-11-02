"""Generate benchmark circuits for optimization experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import efficient_su2, real_amplitudes
from qiskit.synthesis.qft import synth_qft_full


def generate_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate a Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits for the QFT

    Returns:
        QFT quantum circuit (decomposed to basic gates)
    """
    circuit = synth_qft_full(num_qubits, do_swaps=True)
    circuit.name = f"qft_{num_qubits}"
    # Decompose to basic gates for compatibility
    circuit = circuit.decompose()
    circuit = circuit.decompose()  # Second decomposition for nested gates
    return circuit


def generate_real_amplitudes(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Generate a RealAmplitudes variational ansatz.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions of the ansatz pattern

    Returns:
        RealAmplitudes quantum circuit
    """
    circuit = real_amplitudes(num_qubits, reps=reps, entanglement="linear")
    circuit.name = f"real_amplitudes_{num_qubits}"
    # Decompose to expand the ansatz structure
    circuit = circuit.decompose()
    return circuit


def generate_efficient_su2(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Generate an EfficientSU2 variational ansatz.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions of the ansatz pattern

    Returns:
        EfficientSU2 quantum circuit
    """
    circuit = efficient_su2(num_qubits, reps=reps, entanglement="linear")
    circuit.name = f"efficient_su2_{num_qubits}"
    # Decompose to expand the ansatz structure
    circuit = circuit.decompose()
    return circuit


def bind_parameters_to_zero(circuit: QuantumCircuit) -> QuantumCircuit:
    """Bind all parameters in a circuit to zero.

    Args:
        circuit: Circuit with parameters

    Returns:
        Circuit with all parameters bound to zero
    """
    if len(circuit.parameters) == 0:
        return circuit

    param_dict = {param: 0.0 for param in circuit.parameters}
    return circuit.assign_parameters(param_dict)


def _count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count the number of two-qubit gates in a circuit."""
    return sum(1 for instruction in circuit.data if len(instruction.qubits) >= 2)


def _two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Calculate the depth considering only two-qubit gates."""
    swap_free = circuit.decompose(gates_to_decompose=["swap"])
    return swap_free.depth(lambda instruction: len(instruction.qubits) >= 2)


def generate_circuit_metadata(
    circuit: QuantumCircuit,
    name: str,
    description: str,
    tags: Sequence[str],
) -> Dict[str, object]:
    """Generate metadata for a benchmark circuit.

    Args:
        circuit: The quantum circuit
        name: Name of the circuit
        description: Human-readable description
        tags: List of tags for categorization

    Returns:
        Metadata dictionary
    """
    return {
        "name": name,
        "description": description,
        "tags": list(tags),
        "num_qubits": circuit.num_qubits,
        "metrics": {
            "depth": circuit.depth(),
            "two_qubit_gates": _count_two_qubit_gates(circuit),
            "two_qubit_depth": _two_qubit_depth(circuit),
            "total_gates": circuit.size(),
        },
        "file": f"qasm/{name}.qasm",
    }


def save_benchmark_circuit(circuit: QuantumCircuit, output_path: Path) -> None:
    """Save a circuit to a QASM file.

    Args:
        circuit: Circuit to save
        output_path: Path to save QASM file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_str = qasm2.dumps(circuit)
    output_path.write_text(qasm_str)


def update_metadata_file(metadata_path: Path, new_circuit_metadata: Dict[str, object]) -> None:
    """Update the metadata.json file with a new circuit.

    If a circuit with the same name exists, it will be replaced.

    Args:
        metadata_path: Path to metadata.json
        new_circuit_metadata: Metadata for the new circuit
    """
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text())
    else:
        data = {"circuits": []}

    # Remove existing entry with same name if present
    circuits: List[Dict[str, object]] = data["circuits"]
    circuits = [c for c in circuits if c.get("name") != new_circuit_metadata["name"]]
    circuits.append(new_circuit_metadata)
    data["circuits"] = circuits

    metadata_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    """Generate benchmark circuits and update metadata."""
    parser = argparse.ArgumentParser(description="Generate benchmark quantum circuits")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/ai_transpile"),
        help="Output directory for circuits and metadata",
    )
    parser.add_argument(
        "--qft-sizes",
        type=int,
        nargs="+",
        default=[10, 12, 16],
        help="Qubit counts for QFT circuits",
    )
    parser.add_argument(
        "--ansatz-configs",
        action="store_true",
        help="Generate variational ansatz circuits",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    qasm_dir = output_dir / "qasm"
    metadata_file = output_dir / "metadata.json"

    qasm_dir.mkdir(parents=True, exist_ok=True)

    # Generate QFT circuits
    for num_qubits in args.qft_sizes:
        print(f"Generating QFT circuit with {num_qubits} qubits...")
        circuit = generate_qft_circuit(num_qubits)
        name = f"qft_{num_qubits}"

        # Save circuit
        output_file = qasm_dir / f"{name}.qasm"
        save_benchmark_circuit(circuit, output_file)

        # Update metadata
        metadata = generate_circuit_metadata(
            circuit=circuit,
            name=name,
            description=f"Quantum Fourier Transform on {num_qubits} qubits",
            tags=["qft", "algorithmic"],
        )
        update_metadata_file(metadata_file, metadata)
        print(f"  Saved to {output_file}")
        print(f"  Metrics: {metadata['metrics']}")

    # Generate variational ansatz circuits
    if args.ansatz_configs:
        ansatz_configs = [
            (8, 2, "real_amplitudes", generate_real_amplitudes),
            (12, 3, "real_amplitudes", generate_real_amplitudes),
            (16, 2, "real_amplitudes", generate_real_amplitudes),
            (10, 2, "efficient_su2", generate_efficient_su2),
            (12, 2, "efficient_su2", generate_efficient_su2),
        ]

        for num_qubits, reps, ansatz_type, generator_func in ansatz_configs:
            print(f"Generating {ansatz_type} circuit with {num_qubits} qubits, {reps} reps...")
            circuit = generator_func(num_qubits, reps=reps)
            circuit = bind_parameters_to_zero(circuit)
            name = f"{ansatz_type}_{num_qubits}_r{reps}"

            # Save circuit
            output_file = qasm_dir / f"{name}.qasm"
            save_benchmark_circuit(circuit, output_file)

            # Update metadata
            description = (
                f"{ansatz_type.replace('_', ' ').title()} ansatz on "
                f"{num_qubits} qubits with {reps} repetitions"
            )
            metadata = generate_circuit_metadata(
                circuit=circuit,
                name=name,
                description=description,
                tags=[ansatz_type, "variational", "vqe"],
            )
            update_metadata_file(metadata_file, metadata)
            print(f"  Saved to {output_file}")
            print(f"  Metrics: {metadata['metrics']}")

    print(f"\nAll circuits generated and metadata updated in {metadata_file}")


if __name__ == "__main__":
    main()

