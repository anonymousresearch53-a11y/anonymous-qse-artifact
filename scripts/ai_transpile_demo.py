from __future__ import annotations

import json
from typing import Literal, Tuple, TypedDict

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGates
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions, CollectPauliNetworks  # noqa: F401
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis


def build_sample_circuit(num_qubits: int = 8) -> QuantumCircuit:
    """Create a simple parameterized circuit to transpile.

    Uses an efficient_su2 ansatz then decomposes to a basic gate set.
    """
    circuit: QuantumCircuit = efficient_su2(num_qubits, entanglement="circular", reps=1).decompose()
    return circuit


def ring_coupling_map(num_qubits: int) -> CouplingMap:
    """Construct a simple ring connectivity coupling map for local simulation."""
    edges: list[Tuple[int, int]] = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    # Make it undirected by adding reverse edges
    edges += [((i + 1) % num_qubits, i) for i in range(num_qubits)]
    return CouplingMap(edges)


def make_ai_pass_manager(num_qubits: int) -> PassManager:
    """Create a hybrid heuristic + AI pass manager in local mode."""
    cm = ring_coupling_map(num_qubits)

    # Create collection and synthesis passes
    collect_lfs = CollectLinearFunctions(
        do_commutative_analysis=True,
        min_block_size=0,
        max_block_size=2,
        collect_from_back=False,
    )

    ai_lf_synth = AILinearFunctionSynthesis(
        coupling_map=list(cm.get_edges()),
        replace_only_if_better=True,
    )

    pm = PassManager([
        AIRouting(
            backend_name="ibm_torino",  # name used by passes; local_mode avoids cloud
            optimization_level=3,
            layout_mode="optimize",
            local_mode=True,
            coupling_map=cm,
        ),
        collect_lfs,
        ai_lf_synth,
        Optimize1qGates(),
    ])

    # Inject coupling map via property set; AIRouting respects device maps internally
    pm.property_set["coupling_map"] = cm
    return pm


class CircuitMetrics(TypedDict):
    depth: int
    size: int
    num_qubits: int
    two_qubit_count: int
    ops: dict[str, int]


def summarize(circ: QuantumCircuit) -> CircuitMetrics:
    """Return shallow metrics for a circuit."""
    return {
        "depth": circ.depth(),
        "size": circ.size(),
        "num_qubits": circ.num_qubits,
        "two_qubit_count": sum(count for gate, count in circ.count_ops().items() if gate in {"cx", "cz", "swap"}),
        "ops": dict(circ.count_ops()),
    }


def bind_parameters_numeric(circuit: QuantumCircuit, value: float = 0.1) -> QuantumCircuit:
    """Bind all parameters in a circuit to a numeric value to enable optimization passes."""
    if len(circuit.parameters) == 0:
        return circuit
    mapping = {param: float(value) for param in circuit.parameters}
    return circuit.assign_parameters(mapping, inplace=False)


def verify_functional_equivalence(original: QuantumCircuit, optimized: QuantumCircuit, _shots: int = 0) -> bool:
    """Naively check functional equivalence by comparing statevectors on zero parameters.

    For parameterized circuits, efficient_su2 has default parameters; we compare statevectors.
    In noisy or hardware settings, use formal equivalence or sampling-based checks.
    """
    sv_orig = Statevector.from_instruction(original)
    sv_opt = Statevector.from_instruction(optimized)
    fidelity = abs(sv_orig.data.conj().T @ sv_opt.data) ** 2
    return bool(fidelity > 1 - 1e-10)


def main() -> None:
    num_qubits = 8
    circuit = build_sample_circuit(num_qubits)
    circuit = bind_parameters_numeric(circuit, value=0.1)

    # Baseline: only simple 1q optimization
    baseline_pm = PassManager([Optimize1qGates()])
    baseline = baseline_pm.run(circuit)

    # AI hybrid pass manager
    ai_pm = make_ai_pass_manager(num_qubits)
    ai_circuit = ai_pm.run(circuit)

    base_metrics = summarize(baseline)
    ai_metrics = summarize(ai_circuit)

    print("Baseline metrics:\n" + json.dumps(base_metrics, indent=2))
    print("AI metrics:\n" + json.dumps(ai_metrics, indent=2))

    try:
        eq = verify_functional_equivalence(baseline, ai_circuit)
        print(f"Functional equivalence (statevector): {eq}")
    except Exception as exc:  # Statevector may fail for large qubit counts
        print(f"Equivalence check skipped: {exc}")

    # Show diffs that matter for RQ1
    print("\nMetric deltas (AI - Baseline):")
    DeltaKey = Literal["depth", "size", "two_qubit_count"]
    delta_keys: tuple[DeltaKey, ...] = ("depth", "size", "two_qubit_count")
    deltas = {key: ai_metrics[key] - base_metrics[key] for key in delta_keys}
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
