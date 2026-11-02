from __future__ import annotations

import json
from typing import Tuple, TypedDict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGates
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis


def ring_coupling_map(num_qubits: int) -> CouplingMap:
    edges: list[Tuple[int, int]] = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    edges += [((i + 1) % num_qubits, i) for i in range(num_qubits)]
    return CouplingMap(edges)


def make_ai_pass_manager(cm: CouplingMap) -> PassManager:
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

    return PassManager([
        AIRouting(
            backend_name="ibm_torino",
            optimization_level=3,
            layout_mode="optimize",
            local_mode=True,
            coupling_map=cm,
        ),
        collect_lfs,
        ai_lf_synth,
        Optimize1qGates(),
    ])


class CircuitMetrics(TypedDict):
    depth: int
    size: int
    num_qubits: int
    two_qubit_count: int
    ops: dict[str, int]


def summarize(circ: QuantumCircuit) -> CircuitMetrics:
    two_qubit_count = 0
    for instr in circ.data:
        qargs = instr.qubits
        if len(qargs) == 2:
            two_qubit_count += 1
    return {
        "depth": circ.depth(),
        "size": circ.size(),
        "num_qubits": circ.num_qubits,
        "two_qubit_count": two_qubit_count,
        "ops": dict(circ.count_ops()),
    }


def unitary_fidelity(original: QuantumCircuit, optimized: QuantumCircuit) -> float:
    """Compute unitary/process fidelity between two circuits' unitaries.

    F_pro = |Tr(U^\dagger V)|^2 / d^2, where d = 2^n.
    """
    try:
        U = Operator(original).data
        V = Operator(optimized).data
        d = U.shape[0]
        tr = np.trace(U.conj().T @ V)
        f_pro = float((abs(tr) ** 2) / (d * d))
        return f_pro
    except Exception:
        return float("nan")


def main() -> None:
    num_qubits = 10
    depth = 6
    seed1 = 1337
    seed2 = 4242

    # Two different random circuits (different seeds), no measurements
    circuit1 = random_circuit(num_qubits, depth=depth, measure=False, seed=seed1)
    circuit2 = random_circuit(num_qubits, depth=depth, measure=False, seed=seed2)

    cm = ring_coupling_map(num_qubits)
    ai_pm = make_ai_pass_manager(cm)

    ai_run_1 = ai_pm.run(circuit1)
    ai_run_2 = ai_pm.run(circuit2)

    m1 = summarize(ai_run_1)
    m2 = summarize(ai_run_2)

    print("AI run 1 metrics (seed 1337):\n" + json.dumps(m1, indent=2))
    print("AI run 2 metrics (seed 4242):\n" + json.dumps(m2, indent=2))

    deltas = {
        "depth": m2["depth"] - m1["depth"],
        "size": m2["size"] - m1["size"],
        "two_qubit_count": m2["two_qubit_count"] - m1["two_qubit_count"],
    }
    print("Metric deltas (seed 4242 - seed 1337):\n" + json.dumps(deltas, indent=2))

    # Fidelity of each AI run vs its original circuit
    f1 = unitary_fidelity(circuit1, ai_run_1)
    f2 = unitary_fidelity(circuit2, ai_run_2)
    print(json.dumps({"unitary_fidelity_seed1337": f1, "unitary_fidelity_seed4242": f2}, indent=2))


if __name__ == "__main__":
    main()

