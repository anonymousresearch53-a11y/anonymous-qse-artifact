"""
TKET optimization script template.

This script runs in an isolated TKET environment (.venv-tket) to avoid
networkx version conflicts with qiskit-ibm-ai-local-transpiler.

The script receives:
  - QASM string via qasm_str
  - Target gate set name via gate_set_name
  - Target gate names list via gate_names_str

It outputs a JSON object with:
  - qasm: The optimized circuit as QASM string
  - duration: Time taken to run the optimization
"""

import json
import time

from pytket.circuit import OpType
from pytket.passes import AutoRebase, DecomposeBoxes, FullPeepholeOptimise, RemoveRedundancies, SequencePass
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

# Gate set mapping for pytket OpTypes
TKET_OPTYPE_MAP = {
    "u1": OpType.U1,
    "u2": OpType.U2,
    "u3": OpType.U3,
    "cx": OpType.CX,
    "rz": OpType.Rz,
    "sx": OpType.SX,
    "x": OpType.X,
    "h": OpType.H,
    "rx": OpType.Rx,
    "ry": OpType.Ry,
}


def optimize_circuit(qasm_str: str, gate_names: list[str]) -> dict[str, object]:
    """
    Optimize a circuit using TKET's FullPeepholeOptimise.

    Args:
        qasm_str: Input circuit as QASM 2.0 string
        gate_names: List of target gate names (e.g., ["cx", "rz", "sx", "x"])

    Returns:
        Dictionary with 'qasm' (optimized circuit) and 'duration' (seconds)
    """
    # Load circuit from QASM
    tket_circuit = circuit_from_qasm_str(qasm_str)

    # Get target OpTypes from gate names
    target_gates = {TKET_OPTYPE_MAP[g] for g in gate_names if g in TKET_OPTYPE_MAP}

    # Build optimization pass sequence
    seq_pass = SequencePass([
        DecomposeBoxes(),
        FullPeepholeOptimise(),
        RemoveRedundancies(),
        AutoRebase(target_gates),
    ])

    # Run optimization
    start_time = time.perf_counter()
    seq_pass.apply(tket_circuit)
    duration = time.perf_counter() - start_time

    # Convert to QASM and return result
    output_qasm = circuit_to_qasm_str(tket_circuit)
    return {
        "qasm": output_qasm,
        "duration": duration,
    }


# Script execution when run directly (called from TKET environment)
if __name__ == "__main__":
    import sys

    # These variables are injected when the script is executed
    # They are defined in the calling code before exec()
    try:
        # Get input from globals (injected by the runner)
        qasm_input = globals().get("__qasm_str__", "")
        gate_names_input = globals().get("__gate_names__", [])

        if not qasm_input:
            # If not injected, try reading from stdin (alternative method)
            import sys
            if not sys.stdin.isatty():
                input_data = json.loads(sys.stdin.read())
                qasm_input = input_data["qasm_str"]
                gate_names_input = input_data["gate_names"]

        result = optimize_circuit(qasm_input, gate_names_input)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

