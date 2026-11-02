#!/usr/bin/env python3
"""Fix metadata.json by adding missing two_qubit_depth field."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def calculate_two_qubit_depth(qasm_path: Path) -> int:
    """Calculate two-qubit depth for a QASM file.
    
    Args:
        qasm_path: Path to QASM file
        
    Returns:
        Two-qubit depth
    """
    try:
        circuit = qasm2.load(str(qasm_path))
        
        # Extract two-qubit operations
        two_qubit_ops = []
        for instruction in circuit.data:
            if len(instruction.qubits) == 2:
                two_qubit_ops.append(instruction)
        
        # Calculate depth
        if not two_qubit_ops:
            return 0
        
        temp_circuit = QuantumCircuit(circuit.num_qubits)
        for op in two_qubit_ops:
            temp_circuit.append(op.operation, op.qubits)
        
        return temp_circuit.depth()
    except Exception as e:
        print(f"  Warning: Failed to analyze {qasm_path}: {e}", file=sys.stderr)
        return 0


def main() -> None:
    metadata_file = PROJECT_ROOT / "benchmarks/ai_transpile/metadata.json"
    
    # Load metadata
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata with {len(metadata['circuits'])} circuits")
    
    # Fix circuits missing two_qubit_depth
    fixed_count = 0
    error_count = 0
    
    for circuit in metadata["circuits"]:
        if "two_qubit_depth" not in circuit.get("metrics", {}):
            # Calculate it
            qasm_path = PROJECT_ROOT / "benchmarks/ai_transpile" / circuit["file"]
            
            if not qasm_path.exists():
                print(f"  Warning: File not found: {qasm_path}", file=sys.stderr)
                error_count += 1
                continue
            
            two_qubit_depth = calculate_two_qubit_depth(qasm_path)
            circuit["metrics"]["two_qubit_depth"] = two_qubit_depth
            fixed_count += 1
            print(f"  Fixed {circuit['name']}: two_qubit_depth = {two_qubit_depth}")
    
    # Save metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFixed {fixed_count} circuits")
    if error_count > 0:
        print(f"Errors: {error_count} circuits", file=sys.stderr)
    print(f"Saved to {metadata_file}")


if __name__ == "__main__":
    main()
