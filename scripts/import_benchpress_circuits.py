#!/usr/bin/env python3
"""Import Benchpress circuits into local repository.

This script downloads all QASM files from the Qiskit Benchpress repository
and stores them locally with generated metadata.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from qiskit import qasm2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_qasm_file(qasm_path: Path) -> dict | None:
    """Analyze a QASM file and extract metrics.
    
    Args:
        qasm_path: Path to QASM file
        
    Returns:
        Dictionary with circuit metrics, or None if parsing fails
    """
    try:
        circuit = qasm2.load(str(qasm_path))
        
        # Count two-qubit gates and calculate two-qubit depth
        two_qubit_gates = 0
        two_qubit_ops = []
        for i, instruction in enumerate(circuit.data):
            if len(instruction.qubits) == 2:
                two_qubit_gates += 1
                two_qubit_ops.append(instruction)
        
        # Calculate two-qubit depth by creating a subcircuit with only 2q gates
        # and measuring its depth
        two_qubit_depth = 0
        if two_qubit_ops:
            from qiskit import QuantumCircuit
            temp_circuit = QuantumCircuit(circuit.num_qubits)
            for op in two_qubit_ops:
                temp_circuit.append(op.operation, op.qubits)
            two_qubit_depth = temp_circuit.depth()
        
        return {
            "depth": circuit.depth(),
            "two_qubit_gates": two_qubit_gates,
            "two_qubit_depth": two_qubit_depth,
            "total_gates": len(circuit.data),
        }
    except Exception as e:
        print(f"  Warning: Failed to analyze {qasm_path.name}: {e}", file=sys.stderr)
        return None


def import_category(
    source_dir: Path,
    target_dir: Path,
    category: str,
    max_qubits: int | None = None,
) -> list[dict]:
    """Import circuits from a Benchpress category.
    
    Args:
        source_dir: Source directory (e.g., /tmp/benchpress/benchpress/qasm/qft)
        target_dir: Target directory (e.g., benchmarks/ai_transpile/qasm/qft)
        category: Category name
        max_qubits: Maximum number of qubits to include (None for no limit)
        
    Returns:
        List of circuit metadata dictionaries
    """
    if not source_dir.exists():
        print(f"  Warning: Category directory not found: {source_dir}", file=sys.stderr)
        return []
    
    qasm_files = sorted(source_dir.glob("*.qasm"))
    if not qasm_files:
        print(f"  Warning: No QASM files found in {source_dir}", file=sys.stderr)
        return []
    
    target_dir.mkdir(parents=True, exist_ok=True)
    circuits = []
    
    for qasm_file in qasm_files:
        # Copy file
        target_file = target_dir / qasm_file.name
        shutil.copy2(qasm_file, target_file)
        
        # Analyze circuit
        metrics = analyze_qasm_file(target_file)
        if metrics is None:
            continue
        
        # Extract circuit info from filename
        name = qasm_file.stem
        
        # Infer num_qubits from filename (e.g., qft_N100.qasm -> 100 qubits)
        num_qubits = None
        if "_N" in name:
            try:
                num_qubits = int(name.split("_N")[1].split("_")[0])
            except (ValueError, IndexError):
                pass
        
        # If we couldn't infer from filename, get from circuit
        if num_qubits is None:
            try:
                circuit = qasm2.load(str(target_file))
                num_qubits = circuit.num_qubits
            except Exception:
                continue
        
        # Skip if exceeds max_qubits
        if max_qubits and num_qubits > max_qubits:
            target_file.unlink()  # Remove the copied file
            continue
        
        # Create metadata entry
        circuit_entry = {
            "name": name,
            "description": f"{category.replace('-', ' ').title()} benchmark circuit",
            "tags": [category, "benchpress"],
            "num_qubits": num_qubits,
            "metrics": {
                "depth": metrics["depth"],
                "two_qubit_gates": metrics["two_qubit_gates"],
                "two_qubit_depth": metrics["two_qubit_depth"],
                "total_gates": metrics["total_gates"],
            },
            "file": f"qasm/{category}/{qasm_file.name}",
        }
        
        circuits.append(circuit_entry)
    
    return circuits


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Import Benchpress circuits into local repository"
    )
    parser.add_argument(
        "--benchpress-path",
        type=Path,
        default=Path("/tmp/benchpress/benchpress/qasm"),
        help="Path to cloned Benchpress qasm directory",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/ai_transpile/qasm",
        help="Target directory for circuits",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to import (default: all)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=20,
        help="Maximum number of qubits (default: 20)",
    )
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Update metadata.json with new circuits",
    )
    
    args = parser.parse_args()
    
    # Verify source exists
    if not args.benchpress_path.exists():
        print(f"Error: Benchpress path not found: {args.benchpress_path}", file=sys.stderr)
        print("Please clone Benchpress first:", file=sys.stderr)
        print("  cd /tmp && git clone --depth 1 https://github.com/Qiskit/benchpress.git", file=sys.stderr)
        sys.exit(1)
    
    # Find all categories
    all_categories = [
        d.name for d in args.benchpress_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    categories_to_import = args.categories if args.categories else all_categories
    
    print(f"\nImporting Benchpress circuits to {args.target_dir}")
    print(f"Max qubits: {args.max_qubits}")
    print(f"Categories: {', '.join(categories_to_import)}\n")
    
    all_circuits = []
    category_counts = {}
    
    for category in categories_to_import:
        source_dir = args.benchpress_path / category
        target_dir = args.target_dir / category
        
        print(f"Processing {category}...")
        circuits = import_category(source_dir, target_dir, category, args.max_qubits)
        category_counts[category] = len(circuits)
        all_circuits.extend(circuits)
        print(f"  Imported {len(circuits)} circuits")
    
    print(f"\nTotal imported: {len(all_circuits)} circuits")
    print("\nBreakdown by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:20s}: {count:3d} circuits")
    
    # Update metadata.json if requested
    if args.update_metadata:
        metadata_file = PROJECT_ROOT / "benchmarks/ai_transpile/metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"\nLoaded existing metadata with {len(metadata['circuits'])} circuits")
        else:
            metadata = {
                "source_notebook": "notebooks/ai_transpile_variants.ipynb",
                "description": "Quantum circuit benchmarks for transpiler comparison",
                "circuits": [],
            }
        
        # Get existing circuit names
        existing_names = {c["name"] for c in metadata["circuits"]}
        
        # Add new circuits
        new_circuits = [c for c in all_circuits if c["name"] not in existing_names]
        metadata["circuits"].extend(new_circuits)
        
        # Sort by category and name
        metadata["circuits"].sort(key=lambda c: (c.get("tags", [""])[0], c["name"]))
        
        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nUpdated {metadata_file}")
        print(f"  Added {len(new_circuits)} new circuits")
        print(f"  Total circuits in metadata: {len(metadata['circuits'])}")


if __name__ == "__main__":
    main()
