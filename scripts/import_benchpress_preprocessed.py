#!/usr/bin/env python3
"""Import Benchpress circuits into the trajectory database with IBMN gate set preprocessing.

This script:
1. Loads circuits from benchmarks/ai_transpile/qasm/
2. Decomposes them to the IBMN gate set (rz, sx, x, cx)
3. Saves preprocessed circuits to qasm/benchpress_ibmn/
4. Imports them into the trajectory database
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory.database import CircuitRecord, TrajectoryDatabase  # noqa: E402
from benchmarks.ai_transpile.transpilers import analyze_circuit  # noqa: E402

CIRCUIT_DIR = PROJECT_ROOT / "benchmarks/ai_transpile/qasm"
IBMN_BASIS_GATES = ["rz", "sx", "x", "cx"]
EXCLUDED_SUBTREES = {"guoq_ibmnew", "benchpress_ibmn"}


def decompose_to_ibmn_gate_set(circuit: QuantumCircuit) -> QuantumCircuit:
    """Decompose a circuit to the IBMN gate set used by the confirmatory workflow."""
    gates_before = {inst.operation.name for inst in circuit.data}
    if "measure" in gates_before:
        raise ValueError("circuit contains measurement gates")

    qc = transpile(
        circuit.copy(),
        basis_gates=IBMN_BASIS_GATES,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
    )

    remaining = {inst.operation.name for inst in qc.data} - set(IBMN_BASIS_GATES)
    if remaining:
        qc = qc.decompose().decompose().decompose()
    return qc


def get_category_from_path(path: Path) -> str:
    """Extract the category from a QASM file path relative to the repo QASM root."""
    rel = path.relative_to(CIRCUIT_DIR)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "misc"


def iter_benchpress_qasm_files(output_dir: Path) -> list[Path]:
    """Return input QASM files, excluding generated or non-Benchpress subtrees."""
    qasm_files: list[Path] = []
    output_dir_resolved = output_dir.resolve()
    for path in CIRCUIT_DIR.rglob("*.qasm"):
        path_resolved = path.resolve()
        rel = path.relative_to(CIRCUIT_DIR)
        if rel.parts and rel.parts[0] in EXCLUDED_SUBTREES:
            continue
        if output_dir_resolved in path_resolved.parents:
            continue
        qasm_files.append(path)
    return sorted(qasm_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Benchpress circuits with IBMN preprocessing")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to the trajectory database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CIRCUIT_DIR / "benchpress_ibmn",
        help="Directory to save preprocessed circuits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze but do not save or import",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    qasm_files = iter_benchpress_qasm_files(output_dir)
    db = None if args.dry_run else TrajectoryDatabase(args.database)

    print(f"Found {len(qasm_files)} QASM files")
    print(f"Output directory: {output_dir}")
    print(f"Database: {args.database}")
    if args.dry_run:
        print("DRY RUN")
    print()

    imported = 0
    skipped_measure = 0
    skipped_parse = 0
    skipped_exists = 0

    try:
        for qasm_path in qasm_files:
            circuit_name = qasm_path.stem
            category = get_category_from_path(qasm_path)

            if db is not None and db.get_circuit_by_name(circuit_name) is not None:
                skipped_exists += 1
                continue

            try:
                circuit = qasm2.load(str(qasm_path), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
            except Exception as exc:
                print(f"  SKIP {qasm_path.name}: Parse error: {exc}")
                skipped_parse += 1
                continue

            if any(inst.operation.name == "measure" for inst in circuit.data):
                print(f"  SKIP {qasm_path.name}: Contains measurement gates")
                skipped_measure += 1
                continue

            try:
                circuit_ibmn = decompose_to_ibmn_gate_set(circuit)
            except Exception as exc:
                print(f"  SKIP {qasm_path.name}: Preprocess error: {exc}")
                skipped_parse += 1
                continue

            output_path = output_dir / f"{circuit_name}.qasm"
            if not args.dry_run:
                output_path.write_text(qasm2.dumps(circuit_ibmn), encoding="utf-8")

            metrics = analyze_circuit(circuit_ibmn)
            gate_set = {inst.operation.name for inst in circuit_ibmn.data}
            non_ibmn = gate_set - set(IBMN_BASIS_GATES)
            if non_ibmn:
                print(f"  WARNING {qasm_path.name}: Still has non-IBMN gates: {sorted(non_ibmn)}")

            print(
                f"  {circuit_name}: {metrics.two_qubit_gates} 2Q gates, "
                f"depth={metrics.depth}, qubits={circuit_ibmn.num_qubits}"
            )

            if db is not None:
                record = CircuitRecord(
                    id=None,
                    name=circuit_name,
                    category=category,
                    source="benchpress",
                    qasm_path=str(output_path.resolve()),
                    num_qubits=circuit_ibmn.num_qubits,
                    initial_depth=metrics.depth,
                    initial_two_qubit_gates=metrics.two_qubit_gates,
                    initial_two_qubit_depth=metrics.two_qubit_depth,
                    initial_total_gates=metrics.total_gates,
                    gate_density=metrics.total_gates / circuit_ibmn.num_qubits,
                    two_qubit_ratio=(metrics.two_qubit_gates / metrics.total_gates) if metrics.total_gates > 0 else 0.0,
                )
                db.insert_circuit(record)

            imported += 1
    finally:
        if db is not None:
            db.close()

    print("\nSummary:")
    print(f"  Imported:          {imported}")
    print(f"  Skipped (measure): {skipped_measure}")
    print(f"  Skipped (parse):   {skipped_parse}")
    print(f"  Skipped (exists):  {skipped_exists}")
    print(f"  Total processed:   {len(qasm_files)}")

    if not args.dry_run and imported > 0:
        print(f"\n{imported} circuits saved to {output_dir}")
        print(f"and registered in {args.database}")


if __name__ == "__main__":
    main()
