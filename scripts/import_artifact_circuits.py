#!/usr/bin/env python3
"""Import GUOQ ibmnew benchmark circuits into the trajectory database.

Usage:
    python scripts/import_guoq_circuits.py \
        --source benchmarks/ai_transpile/qasm/guoq_ibmnew/ \
        --database data/trajectories_guoq_benchmark.db

The GUOQ ibmnew circuits are in IBM native gate set (cx, u3/u1/id/rz/x/sx)
so no gate-set conversion is needed.  Each circuit is registered with
category="guoq_ibmnew" and source="guoq".

To fetch the circuits first:
    git clone --depth 1 https://github.com/qqq-wisc/guoq.git /tmp/guoq
    cp /tmp/guoq/benchmarks/ibmnew/*.qasm \\
       benchmarks/ai_transpile/qasm/guoq_ibmnew/
    rm -rf /tmp/guoq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory.database import (  # noqa: E402
    CircuitRecord,
    TrajectoryDatabase,
)
from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS  # noqa: E402


def _analyze_qasm(qasm_path: Path) -> dict | None:
    """Parse a QASM file and return circuit metrics.

    GUOQ ibmnew circuits use sx gate which isn't in qelib1.inc,
    so we need LEGACY_CUSTOM_INSTRUCTIONS for parsing.

    Returns None if parsing fails (circuit is skipped).
    """
    try:
        circuit = qasm2.load(str(qasm_path), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as e:
        print(f"  SKIP {qasm_path.name}: QASM parse error: {e}", file=sys.stderr)
        return None

    num_qubits = circuit.num_qubits
    total_gates = len(circuit.data)

    two_qubit_gates = 0
    two_qubit_ops = []
    for inst in circuit.data:
        if len(inst.qubits) == 2:
            two_qubit_gates += 1
            two_qubit_ops.append(inst)

    two_qubit_depth = 0
    if two_qubit_ops:
        try:
            tmp = QuantumCircuit(num_qubits)
            for op in two_qubit_ops:
                qubit_indices = [circuit.find_bit(q).index for q in op.qubits]
                tmp.append(op.operation, qubit_indices)
            two_qubit_depth = tmp.depth()
        except Exception:
            two_qubit_depth = 0

    gate_density = total_gates / num_qubits if num_qubits > 0 else 0.0
    two_qubit_ratio = two_qubit_gates / total_gates if total_gates > 0 else 0.0

    return {
        "num_qubits": num_qubits,
        "depth": circuit.depth(),
        "two_qubit_gates": two_qubit_gates,
        "two_qubit_depth": two_qubit_depth,
        "total_gates": total_gates,
        "gate_density": gate_density,
        "two_qubit_ratio": two_qubit_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import GUOQ ibmnew circuits into a trajectory database"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmarks/ai_transpile/qasm/guoq_ibmnew"),
        help="Directory containing GUOQ ibmnew QASM files",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories_guoq_benchmark.db"),
        help="Path to the target trajectory database",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="artifact",
        help="Circuit category tag in the DB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report without writing to DB",
    )
    args = parser.parse_args()

    source_dir = args.source
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Run the clone command first:")
        print("  git clone --depth 1 https://github.com/qqq-wisc/guoq.git /tmp/guoq")
        print(f"  cp /tmp/guoq/benchmarks/ibmnew/*.qasm {source_dir}/")
        print("  rm -rf /tmp/guoq")
        sys.exit(1)

    qasm_files = sorted(source_dir.rglob("*.qasm"))
    if not qasm_files:
        print(f"No .qasm files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(qasm_files)} QASM files in {source_dir}")
    print(f"Target database: {args.database}")
    print(f"Category: {args.category}")
    if args.dry_run:
        print("DRY RUN — no DB writes")
    print()

    db = None if args.dry_run else TrajectoryDatabase(args.database)

    imported = 0
    skipped_parse = 0
    skipped_exists = 0

    circuits = pd.read_csv("./dbs/mqt/single_opt_master.csv")["opt_chain"].unique().tolist()


    for qasm_path in qasm_files:
        
        circuit_name = f"{qasm_path.parent.name}__{qasm_path.stem}"
        if circuit_name not in circuits:
            continue
        
        

        # Check for duplicates
        if db is not None:
            existing = db.get_circuit_by_name(circuit_name)
            if existing is not None:
                skipped_exists += 1
                continue

        metrics = _analyze_qasm(qasm_path)
        if metrics is None:
            skipped_parse += 1
            continue

        print(
            f"  {circuit_name}: {metrics['num_qubits']}q, "
            f"{metrics['two_qubit_gates']} 2Q gates, "
            f"depth={metrics['depth']}"
        )

        if db is not None:
            record = CircuitRecord(
                id=None,
                name=circuit_name,
                category=circuit_name.split("_")[0],
                source="QASMbench",
                qasm_path=str(qasm_path.resolve()),
                num_qubits=metrics["num_qubits"],
                initial_depth=metrics["depth"],
                initial_two_qubit_gates=metrics["two_qubit_gates"],
                initial_two_qubit_depth=metrics["two_qubit_depth"],
                initial_total_gates=metrics["total_gates"],
                gate_density=metrics["gate_density"],
                two_qubit_ratio=metrics["two_qubit_ratio"],
            )
            db.insert_circuit(record)

        imported += 1

    if db is not None:
        db.close()

    print("\nSummary:")
    print(f"  Imported:          {imported}")
    print(f"  Skipped (parse):   {skipped_parse}")
    print(f"  Skipped (exists):  {skipped_exists}")
    print(f"  Total processed:   {len(qasm_files)}")

    if not args.dry_run and imported > 0:
        print(f"\n{imported} circuits registered in {args.database}")
        print("Next: run grid search on these circuits with extended timeout:")
        print("  python scripts/run_single_step_grid_search.py \\")
        print(f"    --database {args.database} \\")
        print("    --wisq-bqskit-timeout 1800 \\")
        print("    --optimizers tket wisq_rules wisq_bqskit \\")
        print(f"    --circuit-category {args.category}")


if __name__ == "__main__":
    main()
