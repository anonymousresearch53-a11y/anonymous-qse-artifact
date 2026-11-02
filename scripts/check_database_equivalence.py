#!/usr/bin/env python3
"""Audit optimized circuits in the master CSV with a practical equivalence check.

This implementation is intentionally self-contained and avoids the repo's older
comparison helpers. It uses two modes:

1. Exact operator equivalence for smaller circuits.
2. High-confidence randomized statevector agreement for larger circuits.

The goal is to be robust and efficient enough for large CSV audits without the
memory blow-ups caused by full unitary construction on bigger circuits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS
from qiskit.quantum_info import Operator, Statevector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_CSV = PROJECT_ROOT / "dbs" / "combined_master.csv"


@dataclass(frozen=True)
class CheckConfig:
    """Configuration for the equivalence audit."""

    atol: float = 1e-10
    exact_qubits: int = 11
    sampled_qubits: int = 16
    random_trials: int = 2
    seed: int = 1234
    per_check_timeout_seconds: float = 0.0


class CheckTimeoutError(TimeoutError):
    """Raised when one row-level equivalence check exceeds the configured timeout."""


def _timeout_handler(_signum: int, _frame: Any) -> None:
    """Signal handler for row-level timeouts."""
    raise CheckTimeoutError("Equivalence check timed out")


class per_check_timeout:
    """Best-effort Unix signal timeout for one row-level audit."""

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, float(seconds))
        self._enabled = self.seconds > 0 and hasattr(signal, "setitimer")
        self._previous_handler: Any = None

    def __enter__(self) -> None:
        if not self._enabled:
            return
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> bool:
        if not self._enabled:
            return False
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        return False


def load_qasm_circuit(path: Path) -> QuantumCircuit:
    """Load a QASM circuit using the same legacy instruction compatibility as the repo."""
    return qasm2.load(path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)


def normalize_for_comparison(circuit: QuantumCircuit) -> QuantumCircuit:
    """Prepare a circuit for semantic comparison.

    Final measurements are removed so the comparison focuses on the unitary part
    of the circuit. This matches the optimization setting used in the repo.
    """
    normalized = circuit.remove_final_measurements(inplace=False)
    return normalized


def _permutation_circuit(logical_to_physical: list[int]) -> QuantumCircuit:
    circuit = QuantumCircuit(len(logical_to_physical))
    labels = list(range(len(logical_to_physical)))
    for logical, physical in enumerate(logical_to_physical):
        current_physical = labels.index(logical)
        if current_physical != physical:
            circuit.swap(current_physical, physical)
            labels[current_physical], labels[physical] = labels[physical], labels[current_physical]
    return circuit


def restore_logical_wire_order(
    circuit: QuantumCircuit, layout: dict[str, Any]
) -> QuantumCircuit:
    """Convert a mapped circuit back to the original logical wire order."""
    input_layout = [int(index) for index in layout["input_logical_to_physical"]]
    output_layout = [int(index) for index in layout["output_logical_to_physical"]]
    if len(input_layout) != circuit.num_qubits or len(output_layout) != circuit.num_qubits:
        raise ValueError("Layout width does not match the circuit width")
    if sorted(input_layout) != list(range(circuit.num_qubits)):
        raise ValueError("Input layout is not a permutation of the circuit wires")
    if sorted(output_layout) != list(range(circuit.num_qubits)):
        raise ValueError("Output layout is not a permutation of the circuit wires")
    physical_to_logical = [0] * circuit.num_qubits
    for logical, physical in enumerate(output_layout):
        physical_to_logical[physical] = logical
    return (
        _permutation_circuit(input_layout)
        .compose(circuit)
        .compose(_permutation_circuit(physical_to_logical))
    )


def load_hardware_layout(record: dict[str, Any]) -> dict[str, Any] | None:
    """Load the logical-to-physical layout stored in run metadata or an artifact sidecar."""
    metadata_raw = record.get("metadata_json")
    if metadata_raw:
        try:
            metadata = json.loads(str(metadata_raw))
            layout = metadata.get("layout")
            if isinstance(layout, dict):
                return layout
        except json.JSONDecodeError:
            pass

    artifact_path = Path(record["artifact_path"]) if record.get("artifact_path") else None
    if artifact_path is None:
        return None
    layout_path = artifact_path.with_suffix(".layout.json")
    if not layout_path.exists():
        return None
    sidecar = json.loads(layout_path.read_text(encoding="utf-8"))
    expected_hash = sidecar.get("artifact_qasm_sha256")
    actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("Artifact QASM does not match its layout sidecar")
    layout = sidecar.get("layout")
    return layout if isinstance(layout, dict) else None


def is_hardware_aware(record: dict[str, Any]) -> bool:
    metadata_raw = record.get("metadata_json")
    if not metadata_raw:
        return False
    try:
        return bool(json.loads(str(metadata_raw)).get("hardware_aware"))
    except json.JSONDecodeError:
        return False


def parse_success(row: dict[str, str]) -> bool:
    """Infer whether a master-CSV row represents a successful optimization."""
    value = str(row.get("error", "")).strip()
    return value in {"", "False", "false", "0", "nan", "None", "none"}


def parse_float(row: dict[str, str], key: str) -> float:
    """Read a float from a CSV row safely."""
    value = str(row.get(key, "")).strip()
    if value in {"", "nan", "None", "none"}:
        return 0.0


def parse_int(row: dict[str, str], key: str) -> int | None:
    """Read an integer from a CSV row safely."""
    value = str(row.get(key, "")).strip()
    if value in {"", "nan", "None", "none"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
    try:
        return float(value)
    except ValueError:
        return 0.0


def resolve_data_path(raw_path: str | None, csv_path: Path) -> Path | None:
    """Resolve either an absolute or repo-relative data path."""
    if not raw_path:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidate_paths = [
        PROJECT_ROOT / path,
        csv_path.parent / path,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def build_random_product_state(num_qubits: int, rng: np.random.Generator) -> QuantumCircuit:
    """Build a random product-state preparation circuit.

    Product states are cheap to prepare and already catch most real semantic
    mismatches while keeping the check much cheaper than full operator methods.
    """
    prep = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        theta = float(rng.uniform(0.0, math.pi))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        lam = float(rng.uniform(0.0, 2.0 * math.pi))
        prep.u(theta, phi, lam, qubit)
    return prep


def state_fidelity(circuit_a: QuantumCircuit, circuit_b: QuantumCircuit) -> float:
    """Return state fidelity between two state-preparation circuits."""
    state_a = Statevector.from_instruction(circuit_a)
    state_b = Statevector.from_instruction(circuit_b)
    return float(abs(np.vdot(state_a.data, state_b.data)) ** 2)


def exact_operator_equivalent(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    atol: float,
) -> tuple[bool, float]:
    """Check exact equivalence via full unitary operator construction."""
    op_a = Operator(circuit_a)
    op_b = Operator(circuit_b)
    equivalent = bool(op_a.equiv(op_b, atol=atol))
    dim = op_a.data.shape[0]
    fidelity = float((abs(np.trace(op_a.data.conj().T @ op_b.data)) ** 2) / (dim * dim))
    return equivalent, fidelity


def sampled_statevector_equivalent(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    config: CheckConfig,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """Check high-confidence equivalence using several random product-state inputs."""
    min_fidelity = 1.0

    # Always include the all-zero input first.
    base_fidelity = state_fidelity(circuit_a, circuit_b)
    min_fidelity = min(min_fidelity, base_fidelity)
    if base_fidelity < (1.0 - config.atol):
        return False, min_fidelity

    for _ in range(config.random_trials):
        prep = build_random_product_state(circuit_a.num_qubits, rng)
        test_a = prep.compose(circuit_a)
        test_b = prep.compose(circuit_b)
        fidelity = state_fidelity(test_a, test_b)
        min_fidelity = min(min_fidelity, fidelity)
        if fidelity < (1.0 - config.atol):
            return False, min_fidelity

    return True, min_fidelity


def compare_circuits_practical(
    original: QuantumCircuit,
    optimized: QuantumCircuit,
    config: CheckConfig,
    rng: np.random.Generator,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two circuits with an exact-small / sampled-large strategy."""
    original_cmp = normalize_for_comparison(original)
    optimized_cmp = normalize_for_comparison(optimized)
    if layout is not None:
        optimized_cmp = restore_logical_wire_order(optimized_cmp, layout)

    if original_cmp.num_qubits != optimized_cmp.num_qubits:
        return {
            "equivalent": False,
            "method": "qubit_mismatch",
            "fidelity": None,
            "error": (
                f"Different qubit counts: {original_cmp.num_qubits} vs "
                f"{optimized_cmp.num_qubits}"
            ),
        }

    num_qubits = original_cmp.num_qubits

    try:
        if num_qubits <= config.exact_qubits:
            equivalent, fidelity = exact_operator_equivalent(
                original_cmp,
                optimized_cmp,
                atol=config.atol,
            )
            return {
                "equivalent": equivalent,
                "method": "exact_operator",
                "fidelity": fidelity,
                "error": None,
            }

        if num_qubits <= config.sampled_qubits:
            equivalent, fidelity = sampled_statevector_equivalent(
                original_cmp,
                optimized_cmp,
                config=config,
                rng=rng,
            )
            return {
                "equivalent": equivalent,
                "method": "sampled_statevector",
                "fidelity": fidelity,
                "error": None,
            }

        return {
            "equivalent": None,
            "method": "skipped_too_many_qubits",
            "fidelity": None,
            "error": (
                f"Circuit has {num_qubits} qubits, above sampled limit "
                f"{config.sampled_qubits}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "equivalent": None,
            "method": "comparison_failed",
            "fidelity": None,
            "error": str(exc),
        }


def fetch_master_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the master CSV with only the standard library."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "original_circuit",
            "original_circuit_path",
            "artifact_path",
            "chain_name",
            "error",
        }
        missing = required - fieldnames
        if missing:
            raise RuntimeError(
                f"Master CSV {csv_path} is missing required columns: {sorted(missing)}"
            )
        return list(reader)


def fetch_database_rows(
    db_path: Path,
    optimizer_name: str | None = None,
    optimizer_id: int | None = None,
) -> list[dict[str, str]]:
    """Read optimization-run rows from a trajectory SQLite database.

    The returned records are shaped to match the master-CSV field names used by
    the rest of this script so the existing filtering and reporting code can be
    reused without branching everywhere.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Trajectory database not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    conditions: list[str] = []
    params: list[Any] = []
    if optimizer_name is not None:
        conditions.append("o.name = ?")
        params.append(str(optimizer_name))
    if optimizer_id is not None:
        conditions.append("o.id = ?")
        params.append(int(optimizer_id))

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"""
        SELECT
            c.name AS original_circuit,
            c.category AS original_category,
            c.source AS source_repository,
            c.qasm_path AS original_circuit_path,
            o.name AS chain_name,
            r.artifact_path AS artifact_path,
            r.metadata_json AS metadata_json,
            r.duration_seconds AS total_runtime_seconds,
            CASE
                WHEN COALESCE(r.success, 0) = 1 THEN ''
                WHEN COALESCE(r.error_message, '') <> '' THEN r.error_message
                ELSE 'True'
            END AS error,
            1 AS num_optimizers_used
        FROM optimization_runs r
        JOIN circuits c ON c.id = r.circuit_id
        JOIN optimizers o ON o.id = r.optimizer_id
        {where_clause}
        ORDER BY r.id
        """,
        tuple(params),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def row_matches_filters(
    row: dict[str, str],
    source_repository: str | None,
    circuit_prefix: str | None,
    single_only: bool,
    include_qiskit_ai: bool,
) -> bool:
    """Return whether a master-CSV row passes the requested filters."""
    chain_name = str(row.get("chain_name", "")).strip()
    if not include_qiskit_ai and "qiskit_ai" in chain_name:
        return False
    if source_repository is not None:
        source_value = str(row.get("source_repository", "")).strip().lower()
        if source_value != source_repository.strip().lower():
            return False
    if circuit_prefix is not None:
        circuit_name = str(row.get("original_circuit", "")).strip().lower()
        if not circuit_name.startswith(circuit_prefix.strip().lower()):
            return False
    if single_only:
        num_optimizers_used = parse_int(row, "num_optimizers_used")
        if num_optimizers_used is not None:
            if num_optimizers_used != 1:
                return False
        else:
            if "__" in chain_name:
                return False
    return True


def make_result_record(row_index: int, row: dict[str, str], csv_path: Path) -> dict[str, Any]:
    """Convert one CSV row into a mutable audit record."""
    return {
        "row_index": row_index,
        "circuit_name": str(row.get("original_circuit", "")),
        "category": str(row.get("original_category", "")),
        "source": str(row.get("source_repository", "")),
        "optimizer_name": str(row.get("chain_name", "")),
        "success": parse_success(row),
        "error_message": row.get("error"),
        "artifact_path": str(resolve_data_path(row.get("artifact_path"), csv_path) or ""),
        "original_qasm_path": str(
            resolve_data_path(row.get("original_circuit_path"), csv_path) or ""
        ),
        "duration_seconds": parse_float(row, "total_runtime_seconds"),
        "metadata_json": str(row.get("metadata_json", "")),
    }


def print_progress(index: int, total: int, record: dict[str, Any]) -> None:
    """Print one concise progress update."""
    circuit = record["circuit_name"]
    optimizer = record["optimizer_name"]
    status = record["status"]
    if status == "checked":
        verdict = "equivalent" if record["equivalent"] else "NOT equivalent"
        fidelity = record["fidelity"]
        fidelity_text = f" | fidelity={fidelity:.6f}" if fidelity is not None else ""
        print(
            f"[{index}/{total}] {circuit} | {optimizer} | {verdict} | "
            f"method={record['comparison_method']}{fidelity_text}",
            flush=True,
        )
    else:
        print(
            f"[{index}/{total}] {circuit} | {optimizer} | skipped ({status})",
            flush=True,
        )


def print_progress_start(index: int, total: int, record: dict[str, Any]) -> None:
    """Print a start line before a row is processed."""
    print(
        f"[{index}/{total}] starting | {record['circuit_name']} | {record['optimizer_name']}",
        flush=True,
    )


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print the final audit summary."""
    total = len(results)
    checked = [row for row in results if row["status"] == "checked"]
    equivalent = [row for row in checked if row["equivalent"] is True]
    non_equivalent = [row for row in checked if row["equivalent"] is False]
    skipped = [row for row in results if row["status"] != "checked"]

    print(f"Total runs: {total}")
    print(f"Checked: {len(checked)}")
    print(f"Equivalent: {len(equivalent)}")
    print(f"Non-equivalent: {len(non_equivalent)}")
    print(f"Skipped/uncheckable: {len(skipped)}")

    method_counts: dict[str, int] = {}
    for row in checked:
        method = str(row.get("comparison_method"))
        method_counts[method] = method_counts.get(method, 0) + 1
    if method_counts:
        print("\nChecked by method:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method}: {count}")

    if skipped:
        skipped_counts: dict[str, int] = {}
        for row in skipped:
            status = str(row.get("status"))
            skipped_counts[status] = skipped_counts.get(status, 0) + 1
        print("\nSkipped breakdown:")
        for status, count in sorted(skipped_counts.items()):
            print(f"  {status}: {count}")

    if non_equivalent:
        print("\nNon-equivalent runs:")
        for row in non_equivalent[:25]:
            print(
                f"  row_index={row['row_index']} | circuit={row['circuit_name']} | "
                f"optimizer={row['optimizer_name']} | method={row['comparison_method']} | "
                f"fidelity={row['fidelity']} | error={row['comparison_error']}"
            )
        if len(non_equivalent) > 25:
            print(f"  ... and {len(non_equivalent) - 25} more")


def audit_record(
    record: dict[str, Any],
    config: CheckConfig,
    rng: np.random.Generator,
    circuit_cache: dict[str, QuantumCircuit],
) -> dict[str, Any]:
    """Audit one master-CSV row."""
    record["equivalent"] = None
    record["comparison_method"] = None
    record["fidelity"] = None
    record["comparison_error"] = None
    record["status"] = "unchecked"

    if not record["success"]:
        record["status"] = "run_failed"
        return record

    original_path = Path(record["original_qasm_path"]) if record["original_qasm_path"] else None
    artifact_path = Path(record["artifact_path"]) if record["artifact_path"] else None
    if original_path is None or not str(original_path):
        record["status"] = "missing_original_path"
        return record
    if artifact_path is None or not str(artifact_path):
        record["status"] = "missing_artifact_path"
        return record
    if not original_path.exists():
        record["status"] = "missing_original_file"
        return record
    if not artifact_path.exists():
        record["status"] = "missing_artifact_file"
        return record

    try:
        with per_check_timeout(config.per_check_timeout_seconds):
            try:
                original_key = str(original_path.resolve())
                if original_key not in circuit_cache:
                    circuit_cache[original_key] = load_qasm_circuit(original_path)
                original = circuit_cache[original_key]
            except Exception as exc:  # noqa: BLE001
                record["status"] = "original_load_failed"
                record["comparison_error"] = str(exc)
                return record

            try:
                optimized = load_qasm_circuit(artifact_path)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "artifact_load_failed"
                record["comparison_error"] = str(exc)
                return record

            try:
                layout = load_hardware_layout(record)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "layout_load_failed"
                record["comparison_error"] = str(exc)
                return record
            if is_hardware_aware(record) and layout is None:
                record["status"] = "missing_hardware_layout"
                record["comparison_error"] = "Mapped result has no saved wire layout"
                return record
            result = compare_circuits_practical(
                original, optimized, config=config, rng=rng, layout=layout
            )
            if result["equivalent"] is None:
                record["status"] = str(result["method"])
                record["comparison_method"] = str(result["method"])
                record["comparison_error"] = result["error"]
                return record

            record["equivalent"] = bool(result["equivalent"])
            record["comparison_method"] = str(result["method"])
            record["fidelity"] = result["fidelity"]
            record["comparison_error"] = result["error"]
            record["status"] = "checked"
            return record
    except CheckTimeoutError as exc:
        record["status"] = "timed_out"
        record["comparison_error"] = str(exc)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit master-CSV optimized circuits with a practical equivalence checker."
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        default=DEFAULT_MASTER_CSV,
        help="Path to the combined master CSV.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Optional trajectory SQLite database to audit instead of the master CSV.",
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default=None,
        help="Optional optimizer name filter when reading from --database.",
    )
    parser.add_argument(
        "--optimizer-id",
        type=int,
        default=None,
        help="Optional optimizer ID filter when reading from --database.",
    )
    parser.add_argument(
        "--only-successful",
        action="store_true",
        help="Only audit rows that marked the optimization run as successful.",
    )
    parser.add_argument(
        "--source-repository",
        type=str,
        default=None,
        help="Optional source_repository filter, for example 'mqt' or 'guoq'.",
    )
    parser.add_argument(
        "--circuit-prefix",
        type=str,
        default=None,
        help="Optional original_circuit prefix filter, for example 'mqt_'.",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only audit single-optimizer rows (exclude multi-step chains).",
    )
    parser.add_argument(
        "--include-qiskit-ai",
        action="store_true",
        help="Include rows whose optimizer chain contains qiskit_ai. By default they are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for spot checks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress update after every N processed rows.",
    )
    parser.add_argument(
        "--exact-qubits",
        type=int,
        default=11,
        help="Use exact operator equivalence at or below this qubit count.",
    )
    parser.add_argument(
        "--sampled-qubits",
        type=int,
        default=16,
        help="Use sampled statevector checks up to this qubit count; above it, skip.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=2,
        help="Number of random product-state trials for sampled checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Tolerance used for exact and sampled agreement checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampled checks.",
    )
    parser.add_argument(
        "--per-check-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout per row-level check in seconds (0 disables).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full audit results as JSON.",
    )
    args = parser.parse_args()

    if args.database is not None:
        rows = fetch_database_rows(
            args.database,
            optimizer_name=args.optimizer_name,
            optimizer_id=args.optimizer_id,
        )
        source_path = args.database
    else:
        rows = fetch_master_rows(args.master_csv)
        source_path = args.master_csv
    if args.source_repository is not None or args.circuit_prefix is not None:
        rows = [
            row
            for row in rows
            if row_matches_filters(
                row,
                source_repository=args.source_repository,
                circuit_prefix=args.circuit_prefix,
                single_only=bool(args.single_only),
                include_qiskit_ai=bool(args.include_qiskit_ai),
            )
        ]
    elif args.single_only:
        rows = [
            row
            for row in rows
            if row_matches_filters(
                row,
                source_repository=None,
                circuit_prefix=None,
                single_only=True,
                include_qiskit_ai=bool(args.include_qiskit_ai),
            )
        ]
    elif not args.include_qiskit_ai:
        rows = [
            row
            for row in rows
            if row_matches_filters(
                row,
                source_repository=None,
                circuit_prefix=None,
                single_only=False,
                include_qiskit_ai=False,
            )
        ]
    records = [
        make_result_record(index, row, source_path)
        for index, row in enumerate(rows, start=2)
    ]
    if args.only_successful:
        records = [record for record in records if record["success"]]
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]

    total = len(records)
    print(
        f"Starting equivalence audit | rows={total} | exact_qubits={args.exact_qubits} | "
        f"sampled_qubits={args.sampled_qubits} | random_trials={args.random_trials}",
        flush=True,
    )

    config = CheckConfig(
        atol=float(args.atol),
        exact_qubits=int(args.exact_qubits),
        sampled_qubits=int(args.sampled_qubits),
        random_trials=int(args.random_trials),
        seed=int(args.seed),
        per_check_timeout_seconds=float(args.per_check_timeout_seconds),
    )
    rng = np.random.default_rng(config.seed)
    circuit_cache: dict[str, QuantumCircuit] = {}
    progress_every = max(1, int(args.progress_every))

    audited: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        print_progress_start(index, total, record)
        audited_record = audit_record(record, config=config, rng=rng, circuit_cache=circuit_cache)
        audited.append(audited_record)
        if progress_every == 1 or index % progress_every == 0 or index == total:
            print_progress(index, total, audited_record)

    print_summary(audited)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audited, indent=2, default=str))
        print(f"\nWrote JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
