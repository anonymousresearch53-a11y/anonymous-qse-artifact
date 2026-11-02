#!/usr/bin/env python3
"""Build an RL master CSV from the hardware-aware first and second rounds."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "dbs" / "hardware_aware_combined"
FIRST_ROUND_DB = OUTPUT_DIR / "first_round" / "first_round.db"
SECOND_ROUND_DB = OUTPUT_DIR / "second_round" / "second_round.db"
RAW_MASTER_CSV = OUTPUT_DIR / "combined_master_raw.csv"
SUMMARY_JSON = OUTPUT_DIR / "combined_master_build_summary.json"

OPTIMIZER_ALIASES = {
    "qiskit_ai_hardware": "qiskit_ai",
    "qiskit_standard_hardware": "qiskit_standard",
    "tket_hardware": "tket",
}

RAW_COLUMNS = (
    "id",
    "original_circuit_id",
    "original_circuit",
    "original_category",
    "original_circuit_path",
    "parent_circuit_id",
    "parent_circuit",
    "parent_circuit_path",
    "chain_name",
    "artifact_path",
    "opt_level",
    "opt_1",
    "opt_2",
    "opt_3",
    "initial_num_qubits",
    "initial_depth",
    "initial_two_qubit_gates",
    "initial_total_gates",
    "initial_two_qubit_depth",
    "initial_gate_density",
    "initial_two_qubit_ratio",
    "final_num_qubits",
    "final_depth",
    "final_two_qubit_gates",
    "final_total_gates",
    "final_two_qubit_depth",
    "total_runtime_seconds",
    "current_step_runtime",
    "error",
    "opt_chain",
    "source_repository",
)


def load_rows(connection: sqlite3.Connection, query: str) -> list[dict[str, object]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(query)]


def metrics_from_original(circuit: dict[str, object]) -> dict[str, object]:
    return {
        "initial_num_qubits": circuit["num_qubits"],
        "initial_depth": circuit["initial_depth"],
        "initial_two_qubit_gates": circuit["initial_two_qubit_gates"],
        "initial_total_gates": circuit["initial_total_gates"],
        "initial_two_qubit_depth": circuit["initial_two_qubit_depth"],
        "initial_gate_density": circuit["gate_density"],
        "initial_two_qubit_ratio": circuit["two_qubit_ratio"],
        # Keep logical circuit size consistent across optimizer stages. Hardware
        # artifacts may declare a 133-qubit register to preserve physical IDs.
        "final_num_qubits": circuit["num_qubits"],
    }


def output_metrics(run: dict[str, object]) -> dict[str, object]:
    return {
        "final_depth": run["output_depth"],
        "final_two_qubit_gates": run["output_two_qubit_gates"],
        "final_total_gates": run["output_total_gates"],
        "final_two_qubit_depth": run["output_two_qubit_depth"],
    }


def require_artifact(path_value: object, context: str) -> str:
    path_text = str(path_value or "")
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing successful output artifact for {context}: {path_text}")
    return path_text


def build_master_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    if not FIRST_ROUND_DB.is_file() or not SECOND_ROUND_DB.is_file():
        raise FileNotFoundError("Expected first-round and second-round databases under dbs/hardware_aware_combined")

    first = sqlite3.connect(FIRST_ROUND_DB)
    second = sqlite3.connect(SECOND_ROUND_DB)
    try:
        first_circuits = load_rows(first, "SELECT * FROM circuits ORDER BY id")
        first_optimizer_names = {
            row[0]: row[1]
            for row in first.execute("SELECT id, name FROM optimizers")
        }
        first_runs = load_rows(
            first,
            """
            SELECT * FROM optimization_runs
            ORDER BY circuit_id, optimizer_id, id
            """,
        )
        second_circuits = load_rows(second, "SELECT * FROM circuits ORDER BY id")
        second_optimizer_names = {
            row[0]: row[1]
            for row in second.execute("SELECT id, name FROM optimizers")
        }
        second_runs = load_rows(
            second,
            """
            SELECT * FROM optimization_runs
            ORDER BY circuit_id, optimizer_id, id
            """,
        )
    finally:
        first.close()
        second.close()

    expected_optimizers = set(OPTIMIZER_ALIASES)
    if set(first_optimizer_names.values()) != expected_optimizers:
        raise ValueError(f"Unexpected first-round optimizers: {sorted(first_optimizer_names.values())}")
    if set(second_optimizer_names.values()) != expected_optimizers:
        raise ValueError(f"Unexpected second-round optimizers: {sorted(second_optimizer_names.values())}")

    first_runs_by_circuit: dict[int, list[dict[str, object]]] = defaultdict(list)
    for run in first_runs:
        run["optimizer_name"] = first_optimizer_names[run["optimizer_id"]]
        first_runs_by_circuit[run["circuit_id"]].append(run)

    second_circuit_by_name = {row["name"]: row for row in second_circuits}
    second_runs_by_circuit: dict[int, list[dict[str, object]]] = defaultdict(list)
    for run in second_runs:
        run["optimizer_name"] = second_optimizer_names[run["optimizer_id"]]
        second_runs_by_circuit[run["circuit_id"]].append(run)

    rows: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = defaultdict(list)
    eligible_count = 0

    for original in first_circuits:
        runs = first_runs_by_circuit.get(original["id"], [])
        if not runs:
            exclusions["no_first_round_runs"] += 1
            continue
        if len(runs) != len(OPTIMIZER_ALIASES):
            exclusions["incomplete_first_round"] += 1
            continue

        runs_by_optimizer = {run["optimizer_name"]: run for run in runs}
        if set(runs_by_optimizer) != expected_optimizers:
            exclusions["incomplete_first_round"] += 1
            continue
        if any(not run["success"] for run in runs):
            exclusions["first_round_failure"] += 1
            failure_examples["first_round_failure"].append(str(original["name"]))
            continue

        if original["initial_two_qubit_gates"] <= 0:
            exclusions["zero_initial_two_qubit_gates"] += 1
            continue

        chain_runs: dict[tuple[str, str], dict[str, object]] = {}
        failed_second_round = False
        incomplete_second_round = False
        for first_optimizer, first_run in runs_by_optimizer.items():
            seed_name = f"{original['name']}__{first_optimizer}"
            seed_circuit = second_circuit_by_name.get(seed_name)
            if seed_circuit is None:
                incomplete_second_round = True
                break

            seed_runs = second_runs_by_circuit.get(seed_circuit["id"], [])
            seed_runs_by_optimizer = {run["optimizer_name"]: run for run in seed_runs}
            if (
                len(seed_runs) != len(OPTIMIZER_ALIASES)
                or set(seed_runs_by_optimizer) != expected_optimizers
            ):
                incomplete_second_round = True
                break
            if any(not run["success"] for run in seed_runs):
                failed_second_round = True
                break

            for second_optimizer, run in seed_runs_by_optimizer.items():
                chain_runs[(first_optimizer, second_optimizer)] = run

        if incomplete_second_round:
            exclusions["incomplete_second_round"] += 1
            continue
        if failed_second_round:
            exclusions["second_round_failure"] += 1
            failure_examples["second_round_failure"].append(str(original["name"]))
            continue

        eligible_count += 1
        initial = metrics_from_original(original)
        source_repository = str(original["source"] or "unknown")

        for first_optimizer in OPTIMIZER_ALIASES:
            run = runs_by_optimizer[first_optimizer]
            optimizer = OPTIMIZER_ALIASES[first_optimizer]
            artifact_path = require_artifact(run["artifact_path"], f"{original['name']} {first_optimizer}")
            rows.append(
                {
                    "original_circuit_id": original["id"],
                    "original_circuit": original["name"],
                    "original_category": original["category"],
                    "original_circuit_path": original["qasm_path"],
                    "parent_circuit_id": original["id"],
                    "parent_circuit": original["name"],
                    "parent_circuit_path": original["qasm_path"],
                    "chain_name": optimizer,
                    "artifact_path": artifact_path,
                    "opt_level": 1,
                    "opt_1": optimizer,
                    "opt_2": "",
                    "opt_3": "",
                    **initial,
                    **output_metrics(run),
                    "total_runtime_seconds": run["duration_seconds"],
                    "current_step_runtime": run["duration_seconds"],
                    "error": False,
                    "opt_chain": f"{original['name']}__{optimizer}",
                    "source_repository": source_repository,
                }
            )

        for first_optimizer in OPTIMIZER_ALIASES:
            first_run = runs_by_optimizer[first_optimizer]
            first_label = OPTIMIZER_ALIASES[first_optimizer]
            parent_path = require_artifact(
                first_run["artifact_path"], f"{original['name']} {first_optimizer} parent"
            )
            seed_name = f"{original['name']}__{first_optimizer}"
            seed_circuit = second_circuit_by_name[seed_name]
            for second_optimizer in OPTIMIZER_ALIASES:
                run = chain_runs[(first_optimizer, second_optimizer)]
                second_label = OPTIMIZER_ALIASES[second_optimizer]
                artifact_path = require_artifact(
                    run["artifact_path"],
                    f"{original['name']} {first_optimizer} -> {second_optimizer}",
                )
                rows.append(
                    {
                        "original_circuit_id": original["id"],
                        "original_circuit": original["name"],
                        "original_category": original["category"],
                        "original_circuit_path": original["qasm_path"],
                        "parent_circuit_id": seed_circuit["id"],
                        "parent_circuit": f"{original['name']}__{first_label}",
                        "parent_circuit_path": parent_path,
                        "chain_name": f"{first_label}__{second_label}",
                        "artifact_path": artifact_path,
                        "opt_level": 2,
                        "opt_1": first_label,
                        "opt_2": second_label,
                        "opt_3": "",
                        **initial,
                        **output_metrics(run),
                        "total_runtime_seconds": (
                            first_run["duration_seconds"] + run["duration_seconds"]
                        ),
                        "current_step_runtime": run["duration_seconds"],
                        "error": False,
                        "opt_chain": f"{original['name']}__{first_label}__{second_label}",
                        "source_repository": source_repository,
                    }
                )

    for row_id, row in enumerate(rows):
        row["id"] = row_id

    summary = {
        "first_round_circuit_records": len(first_circuits),
        "first_round_optimizer_runs": len(first_runs),
        "second_round_starting_circuits": len(second_circuits),
        "second_round_optimizer_runs": len(second_runs),
        "optimizers": [OPTIMIZER_ALIASES[name] for name in OPTIMIZER_ALIASES],
        "eligible_original_circuits": eligible_count,
        "rows_written": len(rows),
        "expected_rows_per_circuit": len(OPTIMIZER_ALIASES) + len(OPTIMIZER_ALIASES) ** 2,
        "exclusions": dict(exclusions),
        "failed_circuit_examples": dict(failure_examples),
    }
    return rows, summary


def main() -> None:
    rows, summary = build_master_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_MASTER_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows for {summary['eligible_original_circuits']} circuits to {RAW_MASTER_CSV}")
    print(f"Exclusions: {summary['exclusions']}")


if __name__ == "__main__":
    main()
