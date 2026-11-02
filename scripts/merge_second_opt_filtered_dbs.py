#!/usr/bin/env python3
"""Merge two step-2 optimization DBs while rebuilding tket input circuits.

This script creates a fresh SQLite database that:
- merges the two source step-2 databases
- removes artifact circuits whose chain contains an excluded optimizer
- removes optimization runs whose input circuit chain contains an excluded optimizer
- removes optimization runs whose optimizer is excluded
- re-adds corrected single-step tket artifact circuits from a trusted single-step DB

The resulting DB is useful when you want a clean step-2 database without the
old qiskit_ai/tket-derived second-step runs, while still keeping corrected
tket-generated circuits available as valid inputs for future reruns.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DBS = [
    PROJECT_ROOT / "dbs" / "mqt" / "data" / "mqt_two_opt_trajectories.db",
    PROJECT_ROOT / "dbs" / "original_data" / "data_second_opt" / "trajectories.db",
]
DEFAULT_SINGLE_DB = (
    PROJECT_ROOT / "dbs" / "combined_fixed" / "data_single_opt" / "combined_single_opt.db"
)
DEFAULT_OUTPUT_DB = (
    PROJECT_ROOT / "dbs" / "combined_fixed" / "data_second_opt" / "combined_second_opt.db"
)
DEFAULT_EXCLUDED_OPTIMIZERS = ["qiskit_ai", "tket"]
OLD_QASM_PREFIX = "./"
NEW_QASM_PREFIX = "./"


def load_schema_sql() -> str:
    """Load the canonical SQLite schema without importing the full package."""
    module_path = (
        PROJECT_ROOT / "benchmarks" / "ai_transpile" / "rl_trajectory" / "database.py"
    )
    spec = importlib.util.spec_from_file_location("rl_trajectory_database", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load schema module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SCHEMA_SQL


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Run a SELECT and return all rows."""
    return conn.execute(query, params).fetchall()


def normalize_path(path_value: str | None) -> str | None:
    """Rewrite legacy absolute paths and repo-relative artifact paths."""
    if path_value is None:
        return None

    path_text = str(path_value).strip()
    if not path_text:
        return None

    if path_text.startswith(OLD_QASM_PREFIX):
        suffix = path_text.removeprefix(OLD_QASM_PREFIX)
        return f"{NEW_QASM_PREFIX}{suffix}"

    path = Path(path_text)
    if path.is_absolute():
        return str(path)

    return str((PROJECT_ROOT / path).resolve())


def create_output_db(output_path: Path, schema_sql: str) -> sqlite3.Connection:
    """Create a fresh output DB using the standard schema plus original_circuit."""
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(output_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(schema_sql)
    conn.execute("ALTER TABLE circuits ADD COLUMN original_circuit TEXT")
    conn.execute("CREATE INDEX idx_circuits_original_circuit ON circuits(original_circuit)")
    conn.commit()
    return conn


def name_contains_excluded_optimizer(name: str | None, excluded_optimizers: set[str]) -> bool:
    """Return True when a chain-style circuit/trajectory name includes an excluded optimizer."""
    if name is None:
        return False
    parts = [part for part in str(name).split("__")[1:] if part]
    return any(part in excluded_optimizers for part in parts)


def infer_original_circuit_name(circuit_name: str, explicit_original: str | None = None) -> str:
    """Infer the original circuit name for either original or artifact circuit rows."""
    if explicit_original is not None and str(explicit_original).strip():
        return str(explicit_original).strip()
    return str(circuit_name).split("__")[0]


def insert_circuit_row(
    out_conn: sqlite3.Connection,
    *,
    name: str,
    category: str,
    source: str,
    qasm_path: str | None,
    num_qubits: int,
    initial_depth: int,
    initial_two_qubit_gates: int,
    initial_two_qubit_depth: int,
    initial_total_gates: int,
    gate_density: float,
    two_qubit_ratio: float,
    original_circuit: str,
    created_at: str | None,
) -> int:
    """Insert one circuit row and return its new ID."""
    cursor = out_conn.execute(
        """
        INSERT INTO circuits (
            name, category, source, qasm_path, num_qubits,
            initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
            initial_total_gates, gate_density, two_qubit_ratio,
            created_at, original_circuit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
        """,
        (
            name,
            category,
            source,
            qasm_path,
            num_qubits,
            initial_depth,
            initial_two_qubit_gates,
            initial_two_qubit_depth,
            initial_total_gates,
            gate_density,
            two_qubit_ratio,
            created_at,
            original_circuit,
        ),
    )
    return int(cursor.lastrowid)


def add_corrected_tket_circuits(
    out_conn: sqlite3.Connection,
    single_db_path: Path,
) -> int:
    """Rebuild single-step tket artifact circuits from the fixed single-step DB."""
    src_conn = sqlite3.connect(f"file:{single_db_path}?mode=ro", uri=True)
    src_conn.row_factory = sqlite3.Row

    tket_rows = fetch_all(
        src_conn,
        """
        SELECT
            c.name AS original_name,
            c.category,
            r.artifact_path,
            c.num_qubits,
            r.output_depth,
            r.output_two_qubit_gates,
            r.output_two_qubit_depth,
            r.output_total_gates,
            r.created_at
        FROM optimization_runs r
        JOIN circuits c ON c.id = r.circuit_id
        JOIN optimizers o ON o.id = r.optimizer_id
        WHERE o.name = 'tket'
          AND r.success = 1
          AND r.artifact_path IS NOT NULL
          AND TRIM(r.artifact_path) <> ''
        ORDER BY c.name
        """,
    )

    inserted = 0
    for row in tket_rows:
        artifact_name = f"{row['original_name']}__tket"
        existing = out_conn.execute(
            "SELECT id FROM circuits WHERE name = ?",
            (artifact_name,),
        ).fetchone()
        if existing is not None:
            continue

        output_total_gates = int(row["output_total_gates"])
        num_qubits = int(row["num_qubits"])
        output_two_qubit_gates = int(row["output_two_qubit_gates"])
        gate_density = (
            float(output_total_gates) / float(num_qubits)
            if num_qubits > 0
            else 0.0
        )
        two_qubit_ratio = (
            float(output_two_qubit_gates) / float(output_total_gates)
            if output_total_gates > 0
            else 0.0
        )

        insert_circuit_row(
            out_conn,
            name=artifact_name,
            category=str(row["category"]),
            source="artifact",
            qasm_path=normalize_path(row["artifact_path"]),
            num_qubits=num_qubits,
            initial_depth=int(row["output_depth"]),
            initial_two_qubit_gates=output_two_qubit_gates,
            initial_two_qubit_depth=int(row["output_two_qubit_depth"]),
            initial_total_gates=output_total_gates,
            gate_density=gate_density,
            two_qubit_ratio=two_qubit_ratio,
            original_circuit=str(row["original_name"]),
            created_at=row["created_at"],
        )
        inserted += 1

    out_conn.commit()
    src_conn.close()
    return inserted


def merge_databases(
    source_paths: list[Path],
    single_db_path: Path,
    output_path: Path,
    excluded_optimizers: set[str],
) -> None:
    """Merge source DBs into one filtered output DB and rebuild tket circuits."""
    out_conn = create_output_db(output_path, load_schema_sql())

    total_circuits_inserted = 0
    total_runs_inserted = 0
    total_trajectories_inserted = 0
    total_steps_inserted = 0

    for source_path in source_paths:
        print(f"\nProcessing {source_path}")
        src_conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        src_conn.row_factory = sqlite3.Row

        source_has_original_circuit = any(
            row["name"] == "original_circuit"
            for row in fetch_all(src_conn, "PRAGMA table_info(circuits)")
        )

        src_optimizers = fetch_all(
            src_conn,
            "SELECT id, name, runner_type, options_json, description FROM optimizers",
        )

        optimizer_id_map: dict[int, int] = {}
        allowed_optimizer_names: set[str] = set()
        for opt in src_optimizers:
            name = str(opt["name"])
            if name in excluded_optimizers:
                continue

            existing = out_conn.execute(
                "SELECT id FROM optimizers WHERE name = ?",
                (name,),
            ).fetchone()
            if existing is None:
                cursor = out_conn.execute(
                    """
                    INSERT INTO optimizers (name, runner_type, options_json, description)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        opt["runner_type"],
                        opt["options_json"],
                        opt["description"],
                    ),
                )
                out_id = int(cursor.lastrowid)
            else:
                out_id = int(existing["id"])

            optimizer_id_map[int(opt["id"])] = out_id
            allowed_optimizer_names.add(name)

        circuit_select = """
            SELECT id, name, category, source, qasm_path, num_qubits,
                   initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                   initial_total_gates, gate_density, two_qubit_ratio, created_at
        """
        if source_has_original_circuit:
            circuit_select += ", original_circuit"
        else:
            circuit_select += ", NULL AS original_circuit"
        circuit_select += " FROM circuits"

        src_circuits = fetch_all(src_conn, circuit_select)

        circuit_id_map: dict[int, int] = {}
        circuits_inserted_this_db = 0
        circuits_skipped_for_excluded_name = 0
        for circ in src_circuits:
            circuit_name = str(circ["name"])
            if name_contains_excluded_optimizer(circuit_name, excluded_optimizers):
                circuits_skipped_for_excluded_name += 1
                continue

            existing = out_conn.execute(
                "SELECT id FROM circuits WHERE name = ?",
                (circuit_name,),
            ).fetchone()
            if existing is None:
                out_id = insert_circuit_row(
                    out_conn,
                    name=circuit_name,
                    category=str(circ["category"]),
                    source=str(circ["source"]),
                    qasm_path=normalize_path(circ["qasm_path"]),
                    num_qubits=int(circ["num_qubits"]),
                    initial_depth=int(circ["initial_depth"]),
                    initial_two_qubit_gates=int(circ["initial_two_qubit_gates"]),
                    initial_two_qubit_depth=int(circ["initial_two_qubit_depth"]),
                    initial_total_gates=int(circ["initial_total_gates"]),
                    gate_density=float(circ["gate_density"]),
                    two_qubit_ratio=float(circ["two_qubit_ratio"]),
                    original_circuit=infer_original_circuit_name(
                        circuit_name,
                        circ["original_circuit"],
                    ),
                    created_at=circ["created_at"],
                )
                circuits_inserted_this_db += 1
            else:
                out_id = int(existing["id"])

            circuit_id_map[int(circ["id"])] = out_id

        run_rows = fetch_all(
            src_conn,
            """
            SELECT r.id, r.circuit_id, r.optimizer_id,
                   r.input_depth, r.input_two_qubit_gates, r.input_two_qubit_depth, r.input_total_gates,
                   r.output_depth, r.output_two_qubit_gates, r.output_two_qubit_depth, r.output_total_gates,
                   r.duration_seconds, r.success, r.error_message, r.artifact_path, r.created_at,
                   c.name AS circuit_name,
                   o.name AS optimizer_name
            FROM optimization_runs r
            JOIN circuits c ON c.id = r.circuit_id
            JOIN optimizers o ON o.id = r.optimizer_id
            """,
        )

        runs_inserted_this_db = 0
        runs_skipped_for_circuit_name = 0
        runs_skipped_for_optimizer = 0
        for run in run_rows:
            if str(run["optimizer_name"]) in excluded_optimizers:
                runs_skipped_for_optimizer += 1
                continue
            if name_contains_excluded_optimizer(run["circuit_name"], excluded_optimizers):
                runs_skipped_for_circuit_name += 1
                continue

            new_circuit_id = circuit_id_map.get(int(run["circuit_id"]))
            new_optimizer_id = optimizer_id_map.get(int(run["optimizer_id"]))
            if new_circuit_id is None or new_optimizer_id is None:
                continue

            out_conn.execute(
                """
                INSERT INTO optimization_runs (
                    circuit_id, optimizer_id,
                    input_depth, input_two_qubit_gates, input_two_qubit_depth, input_total_gates,
                    output_depth, output_two_qubit_gates, output_two_qubit_depth, output_total_gates,
                    duration_seconds, success, error_message, artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_circuit_id,
                    new_optimizer_id,
                    run["input_depth"],
                    run["input_two_qubit_gates"],
                    run["input_two_qubit_depth"],
                    run["input_total_gates"],
                    run["output_depth"],
                    run["output_two_qubit_gates"],
                    run["output_two_qubit_depth"],
                    run["output_total_gates"],
                    run["duration_seconds"],
                    run["success"],
                    run["error_message"],
                    normalize_path(run["artifact_path"]),
                    run["created_at"],
                ),
            )
            runs_inserted_this_db += 1

        trajectory_rows = fetch_all(
            src_conn,
            """
            SELECT t.id, t.circuit_id, t.chain_name, t.num_steps,
                   t.initial_depth, t.initial_two_qubit_gates, t.initial_two_qubit_depth, t.initial_total_gates,
                   t.final_depth, t.final_two_qubit_gates, t.final_two_qubit_depth, t.final_total_gates,
                   t.total_duration_seconds, t.total_reward, t.improvement_percentage,
                   t.metadata_json, t.created_at,
                   c.name AS circuit_name
            FROM trajectories t
            JOIN circuits c ON c.id = t.circuit_id
            """,
        )

        trajectory_id_map: dict[int, int] = {}
        trajectories_inserted_this_db = 0
        for traj in trajectory_rows:
            if name_contains_excluded_optimizer(traj["circuit_name"], excluded_optimizers):
                continue
            if name_contains_excluded_optimizer(traj["chain_name"], excluded_optimizers):
                continue

            new_circuit_id = circuit_id_map.get(int(traj["circuit_id"]))
            if new_circuit_id is None:
                continue

            cursor = out_conn.execute(
                """
                INSERT INTO trajectories (
                    circuit_id, chain_name, num_steps,
                    initial_depth, initial_two_qubit_gates, initial_two_qubit_depth, initial_total_gates,
                    final_depth, final_two_qubit_gates, final_two_qubit_depth, final_total_gates,
                    total_duration_seconds, total_reward, improvement_percentage,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_circuit_id,
                    traj["chain_name"],
                    traj["num_steps"],
                    traj["initial_depth"],
                    traj["initial_two_qubit_gates"],
                    traj["initial_two_qubit_depth"],
                    traj["initial_total_gates"],
                    traj["final_depth"],
                    traj["final_two_qubit_gates"],
                    traj["final_two_qubit_depth"],
                    traj["final_total_gates"],
                    traj["total_duration_seconds"],
                    traj["total_reward"],
                    traj["improvement_percentage"],
                    traj["metadata_json"],
                    traj["created_at"],
                ),
            )
            trajectory_id_map[int(traj["id"])] = int(cursor.lastrowid)
            trajectories_inserted_this_db += 1

        step_rows = fetch_all(
            src_conn,
            """
            SELECT trajectory_id, step_index, optimizer_id,
                   state_depth, state_two_qubit_gates, state_two_qubit_depth,
                   state_total_gates, state_num_qubits, state_gate_density, state_two_qubit_ratio,
                   state_steps_taken, state_time_budget_remaining, state_category_json,
                   next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                   next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                   next_state_steps_taken, next_state_time_budget_remaining,
                   reward_improvement_only, reward_efficiency, reward_multi_objective, reward_sparse_final,
                   reward_category_relative, reward_efficiency_normalized,
                   done, duration_seconds, created_at
            FROM trajectory_steps
            """,
        )

        steps_inserted_this_db = 0
        for step in step_rows:
            new_trajectory_id = trajectory_id_map.get(int(step["trajectory_id"]))
            new_optimizer_id = optimizer_id_map.get(int(step["optimizer_id"]))
            if new_trajectory_id is None or new_optimizer_id is None:
                continue

            out_conn.execute(
                """
                INSERT INTO trajectory_steps (
                    trajectory_id, step_index, optimizer_id,
                    state_depth, state_two_qubit_gates, state_two_qubit_depth,
                    state_total_gates, state_num_qubits, state_gate_density, state_two_qubit_ratio,
                    state_steps_taken, state_time_budget_remaining, state_category_json,
                    next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                    next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                    next_state_steps_taken, next_state_time_budget_remaining,
                    reward_improvement_only, reward_efficiency, reward_multi_objective, reward_sparse_final,
                    reward_category_relative, reward_efficiency_normalized,
                    done, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_trajectory_id,
                    step["step_index"],
                    new_optimizer_id,
                    step["state_depth"],
                    step["state_two_qubit_gates"],
                    step["state_two_qubit_depth"],
                    step["state_total_gates"],
                    step["state_num_qubits"],
                    step["state_gate_density"],
                    step["state_two_qubit_ratio"],
                    step["state_steps_taken"],
                    step["state_time_budget_remaining"],
                    step["state_category_json"],
                    step["next_state_depth"],
                    step["next_state_two_qubit_gates"],
                    step["next_state_two_qubit_depth"],
                    step["next_state_total_gates"],
                    step["next_state_gate_density"],
                    step["next_state_two_qubit_ratio"],
                    step["next_state_steps_taken"],
                    step["next_state_time_budget_remaining"],
                    step["reward_improvement_only"],
                    step["reward_efficiency"],
                    step["reward_multi_objective"],
                    step["reward_sparse_final"],
                    step["reward_category_relative"],
                    step["reward_efficiency_normalized"],
                    step["done"],
                    step["duration_seconds"],
                    step["created_at"],
                ),
            )
            steps_inserted_this_db += 1

        out_conn.commit()
        src_conn.close()

        total_circuits_inserted += circuits_inserted_this_db
        total_runs_inserted += runs_inserted_this_db
        total_trajectories_inserted += trajectories_inserted_this_db
        total_steps_inserted += steps_inserted_this_db

        print(f"  inserted circuits: {circuits_inserted_this_db}")
        print(f"  skipped circuits with excluded chain names: {circuits_skipped_for_excluded_name}")
        print(f"  inserted allowed optimizers: {sorted(allowed_optimizer_names)}")
        print(f"  inserted optimization_runs: {runs_inserted_this_db}")
        print(f"  skipped optimization_runs with excluded input circuits: {runs_skipped_for_circuit_name}")
        print(f"  skipped optimization_runs with excluded optimizers: {runs_skipped_for_optimizer}")
        print(f"  inserted trajectories: {trajectories_inserted_this_db}")
        print(f"  inserted trajectory_steps: {steps_inserted_this_db}")

    added_tket_circuits = add_corrected_tket_circuits(out_conn, single_db_path)

    final_circuit_count = out_conn.execute("SELECT COUNT(*) FROM circuits").fetchone()[0]
    final_optimizer_count = out_conn.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0]
    final_run_count = out_conn.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()[0]
    final_trajectory_count = out_conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    final_step_count = out_conn.execute("SELECT COUNT(*) FROM trajectory_steps").fetchone()[0]

    print("\nMerge complete")
    print(f"  output_db: {output_path}")
    print(f"  corrected tket circuits added from single-step DB: {added_tket_circuits}")
    print(f"  final circuits: {final_circuit_count}")
    print(f"  final optimizers: {final_optimizer_count}")
    print(f"  final optimization_runs: {final_run_count}")
    print(f"  final trajectories: {final_trajectory_count}")
    print(f"  final trajectory_steps: {final_step_count}")

    out_conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Merge step-2 DBs, drop excluded optimizer content, and rebuild tket input circuits.",
    )
    parser.add_argument(
        "--source-dbs",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCE_DBS,
        help="Source SQLite databases to merge.",
    )
    parser.add_argument(
        "--single-db",
        type=Path,
        default=DEFAULT_SINGLE_DB,
        help="Trusted single-step SQLite database used to rebuild corrected tket circuits.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=DEFAULT_OUTPUT_DB,
        help="Destination SQLite database path.",
    )
    parser.add_argument(
        "--exclude-optimizers",
        nargs="+",
        default=DEFAULT_EXCLUDED_OPTIMIZERS,
        help="Optimizer names to exclude from source circuits and optimization runs.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    merge_databases(
        source_paths=list(args.source_dbs),
        single_db_path=args.single_db,
        output_path=args.output_db,
        excluded_optimizers={str(name) for name in args.exclude_optimizers},
    )


if __name__ == "__main__":
    main()
