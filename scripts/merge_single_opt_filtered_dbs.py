#!/usr/bin/env python3
"""Merge two single-optimizer trajectory DBs into one filtered database.

This script keeps:
- all circuits from both source databases, deduplicated by circuit name
- only optimizers not in the excluded list
- only optimization_runs whose optimizer is allowed
- only single-step trajectories/trajectory_steps whose optimizer is allowed

It writes a brand-new output database using the standard RL trajectory schema.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DBS = [
    PROJECT_ROOT / "dbs" / "mqt" / "data" / "mqt_single_opt_trajectories.db",
    PROJECT_ROOT / "dbs" / "original_data" / "data_single_opt" / "trajectories.db",
]
DEFAULT_OUTPUT_DB = PROJECT_ROOT / "dbs" / "combined_single_opt_no_ai_tket.db"
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


def normalize_qasm_path(path_value: str | None) -> str | None:
    """Rewrite known legacy absolute prefixes to the current repo root."""
    if path_value is None:
        return None
    path_text = str(path_value)
    if path_text.startswith(OLD_QASM_PREFIX):
        suffix = path_text.removeprefix(OLD_QASM_PREFIX)
        return f"{NEW_QASM_PREFIX}{suffix}"
    return path_text


def create_output_db(output_path: Path, schema_sql: str) -> sqlite3.Connection:
    """Create a fresh output DB using the standard schema."""
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(output_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(schema_sql)
    conn.commit()
    return conn


def merge_databases(
    source_paths: list[Path],
    output_path: Path,
    excluded_optimizers: set[str],
) -> None:
    """Merge source DBs into one filtered output DB."""
    out_conn = create_output_db(output_path, load_schema_sql())

    total_circuits_inserted = 0
    total_runs_inserted = 0
    total_trajectories_inserted = 0
    total_steps_inserted = 0

    for source_path in source_paths:
        print(f"\nProcessing {source_path}")
        src_conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        src_conn.row_factory = sqlite3.Row

        src_optimizers = fetch_all(
            src_conn,
            "SELECT id, name, runner_type, options_json, description FROM optimizers",
        )
        src_circuits = fetch_all(
            src_conn,
            """
            SELECT id, name, category, source, qasm_path, num_qubits,
                   initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                   initial_total_gates, gate_density, two_qubit_ratio
            FROM circuits
            """,
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

        circuit_id_map: dict[int, int] = {}
        circuits_inserted_this_db = 0
        for circ in src_circuits:
            existing = out_conn.execute(
                "SELECT id FROM circuits WHERE name = ?",
                (circ["name"],),
            ).fetchone()
            if existing is None:
                cursor = out_conn.execute(
                    """
                    INSERT INTO circuits (
                        name, category, source, qasm_path, num_qubits,
                        initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                        initial_total_gates, gate_density, two_qubit_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        circ["name"],
                        circ["category"],
                        circ["source"],
                        normalize_qasm_path(circ["qasm_path"]),
                        circ["num_qubits"],
                        circ["initial_depth"],
                        circ["initial_two_qubit_gates"],
                        circ["initial_two_qubit_depth"],
                        circ["initial_total_gates"],
                        circ["gate_density"],
                        circ["two_qubit_ratio"],
                    ),
                )
                out_id = int(cursor.lastrowid)
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
                   o.name AS optimizer_name
            FROM optimization_runs r
            JOIN optimizers o ON o.id = r.optimizer_id
            """,
        )

        runs_inserted_this_db = 0
        for run in run_rows:
            if str(run["optimizer_name"]) in excluded_optimizers:
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
                    run["artifact_path"],
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
                   t.metadata_json, t.created_at
            FROM trajectories t
            """,
        )

        trajectory_id_map: dict[int, int] = {}
        trajectories_inserted_this_db = 0
        for traj in trajectory_rows:
            chain_name = str(traj["chain_name"])
            if chain_name in excluded_optimizers:
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
                    chain_name,
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
        print(f"  inserted allowed optimizers: {sorted(allowed_optimizer_names)}")
        print(f"  inserted optimization_runs: {runs_inserted_this_db}")
        print(f"  inserted trajectories: {trajectories_inserted_this_db}")
        print(f"  inserted trajectory_steps: {steps_inserted_this_db}")

    final_circuit_count = out_conn.execute("SELECT COUNT(*) FROM circuits").fetchone()[0]
    final_optimizer_count = out_conn.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0]
    final_run_count = out_conn.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()[0]
    final_trajectory_count = out_conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    final_step_count = out_conn.execute("SELECT COUNT(*) FROM trajectory_steps").fetchone()[0]

    print("\nMerge complete")
    print(f"  output_db: {output_path}")
    print(f"  final circuits: {final_circuit_count}")
    print(f"  final optimizers: {final_optimizer_count}")
    print(f"  final optimization_runs: {final_run_count}")
    print(f"  final trajectories: {final_trajectory_count}")
    print(f"  final trajectory_steps: {final_step_count}")

    out_conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Merge two single-opt DBs and drop excluded optimizers.",
    )
    parser.add_argument(
        "--source-dbs",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCE_DBS,
        help="Source SQLite databases to merge.",
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
        help="Optimizer names to exclude from runs and trajectories.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    merge_databases(
        source_paths=list(args.source_dbs),
        output_path=args.output_db,
        excluded_optimizers={str(name) for name in args.exclude_optimizers},
    )


if __name__ == "__main__":
    main()
