#!/usr/bin/env python3
"""Merge trajectory data from multiple databases into a combined database.

Creates a unified database for RL training by merging trajectory and
trajectory_steps from step2 and step3 databases, with proper deduplication
and foreign key remapping.

Usage:
    python scripts/merge_trajectory_databases.py
    python scripts/merge_trajectory_databases.py \
        --source-dbs data/trajectories_step2.db data/trajectories_step3.db \
        --output data/trajectories_combined.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory.database import SCHEMA_SQL  # noqa: E402


def merge_databases(
    source_dbs: list[Path],
    output_path: Path,
) -> None:
    """Merge trajectory data from source databases into output database.

    Merges:
    - optimizers (deduplicated by name)
    - circuits (deduplicated by name)
    - trajectories (remapped circuit_id)
    - trajectory_steps (remapped trajectory_id and optimizer_id)

    Does NOT merge optimization_runs (not needed for training).

    Args:
        source_dbs: List of source database paths
        output_path: Path for the merged output database
    """
    # Create output database
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_conn = sqlite3.connect(str(output_path))
    out_conn.row_factory = sqlite3.Row
    out_conn.execute("PRAGMA foreign_keys = ON")
    out_conn.execute("PRAGMA journal_mode = WAL")
    out_conn.executescript(SCHEMA_SQL)
    out_conn.commit()

    total_trajectories = 0
    total_steps = 0

    for db_path in source_dbs:
        print(f"\nProcessing {db_path}...")
        src_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        src_conn.row_factory = sqlite3.Row

        # Check if source has any trajectories
        src_traj_count = src_conn.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0]
        src_step_count = src_conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps"
        ).fetchone()[0]
        print(f"  Source trajectories: {src_traj_count}")
        print(f"  Source trajectory steps: {src_step_count}")

        if src_traj_count == 0:
            print("  Skipping (no trajectories).")
            src_conn.close()
            continue

        # --- Merge optimizers (deduplicate by name) ---
        src_optimizers = src_conn.execute(
            "SELECT id, name, runner_type, options_json, description FROM optimizers"
        ).fetchall()

        optimizer_id_map: dict[int, int] = {}  # src_id -> out_id
        for opt in src_optimizers:
            existing = out_conn.execute(
                "SELECT id FROM optimizers WHERE name = ?", (opt["name"],)
            ).fetchone()
            if existing:
                optimizer_id_map[opt["id"]] = existing["id"]
            else:
                cursor = out_conn.execute(
                    "INSERT INTO optimizers (name, runner_type, options_json, description) VALUES (?, ?, ?, ?)",
                    (opt["name"], opt["runner_type"], opt["options_json"], opt["description"]),
                )
                out_conn.commit()
                optimizer_id_map[opt["id"]] = cursor.lastrowid

        print(f"  Optimizers mapped: {len(optimizer_id_map)}")

        # --- Merge circuits (deduplicate by name) ---
        src_circuits = src_conn.execute(
            """SELECT id, name, category, source, qasm_path, num_qubits,
                      initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                      initial_total_gates, gate_density, two_qubit_ratio
               FROM circuits"""
        ).fetchall()

        circuit_id_map: dict[int, int] = {}  # src_id -> out_id
        for circ in src_circuits:
            existing = out_conn.execute(
                "SELECT id FROM circuits WHERE name = ?", (circ["name"],)
            ).fetchone()
            if existing:
                circuit_id_map[circ["id"]] = existing["id"]
            else:
                cursor = out_conn.execute(
                    """INSERT INTO circuits (name, category, source, qasm_path, num_qubits,
                              initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                              initial_total_gates, gate_density, two_qubit_ratio)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        circ["name"], circ["category"], circ["source"], circ["qasm_path"],
                        circ["num_qubits"], circ["initial_depth"], circ["initial_two_qubit_gates"],
                        circ["initial_two_qubit_depth"], circ["initial_total_gates"],
                        circ["gate_density"], circ["two_qubit_ratio"],
                    ),
                )
                out_conn.commit()
                circuit_id_map[circ["id"]] = cursor.lastrowid

        print(f"  Circuits mapped: {len(circuit_id_map)}")

        # --- Merge trajectories (remap circuit_id) ---
        src_trajectories = src_conn.execute(
            """SELECT id, circuit_id, chain_name, num_steps,
                      initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                      initial_total_gates, final_depth, final_two_qubit_gates,
                      final_two_qubit_depth, final_total_gates,
                      total_duration_seconds, total_reward, improvement_percentage,
                      metadata_json
               FROM trajectories"""
        ).fetchall()

        trajectory_id_map: dict[int, int] = {}  # src_id -> out_id
        for traj in src_trajectories:
            new_circuit_id = circuit_id_map.get(traj["circuit_id"])
            if new_circuit_id is None:
                continue

            cursor = out_conn.execute(
                """INSERT INTO trajectories (circuit_id, chain_name, num_steps,
                          initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                          initial_total_gates, final_depth, final_two_qubit_gates,
                          final_two_qubit_depth, final_total_gates,
                          total_duration_seconds, total_reward, improvement_percentage,
                          metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_circuit_id, traj["chain_name"], traj["num_steps"],
                    traj["initial_depth"], traj["initial_two_qubit_gates"],
                    traj["initial_two_qubit_depth"], traj["initial_total_gates"],
                    traj["final_depth"], traj["final_two_qubit_gates"],
                    traj["final_two_qubit_depth"], traj["final_total_gates"],
                    traj["total_duration_seconds"], traj["total_reward"],
                    traj["improvement_percentage"], traj["metadata_json"],
                ),
            )
            trajectory_id_map[traj["id"]] = cursor.lastrowid

        out_conn.commit()
        print(f"  Trajectories merged: {len(trajectory_id_map)}")
        total_trajectories += len(trajectory_id_map)

        # --- Merge trajectory_steps (remap trajectory_id and optimizer_id) ---
        src_steps = src_conn.execute(
            """SELECT trajectory_id, step_index, optimizer_id,
                      state_depth, state_two_qubit_gates, state_two_qubit_depth,
                      state_total_gates, state_num_qubits, state_gate_density,
                      state_two_qubit_ratio, state_steps_taken, state_time_budget_remaining,
                      state_category_json,
                      next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                      next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                      next_state_steps_taken, next_state_time_budget_remaining,
                      reward_improvement_only, reward_efficiency, reward_multi_objective,
                      reward_sparse_final, done, duration_seconds
               FROM trajectory_steps"""
        ).fetchall()

        steps_merged = 0
        for step in src_steps:
            new_traj_id = trajectory_id_map.get(step["trajectory_id"])
            new_opt_id = optimizer_id_map.get(step["optimizer_id"])
            if new_traj_id is None or new_opt_id is None:
                continue

            out_conn.execute(
                """INSERT INTO trajectory_steps (
                          trajectory_id, step_index, optimizer_id,
                          state_depth, state_two_qubit_gates, state_two_qubit_depth,
                          state_total_gates, state_num_qubits, state_gate_density,
                          state_two_qubit_ratio, state_steps_taken, state_time_budget_remaining,
                          state_category_json,
                          next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                          next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                          next_state_steps_taken, next_state_time_budget_remaining,
                          reward_improvement_only, reward_efficiency, reward_multi_objective,
                          reward_sparse_final, done, duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_traj_id, step["step_index"], new_opt_id,
                    step["state_depth"], step["state_two_qubit_gates"],
                    step["state_two_qubit_depth"], step["state_total_gates"],
                    step["state_num_qubits"], step["state_gate_density"],
                    step["state_two_qubit_ratio"], step["state_steps_taken"],
                    step["state_time_budget_remaining"], step["state_category_json"],
                    step["next_state_depth"], step["next_state_two_qubit_gates"],
                    step["next_state_two_qubit_depth"], step["next_state_total_gates"],
                    step["next_state_gate_density"], step["next_state_two_qubit_ratio"],
                    step["next_state_steps_taken"], step["next_state_time_budget_remaining"],
                    step["reward_improvement_only"], step["reward_efficiency"],
                    step["reward_multi_objective"], step["reward_sparse_final"],
                    step["done"], step["duration_seconds"],
                ),
            )
            steps_merged += 1

        out_conn.commit()
        print(f"  Trajectory steps merged: {steps_merged}")
        total_steps += steps_merged

        src_conn.close()

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"MERGE COMPLETE: {output_path}")
    print(f"{'=' * 60}")

    out_circuits = out_conn.execute("SELECT COUNT(*) FROM circuits").fetchone()[0]
    out_optimizers = out_conn.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0]
    out_trajectories = out_conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    out_steps = out_conn.execute("SELECT COUNT(*) FROM trajectory_steps").fetchone()[0]
    done_0 = out_conn.execute("SELECT COUNT(*) FROM trajectory_steps WHERE done = 0").fetchone()[0]
    done_1 = out_conn.execute("SELECT COUNT(*) FROM trajectory_steps WHERE done = 1").fetchone()[0]

    print(f"  Circuits: {out_circuits}")
    print(f"  Optimizers: {out_optimizers}")
    print(f"  Trajectories: {out_trajectories}")
    print(f"  Trajectory steps: {out_steps}")
    print(f"    done=False: {done_0}")
    print(f"    done=True: {done_1}")

    # Show by chain type
    chain_types = out_conn.execute(
        "SELECT num_steps, COUNT(*) as cnt FROM trajectories GROUP BY num_steps ORDER BY num_steps"
    ).fetchall()
    print("\n  By episode length:")
    for row in chain_types:
        print(f"    {row['num_steps']}-step: {row['cnt']} trajectories")

    out_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge trajectory databases for combined RL training"
    )
    parser.add_argument(
        "--source-dbs",
        type=Path,
        nargs="+",
        default=[
            Path("data/trajectories_step2.db"),
            Path("data/trajectories_step3.db"),
        ],
        help="Source databases to merge (default: step2 + step3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trajectories_combined.db"),
        help="Output merged database path",
    )

    args = parser.parse_args()

    for db_path in args.source_dbs:
        if not db_path.exists():
            print(f"Database not found: {db_path}")
            sys.exit(1)

    merge_databases(args.source_dbs, args.output)


if __name__ == "__main__":
    main()
