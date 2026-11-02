#!/usr/bin/env python3
"""Synthesize 2-step chain trajectories by joining step-1 and step-2 data.

Each chain represents: original_circuit → optimizer_A → intermediate → optimizer_B → final
This produces two SARS tuples per chain, with done=False on step 0 and done=True on step 1.

Supports two modes:
  1. Two-database mode (legacy): step-1 runs in one DB, step-2 runs in another
  2. Single-database mode: all runs in one DB, distinguished by circuit naming

Usage (two-database mode):
    python scripts/synthesize_chain_trajectories.py \
        --step1-db data/trajectories.db \
        --step2-db data/trajectories_step2.db

Usage (single-database mode — step-2 circuits named 'artifact_{orig}__{opt}'):
    python scripts/synthesize_chain_trajectories.py \
        --single-db data/trajectories.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory import (  # noqa: E402
    RewardConfig,
    TrajectoryDatabase,
    compute_all_rewards,
    compute_improvement_percentage,
)
from benchmarks.ai_transpile.rl_trajectory.database import TrajectoryStepRecord  # noqa: E402
from benchmarks.ai_transpile.rl_trajectory.state import get_category_encoding  # noqa: E402
from benchmarks.ai_transpile.transpilers import CircuitMetrics  # noqa: E402


def _parse_artifact_name(artifact_name: str) -> tuple[str, str] | None:
    """Parse 'artifact_{circuit_name}__{optimizer_name}' into components.

    Returns:
        (original_circuit_name, step1_optimizer_name) or None if pattern doesn't match.
    """
    m = re.match(r"^artifact_(.+)__(\w+)$", artifact_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _build_step1_lookup(
    step1_conn: sqlite3.Connection,
) -> dict[tuple[str, str], sqlite3.Row]:
    """Build lookup: (circuit_name, optimizer_name) -> best step-1 run.

    For circuit/optimizer pairs with multiple runs, picks the one with
    lowest output_two_qubit_gates (best result).
    """
    rows = step1_conn.execute(
        """
        SELECT
            r.id as run_id, r.circuit_id, r.optimizer_id,
            r.input_depth, r.input_two_qubit_gates,
            r.input_two_qubit_depth, r.input_total_gates,
            r.output_depth, r.output_two_qubit_gates,
            r.output_two_qubit_depth, r.output_total_gates,
            r.duration_seconds,
            c.name as circuit_name, c.num_qubits, c.category,
            c.initial_depth, c.initial_two_qubit_gates,
            c.initial_two_qubit_depth, c.initial_total_gates,
            o.name as optimizer_name
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 1
        ORDER BY c.name, o.name, r.output_two_qubit_gates ASC
        """
    ).fetchall()

    lookup: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:
        key = (row["circuit_name"], row["optimizer_name"])
        if key not in lookup:
            lookup[key] = row  # first row = best (lowest 2Q gates)
    return lookup


def synthesize_chains(
    step1_db_path: Path,
    step2_db: TrajectoryDatabase,
    *,
    dry_run: bool = False,
    time_budget: float = 300.0,
) -> int:
    """Synthesize 2-step chain trajectories from step-1 + step-2 data.

    Each chain: original → step1_optimizer → intermediate → step2_optimizer → final

    Step 0 (done=False): state=original metrics, action=step1_optimizer,
                         next_state=intermediate metrics
    Step 1 (done=True):  state=intermediate metrics, action=step2_optimizer,
                         next_state=final metrics

    Args:
        step1_db_path: Path to step-1 database
        step2_db: Step-2 TrajectoryDatabase (target for inserts)
        dry_run: Count without inserting
        time_budget: Time budget per episode

    Returns:
        Number of chains created
    """
    reward_config = RewardConfig()

    # Open step-1 DB read-only
    step1_conn = sqlite3.connect(f"file:{step1_db_path}?mode=ro", uri=True)
    step1_conn.row_factory = sqlite3.Row

    # Build step-1 lookup
    step1_lookup = _build_step1_lookup(step1_conn)
    print(f"  Step-1 unique (circuit, optimizer) pairs: {len(step1_lookup)}")

    # Get step-2 circuits and their optimization runs
    step2_conn = step2_db._get_connection()

    step2_rows = step2_conn.execute(
        """
        SELECT
            r.id as run_id, r.circuit_id, r.optimizer_id,
            r.input_depth, r.input_two_qubit_gates,
            r.input_two_qubit_depth, r.input_total_gates,
            r.output_depth, r.output_two_qubit_gates,
            r.output_two_qubit_depth, r.output_total_gates,
            r.duration_seconds,
            c.name as artifact_name, c.num_qubits, c.category,
            o.name as step2_optimizer_name
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 1
        ORDER BY c.name, o.name
        """
    ).fetchall()

    print(f"  Step-2 successful runs: {len(step2_rows)}")

    # Build optimizer name → id mapping for step-2 DB
    step2_optimizers = {
        row["name"]: row["id"]
        for row in step2_conn.execute("SELECT id, name FROM optimizers").fetchall()
    }

    created = 0
    skipped = 0

    for s2_row in step2_rows:
        # Parse artifact name to find step-1 origin
        parsed = _parse_artifact_name(s2_row["artifact_name"])
        if parsed is None:
            skipped += 1
            continue

        orig_circuit_name, step1_optimizer_name = parsed

        # Look up the corresponding step-1 run
        s1_row = step1_lookup.get((orig_circuit_name, step1_optimizer_name))
        if s1_row is None:
            skipped += 1
            continue

        # Verify metrics alignment: step-1 output should match step-2 input
        if (
            s1_row["output_two_qubit_gates"] != s2_row["input_two_qubit_gates"]
            or s1_row["output_depth"] != s2_row["input_depth"]
        ):
            # Metrics mismatch — step-1 had multiple runs; the artifact came from
            # a different run than the "best" one we picked. Use step-2 circuit's
            # actual input metrics as the intermediate state (they're correct by
            # definition), and use the original circuit metrics from step-1 DB.
            pass

        # --- Original circuit metrics (episode start) ---
        original_metrics = CircuitMetrics(
            depth=s1_row["input_depth"],
            two_qubit_gates=s1_row["input_two_qubit_gates"],
            two_qubit_depth=s1_row["input_two_qubit_depth"],
            total_gates=s1_row["input_total_gates"],
        )

        # --- Intermediate metrics (step-1 output = step-2 input) ---
        # Use step-2 input metrics (authoritative for what the artifact actually is)
        intermediate_metrics = CircuitMetrics(
            depth=s2_row["input_depth"],
            two_qubit_gates=s2_row["input_two_qubit_gates"],
            two_qubit_depth=s2_row["input_two_qubit_depth"],
            total_gates=s2_row["input_total_gates"],
        )

        # --- Final metrics (step-2 output) ---
        final_metrics = CircuitMetrics(
            depth=s2_row["output_depth"],
            two_qubit_gates=s2_row["output_two_qubit_gates"],
            two_qubit_depth=s2_row["output_two_qubit_depth"],
            total_gates=s2_row["output_total_gates"],
        )

        num_qubits = s2_row["num_qubits"]
        category = s2_row["category"]
        category_encoding = get_category_encoding(category)

        # Estimate step-1 duration from the step-1 run
        step1_duration = s1_row["duration_seconds"]
        step2_duration = s2_row["duration_seconds"]
        total_duration = step1_duration + step2_duration

        # --- Compute rewards ---
        # Step 0: original → intermediate (not final step)
        step0_rewards = compute_all_rewards(
            prev_metrics=original_metrics,
            new_metrics=intermediate_metrics,
            time_cost=step1_duration,
            initial_metrics=original_metrics,
            is_final_step=False,
            config=reward_config,
            time_budget=time_budget,
        )

        # Step 1: intermediate → final (final step)
        step1_rewards = compute_all_rewards(
            prev_metrics=intermediate_metrics,
            new_metrics=final_metrics,
            time_cost=step2_duration,
            initial_metrics=original_metrics,
            is_final_step=True,
            config=reward_config,
            time_budget=time_budget,
        )

        total_reward = step0_rewards.efficiency + step1_rewards.efficiency
        total_improvement = compute_improvement_percentage(
            original_metrics.two_qubit_gates, final_metrics.two_qubit_gates
        )

        if dry_run:
            created += 1
            continue

        # --- Insert trajectory (2-step episode) ---
        chain_name = f"chain_{step1_optimizer_name}__{s2_row['step2_optimizer_name']}"

        # The trajectory's circuit_id should reference the step-2 DB's circuit
        # (the artifact), since that's the DB we're inserting into
        trajectory_id = step2_db.insert_trajectory(
            circuit_id=s2_row["circuit_id"],
            chain_name=chain_name,
            num_steps=2,
            initial_depth=original_metrics.depth,
            initial_two_qubit_gates=original_metrics.two_qubit_gates,
            initial_two_qubit_depth=original_metrics.two_qubit_depth,
            initial_total_gates=original_metrics.total_gates,
            final_depth=final_metrics.depth,
            final_two_qubit_gates=final_metrics.two_qubit_gates,
            final_two_qubit_depth=final_metrics.two_qubit_depth,
            final_total_gates=final_metrics.total_gates,
            total_duration_seconds=total_duration,
            total_reward=total_reward,
            improvement_percentage=total_improvement,
            metadata={
                "source": "chain_step1_step2",
                "step1_run_id": s1_row["run_id"],
                "step2_run_id": s2_row["run_id"],
                "original_circuit": orig_circuit_name,
                "step1_optimizer": step1_optimizer_name,
                "step2_optimizer": s2_row["step2_optimizer_name"],
            },
        )

        # --- Step 0: original → intermediate (done=False) ---
        # Derived state features
        s0_gate_density = (
            original_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        )
        s0_two_qubit_ratio = (
            original_metrics.two_qubit_gates / original_metrics.total_gates
            if original_metrics.total_gates > 0
            else 0.0
        )
        s0_next_gate_density = (
            intermediate_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        )
        s0_next_two_qubit_ratio = (
            intermediate_metrics.two_qubit_gates / intermediate_metrics.total_gates
            if intermediate_metrics.total_gates > 0
            else 0.0
        )

        # Find step-1 optimizer ID in step-2 DB's optimizer table
        step1_opt_id = step2_optimizers[step1_optimizer_name]

        step0 = TrajectoryStepRecord(
            trajectory_id=trajectory_id,
            step_index=0,
            optimizer_id=step1_opt_id,
            state_depth=original_metrics.depth,
            state_two_qubit_gates=original_metrics.two_qubit_gates,
            state_two_qubit_depth=original_metrics.two_qubit_depth,
            state_total_gates=original_metrics.total_gates,
            state_num_qubits=num_qubits,
            state_gate_density=s0_gate_density,
            state_two_qubit_ratio=s0_two_qubit_ratio,
            state_steps_taken=0,
            state_time_budget_remaining=time_budget,
            state_category=category_encoding,
            next_state_depth=intermediate_metrics.depth,
            next_state_two_qubit_gates=intermediate_metrics.two_qubit_gates,
            next_state_two_qubit_depth=intermediate_metrics.two_qubit_depth,
            next_state_total_gates=intermediate_metrics.total_gates,
            next_state_gate_density=s0_next_gate_density,
            next_state_two_qubit_ratio=s0_next_two_qubit_ratio,
            next_state_steps_taken=1,
            next_state_time_budget_remaining=time_budget - step1_duration,
            reward_improvement_only=step0_rewards.improvement_only,
            reward_efficiency=step0_rewards.efficiency,
            reward_multi_objective=step0_rewards.multi_objective,
            reward_sparse_final=step0_rewards.sparse_final,
            reward_efficiency_normalized=step0_rewards.efficiency_normalized,
            done=False,
            duration_seconds=step1_duration,
        )
        step2_db.insert_trajectory_step(step0)

        # --- Step 1: intermediate → final (done=True) ---
        s1_gate_density = (
            intermediate_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        )
        s1_two_qubit_ratio = (
            intermediate_metrics.two_qubit_gates / intermediate_metrics.total_gates
            if intermediate_metrics.total_gates > 0
            else 0.0
        )
        s1_next_gate_density = (
            final_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        )
        s1_next_two_qubit_ratio = (
            final_metrics.two_qubit_gates / final_metrics.total_gates
            if final_metrics.total_gates > 0
            else 0.0
        )

        step1 = TrajectoryStepRecord(
            trajectory_id=trajectory_id,
            step_index=1,
            optimizer_id=s2_row["optimizer_id"],
            state_depth=intermediate_metrics.depth,
            state_two_qubit_gates=intermediate_metrics.two_qubit_gates,
            state_two_qubit_depth=intermediate_metrics.two_qubit_depth,
            state_total_gates=intermediate_metrics.total_gates,
            state_num_qubits=num_qubits,
            state_gate_density=s1_gate_density,
            state_two_qubit_ratio=s1_two_qubit_ratio,
            state_steps_taken=1,
            state_time_budget_remaining=time_budget - step1_duration,
            state_category=category_encoding,
            next_state_depth=final_metrics.depth,
            next_state_two_qubit_gates=final_metrics.two_qubit_gates,
            next_state_two_qubit_depth=final_metrics.two_qubit_depth,
            next_state_total_gates=final_metrics.total_gates,
            next_state_gate_density=s1_next_gate_density,
            next_state_two_qubit_ratio=s1_next_two_qubit_ratio,
            next_state_steps_taken=2,
            next_state_time_budget_remaining=time_budget - total_duration,
            reward_improvement_only=step1_rewards.improvement_only,
            reward_efficiency=step1_rewards.efficiency,
            reward_multi_objective=step1_rewards.multi_objective,
            reward_sparse_final=step1_rewards.sparse_final,
            reward_efficiency_normalized=step1_rewards.efficiency_normalized,
            done=True,
            duration_seconds=step2_duration,
        )
        step2_db.insert_trajectory_step(step1)

        created += 1

    step1_conn.close()

    if skipped > 0:
        print(f"  Skipped {skipped} step-2 runs (no matching step-1 data)")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize 2-step chain trajectories from step-1 + step-2 data"
    )
    parser.add_argument(
        "--step1-db",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to step-1 trajectory database",
    )
    parser.add_argument(
        "--step2-db",
        type=Path,
        default=Path("data/trajectories_step2.db"),
        help="Path to step-2 trajectory database (target for inserts)",
    )
    parser.add_argument(
        "--single-db",
        type=Path,
        default=None,
        help="Single database containing both step-1 and step-2 data "
             "(step-2 circuits named 'artifact_{orig}__{opt}')",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="Time budget per episode in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count chains without inserting",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing trajectories/trajectory_steps before inserting",
    )

    args = parser.parse_args()

    if args.single_db is not None:
        if not args.single_db.exists():
            print(f"Database not found: {args.single_db}")
            sys.exit(1)
        if args.step1_db != Path("data/trajectories.db") or \
           args.step2_db != Path("data/trajectories_step2.db"):
            print("Warning: --single-db overrides --step1-db and --step2-db")
        _main_single_db(args)
    else:
        for db_path in [args.step1_db, args.step2_db]:
            if not db_path.exists():
                print(f"Database not found: {db_path}")
                sys.exit(1)
        _main_two_db(args)


def _main_two_db(args) -> None:
    """Original two-database mode: separate step-1 and step-2 databases."""
    step2_db = TrajectoryDatabase(args.step2_db)

    # Show current stats
    stats = step2_db.get_statistics()
    print(f"Step-1 database: {args.step1_db}")
    print(f"Step-2 database: {args.step2_db}")
    print(f"  Existing trajectories: {stats['num_trajectories']}")
    print(f"  Existing trajectory steps: {stats['num_trajectory_steps']}")
    print()

    if args.clear_existing and not args.dry_run:
        print("Clearing existing trajectories and trajectory_steps...")
        conn = step2_db._get_connection()
        conn.execute("DELETE FROM trajectory_steps")
        conn.execute("DELETE FROM trajectories")
        conn.commit()
        print("  Cleared.")
        print()

    if args.dry_run:
        print("Dry run: counting potential chain trajectories...")
    else:
        print("Synthesizing 2-step chain trajectories...")

    created = synthesize_chains(
        step1_db_path=args.step1_db,
        step2_db=step2_db,
        dry_run=args.dry_run,
        time_budget=args.time_budget,
    )

    if args.dry_run:
        print(f"\nWould create {created} chain trajectories ({created * 2} steps)")
    else:
        print(f"\nCreated {created} chain trajectories ({created * 2} steps)")

        # Show updated stats
        stats = step2_db.get_statistics()
        print("\nUpdated statistics:")
        print(f"  Trajectories: {stats['num_trajectories']}")
        print(f"  Trajectory steps: {stats['num_trajectory_steps']}")

        # Show breakdown
        conn = step2_db._get_connection()
        done_0 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 0"
        ).fetchone()[0]
        done_1 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 1"
        ).fetchone()[0]
        print(f"  Steps with done=False: {done_0}")
        print(f"  Steps with done=True: {done_1}")

    step2_db.close()


def _main_single_db(args) -> None:
    """Single-database mode: both step-1 and step-2 data in one DB.

    Original circuits (source='local', 'guoq', etc.) are step-1.
    Artifact circuits (names starting with 'artifact_') serve as both
    step-1 outputs and step-2 inputs — the optimizer runs on these artifacts
    produce step-2 results.
    """
    db_path = args.single_db
    db = TrajectoryDatabase(db_path)

    stats = db.get_statistics()
    print(f"Single database: {db_path}")
    print(f"  Circuits:     {stats.get('num_circuits', 'N/A')}")
    print(f"  Optimizers:   {stats.get('num_optimizers', 'N/A')}")
    print(f"  Opt. runs:    {stats.get('num_optimization_runs', 'N/A')}")
    print(f"  Trajectories: {stats.get('num_trajectories', 0)}")
    print(f"  Traj. steps:  {stats.get('num_trajectory_steps', 0)}")
    print()

    # Quick audit: count original vs artifact circuits
    conn = db._get_connection()
    orig = conn.execute(
        "SELECT COUNT(*) FROM circuits WHERE name NOT LIKE 'artifact_%'"
    ).fetchone()[0]
    art = conn.execute(
        "SELECT COUNT(*) FROM circuits WHERE name LIKE 'artifact_%'"
    ).fetchone()[0]
    print(f"  Original circuits:  {orig}")
    print(f"  Artifact circuits:  {art}")
    print()

    if args.clear_existing and not args.dry_run:
        print("Clearing existing trajectories and trajectory_steps...")
        conn.execute("DELETE FROM trajectory_steps")
        conn.execute("DELETE FROM trajectories")
        conn.commit()
        print("  Cleared.")
        print()

    if args.dry_run:
        print("Dry run: counting potential chain trajectories...")
    else:
        print("Synthesizing 2-step chain trajectories (single-db mode)...")

    # In single-db mode, open two read-only connections to the same file.
    # synthesize_chains treats them as separate DBs, which works fine since
    # optimization_runs for original circuits (step-1) vs artifact circuits
    # (step-2) are naturally distinguished by the circuit name.
    created = synthesize_chains(
        step1_db_path=db_path,
        step2_db=db,
        dry_run=args.dry_run,
        time_budget=args.time_budget,
    )

    if args.dry_run:
        print(f"\nWould create {created} chain trajectories ({created * 2} steps)")
    else:
        print(f"\nCreated {created} chain trajectories ({created * 2} steps)")

        # Show updated stats
        stats = db.get_statistics()
        print("\nUpdated statistics:")
        print(f"  Trajectories: {stats['num_trajectories']}")
        print(f"  Trajectory steps: {stats['num_trajectory_steps']}")

        # Show breakdown
        conn = db._get_connection()
        done_0 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 0"
        ).fetchone()[0]
        done_1 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 1"
        ).fetchone()[0]
        print(f"  Steps with done=False: {done_0}")
        print(f"  Steps with done=True: {done_1}")

    db.close()


if __name__ == "__main__":
    main()
