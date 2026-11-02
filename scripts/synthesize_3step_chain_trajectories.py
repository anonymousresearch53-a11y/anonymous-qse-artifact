#!/usr/bin/env python3
"""Synthesize 3-step chain trajectories by joining step-1, step-2, and step-3 data.

Each chain represents:
  original_circuit -> opt1 -> intermediate1 -> opt2 -> intermediate2 -> opt3 -> final

This produces three SARS tuples per chain:
  Step 0 (done=False): original -> opt1 -> intermediate1
  Step 1 (done=False): intermediate1 -> opt2 -> intermediate2
  Step 2 (done=True):  intermediate2 -> opt3 -> final

Usage:
    python scripts/synthesize_3step_chain_trajectories.py
    python scripts/synthesize_3step_chain_trajectories.py --dry-run
    python scripts/synthesize_3step_chain_trajectories.py \
        --step1-db data/trajectories.db \
        --step2-db data/trajectories_step2.db \
        --step3-db data/trajectories_step3.db
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


def _parse_step3_artifact_name(name: str) -> tuple[str, str, str] | None:
    """Parse step3 circuit name into (original_circuit, opt1, opt2).

    Step3 names follow: artifact_artifact_{orig}__{opt1}__{opt2}

    We parse in two rounds:
      Round 1: artifact_(.+)__(\\w+) -> (artifact_{orig}__{opt1}, opt2)
      Round 2: artifact_(.+)__(\\w+) -> (orig, opt1)

    Returns:
        (original_circuit_name, step1_optimizer, step2_optimizer) or None
    """
    m1 = re.match(r"^artifact_(.+)__(\w+)$", name)
    if not m1:
        return None
    step2_circuit_name = m1.group(1)
    opt2 = m1.group(2)

    m2 = re.match(r"^artifact_(.+)__(\w+)$", step2_circuit_name)
    if not m2:
        return None
    orig_circuit = m2.group(1)
    opt1 = m2.group(2)

    return orig_circuit, opt1, opt2


def _build_step1_lookup(
    step1_conn: sqlite3.Connection,
) -> dict[tuple[str, str], sqlite3.Row]:
    """Build lookup: (circuit_name, optimizer_name) -> best step-1 run."""
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
            lookup[key] = row
    return lookup


def _build_step2_lookup(
    step2_conn: sqlite3.Connection,
) -> dict[tuple[str, str], sqlite3.Row]:
    """Build lookup: (step2_circuit_name, optimizer_name) -> best step-2 run."""
    rows = step2_conn.execute(
        """
        SELECT
            r.id as run_id, r.circuit_id, r.optimizer_id,
            r.input_depth, r.input_two_qubit_gates,
            r.input_two_qubit_depth, r.input_total_gates,
            r.output_depth, r.output_two_qubit_gates,
            r.output_two_qubit_depth, r.output_total_gates,
            r.duration_seconds,
            c.name as circuit_name, c.num_qubits, c.category,
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
            lookup[key] = row
    return lookup


def _make_step_record(
    trajectory_id: int,
    step_index: int,
    optimizer_id: int,
    prev_metrics: CircuitMetrics,
    next_metrics: CircuitMetrics,
    num_qubits: int,
    category_encoding: list[float],
    steps_taken: int,
    time_budget_remaining: float,
    rewards,  # noqa: ANN001  # RewardSet
    done: bool,
    duration: float,
) -> TrajectoryStepRecord:
    """Create a TrajectoryStepRecord with derived features."""
    s_gate_density = prev_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
    s_two_qubit_ratio = (
        prev_metrics.two_qubit_gates / prev_metrics.total_gates
        if prev_metrics.total_gates > 0
        else 0.0
    )
    ns_gate_density = next_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
    ns_two_qubit_ratio = (
        next_metrics.two_qubit_gates / next_metrics.total_gates
        if next_metrics.total_gates > 0
        else 0.0
    )

    return TrajectoryStepRecord(
        trajectory_id=trajectory_id,
        step_index=step_index,
        optimizer_id=optimizer_id,
        state_depth=prev_metrics.depth,
        state_two_qubit_gates=prev_metrics.two_qubit_gates,
        state_two_qubit_depth=prev_metrics.two_qubit_depth,
        state_total_gates=prev_metrics.total_gates,
        state_num_qubits=num_qubits,
        state_gate_density=s_gate_density,
        state_two_qubit_ratio=s_two_qubit_ratio,
        state_steps_taken=steps_taken,
        state_time_budget_remaining=time_budget_remaining,
        state_category=category_encoding,
        next_state_depth=next_metrics.depth,
        next_state_two_qubit_gates=next_metrics.two_qubit_gates,
        next_state_two_qubit_depth=next_metrics.two_qubit_depth,
        next_state_total_gates=next_metrics.total_gates,
        next_state_gate_density=ns_gate_density,
        next_state_two_qubit_ratio=ns_two_qubit_ratio,
        next_state_steps_taken=steps_taken + 1,
        next_state_time_budget_remaining=time_budget_remaining - duration,
        reward_improvement_only=rewards.improvement_only,
        reward_efficiency=rewards.efficiency,
        reward_multi_objective=rewards.multi_objective,
        reward_sparse_final=rewards.sparse_final,
        reward_efficiency_normalized=rewards.efficiency_normalized,
        done=done,
        duration_seconds=duration,
    )


def synthesize_3step_chains(
    step1_db_path: Path,
    step2_db_path: Path,
    target_db: TrajectoryDatabase,
    *,
    dry_run: bool = False,
    time_budget: float = 300.0,
) -> int:
    """Synthesize 3-step chain trajectories from step-1 + step-2 + step-3 data.

    The target_db is the step-3 database (which also contains the step-3
    optimization_runs and circuits tables).

    Args:
        step1_db_path: Path to step-1 database
        step2_db_path: Path to step-2 database
        target_db: Step-3 TrajectoryDatabase (target for inserts)
        dry_run: Count without inserting
        time_budget: Time budget per episode

    Returns:
        Number of 3-step chains created
    """
    reward_config = RewardConfig()

    # Open step-1 and step-2 DBs read-only
    step1_conn = sqlite3.connect(f"file:{step1_db_path}?mode=ro", uri=True)
    step1_conn.row_factory = sqlite3.Row
    step2_conn = sqlite3.connect(f"file:{step2_db_path}?mode=ro", uri=True)
    step2_conn.row_factory = sqlite3.Row

    # Build lookups
    step1_lookup = _build_step1_lookup(step1_conn)
    print(f"  Step-1 unique (circuit, optimizer) pairs: {len(step1_lookup)}")

    step2_lookup = _build_step2_lookup(step2_conn)
    print(f"  Step-2 unique (circuit, optimizer) pairs: {len(step2_lookup)}")

    # Get step-3 successful runs from target DB
    target_conn = target_db._get_connection()

    step3_rows = target_conn.execute(
        """
        SELECT
            r.id as run_id, r.circuit_id, r.optimizer_id,
            r.input_depth, r.input_two_qubit_gates,
            r.input_two_qubit_depth, r.input_total_gates,
            r.output_depth, r.output_two_qubit_gates,
            r.output_two_qubit_depth, r.output_total_gates,
            r.duration_seconds,
            c.name as circuit_name, c.num_qubits, c.category,
            o.name as step3_optimizer_name
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 1
        ORDER BY c.name, o.name
        """
    ).fetchall()
    print(f"  Step-3 successful runs: {len(step3_rows)}")

    # Build optimizer name -> id mapping for the target (step3) DB
    target_optimizers = {
        row["name"]: row["id"]
        for row in target_conn.execute("SELECT id, name FROM optimizers").fetchall()
    }

    created = 0
    skipped = 0

    for s3_row in step3_rows:
        # Parse step3 artifact name to get full chain lineage
        parsed = _parse_step3_artifact_name(s3_row["circuit_name"])
        if parsed is None:
            skipped += 1
            continue

        orig_circuit, opt1, opt2 = parsed
        opt3 = s3_row["step3_optimizer_name"]

        # Look up step-1 run: original_circuit + opt1
        s1_row = step1_lookup.get((orig_circuit, opt1))
        if s1_row is None:
            skipped += 1
            continue

        # Look up step-2 run: artifact_{orig}__{opt1} + opt2
        step2_circuit_name = f"artifact_{orig_circuit}__{opt1}"
        s2_row = step2_lookup.get((step2_circuit_name, opt2))
        if s2_row is None:
            skipped += 1
            continue

        # --- Build metrics for all 4 states ---
        original_metrics = CircuitMetrics(
            depth=s1_row["input_depth"],
            two_qubit_gates=s1_row["input_two_qubit_gates"],
            two_qubit_depth=s1_row["input_two_qubit_depth"],
            total_gates=s1_row["input_total_gates"],
        )

        # After step 1: use step-2 input (authoritative)
        intermediate1_metrics = CircuitMetrics(
            depth=s2_row["input_depth"],
            two_qubit_gates=s2_row["input_two_qubit_gates"],
            two_qubit_depth=s2_row["input_two_qubit_depth"],
            total_gates=s2_row["input_total_gates"],
        )

        # After step 2: use step-3 input (authoritative)
        intermediate2_metrics = CircuitMetrics(
            depth=s3_row["input_depth"],
            two_qubit_gates=s3_row["input_two_qubit_gates"],
            two_qubit_depth=s3_row["input_two_qubit_depth"],
            total_gates=s3_row["input_total_gates"],
        )

        # After step 3: step-3 output
        final_metrics = CircuitMetrics(
            depth=s3_row["output_depth"],
            two_qubit_gates=s3_row["output_two_qubit_gates"],
            two_qubit_depth=s3_row["output_two_qubit_depth"],
            total_gates=s3_row["output_total_gates"],
        )

        num_qubits = s1_row["num_qubits"]
        category = s1_row["category"]
        category_encoding = get_category_encoding(category)

        s1_duration = s1_row["duration_seconds"]
        s2_duration = s2_row["duration_seconds"]
        s3_duration = s3_row["duration_seconds"]
        total_duration = s1_duration + s2_duration + s3_duration

        # --- Compute rewards for each step ---
        step0_rewards = compute_all_rewards(
            prev_metrics=original_metrics,
            new_metrics=intermediate1_metrics,
            time_cost=s1_duration,
            initial_metrics=original_metrics,
            is_final_step=False,
            config=reward_config,
            time_budget=time_budget,
        )

        step1_rewards = compute_all_rewards(
            prev_metrics=intermediate1_metrics,
            new_metrics=intermediate2_metrics,
            time_cost=s2_duration,
            initial_metrics=original_metrics,
            is_final_step=False,
            config=reward_config,
            time_budget=time_budget,
        )

        step2_rewards = compute_all_rewards(
            prev_metrics=intermediate2_metrics,
            new_metrics=final_metrics,
            time_cost=s3_duration,
            initial_metrics=original_metrics,
            is_final_step=True,
            config=reward_config,
            time_budget=time_budget,
        )

        total_reward = (
            step0_rewards.efficiency + step1_rewards.efficiency + step2_rewards.efficiency
        )
        total_improvement = compute_improvement_percentage(
            original_metrics.two_qubit_gates, final_metrics.two_qubit_gates
        )

        if dry_run:
            created += 1
            continue

        # --- Insert trajectory (3-step episode) ---
        chain_name = f"chain3_{opt1}__{opt2}__{opt3}"

        trajectory_id = target_db.insert_trajectory(
            circuit_id=s3_row["circuit_id"],
            chain_name=chain_name,
            num_steps=3,
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
                "source": "chain_3step",
                "step1_run_id": s1_row["run_id"],
                "step2_run_id": s2_row["run_id"],
                "step3_run_id": s3_row["run_id"],
                "original_circuit": orig_circuit,
                "step1_optimizer": opt1,
                "step2_optimizer": opt2,
                "step3_optimizer": opt3,
            },
        )

        # Map optimizer names to IDs in target DB
        opt1_id = target_optimizers[opt1]
        opt2_id = target_optimizers[opt2]
        opt3_id = s3_row["optimizer_id"]

        # --- Step 0: original -> intermediate1 (done=False) ---
        step0 = _make_step_record(
            trajectory_id=trajectory_id,
            step_index=0,
            optimizer_id=opt1_id,
            prev_metrics=original_metrics,
            next_metrics=intermediate1_metrics,
            num_qubits=num_qubits,
            category_encoding=category_encoding,
            steps_taken=0,
            time_budget_remaining=time_budget,
            rewards=step0_rewards,
            done=False,
            duration=s1_duration,
        )
        target_db.insert_trajectory_step(step0)

        # --- Step 1: intermediate1 -> intermediate2 (done=False) ---
        step1 = _make_step_record(
            trajectory_id=trajectory_id,
            step_index=1,
            optimizer_id=opt2_id,
            prev_metrics=intermediate1_metrics,
            next_metrics=intermediate2_metrics,
            num_qubits=num_qubits,
            category_encoding=category_encoding,
            steps_taken=1,
            time_budget_remaining=time_budget - s1_duration,
            rewards=step1_rewards,
            done=False,
            duration=s2_duration,
        )
        target_db.insert_trajectory_step(step1)

        # --- Step 2: intermediate2 -> final (done=True) ---
        step2 = _make_step_record(
            trajectory_id=trajectory_id,
            step_index=2,
            optimizer_id=opt3_id,
            prev_metrics=intermediate2_metrics,
            next_metrics=final_metrics,
            num_qubits=num_qubits,
            category_encoding=category_encoding,
            steps_taken=2,
            time_budget_remaining=time_budget - s1_duration - s2_duration,
            rewards=step2_rewards,
            done=True,
            duration=s3_duration,
        )
        target_db.insert_trajectory_step(step2)

        created += 1

    step1_conn.close()
    step2_conn.close()

    if skipped > 0:
        print(f"  Skipped {skipped} step-3 runs (no matching step-1/step-2 data or parse failure)")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize 3-step chain trajectories from step-1 + step-2 + step-3 data"
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
        help="Path to step-2 trajectory database",
    )
    parser.add_argument(
        "--step3-db",
        type=Path,
        default=Path("data/trajectories_step3.db"),
        help="Path to step-3 trajectory database (target for inserts)",
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

    for db_path in [args.step1_db, args.step2_db, args.step3_db]:
        if not db_path.exists():
            print(f"Database not found: {db_path}")
            sys.exit(1)

    target_db = TrajectoryDatabase(args.step3_db)

    # Show current stats
    stats = target_db.get_statistics()
    print(f"Step-1 database: {args.step1_db}")
    print(f"Step-2 database: {args.step2_db}")
    print(f"Step-3 database (target): {args.step3_db}")
    print(f"  Existing trajectories: {stats['num_trajectories']}")
    print(f"  Existing trajectory steps: {stats['num_trajectory_steps']}")
    print()

    if args.clear_existing and not args.dry_run:
        print("Clearing existing trajectories and trajectory_steps...")
        conn = target_db._get_connection()
        conn.execute("DELETE FROM trajectory_steps")
        conn.execute("DELETE FROM trajectories")
        conn.commit()
        print("  Cleared.")
        print()

    if args.dry_run:
        print("Dry run: counting potential 3-step chain trajectories...")
    else:
        print("Synthesizing 3-step chain trajectories...")

    created = synthesize_3step_chains(
        step1_db_path=args.step1_db,
        step2_db_path=args.step2_db,
        target_db=target_db,
        dry_run=args.dry_run,
        time_budget=args.time_budget,
    )

    if args.dry_run:
        print(f"\nWould create {created} chain trajectories ({created * 3} steps)")
    else:
        print(f"\nCreated {created} chain trajectories ({created * 3} steps)")

        # Show updated stats
        stats = target_db.get_statistics()
        print("\nUpdated statistics:")
        print(f"  Trajectories: {stats['num_trajectories']}")
        print(f"  Trajectory steps: {stats['num_trajectory_steps']}")

        # Show breakdown
        conn = target_db._get_connection()
        done_0 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 0"
        ).fetchone()[0]
        done_1 = conn.execute(
            "SELECT COUNT(*) FROM trajectory_steps WHERE done = 1"
        ).fetchone()[0]
        print(f"  Steps with done=False: {done_0}")
        print(f"  Steps with done=True: {done_1}")

    target_db.close()


if __name__ == "__main__":
    main()
