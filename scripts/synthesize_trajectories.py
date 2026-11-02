#!/usr/bin/env python3
"""Convert optimization_runs into trajectory_steps for RL training.

Each successful optimization_run becomes a 1-step trajectory in the
trajectory_steps table, with all 4 reward variants computed.

Usage:
    python scripts/synthesize_trajectories.py --database data/trajectories_step2.db
    python scripts/synthesize_trajectories.py --database data/trajectories_step2.db --dry-run
"""

from __future__ import annotations

import argparse
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
from benchmarks.ai_transpile.rl_trajectory.state import (  # noqa: E402
    get_category_encoding,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics  # noqa: E402


def synthesize(db: TrajectoryDatabase, *, dry_run: bool = False, time_budget: float = 300.0) -> int:
    """Convert optimization_runs into trajectory_steps.

    Args:
        db: Trajectory database
        dry_run: If True, count but don't insert
        time_budget: Time budget per episode (for state construction)

    Returns:
        Number of trajectories created
    """
    conn = db._get_connection()
    reward_config = RewardConfig()

    # Get all successful optimization runs with circuit info
    rows = conn.execute(
        """
        SELECT
            r.id as run_id,
            r.circuit_id, r.optimizer_id,
            r.input_depth, r.input_two_qubit_gates,
            r.input_two_qubit_depth, r.input_total_gates,
            r.output_depth, r.output_two_qubit_gates,
            r.output_two_qubit_depth, r.output_total_gates,
            r.duration_seconds,
            c.num_qubits, c.category
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        WHERE r.success = 1
        ORDER BY r.circuit_id, r.optimizer_id
        """
    ).fetchall()

    if not rows:
        print("No successful optimization runs found.")
        return 0

    # Check existing trajectory count to avoid duplicates
    existing_count = db.count_trajectory_steps()
    if existing_count > 0:
        print(f"Database already has {existing_count} trajectory steps.")
        print("Skipping synthesis to avoid duplicates. Clear trajectory_steps first if needed.")
        return 0

    created = 0
    for row in rows:
        num_qubits = row["num_qubits"]
        category = row["category"]

        # Build CircuitMetrics for reward computation
        input_metrics = CircuitMetrics(
            depth=row["input_depth"],
            two_qubit_gates=row["input_two_qubit_gates"],
            two_qubit_depth=row["input_two_qubit_depth"],
            total_gates=row["input_total_gates"],
        )
        output_metrics = CircuitMetrics(
            depth=row["output_depth"],
            two_qubit_gates=row["output_two_qubit_gates"],
            two_qubit_depth=row["output_two_qubit_depth"],
            total_gates=row["output_total_gates"],
        )

        # Compute all reward variants
        rewards = compute_all_rewards(
            prev_metrics=input_metrics,
            new_metrics=output_metrics,
            time_cost=row["duration_seconds"],
            initial_metrics=input_metrics,
            is_final_step=True,
            config=reward_config,
            time_budget=time_budget,
        )

        # Derived features for state
        gate_density = input_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        two_qubit_ratio = (
            input_metrics.two_qubit_gates / input_metrics.total_gates
            if input_metrics.total_gates > 0
            else 0.0
        )

        # Next state derived features
        next_gate_density = output_metrics.total_gates / num_qubits if num_qubits > 0 else 0.0
        next_two_qubit_ratio = (
            output_metrics.two_qubit_gates / output_metrics.total_gates
            if output_metrics.total_gates > 0
            else 0.0
        )

        category_encoding = get_category_encoding(category)

        improvement_pct = compute_improvement_percentage(
            input_metrics.two_qubit_gates, output_metrics.two_qubit_gates
        )

        if dry_run:
            created += 1
            continue

        # Insert trajectory (episode wrapper)
        trajectory_id = db.insert_trajectory(
            circuit_id=row["circuit_id"],
            chain_name=f"single_{row['optimizer_id']}",
            num_steps=1,
            initial_depth=input_metrics.depth,
            initial_two_qubit_gates=input_metrics.two_qubit_gates,
            initial_two_qubit_depth=input_metrics.two_qubit_depth,
            initial_total_gates=input_metrics.total_gates,
            final_depth=output_metrics.depth,
            final_two_qubit_gates=output_metrics.two_qubit_gates,
            final_two_qubit_depth=output_metrics.two_qubit_depth,
            final_total_gates=output_metrics.total_gates,
            total_duration_seconds=row["duration_seconds"],
            total_reward=rewards.efficiency,
            improvement_percentage=improvement_pct,
            metadata={"source": "synthesized_from_optimization_runs", "run_id": row["run_id"]},
        )

        # Insert trajectory step
        from benchmarks.ai_transpile.rl_trajectory.database import TrajectoryStepRecord

        step = TrajectoryStepRecord(
            trajectory_id=trajectory_id,
            step_index=0,
            optimizer_id=row["optimizer_id"],
            state_depth=input_metrics.depth,
            state_two_qubit_gates=input_metrics.two_qubit_gates,
            state_two_qubit_depth=input_metrics.two_qubit_depth,
            state_total_gates=input_metrics.total_gates,
            state_num_qubits=num_qubits,
            state_gate_density=gate_density,
            state_two_qubit_ratio=two_qubit_ratio,
            state_steps_taken=0,
            state_time_budget_remaining=time_budget,
            state_category=category_encoding,
            next_state_depth=output_metrics.depth,
            next_state_two_qubit_gates=output_metrics.two_qubit_gates,
            next_state_two_qubit_depth=output_metrics.two_qubit_depth,
            next_state_total_gates=output_metrics.total_gates,
            next_state_gate_density=next_gate_density,
            next_state_two_qubit_ratio=next_two_qubit_ratio,
            next_state_steps_taken=1,
            next_state_time_budget_remaining=time_budget - row["duration_seconds"],
            reward_improvement_only=rewards.improvement_only,
            reward_efficiency=rewards.efficiency,
            reward_multi_objective=rewards.multi_objective,
            reward_sparse_final=rewards.sparse_final,
            reward_efficiency_normalized=rewards.efficiency_normalized,
            done=True,
            duration_seconds=row["duration_seconds"],
        )
        db.insert_trajectory_step(step)
        created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize trajectory_steps from optimization_runs"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories_step2.db"),
        help="Path to trajectory database",
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
        help="Count trajectories without inserting",
    )

    args = parser.parse_args()

    if not args.database.exists():
        print(f"Database not found: {args.database}")
        sys.exit(1)

    db = TrajectoryDatabase(args.database)

    # Show current stats
    stats = db.get_statistics()
    print(f"Database: {args.database}")
    print(f"  Circuits: {stats['num_circuits']}")
    print(f"  Optimizers: {stats['num_optimizers']}")
    print(f"  Optimization runs: {stats['num_optimization_runs']}")
    print(f"  Existing trajectories: {stats['num_trajectories']}")
    print(f"  Existing trajectory steps: {stats['num_trajectory_steps']}")
    print()

    if args.dry_run:
        print("Dry run: counting potential trajectories...")
    else:
        print("Synthesizing trajectories from optimization runs...")

    created = synthesize(db, dry_run=args.dry_run, time_budget=args.time_budget)

    if args.dry_run:
        print(f"\nWould create {created} trajectories (1 step each)")
    else:
        print(f"\nCreated {created} trajectories ({created} steps)")

        # Show updated stats
        stats = db.get_statistics()
        print("\nUpdated statistics:")
        print(f"  Trajectories: {stats['num_trajectories']}")
        print(f"  Trajectory steps: {stats['num_trajectory_steps']}")

    db.close()


if __name__ == "__main__":
    main()
