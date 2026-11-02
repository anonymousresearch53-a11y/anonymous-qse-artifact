#!/usr/bin/env python3
"""CLI script for analyzing trajectory database.

This script provides analysis and visualization of collected optimization
trajectories for understanding optimizer performance and synergies.

Usage:
    python scripts/analyze_trajectories.py --database data/trajectories.db
    python scripts/analyze_trajectories.py --export-d4rl data/d4rl_export.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory import (  # noqa: E402
    TrajectoryDatabase,
)


def analyze_optimizer_performance(db: TrajectoryDatabase) -> dict:
    """Analyze individual optimizer performance."""
    conn = db._get_connection()

    # Get average improvement by optimizer
    query = """
        SELECT
            o.name as optimizer_name,
            COUNT(*) as num_runs,
            AVG(
                CASE WHEN ts.state_two_qubit_gates > 0
                THEN 100.0 * (ts.state_two_qubit_gates - ts.next_state_two_qubit_gates)
                     / ts.state_two_qubit_gates
                ELSE 0 END
            ) as avg_improvement,
            AVG(ts.duration_seconds) as avg_duration,
            AVG(ts.reward_efficiency) as avg_reward
        FROM trajectory_steps ts
        JOIN optimizers o ON ts.optimizer_id = o.id
        GROUP BY o.name
        ORDER BY avg_improvement DESC
    """

    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        results.append({
            "optimizer": row["optimizer_name"],
            "num_runs": row["num_runs"],
            "avg_improvement_pct": row["avg_improvement"],
            "avg_duration_seconds": row["avg_duration"],
            "avg_reward": row["avg_reward"],
        })

    return {"optimizer_performance": results}


def analyze_category_performance(db: TrajectoryDatabase) -> dict:
    """Analyze performance by circuit category."""
    conn = db._get_connection()

    query = """
        SELECT
            c.category,
            COUNT(DISTINCT t.id) as num_trajectories,
            AVG(t.improvement_percentage) as avg_improvement,
            MAX(t.improvement_percentage) as max_improvement,
            AVG(t.total_duration_seconds) as avg_duration
        FROM trajectories t
        JOIN circuits c ON t.circuit_id = c.id
        GROUP BY c.category
        ORDER BY avg_improvement DESC
    """

    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        results.append({
            "category": row["category"],
            "num_trajectories": row["num_trajectories"],
            "avg_improvement_pct": row["avg_improvement"],
            "max_improvement_pct": row["max_improvement"],
            "avg_duration_seconds": row["avg_duration"],
        })

    return {"category_performance": results}


def analyze_chain_synergies(db: TrajectoryDatabase) -> dict:
    """Analyze optimizer chain synergies."""
    conn = db._get_connection()

    # Get best chains by average improvement
    query = """
        SELECT
            t.chain_name,
            t.num_steps,
            COUNT(*) as num_circuits,
            AVG(t.improvement_percentage) as avg_improvement,
            AVG(t.total_reward) as avg_reward
        FROM trajectories t
        WHERE t.num_steps > 1
        GROUP BY t.chain_name
        HAVING COUNT(*) >= 3
        ORDER BY avg_improvement DESC
        LIMIT 20
    """

    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        results.append({
            "chain": row["chain_name"],
            "num_steps": row["num_steps"],
            "num_circuits": row["num_circuits"],
            "avg_improvement_pct": row["avg_improvement"],
            "avg_reward": row["avg_reward"],
        })

    return {"top_chains": results}


def analyze_step_contributions(db: TrajectoryDatabase) -> dict:
    """Analyze contribution of each step in chains."""
    conn = db._get_connection()

    # Get average improvement by step index
    query = """
        SELECT
            ts.step_index,
            o.name as optimizer_name,
            COUNT(*) as num_runs,
            AVG(ts.reward_improvement_only) as avg_improvement
        FROM trajectory_steps ts
        JOIN optimizers o ON ts.optimizer_id = o.id
        GROUP BY ts.step_index, o.name
        ORDER BY ts.step_index, avg_improvement DESC
    """

    rows = conn.execute(query).fetchall()

    by_step: dict = defaultdict(list)
    for row in rows:
        by_step[row["step_index"]].append({
            "optimizer": row["optimizer_name"],
            "num_runs": row["num_runs"],
            "avg_improvement": row["avg_improvement"],
        })

    return {"step_contributions": dict(by_step)}


def find_best_optimizer_per_category(db: TrajectoryDatabase) -> dict:
    """Find the best optimizer chain for each category."""
    conn = db._get_connection()

    query = """
        WITH ranked AS (
            SELECT
                c.category,
                t.chain_name,
                t.improvement_percentage,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category
                    ORDER BY t.improvement_percentage DESC
                ) as rank
            FROM trajectories t
            JOIN circuits c ON t.circuit_id = c.id
        )
        SELECT category, chain_name, improvement_percentage
        FROM ranked
        WHERE rank = 1
        ORDER BY category
    """

    rows = conn.execute(query).fetchall()

    results = {}
    for row in rows:
        results[row["category"]] = {
            "best_chain": row["chain_name"],
            "improvement_pct": row["improvement_percentage"],
        }

    return {"best_by_category": results}


def export_to_d4rl(db: TrajectoryDatabase, output_path: Path) -> dict:
    """Export trajectories to D4RL format."""
    data = db.export_to_d4rl_format()

    # Save as numpy archive
    np.savez_compressed(
        output_path,
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_observations=data["next_observations"],
        terminals=data["terminals"],
    )

    return {
        "exported_file": str(output_path),
        "num_transitions": len(data["actions"]),
        "observation_dim": data["observations"].shape[1] if len(data["observations"]) > 0 else 0,
        "action_space_size": int(data["actions"].max()) + 1 if len(data["actions"]) > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze trajectory database for RL training"
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to trajectory database (default: data/trajectories.db)",
    )

    parser.add_argument(
        "--export-d4rl",
        type=Path,
        default=None,
        help="Export to D4RL format (.npz file)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON",
    )

    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=["all", "optimizers", "categories", "chains", "steps", "best"],
        default=["all"],
        help="Which analyses to run (default: all)",
    )

    args = parser.parse_args()

    if not args.database.exists():
        print(f"Database not found: {args.database}")
        sys.exit(1)

    db = TrajectoryDatabase(args.database)

    # Get basic statistics
    stats = db.get_statistics()

    if not args.json:
        print("=" * 60)
        print("Trajectory Database Analysis")
        print("=" * 60)
        print(f"\nDatabase: {args.database}")
        print(f"Circuits: {stats['num_circuits']}")
        print(f"Optimizers: {stats['num_optimizers']}")
        print(f"Trajectories: {stats['num_trajectories']}")
        print(f"Trajectory steps: {stats['num_trajectory_steps']}")

        if stats.get("avg_improvement_percentage") is not None:
            print("\nOverall Statistics:")
            print(f"  Avg improvement: {stats['avg_improvement_percentage']:.1f}%")
            print(f"  Max improvement: {stats['max_improvement_percentage']:.1f}%")
            print(f"  Avg total reward: {stats['avg_total_reward']:.3f}")
            print(f"  Avg trajectory length: {stats['avg_trajectory_length']:.1f}")

    analyses = set(args.analysis)
    run_all = "all" in analyses

    results = {"statistics": stats}

    # Run requested analyses
    if run_all or "optimizers" in analyses:
        analysis = analyze_optimizer_performance(db)
        results.update(analysis)

        if not args.json:
            print("\n" + "-" * 40)
            print("Optimizer Performance")
            print("-" * 40)
            for opt in analysis["optimizer_performance"]:
                print(
                    f"  {opt['optimizer']:20s} | "
                    f"Runs: {opt['num_runs']:5d} | "
                    f"Avg Improvement: {opt['avg_improvement_pct']:6.2f}% | "
                    f"Avg Duration: {opt['avg_duration_seconds']:.2f}s"
                )

    if run_all or "categories" in analyses:
        analysis = analyze_category_performance(db)
        results.update(analysis)

        if not args.json:
            print("\n" + "-" * 40)
            print("Performance by Category")
            print("-" * 40)
            for cat in analysis["category_performance"]:
                print(
                    f"  {cat['category']:20s} | "
                    f"Trajectories: {cat['num_trajectories']:5d} | "
                    f"Avg: {cat['avg_improvement_pct']:6.2f}% | "
                    f"Max: {cat['max_improvement_pct']:6.2f}%"
                )

    if run_all or "chains" in analyses:
        analysis = analyze_chain_synergies(db)
        results.update(analysis)

        if not args.json:
            print("\n" + "-" * 40)
            print("Top Optimizer Chains")
            print("-" * 40)
            for chain in analysis["top_chains"][:10]:
                print(
                    f"  {chain['chain'][:50]:50s} | "
                    f"Steps: {chain['num_steps']} | "
                    f"Circuits: {chain['num_circuits']:3d} | "
                    f"Avg: {chain['avg_improvement_pct']:6.2f}%"
                )

    if run_all or "steps" in analyses:
        analysis = analyze_step_contributions(db)
        results.update(analysis)

        if not args.json:
            print("\n" + "-" * 40)
            print("Step Contributions")
            print("-" * 40)
            for step_idx, optimizers in sorted(analysis["step_contributions"].items()):
                print(f"  Step {step_idx}:")
                for opt in optimizers[:3]:  # Top 3 per step
                    print(
                        f"    {opt['optimizer']:20s} | "
                        f"Runs: {opt['num_runs']:5d} | "
                        f"Avg Improvement: {opt['avg_improvement']:.4f}"
                    )

    if run_all or "best" in analyses:
        analysis = find_best_optimizer_per_category(db)
        results.update(analysis)

        if not args.json:
            print("\n" + "-" * 40)
            print("Best Optimizer by Category")
            print("-" * 40)
            for category, best in sorted(analysis["best_by_category"].items()):
                print(
                    f"  {category:20s} | "
                    f"{best['best_chain'][:40]:40s} | "
                    f"{best['improvement_pct']:6.2f}%"
                )

    # Export to D4RL format if requested
    if args.export_d4rl:
        export_info = export_to_d4rl(db, args.export_d4rl)
        results["d4rl_export"] = export_info

        if not args.json:
            print("\n" + "-" * 40)
            print("D4RL Export")
            print("-" * 40)
            print(f"  File: {export_info['exported_file']}")
            print(f"  Transitions: {export_info['num_transitions']}")
            print(f"  Observation dim: {export_info['observation_dim']}")
            print(f"  Action space: {export_info['action_space_size']}")

    db.close()

    # Output as JSON if requested
    if args.json:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        print(json.dumps(results, indent=2, default=convert))


if __name__ == "__main__":
    main()
