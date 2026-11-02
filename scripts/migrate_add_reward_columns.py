#!/usr/bin/env python3
"""Migrate existing trajectory DBs to add reward_category_relative and
reward_efficiency_normalized columns, then populate them.

Usage:
    python scripts/migrate_add_reward_columns.py data/trajectories_combined.db
    python scripts/migrate_add_reward_columns.py data/trajectories_combined.db data/trajectories_step2.db
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

# Default reward config values (must match RewardConfig defaults)
ALPHA = 1.0
BETA = 0.1
GAMMA = 0.01
TIME_BUDGET = 300.0  # seconds; matches synthesis scripts


def _add_column_if_missing(conn: sqlite3.Connection, column: str, definition: str) -> bool:
    """Add a column to trajectory_steps if it doesn't already exist."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trajectory_steps)").fetchall()}
    if column in cols:
        return False
    conn.execute(f"ALTER TABLE trajectory_steps ADD COLUMN {column} {definition}")
    conn.commit()
    print(f"  Added column: {column}")
    return True


def _compute_improvement(s_2q: int, n_2q: int) -> float:
    if s_2q == 0:
        return 0.0
    return (s_2q - n_2q) / s_2q


def migrate(db_path: Path) -> None:
    print(f"\nMigrating: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # --- 1. Add missing columns ---
    _add_column_if_missing(conn, "reward_category_relative", "REAL NOT NULL DEFAULT 0.0")
    _add_column_if_missing(conn, "reward_efficiency_normalized", "REAL NOT NULL DEFAULT 0.0")

    # --- 2. Load all steps with circuit category and optimizer name ---
    rows = conn.execute(
        """
        SELECT
            ts.id,
            ts.state_two_qubit_gates, ts.next_state_two_qubit_gates,
            ts.duration_seconds,
            c.category,
            o.name as optimizer_name
        FROM trajectory_steps ts
        JOIN trajectories t ON ts.trajectory_id = t.id
        JOIN circuits c ON t.circuit_id = c.id
        JOIN optimizers o ON ts.optimizer_id = o.id
        """
    ).fetchall()

    if not rows:
        print("  No trajectory steps found — nothing to migrate.")
        conn.close()
        return

    print(f"  Processing {len(rows)} trajectory steps...")

    # --- 3. Compute per-category and per-(category, optimizer) baselines ---
    category_improvements: dict[str, list[float]] = defaultdict(list)
    catopt_improvements: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        imp = _compute_improvement(row["state_two_qubit_gates"], row["next_state_two_qubit_gates"])
        category_improvements[row["category"]].append(imp)
        catopt_improvements[(row["category"], row["optimizer_name"])].append(imp)

    category_baseline: dict[str, float] = {
        cat: sum(vals) / len(vals) for cat, vals in category_improvements.items()
    }
    catopt_baseline: dict[tuple[str, str], float] = {
        k: sum(v) / len(v) for k, v in catopt_improvements.items()
    }

    print(f"  Category baselines ({len(category_baseline)} categories):")
    for cat, baseline in sorted(category_baseline.items()):
        print(f"    {cat:30s}: {baseline:+.4f}")

    print(f"\n  Per-(category, optimizer) baselines ({len(catopt_baseline)} pairs):")
    for (cat, opt), baseline in sorted(catopt_baseline.items()):
        print(f"    {cat:30s} {opt:20s}: {baseline:+.4f}")

    # --- 4. Compute and update both new rewards ---
    updates_catrel = []
    updates_effnorm = []

    for row in rows:
        imp = _compute_improvement(row["state_two_qubit_gates"], row["next_state_two_qubit_gates"])
        t = row["duration_seconds"]
        cat = row["category"]

        # reward_category_relative (per-(category, optimizer) baseline with category fallback)
        opt = row["optimizer_name"]
        baseline = catopt_baseline.get((cat, opt), category_baseline.get(cat, 0.0))
        reward_catrel = ALPHA * (imp - baseline) - BETA * t - GAMMA
        updates_catrel.append((reward_catrel, row["id"]))

        # reward_efficiency_normalized
        normalized_time = t / TIME_BUDGET
        reward_effnorm = ALPHA * imp - BETA * normalized_time - GAMMA
        updates_effnorm.append((reward_effnorm, row["id"]))

    conn.executemany(
        "UPDATE trajectory_steps SET reward_category_relative = ? WHERE id = ?",
        updates_catrel,
    )
    conn.executemany(
        "UPDATE trajectory_steps SET reward_efficiency_normalized = ? WHERE id = ?",
        updates_effnorm,
    )
    conn.commit()

    # --- 5. Verify ---
    sample = conn.execute(
        """SELECT reward_improvement_only, reward_efficiency,
                  reward_category_relative, reward_efficiency_normalized
           FROM trajectory_steps LIMIT 3"""
    ).fetchall()
    print("  Sample rewards (improvement_only, efficiency, catrel, eff_norm):")
    for r in sample:
        print(f"    {r[0]:+.4f}  {r[1]:+.4f}  {r[2]:+.4f}  {r[3]:+.4f}")

    conn.close()
    print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add reward_category_relative and reward_efficiency_normalized to existing DBs"
    )
    parser.add_argument("databases", nargs="+", type=Path, help="DB paths to migrate")
    args = parser.parse_args()

    for db_path in args.databases:
        if not db_path.exists():
            print(f"WARNING: {db_path} not found, skipping.")
            continue
        migrate(db_path)

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
