#!/usr/bin/env python3
"""Repair stale MQT artifact paths in the combined second-opt database.

This script updates artifact-circuit paths in the `circuits` table from the old
`mqt_artifacts` prefix to the newer `data/mqt_single_opt` prefix, then removes
failed `optimization_runs` rows whose error message indicates the input circuit
could not be loaded.

By default the script performs a dry run and prints the counts it would change.
Pass `--apply` to execute the repair. When applying, a SQLite backup is written
next to the target database unless `--no-backup` is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = (
    PROJECT_ROOT / "dbs" / "combined_fixed" / "data_second_opt" / "combined_second_opt.db"
)
OLD_PREFIX = (
    "./mqt_artifacts/"
)
NEW_PREFIX = (
    "./data/mqt_single_opt/"
)
LOAD_FAILURE_PREFIX = "Failed to load circuit:"


@dataclass(frozen=True)
class RepairPlan:
    circuits_to_update: int
    failed_runs_to_delete: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair combined_second_opt.db by fixing stale MQT artifact prefixes "
            "and deleting failed optimization runs with load errors."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the second-opt SQLite database.",
    )
    parser.add_argument(
        "--old-prefix",
        default=OLD_PREFIX,
        help="Path prefix to replace in circuits.qasm_path.",
    )
    parser.add_argument(
        "--new-prefix",
        default=NEW_PREFIX,
        help="Replacement prefix for circuits.qasm_path.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the repair. Without this flag, the script performs a dry run.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup copy before applying changes.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="How many sample rows to print for each action.",
    )
    return parser


def open_db_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def compute_plan(conn: sqlite3.Connection, old_prefix: str) -> RepairPlan:
    circuits_to_update = conn.execute(
        """
        SELECT COUNT(*)
        FROM circuits
        WHERE qasm_path LIKE ?
        """,
        (f"{old_prefix}%",),
    ).fetchone()[0]

    failed_runs_to_delete = conn.execute(
        """
        SELECT COUNT(*)
        FROM optimization_runs
        WHERE success = 0
          AND error_message LIKE ?
        """,
        (f"{LOAD_FAILURE_PREFIX}%",),
    ).fetchone()[0]

    return RepairPlan(
        circuits_to_update=int(circuits_to_update),
        failed_runs_to_delete=int(failed_runs_to_delete),
    )


def print_samples(
    conn: sqlite3.Connection,
    *,
    old_prefix: str,
    new_prefix: str,
    sample_limit: int,
) -> None:
    print("\nSample circuit path updates:")
    rows = conn.execute(
        """
        SELECT name, qasm_path
        FROM circuits
        WHERE qasm_path LIKE ?
        ORDER BY name
        LIMIT ?
        """,
        (f"{old_prefix}%", sample_limit),
    ).fetchall()
    if not rows:
        print("  (none)")
    for row in rows:
        old_path = str(row["qasm_path"])
        print(f"  {row['name']}")
        print(f"    old: {old_path}")
        print(f"    new: {old_path.replace(old_prefix, new_prefix, 1)}")

    print("\nSample failed runs to delete:")
    rows = conn.execute(
        """
        SELECT r.id, c.name AS circuit_name, o.name AS optimizer_name, r.error_message
        FROM optimization_runs r
        JOIN circuits c ON c.id = r.circuit_id
        JOIN optimizers o ON o.id = r.optimizer_id
        WHERE r.success = 0
          AND r.error_message LIKE ?
        ORDER BY r.id
        LIMIT ?
        """,
        (f"{LOAD_FAILURE_PREFIX}%", sample_limit),
    ).fetchall()
    if not rows:
        print("  (none)")
    for row in rows:
        print(
            f"  run_id={row['id']} circuit={row['circuit_name']} "
            f"optimizer={row['optimizer_name']}"
        )
        print(f"    error: {row['error_message']}")


def make_backup(db_path: Path) -> Path:
    backup_path = db_path.with_name(f"{db_path.stem}.pre_repair_backup{db_path.suffix}")
    shutil.copy2(db_path, backup_path)

    wal_path = db_path.with_name(f"{db_path.name}-wal")
    shm_path = db_path.with_name(f"{db_path.name}-shm")
    if wal_path.exists():
        shutil.copy2(wal_path, backup_path.with_name(f"{backup_path.name}-wal"))
    if shm_path.exists():
        shutil.copy2(shm_path, backup_path.with_name(f"{backup_path.name}-shm"))
    return backup_path


def apply_repair(
    conn: sqlite3.Connection,
    *,
    old_prefix: str,
    new_prefix: str,
) -> RepairPlan:
    with conn:
        conn.execute(
            """
            UPDATE circuits
            SET qasm_path = ? || substr(qasm_path, ?)
            WHERE qasm_path LIKE ?
            """,
            (new_prefix, len(old_prefix) + 1, f"{old_prefix}%"),
        )
        conn.execute(
            """
            DELETE FROM optimization_runs
            WHERE success = 0
              AND error_message LIKE ?
            """,
            (f"{LOAD_FAILURE_PREFIX}%",),
        )

    return compute_plan(conn, old_prefix)


def main() -> None:
    args = build_parser().parse_args()
    db_path = args.database.resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = open_db_rw(db_path)
    try:
        plan = compute_plan(conn, args.old_prefix)

        print(f"Database: {db_path}")
        print(f"Dry run: {not args.apply}")
        print(f"Circuits needing path updates: {plan.circuits_to_update}")
        print(f"Failed load-error runs to delete: {plan.failed_runs_to_delete}")
        print_samples(
            conn,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            sample_limit=args.sample_limit,
        )

        if not args.apply:
            return

        if not args.no_backup:
            backup_path = make_backup(db_path)
            print(f"\nBackup created: {backup_path}")

        remaining = apply_repair(
            conn,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
        )

        print("\nRepair complete.")
        print(f"Remaining circuits with old prefix: {remaining.circuits_to_update}")
        print(f"Remaining failed load-error runs: {remaining.failed_runs_to_delete}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
