#!/usr/bin/env python3
"""Repair stale optimization_runs.artifact_path values in the second-opt DB.

This script rewrites artifact paths in `optimization_runs.artifact_path` from an
old prefix to a new prefix. By default it performs a dry run and prints the
count plus sample rewrites. Pass `--apply` to execute the update. When applying,
the script writes a SQLite backup next to the target database unless
`--no-backup` is supplied.
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
    "./mqt/artifacts/"
)
NEW_PREFIX = (
    "./data/mqt_second_opt_artifacts/"
)


@dataclass(frozen=True)
class RepairPlan:
    runs_to_update: int
    rewritten_paths_existing_on_disk: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair combined_second_opt.db by rewriting stale "
            "optimization_runs.artifact_path prefixes."
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
        help="Path prefix to replace in optimization_runs.artifact_path.",
    )
    parser.add_argument(
        "--new-prefix",
        default=NEW_PREFIX,
        help="Replacement prefix for optimization_runs.artifact_path.",
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
        help="How many sample rows to print.",
    )
    return parser


def open_db_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def rewritten_path(path_value: str, old_prefix: str, new_prefix: str) -> str:
    return f"{new_prefix}{path_value[len(old_prefix):]}"


def compute_plan(conn: sqlite3.Connection, old_prefix: str, new_prefix: str) -> RepairPlan:
    rows = conn.execute(
        """
        SELECT artifact_path
        FROM optimization_runs
        WHERE artifact_path LIKE ?
        """,
        (f"{old_prefix}%",),
    ).fetchall()

    runs_to_update = len(rows)
    rewritten_paths_existing_on_disk = sum(
        1
        for row in rows
        if Path(rewritten_path(str(row["artifact_path"]), old_prefix, new_prefix)).exists()
    )

    return RepairPlan(
        runs_to_update=runs_to_update,
        rewritten_paths_existing_on_disk=rewritten_paths_existing_on_disk,
    )


def print_samples(
    conn: sqlite3.Connection,
    *,
    old_prefix: str,
    new_prefix: str,
    sample_limit: int,
) -> None:
    print("\nSample artifact_path updates:")
    rows = conn.execute(
        """
        SELECT r.id, c.name AS circuit_name, o.name AS optimizer_name, r.artifact_path
        FROM optimization_runs r
        JOIN circuits c ON c.id = r.circuit_id
        JOIN optimizers o ON o.id = r.optimizer_id
        WHERE r.artifact_path LIKE ?
        ORDER BY r.id
        LIMIT ?
        """,
        (f"{old_prefix}%", sample_limit),
    ).fetchall()
    if not rows:
        print("  (none)")
        return

    for row in rows:
        old_path = str(row["artifact_path"])
        new_path = rewritten_path(old_path, old_prefix, new_prefix)
        print(
            f"  run_id={row['id']} circuit={row['circuit_name']} optimizer={row['optimizer_name']}"
        )
        print(f"    old: {old_path}")
        print(f"    new: {new_path}")
        print(f"    exists_after_rewrite: {Path(new_path).exists()}")


def make_backup(db_path: Path) -> Path:
    backup_path = db_path.with_name(f"{db_path.stem}.artifact_path_backup{db_path.suffix}")
    shutil.copy2(db_path, backup_path)

    wal_path = db_path.with_name(f"{db_path.name}-wal")
    shm_path = db_path.with_name(f"{db_path.name}-shm")
    if wal_path.exists():
        shutil.copy2(wal_path, backup_path.with_name(f"{backup_path.name}-wal"))
    if shm_path.exists():
        shutil.copy2(shm_path, backup_path.with_name(f"{backup_path.name}-shm"))
    return backup_path


def apply_repair(conn: sqlite3.Connection, *, old_prefix: str, new_prefix: str) -> None:
    with conn:
        conn.execute(
            """
            UPDATE optimization_runs
            SET artifact_path = ? || substr(artifact_path, ?)
            WHERE artifact_path LIKE ?
            """,
            (new_prefix, len(old_prefix) + 1, f"{old_prefix}%"),
        )


def main() -> None:
    args = build_parser().parse_args()
    db_path = args.database.resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = open_db_rw(db_path)
    try:
        plan = compute_plan(conn, args.old_prefix, args.new_prefix)

        print(f"Database: {db_path}")
        print(f"Dry run: {not args.apply}")
        print(f"Runs needing artifact_path updates: {plan.runs_to_update}")
        print(
            "Rewritten target paths that already exist on disk: "
            f"{plan.rewritten_paths_existing_on_disk}"
        )
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

        apply_repair(conn, old_prefix=args.old_prefix, new_prefix=args.new_prefix)
        remaining = compute_plan(conn, args.old_prefix, args.new_prefix)
        print("\nRepair complete.")
        print(f"Remaining runs with old prefix: {remaining.runs_to_update}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
