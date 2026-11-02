#!/usr/bin/env python3
"""Inspect trajectory database contents.

A simple CLI tool for exploring the trajectory database when GUI tools aren't available.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_table(headers: list[str], rows: list[tuple[Any, ...]], max_width: int = 100) -> None:
    """Print a formatted table."""
    if not rows:
        print("  (no data)")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Truncate if too wide
    total_width = sum(widths) + len(widths) * 3
    if total_width > max_width:
        scale = max_width / total_width
        widths = [max(8, int(w * scale)) for w in widths]

    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"  {header_line}")
    print("  " + "-" * len(header_line))

    # Print rows
    for row in rows:
        row_str = " | ".join(str(v).ljust(w)[:w] for v, w in zip(row, widths))
        print(f"  {row_str}")


def list_tables(db_path: Path) -> None:
    """List all tables in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print("\nTables in database:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table:20s} ({count} rows)")

    conn.close()


def show_circuits(db_path: Path, category: str | None = None, max_qubits: int | None = None) -> None:
    """Show circuits in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT name, category, num_qubits, initial_two_qubit_gates, initial_depth FROM circuits"
    params = []
    conditions = []

    if category:
        conditions.append("category LIKE ?")
        params.append(f"%{category}%")

    if max_qubits:
        conditions.append("num_qubits <= ?")
        params.append(max_qubits)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY category, num_qubits"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    print("\nCircuits:")
    print_table(["Name", "Category", "Qubits", "2Q Gates", "Depth"], rows)
    print(f"\nTotal: {len(rows)} circuits")

    conn.close()


def show_optimizers(db_path: Path) -> None:
    """Show optimizers in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name, config FROM optimizers ORDER BY id")
    rows = cursor.fetchall()

    print("\nOptimizers:")
    if rows:
        print_table(["Name", "Config"], rows)
        print(f"\nTotal: {len(rows)} optimizers")
    else:
        print("  (no optimizers yet - run grid search to populate)")

    conn.close()


def show_trajectories(db_path: Path, limit: int = 10) -> None:
    """Show best trajectories."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT 
            t.chain_name,
            c.name as circuit_name,
            t.improvement_percentage,
            t.total_reward
        FROM trajectories t
        JOIN circuits c ON t.circuit_id = c.id
        ORDER BY t.improvement_percentage DESC
        LIMIT ?
    """

    cursor.execute(query, (limit,))
    rows = cursor.fetchall()

    print(f"\nTop {limit} Trajectories by Improvement:")
    if rows:
        print_table(["Chain Name", "Circuit", "Improvement %", "Total Reward"], rows)
        print(f"\nTotal trajectories: {len(rows)}")
    else:
        print("  (no trajectories yet - run grid search to populate)")

    conn.close()


def show_schema(db_path: Path, table: str) -> None:
    """Show schema for a table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table})")
    rows = cursor.fetchall()

    print(f"\nSchema for table '{table}':")
    headers = ["ID", "Name", "Type", "Not Null", "Default", "PK"]
    print_table(headers, rows)

    conn.close()


def run_custom_query(db_path: Path, query: str) -> None:
    """Run a custom SQL query."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        if cursor.description:
            headers = [desc[0] for desc in cursor.description]
            print("\nQuery Results:")
            print_table(headers, rows)
            print(f"\nRows: {len(rows)}")
        else:
            print("Query executed successfully (no results)")

    except sqlite3.Error as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect trajectory database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --database data/trajectories.db --tables
  %(prog)s --database data/trajectories.db --circuits
  %(prog)s --database data/trajectories.db --circuits --category qft
  %(prog)s --database data/trajectories.db --schema circuits
  %(prog)s --database data/trajectories.db --query "SELECT COUNT(*) FROM circuits"
        """,
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to database (default: data/trajectories.db)",
    )

    parser.add_argument("--tables", action="store_true", help="List all tables")
    parser.add_argument("--circuits", action="store_true", help="Show circuits")
    parser.add_argument("--optimizers", action="store_true", help="Show optimizers")
    parser.add_argument("--trajectories", action="store_true", help="Show best trajectories")

    parser.add_argument("--category", type=str, help="Filter circuits by category")
    parser.add_argument("--max-qubits", type=int, help="Filter circuits by max qubits")
    parser.add_argument("--limit", type=int, default=10, help="Limit for trajectories (default: 10)")

    parser.add_argument("--schema", type=str, help="Show schema for a table")
    parser.add_argument("--query", type=str, help="Run custom SQL query")

    args = parser.parse_args()

    if not args.database.exists():
        print(f"Error: Database not found: {args.database}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {args.database.resolve()}")
    print(f"Size: {args.database.stat().st_size / 1024:.1f} KB")

    # Run requested operations
    if args.tables:
        list_tables(args.database)

    if args.circuits:
        show_circuits(args.database, args.category, args.max_qubits)

    if args.optimizers:
        show_optimizers(args.database)

    if args.trajectories:
        show_trajectories(args.database, args.limit)

    if args.schema:
        show_schema(args.database, args.schema)

    if args.query:
        run_custom_query(args.database, args.query)

    # Default action if nothing specified
    if not any([args.tables, args.circuits, args.optimizers, args.trajectories, args.schema, args.query]):
        list_tables(args.database)
        show_circuits(args.database)


if __name__ == "__main__":
    main()
