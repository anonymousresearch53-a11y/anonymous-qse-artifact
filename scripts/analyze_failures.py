#!/usr/bin/env python3
"""Analyze grid search failures from the trajectory database.

Classifies all failures into categories (harness vs optimizer), generates
summary statistics, and outputs detailed failure reports.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Error message patterns mapped to failure categories and sources
FAILURE_PATTERNS = [
    # (pattern, category, source, description)
    ("Failed to load circuit:%", "CIRCUIT_LOADING", "HARNESS",
     "Circuit QASM file not found or failed to parse"),
    ("%AIRouting runs on physical%", "AIROUTING_PHYSICAL", "OPTIMIZER",
     "AIRouting requires pre-mapped physical circuit"),
    ("Already borrowed%", "RUST_BORROW", "OPTIMIZER",
     "Rust/Python interop memory safety error in qiskit-ibm-transpiler"),
    ("%Argument list too long%", "SUBPROCESS_ARG_LIMIT", "HARNESS",
     "Circuit QASM embedded in subprocess cmd exceeds OS ARG_MAX"),
    ("%OpenQASM 2 cannot represent%", "QASM2_LIMITATION", "HARNESS",
     "Circuit uses OpenQASM 3 features (if_else) incompatible with qasm2.dumps"),
    ("%register name%already exists%", "REGISTER_CONFLICT", "HARNESS",
     "Duplicate register names in QASM parsing/generation"),
    ("%Unable to translate the operations%", "UNSUPPORTED_GATES", "OPTIMIZER",
     "Circuit contains gates not in optimizer's supported gate set"),
    ("Optimizer not found%", "OPTIMIZER_NOT_FOUND", "HARNESS",
     "Optimizer not registered in database"),
    ("No result returned%", "NO_RESULT", "OPTIMIZER",
     "Optimizer returned empty result"),
]


def build_classification_case() -> str:
    """Build SQL CASE expression for classifying error messages."""
    clauses = []
    for pattern, category, source, _ in FAILURE_PATTERNS:
        clauses.append(
            f"WHEN r.error_message LIKE '{pattern}' THEN '{category}'"
        )
    clauses.append("ELSE 'UNCATEGORIZED'")
    return "CASE\n            " + "\n            ".join(clauses) + "\n        END"


def build_source_case() -> str:
    """Build SQL CASE expression for harness vs optimizer source."""
    clauses = []
    for pattern, _, source, _ in FAILURE_PATTERNS:
        clauses.append(
            f"WHEN r.error_message LIKE '{pattern}' THEN '{source}'"
        )
    clauses.append("ELSE 'UNKNOWN'")
    return "CASE\n            " + "\n            ".join(clauses) + "\n        END"


def get_description(category: str) -> str:
    """Get human-readable description for a failure category."""
    for _, cat, _, desc in FAILURE_PATTERNS:
        if cat == category:
            return desc
    return "Unrecognized error pattern"


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_table(headers: list[str], rows: list[tuple], col_widths: list[int] | None = None) -> None:
    if not rows:
        print("  (no data)")
        return

    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
        col_widths = [min(w, 60) for w in col_widths]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print("  " + "-" * len(header_line))
    for row in rows:
        row_str = " | ".join(str(v).ljust(w)[:w] for v, w in zip(row, col_widths))
        print(f"  {row_str}")


def analyze_failures(db_path: Path, export_csv: Path | None = None) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    classification_case = build_classification_case()
    source_case = build_source_case()

    # --- Overall statistics ---
    print_section("OVERALL STATISTICS")

    cursor.execute("SELECT COUNT(*) FROM optimization_runs")
    total_runs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM optimization_runs WHERE success = 0")
    total_failures = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM optimization_runs WHERE success = 1")
    total_successes = cursor.fetchone()[0]

    print(f"  Total runs:     {total_runs}")
    print(f"  Successful:     {total_successes} ({100 * total_successes / total_runs:.1f}%)")
    print(f"  Failed:         {total_failures} ({100 * total_failures / total_runs:.1f}%)")

    # --- Category breakdown ---
    print_section("FAILURES BY CATEGORY")

    query = f"""
        SELECT
            {classification_case} as category,
            {source_case} as source,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / ?, 1) as pct
        FROM optimization_runs r
        WHERE r.success = 0
        GROUP BY category, source
        ORDER BY count DESC
    """
    cursor.execute(query, (total_failures,))
    rows = cursor.fetchall()

    print_table(
        ["Category", "Source", "Count", "% of Failures"],
        rows,
        [25, 10, 8, 14],
    )

    # Summary by source
    print("\n  --- Summary by Source ---")
    for source_label in ("HARNESS", "OPTIMIZER", "UNKNOWN"):
        count = sum(r[2] for r in rows if r[1] == source_label)
        if count > 0:
            print(f"  {source_label:12s}: {count:4d} failures ({100 * count / total_failures:.1f}%)")

    # --- Per-optimizer failure rates ---
    print_section("FAILURE RATES BY OPTIMIZER")

    query = """
        SELECT
            o.name as optimizer,
            COUNT(*) as total,
            SUM(CASE WHEN r.success = 1 THEN 1 ELSE 0 END) as success,
            SUM(CASE WHEN r.success = 0 THEN 1 ELSE 0 END) as failed,
            ROUND(100.0 * SUM(CASE WHEN r.success = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as failure_rate
        FROM optimization_runs r
        JOIN optimizers o ON r.optimizer_id = o.id
        GROUP BY o.name
        ORDER BY failure_rate DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    print_table(
        ["Optimizer", "Total", "Success", "Failed", "Failure %"],
        rows,
        [18, 8, 8, 8, 10],
    )

    # --- Per-category detail with affected circuits ---
    print_section("DETAILED CATEGORY ANALYSIS")

    query = f"""
        SELECT
            {classification_case} as category,
            {source_case} as source,
            o.name as optimizer,
            COUNT(*) as count,
            GROUP_CONCAT(DISTINCT c.name) as circuits
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 0
        GROUP BY category, source, o.name
        ORDER BY category, count DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    current_category = None
    for category, source, optimizer, count, circuits in rows:
        if category != current_category:
            current_category = category
            desc = get_description(category)
            print(f"\n  [{source}] {category} — {desc}")
            print(f"  {'─' * 60}")

        circuit_list = circuits.split(",")
        circuit_display = ", ".join(circuit_list[:5])
        if len(circuit_list) > 5:
            circuit_display += f" ... (+{len(circuit_list) - 5} more)"
        print(f"    {optimizer:18s} {count:4d} failures │ {circuit_display}")

    # --- Circuits failing for ALL optimizers ---
    print_section("CIRCUITS FAILING FOR ALL OPTIMIZERS (COMPLETE FAILURES)")

    cursor.execute("SELECT COUNT(*) FROM optimizers")
    num_optimizers = cursor.fetchone()[0]

    query = """
        SELECT
            c.name,
            c.category,
            c.num_qubits,
            COUNT(DISTINCT o.name) as failed_optimizers,
            GROUP_CONCAT(DISTINCT r.error_message) as errors
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 0
        GROUP BY c.id
        HAVING failed_optimizers >= ?
        ORDER BY c.name
    """
    cursor.execute(query, (num_optimizers,))
    rows = cursor.fetchall()

    if rows:
        for name, category, qubits, failed_opts, errors in rows:
            error_list = errors.split(",")
            unique_errors = list(set(e.strip()[:60] for e in error_list))
            print(f"\n  {name} ({category}, {qubits}q) — fails all {failed_opts} optimizers")
            for err in unique_errors[:3]:
                print(f"    Error: {err}")
    else:
        print("  No circuits fail for ALL optimizers.")

    # --- Circuits with highest failure counts ---
    print_section("CIRCUITS WITH MOST FAILURES")

    query = """
        SELECT
            c.name,
            c.category,
            c.num_qubits,
            COUNT(*) as failure_count,
            COUNT(DISTINCT o.name) as failed_optimizers,
            GROUP_CONCAT(DISTINCT o.name) as optimizers
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 0
        GROUP BY c.id
        ORDER BY failure_count DESC
        LIMIT 15
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    print_table(
        ["Circuit", "Category", "Qubits", "Failures", "# Opts", "Failed Optimizers"],
        rows,
        [25, 12, 8, 10, 8, 40],
    )

    # --- Uncategorized errors ---
    print_section("UNCATEGORIZED ERRORS (need investigation)")

    query = f"""
        SELECT
            r.error_message,
            o.name as optimizer,
            c.name as circuit,
            c.num_qubits
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 0
          AND {classification_case} = 'UNCATEGORIZED'
        ORDER BY o.name, c.name
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if rows:
        for error, optimizer, circuit, qubits in rows:
            error_display = (error or "NULL")[:80]
            print(f"  [{optimizer}] {circuit} ({qubits}q): {error_display}")
    else:
        print("  All errors are categorized.")

    # --- Export CSV ---
    if export_csv:
        query = f"""
            SELECT
                r.id,
                c.name as circuit,
                c.category as circuit_category,
                c.num_qubits,
                o.name as optimizer,
                {classification_case} as failure_category,
                {source_case} as failure_source,
                r.error_message,
                r.duration_seconds,
                r.created_at
            FROM optimization_runs r
            JOIN circuits c ON r.circuit_id = c.id
            JOIN optimizers o ON r.optimizer_id = o.id
            WHERE r.success = 0
            ORDER BY failure_category, o.name, c.name
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]

        with open(export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\n  Exported {len(rows)} failure records to {export_csv}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze grid search failures from the trajectory database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to database (default: data/trajectories.db)",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Export detailed failures to CSV file",
    )

    args = parser.parse_args()

    if not args.database.exists():
        print(f"Error: Database not found: {args.database}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {args.database.resolve()}")
    print(f"Size: {args.database.stat().st_size / 1024:.1f} KB")

    analyze_failures(args.database, args.export_csv)


if __name__ == "__main__":
    main()
