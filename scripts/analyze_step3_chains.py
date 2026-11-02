#!/usr/bin/env python3
"""Analyze 3-step optimization chain performance across all databases.

Read-only analysis that traces full chains through step1, step2, and step3
databases and reports on end-to-end improvement statistics.

Usage:
    python scripts/analyze_step3_chains.py
    python scripts/analyze_step3_chains.py --step1-db data/trajectories.db \
        --step2-db data/trajectories_step2.db --step3-db data/trajectories_step3.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a database in read-only mode."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_step3_artifact_name(name: str) -> tuple[str, str, str] | None:
    """Parse step3 circuit name into (original_circuit, opt1, opt2).

    Step3 names follow: artifact_artifact_{orig}__{opt1}__{opt2}

    We parse in two rounds:
      Round 1: artifact_(.+)__(\w+) -> (artifact_{orig}__{opt1}, opt2)
      Round 2: artifact_(.+)__(\w+) -> (orig, opt1)
    """
    # Round 1: extract the step2 circuit name and step2 optimizer
    m1 = re.match(r"^artifact_(.+)__(\w+)$", name)
    if not m1:
        return None
    step2_circuit_name = m1.group(1)
    opt2 = m1.group(2)

    # Round 2: extract the original circuit name and step1 optimizer
    m2 = re.match(r"^artifact_(.+)__(\w+)$", step2_circuit_name)
    if not m2:
        return None
    orig_circuit = m2.group(1)
    opt1 = m2.group(2)

    return orig_circuit, opt1, opt2


def _build_step1_lookup(
    step1_conn: sqlite3.Connection,
) -> dict[tuple[str, str], sqlite3.Row]:
    """Build lookup: (circuit_name, optimizer_name) -> best step1 run."""
    rows = step1_conn.execute(
        """
        SELECT
            r.id as run_id,
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
    """Build lookup: (step2_circuit_name, optimizer_name) -> best step2 run."""
    rows = step2_conn.execute(
        """
        SELECT
            r.id as run_id,
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


def analyze(
    step1_db_path: Path,
    step2_db_path: Path,
    step3_db_path: Path,
) -> None:
    """Analyze 3-step chains and print summary report."""
    step1_conn = _open_readonly(step1_db_path)
    step2_conn = _open_readonly(step2_db_path)
    step3_conn = _open_readonly(step3_db_path)

    # Build lookups
    print("Building step-1 lookup...")
    step1_lookup = _build_step1_lookup(step1_conn)
    print(f"  Step-1 unique (circuit, optimizer) pairs: {len(step1_lookup)}")

    print("Building step-2 lookup...")
    step2_lookup = _build_step2_lookup(step2_conn)
    print(f"  Step-2 unique (circuit, optimizer) pairs: {len(step2_lookup)}")

    # Get step-3 successful runs
    step3_rows = step3_conn.execute(
        """
        SELECT
            r.id as run_id,
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
    print()

    # Also build step1-only best results per original circuit (for comparison)
    # best step1 result per original circuit (across all optimizers)
    step1_best_per_circuit: dict[str, sqlite3.Row] = {}
    for (circuit_name, _), row in step1_lookup.items():
        if circuit_name not in step1_best_per_circuit:
            step1_best_per_circuit[circuit_name] = row
        elif row["output_two_qubit_gates"] < step1_best_per_circuit[circuit_name]["output_two_qubit_gates"]:
            step1_best_per_circuit[circuit_name] = row

    # Trace chains
    chains = []
    skipped_parse = 0
    skipped_s1 = 0
    skipped_s2 = 0

    for s3_row in step3_rows:
        parsed = _parse_step3_artifact_name(s3_row["circuit_name"])
        if parsed is None:
            skipped_parse += 1
            continue

        orig_circuit, opt1, opt2 = parsed
        opt3 = s3_row["step3_optimizer_name"]

        # Look up step-1 run
        s1_row = step1_lookup.get((orig_circuit, opt1))
        if s1_row is None:
            skipped_s1 += 1
            continue

        # Look up step-2 run
        step2_circuit_name = f"artifact_{orig_circuit}__{opt1}"
        s2_row = step2_lookup.get((step2_circuit_name, opt2))
        if s2_row is None:
            skipped_s2 += 1
            continue

        # Original circuit metrics (from step1 input)
        orig_2q = s1_row["input_two_qubit_gates"]
        orig_depth = s1_row["input_depth"]

        # Final metrics (from step3 output)
        final_2q = s3_row["output_two_qubit_gates"]
        final_depth = s3_row["output_depth"]

        # Intermediate metrics
        after_s1_2q = s1_row["output_two_qubit_gates"]
        after_s2_2q = s2_row["output_two_qubit_gates"]

        # Compute improvements
        e2e_improvement = (
            (orig_2q - final_2q) / orig_2q * 100.0 if orig_2q > 0 else 0.0
        )

        # Step1-only improvement for this circuit's best single optimizer
        best_s1 = step1_best_per_circuit.get(orig_circuit)
        best_s1_2q = best_s1["output_two_qubit_gates"] if best_s1 else orig_2q
        best_s1_improvement = (
            (orig_2q - best_s1_2q) / orig_2q * 100.0 if orig_2q > 0 else 0.0
        )

        # Total duration across all 3 steps
        total_duration = (
            s1_row["duration_seconds"]
            + s2_row["duration_seconds"]
            + s3_row["duration_seconds"]
        )

        chains.append({
            "orig_circuit": orig_circuit,
            "category": s1_row["category"],
            "num_qubits": s1_row["num_qubits"],
            "opt1": opt1,
            "opt2": opt2,
            "opt3": opt3,
            "chain": f"{opt1} -> {opt2} -> {opt3}",
            "orig_2q": orig_2q,
            "after_s1_2q": after_s1_2q,
            "after_s2_2q": after_s2_2q,
            "final_2q": final_2q,
            "e2e_improvement": e2e_improvement,
            "best_s1_improvement": best_s1_improvement,
            "chain_vs_single": e2e_improvement - best_s1_improvement,
            "total_duration": total_duration,
            "orig_depth": orig_depth,
            "final_depth": final_depth,
        })

    print(f"Traced {len(chains)} complete 3-step chains")
    if skipped_parse > 0:
        print(f"  Skipped (parse failed): {skipped_parse}")
    if skipped_s1 > 0:
        print(f"  Skipped (no step-1 match): {skipped_s1}")
    if skipped_s2 > 0:
        print(f"  Skipped (no step-2 match): {skipped_s2}")
    print()

    if not chains:
        print("No chains to analyze.")
        step1_conn.close()
        step2_conn.close()
        step3_conn.close()
        return

    # === Summary Statistics ===
    print("=" * 80)
    print("3-STEP CHAIN ANALYSIS SUMMARY")
    print("=" * 80)

    improvements = [c["e2e_improvement"] for c in chains]
    chain_vs_single = [c["chain_vs_single"] for c in chains]
    durations = [c["total_duration"] for c in chains]

    print(f"\nTotal 3-step chains analyzed: {len(chains)}")
    print("\nEnd-to-End 2Q Gate Improvement (original -> step3 output):")
    print(f"  Mean:   {sum(improvements) / len(improvements):+.2f}%")
    print(f"  Median: {sorted(improvements)[len(improvements) // 2]:+.2f}%")
    print(f"  Min:    {min(improvements):+.2f}%")
    print(f"  Max:    {max(improvements):+.2f}%")
    positive = sum(1 for x in improvements if x > 0)
    print(f"  Chains with improvement: {positive}/{len(chains)} ({positive / len(chains) * 100:.1f}%)")

    print("\n3-Step Chain vs Best Single-Step Optimizer:")
    print(f"  Mean advantage:   {sum(chain_vs_single) / len(chain_vs_single):+.2f}pp")
    print(f"  Median advantage: {sorted(chain_vs_single)[len(chain_vs_single) // 2]:+.2f}pp")
    beats_single = sum(1 for x in chain_vs_single if x > 0)
    print(f"  Chains beating single-step: {beats_single}/{len(chains)} ({beats_single / len(chains) * 100:.1f}%)")

    print("\nTotal Duration (all 3 steps):")
    print(f"  Mean:   {sum(durations) / len(durations):.1f}s")
    print(f"  Median: {sorted(durations)[len(durations) // 2]:.1f}s")
    print(f"  Min:    {min(durations):.1f}s")
    print(f"  Max:    {max(durations):.1f}s")

    # === Per-Category Analysis ===
    print(f"\n{'=' * 80}")
    print("PER-CATEGORY ANALYSIS")
    print(f"{'=' * 80}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for c in chains:
        by_category[c["category"]].append(c)

    print(f"\n{'Category':<25} {'Count':>6} {'Mean E2E':>10} {'Beats 1-step':>14} {'Mean Adv':>10}")
    print("-" * 70)
    for cat in sorted(by_category.keys()):
        cat_chains = by_category[cat]
        mean_e2e = sum(c["e2e_improvement"] for c in cat_chains) / len(cat_chains)
        beats = sum(1 for c in cat_chains if c["chain_vs_single"] > 0)
        mean_adv = sum(c["chain_vs_single"] for c in cat_chains) / len(cat_chains)
        print(f"{cat:<25} {len(cat_chains):>6} {mean_e2e:>+9.2f}% {beats:>6}/{len(cat_chains):<6} {mean_adv:>+9.2f}pp")

    # === Best Chain Combinations ===
    print(f"\n{'=' * 80}")
    print("BEST 3-STEP CHAIN COMBINATIONS (by mean E2E improvement)")
    print(f"{'=' * 80}")

    by_chain: dict[str, list[dict]] = defaultdict(list)
    for c in chains:
        by_chain[c["chain"]].append(c)

    chain_stats = []
    for chain_name, chain_list in by_chain.items():
        mean_e2e = sum(c["e2e_improvement"] for c in chain_list) / len(chain_list)
        mean_adv = sum(c["chain_vs_single"] for c in chain_list) / len(chain_list)
        beats = sum(1 for c in chain_list if c["chain_vs_single"] > 0)
        chain_stats.append((chain_name, len(chain_list), mean_e2e, mean_adv, beats))

    chain_stats.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Chain':<50} {'N':>5} {'Mean E2E':>10} {'Mean Adv':>10} {'Beats':>7}")
    print("-" * 87)
    for name, n, mean_e2e, mean_adv, beats in chain_stats[:20]:
        print(f"{name:<50} {n:>5} {mean_e2e:>+9.2f}% {mean_adv:>+9.2f}pp {beats:>4}/{n}")

    print(f"\n... and {len(chain_stats) - 20} more combinations" if len(chain_stats) > 20 else "")

    # === Worst Chain Combinations ===
    print(f"\n{'=' * 80}")
    print("WORST 3-STEP CHAIN COMBINATIONS (by mean E2E improvement)")
    print(f"{'=' * 80}")

    print(f"\n{'Chain':<50} {'N':>5} {'Mean E2E':>10} {'Mean Adv':>10} {'Beats':>7}")
    print("-" * 87)
    for name, n, mean_e2e, mean_adv, beats in chain_stats[-10:]:
        print(f"{name:<50} {n:>5} {mean_e2e:>+9.2f}% {mean_adv:>+9.2f}pp {beats:>4}/{n}")

    # === Top Individual Chains (by absolute improvement) ===
    print(f"\n{'=' * 80}")
    print("TOP 15 INDIVIDUAL 3-STEP CHAINS (by E2E 2Q gate improvement %)")
    print(f"{'=' * 80}")

    # Filter to circuits with >= 10 original 2Q gates for meaningful percentages
    significant_chains = [c for c in chains if c["orig_2q"] >= 10]
    significant_chains.sort(key=lambda c: c["e2e_improvement"], reverse=True)

    print(f"\n{'Circuit':<30} {'Chain':<45} {'Orig':>6} {'Final':>6} {'E2E%':>8} {'vs1':>8}")
    print("-" * 107)
    for c in significant_chains[:15]:
        name = c["orig_circuit"][:28]
        chain = c["chain"][:43]
        print(
            f"{name:<30} {chain:<45} {c['orig_2q']:>6} {c['final_2q']:>6} "
            f"{c['e2e_improvement']:>+7.1f}% {c['chain_vs_single']:>+7.1f}pp"
        )

    # === Compute Cost vs Improvement Tradeoff ===
    print(f"\n{'=' * 80}")
    print("COMPUTE COST vs IMPROVEMENT TRADEOFF")
    print(f"{'=' * 80}")

    # Bucket chains by duration
    duration_buckets = [
        ("< 10s", 0, 10),
        ("10-60s", 10, 60),
        ("60-300s", 60, 300),
        ("300-1000s", 300, 1000),
        ("> 1000s", 1000, float("inf")),
    ]

    print(f"\n{'Duration':<15} {'Count':>7} {'Mean E2E':>10} {'Mean Adv':>10} {'Beats Single':>14}")
    print("-" * 60)
    for label, lo, hi in duration_buckets:
        bucket = [c for c in chains if lo <= c["total_duration"] < hi]
        if bucket:
            mean_e2e = sum(c["e2e_improvement"] for c in bucket) / len(bucket)
            mean_adv = sum(c["chain_vs_single"] for c in bucket) / len(bucket)
            beats = sum(1 for c in bucket if c["chain_vs_single"] > 0)
            print(f"{label:<15} {len(bucket):>7} {mean_e2e:>+9.2f}% {mean_adv:>+9.2f}pp {beats:>6}/{len(bucket)}")

    # === Chains that consistently beat single-step ===
    print(f"\n{'=' * 80}")
    print("CHAIN COMBINATIONS THAT CONSISTENTLY BEAT SINGLE-STEP")
    print("(>= 50% of instances beat best single optimizer, N >= 10)")
    print(f"{'=' * 80}")

    consistent_chains = [
        (name, n, mean_e2e, mean_adv, beats)
        for name, n, mean_e2e, mean_adv, beats in chain_stats
        if n >= 10 and beats / n >= 0.5
    ]
    consistent_chains.sort(key=lambda x: x[4] / x[1], reverse=True)

    if consistent_chains:
        print(f"\n{'Chain':<50} {'N':>5} {'Win%':>7} {'Mean E2E':>10} {'Mean Adv':>10}")
        print("-" * 87)
        for name, n, mean_e2e, mean_adv, beats in consistent_chains:
            print(f"{name:<50} {n:>5} {beats / n * 100:>5.1f}% {mean_e2e:>+9.2f}% {mean_adv:>+9.2f}pp")
    else:
        print("\n  No chain combinations consistently beat single-step (>50%, N>=10).")

    # === Step-by-step improvement breakdown ===
    print(f"\n{'=' * 80}")
    print("STEP-BY-STEP IMPROVEMENT BREAKDOWN")
    print(f"{'=' * 80}")

    s1_improvements = []
    s2_improvements = []
    s3_improvements = []
    for c in chains:
        if c["orig_2q"] > 0:
            s1_improvements.append((c["orig_2q"] - c["after_s1_2q"]) / c["orig_2q"] * 100.0)
        if c["after_s1_2q"] > 0:
            s2_improvements.append((c["after_s1_2q"] - c["after_s2_2q"]) / c["after_s1_2q"] * 100.0)
        if c["after_s2_2q"] > 0:
            s3_improvements.append((c["after_s2_2q"] - c["final_2q"]) / c["after_s2_2q"] * 100.0)

    for label, imps in [
        ("Step 1 (original -> opt1)", s1_improvements),
        ("Step 2 (opt1 -> opt2)", s2_improvements),
        ("Step 3 (opt2 -> opt3)", s3_improvements),
    ]:
        if imps:
            mean_imp = sum(imps) / len(imps)
            positive = sum(1 for x in imps if x > 0)
            print(f"\n  {label}:")
            print(f"    Mean improvement: {mean_imp:+.2f}%")
            print(f"    Positive: {positive}/{len(imps)} ({positive / len(imps) * 100:.1f}%)")

    step1_conn.close()
    step2_conn.close()
    step3_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze 3-step optimization chain performance"
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
        help="Path to step-3 trajectory database",
    )

    args = parser.parse_args()

    for db_path in [args.step1_db, args.step2_db, args.step3_db]:
        if not db_path.exists():
            print(f"Database not found: {db_path}")
            sys.exit(1)

    analyze(args.step1_db, args.step2_db, args.step3_db)


if __name__ == "__main__":
    main()
