#!/usr/bin/env python3
"""Analyze GUOQ benchmark results: single-step vs chaining comparison.

Compares our optimizer results on the 250 GUOQ ibmnew benchmark circuits
against the GUOQ paper claims (ASPLOS 2025, arXiv 2411.04104).

Key insight: wisq_bqskit IS GUOQ (via the wisq wrapper), so its results
are directly comparable to the paper's headline numbers.

Fairness: tket, wisq_rules, wisq_bqskit produce LOGICAL (unmapped) circuits.
qiskit_ai, qiskit_standard add routing (SabreLayout -> ring topology), which
inflates 2Q gate counts. The GUOQ paper reports unmapped results, so the
fair comparison uses only the unmapped optimizers.

Usage:
    python scripts/analyze_guoq_comparison.py
    python scripts/analyze_guoq_comparison.py --step1-db data/trajectories_guoq.db
    python scripts/analyze_guoq_comparison.py --step1-db data/trajectories_guoq.db \
        --step2-db data/trajectories_guoq_step2.db
    python scripts/analyze_guoq_comparison.py --fair-only
    python scripts/analyze_guoq_comparison.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Optimizers that do NOT perform routing (fair comparison to GUOQ paper)
UNMAPPED_OPTIMIZERS = {"tket", "wisq_rules", "wisq_bqskit"}
# Optimizers that ADD routing overhead (ring topology SWAPs)
ROUTED_OPTIMIZERS = {"qiskit_ai", "qiskit_standard"}
ALL_OPTIMIZERS = UNMAPPED_OPTIMIZERS | ROUTED_OPTIMIZERS

# GUOQ paper reference: ~28% mean 2Q gate reduction on ibmnew
GUOQ_PAPER_CLAIM_PCT = 28.0


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a database in read-only mode."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _build_step1_lookup(
    conn: sqlite3.Connection,
) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    """Build circuit info and per-optimizer lookup from step-1 DB.

    Returns:
        circuit_info: {name: {init_2q, init_depth, init_total, num_qubits, category}}
        optimizer_results: {circuit_name: {optimizer_name: {output_2q, output_depth, ...}}}
    """
    rows = conn.execute(
        """
        SELECT
            c.name as circuit_name, c.category, c.num_qubits,
            c.initial_two_qubit_gates, c.initial_depth, c.initial_total_gates,
            o.name as optimizer_name,
            MIN(r.output_two_qubit_gates) as best_output_2q,
            r.output_depth, r.output_total_gates, r.duration_seconds
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 1
        GROUP BY c.id, o.id
        ORDER BY c.name, o.name
        """
    ).fetchall()

    circuit_info: dict[str, dict] = {}
    optimizer_results: dict[str, dict[str, dict]] = defaultdict(dict)

    for row in rows:
        cn = row["circuit_name"]
        if cn not in circuit_info:
            circuit_info[cn] = {
                "init_2q": row["initial_two_qubit_gates"],
                "init_depth": row["initial_depth"],
                "init_total": row["initial_total_gates"],
                "num_qubits": row["num_qubits"],
                "category": row["category"],
            }
        optimizer_results[cn][row["optimizer_name"]] = {
            "output_2q": row["best_output_2q"],
            "output_depth": row["output_depth"],
            "output_total": row["output_total_gates"],
            "duration": row["duration_seconds"],
        }

    return circuit_info, optimizer_results


def _build_step2_best_chains(
    step1_info: dict[str, dict],
    step1_results: dict[str, dict[str, dict]],
    step2_conn: sqlite3.Connection,
) -> dict[str, dict]:
    """Find best 2-step chain per original circuit.

    Returns:
        {orig_circuit_name: {final_2q, chain_desc, opt1, opt2, duration}}
    """
    rows = step2_conn.execute(
        """
        SELECT
            c.name as circuit_name, o.name as opt2,
            r.output_two_qubit_gates, r.duration_seconds
        FROM optimization_runs r
        JOIN circuits c ON r.circuit_id = c.id
        JOIN optimizers o ON r.optimizer_id = o.id
        WHERE r.success = 1
        ORDER BY c.name, o.name
        """
    ).fetchall()

    best_chains: dict[str, dict] = {}

    for row in rows:
        # Parse artifact name: artifact_{orig}__{opt1}
        m = re.match(r"^artifact_(.+?)__(\w+)$", row["circuit_name"])
        if not m:
            continue
        orig = m.group(1)
        opt1 = m.group(2)

        if orig not in step1_info:
            continue

        final_2q = row["output_two_qubit_gates"]
        opt2 = row["opt2"]
        chain_desc = f"{opt1} -> {opt2}"

        # Compute total duration (step1 + step2)
        s1_dur = step1_results.get(orig, {}).get(opt1, {}).get("duration", 0) or 0
        s2_dur = row["duration_seconds"] or 0
        total_dur = s1_dur + s2_dur

        if orig not in best_chains or final_2q < best_chains[orig]["final_2q"]:
            best_chains[orig] = {
                "final_2q": final_2q,
                "chain_desc": chain_desc,
                "opt1": opt1,
                "opt2": opt2,
                "duration": total_dur,
            }

    return best_chains


def _pct(before: int, after: int) -> float:
    """Compute reduction percentage."""
    return (before - after) / before * 100.0 if before > 0 else 0.0


def _print_header(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")


def analyze(
    step1_db_path: Path,
    step2_db_path: Path | None,
    fair_only: bool,
    csv_path: Path | None,
    original_db_path: Path | None,
) -> None:
    """Run the full GUOQ benchmark comparison analysis."""
    step1_conn = _open_readonly(step1_db_path)
    step2_conn = _open_readonly(step2_db_path) if step2_db_path else None
    original_conn = _open_readonly(original_db_path) if original_db_path else None

    # Get total circuit count
    total_circuits = step1_conn.execute("SELECT COUNT(*) FROM circuits").fetchone()[0]
    total_runs = step1_conn.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()[0]
    successful_runs = step1_conn.execute(
        "SELECT COUNT(*) FROM optimization_runs WHERE success = 1"
    ).fetchone()[0]

    print("GUOQ Benchmark Comparison Analysis")
    print(f"Database: {step1_db_path}")
    print(f"Circuits: {total_circuits}")
    print(f"Optimization runs: {successful_runs}/{total_runs} successful ({100 * successful_runs / total_runs:.1f}%)")

    # Build lookups
    circuit_info, optimizer_results = _build_step1_lookup(step1_conn)
    print(f"Circuits with results: {len(circuit_info)}")

    # Build chain lookups if step2 available
    best_chains = None
    if step2_conn:
        best_chains = _build_step2_best_chains(circuit_info, optimizer_results, step2_conn)
        print(f"Circuits with chain results: {len(best_chains)}")

    # Determine which optimizers to show
    show_optimizers = UNMAPPED_OPTIMIZERS if fair_only else ALL_OPTIMIZERS
    fair_label = " (FAIR: unmapped only)" if fair_only else ""

    # =========================================================================
    # TABLE 1: Fair Single-Step Comparison
    # =========================================================================
    _print_header(f"TABLE 1: Single-Step Optimizer Results{fair_label}")

    # Per-optimizer aggregate stats
    opt_reductions: dict[str, list[float]] = defaultdict(list)
    opt_counts: dict[str, int] = defaultdict(int)

    for cn, info in sorted(circuit_info.items()):
        init_2q = info["init_2q"]
        if init_2q <= 0:
            continue
        for opt_name, opt_data in optimizer_results[cn].items():
            if opt_name not in show_optimizers:
                continue
            red = _pct(init_2q, opt_data["output_2q"])
            opt_reductions[opt_name].append(red)
            opt_counts[opt_name] += 1

    print(f"\n{'Optimizer':<20} {'N':>5} {'Mean Red%':>10} {'Median Red%':>12} {'>0%':>6} {'<0%':>6}")
    print("-" * 65)
    for opt_name in sorted(show_optimizers):
        reds = opt_reductions.get(opt_name, [])
        if not reds:
            continue
        mean_red = sum(reds) / len(reds)
        sorted_reds = sorted(reds)
        median_red = sorted_reds[len(reds) // 2]
        positive = sum(1 for r in reds if r > 0)
        negative = sum(1 for r in reds if r < 0)
        routed = " *" if opt_name in ROUTED_OPTIMIZERS else ""
        print(
            f"{opt_name + routed:<20} {len(reds):>5} {mean_red:>+9.1f}% {median_red:>+11.1f}% "
            f"{positive:>5} {negative:>5}"
        )

    if not fair_only:
        print("\n  * = includes routing overhead (SabreLayout -> ring topology)")

    # =========================================================================
    # TABLE 2: Verification vs GUOQ Paper Claims
    # =========================================================================
    _print_header("TABLE 2: Verification vs GUOQ Paper (~28% mean 2Q reduction)")

    wisq_reds = opt_reductions.get("wisq_bqskit", [])
    if wisq_reds:
        mean_wisq = sum(wisq_reds) / len(wisq_reds)
        median_wisq = sorted(wisq_reds)[len(wisq_reds) // 2]
        print("\n  Our wisq_bqskit (= GUOQ + BQSKit resynthesis):")
        print(f"    Circuits tested:  {len(wisq_reds)}")
        print(f"    Mean reduction:   {mean_wisq:+.1f}%")
        print(f"    Median reduction: {median_wisq:+.1f}%")
        print(f"    GUOQ paper claim: ~{GUOQ_PAPER_CLAIM_PCT:+.1f}%")
        delta = mean_wisq - GUOQ_PAPER_CLAIM_PCT
        print(f"    Delta:            {delta:+.1f}pp")
        if abs(delta) > 10:
            print("    WARNING: Large discrepancy. Check timeout settings (paper uses 3600s, we use 1800s).")
        elif abs(delta) < 5:
            print("    GOOD: Results are consistent with paper claims.")
    else:
        print("\n  No wisq_bqskit results available.")

    # =========================================================================
    # TABLE 3: Best Single-Step per Circuit
    # =========================================================================
    _print_header("TABLE 3: Best Single-Step Results (per circuit)")

    # Compute best single optimizer per circuit
    best_single: dict[str, dict] = {}
    for cn, info in circuit_info.items():
        init_2q = info["init_2q"]
        if init_2q <= 0:
            continue
        best_2q = init_2q
        best_opt = "none"
        best_2q_unmapped = init_2q
        best_opt_unmapped = "none"
        for opt_name, opt_data in optimizer_results[cn].items():
            if opt_data["output_2q"] < best_2q:
                best_2q = opt_data["output_2q"]
                best_opt = opt_name
            if opt_name in UNMAPPED_OPTIMIZERS and opt_data["output_2q"] < best_2q_unmapped:
                best_2q_unmapped = opt_data["output_2q"]
                best_opt_unmapped = opt_name
        best_single[cn] = {
            "best_2q": best_2q,
            "best_opt": best_opt,
            "best_2q_unmapped": best_2q_unmapped,
            "best_opt_unmapped": best_opt_unmapped,
        }

    # Summary stats
    unmapped_reds = [_pct(circuit_info[cn]["init_2q"], bs["best_2q_unmapped"])
                     for cn, bs in best_single.items()]
    all_reds = [_pct(circuit_info[cn]["init_2q"], bs["best_2q"])
                for cn, bs in best_single.items()]

    print("\n  Best unmapped optimizer (per circuit):")
    print(f"    Mean reduction:   {sum(unmapped_reds) / len(unmapped_reds):+.1f}%")
    print(f"    Median reduction: {sorted(unmapped_reds)[len(unmapped_reds) // 2]:+.1f}%")

    if not fair_only:
        print("  Best any optimizer (incl. routed):")
        print(f"    Mean reduction:   {sum(all_reds) / len(all_reds):+.1f}%")
        print(f"    Median reduction: {sorted(all_reds)[len(all_reds) // 2]:+.1f}%")

    # Which optimizer wins most often?
    unmapped_wins: dict[str, int] = defaultdict(int)
    all_wins: dict[str, int] = defaultdict(int)
    for cn, bs in best_single.items():
        unmapped_wins[bs["best_opt_unmapped"]] += 1
        all_wins[bs["best_opt"]] += 1

    print("\n  Unmapped optimizer win counts:")
    for opt, count in sorted(unmapped_wins.items(), key=lambda x: -x[1]):
        print(f"    {opt:<20} {count:>5} ({100 * count / len(best_single):.1f}%)")

    # =========================================================================
    # TABLE 4: 2-Step Chaining vs Best Single (if step2 available)
    # =========================================================================
    if best_chains:
        _print_header("TABLE 4: 2-Step Chaining vs Best Single-Step")

        chain_reds = []
        chain_vs_single = []
        chain_vs_wisq = []
        chain_wins = 0
        chain_ties = 0
        chain_loses = 0

        for cn in sorted(best_chains.keys()):
            if cn not in best_single:
                continue
            init_2q = circuit_info[cn]["init_2q"]
            if init_2q <= 0:
                continue

            chain_2q = best_chains[cn]["final_2q"]
            single_2q = best_single[cn]["best_2q_unmapped"]
            wisq_2q = optimizer_results[cn].get("wisq_bqskit", {}).get("output_2q", init_2q)

            chain_red = _pct(init_2q, chain_2q)
            single_red = _pct(init_2q, single_2q)
            wisq_red = _pct(init_2q, wisq_2q)

            chain_reds.append(chain_red)
            chain_vs_single.append(chain_red - single_red)
            chain_vs_wisq.append(chain_red - wisq_red)

            if chain_2q < single_2q:
                chain_wins += 1
            elif chain_2q == single_2q:
                chain_ties += 1
            else:
                chain_loses += 1

        total = chain_wins + chain_ties + chain_loses
        print(f"\n  Circuits analyzed: {total}")
        print(f"  Chain WINS over best unmapped single: {chain_wins}/{total} ({100 * chain_wins / total:.1f}%)")
        print(f"  Ties:                                 {chain_ties}/{total} ({100 * chain_ties / total:.1f}%)")
        print(f"  Single WINS:                          {chain_loses}/{total} ({100 * chain_loses / total:.1f}%)")

        print("\n  Chain reduction:")
        print(f"    Mean:   {sum(chain_reds) / len(chain_reds):+.1f}%")
        print(f"    Median: {sorted(chain_reds)[len(chain_reds) // 2]:+.1f}%")

        print("  Chain advantage over best unmapped single:")
        print(f"    Mean:   {sum(chain_vs_single) / len(chain_vs_single):+.1f}pp")
        print(f"    Median: {sorted(chain_vs_single)[len(chain_vs_single) // 2]:+.1f}pp")

        print("  Chain advantage over wisq_bqskit (= GUOQ):")
        print(f"    Mean:   {sum(chain_vs_wisq) / len(chain_vs_wisq):+.1f}pp")
        print(f"    Median: {sorted(chain_vs_wisq)[len(chain_vs_wisq) // 2]:+.1f}pp")

        # Top chain patterns
        chain_pattern_count: dict[str, int] = defaultdict(int)
        chain_pattern_wins: dict[str, int] = defaultdict(int)
        for cn, chain_data in best_chains.items():
            if cn not in best_single:
                continue
            chain_pattern_count[chain_data["chain_desc"]] += 1
            single_2q = best_single[cn]["best_2q_unmapped"]
            if chain_data["final_2q"] < single_2q:
                chain_pattern_wins[chain_data["chain_desc"]] += 1

        print("\n  Most common winning chain patterns:")
        print(f"    {'Chain':<40} {'Best For':>8} {'Wins':>6}")
        print(f"    {'-' * 58}")
        for pattern, count in sorted(chain_pattern_count.items(), key=lambda x: -x[1])[:10]:
            wins = chain_pattern_wins.get(pattern, 0)
            print(f"    {pattern:<40} {count:>8} {wins:>6}")

    # =========================================================================
    # TABLE 5: Results by Circuit Size Bucket
    # =========================================================================
    _print_header("TABLE 5: Results by Circuit Size (2Q gate count)")

    size_buckets = [
        ("0-50", 0, 51),
        ("51-200", 51, 201),
        ("201-1K", 201, 1001),
        ("1K-5K", 1001, 5001),
        ("5K-50K", 5001, 50001),
        ("50K+", 50001, float("inf")),
    ]

    header = f"{'Bucket':<10} {'N':>5} {'Best Unmapped':>14} {'wisq_bqskit':>12} {'tket':>8}"
    if best_chains:
        header += f" {'Best Chain':>11} {'Chain Adv':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for label, lo, hi in size_buckets:
        bucket_circuits = [
            cn for cn, info in circuit_info.items()
            if lo <= info["init_2q"] < hi and info["init_2q"] > 0
        ]
        if not bucket_circuits:
            continue

        n = len(bucket_circuits)
        unmapped_reds_b = [_pct(circuit_info[cn]["init_2q"], best_single[cn]["best_2q_unmapped"])
                           for cn in bucket_circuits if cn in best_single]
        wisq_reds_b = [_pct(circuit_info[cn]["init_2q"],
                            optimizer_results[cn].get("wisq_bqskit", {}).get("output_2q", circuit_info[cn]["init_2q"]))
                       for cn in bucket_circuits]
        tket_reds_b = [_pct(circuit_info[cn]["init_2q"],
                            optimizer_results[cn].get("tket", {}).get("output_2q", circuit_info[cn]["init_2q"]))
                       for cn in bucket_circuits]

        row = (
            f"{label:<10} {n:>5} "
            f"{sum(unmapped_reds_b) / len(unmapped_reds_b):>+13.1f}% "
            f"{sum(wisq_reds_b) / len(wisq_reds_b):>+11.1f}% "
            f"{sum(tket_reds_b) / len(tket_reds_b):>+7.1f}%"
        )

        if best_chains:
            chain_reds_b = []
            chain_adv_b = []
            for cn in bucket_circuits:
                if cn in best_chains and cn in best_single:
                    init = circuit_info[cn]["init_2q"]
                    cr = _pct(init, best_chains[cn]["final_2q"])
                    sr = _pct(init, best_single[cn]["best_2q_unmapped"])
                    chain_reds_b.append(cr)
                    chain_adv_b.append(cr - sr)
            if chain_reds_b:
                row += (
                    f" {sum(chain_reds_b) / len(chain_reds_b):>+10.1f}%"
                    f" {sum(chain_adv_b) / len(chain_adv_b):>+9.1f}pp"
                )

        print(row)

    # =========================================================================
    # TABLE 6: Overlapping Circuits (if original DB provided)
    # =========================================================================
    if original_conn:
        _print_header("TABLE 6: Overlapping Circuits (present in both suites)")

        orig_info, orig_results = _build_step1_lookup(original_conn)
        overlap = set(circuit_info.keys()) & set(orig_info.keys())

        if overlap:
            print(f"\n  Overlapping circuits: {len(overlap)}")
            print(f"\n  {'Circuit':<25} {'Init(us)':>8} {'Init(guoq)':>10} "
                  f"{'Best(us)':>8} {'Best(guoq)':>10} {'Consistent?'}")
            print(f"  {'-' * 75}")

            for cn in sorted(overlap):
                init_ours = orig_info[cn]["init_2q"]
                init_guoq = circuit_info[cn]["init_2q"]
                best_ours = min((d["output_2q"] for d in orig_results[cn].values()), default=init_ours)
                best_guoq = min(
                    (d["output_2q"] for opt, d in optimizer_results[cn].items()
                     if opt in UNMAPPED_OPTIMIZERS),
                    default=init_guoq,
                )
                consistent = "YES" if best_ours == best_guoq or abs(best_ours - best_guoq) <= 2 else "DIFFERS"
                print(
                    f"  {cn:<25} {init_ours:>8} {init_guoq:>10} "
                    f"{best_ours:>8} {best_guoq:>10} {consistent}"
                )
        else:
            print("\n  No overlapping circuits found.")

        original_conn.close()

    # =========================================================================
    # TABLE 7: Detailed Per-Circuit Results (top/bottom)
    # =========================================================================
    _print_header("TABLE 7: Top 20 Circuits by Chain Improvement over Single-Step")

    if best_chains:
        detailed = []
        for cn in best_chains:
            if cn not in best_single or circuit_info[cn]["init_2q"] <= 5:
                continue
            init = circuit_info[cn]["init_2q"]
            single = best_single[cn]["best_2q_unmapped"]
            chain = best_chains[cn]["final_2q"]
            detailed.append({
                "circuit": cn,
                "qubits": circuit_info[cn]["num_qubits"],
                "init": init,
                "single": single,
                "single_red": _pct(init, single),
                "chain": chain,
                "chain_red": _pct(init, chain),
                "chain_desc": best_chains[cn]["chain_desc"],
                "advantage": _pct(init, chain) - _pct(init, single),
            })

        detailed.sort(key=lambda x: x["advantage"], reverse=True)

        print(f"\n{'Circuit':<25} {'Q':>3} {'Init':>6} {'Single':>7} {'Red%':>6} "
              f"{'Chain':>6} {'Red%':>6} {'Adv':>6} {'Chain Pattern':<30}")
        print("-" * 105)
        for d in detailed[:20]:
            print(
                f"{d['circuit'][:24]:<25} {d['qubits']:>3} {d['init']:>6} "
                f"{d['single']:>7} {d['single_red']:>+5.1f}% "
                f"{d['chain']:>6} {d['chain_red']:>+5.1f}% "
                f"{d['advantage']:>+5.1f}pp {d['chain_desc']:<30}"
            )

        # Also show bottom 10 (where chaining hurts)
        print("\n  Bottom 10 (where chaining hurts or doesn't help):")
        print(f"  {'Circuit':<25} {'Init':>6} {'Single':>7} {'Chain':>6} {'Adv':>6}")
        print(f"  {'-' * 55}")
        for d in detailed[-10:]:
            print(
                f"  {d['circuit'][:24]:<25} {d['init']:>6} "
                f"{d['single']:>7} {d['chain']:>6} {d['advantage']:>+5.1f}pp"
            )
    else:
        print("\n  No step-2 data available. Run step-2 grid search first.")

    # =========================================================================
    # CSV Export
    # =========================================================================
    if csv_path:
        _export_csv(csv_path, circuit_info, optimizer_results, best_single, best_chains)

    # Cleanup
    step1_conn.close()
    if step2_conn:
        step2_conn.close()


def _export_csv(
    csv_path: Path,
    circuit_info: dict[str, dict],
    optimizer_results: dict[str, dict[str, dict]],
    best_single: dict[str, dict],
    best_chains: dict[str, dict] | None,
) -> None:
    """Export all results to CSV."""
    fieldnames = [
        "circuit", "category", "num_qubits", "init_2q",
        "tket_2q", "wisq_rules_2q", "wisq_bqskit_2q",
        "qiskit_ai_2q", "qiskit_standard_2q",
        "best_unmapped_2q", "best_unmapped_opt",
        "best_any_2q", "best_any_opt",
    ]
    if best_chains:
        fieldnames.extend(["best_chain_2q", "chain_pattern", "chain_advantage_pp"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cn in sorted(circuit_info.keys()):
            info = circuit_info[cn]
            bs = best_single.get(cn, {})
            row = {
                "circuit": cn,
                "category": info["category"],
                "num_qubits": info["num_qubits"],
                "init_2q": info["init_2q"],
                "tket_2q": optimizer_results[cn].get("tket", {}).get("output_2q", ""),
                "wisq_rules_2q": optimizer_results[cn].get("wisq_rules", {}).get("output_2q", ""),
                "wisq_bqskit_2q": optimizer_results[cn].get("wisq_bqskit", {}).get("output_2q", ""),
                "qiskit_ai_2q": optimizer_results[cn].get("qiskit_ai", {}).get("output_2q", ""),
                "qiskit_standard_2q": optimizer_results[cn].get("qiskit_standard", {}).get("output_2q", ""),
                "best_unmapped_2q": bs.get("best_2q_unmapped", ""),
                "best_unmapped_opt": bs.get("best_opt_unmapped", ""),
                "best_any_2q": bs.get("best_2q", ""),
                "best_any_opt": bs.get("best_opt", ""),
            }
            if best_chains:
                bc = best_chains.get(cn, {})
                init = info["init_2q"]
                if bc and cn in bs:
                    chain_red = _pct(init, bc["final_2q"])
                    single_red = _pct(init, bs.get("best_2q_unmapped", init))
                    row["best_chain_2q"] = bc["final_2q"]
                    row["chain_pattern"] = bc["chain_desc"]
                    row["chain_advantage_pp"] = f"{chain_red - single_red:+.1f}"
                else:
                    row["best_chain_2q"] = ""
                    row["chain_pattern"] = ""
                    row["chain_advantage_pp"] = ""
            writer.writerow(row)

    print(f"\nResults exported to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GUOQ benchmark: single-step vs chaining comparison"
    )
    parser.add_argument(
        "--step1-db",
        type=Path,
        default=Path("data/trajectories_guoq.db"),
        help="Path to GUOQ step-1 trajectory database",
    )
    parser.add_argument(
        "--step2-db",
        type=Path,
        default=None,
        help="Path to GUOQ step-2 trajectory database (optional)",
    )
    parser.add_argument(
        "--original-db",
        type=Path,
        default=None,
        help="Path to original trajectory DB for overlap comparison (optional)",
    )
    parser.add_argument(
        "--fair-only",
        action="store_true",
        help="Show only unmapped optimizers (fair comparison to GUOQ paper)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export results to CSV file",
    )

    args = parser.parse_args()

    if not args.step1_db.exists():
        print(f"Step-1 database not found: {args.step1_db}")
        print("Run step-1 grid search first:")
        print("  ./scripts/run_guoq_step1_tmux.sh create")
        sys.exit(1)

    step2_path = args.step2_db
    if step2_path and not step2_path.exists():
        print(f"Step-2 database not found: {step2_path}")
        print("Run step-2 grid search first:")
        print("  ./scripts/run_guoq_step2_tmux.sh create")
        print("Continuing without step-2 data...")
        step2_path = None

    # Auto-detect step2 DB if not specified
    if step2_path is None:
        default_step2 = Path("data/trajectories_guoq_step2.db")
        if default_step2.exists():
            step2_path = default_step2
            print(f"Auto-detected step-2 database: {step2_path}")

    # Auto-detect original DB for overlap
    original_path = args.original_db
    if original_path is None:
        default_orig = Path("data/trajectories.db")
        if default_orig.exists():
            original_path = default_orig

    analyze(args.step1_db, step2_path, args.fair_only, args.csv, original_path)


if __name__ == "__main__":
    main()
