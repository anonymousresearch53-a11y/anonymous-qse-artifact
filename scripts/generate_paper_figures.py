#!/usr/bin/env python3
"""Generate paper-ready figures from the clean confirmatory rerun database.

Figure 1: Cost vs Quality scatter plot (runtime vs 2Q gate reduction %)
Figure 2: Per-optimizer winner count bar chart

Source: data/confirmatory/full_unmapped_r1.db
"""

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "confirmatory" / "full_unmapped_r1.db"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optimizer display names and colors (consistent across both figures)
OPTIMIZER_DISPLAY = {
    "qiskit_ai": "Qiskit AI",
    "qiskit_standard": "Qiskit standard",
    "tket": "TKET",
    "wisq_rules": "WISQ rules",
    "wisq_bqskit": "WISQ+BQSKit",
}
OPTIMIZER_ORDER = ["qiskit_ai", "qiskit_standard", "tket", "wisq_rules", "wisq_bqskit"]

# Colorblind-friendly palette (Tol bright)
OPTIMIZER_COLORS = {
    "qiskit_ai": "#EE6677",
    "qiskit_standard": "#228833",
    "tket": "#4477AA",
    "wisq_rules": "#CCBB44",
    "wisq_bqskit": "#AA3377",
}

OPTIMIZER_MARKERS = {
    "qiskit_ai": "o",
    "qiskit_standard": "s",
    "tket": "^",
    "wisq_rules": "D",
    "wisq_bqskit": "v",
}


def load_runs(db_path: Path) -> list[dict]:
    """Load all successful optimization runs from the confirmatory DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            c.name AS circuit_name,
            o.name AS optimizer,
            r.input_two_qubit_gates AS input_2q,
            r.output_two_qubit_gates AS output_2q,
            r.output_depth AS output_depth,
            r.duration_seconds AS duration
        FROM optimization_runs r
        JOIN circuits c ON c.id = r.circuit_id
        JOIN optimizers o ON o.id = r.optimizer_id
        WHERE r.success = 1
        ORDER BY c.name, o.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_reduction(run: dict) -> float:
    """Compute 2Q gate reduction %: (in - out) / in * 100. Positive = improvement."""
    if run["input_2q"] == 0:
        return 0.0
    return (run["input_2q"] - run["output_2q"]) / run["input_2q"] * 100.0


def compute_winners(runs: list[dict]) -> dict[str, int]:
    """Compute per-optimizer winner counts with depth tiebreaker.

    For each circuit, the winner is the optimizer with the lowest output 2Q gates.
    If tied on 2Q gates, lowest output depth wins.
    Returns {optimizer_name: count}.
    """
    # Group by circuit
    circuits: dict[str, list[dict]] = {}
    for r in runs:
        circuits.setdefault(r["circuit_name"], []).append(r)

    counts = {opt: 0 for opt in OPTIMIZER_ORDER}
    for circuit_name, circuit_runs in sorted(circuits.items()):
        # Sort by (output_2q ASC, output_depth ASC) and pick first
        best = min(circuit_runs, key=lambda r: (r["output_2q"], r["output_depth"]))
        if best["optimizer"] in counts:
            counts[best["optimizer"]] += 1
    return counts


# ── Figure 1: Cost vs Quality scatter ──────────────────────────────────────
def make_figure1(runs: list[dict]) -> Path:
    """Create cost (runtime) vs quality (2Q reduction %) scatter plot."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Plot in Table 1 order so legend matches paper narrative
    for opt in OPTIMIZER_ORDER:
        opt_runs = [r for r in runs if r["optimizer"] == opt]
        if not opt_runs:
            continue
        xs = [r["duration"] for r in opt_runs]
        ys = [compute_reduction(r) for r in opt_runs]
        ax.scatter(
            xs, ys,
            c=OPTIMIZER_COLORS[opt],
            marker=OPTIMIZER_MARKERS[opt],
            label=OPTIMIZER_DISPLAY[opt],
            s=36,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Compilation time (seconds, log scale)", fontsize=9)
    ax.set_ylabel("Two-qubit gate reduction (%, positive = improvement)", fontsize=9)
    ax.axhline(0, color="0.55", linewidth=0.6, linestyle="--", zorder=1)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor="0.8")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))  # leave headroom so y-label doesn't clip
    out_png = OUTPUT_DIR / "fig1_cost_vs_quality.png"
    out_pdf = OUTPUT_DIR / "fig1_cost_vs_quality.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Figure 1 saved: {out_png}")
    print(f"Figure 1 saved: {out_pdf}")
    return out_png


# ── Figure 2: Winner count bar chart ──────────────────────────────────────
def make_figure2(winners: dict[str, int]) -> Path:
    """Create per-optimizer winner count bar chart."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Sort by count descending so the "tie at top" is visually clear
    sorted_opts = sorted(OPTIMIZER_ORDER, key=lambda o: winners[o], reverse=True)
    labels = [OPTIMIZER_DISPLAY[opt] for opt in sorted_opts]
    counts = [winners[opt] for opt in sorted_opts]
    colors = [OPTIMIZER_COLORS[opt] for opt in sorted_opts]
    x = np.arange(len(labels))

    bars = ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.6, width=0.6)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0.1) + 0.3,
            str(count), ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Circuits with best result", fontsize=9)
    ax.set_ylim(0, max(counts) + 1.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    out_png = OUTPUT_DIR / "fig2_winner_counts.png"
    out_pdf = OUTPUT_DIR / "fig2_winner_counts.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Figure 2 saved: {out_png}")
    print(f"Figure 2 saved: {out_pdf}")
    return out_png


# ── Verification ──────────────────────────────────────────────────────────
def verify(runs: list[dict], winners: dict[str, int]):
    """Verify computed values match the paper's Table 1."""
    # Expected "Best #" from verified paper Table 1
    expected_best = {
        "qiskit_ai": 0,
        "qiskit_standard": 7,
        "tket": 7,
        "wisq_rules": 3,
        "wisq_bqskit": 5,
    }
    # Expected run counts
    expected_runs = {
        "qiskit_ai": 21,
        "qiskit_standard": 21,
        "tket": 22,
        "wisq_rules": 10,
        "wisq_bqskit": 9,
    }
    # Expected avg 2Q reduction
    expected_avg = {
        "qiskit_ai": -8.4,
        "qiskit_standard": 1.1,
        "tket": 4.9,
        "wisq_rules": 11.0,
        "wisq_bqskit": 14.9,
    }

    print("\n── Verification ──")
    all_ok = True

    # Check winner counts
    for opt in OPTIMIZER_ORDER:
        actual = winners[opt]
        expect = expected_best[opt]
        ok = "✓" if actual == expect else "✗"
        if actual != expect:
            all_ok = False
        print(f"  Best# {OPTIMIZER_DISPLAY[opt]:20s}: expected={expect}, actual={actual}  {ok}")

    # Check run counts
    run_counts = {}
    for r in runs:
        run_counts[r["optimizer"]] = run_counts.get(r["optimizer"], 0) + 1
    for opt in OPTIMIZER_ORDER:
        actual = run_counts.get(opt, 0)
        expect = expected_runs[opt]
        ok = "✓" if actual == expect else "✗"
        if actual != expect:
            all_ok = False
        print(f"  Runs  {OPTIMIZER_DISPLAY[opt]:20s}: expected={expect}, actual={actual}  {ok}")

    # Check avg 2Q reduction
    for opt in OPTIMIZER_ORDER:
        opt_runs = [r for r in runs if r["optimizer"] == opt]
        if not opt_runs:
            continue
        avg = sum(compute_reduction(r) for r in opt_runs) / len(opt_runs)
        expect = expected_avg[opt]
        ok = "✓" if abs(avg - expect) < 0.15 else "✗"
        if abs(avg - expect) >= 0.15:
            all_ok = False
        print(f"  Avg%  {OPTIMIZER_DISPLAY[opt]:20s}: expected={expect:+.1f}%, actual={avg:+.1f}%  {ok}")

    if all_ok:
        print("\n  ✓ ALL VERIFICATION CHECKS PASSED")
    else:
        print("\n  ✗ SOME CHECKS FAILED — investigate before using figures")
    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Database: {DB_PATH}")
    print(f"Output:   {OUTPUT_DIR}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    runs = load_runs(DB_PATH)
    print(f"Loaded {len(runs)} successful runs")

    winners = compute_winners(runs)
    print(f"Winner counts: { {OPTIMIZER_DISPLAY[k]: v for k, v in winners.items()} }")

    ok = verify(runs, winners)

    fig1_path = make_figure1(runs)
    fig2_path = make_figure2(winners)

    # Print summary
    n_circuits = len(set(r["circuit_name"] for r in runs))
    print("\n── Summary ──")
    print(f"  Database:      {DB_PATH}")
    print(f"  Runs plotted:  {len(runs)} successful runs across {n_circuits} circuits")
    print("  Failed runs:   excluded (only success=1 queried)")
    print(f"  Figure 1:      {fig1_path}")
    print(f"  Figure 2:      {fig2_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
