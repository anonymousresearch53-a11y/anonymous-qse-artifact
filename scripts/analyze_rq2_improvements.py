"""RQ2 Analysis: Improvement from state-of-the-art optimizers over baselines.

Research Question 2: Given a circuit produced by the standard Qiskit transpiler
baseline, how much additional improvement in two-qubit gate count, depth, and
T-count is available by applying state-of-the-art optimizers such as AI-powered
transpilation, GUOQ-style rewrite+resynthesis, and RL-based synthesis/routing?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path to enable imports from benchmarks module
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.ai_transpile.statistics import (  # noqa: E402
    compare_runners,
)
from benchmarks.ai_transpile.visualization import (  # noqa: E402
    plot_improvement_bars,
    plot_runtime_vs_improvement_scatter,
    setup_matplotlib_style,
)


def load_benchmark_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file.

    Args:
        results_file: Path to latest_results.json

    Returns:
        List of result dictionaries
    """
    data = json.loads(results_file.read_text())
    return data["results"]


def compare_optimizers_vs_baseline(
    results: List[Dict[str, Any]],
    baseline_runner: str,
    optimizer_runners: List[str],
    metrics: List[str],
) -> List[Dict[str, Any]]:
    """Compare multiple optimizers against a baseline.

    Args:
        results: All benchmark results
        baseline_runner: Name of baseline runner
        optimizer_runners: List of optimizer runner names
        metrics: List of metrics to compare

    Returns:
        List of comparison dictionaries
    """
    comparisons = []

    for optimizer in optimizer_runners:
        runner_comparisons = compare_runners(
            results,
            baseline_runner=baseline_runner,
            optimized_runner=optimizer,
            metrics=metrics,
            perform_significance_test=True,
        )
        comparisons.extend(runner_comparisons)

    return comparisons


def create_improvement_summary_table(
    comparisons: List[Any],
    output_dir: Path,
    latex: bool = False,
) -> pd.DataFrame:
    """Create summary table of improvements.

    Args:
        comparisons: List of RunnerComparison objects
        output_dir: Output directory
        latex: Whether to export as LaTeX

    Returns:
        Summary DataFrame
    """
    rows = []

    for comp in comparisons:
        # Determine significance marker
        if comp.statistically_significant and comp.p_value:
            if comp.p_value < 0.001:
                significance_marker = "***"
            elif comp.p_value < 0.01:
                significance_marker = "**"
            elif comp.p_value < 0.05:
                significance_marker = "*"
            else:
                significance_marker = ""
        else:
            significance_marker = ""

        rows.append(
            {
                "Circuit": comp.circuit,
                "Metric": comp.metric,
                "Optimizer": comp.optimized_runner,
                "Baseline Mean": f"{comp.baseline_mean:.2f}",
                "Optimized Mean": f"{comp.optimized_mean:.2f}",
                "Improvement %": f"{comp.improvement_pct:.2f}{significance_marker}",
                "p-value": f"{comp.p_value:.4f}" if comp.p_value else "N/A",
            }
        )

    df = pd.DataFrame(rows)

    output_file = output_dir / ("rq2_improvements_table.tex" if latex else "rq2_improvements_table.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    if latex:
        latex_str = df.to_latex(
            index=False,
            escape=False,
            caption="RQ2: Optimizer Improvements over Baseline",
            label="tab:rq2",
        )
        output_file.write_text(latex_str)
    else:
        df.to_csv(output_file, index=False)

    print(f"Saved improvement table to {output_file}")

    return df


def create_per_metric_comparison_plots(
    comparisons: List[Any],
    output_dir: Path,
    metrics: List[str],
) -> None:
    """Create comparison plots for each metric.

    Args:
        comparisons: List of RunnerComparison objects
        output_dir: Output directory
        metrics: List of metrics to plot
    """
    for metric in metrics:
        metric_comparisons = [c for c in comparisons if c.metric == metric]

        if not metric_comparisons:
            continue

        # Create bar chart
        fig_path = output_dir / f"rq2_{metric}_improvements.pdf"
        try:
            plot_improvement_bars(
                metric_comparisons,
                metric=metric,
                title=f"RQ2: Optimizer Improvements - {metric.replace('_', ' ').title()}",
                output_path=fig_path,
            )
            print(f"Saved improvement plot to {fig_path}")
        except Exception as e:
            print(f"Warning: Could not create plot for {metric}: {e}")


def create_pareto_analysis(
    results: List[Dict[str, Any]],
    output_dir: Path,
    metric: str = "two_qubit_gates",
) -> None:
    """Create runtime vs improvement analysis showing optimization cost-benefit.

    Args:
        results: All benchmark results
        output_dir: Output directory
        metric: Metric to analyze
    """
    # Calculate baseline (best standard Qiskit result) for each circuit
    baselines: Dict[str, float] = {}
    for result in results:
        if result["optimizer"] == "qiskit_standard":
            circuit = result["circuit"]
            value = result["metrics"][metric]
            if circuit not in baselines:
                baselines[circuit] = value
            else:
                baselines[circuit] = min(baselines[circuit], value)

    # Extract data for plot
    plot_data = []

    for result in results:
        if "duration_seconds" not in result.get("metadata", {}):
            continue

        circuit = result["circuit"]
        if circuit not in baselines:
            continue

        baseline_value = baselines[circuit]
        optimized_value = result["metrics"][metric]
        improvement_pct = ((baseline_value - optimized_value) / baseline_value) * 100

        plot_data.append(
            {
                "duration_seconds": result["metadata"]["duration_seconds"],
                "improvement_pct": improvement_pct,
                "label": f"{result['runner'][:15]}",
                "circuit": circuit,
                "runner": result["runner"],
            }
        )

    if plot_data:
        fig_path = output_dir / f"rq2_runtime_vs_improvement_{metric}.pdf"
        try:
            plot_runtime_vs_improvement_scatter(
                plot_data,
                metric=metric,
                title=f"RQ2: Optimization Cost-Benefit Analysis - {metric.replace('_', ' ').title()}",
                output_path=fig_path,
            )
            print(f"Saved runtime vs improvement plot to {fig_path}")
        except Exception as e:
            print(f"Warning: Could not create runtime plot: {e}")


def print_summary_statistics(comparisons: List[Any]) -> None:
    """Print summary statistics to console.

    Args:
        comparisons: List of RunnerComparison objects
    """
    print("\n" + "=" * 80)
    print("RQ2: IMPROVEMENT ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by optimizer
    optimizers = sorted(set(c.optimized_runner for c in comparisons))

    for optimizer in optimizers:
        optimizer_comparisons = [c for c in comparisons if c.optimized_runner == optimizer]

        print(f"\nOptimizer: {optimizer}")
        print("-" * 80)

        # Calculate average improvement across all metrics
        avg_improvement = sum(c.improvement_pct for c in optimizer_comparisons) / len(optimizer_comparisons)
        print(f"  Average improvement: {avg_improvement:.2f}%")

        # Count significant improvements
        significant = sum(1 for c in optimizer_comparisons if c.statistically_significant)
        print(f"  Statistically significant: {significant}/{len(optimizer_comparisons)}")

        # Show per-metric breakdown
        metrics = sorted(set(c.metric for c in optimizer_comparisons))
        for metric in metrics:
            metric_comps = [c for c in optimizer_comparisons if c.metric == metric]
            avg_metric_imp = sum(c.improvement_pct for c in metric_comps) / len(metric_comps)
            print(f"    {metric}: {avg_metric_imp:.2f}% average improvement")


def main() -> None:
    """Main analysis function for RQ2."""
    parser = argparse.ArgumentParser(description="Analyze optimizer improvements over baseline (RQ2)")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/circuit_benchmark/full/latest_results.json"),
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/paper_figures"),
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="qiskit_standard",
        help="Baseline runner name",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["qiskit_ai", "wisq_rules_only", "wisq_bqskit", "tket_full_peephole"],
        help="Optimizer runner names to compare",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["two_qubit_gates", "depth", "two_qubit_depth"],
        help="Metrics to analyze",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Export tables as LaTeX",
    )
    args = parser.parse_args()

    # Setup matplotlib
    setup_matplotlib_style()

    # Load results
    print(f"Loading results from {args.results}...")
    results = load_benchmark_results(args.results)
    print(f"Loaded {len(results)} total results")

    # Filter to available optimizers
    available_optimizers = set(r["runner"] for r in results)
    optimizers = [opt for opt in args.optimizers if opt in available_optimizers]

    if not optimizers:
        print("Warning: None of the specified optimizers found in results.")
        print(f"Available runners: {available_optimizers}")
        print("Proceeding with available runners...")
        optimizers = list(available_optimizers - {args.baseline})

    print(f"\nComparing {len(optimizers)} optimizers against baseline '{args.baseline}':")
    for opt in optimizers:
        print(f"  - {opt}")

    # Compare optimizers
    print("\nPerforming comparisons...")
    comparisons = compare_optimizers_vs_baseline(
        results,
        baseline_runner=args.baseline,
        optimizer_runners=optimizers,
        metrics=args.metrics,
    )

    print(f"Completed {len(comparisons)} comparisons")

    if not comparisons:
        print("No comparisons available. Check that baseline and optimizers have common circuits.")
        return

    # Print summary
    print_summary_statistics(comparisons)

    # Create output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create tables
    print("\nGenerating comparison table...")
    create_improvement_summary_table(comparisons, args.output_dir, latex=args.latex)

    # Create plots
    print("\nGenerating comparison plots...")
    create_per_metric_comparison_plots(comparisons, args.output_dir, args.metrics)

    # Create Pareto analysis
    print("\nGenerating Pareto analysis...")
    create_pareto_analysis(results, args.output_dir, metric=args.metrics[0])

    print(f"\n✓ RQ2 analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

