"""Visualization utilities for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .statistics import BenchmarkStatistics, RunnerComparison


def setup_matplotlib_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "text.usetex": False,  # Set to True if LaTeX is available
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def plot_variance_boxplot(
    stats_list: list[BenchmarkStatistics],
    metric: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create box plot showing variance across optimization levels.

    Args:
        stats_list: List of BenchmarkStatistics objects
        metric: Metric name to plot
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    # Filter to specified metric
    filtered = [s for s in stats_list if s.metric == metric]

    if not filtered:
        raise ValueError(f"No data found for metric '{metric}'")

    # Organize data by circuit and runner
    data_dict: dict[str, list[float]] = {}

    for stat in filtered:
        label = f"{stat.circuit}\n{stat.runner}"
        # Approximate data points from statistics (for visualization)
        # In real usage, we'd want the raw values
        values = [stat.mean] * stat.count
        data_dict[label] = values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    positions = range(len(data_dict))
    bp = ax.boxplot(
        data_dict.values(),
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
    )

    # Color boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys(), rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Variance in {metric}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_variance_boxplot_raw(
    results: list[dict[str, Any]],
    metric: str,
    group_by: str = "optimization_level",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create box plot showing variance using raw result data.

    Args:
        results: List of raw benchmark result dictionaries
        metric: Metric name to plot
        group_by: Field to group results by (e.g., 'optimization_level', 'runner')
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    # Organize data by group
    data_dict: dict[str, list[float]] = {}

    for result in results:
        if metric not in result.get("metrics", {}):
            continue

        # Determine group label
        if group_by in result.get("metadata", {}):
            group_value = result["metadata"][group_by]
            label = f"Level {group_value}" if group_by == "optimization_level" else str(group_value)
        elif group_by in result:
            label = str(result[group_by])
        else:
            continue

        if label not in data_dict:
            data_dict[label] = []

        data_dict[label].append(result["metrics"][metric])

    if not data_dict:
        raise ValueError(f"No data found for metric '{metric}' grouped by '{group_by}'")

    # Sort labels for consistent ordering
    sorted_labels = sorted(data_dict.keys())
    sorted_data = [data_dict[label] for label in sorted_labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    positions = range(len(sorted_labels))
    bp = ax.boxplot(
        sorted_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
        medianprops={"color": "darkblue", "linewidth": 2},
    )

    # Color boxes with a gradient
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.3, 0.9, len(sorted_labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Variance in {metric}")
    ax.grid(True, alpha=0.3, axis="y")

    # Add sample size annotations
    for i, (label, values) in enumerate(zip(sorted_labels, sorted_data)):
        ax.text(
            i,
            ax.get_ylim()[0],
            f"n={len(values)}",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
        )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_improvement_bars(
    comparisons: list[RunnerComparison],
    metric: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create bar chart showing improvement percentages.

    Args:
        comparisons: List of RunnerComparison objects
        metric: Metric name to plot
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    # Filter to specified metric
    filtered = [c for c in comparisons if c.metric == metric]

    if not filtered:
        raise ValueError(f"No data found for metric '{metric}'")

    # Get unique circuits and optimizers
    circuits = sorted(set(c.circuit for c in filtered))
    optimizers = sorted(set(c.optimized_runner for c in filtered))

    # Create shorter optimizer names for legend
    optimizer_short_names = {
        "wisq_rules_only": "WISQ Rules",
        "wisq_bqskit": "WISQ BQSKit",
        "tket_full_peephole": "TKET",
    }

    # Assign colors to optimizers
    optimizer_colors = {
        "wisq_rules_only": "#2ecc71",  # Green
        "wisq_bqskit": "#3498db",  # Blue
        "tket_full_peephole": "#e74c3c",  # Red
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate bar positions
    bar_width = 0.25
    x_positions = np.arange(len(circuits))

    # Plot bars for each optimizer
    for i, optimizer in enumerate(optimizers):
        optimizer_data = []
        for circuit in circuits:
            matching = [c for c in filtered if c.circuit == circuit and c.optimized_runner == optimizer]
            if matching:
                optimizer_data.append(matching[0].improvement_pct)
            else:
                optimizer_data.append(0)

        # Determine bar colors based on improvement direction
        colors = []
        for imp in optimizer_data:
            base_color = optimizer_colors.get(optimizer, "#95a5a6")
            if imp < 0:
                # Make negative bars darker/muted
                colors.append(base_color)
            else:
                colors.append(base_color)

        positions = x_positions + (i - len(optimizers) / 2 + 0.5) * bar_width
        bars = ax.bar(
            positions,
            optimizer_data,
            bar_width,
            label=optimizer_short_names.get(optimizer, optimizer),
            color=colors,
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, imp in zip(bars, optimizer_data):
            if abs(imp) > 2:  # Only show labels for significant improvements
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (3 if height > 0 else -3),
                    f"{imp:.1f}%",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_xlabel("Circuit")
    ax.set_ylabel("Improvement over Baseline (%)")
    ax.set_title(title or f"Optimizer Improvements: {metric.replace('_', ' ').title()}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(circuits, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)

    # Add note about baseline
    ax.text(
        0.98,
        0.02,
        "Baseline: qiskit_ai (best of 3 AI transpiler runs)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_runtime_vs_improvement(
    data: list[dict[str, Any]],
    metric: str = "two_qubit_gates",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create scatter plot of runtime vs improvement.

    Args:
        data: List of dictionaries with 'duration_seconds' and 'improvement_pct'
        metric: Metric name for labeling
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    runtimes = [d["duration_seconds"] for d in data]
    improvements = [d["improvement_pct"] for d in data]
    labels = [d.get("label", f"Run {i}") for i, d in enumerate(data)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by improvement
    colors = ["green" if imp > 0 else "red" for imp in improvements]
    ax.scatter(runtimes, improvements, c=colors, alpha=0.6, s=100)

    # Add labels for points
    for i, (x, y, label) in enumerate(zip(runtimes, improvements, labels)):
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel(f"Improvement in {metric} (%)")
    ax.set_title(title or "Runtime vs Improvement Trade-off")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_runtime_vs_improvement_scatter(
    data: list[dict[str, Any]],
    metric: str = "two_qubit_gates",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create enhanced scatter plot showing cost-benefit of different optimizers.

    Args:
        data: List of dictionaries with 'duration_seconds', 'improvement_pct', 'runner', 'circuit'
        metric: Metric name for labeling
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    # Group by optimizer
    optimizers = sorted(set(d["runner"] for d in data))
    circuits = sorted(set(d["circuit"] for d in data))

    # Optimizer colors and markers
    optimizer_config = {
        "qiskit_standard": {"color": "#95a5a6", "marker": "s", "name": "Qiskit Standard"},
        "qiskit_ai": {"color": "#9b59b6", "marker": "D", "name": "Qiskit AI"},
        "wisq_rules_only": {"color": "#2ecc71", "marker": "o", "name": "WISQ Rules"},
        "wisq_bqskit": {"color": "#3498db", "marker": "^", "name": "WISQ BQSKit"},
        "tket_full_peephole": {"color": "#e74c3c", "marker": "v", "name": "TKET"},
    }

    # Circuit markers (different sizes)
    circuit_sizes = {circuit: 80 + i * 40 for i, circuit in enumerate(circuits)}

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each optimizer
    for optimizer in optimizers:
        optimizer_data = [d for d in data if d["runner"] == optimizer]
        if not optimizer_data:
            continue

        config = optimizer_config.get(optimizer, {"color": "#7f8c8d", "marker": "o", "name": optimizer})

        for circuit in circuits:
            circuit_data = [d for d in optimizer_data if d["circuit"] == circuit]
            if not circuit_data:
                continue

            runtimes = [d["duration_seconds"] for d in circuit_data]
            improvements = [d["improvement_pct"] for d in circuit_data]

            # Only show label for first circuit to avoid legend clutter
            label = config["name"] if circuit == circuits[0] else None

            ax.scatter(
                runtimes,
                improvements,
                c=config["color"],
                marker=config["marker"],
                s=circuit_sizes[circuit],
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=label,
            )

    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, label="No improvement")
    ax.axhline(y=20, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.4)

    # Use log scale for runtime if range is large
    if max(d["duration_seconds"] for d in data) / min(d["duration_seconds"] for d in data) > 100:
        ax.set_xscale("log")
        ax.set_xlabel("Runtime (seconds, log scale)")
    else:
        ax.set_xlabel("Runtime (seconds)")

    ax.set_ylabel("Improvement over AI Baseline (%)")
    ax.set_title(title or f"Cost-Benefit Analysis: {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3)

    # Create legend
    legend1 = ax.legend(loc="upper left", framealpha=0.9, title="Optimizer")

    # Add second legend for circuit sizes
    from matplotlib.lines import Line2D

    size_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=np.sqrt(circuit_sizes[circuit] / 10),
            label=circuit,
        )
        for circuit in circuits
    ]
    ax.legend(handles=size_legend_elements, loc="lower right", framealpha=0.9, title="Circuit", fontsize=8)
    ax.add_artist(legend1)  # Add back the first legend

    # Add annotation about the quadrants
    ax.text(
        0.02,
        0.98,
        "↖ Faster & Better\n(Ideal)\n\n↙ Faster & Worse",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    ax.text(
        0.98,
        0.98,
        "↗ Slower & Better\n(Trade-off)\n\n↘ Slower & Worse\n(Avoid)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_pareto_frontier(
    data: list[dict[str, Any]],
    x_metric: str = "duration_seconds",
    y_metric: str = "improvement_pct",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create Pareto frontier plot showing trade-offs.

    Args:
        data: List of dictionaries with metrics
        x_metric: Metric for x-axis (e.g., runtime)
        y_metric: Metric for y-axis (e.g., improvement)
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    x_values = [d[x_metric] for d in data]
    y_values = [d[y_metric] for d in data]
    labels = [d.get("label", f"Config {i}") for i, d in enumerate(data)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all points
    ax.scatter(x_values, y_values, alpha=0.6, s=100, label="Configurations")

    # Identify Pareto frontier (maximize y, minimize x)
    points = list(zip(x_values, y_values, labels))
    points.sort()  # Sort by x

    pareto_x, pareto_y = [], []
    max_y = float("-inf")

    for x, y, label in points:
        if y > max_y:
            pareto_x.append(x)
            pareto_y.append(y)
            max_y = y

    # Plot Pareto frontier
    if pareto_x:
        ax.plot(pareto_x, pareto_y, "r--", linewidth=2, alpha=0.7, label="Pareto Frontier")
        ax.scatter(pareto_x, pareto_y, c="red", s=150, marker="*", label="Pareto Optimal")

    # Label points
    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.7,
        )

    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(title or "Pareto Frontier Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_comparison_heatmap(
    comparisons: list[RunnerComparison],
    metrics: Sequence[str] | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Create heatmap of improvements across circuits and metrics.

    Args:
        comparisons: List of RunnerComparison objects
        metrics: List of metrics to include (None = all)
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    setup_matplotlib_style()

    # Build matrix of improvements
    circuits = sorted(set(c.circuit for c in comparisons))
    all_metrics = sorted(set(c.metric for c in comparisons))

    if metrics:
        all_metrics = [m for m in all_metrics if m in metrics]

    # Create data matrix
    data_matrix = np.zeros((len(circuits), len(all_metrics)))

    for i, circuit in enumerate(circuits):
        for j, metric in enumerate(all_metrics):
            matches = [c for c in comparisons if c.circuit == circuit and c.metric == metric]
            if matches:
                data_matrix[i, j] = matches[0].improvement_pct

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)

    # Set ticks
    ax.set_xticks(np.arange(len(all_metrics)))
    ax.set_yticks(np.arange(len(circuits)))
    ax.set_xticklabels([m.replace("_", " ") for m in all_metrics], rotation=45, ha="right")
    ax.set_yticklabels(circuits)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Improvement (%)")

    # Add text annotations
    for i in range(len(circuits)):
        for j in range(len(all_metrics)):
            ax.text(
                j,
                i,
                f"{data_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black" if abs(data_matrix[i, j]) < 10 else "white",
                fontsize=8,
            )

    ax.set_title(title or "Improvement Heatmap")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def create_summary_table(
    stats_list: list[BenchmarkStatistics],
    output_path: Path | None = None,
    latex: bool = False,
) -> pd.DataFrame:
    """Create summary table of benchmark statistics.

    Args:
        stats_list: List of BenchmarkStatistics objects
        output_path: Optional path to save table
        latex: Whether to export as LaTeX

    Returns:
        Pandas DataFrame with summary table
    """
    rows = []

    for stat in stats_list:
        rows.append(
            {
                "Circuit": stat.circuit,
                "Runner": stat.runner,
                "Metric": stat.metric,
                "Mean": f"{stat.mean:.2f}",
                "Std": f"{stat.std:.2f}",
                "Min": f"{stat.min_val:.0f}",
                "Max": f"{stat.max_val:.0f}",
                "Count": stat.count,
            }
        )

    df = pd.DataFrame(rows)

    if output_path:
        if latex:
            latex_str = df.to_latex(index=False)
            output_path.write_text(latex_str)
        else:
            df.to_csv(output_path, index=False)

    return df


def create_comparison_table(
    comparisons: list[RunnerComparison],
    output_path: Path | None = None,
    latex: bool = False,
) -> pd.DataFrame:
    """Create comparison table showing improvements.

    Args:
        comparisons: List of RunnerComparison objects
        output_path: Optional path to save table
        latex: Whether to export as LaTeX

    Returns:
        Pandas DataFrame with comparison table
    """
    rows = []

    for comp in comparisons:
        rows.append(
            {
                "Circuit": comp.circuit,
                "Metric": comp.metric,
                "Baseline": comp.baseline_runner,
                "Optimized": comp.optimized_runner,
                "Baseline Mean": f"{comp.baseline_mean:.2f}",
                "Optimized Mean": f"{comp.optimized_mean:.2f}",
                "Improvement %": f"{comp.improvement_pct:.2f}",
            }
        )

    df = pd.DataFrame(rows)

    if output_path:
        if latex:
            latex_str = df.to_latex(index=False)
            output_path.write_text(latex_str)
        else:
            df.to_csv(output_path, index=False)

    return df

