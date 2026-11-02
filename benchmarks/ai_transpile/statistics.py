"""Statistical analysis utilities for comparing circuit optimization results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats


@dataclass
class VarianceReport:
    """Report of variance analysis across multiple optimization runs."""

    metric_name: str
    values: list[float]
    mean: float
    std: float
    variance: float
    min_value: float
    max_value: float
    range: float
    coefficient_of_variation: float
    sample_size: int

@dataclass
class ComparisonReport:
    """Report of statistical comparison between baseline and optimized results."""

    metric_name: str
    baseline_mean: float
    baseline_std: float
    optimized_mean: float
    optimized_std: float
    improvement_pct: float
    p_value: float
    statistically_significant: bool
    effect_size: float
    confidence_level: float
    ci_lower: float
    ci_upper: float
    test_method: str

@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    bootstrap_samples: int


def compute_variance_analysis(
    values: Sequence[float],
    metric_name: str = "metric",
) -> VarianceReport:
    """Compute variance analysis for a set of values.

    Args:
        values: List of metric values
        metric_name: Name of the metric being analyzed

    Returns:
        VarianceReport with statistical summaries
    """
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))  # Sample standard deviation
    variance_val = float(np.var(arr, ddof=1))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = max_val - min_val

    # Coefficient of variation (CV = std / mean)
    cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")

    return VarianceReport(
        metric_name=metric_name,
        values=list(values),
        mean=mean_val,
        std=std_val,
        variance=variance_val,
        min_value=min_val,
        max_value=max_val,
        range=range_val,
        coefficient_of_variation=cv,
        sample_size=len(values),
    )


def bootstrap_confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: int | None = None,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with mean and confidence interval
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    arr = np.array(data)
    n = len(arr)

    # Generate bootstrap samples
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(arr))

    return BootstrapResult(
        mean=mean_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence,
        n_samples=n,
        bootstrap_samples=n_bootstrap,
    )


def cohens_d(
    group1: Sequence[float],
    group2: Sequence[float],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Cohen's d is the difference between means divided by pooled standard deviation.
    |d| < 0.2 = small, 0.2-0.5 = medium, 0.5-0.8 = large, > 0.8 = very large

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    n1, n2 = len(arr1), len(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    mean_diff = np.mean(arr1) - np.mean(arr2)

    if pooled_std == 0:
        return 0.0

    return float(mean_diff / pooled_std)


def compare_optimizers_statistical(
    baseline: Sequence[float],
    optimized: Sequence[float],
    metric_name: str = "metric",
    confidence: float = 0.95,
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> ComparisonReport:
    """Compare two optimization approaches with statistical significance testing.

    Uses Mann-Whitney U test (non-parametric) for comparing distributions.
    Computes effect size (Cohen's d) and bootstrap confidence intervals.

    Args:
        baseline: Baseline metric values
        optimized: Optimized metric values
        metric_name: Name of the metric
        confidence: Confidence level for intervals
        alpha: Significance level for hypothesis testing
        lower_is_better: Whether lower values are better (e.g., gate count)

    Returns:
        ComparisonReport with statistical comparison results
    """
    baseline_arr = np.array(baseline)
    optimized_arr = np.array(optimized)

    baseline_mean = float(np.mean(baseline_arr))
    baseline_std = float(np.std(baseline_arr, ddof=1))
    optimized_mean = float(np.mean(optimized_arr))
    optimized_std = float(np.std(optimized_arr, ddof=1))

    # Compute improvement percentage
    if baseline_mean != 0:
        if lower_is_better:
            improvement_pct = 100 * (baseline_mean - optimized_mean) / baseline_mean
        else:
            improvement_pct = 100 * (optimized_mean - baseline_mean) / baseline_mean
    else:
        improvement_pct = 0.0

    # Mann-Whitney U test (non-parametric alternative to t-test)
    # Tests if distributions are different
    if lower_is_better:
        # One-sided test: optimized < baseline
        statistic, p_value = stats.mannwhitneyu(
            optimized_arr, baseline_arr, alternative="less"
        )
    else:
        # One-sided test: optimized > baseline
        statistic, p_value = stats.mannwhitneyu(
            optimized_arr, baseline_arr, alternative="greater"
        )

    p_value = float(p_value)
    significant = p_value < alpha

    # Compute effect size (Cohen's d)
    effect_size = abs(cohens_d(baseline_arr, optimized_arr))

    # Bootstrap confidence interval for the mean improvement
    combined = np.concatenate([baseline_arr, optimized_arr])
    bootstrap_result = bootstrap_confidence_interval(
        combined,
        confidence=confidence,
        n_bootstrap=5000,
    )

    return ComparisonReport(
        metric_name=metric_name,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        optimized_mean=optimized_mean,
        optimized_std=optimized_std,
        improvement_pct=improvement_pct,
        p_value=p_value,
        statistically_significant=significant,
        effect_size=effect_size,
        confidence_level=confidence,
        ci_lower=bootstrap_result.ci_lower,
        ci_upper=bootstrap_result.ci_upper,
        test_method="Mann-Whitney U",
    )


@dataclass
class MultiGroupComparison:
    """Results of comparing multiple optimization groups."""

    metric_name: str
    group_names: list[str]
    group_means: list[float]
    group_stds: list[float]
    kruskal_statistic: float
    kruskal_p_value: float
    overall_significant: bool
    pairwise_comparisons: dict[tuple[str, str], ComparisonReport] = field(default_factory=dict)


# ============================================================================
# Benchmark Results-Specific Statistics
# ============================================================================


@dataclass
class BenchmarkStatistics:
    """Statistics for a single circuit-runner-metric combination."""

    circuit: str
    runner: str
    metric: str
    mean: float
    std: float
    min_val: float
    max_val: float
    count: int
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class RunnerComparison:
    """Comparison between two runners for a specific circuit and metric."""

    circuit: str
    metric: str
    baseline_runner: str
    optimized_runner: str
    baseline_mean: float
    optimized_mean: float
    improvement_pct: float
    baseline_std: float
    optimized_std: float
    p_value: float | None = None
    statistically_significant: bool | None = None


def compute_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for the mean using t-distribution.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If values is empty
    """
    if len(values) == 0:
        raise ValueError("Cannot compute confidence interval without at least one value")

    arr = np.array(values)
    n = len(arr)
    mean_val = float(np.mean(arr))

    if n == 1:
        return mean_val, mean_val

    std_val = float(np.std(arr, ddof=1))

    if std_val == 0:
        return mean_val, mean_val

    # Use t-distribution for small samples
    se = std_val / np.sqrt(n)
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * se

    return float(mean_val - margin), float(mean_val + margin)


def compute_improvement_percentage(baseline: float, optimized: float) -> float:
    """Compute percentage improvement from baseline to optimized.

    Positive values indicate improvement (optimized < baseline for gate count).

    Args:
        baseline: Baseline metric value
        optimized: Optimized metric value

    Returns:
        Improvement percentage

    Raises:
        ValueError: If baseline is zero
    """
    if baseline == 0:
        raise ValueError("Cannot compute improvement percentage with baseline equal to zero")

    return 100.0 * (baseline - optimized) / baseline


def aggregate_runner_stats(
    results: list[dict[str, Any]],
    metrics: list[str] | None = None,
    compute_ci: bool = False,
    confidence: float = 0.95,
) -> list[BenchmarkStatistics]:
    """Aggregate statistics for each circuit-runner-metric combination.

    Args:
        results: List of benchmark result dictionaries
        metrics: List of metrics to aggregate (None = all metrics)
        compute_ci: Whether to compute confidence intervals
        confidence: Confidence level for intervals

    Returns:
        List of BenchmarkStatistics objects
    """
    if not results:
        return []

    # Group results by (circuit, runner, metric)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for result in results:
        circuit = result["circuit"]
        runner = result["runner"]
        result_metrics = result["metrics"]

        for metric_name, metric_value in result_metrics.items():
            if metrics is None or metric_name in metrics:
                grouped[(circuit, runner, metric_name)].append(float(metric_value))

    # Compute statistics for each group
    stats_list: list[BenchmarkStatistics] = []

    for (circuit, runner, metric_name), values in grouped.items():
        arr = np.array(values)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        ci_lower: float | None = None
        ci_upper: float | None = None

        if compute_ci and len(values) > 0:
            ci_lower, ci_upper = compute_confidence_interval(values, confidence)

        stats_list.append(
            BenchmarkStatistics(
                circuit=circuit,
                runner=runner,
                metric=metric_name,
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
                count=len(values),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return stats_list


def compare_runners(
    results: list[dict[str, Any]],
    baseline_runner: str,
    optimized_runner: str,
    metrics: list[str] | None = None,
    perform_significance_test: bool = False,
    alpha: float = 0.05,
) -> list[RunnerComparison]:
    """Compare two runners across all circuits and metrics.

    Args:
        results: List of benchmark result dictionaries
        baseline_runner: Name of the baseline runner
        optimized_runner: Name of the optimized runner
        metrics: List of metrics to compare (None = all metrics)
        perform_significance_test: Whether to perform statistical significance testing
        alpha: Significance level for tests

    Returns:
        List of RunnerComparison objects
    """
    # Get aggregated stats for both runners
    baseline_stats = aggregate_runner_stats(
        [r for r in results if r["runner"] == baseline_runner],
        metrics=metrics,
    )
    optimized_stats = aggregate_runner_stats(
        [r for r in results if r["runner"] == optimized_runner],
        metrics=metrics,
    )

    # Create lookup dictionaries
    baseline_dict = {(s.circuit, s.metric): s for s in baseline_stats}
    optimized_dict = {(s.circuit, s.metric): s for s in optimized_stats}

    # Find common circuit-metric combinations
    common_keys = set(baseline_dict.keys()) & set(optimized_dict.keys())

    comparisons: list[RunnerComparison] = []

    for circuit, metric_name in common_keys:
        baseline = baseline_dict[(circuit, metric_name)]
        optimized = optimized_dict[(circuit, metric_name)]

        improvement = compute_improvement_percentage(baseline.mean, optimized.mean)

        p_value: float | None = None
        significant: bool | None = None

        if perform_significance_test:
            # Get raw values for significance testing
            baseline_values = [
                r["metrics"][metric_name]
                for r in results
                if r["runner"] == baseline_runner and r["circuit"] == circuit
            ]
            optimized_values = [
                r["metrics"][metric_name]
                for r in results
                if r["runner"] == optimized_runner and r["circuit"] == circuit
            ]

            if len(baseline_values) > 1 and len(optimized_values) > 1:
                _, p_value = stats.mannwhitneyu(
                    optimized_values, baseline_values, alternative="less"
                )
                p_value = float(p_value)
                significant = p_value < alpha

        comparisons.append(
            RunnerComparison(
                circuit=circuit,
                metric=metric_name,
                baseline_runner=baseline_runner,
                optimized_runner=optimized_runner,
                baseline_mean=baseline.mean,
                optimized_mean=optimized.mean,
                improvement_pct=improvement,
                baseline_std=baseline.std,
                optimized_std=optimized.std,
                p_value=p_value,
                statistically_significant=significant,
            )
        )

    return comparisons


# ============================================================================
# Chain Experiment Statistics
# ============================================================================


@dataclass
class ChainStepStatistics:
    """Statistics for a single step in a chain."""

    step_name: str
    step_index: int
    runner_type: str
    input_two_qubit_gates: float
    output_two_qubit_gates: float
    improvement_pct: float
    duration_seconds: float


@dataclass
class ChainComparisonResult:
    """Result of comparing a chain against individual optimizers."""

    circuit: str
    chain_name: str
    chain_final_metric: float
    chain_total_duration: float
    individual_results: dict[str, float]
    chain_vs_individual: dict[str, float]
    best_individual: str
    best_individual_metric: float
    chain_improvement_over_best: float
    time_efficiency: float | None = None


def analyze_chain_results(
    chain_result_dict: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a chain result dictionary and compute statistics.

    Args:
        chain_result_dict: Dictionary from ChainResult.to_dict()

    Returns:
        Dictionary with chain analysis including per-step stats
    """
    step_stats: list[ChainStepStatistics] = []

    for sr in chain_result_dict.get("step_results", []):
        input_2q = sr["input_metrics"]["two_qubit_gates"]
        output_2q = sr["output_metrics"]["two_qubit_gates"]
        improvement = compute_improvement_percentage(input_2q, output_2q) if input_2q > 0 else 0.0

        step_stats.append(
            ChainStepStatistics(
                step_name=sr["step_name"],
                step_index=sr["step_index"],
                runner_type=sr["step_name"],  # step_name includes runner info
                input_two_qubit_gates=input_2q,
                output_two_qubit_gates=output_2q,
                improvement_pct=improvement,
                duration_seconds=sr["duration_seconds"],
            )
        )

    initial = chain_result_dict["initial_metrics"]
    final = chain_result_dict["final_metrics"]

    total_improvement = (
        compute_improvement_percentage(
            initial["two_qubit_gates"],
            final["two_qubit_gates"],
        )
        if initial["two_qubit_gates"] > 0
        else 0.0
    )

    return {
        "chain_name": chain_result_dict["chain_name"],
        "num_steps": len(step_stats),
        "step_statistics": [
            {
                "step_name": s.step_name,
                "step_index": s.step_index,
                "input_two_qubit_gates": s.input_two_qubit_gates,
                "output_two_qubit_gates": s.output_two_qubit_gates,
                "improvement_pct": s.improvement_pct,
                "duration_seconds": s.duration_seconds,
            }
            for s in step_stats
        ],
        "initial_two_qubit_gates": initial["two_qubit_gates"],
        "final_two_qubit_gates": final["two_qubit_gates"],
        "total_improvement_pct": total_improvement,
        "total_duration_seconds": chain_result_dict["total_duration_seconds"],
    }


def compare_chain_vs_individual(
    chain_results: list[dict[str, Any]],
    individual_results: list[dict[str, Any]],
    circuit_name: str,
    metric: str = "two_qubit_gates",
) -> ChainComparisonResult | None:
    """Compare chain results against individual optimizer results for a circuit.

    Args:
        chain_results: List of benchmark results from chain runners
        individual_results: List of benchmark results from individual runners
        circuit_name: Name of the circuit to compare
        metric: Metric to compare (default: two_qubit_gates)

    Returns:
        ChainComparisonResult or None if no matching results found
    """
    # Find chain result for this circuit
    chain_result = None
    for r in chain_results:
        if r["circuit"] == circuit_name and r["optimizer"] == "chain":
            chain_result = r
            break

    if chain_result is None:
        return None

    chain_metric = chain_result["metrics"][metric]
    chain_duration = chain_result["metadata"].get("total_duration_seconds", 0.0)
    chain_name = chain_result["metadata"].get("chain_name", "unknown_chain")

    # Collect individual results
    individual_metrics: dict[str, float] = {}
    for r in individual_results:
        if r["circuit"] == circuit_name and r["optimizer"] != "chain":
            runner = r["runner"]
            if runner not in individual_metrics:
                individual_metrics[runner] = r["metrics"][metric]
            else:
                # Take the best result if multiple
                individual_metrics[runner] = min(individual_metrics[runner], r["metrics"][metric])

    if not individual_metrics:
        return None

    # Calculate chain vs individual improvements
    chain_vs_individual: dict[str, float] = {}
    for runner, ind_metric in individual_metrics.items():
        if ind_metric > 0:
            improvement = 100.0 * (ind_metric - chain_metric) / ind_metric
        else:
            improvement = 0.0
        chain_vs_individual[runner] = improvement

    # Find best individual
    best_individual = min(individual_metrics.keys(), key=lambda k: individual_metrics[k])
    best_individual_metric = individual_metrics[best_individual]

    # Chain improvement over best individual
    if best_individual_metric > 0:
        chain_improvement_over_best = (
            100.0 * (best_individual_metric - chain_metric) / best_individual_metric
        )
    else:
        chain_improvement_over_best = 0.0

    return ChainComparisonResult(
        circuit=circuit_name,
        chain_name=chain_name,
        chain_final_metric=chain_metric,
        chain_total_duration=chain_duration,
        individual_results=individual_metrics,
        chain_vs_individual=chain_vs_individual,
        best_individual=best_individual,
        best_individual_metric=best_individual_metric,
        chain_improvement_over_best=chain_improvement_over_best,
    )


def compute_chain_efficiency(
    chain_duration: float,
    chain_improvement: float,
    individual_durations: dict[str, float],
    individual_improvements: dict[str, float],
) -> dict[str, Any]:
    """Compute efficiency metrics comparing chain to individual optimizers.

    Efficiency is defined as improvement_per_second.

    Args:
        chain_duration: Total chain execution time
        chain_improvement: Chain improvement percentage
        individual_durations: Duration for each individual optimizer
        individual_improvements: Improvement percentage for each individual optimizer

    Returns:
        Dictionary with efficiency comparisons
    """
    chain_efficiency = chain_improvement / chain_duration if chain_duration > 0 else 0.0

    individual_efficiencies: dict[str, float] = {}
    for runner in individual_durations:
        duration = individual_durations[runner]
        improvement = individual_improvements.get(runner, 0.0)
        if duration > 0:
            individual_efficiencies[runner] = improvement / duration
        else:
            individual_efficiencies[runner] = 0.0

    best_individual_efficiency = max(individual_efficiencies.values()) if individual_efficiencies else 0.0
    best_efficient_runner = (
        max(individual_efficiencies.keys(), key=lambda k: individual_efficiencies[k])
        if individual_efficiencies
        else None
    )

    return {
        "chain_efficiency": chain_efficiency,
        "individual_efficiencies": individual_efficiencies,
        "best_individual_efficiency": best_individual_efficiency,
        "best_efficient_runner": best_efficient_runner,
        "chain_vs_best_efficiency_ratio": (
            chain_efficiency / best_individual_efficiency
            if best_individual_efficiency > 0
            else float("inf")
        ),
    }

