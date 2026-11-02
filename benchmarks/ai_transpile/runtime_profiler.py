"""Runtime profiling utilities for tracking optimization time and performance."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RuntimeStatistics:
    """Statistics for operation runtimes."""

    operation: str
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    total_seconds: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operation": self.operation,
            "mean_seconds": self.mean_seconds,
            "std_seconds": self.std_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "total_seconds": self.total_seconds,
            "count": self.count,
        }


class Timer:
    """Context manager for measuring execution time."""

    def __init__(self) -> None:
        """Initialize the timer."""
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer and record elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def __enter__(self) -> Timer:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.stop()


class RuntimeProfiler:
    """Profiler for tracking operation runtimes."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.timings: dict[str, list[float]] = defaultdict(list)

    def record(self, operation: str, duration: float) -> None:
        """Record a timing measurement.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        self.timings[operation].append(duration)

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        """Context manager to measure an operation.

        Args:
            operation: Name of the operation

        Yields:
            None
        """
        timer = Timer()
        timer.start()
        try:
            yield
        finally:
            timer.stop()
            self.record(operation, timer.elapsed)

    def get_statistics(self) -> list[RuntimeStatistics]:
        """Compute statistics for all recorded operations.

        Returns:
            List of RuntimeStatistics objects
        """
        stats: list[RuntimeStatistics] = []

        for operation, durations in self.timings.items():
            if not durations:
                continue

            arr = np.array(durations)
            stats.append(
                RuntimeStatistics(
                    operation=operation,
                    mean_seconds=float(np.mean(arr)),
                    std_seconds=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    min_seconds=float(np.min(arr)),
                    max_seconds=float(np.max(arr)),
                    total_seconds=float(np.sum(arr)),
                    count=len(durations),
                )
            )

        return stats

    def export_json(self, output_path: Path) -> None:
        """Export profiler data to JSON.

        Args:
            output_path: Path to output JSON file
        """
        stats = self.get_statistics()
        data = {"statistics": [s.to_dict() for s in stats]}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))

    def clear(self) -> None:
        """Clear all recorded timings."""
        self.timings.clear()


def profile_function(func: F) -> F:
    """Decorator to profile a function's execution time.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that records timing
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timer = Timer()
        timer.start()
        try:
            result = func(*args, **kwargs)
        finally:
            timer.stop()
            # Note: In a real application, you'd pass a profiler instance
            # For testing, we just measure without recording
        return result

    return wrapper  # type: ignore[return-value]


def aggregate_timing_data(
    timing_dicts: list[dict[str, list[float]]],
) -> dict[str, list[float]]:
    """Aggregate timing data from multiple profiler runs.

    Args:
        timing_dicts: List of timing dictionaries from different runs

    Returns:
        Aggregated timing dictionary
    """
    aggregated: dict[str, list[float]] = defaultdict(list)

    for timing_dict in timing_dicts:
        for operation, durations in timing_dict.items():
            aggregated[operation].extend(durations)

    return dict(aggregated)


def compute_cost_benefit_ratio(
    improvement_pct: float,
    duration_seconds: float,
) -> float:
    """Compute cost-benefit ratio for an optimization.

    The ratio is improvement percentage per second of runtime.
    Higher values indicate better cost-benefit.

    Args:
        improvement_pct: Percentage improvement in metric
        duration_seconds: Time taken in seconds

    Returns:
        Cost-benefit ratio (improvement per second)
    """
    if duration_seconds == 0:
        return float("inf") if improvement_pct > 0 else 0.0

    return improvement_pct / duration_seconds

