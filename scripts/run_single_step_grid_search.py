#!/usr/bin/env python3
"""CLI script for running single-step grid search over quantum circuit optimizers.

This script runs exhaustive single-step optimization across circuits and optimizers,
with async execution and proper concurrency control for resource-intensive optimizers.

Usage:
    # Initialize database and run full search
    uv run python scripts/run_single_step_grid_search.py --resume

    # Run specific categories
    uv run python scripts/run_single_step_grid_search.py --categories qft qaoa --resume

    # Run specific optimizers only
    uv run python scripts/run_single_step_grid_search.py --optimizers tket qiskit_standard

    # Run the same optimizers with hardware-aware implementations
    uv run python scripts/run_single_step_grid_search.py --hardware-aware yes --optimizers tket qiskit_standard

    # Set custom WISQ+BQSKit timeout (default 5 min)
    uv run python scripts/run_single_step_grid_search.py --wisq-bqskit-timeout 180

    # Force rerun all optimizers (keeps history in database)
    uv run python scripts/run_single_step_grid_search.py --resume --rerun

    # Force rerun specific optimizers only (keeps history)
    uv run python scripts/run_single_step_grid_search.py --resume --rerun-optimizers wisq_rules wisq_bqskit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# WISQ/GUOQ requires Java 21 (class file version 65).
# Set JAVA_HOME if not already configured and java-21 is available.
_java21_home = "/usr/lib/jvm/java-21-openjdk-amd64"
if "JAVA_HOME" not in os.environ and os.path.isdir(_java21_home):
    os.environ["JAVA_HOME"] = _java21_home
    os.environ["PATH"] = f"{_java21_home}/bin{os.pathsep}{os.environ.get('PATH', '')}"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory import (  # noqa: E402
    ALL_OPTIMIZER_CONFIGS,
    OPTIMIZER_CONFIGS,
    OptimizerRecord,
    SingleStepConfig,
    TrajectoryDatabase,
    import_from_artifacts_dir,
    import_from_metadata_json,
)
from benchmarks.ai_transpile.rl_trajectory.single_step_search import (  # noqa: E402
    AsyncSingleStepRunner,
    OptimizersProgressTracker,
)


HARDWARE_AWARE_OPTIMIZER_NAMES = {
    "wisq_rules": "wisq_rules_hardware",
    "wisq_bqskit": "wisq_bqskit_hardware",
    "tket": "tket_hardware",
    "qiskit_ai": "qiskit_ai_hardware",
    "qiskit_standard": "qiskit_standard_hardware",
}


def resolve_optimizer_names(
    requested_names: list[str] | None,
    hardware_aware: bool,
) -> list[str]:
    """Resolve optimizer names to the selected implementation family."""
    selected_names = requested_names or list(OPTIMIZER_CONFIGS.keys())

    if hardware_aware:
        return [HARDWARE_AWARE_OPTIMIZER_NAMES.get(name, name) for name in selected_names]

    hardware_names = set(HARDWARE_AWARE_OPTIMIZER_NAMES.values())
    invalid_names = [name for name in selected_names if name in hardware_names]
    if invalid_names:
        raise ValueError(
            "Hardware-aware optimizer names require --hardware-aware yes: "
            + ", ".join(invalid_names)
        )

    return selected_names


def register_optimizers(
    db: TrajectoryDatabase,
    optimizer_names: list[str],
    wisq_bqskit_timeout: int = 300,
) -> int:
    """Register the selected optimizers in the database.

    Args:
        db: TrajectoryDatabase instance
        optimizer_names: Optimizer configurations to register
        wisq_bqskit_timeout: Timeout in seconds for WISQ+BQSKit optimizer
    Returns:
        Number of optimizers registered
    """
    registered = 0

    for name in optimizer_names:
        config = ALL_OPTIMIZER_CONFIGS[name]
        options = dict(config["options"])
        if name in {"wisq_bqskit", "wisq_bqskit_hardware"}:
            options["opt_timeout"] = wisq_bqskit_timeout
        optimizer = OptimizerRecord(
            id=None,
            name=name,
            runner_type=config["runner_type"],
            options=options,
            description=config.get("description"),
        )

        existing = db.get_optimizer_by_name(name)
        db.get_or_create_optimizer(optimizer)
        if existing is None:
            registered += 1

    return registered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-step grid search over quantum circuit optimizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --resume                            # Run full search, resuming from existing progress
  %(prog)s --categories qft qaoa --resume      # Run on specific categories only
  %(prog)s --optimizers tket qiskit_standard   # Run specific optimizers only
  %(prog)s --hardware-aware yes --optimizers qiskit_standard tket
  %(prog)s --wisq-bqskit-timeout 180           # Set 3 minute timeout for WISQ+BQSKit
  %(prog)s --no-artifacts                      # Don't save output circuit files
        """,
    )

    # Database and import options
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to trajectory database (default: data/trajectories.db)",
    )
    parser.add_argument(
        "--import-metadata",
        type=Path,
        default=None,
        help="Import circuits from metadata.json file before running",
    )
    parser.add_argument(
        "--import-artifacts",
        type=Path,
        default=None,
        help="Import circuits from optimized artifact outputs (step-1 QASM files)",
    )
    parser.add_argument(
        "--step1-database",
        type=Path,
        default=None,
        help="Path to step-1 trajectory DB for category lookup during artifact import",
    )

    # Logging
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to a clean plain-text log file (no ANSI escape sequences)",
    )

    # Circuit filters
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Circuit categories to include (e.g., qft qaoa clifford)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Circuit sources to include (e.g., guoq, artifact, benchpress)",
    )
    parser.add_argument(
        "--exclude-name-like",
        nargs="+",
        default=None,
        help="Exclude circuits whose names match these patterns (e.g., 'artifact_%%')",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=20,
        help="Maximum number of qubits (default: 20)",
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        choices=list(ALL_OPTIMIZER_CONFIGS.keys()),
        help="Optimizers to use (default: all existing logical optimizers)",
    )
    parser.add_argument(
        "--hardware-aware",
        choices=("yes", "no"),
        default="no",
        help="Use full Fake Torino hardware-aware optimizers (default: no)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun all optimizers even if results exist (keeps history)",
    )
    parser.add_argument(
        "--rerun-optimizers",
        nargs="+",
        default=None,
        choices=list(ALL_OPTIMIZER_CONFIGS.keys()),
        help="Force rerun specific optimizers even if results exist (keeps history)",
    )
    parser.add_argument(
        "--wisq-bqskit-timeout",
        type=int,
        default=120,
        help="Timeout for WISQ+BQSKit in seconds (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--optimizer-timeout",
        type=int,
        default=900,
        help=(
            "Hard timeout per circuit/optimizer run in seconds "
            "(default: 900 = 15 min, 0 disables)"
        ),
    )

    # Concurrency control
    parser.add_argument(
        "--max-concurrent-fast",
        type=int,
        default=4,
        help="Max concurrent fast optimizers (tket, qiskit) (default: 4)",
    )
    parser.add_argument(
        "--max-concurrent-wisq-rules",
        type=int,
        default=2,
        help="Max concurrent wisq_rules runs (default: 2)",
    )
    parser.add_argument(
        "--max-concurrent-wisq-bqskit",
        type=int,
        default=1,
        help="Max concurrent wisq_bqskit runs (default: 1, SHOULD NOT BE INCREASED)",
    )

    # Artifact storage
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory for output circuit artifacts (default: data/artifacts)",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Don't save output circuit files",
    )

    # Execution
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing database, skipping completed runs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Ensure database directory exists
    args.database.parent.mkdir(parents=True, exist_ok=True)

    # Create database
    db = TrajectoryDatabase(args.database)

    # Import circuits if requested
    if args.import_metadata is not None:
        print(f"Importing circuits from: {args.import_metadata}")
        num_imported = import_from_metadata_json(
            db=db,
            metadata_path=args.import_metadata,
            source="local",
        )
        if num_imported > 0:
            print(f"Imported {num_imported} new circuits")
        else:
            print("All circuits already in database (0 new imports)")

    if args.import_artifacts is not None:
        print(f"Importing circuits from artifacts in: {args.import_artifacts}")
        num_imported = import_from_artifacts_dir(
            db=db,
            artifacts_dir=args.import_artifacts,
            category=None,  # Auto-detect categories
            max_qubits=args.max_qubits,
            step1_db_path=args.step1_database,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        if num_imported > 0:
            print(f"Imported {num_imported} new artifact circuits")
        else:
            print("All artifact circuits already in database (0 new imports)")

    hardware_aware = args.hardware_aware == "yes"
    try:
        optimizer_names = resolve_optimizer_names(args.optimizers, hardware_aware)
        requested_reruns = (
            resolve_optimizer_names(args.rerun_optimizers, hardware_aware)
            if args.rerun_optimizers
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Register optimizers
    print(f"\nRegistering optimizers (WISQ+BQSKit timeout: {args.wisq_bqskit_timeout}s)...")
    num_optimizers = register_optimizers(
        db,
        optimizer_names,
        wisq_bqskit_timeout=args.wisq_bqskit_timeout,
    )
    if num_optimizers > 0:
        print(f"Registered {num_optimizers} new optimizers")

    # Check circuits
    circuits = db.list_circuits(max_qubits=args.max_qubits)
    if args.sources:
        circuits = [c for c in circuits if c.source in args.sources]
    if args.categories:
        circuits = [c for c in circuits if c.category in args.categories]
    if args.exclude_name_like:
        for pattern in args.exclude_name_like:
            import fnmatch
            circuits = [c for c in circuits if not fnmatch.fnmatch(c.name, pattern)]

    if not circuits:
        print("\nNo circuits found. Use --import-metadata or --import-artifacts to add circuits first.")
        print("Examples:")
        print("  --import-metadata benchmarks/ai_transpile/metadata.json")
        print("  --import-artifacts data/artifacts --step1-database data/trajectories.db")
        db.close()
        sys.exit(1)

    # Get optimizers
    optimizers = [db.get_optimizer_by_name(name) for name in optimizer_names]
    optimizers = [o for o in optimizers if o is not None]

    total_possible = len(circuits) * len(optimizers)
    print("\nSearch Configuration:")
    print(f"  Circuits: {len(circuits)}")
    print(f"  Optimizers: {', '.join(optimizer_names)}")
    print(f"  Hardware-aware: {args.hardware_aware}")
    print(f"  Total possible runs: {total_possible}")
    print(f"  Max qubits: {args.max_qubits}")
    print(f"  Categories: {args.categories or 'all'}")
    print(f"  Save artifacts: {not args.no_artifacts}")
    print(f"  Resume mode: {args.resume}")
    print(f"  Optimizer timeout: {args.optimizer_timeout}s")
    if args.rerun:
        print("  Rerun: all optimizers (keeping history)")
    elif requested_reruns:
        print(f"  Rerun: {', '.join(requested_reruns)} (keeping history)")
    print()
    print("Concurrency settings:")
    print(f"  Fast optimizers: {args.max_concurrent_fast}")
    print(f"  WISQ rules: {args.max_concurrent_wisq_rules}")
    print(f"  WISQ+BQSKit: {args.max_concurrent_wisq_bqskit}")
    print()

    # Calculate optimizer totals for progress tracker
    optimizer_totals = {name: len(circuits) for name in optimizer_names}

    # Determine which optimizers to rerun
    # --rerun: rerun all optimizers
    # --rerun-optimizers: rerun only specified optimizers
    if args.rerun:
        rerun_optimizers = optimizer_names  # Rerun all
    else:
        rerun_optimizers = requested_reruns

    db.close()  # Close before starting async runner

    # Create config
    config = SingleStepConfig(
        database_path=args.database,
        max_qubits=args.max_qubits,
        categories=args.categories,
        optimizers=optimizer_names,
        rerun_optimizers=rerun_optimizers,
        wisq_bqskit_timeout=args.wisq_bqskit_timeout,
        optimizer_timeout_seconds=args.optimizer_timeout,
        max_concurrent_fast=args.max_concurrent_fast,
        max_concurrent_wisq_rules=args.max_concurrent_wisq_rules,
        max_concurrent_wisq_bqskit=args.max_concurrent_wisq_bqskit,
        artifact_dir=args.artifact_dir,
        save_artifacts=not args.no_artifacts,
    )

    # Run grid search
    print("Starting single-step grid search...")
    print()

    # Create progress tracker if not in quiet mode
    try:
        if not args.quiet:
            progress_tracker = OptimizersProgressTracker(
                optimizer_names,
                optimizer_totals,
                log_file=args.log_file,
            )
            with progress_tracker:
                with AsyncSingleStepRunner(config, progress_tracker=progress_tracker) as runner:
                    report = runner.run_sync(resume=args.resume)
        else:
            with AsyncSingleStepRunner(config) as runner:
                report = runner.run_sync(resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Active optimizer processes were asked to stop.")
        sys.exit(130)

    print()  # New line after progress
    print()

    # Print report
    print("=" * 70)
    print("Single-Step Grid Search Complete")
    print("=" * 70)
    print(f"Total circuits:     {report.total_circuits}")
    print(f"Total optimizers:   {report.total_optimizers}")
    print(f"Total runs:         {report.total_runs}")
    print(f"  Completed:        {report.completed_runs}")
    print(f"  Skipped:          {report.skipped_runs}")
    print(f"  Failed:           {report.failed_runs}")
    if report.equivalence_counts:
        print(
            "  Equivalence:      "
            f"verified={report.equivalence_counts.get('verified', 0)}, "
            f"skipped={report.equivalence_counts.get('skipped', 0)}, "
            f"failed={report.equivalence_counts.get('failed', 0)}"
        )
    print(f"Total duration:     {report.total_duration_seconds:.1f}s")

    if report.failures:
        print(f"\nFailures ({len(report.failures)}):")
        for failure in report.failures[:10]:
            print(f"  - {failure['circuit']} / {failure['optimizer']}: {failure.get('error', 'Unknown')[:60]}")
        if len(report.failures) > 10:
            print(f"  ... and {len(report.failures) - 10} more")

    if report.best_by_optimizer:
        print("\nBest improvement by optimizer:")
        for optimizer, best in sorted(report.best_by_optimizer.items()):
            print(f"  {optimizer:20s}: {best['improvement']:6.2f}% ({best['circuit']})")

    # Auto-import artifacts after the run so they become circuits for the next step
    if args.import_artifacts is not None:
        print("\nImporting artifacts from Step 1 runs for Step 2...")
        import_db = TrajectoryDatabase(args.database)
        num_imported = import_from_artifacts_dir(
            db=import_db,
            artifacts_dir=args.import_artifacts,
            category=None,  # Auto-detect categories
            max_qubits=args.max_qubits,
            step1_db_path=args.step1_database,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        if num_imported > 0:
            print(f"Imported {num_imported} new artifact circuits for Step 2")
        else:
            print("All artifact circuits already in database (0 new imports)")
        import_db.close()

    # Final database statistics
    db = TrajectoryDatabase(args.database)
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Circuits:          {stats['num_circuits']}")
    print(f"  Optimizers:        {stats['num_optimizers']}")
    print(f"  Optimization runs: {stats['num_optimization_runs']}")
    artifact_count = len(db.list_circuits(source='artifact'))
    if artifact_count > 0:
        print(f"  Artifact circuits: {artifact_count}")
    db.close()

    print(f"\nResults saved to {args.database}")


if __name__ == "__main__":
    main()
