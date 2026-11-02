#!/usr/bin/env python3
"""CLI script for running grid search over quantum circuit optimizers.

This script runs exhaustive grid search across circuits and optimizer
combinations, recording trajectories for offline RL training.

Usage:
    python scripts/run_grid_search.py --categories qft qaoa --max-chain-length 2
    python scripts/run_grid_search.py --database data/trajectories.db --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ai_transpile.rl_trajectory import (  # noqa: E402
    OPTIMIZER_CONFIGS,
    BenchpressImporter,
    GridSearchConfig,
    GridSearchProgress,
    GridSearchRunner,
    LocalCircuitImporter,
    TrajectoryDatabase,
    import_from_artifacts_dir,
    import_from_metadata_json,
)


def progress_callback(progress: GridSearchProgress) -> None:
    """Print progress updates."""
    pct_circuits = 100 * progress.completed_circuits / progress.total_circuits if progress.total_circuits > 0 else 0
    pct_combos = (
        100 * progress.completed_combinations / progress.total_combinations if progress.total_combinations > 0 else 0
    )

    print(
        f"\r[{pct_circuits:5.1f}%] Circuit {progress.completed_circuits}/{progress.total_circuits}: "
        f"{progress.current_circuit[:30]:30s} | "
        f"[{pct_combos:5.1f}%] Combo: {progress.current_combination[:40]:40s} | "
        f"Elapsed: {progress.elapsed_seconds:.1f}s",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid search over quantum circuit optimizers")

    # Circuit sources
    parser.add_argument(
        "--import-benchpress",
        action="store_true",
        help="Import circuits from Qiskit Benchpress repository",
    )
    parser.add_argument(
        "--import-local",
        type=Path,
        default=None,
        help="Import circuits from a local directory",
    )
    parser.add_argument(
        "--import-artifacts",
        type=Path,
        default=None,
        help="Import circuits from optimized artifact outputs",
    )
    parser.add_argument(
        "--import-metadata",
        type=Path,
        default=None,
        help="Import circuits from a metadata.json file",
    )
    parser.add_argument(
        "--artifact-category",
        type=str,
        default="artifact_step1",
        help="Category name to assign to artifact circuits",
    )

    # Circuit filters
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Circuit categories to include (e.g., qft qaoa clifford)",
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
        choices=list(OPTIMIZER_CONFIGS.keys()),
        help="Optimizers to use (default: all)",
    )
    parser.add_argument(
        "--max-chain-length",
        type=int,
        default=3,
        help="Maximum chain length (default: 3)",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single optimizers, not chains",
    )

    # Database
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to trajectory database (default: data/trajectories.db)",
    )

    # Execution
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing database, skipping completed trajectories",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="Time budget per episode in seconds (default: 300)",
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

    # Import circuits
    imported = 0

    if args.import_benchpress:
        print("Importing circuits from Benchpress...")
        importer = BenchpressImporter()
        imported += importer.import_to_database(
            db,
            categories=args.categories,
            max_qubits=args.max_qubits,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

    if args.import_local is not None:
        print(f"Importing circuits from {args.import_local}...")
        local_importer = LocalCircuitImporter(args.import_local)
        imported += local_importer.import_to_database(
            db,
            max_qubits=args.max_qubits,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

    if args.import_artifacts is not None:
        print(f"Importing circuits from artifacts in {args.import_artifacts}...")
        imported += import_from_artifacts_dir(
            db,
            args.import_artifacts,
            category=args.artifact_category,
            max_qubits=args.max_qubits,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

    if args.import_metadata is not None:
        print(f"Importing circuits from {args.import_metadata}...")
        imported += import_from_metadata_json(
            db,
            args.import_metadata,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

    if imported > 0:
        print(f"Imported {imported} circuits")

    # Check if we have circuits
    circuits = db.list_circuits(max_qubits=args.max_qubits)
    if args.categories:
        circuits = [c for c in circuits if c.category in args.categories]

    if not circuits:
        print("No circuits available. Use --import-benchpress, --import-local, or --import-metadata to add circuits.")
        db.close()
        return

    print(f"Database has {len(circuits)} circuits matching criteria")

    # Configure grid search
    config = GridSearchConfig(
        categories=args.categories,
        optimizers=args.optimizers or list(OPTIMIZER_CONFIGS.keys()),
        max_chain_length=args.max_chain_length,
        enable_chain_search=not args.single_only,
        time_budget=args.time_budget,
        max_qubits=args.max_qubits,
        database_path=args.database,
    )

    # Run grid search
    print("\nStarting grid search...")
    print(f"  Optimizers: {config.optimizers}")
    print(f"  Max chain length: {config.max_chain_length}")
    print(f"  Categories: {config.categories or 'all'}")
    print()

    callback = None if args.quiet else progress_callback

    with GridSearchRunner(config, progress_callback=callback) as runner:
        report = runner.run_exhaustive_search(resume=args.resume)

    print()  # New line after progress
    print()

    # Print report
    print("=" * 60)
    print("Grid Search Complete")
    print("=" * 60)
    print(f"Total circuits:      {report.total_circuits}")
    print(f"Total trajectories:  {report.total_trajectories}")
    print(f"Total steps:         {report.total_steps}")
    print(f"Total duration:      {report.total_duration_seconds:.1f}s")

    if report.failures:
        print(f"\nFailures ({len(report.failures)}):")
        for failure in report.failures[:10]:  # Show first 10
            print(f"  - {failure['circuit']}: {failure.get('error', 'Unknown error')}")
        if len(report.failures) > 10:
            print(f"  ... and {len(report.failures) - 10} more")

    if report.best_by_category:
        print("\nBest optimizer chains by category:")
        for category, best in sorted(report.best_by_category.items()):
            print(f"  {category}:")
            print(f"    Chain: {best['chain']}")
            print(f"    Improvement: {best['improvement']:.1f}%")

    # Print database statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Circuits: {stats['num_circuits']}")
    print(f"  Optimizers: {stats['num_optimizers']}")
    print(f"  Trajectories: {stats['num_trajectories']}")
    print(f"  Trajectory steps: {stats['num_trajectory_steps']}")

    if stats.get("avg_improvement_percentage") is not None:
        print(f"  Avg improvement: {stats['avg_improvement_percentage']:.1f}%")
        print(f"  Max improvement: {stats['max_improvement_percentage']:.1f}%")
        print(f"  Avg trajectory length: {stats['avg_trajectory_length']:.1f}")

    db.close()
    print(f"\nResults saved to {args.database}")


if __name__ == "__main__":
    main()
