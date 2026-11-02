#!/usr/bin/env python3
"""Initialize trajectory database with circuits and optimizers.

This script populates the database with circuit records from metadata.json
and registers all available optimizers without running any optimization.
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
    OptimizerRecord,
    TrajectoryDatabase,
    import_from_metadata_json,
)


def register_optimizers(db: TrajectoryDatabase, wisq_bqskit_timeout: int = 300) -> int:
    """Register all optimizers in the database.
    
    Args:
        db: TrajectoryDatabase instance
        wisq_bqskit_timeout: Timeout in seconds for WISQ+BQSKit optimizer (default 5 min)
    
    Returns:
        Number of optimizers registered
    """
    registered = 0
    
    for name, config in OPTIMIZER_CONFIGS.items():
        # Override wisq_bqskit timeout to 5 minutes
        options = dict(config["options"])
        if name == "wisq_bqskit":
            options["opt_timeout"] = wisq_bqskit_timeout
        
        optimizer = OptimizerRecord(
            id=None,
            name=name,
            runner_type=config["runner_type"],
            options=options,
            description=config.get("description"),
        )
        
        # Check if already exists
        existing = db.get_optimizer_by_name(name)
        if existing is None:
            db.insert_optimizer(optimizer)
            registered += 1
            print(f"  Registered: {name} ({config['runner_type']})")
        else:
            print(f"  Already exists: {name}")
    
    return registered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize trajectory database with circuits and optimizers"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("benchmarks/ai_transpile/metadata.json"),
        help="Path to metadata.json file (default: benchmarks/ai_transpile/metadata.json)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/trajectories.db"),
        help="Path to trajectory database (default: data/trajectories.db)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Circuit categories to filter (e.g., qft efficient_su2 real_amplitudes)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Maximum number of qubits to include",
    )
    parser.add_argument(
        "--wisq-bqskit-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for WISQ+BQSKit optimizer (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--skip-optimizers",
        action="store_true",
        help="Skip registering optimizers (circuits only)",
    )

    args = parser.parse_args()

    # Ensure database directory exists
    args.database.parent.mkdir(parents=True, exist_ok=True)

    # Create database
    db = TrajectoryDatabase(args.database)
    print(f"Created database: {args.database}")

    # Import circuits from metadata
    print(f"\nImporting circuits from: {args.metadata}")
    num_circuits_imported = import_from_metadata_json(
        db=db,
        metadata_path=args.metadata,
        source="local",
    )
    print(f"Imported {num_circuits_imported} new circuits")

    # Register optimizers
    if not args.skip_optimizers:
        print(f"\nRegistering optimizers (WISQ+BQSKit timeout: {args.wisq_bqskit_timeout}s):")
        num_optimizers = register_optimizers(db, wisq_bqskit_timeout=args.wisq_bqskit_timeout)
        print(f"Registered {num_optimizers} new optimizers")

    # Get all circuits from database
    all_circuits = db.list_circuits()
    
    # Apply filters if specified
    filtered_circuits = all_circuits
    if args.categories:
        category_set = set(args.categories)
        filtered_circuits = [
            c for c in filtered_circuits 
            if any(cat in c.category for cat in category_set)
        ]
    
    if args.max_qubits:
        filtered_circuits = [
            c for c in filtered_circuits 
            if c.num_qubits <= args.max_qubits
        ]

    print(f"\nDatabase contains {len(all_circuits)} total circuits")
    if args.categories or args.max_qubits:
        print(f"Filtered to {len(filtered_circuits)} circuits matching criteria")
        circuits_to_show = filtered_circuits
    else:
        circuits_to_show = all_circuits

    print("\nCircuits in database:")
    for circuit in circuits_to_show:
        print(
            f"  - {circuit.name:30s} | {circuit.category:15s} | "
            f"{circuit.num_qubits:2d} qubits | "
            f"{circuit.initial_two_qubit_gates:3d} 2q gates"
        )

    # Print optimizer summary
    optimizers = db.list_optimizers()
    if optimizers:
        print(f"\nOptimizers in database ({len(optimizers)}):")
        for opt in optimizers:
            print(f"  - {opt.name:20s} | {opt.runner_type:15s} | {opt.description or ''}")

    # Print database location
    print(f"\nDatabase ready at: {args.database.resolve()}")
    print(f"Size: {args.database.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
