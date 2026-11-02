"""RL Trajectory Database for Quantum Circuit Optimizer Selection.

This package provides infrastructure for collecting and storing optimization
trajectories for offline RL training. It includes:

- TrajectoryDatabase: SQLite-backed storage for circuits, optimizers, and trajectories
- RLState: State representation for RL with feature extraction
- Reward functions: Multiple reward variants for offline experimentation
- BenchpressImporter: Import circuits from Qiskit Benchpress
- GridSearchRunner: Exhaustive search over optimizer combinations

Example usage:

    from benchmarks.ai_transpile.rl_trajectory import (
        TrajectoryDatabase,
        GridSearchConfig,
        GridSearchRunner,
        BenchpressImporter,
    )

    # Create database and import circuits
    db = TrajectoryDatabase("data/trajectories.db")
    importer = BenchpressImporter()
    importer.import_to_database(db, categories=["qft"], max_qubits=10)

    # Run grid search
    config = GridSearchConfig(
        categories=["qft"],
        max_chain_length=2,
        database_path=Path("data/trajectories.db"),
    )
    with GridSearchRunner(config) as runner:
        report = runner.run_exhaustive_search()

    # Export for RL training
    d4rl_data = db.export_to_d4rl_format()
"""

from .database import (
    CircuitRecord,
    OptimizerRecord,
    SARSTuple,
    TrajectoryDatabase,
    TrajectoryStepRecord,
)
from .grid_search import (
    ALL_OPTIMIZER_CONFIGS,
    HARDWARE_OPTIMIZER_CONFIGS,
    OPTIMIZER_CONFIGS,
    GridSearchConfig,
    GridSearchProgress,
    GridSearchReport,
    GridSearchRunner,
    generate_optimizer_combinations,
    run_quick_grid_search,
)
from .importer import (
    ArtifactCircuitImporter,
    BenchpressImporter,
    CircuitInfo,
    LocalCircuitImporter,
    import_from_artifacts_dir,
    import_from_metadata_json,
)
from .reward import (
    RewardConfig,
    RewardSet,
    compute_all_rewards,
    compute_efficiency_normalized_reward,
    compute_efficiency_reward,
    compute_improvement_only_reward,
    compute_improvement_percentage,
    compute_multi_objective_reward,
    compute_sparse_final_reward,
    get_default_config,
    summarize_trajectory_rewards,
)
from .single_step_search import (
    AsyncSingleStepRunner,
    OptimizersProgressTracker,
    SingleStepConfig,
    SingleStepProgress,
    SingleStepReport,
    SingleStepResult,
    run_single_step_grid_search,
)
from .state import (
    CATEGORIES,
    RLState,
    compute_circuit_features,
    get_category_encoding,
    normalize_state,
)

__all__ = [
    # Database
    "CircuitRecord",
    "OptimizerRecord",
    "SARSTuple",
    "TrajectoryDatabase",
    "TrajectoryStepRecord",
    # Grid Search
    "GridSearchConfig",
    "GridSearchProgress",
    "GridSearchReport",
    "GridSearchRunner",
    "ALL_OPTIMIZER_CONFIGS",
    "HARDWARE_OPTIMIZER_CONFIGS",
    "OPTIMIZER_CONFIGS",
    "generate_optimizer_combinations",
    "run_quick_grid_search",
    # Single-Step Search
    "AsyncSingleStepRunner",
    "OptimizersProgressTracker",
    "SingleStepConfig",
    "SingleStepProgress",
    "SingleStepReport",
    "SingleStepResult",
    "run_single_step_grid_search",
    # Importer
    "BenchpressImporter",
    "CircuitInfo",
    "ArtifactCircuitImporter",
    "LocalCircuitImporter",
    "import_from_artifacts_dir",
    "import_from_metadata_json",
    # Reward
    "RewardConfig",
    "RewardSet",
    "compute_all_rewards",
    "compute_efficiency_normalized_reward",
    "compute_efficiency_reward",
    "compute_improvement_only_reward",
    "compute_improvement_percentage",
    "compute_multi_objective_reward",
    "compute_sparse_final_reward",
    "get_default_config",
    "summarize_trajectory_rewards",
    # State
    "CATEGORIES",
    "RLState",
    "compute_circuit_features",
    "get_category_encoding",
    "normalize_state",
]
