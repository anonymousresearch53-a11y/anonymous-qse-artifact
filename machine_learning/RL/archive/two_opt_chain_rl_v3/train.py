"""Training entry point for the v2 two-step chain RL setup."""

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

try:
    from .data_processing import RANDOM_STATE, feature_batch_names, prepare_datasets
    from .environment import OptimizerChainEnv
    from .evaluation import append_training_log, evaluate_model_on_frame
    from .hyperparameter_search import run_grid_search
    from .rewards import CURRENT_REWARD, SUPPORTED_REWARD_TYPES, apply_reward
except ImportError:
    from data_processing import RANDOM_STATE, feature_batch_names, prepare_datasets
    from environment import OptimizerChainEnv
    from evaluation import append_training_log, evaluate_model_on_frame
    from hyperparameter_search import run_grid_search
    from rewards import CURRENT_REWARD, SUPPORTED_REWARD_TYPES, apply_reward


# Main training defaults live here so experimentation is easy.
ALGORITHM_NAME = "ppo"
REWARD_TYPE = CURRENT_REWARD
RUNTIME_WEIGHT = 0.1
TWO_QUBIT_WEIGHT = 0.5
DEPTH_WEIGHT = 0.5
FINAL_REWARD_WEIGHT = 0.5
INVALID_STOP_PENALTY = -1.0
TOTAL_TIMESTEPS = 30_000
RUN_GRID_SEARCH = False
GRID_SEARCH_TIMESTEPS = 10_000
SAVE_SPLITS = True
FEATURE_BATCH = "original"
ALGORITHM_SEED = RANDOM_STATE

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Keep the algorithm map explicit rather than building a plugin system.
ALGORITHMS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}

# Per-algorithm defaults are separated so switching algorithms is simple.
DEFAULT_CONFIGS = {
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 0.001,
        "n_steps": 256,
        "batch_size": 64,
        "gamma": 1.0,
        "ent_coef": 0.05,
        "verbose": 0,
        "seed": RANDOM_STATE,
    },
    "a2c": {
        "policy": "MlpPolicy",
        "learning_rate": 0.001,
        "n_steps": 256,
        "gamma": 1.0,
        "ent_coef": 0.0,
        "verbose": 0,
        "seed": RANDOM_STATE,
    },
    "dqn": {
        "policy": "MlpPolicy",
        "learning_rate": 0.001,
        "buffer_size": 10_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 1.0,
        "train_freq": 1,
        "gradient_steps": 1,
        "verbose": 0,
        "seed": RANDOM_STATE,
    },
}

# Keep default search grids simple and explicit for each supported algorithm.
DEFAULT_SEARCH_GRIDS = {
    "ppo": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [256, 512, 1024],
        "batch_size": [64, 128],
        "ent_coef": [0.0, 0.01, 0.05],
        "gamma": [1.0],
    },
    "a2c": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [128, 256, 512],
        "ent_coef": [0.0, 0.01, 0.05],
        "gamma": [1.0],
    },
    "dqn": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "buffer_size": [5_000, 10_000, 20_000],
        "learning_starts": [500, 1_000, 2_000],
        "batch_size": [32, 64, 128],
        "gamma": [1.0],
    }
}


class RewardHistoryCallback(BaseCallback):
    """Capture episode rewards during learning for quick diagnostics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def parse_bool_arg(value):
    """Parse a flexible true/false CLI argument."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_search_flag(grid_search_arg, hyperparameter_search_arg):
    """Resolve the two search flag names into one internal boolean.

    The two CLI names are treated as aliases for the same behavior.
    """
    if grid_search_arg is None and hyperparameter_search_arg is None:
        return RUN_GRID_SEARCH

    if (
        grid_search_arg is not None
        and hyperparameter_search_arg is not None
        and grid_search_arg != hyperparameter_search_arg
    ):
        raise ValueError(
            "grid_search and hyperparameter_search arguments disagree. "
            "Use matching values or provide only one of them."
        )

    if hyperparameter_search_arg is not None:
        return hyperparameter_search_arg
    return grid_search_arg


def build_arg_parser():
    """Create the command-line interface for training runs."""
    parser = argparse.ArgumentParser(
        description="Train the v2 two-step optimizer-chain RL model.",
    )
    parser.add_argument(
        "--algorithm",
        "--algorithm-type",
        dest="algorithm_name",
        default=ALGORITHM_NAME,
        choices=sorted(ALGORITHMS.keys()),
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--reward-type",
        default=REWARD_TYPE,
        choices=SUPPORTED_REWARD_TYPES,
        help="Reward function to apply before training.",
    )
    parser.add_argument(
        "--grid-search",
        type=parse_bool_arg,
        default=None,
        help="Whether to run the grid search helper.",
    )
    parser.add_argument(
        "--hyperparameter-search",
        type=parse_bool_arg,
        default=None,
        help="Alias for grid-search. Uses the same internal behavior.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Total training timesteps for the final model.",
    )
    parser.add_argument(
        "--grid-search-timesteps",
        type=int,
        default=GRID_SEARCH_TIMESTEPS,
        help="Training timesteps for each hyperparameter-search candidate.",
    )
    parser.add_argument(
        "--runtime-weight",
        type=float,
        default=RUNTIME_WEIGHT,
        help="Runtime penalty weight used by the selected reward.",
    )
    parser.add_argument(
        "--two-qubit-weight",
        type=float,
        default=TWO_QUBIT_WEIGHT,
        help="Two-qubit weight used by weighted mixed rewards.",
    )
    parser.add_argument(
        "--depth-weight",
        type=float,
        default=DEPTH_WEIGHT,
        help="Depth weight used by weighted or lexicographic rewards.",
    )
    parser.add_argument(
        "--final-reward-weight",
        type=float,
        default=FINAL_REWARD_WEIGHT,
        help="Weight for the terminal final-vs-original correction.",
    )
    parser.add_argument(
        "--feature-batch",
        default=FEATURE_BATCH,
        choices=feature_batch_names(),
        help="Observation feature batch to use.",
    )
    parser.add_argument(
        "--algorithm-seed",
        type=int,
        default=ALGORITHM_SEED,
        help="Random seed for the RL algorithm. Does not change the data split seed.",
    )
    return parser


def reward_splits(
    runtime_weight=RUNTIME_WEIGHT,
    reward_type=REWARD_TYPE,
    two_qubit_weight=TWO_QUBIT_WEIGHT,
    depth_weight=DEPTH_WEIGHT,
    save_splits=SAVE_SPLITS,
    feature_batch=FEATURE_BATCH,
):
    """Prepare datasets, then apply the currently selected reward function."""
    bundle = prepare_datasets(save=save_splits, feature_batch=feature_batch)
    for key in ("full_df", "train_df", "validation_df", "test_df"):
        bundle[key] = apply_reward(
            bundle[key],
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )
    return bundle


def build_env(
    frame,
    optimizer_names,
    category_names,
    metric_scale,
    observation_feature_scale,
    static_feature_columns,
    dynamic_feature_columns,
    include_category,
    shuffle_each_epoch,
    reward_type=REWARD_TYPE,
    runtime_weight=RUNTIME_WEIGHT,
    two_qubit_weight=TWO_QUBIT_WEIGHT,
    depth_weight=DEPTH_WEIGHT,
    final_reward_weight=FINAL_REWARD_WEIGHT,
    invalid_stop_penalty=INVALID_STOP_PENALTY,
):
    """Create one environment instance with the shared v2 defaults."""
    return OptimizerChainEnv(
        frame,
        optimizer_names,
        category_names,
        metric_scale,
        observation_feature_scale=observation_feature_scale,
        static_feature_columns=static_feature_columns,
        dynamic_feature_columns=dynamic_feature_columns,
        include_category=include_category,
        shuffle_each_epoch=shuffle_each_epoch,
        invalid_stop_penalty=invalid_stop_penalty,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
    )


def build_model(algorithm_name, env, config):
    """Instantiate the selected Stable Baselines model."""
    algorithm_key = algorithm_name.lower()
    if algorithm_key not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    # The SB3 constructors take the policy separately from the rest of config.
    config = config.copy()
    policy = config.pop("policy")
    return ALGORITHMS[algorithm_key](policy, env, **config)


def resolved_search_grid(algorithm_name, search_grid=None):
    """Return the search grid that will actually be used for this run."""
    algorithm_key = algorithm_name.lower()
    if search_grid is not None:
        return search_grid
    return DEFAULT_SEARCH_GRIDS.get(algorithm_key, {})


def print_search_grid(algorithm_name, search_grid):
    """Print the chosen search grid in a short, readable format."""
    print(f"Hyperparameter search grid for {algorithm_name.lower()}:")
    if not search_grid:
        print("  (no search grid available)")
        return

    for key, values in search_grid.items():
        print(f"  {key}: {list(values)}")


def coerce_config_value(example_value, candidate_value):
    """Cast search-result values back to the config type expected by SB3."""
    if isinstance(example_value, bool):
        return bool(candidate_value)
    if isinstance(example_value, int) and not isinstance(example_value, bool):
        return int(candidate_value)
    if isinstance(example_value, float):
        return float(candidate_value)
    return candidate_value


def validate_training_inputs(
    reward_type,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
    total_timesteps,
):
    """Catch simple invalid training settings early."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if runtime_weight < 0:
        raise ValueError("runtime_weight must be non-negative.")
    if two_qubit_weight < 0:
        raise ValueError("two_qubit_weight must be non-negative.")
    if depth_weight < 0:
        raise ValueError("depth_weight must be non-negative.")
    if final_reward_weight < 0:
        raise ValueError("final_reward_weight must be non-negative.")
    if reward_type == "mixed_weighted" and (
        two_qubit_weight == 0 and depth_weight == 0
    ):
        raise ValueError(
            "mixed_weighted reward needs at least one positive metric weight."
        )


def _slugify_float(value):
    """Convert a float into a filename-friendly short token."""
    return str(float(value)).replace("-", "neg").replace(".", "p")


def build_model_path(
    algorithm_name,
    algorithm_seed,
    feature_batch,
    reward_type,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
):
    """Build a model path that stays unique across reward settings."""
    filename = (
        f"optimizer_chain_{algorithm_name.lower()}_{reward_type}"
        f"_seed{int(algorithm_seed)}"
        f"_fb{feature_batch}"
        f"_rw{_slugify_float(runtime_weight)}"
        f"_tqw{_slugify_float(two_qubit_weight)}"
        f"_dw{_slugify_float(depth_weight)}"
        f"_frw{_slugify_float(final_reward_weight)}.zip"
    )
    return MODEL_DIR / filename


def train_model(
    algorithm_name=ALGORITHM_NAME,
    algorithm_seed=ALGORITHM_SEED,
    feature_batch=FEATURE_BATCH,
    reward_type=REWARD_TYPE,
    runtime_weight=RUNTIME_WEIGHT,
    two_qubit_weight=TWO_QUBIT_WEIGHT,
    depth_weight=DEPTH_WEIGHT,
    final_reward_weight=FINAL_REWARD_WEIGHT,
    total_timesteps=TOTAL_TIMESTEPS,
    grid_search_timesteps=GRID_SEARCH_TIMESTEPS,
    run_search=RUN_GRID_SEARCH,
    search_grid=None,
):
    """Train one algorithm and return the useful outputs in a single bundle."""
    validate_training_inputs(
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
        total_timesteps=total_timesteps,
    )

    bundle = reward_splits(
        runtime_weight=runtime_weight,
        reward_type=reward_type,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        save_splits=SAVE_SPLITS,
        feature_batch=feature_batch,
    )

    optimizer_names = bundle["optimizer_names"]
    category_names = bundle["category_names"]
    metric_scale = bundle["metric_scale"]
    observation_feature_scale = bundle["observation_feature_scale"]
    static_feature_columns = bundle["static_feature_columns"]
    dynamic_feature_columns = bundle["dynamic_feature_columns"]
    include_category = bundle["feature_batch_config"]["include_category"]
    train_df = bundle["train_df"]
    validation_df = bundle["validation_df"]
    test_df = bundle["test_df"]

    train_env = Monitor(
        build_env(
            train_df,
            optimizer_names,
            category_names,
            metric_scale,
            observation_feature_scale,
            static_feature_columns,
            dynamic_feature_columns,
            include_category,
            shuffle_each_epoch=True,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
            final_reward_weight=final_reward_weight,
        )
    )

    algorithm_key = algorithm_name.lower()
    base_config = DEFAULT_CONFIGS[algorithm_key].copy()
    base_config["seed"] = int(algorithm_seed)
    chosen_search_grid = resolved_search_grid(algorithm_key, search_grid=search_grid)

    if run_search:
        print_search_grid(algorithm_key, chosen_search_grid)
        # If search is enabled, override the defaults with the best result.
        search_results = run_grid_search(
            algorithm_name=algorithm_key,
            base_config=base_config,
            search_grid=chosen_search_grid,
            train_df=train_df,
            validation_df=validation_df,
            optimizer_names=optimizer_names,
            category_names=category_names,
            metric_scale=metric_scale,
            observation_feature_scale=observation_feature_scale,
            static_feature_columns=static_feature_columns,
            dynamic_feature_columns=dynamic_feature_columns,
            include_category=include_category,
            build_env_fn=build_env,
            build_model_fn=build_model,
            evaluate_model_on_frame_fn=evaluate_model_on_frame,
            tuning_timesteps=grid_search_timesteps,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
            final_reward_weight=final_reward_weight,
        )
        if not search_results.empty:
            best_row = search_results.iloc[0].to_dict()
            for key in chosen_search_grid:
                base_config[key] = coerce_config_value(base_config[key], best_row[key])
    else:
        search_results = None

    model = build_model(algorithm_key, train_env, base_config)
    reward_callback = RewardHistoryCallback()
    model.learn(total_timesteps=total_timesteps, callback=reward_callback)

    # Keep evaluation simple: one deterministic pass on validation and test.
    validation_selected = evaluate_model_on_frame(
        model,
        validation_df,
        build_env,
        optimizer_names,
        category_names,
        metric_scale,
        observation_feature_scale,
        static_feature_columns,
        dynamic_feature_columns,
        include_category,
        method_name=f"{algorithm_key.upper()} validation",
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
    )
    test_selected = evaluate_model_on_frame(
        model,
        test_df,
        build_env,
        optimizer_names,
        category_names,
        metric_scale,
        observation_feature_scale,
        static_feature_columns,
        dynamic_feature_columns,
        include_category,
        method_name=f"{algorithm_key.upper()} test",
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = build_model_path(
        algorithm_key,
        algorithm_seed,
        feature_batch,
        reward_type,
        runtime_weight,
        two_qubit_weight,
        depth_weight,
        final_reward_weight,
    )
    model.save(model_path)
    append_training_log(
        algorithm_name=algorithm_key,
        algorithm_seed=algorithm_seed,
        feature_batch=feature_batch,
        reward_type=reward_type,
        model_path=model_path,
        config=base_config,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
        invalid_stop_penalty=INVALID_STOP_PENALTY,
        used_grid_search=run_search,
        validation_selected=validation_selected,
        test_selected=test_selected,
    )

    return {
        "model": model,
        "model_path": model_path,
        "config": base_config,
        "split_summary": bundle["split_summary"],
        "search_results": search_results,
        "validation_selected": validation_selected,
        "test_selected": test_selected,
        "reward_history": np.array(reward_callback.episode_rewards, dtype=float),
    }


def main():
    """Default script entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_search = resolve_search_flag(args.grid_search, args.hyperparameter_search)

    results = train_model(
        algorithm_name=args.algorithm_name,
        algorithm_seed=args.algorithm_seed,
        feature_batch=args.feature_batch,
        reward_type=args.reward_type,
        runtime_weight=args.runtime_weight,
        two_qubit_weight=args.two_qubit_weight,
        depth_weight=args.depth_weight,
        final_reward_weight=args.final_reward_weight,
        total_timesteps=args.total_timesteps,
        grid_search_timesteps=args.grid_search_timesteps,
        run_search=run_search,
    )
    print(results["split_summary"])
    print(f"Saved model to {results['model_path']}")
    print(f"Validation mean utility: {results['validation_selected']['episode_reward'].mean():.4f}")
    print(f"Test mean utility: {results['test_selected']['episode_reward'].mean():.4f}")


if __name__ == "__main__":
    main()
