"""Hyperparameter search helpers for the v2 two-step chain RL setup."""

from itertools import product

import pandas as pd
from stable_baselines3.common.monitor import Monitor


def expand_search_grid(search_grid):
    """Yield one concrete hyperparameter dictionary at a time."""
    keys = list(search_grid.keys())
    values = [search_grid[key] for key in keys]
    for combination in product(*values):
        yield dict(zip(keys, combination))


def run_grid_search(
    algorithm_name,
    base_config,
    search_grid,
    train_df,
    validation_df,
    optimizer_names,
    category_names,
    metric_scale,
    build_env_fn,
    build_model_fn,
    evaluate_model_on_frame_fn,
    tuning_timesteps,
    reward_type,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
):
    """Run a plain grid search on validation utility for any supported algorithm."""
    if not search_grid:
        return pd.DataFrame()

    results = []
    all_configs = list(expand_search_grid(search_grid))

    for config_index, overrides in enumerate(all_configs, start=1):
        print(f"[{config_index}/{len(all_configs)}] testing {overrides}")

        candidate_config = base_config.copy()
        candidate_config.update(overrides)

        candidate_train_env = Monitor(
            build_env_fn(
                train_df,
                optimizer_names,
                category_names,
                metric_scale,
                shuffle_each_epoch=True,
                reward_type=reward_type,
                runtime_weight=runtime_weight,
                two_qubit_weight=two_qubit_weight,
                depth_weight=depth_weight,
                final_reward_weight=final_reward_weight,
            )
        )
        candidate_model = build_model_fn(
            algorithm_name,
            candidate_train_env,
            candidate_config,
        )
        candidate_model.learn(total_timesteps=tuning_timesteps)

        validation_selected = evaluate_model_on_frame_fn(
            candidate_model,
            validation_df,
            build_env_fn,
            optimizer_names,
            category_names,
            metric_scale,
            method_name=f"{algorithm_name.upper()} validation",
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
            final_reward_weight=final_reward_weight,
        )

        results.append(
            {
                **overrides,
                "avg_val_utility": validation_selected["episode_reward"].mean(),
                "avg_val_runtime_seconds": validation_selected[
                    "episode_total_runtime_seconds"
                ].mean(),
                "avg_val_total_two_qubit_reduction_pct": validation_selected[
                    "total_two_qubit_gate_reduction_pct"
                ].mean(),
                "selected_unique_chains": validation_selected["selected_chain"].nunique(),
            }
        )

    return pd.DataFrame(results).sort_values("avg_val_utility", ascending=False)
