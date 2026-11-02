"""Evaluation helpers for the v2 two-step chain RL setup."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_LOG_PATH = Path(__file__).resolve().parent / "training_runs.csv"
TRAINING_LOG_COLUMNS = [
    "algorithm_type",
    "reward_type",
    "model_path",
    "hyperparameters",
    "runtime_weight",
    "two_qubit_weight",
    "depth_weight",
    "final_reward_weight",
    "stop_penalty",
    "used_hyperparameter_grid_search",
    "validation_utility",
    "test_utility",
    "validation_avg_two_qubit_gate_reduction_pct",
    "test_avg_two_qubit_gate_reduction_pct",
    "validation_avg_depth_reduction_pct",
    "test_avg_depth_reduction_pct",
    "validation_avg_runtime_seconds",
    "test_avg_runtime_seconds",
    "validation_avg_runtime_score",
    "test_avg_runtime_score",
]


def evaluate_model_on_frame(
    model,
    frame,
    build_env_fn,
    optimizer_names,
    category_names,
    metric_scale,
    method_name,
    reward_type,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
):
    """Run a trained policy once per circuit and collect chosen chains.

    The selected row's `utility_reward` is only the reward for the step
    represented by that row, so this function also stores `episode_reward`
    as the sum across both environment steps.

    For runtime reporting we intentionally use the selected chain's total
    runtime columns rather than summing per-step runtime penalties. That keeps
    the logged runtime score aligned with the logged total runtime seconds.
    """
    env = build_env_fn(
        frame,
        optimizer_names,
        category_names,
        metric_scale,
        shuffle_each_epoch=False,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
    )
    selected_rows = []

    for circuit in sorted(frame["original_circuit"].unique()):
        obs, _ = env.reset(options={"circuit": circuit})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        episode_reward = float(reward)

        selected_chain = info["selected_chain"]
        if selected_chain is None:
            raise ValueError(
                f"Evaluation produced no selected chain for circuit {circuit}."
            )

        selected_row = frame[
            (frame["original_circuit"] == circuit)
            & (frame["chain_name"] == selected_chain)
        ].iloc[0].copy()

        if not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)

            selected_chain = info["selected_chain"]
            if (
                not info["invalid_action"]
                and selected_chain is not None
                and "__" in selected_chain
            ):
                selected_row = frame[
                    (frame["original_circuit"] == circuit)
                    & (frame["chain_name"] == selected_chain)
                ].iloc[0].copy()

        row = selected_row.copy()
        row["method"] = method_name
        row["selected_chain"] = selected_row["chain_name"]
        row["episode_reward"] = episode_reward
        row["episode_total_runtime_seconds"] = float(selected_row["total_runtime_seconds"])
        row["episode_runtime_score"] = float(selected_row["total_runtime_score"])
        selected_rows.append(row)

    return pd.DataFrame(selected_rows)


def summarize_selected_frame(frame):
    """Compute the evaluation summary values stored in the training log."""
    avg_runtime_seconds = float(frame["episode_total_runtime_seconds"].mean())
    runtime_denominator = float(frame["runtime_denominator"].iloc[0])

    return {
        "utility": float(frame["episode_reward"].mean()),
        "avg_two_qubit_gate_reduction_pct": float(
            frame["total_two_qubit_gate_reduction_pct"].mean()
        ),
        "avg_depth_reduction_pct": float(frame["depth_reduction_pct"].mean()),
        "avg_runtime_seconds": avg_runtime_seconds,
        "avg_runtime_score": float(
            np.clip(np.log1p(max(avg_runtime_seconds, 0.0)) / runtime_denominator, 0, 2)
        ),
    }


def append_training_log(
    algorithm_name,
    reward_type,
    model_path,
    config,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
    invalid_stop_penalty,
    used_grid_search,
    validation_selected,
    test_selected,
):
    """Append one training result row to the persistent CSV log."""
    validation_summary = summarize_selected_frame(validation_selected)
    test_summary = summarize_selected_frame(test_selected)

    row = {
        "algorithm_type": algorithm_name,
        "reward_type": reward_type,
        "model_path": str(model_path),
        "hyperparameters": json.dumps(config, sort_keys=True),
        "runtime_weight": float(runtime_weight),
        "two_qubit_weight": float(two_qubit_weight),
        "depth_weight": float(depth_weight),
        "final_reward_weight": float(final_reward_weight),
        "stop_penalty": float(invalid_stop_penalty),
        "used_hyperparameter_grid_search": bool(used_grid_search),
        "validation_utility": validation_summary["utility"],
        "test_utility": test_summary["utility"],
        "validation_avg_two_qubit_gate_reduction_pct": validation_summary[
            "avg_two_qubit_gate_reduction_pct"
        ],
        "test_avg_two_qubit_gate_reduction_pct": test_summary[
            "avg_two_qubit_gate_reduction_pct"
        ],
        "validation_avg_depth_reduction_pct": validation_summary[
            "avg_depth_reduction_pct"
        ],
        "test_avg_depth_reduction_pct": test_summary["avg_depth_reduction_pct"],
        "validation_avg_runtime_seconds": validation_summary["avg_runtime_seconds"],
        "test_avg_runtime_seconds": test_summary["avg_runtime_seconds"],
        "validation_avg_runtime_score": validation_summary["avg_runtime_score"],
        "test_avg_runtime_score": test_summary["avg_runtime_score"],
    }

    row_frame = pd.DataFrame([row]).reindex(columns=TRAINING_LOG_COLUMNS)

    if TRAINING_LOG_PATH.exists():
        existing = pd.read_csv(TRAINING_LOG_PATH)
        existing = existing.reindex(columns=TRAINING_LOG_COLUMNS)
        existing["two_qubit_weight"] = existing["two_qubit_weight"].fillna(0.5)
        existing["depth_weight"] = existing["depth_weight"].fillna(0.5)
        existing["final_reward_weight"] = existing["final_reward_weight"].fillna(0.5)
        combined = pd.concat([existing, row_frame], ignore_index=True)
    else:
        combined = row_frame

    combined.to_csv(TRAINING_LOG_PATH, index=False)
