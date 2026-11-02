"""Evaluation helpers for the v4 two-step chain RL setup."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


TRAINING_LOG_PATH = Path(__file__).resolve().parent / "training_runs.csv"
TRAINING_LOG_COLUMNS = [
    "algorithm_type",
    "algorithm_seed",
    "feature_batch",
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


def original_selection_row(frame, circuit):
    """Build a synthetic evaluation row for the 'stop immediately' choice."""
    source_row = frame[frame["original_circuit"] == circuit].iloc[0].copy()
    source_row["chain_name"] = "__original__"
    source_row["selected_chain"] = "__original__"
    source_row["final_num_qubits"] = source_row["initial_num_qubits"]
    source_row["final_depth"] = source_row["initial_depth"]
    source_row["final_two_qubit_gates"] = source_row["initial_two_qubit_gates"]
    source_row["final_total_gates"] = source_row["initial_total_gates"]
    source_row["final_two_qubit_depth"] = source_row["initial_two_qubit_depth"]
    source_row["total_runtime_seconds"] = 0.0
    source_row["total_runtime_score"] = 0.0
    source_row["total_two_qubit_gate_reduction"] = 0.0
    source_row["total_two_qubit_gate_reduction_pct"] = 0.0
    source_row["total_depth_reduction"] = 0.0
    source_row["total_depth_reduction_pct"] = 0.0
    source_row["two_qubit_gate_reduction"] = 0.0
    source_row["two_qubit_gate_reduction_pct"] = 0.0
    source_row["depth_reduction"] = 0.0
    source_row["depth_reduction_pct"] = 0.0
    source_row["current_step_two_qubit_gate_reduction"] = 0.0
    source_row["current_step_two_qubit_gate_reduction_pct"] = 0.0
    source_row["current_step_depth_reduction"] = 0.0
    source_row["current_step_depth_reduction_pct"] = 0.0
    source_row["runtime_score"] = 0.0
    source_row["local_step_reward"] = 0.0
    source_row["utility_reward"] = 0.0
    return source_row


def evaluate_model_on_frame(
    model,
    frame,
    build_env_fn,
    optimizer_names,
    category_names,
    metric_scale,
    observation_feature_scale,
    static_feature_columns,
    dynamic_feature_columns,
    include_category,
    method_name,
    reward_type,
    runtime_weight,
    two_qubit_weight,
    depth_weight,
    final_reward_weight,
    verbose=False,
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
        observation_feature_scale,
        static_feature_columns,
        dynamic_feature_columns,
        include_category=include_category,
        shuffle_each_epoch=False,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
        final_reward_weight=final_reward_weight,
    )
    selected_rows = []

    def row_for_selected_chain(circuit_name, selected_chain):
        if selected_chain == "__original__":
            return original_selection_row(frame, circuit_name)
        if selected_chain is None:
            raise ValueError(
                f"Evaluation produced no selected chain for circuit {circuit_name}."
            )
        return frame[
            (frame["original_circuit"] == circuit_name)
            & (frame["chain_name"] == selected_chain)
        ].iloc[0].copy()

    def print_step(circuit_name, step_index, action, selected_chain, reward, row, terminated, truncated, invalid_action):
        if not verbose:
            return
        print(
            f"[{circuit_name}] step {step_index}: action={int(action)} "
            f"chain={selected_chain} reward={float(reward):.4f} "
            f"invalid={bool(invalid_action)} terminated={bool(terminated)} truncated={bool(truncated)}"
        )
        print(
            f"  depth={float(row['final_depth']):.3f} "
            f"depth_reduction_pct={float(row['total_depth_reduction_pct']):.3f} "
            f"twoq_reduction_pct={float(row['total_two_qubit_gate_reduction_pct']):.3f} "
            f"runtime_s={float(row['total_runtime_seconds']):.3f} "
            f"runtime_score={float(row['total_runtime_score']):.3f}"
        )

    for circuit in sorted(frame["original_circuit"].unique()):
        obs, _ = env.reset(options={"circuit": circuit})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        episode_reward = float(reward)

        selected_chain = info["selected_chain"]
        selected_row = row_for_selected_chain(circuit, selected_chain)
        print_step(
            circuit,
            1,
            action,
            selected_chain,
            reward,
            selected_row,
            terminated,
            truncated,
            info.get("invalid_action", False),
        )

        if not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)

            selected_chain = info["selected_chain"]
            if not info["invalid_action"] and selected_chain is not None and "__" in selected_chain:
                selected_row = frame[
                    (frame["original_circuit"] == circuit)
                    & (frame["chain_name"] == selected_chain)
                ].iloc[0].copy()
            else:
                selected_row = row_for_selected_chain(circuit, selected_chain)
            print_step(
                circuit,
                2,
                action,
                selected_chain,
                reward,
                selected_row,
                terminated,
                truncated,
                info.get("invalid_action", False),
            )

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
        "avg_depth_reduction_pct": float(frame["total_depth_reduction_pct"].mean()),
        "avg_runtime_seconds": avg_runtime_seconds,
        "avg_runtime_score": float(
            np.clip(np.log1p(max(avg_runtime_seconds, 0.0)) / runtime_denominator, 0, 2)
        ),
    }


def append_training_log(
    algorithm_name,
    algorithm_seed,
    feature_batch,
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
        "algorithm_seed": int(algorithm_seed),
        "feature_batch": feature_batch,
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
        try:
            existing = pd.read_csv(TRAINING_LOG_PATH)
        except EmptyDataError:
            existing = pd.DataFrame(columns=TRAINING_LOG_COLUMNS)

        if existing.empty:
            combined = row_frame
        else:
            existing = existing.reindex(columns=TRAINING_LOG_COLUMNS)
            existing["algorithm_seed"] = existing["algorithm_seed"].fillna(42).astype(int)
            existing["feature_batch"] = existing["feature_batch"].fillna("original")
            existing["two_qubit_weight"] = existing["two_qubit_weight"].fillna(0.5)
            existing["depth_weight"] = existing["depth_weight"].fillna(0.5)
            existing["final_reward_weight"] = existing["final_reward_weight"].fillna(0.5)
            combined = pd.concat([existing, row_frame], ignore_index=True)
    else:
        combined = row_frame

    combined.to_csv(TRAINING_LOG_PATH, index=False)
