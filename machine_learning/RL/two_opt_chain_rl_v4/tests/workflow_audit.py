"""Reusable, data-backed audits for the v4 RL workflow.

The functions in this module intentionally keep the independent calculations
separate from ``rewards.py``.  They are used by pytest and can also be run as
an end-to-end audit report with:

    python machine_learning/RL/two_opt_chain_rl_v4/tests/workflow_audit.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (  # noqa: E402
    OPTIMIZER_NAMES,
    prepare_datasets,
)
from machine_learning.RL.two_opt_chain_rl_v4.environment import (  # noqa: E402
    METRIC_NAMES,
    UNKNOWN_CATEGORY,
    OptimizerChainEnv,
)
from machine_learning.RL.two_opt_chain_rl_v4.rewards import (  # noqa: E402
    CURRENT_REWARD,
    DEPTH_ONLY,
    LEXICOGRAPHIC,
    MIXED_EQUAL,
    MIXED_WEIGHTED,
    NO_RUNTIME,
    SUPPORTED_REWARD_TYPES,
    TWO_QUBIT_ONLY,
)

V4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_CSV = REPO_ROOT / "dbs" / "combined_master.csv"


def load_bundle(*, feature_batch: str = "original") -> dict[str, Any]:
    """Load the real v4 dataset without changing cached split artifacts."""
    return prepare_datasets(
        csv_path=DEFAULT_MASTER_CSV,
        save=False,
        feature_batch=feature_batch,
    )


def build_environment(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    reward_type: str = MIXED_WEIGHTED,
) -> OptimizerChainEnv:
    """Build the same environment configuration used by training/evaluation."""
    return OptimizerChainEnv(
        frame=frame,
        optimizer_names=bundle["optimizer_names"],
        category_names=bundle["category_names"],
        metric_scale=bundle["metric_scale"],
        observation_feature_scale=bundle["observation_feature_scale"],
        static_feature_columns=bundle["static_feature_columns"],
        dynamic_feature_columns=bundle["dynamic_feature_columns"],
        include_category=bundle["feature_batch_config"]["include_category"],
        shuffle_each_epoch=False,
        reward_type=reward_type,
        runtime_weight=0.1,
        two_qubit_weight=0.5,
        depth_weight=0.5,
        final_reward_weight=0.5,
    )


def _value(row: pd.Series, name: str) -> float:
    return float(row[name])


def manual_step_reward(
    row: pd.Series,
    *,
    reward_type: str,
    runtime_weight: float = 0.1,
    two_qubit_weight: float = 0.5,
    depth_weight: float = 0.5,
) -> float:
    """Independently calculate one local reward from stored columns."""
    two_qubit = _value(row, "current_step_two_qubit_gate_reduction_pct") / 100.0
    depth = _value(row, "current_step_depth_reduction_pct") / 100.0
    runtime = _value(row, "runtime_score")

    if reward_type in {CURRENT_REWARD, TWO_QUBIT_ONLY}:
        return two_qubit - runtime_weight * runtime
    if reward_type == DEPTH_ONLY:
        return depth - runtime_weight * runtime
    if reward_type == MIXED_WEIGHTED:
        return two_qubit_weight * two_qubit + depth_weight * depth - runtime_weight * runtime
    if reward_type == MIXED_EQUAL:
        return 0.5 * two_qubit + 0.5 * depth - runtime_weight * runtime
    if reward_type == LEXICOGRAPHIC:
        return two_qubit + depth_weight * depth - runtime_weight * runtime
    if reward_type == NO_RUNTIME:
        return two_qubit
    raise ValueError(f"Unsupported reward type: {reward_type}")


def manual_final_correction(
    row: pd.Series,
    *,
    reward_type: str,
    final_reward_weight: float = 0.5,
    two_qubit_weight: float = 0.5,
    depth_weight: float = 0.5,
) -> float:
    """Independently calculate the terminal correction."""
    two_qubit = _value(row, "total_two_qubit_gate_reduction_pct") / 100.0
    depth = _value(row, "total_depth_reduction_pct") / 100.0

    if reward_type in {CURRENT_REWARD, TWO_QUBIT_ONLY, NO_RUNTIME}:
        return final_reward_weight * two_qubit
    if reward_type == DEPTH_ONLY:
        return final_reward_weight * depth
    if reward_type == MIXED_WEIGHTED:
        return final_reward_weight * (two_qubit_weight * two_qubit + depth_weight * depth)
    if reward_type == MIXED_EQUAL:
        return final_reward_weight * (0.5 * two_qubit + 0.5 * depth)
    if reward_type == LEXICOGRAPHIC:
        return final_reward_weight * (two_qubit + depth_weight * depth)
    raise ValueError(f"Unsupported reward type: {reward_type}")


def manual_episode_reward(
    circuit_frame: pd.DataFrame,
    chain_name: str,
    *,
    reward_type: str,
    runtime_weight: float = 0.1,
    two_qubit_weight: float = 0.5,
    depth_weight: float = 0.5,
    final_reward_weight: float = 0.5,
) -> float:
    """Calculate the complete episode reward for one available choice."""
    if chain_name == "__original__":
        return 0.0

    rows = circuit_frame.set_index("chain_name")
    if "__" not in chain_name:
        row = rows.loc[chain_name]
        return manual_step_reward(
            row,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        ) + manual_final_correction(
            row,
            reward_type=reward_type,
            final_reward_weight=final_reward_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )

    first_optimizer, _ = chain_name.split("__", maxsplit=1)
    first_row = rows.loc[first_optimizer]
    chain_row = rows.loc[chain_name]
    return (
        manual_step_reward(
            first_row,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )
        + manual_step_reward(
            chain_row,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )
        + manual_final_correction(
            chain_row,
            reward_type=reward_type,
            final_reward_weight=final_reward_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )
    )


def assert_reward_parity(
    bundle: dict[str, Any],
    *,
    reward_type: str,
    atol: float = 1e-10,
) -> int:
    """Check every test-circuit choice against the independent calculation.

    There are 21 choices per circuit: immediate stop, four single optimizers,
    and sixteen ordered two-optimizer chains.
    """
    frame = bundle["test_df"]
    env = build_environment(bundle, frame, reward_type=reward_type)
    checked = 0

    for circuit, circuit_frame in frame.groupby("original_circuit", sort=True):
        expected_original = manual_episode_reward(
            circuit_frame,
            "__original__",
            reward_type=reward_type,
        )
        env.reset(options={"circuit": circuit})
        _, actual, terminated, truncated, _ = env.step(env.stop_action)
        assert terminated and not truncated
        assert math.isclose(actual, expected_original, abs_tol=atol)
        checked += 1

        for first_index, first_optimizer in enumerate(OPTIMIZER_NAMES):
            env.reset(options={"circuit": circuit})
            _, first_reward, terminated, truncated, _ = env.step(first_index)
            assert not terminated and not truncated

            _, stop_reward, terminated, truncated, _ = env.step(env.stop_action)
            assert terminated and not truncated
            expected_single = manual_episode_reward(
                circuit_frame,
                first_optimizer,
                reward_type=reward_type,
            )
            single_row = circuit_frame.set_index("chain_name").loc[first_optimizer]
            actual_single = float(first_reward) + float(stop_reward)
            assert math.isclose(
                float(first_reward),
                manual_step_reward(
                    single_row,
                    reward_type=reward_type,
                ),
                abs_tol=atol,
            )
            assert math.isclose(
                float(stop_reward),
                manual_final_correction(single_row, reward_type=reward_type),
                abs_tol=atol,
            )
            assert math.isclose(actual_single, expected_single, abs_tol=atol)
            checked += 1

            for second_index, second_optimizer in enumerate(OPTIMIZER_NAMES):
                chain_name = f"{first_optimizer}__{second_optimizer}"
                env.reset(options={"circuit": circuit})
                _, first_reward, terminated, truncated, _ = env.step(first_index)
                assert not terminated and not truncated
                _, second_reward, terminated, truncated, _ = env.step(second_index)
                assert terminated and not truncated
                actual_chain = float(first_reward) + float(second_reward)
                expected_chain = manual_episode_reward(
                    circuit_frame,
                    chain_name,
                    reward_type=reward_type,
                )
                assert math.isclose(actual_chain, expected_chain, abs_tol=atol), (
                    circuit,
                    chain_name,
                    actual_chain,
                    expected_chain,
                )
                checked += 1

    return checked


def seed_uncertainty_summary(
    path: Path,
    *,
    feature_batch: str = "original",
) -> dict[str, float | int | list[int]]:
    """Summarize held-out utility across matched algorithm seeds."""
    frame = pd.read_csv(path)
    required = {"status", "algorithm_seed", "feature_batch", "test_utility"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Seed artifact is missing columns: {sorted(missing)}")

    matched = frame[
        (frame["status"] == "success")
        & (frame["feature_batch"] == feature_batch)
    ].copy()
    matched = matched.drop_duplicates(subset=["algorithm_seed"], keep="last")
    seeds = sorted(int(seed) for seed in matched["algorithm_seed"].dropna().unique())
    if len(seeds) < 2:
        raise ValueError(f"Expected at least two matched seeds, found {seeds}")

    scores = matched["test_utility"].astype(float).to_numpy()
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    standard_error = std / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean_test_utility": mean,
        "std_test_utility": std,
        "approx_95pct_half_width": 1.96 * standard_error,
        "min_test_utility": float(scores.min()),
        "max_test_utility": float(scores.max()),
    }


def per_circuit_comparison_summary(path: Path) -> dict[str, int | float]:
    """Summarize wins, losses, ties, and utility differences per circuit."""
    frame = pd.read_csv(path)
    required = {"original_circuit", "utility_diff"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Comparison artifact is missing columns: {sorted(missing)}")

    differences = frame["utility_diff"].astype(float)
    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    return {
        "circuits": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_utility_difference": float(differences.mean()),
    }


def _print_report(bundle: dict[str, Any]) -> None:
    print("v4 split summary")
    print(bundle["split_summary"].to_string(index=False))

    print("\nreward parity")
    for reward_type in SUPPORTED_REWARD_TYPES:
        checked = assert_reward_parity(bundle, reward_type=reward_type)
        print(f"  {reward_type}: {checked} choices checked")

    seed_path = V4_ROOT / "grid_search_runs_features.csv"
    if seed_path.exists():
        summary = seed_uncertainty_summary(seed_path, feature_batch="original")
        print("\nseed uncertainty")
        print(summary)

    print("\nper-circuit comparisons")
    for path in sorted((V4_ROOT / "model_evaluator_figures").glob("per_circuit_model_vs_*.csv")):
        print(f"  {path.name}: {per_circuit_comparison_summary(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the v4 RL workflow artifacts.")
    parser.add_argument("--feature-batch", default="original")
    args = parser.parse_args()
    _print_report(load_bundle(feature_batch=args.feature_batch))


if __name__ == "__main__":
    main()
