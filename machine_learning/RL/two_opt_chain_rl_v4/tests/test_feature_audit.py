"""Feature-shape, feature-timing, and leakage audits for RL v4."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.RL.two_opt_chain_rl_v4 import data_processing  # noqa: E402
from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (  # noqa: E402
    BASELINE_FEATURE_COLUMNS,
    FEATURE_BATCH_CONFIGS,
    compute_observation_feature_scale,
    feature_batch_config,
)
from machine_learning.RL.two_opt_chain_rl_v4.environment import (  # noqa: E402
    METRIC_NAMES,
    UNKNOWN_CATEGORY,
    OptimizerChainEnv,
)
from machine_learning.RL.two_opt_chain_rl_v4.tests.workflow_audit import (  # noqa: E402
    build_environment,
    load_bundle,
)


@pytest.fixture(scope="module")
def original_bundle():
    return load_bundle(feature_batch="original")


def _build_feature_batch_environment(bundle, feature_batch):
    config = feature_batch_config(feature_batch)
    feature_scale = compute_observation_feature_scale(
        bundle["train_df"],
        static_feature_columns=config["static_feature_columns"],
        dynamic_feature_columns=config["dynamic_feature_columns"],
    )
    return OptimizerChainEnv(
        frame=bundle["test_df"],
        optimizer_names=bundle["optimizer_names"],
        category_names=bundle["category_names"],
        metric_scale=bundle["metric_scale"],
        observation_feature_scale=feature_scale,
        static_feature_columns=config["static_feature_columns"],
        dynamic_feature_columns=config["dynamic_feature_columns"],
        include_category=config["include_category"],
        shuffle_each_epoch=False,
    ), config


def _outcome_columns(frame):
    """Return columns that describe outcomes rather than available state."""
    exact = {
        "current_step_runtime",
        "total_runtime_seconds",
        "runtime_score",
        "total_runtime_score",
        "local_step_reward",
        "utility_reward",
    }
    return [
        column
        for column in frame.columns
        if column in exact
        or column.startswith("final_")
        or column.startswith("total_")
        or column.startswith("current_step_")
    ]


def _overwrite_numeric_outcomes(frame, columns, value):
    result = frame.copy(deep=True)
    for column in columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = float(value)
    return result


def test_every_feature_batch_has_expected_shape_and_finite_states(original_bundle):
    """Exercise initial and post-first-optimizer states for every feature batch."""
    circuits = sorted(original_bundle["test_df"]["original_circuit"].unique())

    for feature_batch in FEATURE_BATCH_CONFIGS:
        env, config = _build_feature_batch_environment(original_bundle, feature_batch)
        category_size = len(original_bundle["category_names"]) if config["include_category"] else 0
        expected_dim = (
            len(METRIC_NAMES)
            + len(config["static_feature_columns"])
            + len(config["dynamic_feature_columns"])
            + category_size
            + len(env.prev_optimizer_names)
            + 2
        )
        assert env.observation_space.shape == (expected_dim,), feature_batch

        for circuit in circuits:
            initial_observation, _ = env.reset(options={"circuit": circuit})
            assert initial_observation.shape == (expected_dim,)
            assert np.isfinite(initial_observation).all(), feature_batch
            assert initial_observation[-2:].tolist() == [0.0, 1.0]

            next_observation, _, terminated, truncated, _ = env.step(0)
            assert not terminated and not truncated
            assert next_observation.shape == (expected_dim,)
            assert np.isfinite(next_observation).all(), feature_batch
            assert next_observation[-2:].tolist() == [1.0, 0.0]


def test_step_zero_observation_is_invariant_to_future_outcomes(original_bundle):
    """Changing optimizer outcomes must not alter the initial state."""
    clean_frame = original_bundle["test_df"]
    mutated_frame = _overwrite_numeric_outcomes(
        clean_frame,
        _outcome_columns(clean_frame),
        value=999_999.0,
    )

    clean_env = build_environment(original_bundle, clean_frame)
    mutated_env = build_environment(original_bundle, mutated_frame)

    for circuit in sorted(clean_frame["original_circuit"].unique()):
        clean_observation, _ = clean_env.reset(options={"circuit": circuit})
        mutated_observation, _ = mutated_env.reset(options={"circuit": circuit})
        np.testing.assert_array_equal(clean_observation, mutated_observation)


def test_step_one_observation_does_not_see_second_optimizer_outcome(original_bundle):
    """The step-1 state may use the first result, not the second result."""
    clean_frame = original_bundle["test_df"]
    chain_mask = clean_frame["chain_name"].str.contains("__", regex=False)
    mutated_frame = clean_frame.copy(deep=True)
    mutated_chain_rows = _overwrite_numeric_outcomes(
        clean_frame.loc[chain_mask],
        _outcome_columns(clean_frame),
        value=888_888.0,
    )
    mutated_frame.loc[chain_mask, mutated_chain_rows.columns] = mutated_chain_rows

    clean_env = build_environment(original_bundle, clean_frame)
    mutated_env = build_environment(original_bundle, mutated_frame)

    for circuit in sorted(clean_frame["original_circuit"].unique()):
        clean_env.reset(options={"circuit": circuit})
        mutated_env.reset(options={"circuit": circuit})
        clean_observation, _, clean_terminated, clean_truncated, _ = clean_env.step(0)
        mutated_observation, _, mutated_terminated, mutated_truncated, _ = mutated_env.step(0)
        assert not clean_terminated and not clean_truncated
        assert not mutated_terminated and not mutated_truncated
        np.testing.assert_array_equal(clean_observation, mutated_observation)


def test_step_one_observation_changes_when_first_optimizer_state_changes(original_bundle):
    """The first optimizer's output is legitimate information at step 1."""
    clean_frame = original_bundle["test_df"]
    mutated_frame = clean_frame.copy(deep=True)
    circuit = sorted(clean_frame["original_circuit"].unique())[0]
    first_single_index = mutated_frame.index[
        (mutated_frame["original_circuit"] == circuit)
        & (mutated_frame["chain_name"] == original_bundle["optimizer_names"][0])
    ][0]
    mutated_frame.loc[first_single_index, "final_depth"] += 1000.0
    mutated_frame.loc[first_single_index, "final_total_gates"] += 1000.0

    clean_env = build_environment(original_bundle, clean_frame)
    mutated_env = build_environment(original_bundle, mutated_frame)
    clean_env.reset(options={"circuit": circuit})
    mutated_env.reset(options={"circuit": circuit})
    clean_observation, _, _, _, _ = clean_env.step(0)
    mutated_observation, _, _, _, _ = mutated_env.step(0)
    assert not np.array_equal(clean_observation, mutated_observation)


def test_preprocessing_statistics_are_unchanged_by_test_only_perturbations(monkeypatch):
    """Test outcomes cannot affect train-derived scales or vocabulary."""
    original_loader = data_processing.load_master_data
    raw_frame = original_loader(data_processing.default_master_csv_path())
    derived_frame = data_processing.add_derived_columns(raw_frame)
    split_frames = data_processing.split_data(derived_frame, random_state=42)
    test_circuits = set(split_frames["test"]["original_circuit"])

    mutated_raw = raw_frame.copy(deep=True)
    test_mask = mutated_raw["original_circuit"].isin(test_circuits)
    for column in (
        "final_depth",
        "final_two_qubit_gates",
        "final_total_gates",
        "final_two_qubit_depth",
        "total_runtime_seconds",
        "current_step_runtime",
    ):
        mutated_raw.loc[test_mask, column] = 1_000_000.0

    datasets = iter((raw_frame, mutated_raw))
    monkeypatch.setattr(
        data_processing,
        "load_master_data",
        lambda _csv_path: next(datasets).copy(deep=True),
    )

    clean_bundle = data_processing.prepare_datasets(save=False, feature_batch="plus_all_features")
    mutated_bundle = data_processing.prepare_datasets(save=False, feature_batch="plus_all_features")

    assert clean_bundle["runtime_reference_seconds"] == pytest.approx(
        mutated_bundle["runtime_reference_seconds"]
    )
    assert clean_bundle["runtime_denominator"] == pytest.approx(
        mutated_bundle["runtime_denominator"]
    )
    assert clean_bundle["metric_scale"] == mutated_bundle["metric_scale"]
    assert clean_bundle["observation_feature_scale"] == mutated_bundle[
        "observation_feature_scale"
    ]
    assert clean_bundle["category_names"] == mutated_bundle["category_names"]


def test_observation_feature_contract_contains_no_reward_or_test_fields(original_bundle):
    """The configured feature lists contain state features only."""
    forbidden = {
        "runtime_score",
        "total_runtime_score",
        "local_step_reward",
        "utility_reward",
        "total_two_qubit_gate_reduction_pct",
        "total_depth_reduction_pct",
        "current_step_two_qubit_gate_reduction_pct",
        "current_step_depth_reduction_pct",
    }

    configured = set(BASELINE_FEATURE_COLUMNS)
    for config in FEATURE_BATCH_CONFIGS.values():
        configured.update(config["static_feature_columns"])
        configured.update(config["dynamic_feature_columns"])

    assert configured.isdisjoint(forbidden)
    assert UNKNOWN_CATEGORY in original_bundle["category_names"]
