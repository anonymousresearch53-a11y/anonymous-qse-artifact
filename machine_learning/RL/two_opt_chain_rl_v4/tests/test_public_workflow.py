from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (
    METRIC_NAMES,
    add_derived_columns,
    add_runtime_score,
    compute_metric_scale,
    compute_observation_feature_scale,
    feature_batch_config,
    filter_supported_optimizer_rows,
    split_data,
    supported_chain_name,
)
from machine_learning.RL.two_opt_chain_rl_v4.environment import OptimizerChainEnv
from machine_learning.RL.two_opt_chain_rl_v4.evaluation import (
    evaluate_model_on_frame,
    original_selection_row,
    summarize_selected_frame,
)
from machine_learning.RL.two_opt_chain_rl_v4.rewards import (
    CURRENT_REWARD,
    DEPTH_ONLY,
    LEXICOGRAPHIC,
    MIXED_EQUAL,
    MIXED_WEIGHTED,
    NO_RUNTIME,
    TWO_QUBIT_ONLY,
    apply_reward,
    compute_final_correction,
    compute_step_reward,
)


OPTIMIZERS = ["opt_a", "opt_b"]


def _raw_result_row(
    chain_name: str,
    *,
    final_two_qubit_gates: float,
    final_depth: float,
    current_step_runtime: float = 1.0,
) -> dict[str, object]:
    return {
        "original_circuit": "circuit_1",
        "original_category": "category_a",
        "chain_name": chain_name,
        "parent_circuit": None,
        "opt_1": chain_name.split("__")[0],
        "initial_num_qubits": 4.0,
        "initial_depth": 100.0,
        "initial_two_qubit_gates": 10.0,
        "initial_total_gates": 40.0,
        "initial_two_qubit_depth": 8.0,
        "initial_gate_density": 10.0,
        "initial_two_qubit_ratio": 0.25,
        "final_num_qubits": 4.0,
        "final_depth": final_depth,
        "final_two_qubit_gates": final_two_qubit_gates,
        "final_total_gates": 30.0,
        "final_two_qubit_depth": 6.0,
        "total_runtime_seconds": current_step_runtime,
        "current_step_runtime": current_step_runtime,
    }


def _environment_frame(circuits=("circuit_1",)) -> pd.DataFrame:
    rows = []
    outcomes = {
        "opt_a": (8.0, 80.0, 1.0, 0.1, 0.2, 0.2),
        "opt_b": (9.0, 90.0, 1.5, 0.15, 0.1, 0.1),
        "opt_a__opt_b": (6.0, 60.0, 3.0, 0.2, 0.25, 0.25),
    }
    for circuit in circuits:
        for chain_name, (
            final_two_qubit,
            final_depth,
            total_runtime,
            runtime_score,
            step_two_qubit_reduction,
            step_depth_reduction,
        ) in outcomes.items():
            rows.append(
                {
                    "original_circuit": circuit,
                    "original_category": "category_a",
                    "chain_name": chain_name,
                    "initial_num_qubits": 4.0,
                    "initial_depth": 100.0,
                    "initial_two_qubit_gates": 10.0,
                    "initial_total_gates": 40.0,
                    "initial_two_qubit_depth": 8.0,
                    "initial_gate_density": 10.0,
                    "initial_two_qubit_ratio": 0.25,
                    "final_num_qubits": 4.0,
                    "final_depth": final_depth,
                    "final_two_qubit_gates": final_two_qubit,
                    "final_total_gates": 30.0,
                    "final_two_qubit_depth": 6.0,
                    "current_step_two_qubit_gate_reduction_pct": (
                        100.0 * step_two_qubit_reduction
                    ),
                    "current_step_depth_reduction_pct": 100.0 * step_depth_reduction,
                    "total_two_qubit_gate_reduction_pct": 100.0
                    * (10.0 - final_two_qubit)
                    / 10.0,
                    "total_depth_reduction_pct": 100.0
                    * (100.0 - final_depth)
                    / 100.0,
                    "total_runtime_seconds": total_runtime,
                    "runtime_score": runtime_score,
                    "total_runtime_score": 0.4 if "__" in chain_name else runtime_score,
                    "runtime_denominator": 2.0,
                }
            )
    return pd.DataFrame(rows)


def _build_env(frame: pd.DataFrame, **overrides) -> OptimizerChainEnv:
    config = {
        "frame": frame,
        "optimizer_names": OPTIMIZERS,
        "category_names": ["category_a"],
        "metric_scale": {name: 100.0 for name in METRIC_NAMES},
        "observation_feature_scale": {},
        "static_feature_columns": [],
        "dynamic_feature_columns": [],
        "include_category": True,
        "shuffle_each_epoch": False,
        "reward_type": MIXED_WEIGHTED,
        "runtime_weight": 0.1,
        "two_qubit_weight": 0.6,
        "depth_weight": 0.4,
        "final_reward_weight": 0.5,
    }
    config.update(overrides)
    return OptimizerChainEnv(**config)


class _ActionModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.deterministic_values = []

    def predict(self, observation, deterministic):
        assert observation.ndim == 1
        self.deterministic_values.append(deterministic)
        return next(self.actions), None


def test_split_data_keeps_every_original_circuit_in_exactly_one_split():
    rows = []
    for category, circuit_count in (("large", 12), ("small", 3)):
        for circuit_index in range(circuit_count):
            circuit = f"{category}_{circuit_index:02d}"
            for chain_name in ("opt_a", "opt_b", "opt_a__opt_b"):
                rows.append(
                    {
                        "original_circuit": circuit,
                        "original_category": category,
                        "chain_name": chain_name,
                    }
                )
    frame = pd.DataFrame(rows)

    splits = split_data(frame, random_state=17)
    repeated_splits = split_data(frame, random_state=17)
    circuit_sets = {
        name: set(split["original_circuit"])
        for name, split in splits.items()
    }

    assert circuit_sets["train"].isdisjoint(circuit_sets["validation"])
    assert circuit_sets["train"].isdisjoint(circuit_sets["test"])
    assert circuit_sets["validation"].isdisjoint(circuit_sets["test"])
    assert set().union(*circuit_sets.values()) == set(frame["original_circuit"])
    assert sum(len(split) for split in splits.values()) == len(frame)
    assert circuit_sets == {
        name: set(split["original_circuit"])
        for name, split in repeated_splits.items()
    }

    large_counts = {
        name: split.loc[split["original_category"] == "large", "original_circuit"].nunique()
        for name, split in splits.items()
    }
    small_counts = {
        name: split.loc[split["original_category"] == "small", "original_circuit"].nunique()
        for name, split in splits.items()
    }
    assert large_counts == {"train": 10, "validation": 1, "test": 1}
    assert small_counts == {"train": 1, "validation": 1, "test": 1}


def test_add_derived_columns_calculates_total_current_step_and_alias_columns():
    frame = pd.DataFrame(
        [
            _raw_result_row("opt_a", final_two_qubit_gates=8.0, final_depth=80.0),
            _raw_result_row(
                "opt_a__opt_b",
                final_two_qubit_gates=6.0,
                final_depth=60.0,
                current_step_runtime=2.0,
            ),
        ]
    )

    derived = add_derived_columns(frame).set_index("chain_name")
    single = derived.loc["opt_a"]
    chain = derived.loc["opt_a__opt_b"]

    assert single["total_two_qubit_gate_reduction"] == pytest.approx(2.0)
    assert single["total_two_qubit_gate_reduction_pct"] == pytest.approx(20.0)
    assert single["total_depth_reduction"] == pytest.approx(20.0)
    assert single["total_depth_reduction_pct"] == pytest.approx(20.0)
    assert single["current_step_parent_two_qubit_gates"] == pytest.approx(10.0)
    assert single["current_step_two_qubit_gate_reduction"] == pytest.approx(2.0)
    assert single["current_step_two_qubit_gate_reduction_pct"] == pytest.approx(20.0)
    assert single["current_step_parent_depth"] == pytest.approx(100.0)
    assert single["current_step_depth_reduction"] == pytest.approx(20.0)
    assert single["current_step_depth_reduction_pct"] == pytest.approx(20.0)

    assert chain["parent_circuit"] == "circuit_1"
    assert chain["total_two_qubit_gate_reduction"] == pytest.approx(4.0)
    assert chain["total_two_qubit_gate_reduction_pct"] == pytest.approx(40.0)
    assert chain["total_depth_reduction"] == pytest.approx(40.0)
    assert chain["total_depth_reduction_pct"] == pytest.approx(40.0)
    assert chain["current_step_parent_two_qubit_gates"] == pytest.approx(8.0)
    assert chain["current_step_two_qubit_gate_reduction"] == pytest.approx(2.0)
    assert chain["current_step_two_qubit_gate_reduction_pct"] == pytest.approx(25.0)
    assert chain["current_step_parent_depth"] == pytest.approx(80.0)
    assert chain["current_step_depth_reduction"] == pytest.approx(20.0)
    assert chain["current_step_depth_reduction_pct"] == pytest.approx(25.0)

    for alias, source in (
        ("two_qubit_gate_reduction", "total_two_qubit_gate_reduction"),
        ("two_qubit_gate_reduction_pct", "total_two_qubit_gate_reduction_pct"),
        ("depth_reduction", "total_depth_reduction"),
        ("depth_reduction_pct", "total_depth_reduction_pct"),
    ):
        pd.testing.assert_series_equal(derived[alias], derived[source], check_names=False)


def test_derived_current_step_percentages_are_zero_when_parent_is_zero():
    single = _raw_result_row("opt_a", final_two_qubit_gates=0.0, final_depth=0.0)
    chain = _raw_result_row("opt_a__opt_b", final_two_qubit_gates=0.0, final_depth=0.0)

    derived = add_derived_columns(pd.DataFrame([single, chain])).set_index("chain_name")

    assert derived.loc["opt_a__opt_b", "current_step_two_qubit_gate_reduction_pct"] == 0.0
    assert derived.loc["opt_a__opt_b", "current_step_depth_reduction_pct"] == 0.0


def test_runtime_and_observation_scales_use_expected_formulas():
    frame = _environment_frame().copy()
    frame["current_step_runtime"] = [-2.0, np.expm1(2.0), np.expm1(10.0)]
    frame["initial_custom"] = [2.0, 4.0, 6.0]
    frame["custom"] = [4.0, 8.0, 12.0]
    frame["progress"] = [-2.0, 4.0, -8.0]

    scored = add_runtime_score(frame, runtime_denominator=2.0)
    assert scored["runtime_score"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert scored["runtime_denominator"].eq(2.0).all()

    metric_scale = compute_metric_scale(frame)
    assert set(metric_scale) == set(METRIC_NAMES)
    assert all(value > 0 for value in metric_scale.values())

    feature_scale = compute_observation_feature_scale(
        frame,
        static_feature_columns=["custom", "missing"],
        dynamic_feature_columns=["progress", "missing_progress"],
    )
    combined_custom = pd.concat([frame["initial_custom"], frame["custom"]], ignore_index=True)
    assert feature_scale["custom"] == pytest.approx(combined_custom.quantile(0.95))
    assert feature_scale["progress"] == pytest.approx(frame["progress"].abs().quantile(0.95))
    assert feature_scale["missing"] == 1.0
    assert feature_scale["missing_progress"] == 1.0


@pytest.mark.parametrize(
    ("reward_type", "expected_step", "expected_final"),
    [
        (CURRENT_REWARD, 0.37, 0.20),
        (TWO_QUBIT_ONLY, 0.37, 0.20),
        (DEPTH_ONLY, 0.17, 0.12),
        (MIXED_WEIGHTED, 0.32, 0.18),
        (MIXED_EQUAL, 0.27, 0.16),
        (LEXICOGRAPHIC, 0.42, 0.23),
        (NO_RUNTIME, 0.40, 0.20),
    ],
)
def test_each_reward_type_uses_the_expected_formula(
    reward_type, expected_step, expected_final
):
    row = {
        "current_step_two_qubit_gate_reduction_pct": 40.0,
        "current_step_depth_reduction_pct": 20.0,
        "runtime_score": 0.3,
        "total_two_qubit_gate_reduction_pct": 50.0,
        "total_depth_reduction_pct": 30.0,
    }

    step_reward = compute_step_reward(
        row,
        reward_type=reward_type,
        runtime_weight=0.1,
        two_qubit_weight=0.75,
        depth_weight=0.25,
    )
    final_reward = compute_final_correction(
        row,
        reward_type=reward_type,
        final_reward_weight=0.4,
        two_qubit_weight=0.75,
        depth_weight=0.25,
    )

    assert step_reward == pytest.approx(expected_step)
    assert final_reward == pytest.approx(expected_final)


def test_reward_helpers_add_columns_without_mutating_input_and_reject_unknown_type():
    frame = pd.DataFrame(
        [
            {
                "current_step_two_qubit_gate_reduction_pct": 40.0,
                "current_step_depth_reduction_pct": 20.0,
                "runtime_score": 0.3,
            }
        ]
    )

    rewarded = apply_reward(frame, reward_type=MIXED_EQUAL, runtime_weight=0.1)

    assert list(frame.columns) == [
        "current_step_two_qubit_gate_reduction_pct",
        "current_step_depth_reduction_pct",
        "runtime_score",
    ]
    assert rewarded.loc[0, "two_qubit_score"] == pytest.approx(0.4)
    assert rewarded.loc[0, "depth_score"] == pytest.approx(0.2)
    assert rewarded.loc[0, "local_step_reward"] == pytest.approx(0.27)
    assert rewarded.loc[0, "utility_reward"] == pytest.approx(0.27)

    with pytest.raises(ValueError, match="Unsupported reward_type"):
        compute_step_reward(frame.iloc[0], reward_type="unknown")


def test_environment_returns_expected_rewards_for_two_optimizer_chain():
    env = _build_env(_environment_frame())

    initial_observation, info = env.reset(options={"circuit": "circuit_1"})
    next_observation, first_reward, terminated, truncated, first_info = env.step(0)

    assert info == {"circuit": "circuit_1"}
    assert initial_observation.shape == env.observation_space.shape
    assert np.isfinite(initial_observation).all()
    assert next_observation.shape == env.observation_space.shape
    assert first_reward == pytest.approx(0.19)
    assert not terminated
    assert not truncated
    assert first_info["selected_chain"] == "opt_a"

    _, second_reward, terminated, truncated, second_info = env.step(1)

    assert second_reward == pytest.approx(0.43)
    assert terminated
    assert not truncated
    assert second_info == {
        "invalid_action": False,
        "selected_chain": "opt_a__opt_b",
        "circuit": "circuit_1",
    }


def test_environment_stop_and_missing_chain_paths_return_expected_rewards():
    frame = _environment_frame()
    env = _build_env(frame, invalid_stop_penalty=-0.75)

    env.reset(options={"circuit": "circuit_1"})
    _, reward, terminated, _, info = env.step(env.stop_action)
    assert reward == 0.0
    assert terminated
    assert info["selected_chain"] == "__original__"

    env.reset(options={"circuit": "circuit_1"})
    env.step(0)
    _, reward, terminated, _, info = env.step(env.stop_action)
    assert reward == pytest.approx(0.10)
    assert terminated
    assert info["selected_chain"] == "opt_a"

    env = _build_env(frame[frame["chain_name"] != "opt_a__opt_b"], invalid_stop_penalty=-0.75)
    env.reset(options={"circuit": "circuit_1"})
    env.step(0)
    _, reward, terminated, _, info = env.step(1)
    assert reward == -0.75
    assert terminated
    assert info["invalid_action"] is True
    assert info["selected_chain"] == "opt_a"


def test_evaluate_model_on_frame_tracks_selected_chain_episode_reward_and_runtime():
    frame = _environment_frame(("circuit_1", "circuit_2"))
    model = _ActionModel([0, 1, 0, 1])

    selected = evaluate_model_on_frame(
        model=model,
        frame=frame,
        build_env_fn=OptimizerChainEnv,
        optimizer_names=OPTIMIZERS,
        category_names=["category_a"],
        metric_scale={name: 100.0 for name in METRIC_NAMES},
        observation_feature_scale={},
        static_feature_columns=[],
        dynamic_feature_columns=[],
        include_category=True,
        method_name="test_model",
        reward_type=MIXED_WEIGHTED,
        runtime_weight=0.1,
        two_qubit_weight=0.6,
        depth_weight=0.4,
        final_reward_weight=0.5,
    )

    assert selected["original_circuit"].tolist() == ["circuit_1", "circuit_2"]
    assert selected["selected_chain"].tolist() == ["opt_a__opt_b", "opt_a__opt_b"]
    assert selected["method"].eq("test_model").all()
    assert selected["episode_reward"].tolist() == pytest.approx([0.62, 0.62])
    assert selected["episode_total_runtime_seconds"].tolist() == [3.0, 3.0]
    assert selected["episode_runtime_score"].tolist() == [0.4, 0.4]
    assert model.deterministic_values == [True, True, True, True]

    summary = summarize_selected_frame(selected)
    assert summary["utility"] == pytest.approx(0.62)
    assert summary["avg_two_qubit_gate_reduction_pct"] == pytest.approx(40.0)
    assert summary["avg_depth_reduction_pct"] == pytest.approx(40.0)
    assert summary["avg_runtime_seconds"] == pytest.approx(3.0)
    assert summary["avg_runtime_score"] == pytest.approx(np.log1p(3.0) / 2.0)


def test_evaluation_immediate_stop_returns_original_circuit_metrics():
    frame = _environment_frame()
    model = _ActionModel([len(OPTIMIZERS)])

    selected = evaluate_model_on_frame(
        model=model,
        frame=frame,
        build_env_fn=OptimizerChainEnv,
        optimizer_names=OPTIMIZERS,
        category_names=["category_a"],
        metric_scale={name: 100.0 for name in METRIC_NAMES},
        observation_feature_scale={},
        static_feature_columns=[],
        dynamic_feature_columns=[],
        include_category=True,
        method_name="stop_model",
        reward_type=CURRENT_REWARD,
        runtime_weight=0.1,
        two_qubit_weight=0.5,
        depth_weight=0.5,
        final_reward_weight=0.5,
    )

    row = selected.iloc[0]
    assert row["selected_chain"] == "__original__"
    assert row["final_depth"] == row["initial_depth"]
    assert row["final_two_qubit_gates"] == row["initial_two_qubit_gates"]
    assert row["episode_reward"] == 0.0
    assert row["episode_total_runtime_seconds"] == 0.0


def test_chain_filtering_feature_batches_and_original_row_helpers():
    assert supported_chain_name("qiskit_standard")
    assert supported_chain_name("qiskit_standard__tket")
    assert not supported_chain_name("qiskit_standard__tket__wisq_rules")
    assert not supported_chain_name("unsupported")

    frame = pd.DataFrame(
        {
            "chain_name": ["qiskit_standard", "qiskit_standard__tket", "unsupported"]
        }
    )
    filtered = filter_supported_optimizer_rows(frame)
    assert filtered["chain_name"].tolist() == ["qiskit_standard", "qiskit_standard__tket"]
    assert filtered.attrs["dropped_unsupported_optimizer_rows"] == 1

    config = feature_batch_config("plus_all_no_history")
    assert config == feature_batch_config("plus_all_features")
    assert config["include_category"] is False
    with pytest.raises(ValueError, match="Unsupported feature_batch"):
        feature_batch_config("unknown")

    original = original_selection_row(_environment_frame(), "circuit_1")
    assert original["chain_name"] == "__original__"
    assert original["final_depth"] == original["initial_depth"]
    assert original["total_runtime_seconds"] == 0.0
    assert original["total_two_qubit_gate_reduction_pct"] == 0.0
