"""End-to-end audits for the v4 RL workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (  # noqa: E402
    OPTIMIZER_NAMES,
)
from machine_learning.RL.two_opt_chain_rl_v4.environment import (  # noqa: E402
    METRIC_NAMES,
    UNKNOWN_CATEGORY,
    OptimizerChainEnv,
)
from machine_learning.RL.two_opt_chain_rl_v4.rewards import (  # noqa: E402
    SUPPORTED_REWARD_TYPES,
)
from machine_learning.RL.two_opt_chain_rl_v4.tests.workflow_audit import (  # noqa: E402
    V4_ROOT,
    assert_reward_parity,
    load_bundle,
    per_circuit_comparison_summary,
    seed_uncertainty_summary,
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(feature_batch="original")


def test_reward_matches_independent_calculation_for_every_test_choice(bundle):
    for reward_type in SUPPORTED_REWARD_TYPES:
        assert_reward_parity(bundle, reward_type=reward_type)


def test_test_split_is_circuit_disjoint_and_complete(bundle):
    train_circuits = set(bundle["train_df"]["original_circuit"])
    validation_circuits = set(bundle["validation_df"]["original_circuit"])
    test_circuits = set(bundle["test_df"]["original_circuit"])

    assert train_circuits.isdisjoint(validation_circuits)
    assert train_circuits.isdisjoint(test_circuits)
    assert validation_circuits.isdisjoint(test_circuits)

    for split_name, frame in (
        ("train", bundle["train_df"]),
        ("validation", bundle["validation_df"]),
        ("test", bundle["test_df"]),
    ):
        counts = frame.groupby("original_circuit")["chain_name"].nunique()
        assert counts.size > 0, split_name
        assert counts.eq(20).all(), split_name


def test_seed_uncertainty_is_available_for_matched_runs():
    path = V4_ROOT / "grid_search_runs_features.csv"
    if not path.exists():
        pytest.skip("No feature sweep artifact is available.")

    summary = seed_uncertainty_summary(path, feature_batch="original")
    assert summary["n_seeds"] >= 3
    assert np.isfinite(summary["mean_test_utility"])
    assert np.isfinite(summary["std_test_utility"])
    assert np.isfinite(summary["approx_95pct_half_width"])
    print(f"seed uncertainty: {summary}")


def test_per_circuit_baseline_artifacts_report_wins_losses_and_ties():
    paths = sorted((V4_ROOT / "model_evaluator_figures").glob("per_circuit_model_vs_*.csv"))
    if not paths:
        pytest.skip("No per-circuit comparison artifacts are available.")

    for path in paths:
        summary = per_circuit_comparison_summary(path)
        assert summary["circuits"] > 0
        assert summary["wins"] + summary["losses"] + summary["ties"] == summary["circuits"]
        assert np.isfinite(summary["mean_utility_difference"])
        print(f"{path.name}: {summary}")


def test_category_vocabulary_comes_from_training_and_has_unknown_bucket(bundle):
    training_categories = sorted(bundle["train_df"]["original_category"].dropna().unique())
    assert bundle["category_names"] == training_categories + [UNKNOWN_CATEGORY]


def test_unseen_category_maps_to_explicit_unknown_bucket(bundle):
    circuit = bundle["test_df"]["original_circuit"].iloc[0]
    frame = bundle["test_df"]
    unseen_frame = frame[frame["original_circuit"] == circuit].copy()
    unseen_frame["original_category"] = "category_seen_only_in_test"

    env = OptimizerChainEnv(
        frame=unseen_frame,
        optimizer_names=OPTIMIZER_NAMES,
        category_names=["category_seen_in_train"],
        metric_scale=bundle["metric_scale"],
        observation_feature_scale={},
        static_feature_columns=[],
        dynamic_feature_columns=[],
        include_category=True,
        shuffle_each_epoch=False,
    )
    observation, _ = env.reset(options={"circuit": circuit})
    category_start = len(METRIC_NAMES)
    assert env.current_category == UNKNOWN_CATEGORY
    assert observation[category_start : category_start + 2].tolist() == [0.0, 1.0]


class _AuditEnv(gym.Env):
    """Minimal environment accepted by Stable-Baselines' Monitor wrapper."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class _AuditModel:
    def learn(self, total_timesteps):
        return self


def test_hyperparameter_search_uses_validation_only_for_selection():
    pytest.importorskip("stable_baselines3")
    from machine_learning.RL.two_opt_chain_rl_v4.hyperparameter_search import run_grid_search

    train_frame = pd.DataFrame({"split": ["train"]})
    validation_frame = pd.DataFrame({"split": ["validation"]})
    evaluation_frames = []

    def build_env_fn(frame, *args, **kwargs):
        return _AuditEnv()

    def build_model_fn(algorithm_name, env, config):
        return _AuditModel()

    def evaluate_fn(model, frame, *args, **kwargs):
        evaluation_frames.append(frame)
        return pd.DataFrame(
            {
                "episode_reward": [1.0],
                "episode_total_runtime_seconds": [1.0],
                "total_two_qubit_gate_reduction_pct": [1.0],
                "selected_chain": ["qiskit_standard"],
            }
        )

    result = run_grid_search(
        algorithm_name="ppo",
        base_config={"policy": "MlpPolicy"},
        search_grid={"learning_rate": [0.001, 0.002]},
        train_df=train_frame,
        validation_df=validation_frame,
        optimizer_names=OPTIMIZER_NAMES,
        category_names=[],
        metric_scale={name: 1.0 for name in METRIC_NAMES},
        observation_feature_scale={},
        static_feature_columns=[],
        dynamic_feature_columns=[],
        include_category=False,
        build_env_fn=build_env_fn,
        build_model_fn=build_model_fn,
        evaluate_model_on_frame_fn=evaluate_fn,
        tuning_timesteps=1,
        reward_type="current_reward",
        runtime_weight=0.1,
        two_qubit_weight=0.5,
        depth_weight=0.5,
        final_reward_weight=0.5,
    )

    assert len(result) == 2
    assert evaluation_frames
    assert all(frame is validation_frame for frame in evaluation_frames)
    assert all(frame is not train_frame for frame in evaluation_frames)


def test_train_model_search_call_does_not_receive_test_data():
    train_source = REPO_ROOT / "machine_learning/RL/two_opt_chain_rl_v4/train.py"
    source = train_source.read_text()
    function_source = source[source.index("def train_model(") :]
    search_start = function_source.index("run_grid_search(")
    final_model_start = function_source.index("model = build_model", search_start)
    search_block = function_source[search_start:final_model_start]
    assert "test_df" not in search_block
