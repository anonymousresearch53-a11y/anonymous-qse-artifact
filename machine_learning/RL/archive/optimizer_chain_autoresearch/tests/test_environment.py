"""Gymnasium contract and transition tests for the editable environment."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from machine_learning.RL.optimizer_chain_autoresearch.optimizer_chain_env import (
    OptimizerChainEnv,
)


def _environment(data_bundle, experiment_config, *, frame=None):
    stats = data_bundle.statistics
    return OptimizerChainEnv(
        frame if frame is not None else data_bundle.train,
        stats["optimizer_names"],
        stats["category_names"],
        stats["metric_scale"],
        stats["runtime_99_seconds"],
        experiment_config["reward"],
        experiment_config["observation"],
        **experiment_config["environment"],
    )


def test_gymnasium_environment_contract(data_bundle, experiment_config):
    """Stable Baselines3 should accept the environment without API warnings."""

    env = _environment(data_bundle, experiment_config)
    check_env(env, warn=True)
    env.close()


def test_stop_and_two_step_chain_behavior(data_bundle, experiment_config):
    """STOP is invalid first, valid second, and a second optimizer forms a chain."""

    env = _environment(data_bundle, experiment_config)
    circuit = data_bundle.splits["train"][0]

    env.reset(options={"circuit": circuit})
    _, reward, terminated, _, info = env.step(env.stop_action)
    assert terminated
    assert info["invalid_action"]
    assert reward == experiment_config["environment"]["invalid_stop_penalty"]

    env.reset(options={"circuit": circuit})
    _, _, first_terminated, _, first_info = env.step(0)
    if not first_info["failed"]:
        assert not first_terminated
        _, _, terminated, _, info = env.step(1)
        assert terminated
        assert info["selected_chain"] == "qiskit_standard__qiskit_ai"

    env.reset(options={"circuit": circuit})
    observation, _, first_terminated, _, first_info = env.step(0)
    if not first_info["failed"]:
        _, _, terminated, _, info = env.step(env.stop_action)
        assert terminated
        assert info["selected_chain"] == "qiskit_standard"
        assert observation.dtype == np.float32
    env.close()


def test_known_error_row_terminates_with_failure(data_bundle, experiment_config):
    """Selecting a recorded optimizer error must terminate and report failure."""

    error_row = data_bundle.train[
        (data_bundle.train["chain_name"] == "qiskit_ai")
        & data_bundle.train["error"]
    ].iloc[0]
    env = _environment(data_bundle, experiment_config)
    env.reset(options={"circuit": error_row["original_circuit"]})
    _, reward, terminated, _, info = env.step(1)
    assert terminated
    assert info["failed"]
    assert reward == experiment_config["reward"]["failure_penalty"]
    env.close()
