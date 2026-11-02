"""Smoke tests for all supported Stable Baselines3 algorithms."""

from __future__ import annotations

from copy import deepcopy

import pytest

from machine_learning.RL.optimizer_chain_autoresearch.evaluate import evaluate_model
from machine_learning.RL.optimizer_chain_autoresearch.train import train_one_seed


@pytest.mark.parametrize("algorithm", ["ppo", "dqn", "a2c"])
def test_supported_algorithm_smoke(
    algorithm, tmp_path, data_bundle, experiment_config
):
    """Each supported algorithm should train, save, load, and select 75 scores."""

    config = deepcopy(experiment_config)
    config["algorithm"] = algorithm
    # Keep algorithm-specific dictionaries available after changing the selected
    # family in this isolated test.
    if algorithm == "ppo":
        config["algorithm_config"] = {
            "learning_rate": 1e-3,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    elif algorithm == "dqn":
        config["algorithm_config"] = {
            "learning_rate": 1e-3,
            "buffer_size": 100,
            "learning_starts": 1,
            "batch_size": 4,
            "gamma": 1.0,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 4,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "max_grad_norm": 10.0,
        }
    else:
        config["algorithm_config"] = {
            "learning_rate": 1e-3,
            "n_steps": 4,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": False,
        }

    metadata = train_one_seed(
        config=config,
        bundle=data_bundle,
        output_dir=tmp_path / algorithm,
        seed=42,
        total_timesteps=8,
        device="cpu",
    )
    result = evaluate_model(
        model_path=tmp_path / algorithm / "model.zip",
        config=config,
        bundle=data_bundle,
        device="cpu",
        include_baselines=False,
    )
    assert metadata["algorithm"] == algorithm
    assert result.summary["objective_track_count"] == 75
    assert len(result.selected) == 34
