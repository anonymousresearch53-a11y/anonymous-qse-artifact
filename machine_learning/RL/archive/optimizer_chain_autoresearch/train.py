"""Editable multi-algorithm training entry point.

EDITABLE BY THE RESEARCH AGENT.

The module converts the author's optimizer-chain training notebook into a
repeatable Python interface.  PPO, DQN, and A2C share the same environment and
configuration contract while retaining their algorithm-specific parameters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

try:
    from .data_setup import DataBundle, ensure_data_setup
    from .experiment_config import get_experiment_config
    from .optimizer_chain_env import OptimizerChainEnv
except ImportError:  # pragma: no cover - supports ``python train.py``
    from data_setup import DataBundle, ensure_data_setup
    from experiment_config import get_experiment_config
    from optimizer_chain_env import OptimizerChainEnv


ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "silu": torch.nn.SiLU,
}


class EpisodeHistoryCallback(BaseCallback):
    """Collect episode returns without flooding stdout during autonomous runs."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible candidate training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _policy_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the architecture declaration into Stable Baselines3 kwargs."""

    architecture = config["architecture"]
    activation_name = str(architecture.get("activation", "tanh")).lower()
    try:
        activation = ACTIVATIONS[activation_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {activation_name!r}; choose {sorted(ACTIVATIONS)}."
        ) from exc

    if config["algorithm"] == "dqn":
        net_arch: Any = list(architecture.get("q_network", [128, 128]))
    else:
        net_arch = {
            "pi": list(architecture.get("policy_network", [128, 128])),
            "vf": list(architecture.get("value_network", [128, 128])),
        }
    return {"net_arch": net_arch, "activation_fn": activation}


def make_environment(
    frame: Any,
    bundle: DataBundle,
    config: Mapping[str, Any],
    *,
    shuffle_each_epoch: bool,
    monitored: bool,
) -> Any:
    """Construct the editable environment using fixed training statistics."""

    statistics = bundle.statistics
    environment_config = dict(config["environment"])
    environment_config["shuffle_each_epoch"] = shuffle_each_epoch
    env = OptimizerChainEnv(
        frame,
        statistics["optimizer_names"],
        statistics["category_names"],
        statistics["metric_scale"],
        statistics["runtime_99_seconds"],
        config["reward"],
        config["observation"],
        **environment_config,
    )
    return Monitor(env) if monitored else env


def build_model(
    config: Mapping[str, Any],
    env: Any,
    *,
    seed: int,
    total_timesteps: int,
    device: str = "auto",
) -> BaseAlgorithm:
    """Build one Stable Baselines3 model from the active experiment.

    Small smoke runs automatically cap rollout lengths and batch sizes so all
    three algorithms can be tested without changing the production config.
    """

    algorithm = str(config["algorithm"]).lower()
    if algorithm not in ALGORITHM_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")

    kwargs = dict(config["algorithm_config"])
    kwargs["policy_kwargs"] = _policy_kwargs(config)
    kwargs["seed"] = seed
    kwargs["device"] = device
    kwargs["verbose"] = 0

    if algorithm == "ppo":
        kwargs["n_steps"] = max(2, min(int(kwargs["n_steps"]), total_timesteps))
        kwargs["batch_size"] = min(int(kwargs["batch_size"]), kwargs["n_steps"])
    elif algorithm == "a2c":
        kwargs["n_steps"] = max(1, min(int(kwargs["n_steps"]), total_timesteps))
    elif algorithm == "dqn":
        kwargs["buffer_size"] = max(
            int(kwargs.get("batch_size", 32)) + 1,
            min(int(kwargs["buffer_size"]), max(total_timesteps * 2, 100)),
        )

    model_class = ALGORITHM_CLASSES[algorithm]
    return model_class("MlpPolicy", env, **kwargs)


def load_model(
    algorithm: str,
    model_path: Path,
    *,
    env: Any | None = None,
    device: str = "auto",
) -> BaseAlgorithm:
    """Load a saved model using its declared algorithm family."""

    normalized = algorithm.lower()
    if normalized not in ALGORITHM_CLASSES:
        raise ValueError(f"Unsupported saved-model algorithm: {algorithm!r}")
    return ALGORITHM_CLASSES[normalized].load(
        str(model_path), env=env, device=device
    )


def train_one_seed(
    *,
    config: Mapping[str, Any],
    bundle: DataBundle,
    output_dir: Path,
    seed: int,
    total_timesteps: int,
    device: str = "auto",
) -> dict[str, Any]:
    """Train and save one seeded candidate model.

    Args:
        config: Experiment declaration.
        bundle: Fixed train/validation/test data and statistics.
        output_dir: Seed-specific artifact directory.
        seed: Random seed.
        total_timesteps: Stable Baselines3 learning budget.
        device: Torch device accepted by Stable Baselines3.

    Returns:
        JSON-serializable training metadata including the model path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    env = make_environment(
        bundle.train,
        bundle,
        config,
        shuffle_each_epoch=True,
        monitored=True,
    )
    callback = EpisodeHistoryCallback()
    model = build_model(
        config,
        env,
        seed=seed,
        total_timesteps=total_timesteps,
        device=device,
    )

    started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.perf_counter() - started

    model_base = output_dir / "model"
    model.save(str(model_base))
    model_path = output_dir / "model.zip"
    metadata = {
        "algorithm": config["algorithm"],
        "seed": seed,
        "requested_timesteps": total_timesteps,
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "observation_size": int(env.observation_space.shape[0]),
        "action_count": int(env.action_space.n),
        "episodes_completed": len(callback.episode_rewards),
        "mean_episode_reward": (
            float(np.mean(callback.episode_rewards))
            if callback.episode_rewards
            else None
        ),
        "mean_episode_length": (
            float(np.mean(callback.episode_lengths))
            if callback.episode_lengths
            else None
        ),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env.close()
    return metadata


def main() -> None:
    """Train one standalone model using the active editable configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = ensure_data_setup()
    metadata = train_one_seed(
        config=get_experiment_config(),
        bundle=bundle,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.timesteps,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
