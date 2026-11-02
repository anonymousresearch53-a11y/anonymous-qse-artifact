"""Shared fixtures for optimizer-chain autoresearch tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from machine_learning.RL.optimizer_chain_autoresearch.data_setup import (
    ensure_data_setup,
)
from machine_learning.RL.optimizer_chain_autoresearch.experiment_config import (
    get_experiment_config,
)


@pytest.fixture(scope="session")
def data_bundle():
    """Return the real immutable optimizer-chain data bundle."""

    return ensure_data_setup()


@pytest.fixture
def experiment_config():
    """Return an isolated mutable copy of the baseline experiment."""

    return deepcopy(get_experiment_config())
