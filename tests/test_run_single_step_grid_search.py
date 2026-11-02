from __future__ import annotations

import pytest

from scripts.run_single_step_grid_search import (
    HARDWARE_AWARE_OPTIMIZER_NAMES,
    resolve_optimizer_names,
)


def test_logical_mode_preserves_requested_optimizer_names() -> None:
    requested = ["qiskit_standard", "tket"]

    assert resolve_optimizer_names(requested, hardware_aware=False) == requested


def test_hardware_mode_maps_requested_optimizer_names() -> None:
    assert resolve_optimizer_names(
        ["qiskit_standard", "tket"],
        hardware_aware=True,
    ) == ["qiskit_standard_hardware", "tket_hardware"]


def test_hardware_mode_maps_default_optimizer_set() -> None:
    resolved = resolve_optimizer_names(None, hardware_aware=True)

    assert resolved == list(HARDWARE_AWARE_OPTIMIZER_NAMES.values())


def test_logical_mode_rejects_explicit_hardware_optimizer_name() -> None:
    with pytest.raises(ValueError, match="--hardware-aware yes"):
        resolve_optimizer_names(["qiskit_standard_hardware"], hardware_aware=False)
