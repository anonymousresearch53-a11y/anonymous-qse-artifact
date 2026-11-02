"""Tests for immutable data preparation and circuit-level splits."""

from __future__ import annotations

import json

import pytest

from machine_learning.RL.optimizer_chain_autoresearch.data_setup import (
    EXPECTED_SPLIT_SIZES,
    create_stratified_splits,
    ensure_data_setup,
    load_splits,
    save_splits,
)


def test_real_split_contract(data_bundle):
    """The generated benchmark must exactly match the planned 272/34/34 split."""

    assert {
        name: len(circuits) for name, circuits in data_bundle.splits.items()
    } == EXPECTED_SPLIT_SIZES
    split_sets = {name: set(values) for name, values in data_bundle.splits.items()}
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert len(set.union(*split_sets.values())) == 340


def test_split_generation_is_deterministic(data_bundle):
    """The same source and seed must always produce identical circuit names."""

    first = create_stratified_splits(data_bundle.full)
    second = create_stratified_splits(data_bundle.full)
    assert first == second == data_bundle.splits


def test_partial_split_directory_is_rejected(tmp_path, data_bundle):
    """A partial benchmark must never be silently overwritten."""

    (tmp_path / "train_circuits.txt").write_text("partial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partially initialized"):
        ensure_data_setup(split_dir=tmp_path)


def test_source_hash_change_is_rejected(tmp_path, data_bundle):
    """Existing manifests are tied to the exact source CSV digest."""

    save_splits(
        data_bundle.splits,
        tmp_path,
        source_csv=data_bundle.source_csv,
        frame=data_bundle.full,
        seed=42,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_sha256"] = "not-the-source-hash"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Master CSV changed"):
        load_splits(
            data_bundle.full,
            tmp_path,
            source_csv=data_bundle.source_csv,
        )
