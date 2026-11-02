"""End-to-end registry and finalized baseline smoke test."""

from __future__ import annotations

import json

import pytest

from machine_learning.RL.optimizer_chain_autoresearch.run_experiment import execute


def test_finalize_populates_research_memory(tmp_path):
    """Five tiny seeded models should exercise every persistent registry."""

    registry = tmp_path / "registry"
    runs = tmp_path / "runs"
    event = execute(
        stage="finalize",
        label="test baseline",
        device="cpu",
        timesteps_override=4,
        seeds_override=[42, 43, 44, 45, 46],
        runs_dir=runs,
        registry_dir=registry,
    )
    champions = json.loads((registry / "champions.json").read_text(encoding="utf-8"))
    pareto = json.loads((registry / "pareto.json").read_text(encoding="utf-8"))

    assert event["status"] == "retained"
    assert len(champions["objective_tracks"]) == 75
    assert len(pareto) == 1
    assert (registry / "experiments.jsonl").exists()
    assert (registry / "results.tsv").exists()
    assert (registry / "research_report.md").exists()

    # The second call must fail before creating another run directory or
    # spending any training compute.
    before = sorted(path.name for path in runs.iterdir())
    with pytest.raises(RuntimeError, match="Exact duplicate"):
        execute(
            stage="finalize",
            label="test baseline",
            device="cpu",
            timesteps_override=4,
            seeds_override=[42, 43, 44, 45, 46],
            runs_dir=runs,
            registry_dir=registry,
        )
    assert sorted(path.name for path in runs.iterdir()) == before
