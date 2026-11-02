"""Data loading helpers for the LLM chain predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib
import json

import pandas as pd

try:
    from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (
        OPTIMIZER_NAMES,
        RANDOM_STATE,
        default_master_csv_path,
        default_split_dir,
        prepare_datasets,
    )
except ImportError:
    from ..RL.two_opt_chain_rl_v4.data_processing import (
        OPTIMIZER_NAMES,
        RANDOM_STATE,
        default_master_csv_path,
        default_split_dir,
        prepare_datasets,
    )


CHAIN_ID_VERSION = "v1"
CANDIDATE_CHAIN_LIST_VERSION = "v1"
DEFAULT_SPLIT_SOURCE = "rl_v4_saved_splits"


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded data bundle for one experiment run."""

    full_df: pd.DataFrame
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    split_seed: int
    split_source: str
    master_csv_path: Path
    master_csv_fingerprint: str


def build_candidate_chains(
    optimizer_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed action space of 30 valid chains with stable ids."""
    optimizer_names = list(optimizer_names or OPTIMIZER_NAMES)
    candidates: list[dict[str, Any]] = []
    index = 1

    for optimizer in optimizer_names:
        candidates.append(
            {
                "chain_id": f"chain_{index:02d}",
                "chain": [optimizer],
                "chain_name": optimizer,
                "length": 1,
            }
        )
        index += 1

    for first_optimizer in optimizer_names:
        for second_optimizer in optimizer_names:
            chain_name = f"{first_optimizer}__{second_optimizer}"
            candidates.append(
                {
                    "chain_id": f"chain_{index:02d}",
                    "chain": [first_optimizer, second_optimizer],
                    "chain_name": chain_name,
                    "length": 2,
                }
            )
            index += 1

    if len(candidates) != 30:
        raise ValueError(f"Expected 30 candidate chains, found {len(candidates)}")

    return candidates


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_splits(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    if not (train_path.exists() and validation_path.exists() and test_path.exists()):
        bundle = prepare_datasets(save=True, feature_batch="plus_all_no_history")
        return bundle["train_df"], bundle["validation_df"], bundle["test_df"]

    return (
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        pd.read_csv(test_path),
    )


def load_dataset_bundle(
    master_csv_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_seed: int = RANDOM_STATE,
) -> DatasetBundle:
    """Load the cleaned master dataset plus train/validation/test split frames."""
    master_path = Path(master_csv_path) if master_csv_path else default_master_csv_path()
    master_path = master_path.expanduser().resolve()
    split_root = Path(split_dir) if split_dir else default_split_dir()
    split_root = split_root.expanduser().resolve()

    full_df = pd.read_csv(master_path)
    train_df, validation_df, test_df = _load_saved_splits(split_root)

    return DatasetBundle(
        full_df=full_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        split_seed=int(split_seed),
        split_source=DEFAULT_SPLIT_SOURCE,
        master_csv_path=master_path,
        master_csv_fingerprint=_fingerprint_file(master_path),
    )


def split_frame_for_name(bundle: DatasetBundle, split_name: str) -> pd.DataFrame:
    """Return one evaluation split by its public name."""
    normalized = str(split_name).strip().lower()
    if normalized == "train":
        return bundle.train_df.copy()
    if normalized == "validation":
        return bundle.validation_df.copy()
    if normalized == "test":
        return bundle.test_df.copy()
    raise ValueError(f"Unsupported split: {split_name}")


def circuit_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one representative row per original circuit."""
    representatives = (
        frame.sort_values(["original_circuit", "chain_name"])
        .groupby("original_circuit", as_index=False)
        .first()
    )
    return representatives.copy()


def candidate_rows_for_circuit(
    frame: pd.DataFrame,
    circuit: str,
) -> pd.DataFrame:
    """Return all valid candidate rows for one circuit."""
    rows = frame[frame["original_circuit"] == circuit].copy()
    if rows.empty:
        raise KeyError(f"No rows found for circuit {circuit}")
    return rows


def dataset_metadata(bundle: DatasetBundle) -> dict[str, Any]:
    """Return metadata that should be logged with each run."""
    return {
        "master_csv_path": str(bundle.master_csv_path),
        "master_csv_fingerprint": bundle.master_csv_fingerprint,
        "split_seed": bundle.split_seed,
        "split_source": bundle.split_source,
        "candidate_chain_list_version": CANDIDATE_CHAIN_LIST_VERSION,
        "candidate_chain_count": len(build_candidate_chains()),
    }


def write_resolved_split_ids(bundle: DatasetBundle) -> dict[str, list[str]]:
    """Return circuit ids per split for logging."""
    return {
        "train": sorted(bundle.train_df["original_circuit"].unique().tolist()),
        "validation": sorted(bundle.validation_df["original_circuit"].unique().tolist()),
        "test": sorted(bundle.test_df["original_circuit"].unique().tolist()),
    }


def to_pretty_json(value: Any) -> str:
    """Serialize a value deterministically for logging."""
    return json.dumps(value, indent=2, sort_keys=True)
