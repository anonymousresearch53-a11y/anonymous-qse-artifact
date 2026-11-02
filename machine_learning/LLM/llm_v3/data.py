"""Data loading helpers for the LLM-based orchestration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib

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
    from ...RL.two_opt_chain_rl_v4.data_processing import (
        OPTIMIZER_NAMES,
        RANDOM_STATE,
        default_master_csv_path,
        default_split_dir,
        prepare_datasets,
    )


CHAIN_ID_VERSION = "v2"
CANDIDATE_CHAIN_LIST_VERSION = "v2_with_original_action"
DEFAULT_SPLIT_SOURCE = "rl_v4_saved_splits"
ORIGINAL_CHAIN_ID = "chain_00"
ORIGINAL_CHAIN_NAME = "__original__"


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded dataset plus resolved train/validation/test split frames."""

    full_df: pd.DataFrame
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    split_seed: int
    split_source: str
    split_dir: Path
    master_csv_path: Path
    master_csv_fingerprint: str
    train_csv_fingerprint: str
    validation_csv_fingerprint: str
    test_csv_fingerprint: str


def original_candidate() -> dict[str, Any]:
    """Return the explicit no-optimization action."""
    return {
        "chain_id": ORIGINAL_CHAIN_ID,
        "chain": [],
        "chain_name": ORIGINAL_CHAIN_NAME,
        "length": 0,
    }


def build_candidate_chains(
    optimizer_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed action space of valid 0/1/2-step chains."""
    optimizer_names = list(optimizer_names or OPTIMIZER_NAMES)
    candidates: list[dict[str, Any]] = [original_candidate()]
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
            candidates.append(
                {
                    "chain_id": f"chain_{index:02d}",
                    "chain": [first_optimizer, second_optimizer],
                    "chain_name": f"{first_optimizer}__{second_optimizer}",
                    "length": 2,
                }
            )
            index += 1

    expected_count = 1 + len(optimizer_names) + (len(optimizer_names) * len(optimizer_names))
    if len(candidates) != expected_count:
        raise ValueError(
            f"Expected {expected_count} candidate chains, found {len(candidates)}"
        )

    return candidates


def candidate_chain_lookup(
    optimizer_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return candidate chains keyed by stable chain id."""
    return {
        candidate["chain_id"]: candidate
        for candidate in build_candidate_chains(optimizer_names=optimizer_names)
    }


def _fingerprint_file(path: Path) -> str:
    """Return a stable SHA256 fingerprint for the dataset file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_splits(
    split_dir: Path,
    master_csv_path: Path,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load precomputed RL v4 split CSVs, or regenerate them if missing."""
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    if int(split_seed) != int(RANDOM_STATE):
        train_df, validation_df, test_df = _regenerate_saved_splits(
            split_dir=split_dir,
            master_csv_path=master_csv_path,
            split_seed=split_seed,
        )
        return train_df, validation_df, test_df, True

    if not (train_path.exists() and validation_path.exists() and test_path.exists()):
        train_df, validation_df, test_df = _regenerate_saved_splits(
            split_dir=split_dir,
            master_csv_path=master_csv_path,
            split_seed=split_seed,
        )
        return train_df, validation_df, test_df, True

    return (
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        pd.read_csv(test_path),
        False,
    )


def _regenerate_saved_splits(
    split_dir: Path,
    master_csv_path: Path,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regenerate RL v4 split CSVs from the current master dataset."""
    bundle = prepare_datasets(
        csv_path=master_csv_path,
        split_dir=split_dir,
        random_state=int(split_seed),
        save=True,
        feature_batch="plus_all_no_history",
    )
    return bundle["train_df"], bundle["validation_df"], bundle["test_df"]


def _normalized_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return a deterministically ordered view for split validation."""
    normalized = frame.loc[:, columns].copy()
    sort_columns = [
        column
        for column in ["original_circuit", "chain_name"]
        if column in normalized.columns
    ]
    if not sort_columns:
        sort_columns = list(columns)
    return normalized.sort_values(sort_columns).reset_index(drop=True)


def _split_matches_master_subset(
    split_frame: pd.DataFrame,
    full_df: pd.DataFrame,
) -> bool:
    """Return whether one split frame matches the current master rows for its ids."""
    if split_frame.empty:
        return False

    circuit_ids = split_frame["original_circuit"].astype(str).unique().tolist()
    master_subset = full_df[full_df["original_circuit"].astype(str).isin(circuit_ids)].copy()
    shared_columns = sorted(set(master_subset.columns) & set(split_frame.columns))
    if not shared_columns:
        return False

    normalized_split = _normalized_frame(split_frame, shared_columns)
    normalized_master = _normalized_frame(master_subset, shared_columns)
    return normalized_split.equals(normalized_master)


def _saved_splits_match_master(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """Validate that cached split CSVs match the current master dataset."""
    split_frames = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    required_columns = {"original_circuit", "chain_name"}
    if any(not required_columns.issubset(frame.columns) for frame in split_frames.values()):
        return False

    full_ids = set(full_df["original_circuit"].astype(str).unique().tolist())
    split_ids = {
        name: set(frame["original_circuit"].astype(str).unique().tolist())
        for name, frame in split_frames.items()
    }
    if split_ids["train"] & split_ids["validation"]:
        return False
    if split_ids["train"] & split_ids["test"]:
        return False
    if split_ids["validation"] & split_ids["test"]:
        return False
    if split_ids["train"] | split_ids["validation"] | split_ids["test"] != full_ids:
        return False
    if sum(len(frame) for frame in split_frames.values()) != len(full_df):
        return False

    return all(
        _split_matches_master_subset(frame, full_df)
        for frame in split_frames.values()
    )


def load_dataset_bundle(
    master_csv_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_seed: int = RANDOM_STATE,
) -> DatasetBundle:
    """Load the cleaned master dataset and resolved train/validation/test splits."""
    master_path = Path(master_csv_path) if master_csv_path else default_master_csv_path()
    master_path = master_path.expanduser().resolve()

    split_root = Path(split_dir) if split_dir else default_split_dir()
    split_root = split_root.expanduser().resolve()
    train_path = split_root / "train.csv"
    validation_path = split_root / "validation.csv"
    test_path = split_root / "test.csv"

    full_df = pd.read_csv(master_path)
    train_df, validation_df, test_df, regenerated_splits = _load_saved_splits(
        split_root,
        master_csv_path=master_path,
        split_seed=split_seed,
    )
    if not _saved_splits_match_master(full_df, train_df, validation_df, test_df):
        train_df, validation_df, test_df = _regenerate_saved_splits(
            split_dir=split_root,
            master_csv_path=master_path,
            split_seed=split_seed,
        )
        regenerated_splits = True

    split_source = (
        f"rl_v4_regenerated_splits_seed_{int(split_seed)}"
        if regenerated_splits
        else DEFAULT_SPLIT_SOURCE
    )

    return DatasetBundle(
        full_df=full_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        split_seed=int(split_seed),
        split_source=split_source,
        split_dir=split_root,
        master_csv_path=master_path,
        master_csv_fingerprint=_fingerprint_file(master_path),
        train_csv_fingerprint=_fingerprint_file(train_path),
        validation_csv_fingerprint=_fingerprint_file(validation_path),
        test_csv_fingerprint=_fingerprint_file(test_path),
    )


def split_frame_for_name(bundle: DatasetBundle, split_name: str) -> pd.DataFrame:
    """Return one split frame by name."""
    normalized = str(split_name).strip().lower()
    if normalized == "train":
        return bundle.train_df.copy()
    if normalized == "validation":
        return bundle.validation_df.copy()
    if normalized == "test":
        return bundle.test_df.copy()
    raise ValueError(f"Unsupported split name: {split_name}")


def circuit_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic representative row per original circuit."""
    representatives = (
        frame.sort_values(["original_circuit", "chain_name"])
        .groupby("original_circuit", as_index=False)
        .first()
    )
    return representatives.copy()


def candidate_rows_for_circuit(
    frame: pd.DataFrame,
    original_circuit: str,
) -> pd.DataFrame:
    """Return all candidate rows for one original circuit."""
    rows = frame[frame["original_circuit"] == original_circuit].copy()
    if rows.empty:
        raise KeyError(f"No rows found for circuit {original_circuit}")
    return rows


def dataset_metadata(bundle: DatasetBundle) -> dict[str, Any]:
    """Return reproducibility metadata for the currently loaded dataset."""
    return {
        "master_csv_path": str(bundle.master_csv_path),
        "master_csv_fingerprint": bundle.master_csv_fingerprint,
        "split_dir": str(bundle.split_dir),
        "train_csv_fingerprint": bundle.train_csv_fingerprint,
        "validation_csv_fingerprint": bundle.validation_csv_fingerprint,
        "test_csv_fingerprint": bundle.test_csv_fingerprint,
        "split_seed": int(bundle.split_seed),
        "split_source": bundle.split_source,
        "candidate_chain_list_version": CANDIDATE_CHAIN_LIST_VERSION,
        "candidate_chain_count": len(build_candidate_chains()),
        "original_chain_id": ORIGINAL_CHAIN_ID,
        "original_chain_name": ORIGINAL_CHAIN_NAME,
    }


def resolved_split_ids(bundle: DatasetBundle) -> dict[str, list[str]]:
    """Return sorted original-circuit ids for each split."""
    return {
        "train": sorted(bundle.train_df["original_circuit"].unique().tolist()),
        "validation": sorted(bundle.validation_df["original_circuit"].unique().tolist()),
        "test": sorted(bundle.test_df["original_circuit"].unique().tolist()),
    }
