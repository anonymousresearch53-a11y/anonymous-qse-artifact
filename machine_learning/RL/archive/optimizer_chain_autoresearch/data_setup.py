"""Fixed data preparation and immutable split management.

READ-ONLY DURING AUTONOMOUS RESEARCH.

This module is the source of truth for the master CSV, circuit assignments,
training-only scaling constants, and evaluator winsorization bounds.  It never
imports the unrelated RL implementation elsewhere in the repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = PACKAGE_DIR / "splits"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FILENAMES = {
    "train": "train_circuits.txt",
    "validation": "validation_circuits.txt",
    "test": "test_circuits.txt",
}
SPLIT_SEED = 42
EXPECTED_SPLIT_SIZES = {"train": 272, "validation": 34, "test": 34}

OPTIMIZER_NAMES = [
    "qiskit_standard",
    "qiskit_ai",
    "tket",
    "wisq_rules",
    "wisq_bqskit",
]

QUALITY_METRICS: dict[str, tuple[str, str]] = {
    "two_qubit_gates": ("initial_two_qubit_gates", "final_two_qubit_gates"),
    "depth": ("initial_depth", "final_depth"),
    "total_gates": ("initial_total_gates", "final_total_gates"),
    "two_qubit_depth": ("initial_two_qubit_depth", "final_two_qubit_depth"),
}

REQUIRED_COLUMNS = {
    "original_circuit",
    "original_category",
    "chain_name",
    "initial_num_qubits",
    "initial_depth",
    "initial_two_qubit_gates",
    "initial_total_gates",
    "initial_two_qubit_depth",
    "initial_gate_density",
    "initial_two_qubit_ratio",
    "final_num_qubits",
    "final_depth",
    "final_two_qubit_gates",
    "final_total_gates",
    "final_two_qubit_depth",
    "total_runtime_seconds",
    "error",
}


@dataclass(frozen=True)
class DataBundle:
    """Prepared dataframes and immutable training-derived statistics."""

    full: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    splits: dict[str, list[str]]
    statistics: dict[str, Any]
    source_csv: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for the master CSV."""

    current = (start or PACKAGE_DIR).resolve()
    for candidate in (current, *current.parents):
        csv_path = candidate / "dbs" / "original_data" / "one_and_two_opt_master.csv"
        if csv_path.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate dbs/original_data/one_and_two_opt_master.csv."
    )


def default_csv_path() -> Path:
    """Return the canonical optimizer-chain master CSV path."""

    return find_project_root() / "dbs" / "original_data" / "one_and_two_opt_master.csv"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_error_column(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean representations into real booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    truthy = {"1", "true", "yes", "y", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def load_master_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the complete optimizer-chain outcome table.

    Raises:
        ValueError: If required columns, circuits, or optimizer chains are
            missing.
    """

    path = (csv_path or default_csv_path()).resolve()
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Master CSV is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Master CSV contains no rows.")

    frame = frame.copy()
    frame["original_circuit"] = frame["original_circuit"].astype(str)
    frame["original_category"] = frame["original_category"].astype(str)
    frame["chain_name"] = frame["chain_name"].astype(str)
    frame["error"] = _coerce_error_column(frame["error"])

    expected_chains = {
        *OPTIMIZER_NAMES,
        *{
            f"{first}__{second}"
            for first in OPTIMIZER_NAMES
            for second in OPTIMIZER_NAMES
        },
    }
    actual_chains = set(frame["chain_name"].unique())
    if actual_chains != expected_chains:
        raise ValueError(
            "Master CSV chain set differs from the fixed five-optimizer action space. "
            f"Missing={sorted(expected_chains - actual_chains)}, "
            f"extra={sorted(actual_chains - expected_chains)}"
        )

    counts = frame.groupby("original_circuit")["chain_name"].nunique()
    incomplete = counts[counts != len(expected_chains)]
    if not incomplete.empty:
        raise ValueError(
            "Every circuit must contain all 30 one/two-step chains; "
            f"incomplete circuits: {incomplete.index.tolist()[:10]}"
        )
    return frame


def _allocate_category_counts(
    category_counts: dict[str, int],
    target: int,
    *,
    reserve_for_other_split: bool,
) -> dict[str, int]:
    """Allocate an exact split size while preserving rare categories in train.

    Categories with at least three circuits receive one validation/test example
    before proportional largest-remainder allocation.  Singleton categories
    therefore always remain in training, and three-example categories can be
    represented in all three splits.
    """

    eligible = {name: count for name, count in category_counts.items() if count >= 3}
    allocation = {name: (1 if name in eligible else 0) for name in category_counts}
    remaining = target - sum(allocation.values())
    if remaining < 0:
        raise ValueError("Target split is too small for category-aware allocation.")

    # Reserve one training circuit and, while allocating validation, one test
    # circuit in every eligible category.
    reserve = 2 if reserve_for_other_split else 1
    capacity = {
        name: max(category_counts[name] - reserve - allocation[name], 0)
        for name in category_counts
    }
    total_weight = sum(category_counts.values())
    ideals = {
        name: remaining * category_counts[name] / total_weight
        for name in category_counts
    }
    floors = {
        name: min(int(np.floor(ideals[name])), capacity[name])
        for name in category_counts
    }
    for name, value in floors.items():
        allocation[name] += value
        capacity[name] -= value
    remaining -= sum(floors.values())

    order = sorted(
        category_counts,
        key=lambda name: (
            -(ideals[name] - np.floor(ideals[name])),
            -category_counts[name],
            name,
        ),
    )
    while remaining > 0:
        progressed = False
        for name in order:
            if capacity[name] <= 0:
                continue
            allocation[name] += 1
            capacity[name] -= 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Unable to satisfy requested split size.")
    return allocation


def create_stratified_splits(
    frame: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Create deterministic category-aware 272/34/34 circuit assignments."""

    circuit_categories = (
        frame[["original_circuit", "original_category"]]
        .drop_duplicates()
        .sort_values(["original_category", "original_circuit"])
    )
    if circuit_categories["original_circuit"].duplicated().any():
        raise ValueError("A circuit is assigned to multiple categories.")
    if len(circuit_categories) != sum(EXPECTED_SPLIT_SIZES.values()):
        raise ValueError(
            "The locked 272/34/34 contract expects exactly 340 circuits; "
            f"found {len(circuit_categories)}."
        )

    grouped = {
        category: sorted(group["original_circuit"].tolist())
        for category, group in circuit_categories.groupby("original_category")
    }
    category_counts = {category: len(circuits) for category, circuits in grouped.items()}
    validation_counts = _allocate_category_counts(
        category_counts,
        EXPECTED_SPLIT_SIZES["validation"],
        reserve_for_other_split=True,
    )

    # Test allocation is calculated from the same original category counts.
    # The shared reserve rule guarantees that validation + test still leave at
    # least one training circuit in every category, including a 1/1/1 split for
    # categories containing exactly three circuits.
    test_counts = _allocate_category_counts(
        category_counts,
        EXPECTED_SPLIT_SIZES["test"],
        reserve_for_other_split=True,
    )

    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    for category in sorted(grouped):
        circuits = np.asarray(grouped[category], dtype=object)
        rng.shuffle(circuits)
        n_validation = validation_counts[category]
        n_test = test_counts[category]
        splits["validation"].extend(map(str, circuits[:n_validation]))
        splits["test"].extend(
            map(str, circuits[n_validation : n_validation + n_test])
        )
        splits["train"].extend(map(str, circuits[n_validation + n_test :]))

    for name in SPLIT_NAMES:
        splits[name] = sorted(splits[name])
    validate_splits(frame, splits)
    return splits


def validate_splits(
    frame: pd.DataFrame,
    splits: dict[str, list[str]],
    *,
    enforce_expected_sizes: bool = True,
) -> None:
    """Validate split completeness, disjointness, and singleton handling."""

    missing_names = set(SPLIT_NAMES) - set(splits)
    if missing_names:
        raise ValueError(f"Missing split definitions: {sorted(missing_names)}")

    sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    if any(
        sets[left] & sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise ValueError("Circuit splits overlap.")

    expected_circuits = set(frame["original_circuit"].unique())
    assigned = set().union(*sets.values())
    if assigned != expected_circuits:
        raise ValueError(
            "Split membership does not exactly match the master CSV circuits."
        )
    if enforce_expected_sizes:
        actual_sizes = {name: len(sets[name]) for name in SPLIT_NAMES}
        if actual_sizes != EXPECTED_SPLIT_SIZES:
            raise ValueError(
                f"Expected split sizes {EXPECTED_SPLIT_SIZES}, found {actual_sizes}."
            )

    category_counts = (
        frame[["original_circuit", "original_category"]]
        .drop_duplicates()["original_category"]
        .value_counts()
    )
    singleton_categories = set(category_counts[category_counts == 1].index)
    for category in singleton_categories:
        singleton = frame.loc[
            frame["original_category"] == category, "original_circuit"
        ].iloc[0]
        if singleton not in sets["train"]:
            raise ValueError(
                f"Singleton category {category!r} must remain in training."
            )


def _write_lines(path: Path, values: list[str]) -> None:
    """Write one stable, newline-terminated value per line."""

    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def save_splits(
    splits: dict[str, list[str]],
    split_dir: Path,
    *,
    source_csv: Path,
    frame: pd.DataFrame,
    seed: int,
) -> None:
    """Persist split manifests and source-integrity metadata."""

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in SPLIT_FILENAMES.items():
        _write_lines(split_dir / filename, splits[name])

    circuit_categories = (
        frame[["original_circuit", "original_category"]].drop_duplicates()
    )
    metadata = {
        "seed": seed,
        "source_csv": str(source_csv.resolve()),
        "source_sha256": sha256_file(source_csv),
        "row_count": len(frame),
        "circuit_count": frame["original_circuit"].nunique(),
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "category_counts": {
            name: int(value)
            for name, value in circuit_categories["original_category"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    (split_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_splits(
    frame: pd.DataFrame,
    split_dir: Path,
    *,
    source_csv: Path,
) -> dict[str, list[str]]:
    """Load existing immutable split manifests and verify source integrity."""

    required = [split_dir / name for name in SPLIT_FILENAMES.values()]
    required.append(split_dir / "metadata.json")
    existing = [path.exists() for path in required]
    if any(existing) and not all(existing):
        raise ValueError(
            "Split directory is partially initialized; refuse to regenerate it "
            "because that could silently change the benchmark."
        )
    if not all(existing):
        raise FileNotFoundError("Split manifests have not been generated.")

    metadata = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    actual_hash = sha256_file(source_csv)
    if metadata.get("source_sha256") != actual_hash:
        raise ValueError(
            "Master CSV changed after splits were created. Regeneration must be "
            "an explicit human decision."
        )
    splits = {
        name: [
            line.strip()
            for line in (split_dir / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for name, filename in SPLIT_FILENAMES.items()
    }
    validate_splits(frame, splits)
    return splits


def _relative_improvement(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Compute per-row fractional reduction for one quality metric."""

    initial_column, final_column = QUALITY_METRICS[metric]
    denominator = frame[initial_column].replace(0, np.nan)
    return ((frame[initial_column] - frame[final_column]) / denominator).fillna(0.0)


def compute_training_statistics(train_frame: pd.DataFrame) -> dict[str, Any]:
    """Compute all scaling and evaluation constants from training circuits only."""

    initial_metrics = pd.DataFrame(
        {
            "num_qubits": train_frame["initial_num_qubits"],
            "depth": train_frame["initial_depth"],
            "two_qubit_gates": train_frame["initial_two_qubit_gates"],
            "total_gates": train_frame["initial_total_gates"],
            "two_qubit_depth": train_frame["initial_two_qubit_depth"],
            "gate_density": train_frame["initial_gate_density"],
            "two_qubit_ratio": train_frame["initial_two_qubit_ratio"],
        }
    )
    singles = train_frame[~train_frame["chain_name"].str.contains("__", regex=False)]
    final_qubits = singles["final_num_qubits"].replace(0, np.nan)
    final_total = singles["final_total_gates"].replace(0, np.nan)
    final_metrics = pd.DataFrame(
        {
            "num_qubits": singles["final_num_qubits"],
            "depth": singles["final_depth"],
            "two_qubit_gates": singles["final_two_qubit_gates"],
            "total_gates": singles["final_total_gates"],
            "two_qubit_depth": singles["final_two_qubit_depth"],
            "gate_density": singles["final_total_gates"] / final_qubits,
            "two_qubit_ratio": singles["final_two_qubit_gates"] / final_total,
        }
    ).fillna(0.0)
    scale_source = pd.concat([initial_metrics, final_metrics], ignore_index=True)
    metric_scale = (
        scale_source.quantile(0.95).replace(0, 1.0).astype(float).to_dict()
    )

    winsor_bounds = {}
    for metric in QUALITY_METRICS:
        values = _relative_improvement(train_frame, metric)
        winsor_bounds[metric] = {
            "lower": float(values.quantile(0.01)),
            "upper": float(values.quantile(0.99)),
        }

    return {
        "metric_scale": metric_scale,
        "runtime_99_seconds": float(
            max(train_frame["total_runtime_seconds"].quantile(0.99), 1e-9)
        ),
        "winsor_bounds": winsor_bounds,
        "category_names": sorted(train_frame["original_category"].unique().tolist()),
        "optimizer_names": list(OPTIMIZER_NAMES),
    }


def ensure_data_setup(
    *,
    csv_path: Path | None = None,
    split_dir: Path = DEFAULT_SPLIT_DIR,
    create_if_missing: bool = True,
) -> DataBundle:
    """Load data, creating immutable split manifests only when wholly absent."""

    source_csv = (csv_path or default_csv_path()).resolve()
    frame = load_master_data(source_csv)
    try:
        splits = load_splits(frame, split_dir, source_csv=source_csv)
    except FileNotFoundError:
        if not create_if_missing:
            raise
        splits = create_stratified_splits(frame, seed=SPLIT_SEED)
        save_splits(
            splits,
            split_dir,
            source_csv=source_csv,
            frame=frame,
            seed=SPLIT_SEED,
        )

    frames = {
        name: frame[frame["original_circuit"].isin(circuits)].copy()
        for name, circuits in splits.items()
    }
    statistics = compute_training_statistics(frames["train"])
    return DataBundle(
        full=frame,
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        splits=splits,
        statistics=statistics,
        source_csv=source_csv,
    )


def main() -> None:
    """Create or validate the immutable data setup and print a summary."""

    bundle = ensure_data_setup()
    print(f"Source: {bundle.source_csv}")
    for name in SPLIT_NAMES:
        frame = getattr(bundle, name)
        print(
            f"{name:10s}: {frame['original_circuit'].nunique():3d} circuits, "
            f"{len(frame):5d} rows"
        )
    print(f"Splits stored in: {DEFAULT_SPLIT_DIR}")


if __name__ == "__main__":
    main()
