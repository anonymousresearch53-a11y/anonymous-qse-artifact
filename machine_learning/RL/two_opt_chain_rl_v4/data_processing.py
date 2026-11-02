"""Dataset preparation utilities for the v4 two-step chain RL setup.

This module is responsible for:
- loading the master CSV,
- adding the derived columns used by rewards and analysis,
- creating train/validation/test splits by original circuit, and
- saving those split CSVs for reuse.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .environment import METRIC_NAMES, UNKNOWN_CATEGORY
except ImportError:
    from environment import METRIC_NAMES, UNKNOWN_CATEGORY


# Keep the defaults near the top so experiments can change them easily.
RANDOM_STATE = 42
OPTIMIZER_NAMES = [
    "qiskit_standard",
    "tket",
    "wisq_rules",
    "wisq_bqskit",
]
EXPECTED_CHAIN_ROWS_PER_CIRCUIT = len(OPTIMIZER_NAMES) + len(OPTIMIZER_NAMES) ** 2
BASELINE_FEATURE_COLUMNS = [
    "num_qubits",
    "depth",
    "two_qubit_gates",
    "total_gates",
    "two_qubit_depth",
    "gate_density",
    "two_qubit_ratio",
]
GATE_TYPE_FEATURE_COLUMNS = [
    "rz_gate_fraction",
    "sx_gate_fraction",
    "x_gate_fraction",
    "cx_gate_fraction",
]
RATIO_FEATURE_COLUMNS = [
    "depth_per_qubit",
    "gates_per_qubit",
    "two_qubit_gates_per_qubit",
]
INTERACTION_FEATURE_COLUMNS = [
    "num_interaction_edges",
    "avg_interaction_degree",
    "max_interaction_degree",
    "active_interaction_qubits",
    "active_interaction_qubit_fraction",
]
STATIC_STATE_FEATURE_COLUMNS = (
    GATE_TYPE_FEATURE_COLUMNS + RATIO_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS
)
DYNAMIC_PROGRESS_FEATURE_COLUMNS = [
    "depth_reduction_so_far",
    "depth_reduction_so_far_fraction",
    "two_qubit_gate_reduction_so_far",
    "two_qubit_gate_reduction_so_far_fraction",
]
FEATURE_BATCH_CONFIGS = {
    "original": {
        "include_category": True,
        "static_feature_columns": [],
        "dynamic_feature_columns": [],
        "description": "Baseline metrics plus category, previous optimizer, and step flags.",
    },
    "original_no_category": {
        "include_category": False,
        "static_feature_columns": [],
        "dynamic_feature_columns": [],
        "description": "Baseline metrics without the category one-hot features.",
    },
    "plus_gate_types": {
        "include_category": False,
        "static_feature_columns": GATE_TYPE_FEATURE_COLUMNS,
        "dynamic_feature_columns": [],
        "description": "Baseline without category plus gate-type fraction features.",
    },
    "plus_ratios": {
        "include_category": False,
        "static_feature_columns": RATIO_FEATURE_COLUMNS,
        "dynamic_feature_columns": [],
        "description": "Baseline without category plus derived ratio features.",
    },
    "plus_interaction": {
        "include_category": False,
        "static_feature_columns": INTERACTION_FEATURE_COLUMNS,
        "dynamic_feature_columns": [],
        "description": "Baseline without category plus interaction-graph features.",
    },
    "plus_dynamic_progress": {
        "include_category": False,
        "static_feature_columns": [],
        "dynamic_feature_columns": DYNAMIC_PROGRESS_FEATURE_COLUMNS,
        "description": "Baseline without category plus dynamic progress features.",
    },
    "plus_gate_types_ratios": {
        "include_category": False,
        "static_feature_columns": GATE_TYPE_FEATURE_COLUMNS + RATIO_FEATURE_COLUMNS,
        "dynamic_feature_columns": [],
        "description": "Baseline without category plus gate-type and ratio features.",
    },
    "plus_all_static": {
        "include_category": False,
        "static_feature_columns": (
            GATE_TYPE_FEATURE_COLUMNS
            + RATIO_FEATURE_COLUMNS
            + INTERACTION_FEATURE_COLUMNS
        ),
        "dynamic_feature_columns": [],
        "description": "Baseline without category plus all static structural features.",
    },
    "plus_all_features": {
        "include_category": False,
        "static_feature_columns": (
            GATE_TYPE_FEATURE_COLUMNS
            + RATIO_FEATURE_COLUMNS
            + INTERACTION_FEATURE_COLUMNS
        ),
        "dynamic_feature_columns": DYNAMIC_PROGRESS_FEATURE_COLUMNS,
        "description": "Baseline without category plus all static and progress features.",
    },
}
FEATURE_BATCH_ALIASES = {
    "plus_all_no_history": "plus_all_features",
}
REQUIRED_COLUMNS = [
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
    "current_step_runtime",
]


def find_project_root(start_path=None):
    """Walk upward until the repository root with the master CSV is found."""
    current = Path(start_path or __file__).resolve()
    if current.is_file():
        current = current.parent

    while not (current / "dbs" /  "combined_master.csv").exists():
        if current.parent == current:
            raise FileNotFoundError("Could not locate project root.")
        current = current.parent
    return current


def default_master_csv_path():
    """Return the default v4 source CSV path."""
    return find_project_root() / "dbs" / "combined_master.csv"


def default_split_dir():
    """Return the default location for cached split CSVs."""
    return Path(__file__).resolve().parent / "splits"


def feature_batch_names():
    """Return the supported feature-batch names."""
    return sorted(FEATURE_BATCH_CONFIGS.keys())


def feature_batch_config(feature_batch):
    """Validate and return one feature-batch config."""
    feature_batch = FEATURE_BATCH_ALIASES.get(str(feature_batch).strip(), str(feature_batch).strip())
    if feature_batch not in FEATURE_BATCH_CONFIGS:
        raise ValueError(
            f"Unsupported feature_batch: {feature_batch}. "
            f"Supported values: {feature_batch_names()}"
        )
    config = FEATURE_BATCH_CONFIGS[feature_batch].copy()
    config["static_feature_columns"] = list(config["static_feature_columns"])
    config["dynamic_feature_columns"] = list(config["dynamic_feature_columns"])
    return config


def supported_chain_name(chain_name):
    """Return whether a chain name uses only the supported four optimizers."""
    parts = [part for part in str(chain_name).split("__") if part]
    return len(parts) in {1, 2} and all(part in OPTIMIZER_NAMES for part in parts)


def required_master_columns():
    """Return the full set of master columns needed by v4."""
    static_feature_columns = set()
    dynamic_feature_columns = set()
    for config in FEATURE_BATCH_CONFIGS.values():
        static_feature_columns.update(config["static_feature_columns"])
        dynamic_feature_columns.update(config["dynamic_feature_columns"])

    initial_static_feature_columns = {
        f"initial_{column}" for column in static_feature_columns
    }
    return sorted(
        set(REQUIRED_COLUMNS)
        | static_feature_columns
        | initial_static_feature_columns
        | dynamic_feature_columns
    )


def filter_supported_optimizer_rows(frame):
    """Keep only rows whose chain uses the supported four optimizers."""
    df = frame.copy()
    supported_mask = df["chain_name"].map(supported_chain_name)
    filtered = df.loc[supported_mask].copy()
    filtered.attrs["dropped_unsupported_optimizer_rows"] = int((~supported_mask).sum())
    return filtered


def validate_supported_optimizer_rows(frame):
    """Ensure every retained circuit still has the full 4-optimizer action set."""
    if frame.empty:
        raise ValueError(
            "No supported optimizer rows remain after filtering. "
            "Expected chains built from qiskit_standard, tket, wisq_rules, and wisq_bqskit."
        )

    chain_counts = frame.groupby("original_circuit")["chain_name"].nunique()
    invalid_counts = chain_counts[chain_counts != EXPECTED_CHAIN_ROWS_PER_CIRCUIT]
    if invalid_counts.empty:
        return frame

    examples = ", ".join(
        f"{circuit}={int(count)}" for circuit, count in invalid_counts.head(10).items()
    )
    raise ValueError(
        "RL v4 expects exactly "
        f"{EXPECTED_CHAIN_ROWS_PER_CIRCUIT} supported optimization outcomes per original circuit "
        "after filtering to qiskit_standard, tket, wisq_rules, and wisq_bqskit. "
        f"Found mismatches for {len(invalid_counts)} circuits. Examples: {examples}. "
        "Regenerate dbs/combined_master.csv with the four-optimizer feature-extraction pipeline."
    )


def load_master_data(csv_path=None):
    """Load the raw master CSV and validate required columns."""
    path = Path(csv_path) if csv_path else default_master_csv_path()
    frame = pd.read_csv(path)

    missing = sorted(set(required_master_columns()) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Regenerate dbs/combined_master.csv with "
            "machine_learning/feature_extraction/build_combined_master.py so "
            "initial structural features are available for the RL environment."
        )

    filtered = filter_supported_optimizer_rows(frame)
    validate_supported_optimizer_rows(filtered)
    return filtered


def add_derived_columns(frame):
    """Add reduction percentages used later by rewards and reporting.

    We keep two versions of the two-qubit reduction columns:
    - total_*: improvement versus the original circuit
    - current_step_*: improvement caused by the step represented by this row

    For single-optimizer rows these are the same.
    For two-step rows the current-step version is measured against the
    parent circuit state produced by the first optimizer.
    """
    df = frame.copy()

    # Some failed chained runs do not record a parent circuit name.
    # For those rows, treat the parent circuit as the original circuit.
    chained_rows = df["chain_name"].str.contains("__", regex=False)
    df.loc[chained_rows & df["parent_circuit"].isna(), "parent_circuit"] = df.loc[
        chained_rows & df["parent_circuit"].isna(), "original_circuit"
    ]

    # Absolute and percentage two-qubit gate improvement versus the original circuit.
    df["total_two_qubit_gate_reduction"] = (
        df["initial_two_qubit_gates"] - df["final_two_qubit_gates"]
    )
    df["total_two_qubit_gate_reduction_pct"] = (
        100
        * df["total_two_qubit_gate_reduction"]
        / df["initial_two_qubit_gates"].replace(0, np.nan)
    )

    # Absolute and percentage depth improvement versus the original circuit.
    df["total_depth_reduction"] = df["initial_depth"] - df["final_depth"]
    df["total_depth_reduction_pct"] = (
        100 * df["total_depth_reduction"] / df["initial_depth"].replace(0, np.nan)
    )

    # Build a lookup for the single-step parent states.
    # Example:
    # original circuit = foo
    # single row chain_name = qiskit_standard
    # resulting parent state name = foo__qiskit_standard
    single_step_rows = df[~df["chain_name"].str.contains("__", regex=False)].copy()
    single_step_rows["result_circuit_name"] = (
        single_step_rows["original_circuit"] + "__" + single_step_rows["chain_name"]
    )
    parent_two_qubit_lookup = single_step_rows.set_index("result_circuit_name")[
        "final_two_qubit_gates"
    ].to_dict()
    parent_depth_lookup = single_step_rows.set_index("result_circuit_name")[
        "final_depth"
    ].to_dict()

    def current_step_parent_two_qubit_gates(row):
        if "__" not in row["chain_name"]:
            return row["initial_two_qubit_gates"]

        # For chained rows, the most reliable parent key is based on the
        # original circuit plus the first optimizer name. This still works
        # when parent_circuit is missing on failed rows that were treated as
        # "no further update".
        opt_1 = row.get("opt_1")
        if pd.notna(opt_1):
            derived_parent_name = f"{row['original_circuit']}__{opt_1}"
            if derived_parent_name in parent_two_qubit_lookup:
                return parent_two_qubit_lookup[derived_parent_name]

        # Fall back to the stored parent_circuit name when available.
        parent_circuit = row.get("parent_circuit")
        if pd.notna(parent_circuit):
            return parent_two_qubit_lookup.get(parent_circuit, np.nan)

        return np.nan

    df["current_step_parent_two_qubit_gates"] = df.apply(
        current_step_parent_two_qubit_gates,
        axis=1,
    )
    df["current_step_two_qubit_gate_reduction"] = (
        df["current_step_parent_two_qubit_gates"] - df["final_two_qubit_gates"]
    )
    # If the parent already has zero two-qubit gates, there is no remaining
    # two-qubit-gate improvement available for this step. Keep those rows and
    # define the current-step percentage improvement as 0.0 instead of NaN so
    # valid chained results are not dropped later.
    parent_two_qubit_denominator = df["current_step_parent_two_qubit_gates"].replace(
        0, np.nan
    )
    df["current_step_two_qubit_gate_reduction_pct"] = (
        100
        * df["current_step_two_qubit_gate_reduction"]
        / parent_two_qubit_denominator
    )
    zero_parent_two_qubit = df["current_step_parent_two_qubit_gates"] == 0
    df.loc[
        zero_parent_two_qubit, "current_step_two_qubit_gate_reduction_pct"
    ] = 0.0

    def current_step_parent_depth(row):
        if "__" not in row["chain_name"]:
            return row["initial_depth"]

        opt_1 = row.get("opt_1")
        if pd.notna(opt_1):
            derived_parent_name = f"{row['original_circuit']}__{opt_1}"
            if derived_parent_name in parent_depth_lookup:
                return parent_depth_lookup[derived_parent_name]

        parent_circuit = row.get("parent_circuit")
        if pd.notna(parent_circuit):
            return parent_depth_lookup.get(parent_circuit, np.nan)

        return np.nan

    df["current_step_parent_depth"] = df.apply(
        current_step_parent_depth,
        axis=1,
    )
    df["current_step_depth_reduction"] = (
        df["current_step_parent_depth"] - df["final_depth"]
    )
    # If the parent already has zero depth, there is no remaining depth
    # improvement available for this step. Keep those valid chained rows and
    # define the current-step percentage improvement as 0.0 instead of NaN.
    parent_depth_denominator = df["current_step_parent_depth"].replace(0, np.nan)
    df["current_step_depth_reduction_pct"] = (
        100
        * df["current_step_depth_reduction"]
        / parent_depth_denominator
    )
    zero_parent_depth = df["current_step_parent_depth"] == 0
    df.loc[zero_parent_depth, "current_step_depth_reduction_pct"] = 0.0

    # Keep the old names as aliases to the total-vs-original version so
    # any downstream reporting that still expects them continues to work.
    # This makes the reward bookkeeping explicit:
    # - total_* columns compare final state vs original circuit
    # - current_step_* columns compare the row vs its immediate parent state
    df["two_qubit_gate_reduction"] = df["total_two_qubit_gate_reduction"]
    df["two_qubit_gate_reduction_pct"] = df["total_two_qubit_gate_reduction_pct"]
    df["depth_reduction"] = df["total_depth_reduction"]
    df["depth_reduction_pct"] = df["total_depth_reduction_pct"]

    # Drop rows that cannot support the current reward calculation.
    df = df.dropna(
        subset=[
            "total_two_qubit_gate_reduction_pct",
            "total_depth_reduction_pct",
            "current_step_two_qubit_gate_reduction_pct",
            "current_step_depth_reduction_pct",
            "total_runtime_seconds",
            "current_step_runtime",
            "final_two_qubit_gates",
        ]
    ).copy()

    return df


def split_data(frame, random_state=RANDOM_STATE):
    """Split by original circuit with category-aware stratification.

    Larger categories are split approximately 80/10/10 across train,
    validation, and test. Categories with too few circuits to support all
    three splits are assigned deterministically in the least lossy way:
    - 1 circuit: train only
    - 2 circuits: 1 train, 1 test
    - 3 circuits: 1 train, 1 validation, 1 test
    """
    circuit_categories = (
        frame[["original_circuit", "original_category"]]
        .drop_duplicates()
        .sort_values(["original_category", "original_circuit"])
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(random_state)
    train_circuits: list[str] = []
    validation_circuits: list[str] = []
    test_circuits: list[str] = []

    def split_counts_for_category(num_circuits: int) -> tuple[int, int, int]:
        if num_circuits <= 1:
            return num_circuits, 0, 0
        if num_circuits == 2:
            return 1, 0, 1
        if num_circuits == 3:
            return 1, 1, 1

        test_count = max(1, int(round(num_circuits * 0.10)))
        validation_count = max(1, int(round(num_circuits * 0.10)))
        train_count = num_circuits - test_count - validation_count

        while train_count < 1:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            train_count = num_circuits - test_count - validation_count

        return train_count, validation_count, test_count

    for _, category_rows in circuit_categories.groupby("original_category", dropna=False):
        circuits = category_rows["original_circuit"].to_numpy(copy=True)
        rng.shuffle(circuits)

        train_count, validation_count, test_count = split_counts_for_category(len(circuits))
        train_end = train_count
        validation_end = train_count + validation_count

        train_circuits.extend(circuits[:train_end].tolist())
        validation_circuits.extend(circuits[train_end:validation_end].tolist())
        test_circuits.extend(circuits[validation_end:validation_end + test_count].tolist())

    splits = {
        "train": frame[frame["original_circuit"].isin(train_circuits)].copy(),
        "validation": frame[
            frame["original_circuit"].isin(validation_circuits)
        ].copy(),
        "test": frame[frame["original_circuit"].isin(test_circuits)].copy(),
    }

    return splits


def add_runtime_score(frame, runtime_denominator):
    """Add runtime normalization columns used by reward and reporting.

    `runtime_score` is step-local and is the version used by rewards.
    `total_runtime_score` is based on the full selected chain runtime and
    is used for cleaner evaluation reporting.
    """
    df = frame.copy()

    # Rewards should charge only for the runtime of the step represented
    # by the row. For first-step rows this matches the optimizer runtime,
    # and for two-step rows it should be the runtime of the second step only.
    df["runtime_score"] = (
        np.log1p(df["current_step_runtime"].clip(lower=0)) / runtime_denominator
    ).clip(0, 2)
    df["total_runtime_score"] = (
        np.log1p(df["total_runtime_seconds"].clip(lower=0)) / runtime_denominator
    ).clip(0, 2)
    df["runtime_denominator"] = float(runtime_denominator)
    return df


def initial_metric_frame(frame):
    """Return the initial metrics in the environment feature layout."""
    return pd.DataFrame(
        {
            "num_qubits": frame["initial_num_qubits"],
            "depth": frame["initial_depth"],
            "two_qubit_gates": frame["initial_two_qubit_gates"],
            "total_gates": frame["initial_total_gates"],
            "two_qubit_depth": frame["initial_two_qubit_depth"],
            "gate_density": frame["initial_gate_density"],
            "two_qubit_ratio": frame["initial_two_qubit_ratio"],
        }
    )


def final_metric_frame(frame):
    """Return the final metrics in the environment feature layout."""
    final_total = frame["final_total_gates"].replace(0, np.nan)
    final_qubits = frame["final_num_qubits"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "num_qubits": frame["final_num_qubits"],
            "depth": frame["final_depth"],
            "two_qubit_gates": frame["final_two_qubit_gates"],
            "total_gates": frame["final_total_gates"],
            "two_qubit_depth": frame["final_two_qubit_depth"],
            "gate_density": frame["final_total_gates"] / final_qubits,
            "two_qubit_ratio": frame["final_two_qubit_gates"] / final_total,
        }
    ).fillna(0.0)


def compute_metric_scale(train_frame):
    """Use training data only to build metric normalization scales."""
    # Metric scale uses all initial states plus single-step final states.
    # That matches the states the agent actually observes.
    train_single_df = train_frame[
        ~train_frame["chain_name"].str.contains("__", regex=False)
    ]
    scale_source = pd.concat(
        [initial_metric_frame(train_frame), final_metric_frame(train_single_df)],
        ignore_index=True,
    )
    return scale_source[METRIC_NAMES].quantile(0.95).replace(0, 1.0).to_dict()


def compute_observation_feature_scale(
    train_frame,
    static_feature_columns,
    dynamic_feature_columns,
):
    """Build normalization scales for the selected add-on observation features."""
    scales = {}

    for column in static_feature_columns:
        source_columns = [
            source_column
            for source_column in (f"initial_{column}", column)
            if source_column in train_frame.columns
        ]
        if not source_columns:
            scales[column] = 1.0
            continue

        values = pd.concat(
            [
                pd.to_numeric(train_frame[source_column], errors="coerce")
                for source_column in source_columns
            ],
            ignore_index=True,
        )
        scale = float(values.quantile(0.95))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scales[column] = scale

    for column in dynamic_feature_columns:
        if column not in train_frame.columns:
            scales[column] = 1.0
            continue
        values = pd.to_numeric(train_frame[column], errors="coerce")
        scale = float(values.abs().quantile(0.95))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scales[column] = scale

    return scales


def split_summary_frame(split_frames):
    """Create a small summary table for train/validation/test sizes."""
    return pd.DataFrame(
        {
            "split": list(split_frames.keys()),
            "rows": [len(frame) for frame in split_frames.values()],
            "circuits": [
                frame["original_circuit"].nunique() for frame in split_frames.values()
            ],
            "chains": [frame["chain_name"].nunique() for frame in split_frames.values()],
        }
    )


def save_splits(split_frames, split_dir=None):
    """Persist each split to CSV so training can reuse the same split."""
    output_dir = Path(split_dir) if split_dir else default_split_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, frame in split_frames.items():
        frame.to_csv(output_dir / f"{split_name}.csv", index=False)

    split_summary_frame(split_frames).to_csv(output_dir / "split_summary.csv", index=False)
    return output_dir


def prepare_datasets(
    csv_path=None,
    split_dir=None,
    random_state=RANDOM_STATE,
    save=True,
    feature_batch="original",
):
    """Run the full v4 data-prep pipeline.

    Returns a dictionary bundle so training code can stay simple and
    avoid passing many separate objects around.
    """
    batch_config = feature_batch_config(feature_batch)
    full_df = add_derived_columns(load_master_data(csv_path))
    split_frames = split_data(full_df, random_state=random_state)

    # Runtime normalization is based on current-step runtime from the training split only.
    train_runtime_99 = float(split_frames["train"]["current_step_runtime"].quantile(0.99))
    runtime_denominator = np.log1p(max(train_runtime_99, 1e-9))

    full_df = add_runtime_score(full_df, runtime_denominator)
    split_frames = {
        split_name: add_runtime_score(frame, runtime_denominator)
        for split_name, frame in split_frames.items()
    }

    if save:
        save_splits(split_frames, split_dir=split_dir)

    static_feature_columns = list(batch_config["static_feature_columns"])
    dynamic_feature_columns = list(batch_config["dynamic_feature_columns"])

    return {
        "full_df": full_df,
        "train_df": split_frames["train"],
        "validation_df": split_frames["validation"],
        "test_df": split_frames["test"],
        "optimizer_names": OPTIMIZER_NAMES,
        "category_names": (
            sorted(split_frames["train"]["original_category"].dropna().unique())
            + [UNKNOWN_CATEGORY]
            if batch_config["include_category"]
            else []
        ),
        "feature_batch": feature_batch,
        "feature_batch_config": batch_config,
        "metric_scale": compute_metric_scale(split_frames["train"]),
        "baseline_feature_columns": list(BASELINE_FEATURE_COLUMNS),
        "static_feature_columns": static_feature_columns,
        "dynamic_feature_columns": dynamic_feature_columns,
        "observation_feature_columns": (
            list(BASELINE_FEATURE_COLUMNS)
            + static_feature_columns
            + dynamic_feature_columns
        ),
        "observation_feature_scale": compute_observation_feature_scale(
            split_frames["train"],
            static_feature_columns=static_feature_columns,
            dynamic_feature_columns=dynamic_feature_columns,
        ),
        "runtime_reference_seconds": train_runtime_99,
        "runtime_denominator": runtime_denominator,
        "split_summary": split_summary_frame(split_frames),
    }


if __name__ == "__main__":
    # Handy direct entry point for regenerating cached split files.
    bundle = prepare_datasets(save=True)
    print(bundle["split_summary"])
