"""Dataset preparation utilities for the v2 two-step chain RL setup.

This module is responsible for:
- loading the master CSV,
- adding the derived columns used by rewards and analysis,
- creating train/validation/test splits by original circuit, and
- saving those split CSVs for reuse.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

try:
    from .environment import METRIC_NAMES
except ImportError:
    from environment import METRIC_NAMES


# Keep the defaults near the top so experiments can change them easily.
RANDOM_STATE = 42
OPTIMIZER_NAMES = [
    "qiskit_standard",
    "qiskit_ai",
    "tket",
    "wisq_rules",
    "wisq_bqskit",
]
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

    while not (current / "dbs" / "original_data" / "one_and_two_opt_master.csv").exists():
        if current.parent == current:
            raise FileNotFoundError("Could not locate project root.")
        current = current.parent
    return current


def default_master_csv_path():
    """Return the default v2 source CSV path."""
    return find_project_root() / "dbs" / "original_data" / "one_and_two_opt_master.csv"


def default_split_dir():
    """Return the default location for cached split CSVs."""
    return Path(__file__).resolve().parent / "splits"


def load_master_data(csv_path=None):
    """Load the raw master CSV and validate required columns."""
    path = Path(csv_path) if csv_path else default_master_csv_path()
    frame = pd.read_csv(path)

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return frame


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

    # Absolute and percentage depth improvement.
    df["depth_reduction"] = df["initial_depth"] - df["final_depth"]
    df["depth_reduction_pct"] = (
        100 * df["depth_reduction"] / df["initial_depth"].replace(0, np.nan)
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
    df["current_step_depth_reduction_pct"] = (
        100
        * df["current_step_depth_reduction"]
        / df["current_step_parent_depth"].replace(0, np.nan)
    )

    # Keep the old names as aliases to the total-vs-original version so
    # any downstream reporting that still expects them continues to work.
    df["two_qubit_gate_reduction"] = df["total_two_qubit_gate_reduction"]
    df["two_qubit_gate_reduction_pct"] = df["total_two_qubit_gate_reduction_pct"]

    # Drop rows that cannot support the current reward calculation.
    df = df.dropna(
        subset=[
            "total_two_qubit_gate_reduction_pct",
            "current_step_two_qubit_gate_reduction_pct",
            "current_step_depth_reduction_pct",
            "total_runtime_seconds",
            "current_step_runtime",
            "final_two_qubit_gates",
        ]
    ).copy()

    return df


def split_data(frame, random_state=RANDOM_STATE):
    """Split by original circuit so the same circuit never crosses splits."""
    circuits = np.array(sorted(frame["original_circuit"].unique()))

    train_val_circuits, test_circuits = train_test_split(
        circuits,
        test_size=0.10,
        random_state=random_state,
    )
    train_circuits, validation_circuits = train_test_split(
        train_val_circuits,
        test_size=0.10 / 0.90,
        random_state=random_state,
    )

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


def prepare_datasets(csv_path=None, split_dir=None, random_state=RANDOM_STATE, save=True):
    """Run the full v2 data-prep pipeline.

    Returns a dictionary bundle so training code can stay simple and
    avoid passing many separate objects around.
    """
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

    return {
        "full_df": full_df,
        "train_df": split_frames["train"],
        "validation_df": split_frames["validation"],
        "test_df": split_frames["test"],
        "optimizer_names": OPTIMIZER_NAMES,
        "category_names": sorted(full_df["original_category"].unique()),
        "metric_scale": compute_metric_scale(split_frames["train"]),
        "runtime_reference_seconds": train_runtime_99,
        "runtime_denominator": runtime_denominator,
        "split_summary": split_summary_frame(split_frames),
    }


if __name__ == "__main__":
    # Handy direct entry point for regenerating cached split files.
    bundle = prepare_datasets(save=True)
    print(bundle["split_summary"])
