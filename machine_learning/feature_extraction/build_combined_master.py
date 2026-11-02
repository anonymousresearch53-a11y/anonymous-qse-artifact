#!/usr/bin/env python3
"""Build source masters and the canonical combined master with extracted features."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .extract_combined_master_features import (
        build_cleaned_master_with_features,
        clean_text,
        safe_divide,
    )
except ImportError:
    from extract_combined_master_features import (  # type: ignore
        build_cleaned_master_with_features,
        clean_text,
        safe_divide,
    )


OPTIMIZERS = ["wisq_rules", "wisq_bqskit", "qiskit_standard", "tket"]
RUNTIME_THRESHOLD_SECONDS = 900.0
OLD_PREFIX = (
    "./"
    "quantum-circuit-chaining/"
)
ALLOWED_GATES = ("rz", "sx", "x", "cx")
RARE_CATEGORY_MIN_CIRCUITS = 3

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DBS_DIR = REPO_ROOT / "dbs"
COMBINED_FIXED_DIR = DBS_DIR / "combined_fixed"
FEATURE_OUTPUT_CSV = SCRIPT_DIR / "combined_master_features.csv"
FEATURE_SUMMARY_JSON = SCRIPT_DIR / "combined_master_features_summary.json"
BUILD_SUMMARY_JSON = SCRIPT_DIR / "combined_master_build_summary.json"
CLEANED_MASTER_OUTPUT = SCRIPT_DIR / "combined_master_cleaned_with_features.csv"
COMBINED_MASTER_OUTPUT = DBS_DIR / "combined_master.csv"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    source_dir: Path
    parse_dir: Path
    single_db_path: Path
    two_step_db_path: Path
    prefix_replacements: tuple[tuple[str, str], ...] = ()
    deduplicate_single_runs: bool = False

    @property
    def original_circuits_csv(self) -> Path:
        return self.source_dir / "original_circuits.csv"

    @property
    def single_master_csv(self) -> Path:
        return self.source_dir / "single_opt_master.csv"

    @property
    def combined_master_csv(self) -> Path:
        return self.source_dir / "one_and_two_opt_master.csv"

    @property
    def error_csv(self) -> Path:
        return self.source_dir / "error.csv"

    @property
    def one_opt_circuits_csv(self) -> Path:
        return self.parse_dir / "one_opt_circuits.csv"


@dataclass
class SourceBuildResult:
    config: SourceConfig
    original_circuit_df: pd.DataFrame
    two_step_circuit_df: pd.DataFrame
    single_master: pd.DataFrame
    two_step_master: pd.DataFrame
    combined_master: pd.DataFrame
    summary: pd.DataFrame


COMBINED_FIXED_CONFIG = SourceConfig(
    name="combined_fixed",
    source_dir=COMBINED_FIXED_DIR,
    parse_dir=COMBINED_FIXED_DIR / "parse_dbs",
    single_db_path=COMBINED_FIXED_DIR / "data_single_opt" / "combined_single_opt.db",
    two_step_db_path=COMBINED_FIXED_DIR / "data_second_opt" / "combined_second_opt.db",
    prefix_replacements=(
        (OLD_PREFIX, str(REPO_ROOT) + "/"),
        ("mqt_artifacts/", str(REPO_ROOT / "data" / "artifacts") + "/"),
        ("mqt/artifacts/", str(REPO_ROOT / "mqt" / "artifacts") + "/"),
        ("data/artifacts/", str(REPO_ROOT / "data" / "artifacts") + "/"),
    ),
)

SOURCE_CONFIGS = {
    COMBINED_FIXED_CONFIG.name: COMBINED_FIXED_CONFIG,
}

MANUAL_CATEGORY_OVERRIDES = {
    "guoq_0410184_169": "logic",
    "guoq_3_17_13": "logic",
    "guoq_4_49_16": "logic",
    "guoq_qiskit-3_17_13": "logic",
    "guoq_ex-1_166": "logic",
    "guoq_cnt3-5_179": "counter",
    "guoq_cnt3-5_180": "counter",
    "guoq_max46_240": "arithmetic_logic",
    "guoq_mini-alu_167": "alu",
    "guoq_mini_alu_305": "alu",
    "guoq_qiskit-alu-v0_26": "alu",
    "guoq_qe_qft_4": "qft",
    "guoq_qe_qft_5": "qft",
}


def strip_source_prefix(name: str) -> str:
    if name.startswith("mqt_"):
        return name[4:]
    if name.startswith("guoq_"):
        return name[5:]
    return name


def infer_source_repository(original_circuit: object, legacy_category: object) -> str:
    circuit_name = clean_text(original_circuit).lower()
    if circuit_name.startswith("mqt_"):
        return "mqt"
    if circuit_name.startswith("guoq_"):
        return "guoq"

    legacy_name = clean_text(legacy_category)
    if legacy_name in {
        "feynman",
        "clifford",
        "qaoa",
        "qft",
        "qv",
        "square-heisenberg",
    }:
        return "benchpress"
    if legacy_name == "misc":
        return "local_qiskit"
    return "unknown"


def infer_circuit_family(original_circuit: object) -> str:
    circuit_name = clean_text(original_circuit)
    if circuit_name in MANUAL_CATEGORY_OVERRIDES:
        return MANUAL_CATEGORY_OVERRIDES[circuit_name]

    name = strip_source_prefix(circuit_name.lower().replace("-", "_"))

    exact_prefixes = [
        "bmw_quark_cardinality",
        "bmw_quark_copula",
        "cdkm_ripple_carry_adder",
        "draper_qft_adder",
        "full_adder",
        "half_adder",
        "modular_adder",
        "vbe_ripple_carry_adder",
        "rg_qft_multiplier",
        "hrs_cumulative_multiplier",
        "ground_state_estimation",
        "portfolioqaoa",
        "portfoliovqe",
        "qftentangled",
        "qpeexact",
        "qpeinexact",
        "randomcircuit",
        "graphstate",
        "efficient_su2",
        "real_amplitudes",
        "square_heisenberg",
    ]
    for prefix in exact_prefixes:
        if name.startswith(prefix):
            return prefix

    simple_prefixes = [
        "ae",
        "bv",
        "dj",
        "ghz",
        "grover",
        "qaoa",
        "qft",
        "qv",
        "vqe",
        "qnn",
        "qwalk",
        "wstate",
        "tfim",
        "xy",
        "heisenberg",
        "ising_model",
        "shor",
        "qgan",
        "alu",
        "hwb",
        "ham",
        "tof",
        "barenco_tof",
        "csla_mux",
        "rc_adder",
        "vbe_adder",
        "adder",
        "decod24",
        "graycode",
        "mod5",
        "mod_mult",
        "mod_red",
        "clifford",
        "sym",
        "urf",
        "rd",
        "sqn",
        "sao2",
        "gf2^4_mult",
        "gf2^5_mult",
        "gf2^6_mult",
        "life",
        "majority",
        "miller",
        "misex1",
        "mlp4",
        "pm1",
        "co14",
        "con1",
        "inc",
        "clip",
        "dist",
        "cycle",
        "c17",
        "adr4",
        "aj_e11",
        "cm152a",
        "cm42a",
        "cm82a",
        "cm85a",
        "dc1",
        "dc2",
        "ex1",
        "ex2",
        "ex3",
        "f2",
        "hlf",
        "radd",
        "hhl",
    ]
    for prefix in simple_prefixes:
        if name.startswith(prefix):
            return prefix

    if "_alg_" in name:
        return name.split("_alg_", 1)[0]
    if "_step_" in name:
        return name.split("_step_", 1)[0]
    if "_nativegates_" in name:
        return name.split("_nativegates_", 1)[0]

    if name.startswith("4gt"):
        return "4gt"
    if name.startswith("one_two_three"):
        return "one_two_three"
    if "mod" in name:
        return "modular_arithmetic"
    if "multiply" in name or "muliply" in name or "multiplier" in name:
        return "multiplier"
    if "square_root" in name or name.startswith("sqrt") or name.startswith("squar"):
        return "square_root"
    if name.startswith("root"):
        return "root"
    if name.startswith("xor"):
        return "xor"
    if name.startswith("z4"):
        return "z4"
    if name.startswith("wim"):
        return "wim"
    if name.startswith("sf"):
        return "sf"
    if name.startswith("sys6"):
        return "sys6"
    if name.startswith("9symml"):
        return "9symml"

    return "other"


def apply_category_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    mapped = frame.copy()
    mapped["source_repository"] = [
        infer_source_repository(circuit, category)
        for circuit, category in zip(
            mapped["original_circuit"],
            mapped["original_category"],
        )
    ]
    mapped["_inferred_category"] = mapped["original_circuit"].map(infer_circuit_family)

    category_counts = (
        mapped[["original_circuit", "_inferred_category"]]
        .drop_duplicates()["_inferred_category"]
        .value_counts()
    )
    rare_categories = set(category_counts[category_counts < RARE_CATEGORY_MIN_CIRCUITS].index)
    mapped["original_category"] = mapped["_inferred_category"].where(
        ~mapped["_inferred_category"].isin(rare_categories),
        "misc",
    )
    mapped = mapped.drop(columns=["_inferred_category"])
    return mapped


def normalize_repo_path(value: object, config: SourceConfig) -> object:
    if not isinstance(value, str) or not value:
        return value

    updated = value
    for old_prefix, new_prefix in config.prefix_replacements:
        if updated.startswith(old_prefix):
            return updated.replace(old_prefix, new_prefix, 1)
    return updated


def normalize_path_columns(
    frame: pd.DataFrame, columns: list[str], config: SourceConfig
) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: normalize_repo_path(value, config))
    return frame


def deduplicate_single_opt_runs(opt_runs: pd.DataFrame) -> pd.DataFrame:
    dup_cols = ["circuit_id", "optimizer_id"]
    return (
        opt_runs.assign(
            _orig_order=np.arange(len(opt_runs)),
            _success_rank=opt_runs["success"].eq(1).astype(int),
        )
        .sort_values(
            dup_cols + ["_success_rank", "_orig_order"],
            ascending=[True] * len(dup_cols) + [False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=dup_cols, keep="first")
        .sort_values("_orig_order")
        .drop(columns=["_orig_order", "_success_rank"])
        .reset_index(drop=True)
    )


def load_single_source_data(config: SourceConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(config.single_db_path) as conn:
        circuit_df = pd.read_sql_query("SELECT * FROM circuits;", conn)
        opt_runs = pd.read_sql_query("SELECT * FROM optimization_runs;", conn)
        optimizer_df = pd.read_sql_query("SELECT id, name FROM optimizers;", conn)

    optimizer_lookup = optimizer_df.set_index("id")["name"].to_dict()
    opt_runs["optimizer_name"] = opt_runs["optimizer_id"].map(optimizer_lookup)
    opt_runs = opt_runs.loc[opt_runs["optimizer_name"].isin(OPTIMIZERS)].copy()

    circuit_df["optimizer_chain"] = circuit_df["name"].str.split("__").str[1:].str.join("__")
    circuit_df["original_circuit"] = (
        circuit_df["name"].str.split("__").str[0].str.replace(r"^(artifact_)+", "", regex=True)
    )
    circuit_df = normalize_path_columns(circuit_df, ["qasm_path"], config)
    opt_runs = normalize_path_columns(opt_runs, ["artifact_path"], config)
    if config.deduplicate_single_runs:
        opt_runs = deduplicate_single_opt_runs(opt_runs)
    return circuit_df, opt_runs


def load_two_step_source_data(config: SourceConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(config.two_step_db_path) as conn:
        circuit_df = pd.read_sql_query("SELECT * FROM circuits;", conn)
        opt_runs = pd.read_sql_query("SELECT * FROM optimization_runs;", conn)
        optimizer_df = pd.read_sql_query("SELECT id, name FROM optimizers;", conn)

    optimizer_lookup = optimizer_df.set_index("id")["name"].to_dict()
    opt_runs["optimizer_name"] = opt_runs["optimizer_id"].map(optimizer_lookup)
    opt_runs = opt_runs.loc[opt_runs["optimizer_name"].isin(OPTIMIZERS)].copy()
    circuit_df["optimizer_chain"] = circuit_df["name"].str.split("__").str[1:].str.join("__")
    if "original_circuit" not in circuit_df.columns:
        circuit_df["original_circuit"] = (
            circuit_df["name"].str.split("__").str[0].str.replace(r"^(artifact_)+", "", regex=True)
        )
    else:
        circuit_df["original_circuit"] = (
            circuit_df["original_circuit"]
            .fillna(circuit_df["name"].str.split("__").str[0])
            .astype(str)
            .str.replace(r"^(artifact_)+", "", regex=True)
        )
    circuit_df = normalize_path_columns(circuit_df, ["qasm_path"], config)
    opt_runs = normalize_path_columns(opt_runs, ["artifact_path"], config)
    return circuit_df, opt_runs


def build_original_circuits(single_circuit_df: pd.DataFrame) -> pd.DataFrame:
    return (
        single_circuit_df[single_circuit_df["name"] == single_circuit_df["original_circuit"]]
        .copy()
        .sort_values("name")
        .reset_index(drop=True)
    )


def build_single_master(circuit_df: pd.DataFrame, opt_runs: pd.DataFrame) -> pd.DataFrame:
    circuits_by_id = circuit_df.set_index("id")
    originals_by_name = circuit_df.set_index("name")

    rows: list[dict[str, object]] = []
    for index, row in opt_runs.iterrows():
        source_circuit = circuits_by_id.loc[row["circuit_id"]]
        original_circuit = source_circuit["original_circuit"]
        original_row = originals_by_name.loc[original_circuit]
        optimizer = clean_text(row.get("optimizer_name"))
        if optimizer not in OPTIMIZERS:
            raise KeyError(f"Unexpected optimizer for single-step row: {optimizer!r}")

        rows.append(
            {
                "id": index,
                "original_circuit_id": original_row["id"],
                "original_circuit": original_circuit,
                "original_category": original_row["category"],
                "original_circuit_path": original_row["qasm_path"],
                "parent_circuit_id": original_row["id"],
                "parent_circuit": original_circuit,
                "parent_circuit_path": original_row["qasm_path"],
                "chain_name": optimizer,
                "artifact_path": row["artifact_path"],
                "opt_level": 1,
                "opt_1": optimizer,
                "opt_2": pd.NA,
                "opt_3": pd.NA,
                "initial_num_qubits": original_row["num_qubits"],
                "initial_depth": original_row["initial_depth"],
                "initial_two_qubit_gates": original_row["initial_two_qubit_gates"],
                "initial_total_gates": original_row["initial_total_gates"],
                "initial_two_qubit_depth": original_row["initial_two_qubit_depth"],
                "initial_gate_density": original_row["gate_density"],
                "initial_two_qubit_ratio": original_row["two_qubit_ratio"],
                "final_num_qubits": original_row["num_qubits"],
                "final_depth": row["output_depth"],
                "final_two_qubit_gates": row["output_two_qubit_gates"],
                "final_total_gates": row["output_total_gates"],
                "final_two_qubit_depth": row["output_two_qubit_depth"],
                "total_runtime_seconds": row["duration_seconds"],
                "current_step_runtime": row["duration_seconds"],
                "error": bool(row["success"] == 0),
            }
        )

    frame = pd.DataFrame(rows)
    frame["opt_chain"] = frame["original_circuit"] + "__" + frame["opt_1"]
    return frame


def final_initial_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for final_column in frame.columns:
        if not final_column.startswith("final_"):
            continue
        initial_column = f"initial_{final_column[len('final_'):]}"
        if initial_column in frame.columns:
            pairs.append((final_column, initial_column))
    return pairs


def normalize_single_failures(frame: pd.DataFrame, runtime_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    failure_mask = frame["error"].fillna(False) | (frame["current_step_runtime"] >= runtime_threshold)

    for final_column, initial_column in final_initial_pairs(frame):
        frame.loc[failure_mask, final_column] = frame.loc[failure_mask, initial_column]

    frame.loc[failure_mask, "current_step_runtime"] = 0.0
    frame.loc[failure_mask, "total_runtime_seconds"] = 0.0
    frame.loc[failure_mask, "artifact_path"] = frame.loc[failure_mask, "original_circuit_path"]
    frame.loc[failure_mask, "parent_circuit_id"] = frame.loc[failure_mask, "original_circuit_id"]
    frame.loc[failure_mask, "parent_circuit"] = frame.loc[failure_mask, "original_circuit"]
    frame.loc[failure_mask, "parent_circuit_path"] = frame.loc[failure_mask, "original_circuit_path"]
    frame.loc[failure_mask, "error"] = True

    missing_artifact_mask = frame["artifact_path"].isna()
    frame.loc[missing_artifact_mask, "artifact_path"] = frame.loc[missing_artifact_mask, "original_circuit_path"]
    return frame


def build_two_step_master(
    two_step_circuit_df: pd.DataFrame,
    opt_runs: pd.DataFrame,
    original_circuit_df: pd.DataFrame,
    cleaned_single_master: pd.DataFrame,
) -> pd.DataFrame:
    circuits_by_id = two_step_circuit_df.set_index("id")
    originals_by_name = original_circuit_df.set_index("name")
    single_by_opt_chain = cleaned_single_master.set_index("opt_chain")

    rows: list[dict[str, object]] = []
    for index, row in opt_runs.iterrows():
        parent_source = circuits_by_id.loc[row["circuit_id"]]
        parent_circuit = parent_source["name"]
        original_name = clean_text(parent_source.get("original_circuit")) or parent_circuit.split("__")[0]
        original_row = originals_by_name.loc[original_name]

        if parent_circuit not in single_by_opt_chain.index:
            raise KeyError(f"Missing parent single-step row for {parent_circuit}")
        parent_single_row = single_by_opt_chain.loc[parent_circuit]
        if isinstance(parent_single_row, pd.DataFrame):
            raise ValueError(f"Multiple parent rows found for {parent_circuit}")

        opt_1 = parent_circuit.split("__")[1]
        opt_2 = clean_text(row.get("optimizer_name"))
        if opt_2 not in OPTIMIZERS:
            raise KeyError(f"Unexpected optimizer for two-step row: {opt_2!r}")

        rows.append(
            {
                "id": index,
                "original_circuit_id": original_row["id"],
                "original_circuit": original_name,
                "original_category": original_row["category"],
                "original_circuit_path": original_row["qasm_path"],
                "parent_circuit_id": row["circuit_id"],
                "parent_circuit": parent_circuit,
                "parent_circuit_path": parent_source["qasm_path"],
                "chain_name": f"{opt_1}__{opt_2}",
                "artifact_path": row["artifact_path"],
                "opt_level": 2,
                "opt_1": opt_1,
                "opt_2": opt_2,
                "opt_3": pd.NA,
                "initial_num_qubits": original_row["num_qubits"],
                "initial_depth": original_row["initial_depth"],
                "initial_two_qubit_gates": original_row["initial_two_qubit_gates"],
                "initial_total_gates": original_row["initial_total_gates"],
                "initial_two_qubit_depth": original_row["initial_two_qubit_depth"],
                "initial_gate_density": original_row["gate_density"],
                "initial_two_qubit_ratio": original_row["two_qubit_ratio"],
                "final_num_qubits": original_row["num_qubits"],
                "final_depth": row["output_depth"],
                "final_two_qubit_gates": row["output_two_qubit_gates"],
                "final_total_gates": row["output_total_gates"],
                "final_two_qubit_depth": row["output_two_qubit_depth"],
                "total_runtime_seconds": row["duration_seconds"] + parent_single_row["total_runtime_seconds"],
                "current_step_runtime": row["duration_seconds"],
                "error": bool(row["success"] == 0),
            }
        )

    frame = pd.DataFrame(rows)
    frame["opt_chain"] = frame["original_circuit"] + "__" + frame["opt_1"] + "__" + frame["opt_2"]
    return frame


def add_missing_children_for_failed_singles(
    two_step_master: pd.DataFrame,
    cleaned_single_master: pd.DataFrame,
) -> pd.DataFrame:
    two_step_master = two_step_master.copy()
    existing_chains = set(two_step_master["opt_chain"])
    single_lookup = cleaned_single_master.set_index(["original_circuit", "opt_1"])
    failed_single_rows = cleaned_single_master.loc[cleaned_single_master["error"].fillna(False)].copy()

    synthetic_rows: list[dict[str, object]] = []
    for _, failed_row in failed_single_rows.iterrows():
        for opt_2 in OPTIMIZERS:
            opt_chain = f"{failed_row['original_circuit']}__{failed_row['opt_1']}__{opt_2}"
            if opt_chain in existing_chains:
                continue

            proxy_key = (failed_row["original_circuit"], opt_2)
            if proxy_key not in single_lookup.index:
                raise KeyError(f"Missing single-step proxy row for {proxy_key}")

            proxy_row = single_lookup.loc[proxy_key]
            if isinstance(proxy_row, pd.DataFrame):
                raise ValueError(f"Multiple proxy rows found for {proxy_key}")

            synthetic_rows.append(
                {
                    "id": -1,
                    "original_circuit_id": failed_row["original_circuit_id"],
                    "original_circuit": failed_row["original_circuit"],
                    "original_category": failed_row["original_category"],
                    "original_circuit_path": failed_row["original_circuit_path"],
                    "parent_circuit_id": failed_row["original_circuit_id"],
                    "parent_circuit": failed_row["original_circuit"],
                    "parent_circuit_path": failed_row["original_circuit_path"],
                    "chain_name": f"{failed_row['opt_1']}__{opt_2}",
                    "artifact_path": proxy_row["artifact_path"],
                    "opt_level": 2,
                    "opt_1": failed_row["opt_1"],
                    "opt_2": opt_2,
                    "opt_3": pd.NA,
                    "initial_num_qubits": proxy_row["initial_num_qubits"],
                    "initial_depth": proxy_row["initial_depth"],
                    "initial_two_qubit_gates": proxy_row["initial_two_qubit_gates"],
                    "initial_total_gates": proxy_row["initial_total_gates"],
                    "initial_two_qubit_depth": proxy_row["initial_two_qubit_depth"],
                    "initial_gate_density": proxy_row["initial_gate_density"],
                    "initial_two_qubit_ratio": proxy_row["initial_two_qubit_ratio"],
                    "final_num_qubits": proxy_row["final_num_qubits"],
                    "final_depth": proxy_row["final_depth"],
                    "final_two_qubit_gates": proxy_row["final_two_qubit_gates"],
                    "final_total_gates": proxy_row["final_total_gates"],
                    "final_two_qubit_depth": proxy_row["final_two_qubit_depth"],
                    "total_runtime_seconds": proxy_row["total_runtime_seconds"],
                    "current_step_runtime": proxy_row["current_step_runtime"],
                    "error": bool(proxy_row["error"]),
                    "opt_chain": opt_chain,
                }
            )
            existing_chains.add(opt_chain)

    if synthetic_rows:
        two_step_master = pd.concat([two_step_master, pd.DataFrame(synthetic_rows)], ignore_index=True)
    return two_step_master


def normalize_two_step_failures(
    frame: pd.DataFrame,
    cleaned_single_master: pd.DataFrame,
    runtime_threshold: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["error"] = frame["error"].fillna(False).astype(bool)

    single_lookup = cleaned_single_master.set_index(["original_circuit", "opt_1"])
    single_by_chain = cleaned_single_master.set_index("opt_chain")
    failed_single_keys = set(
        cleaned_single_master.loc[cleaned_single_master["error"], ["original_circuit", "opt_1"]]
        .itertuples(index=False, name=None)
    )
    final_columns = [final_column for final_column, _ in final_initial_pairs(frame)]

    rerouted_rows = 0
    rolled_back_rows = 0
    for idx, row in frame.iterrows():
        parent_failed = (row["original_circuit"], row["opt_1"]) in failed_single_keys
        long_runtime = row["current_step_runtime"] >= runtime_threshold
        explicit_error = bool(row["error"])

        if parent_failed:
            proxy_key = (row["original_circuit"], row["opt_2"])
            if proxy_key not in single_lookup.index:
                raise KeyError(f"Missing reroute proxy row for {proxy_key}")
            proxy_row = single_lookup.loc[proxy_key]
            if isinstance(proxy_row, pd.DataFrame):
                raise ValueError(f"Multiple reroute proxy rows found for {proxy_key}")

            for column in (
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
                "artifact_path",
            ):
                frame.at[idx, column] = proxy_row[column]

            frame.at[idx, "parent_circuit_id"] = row["original_circuit_id"]
            frame.at[idx, "parent_circuit"] = row["original_circuit"]
            frame.at[idx, "parent_circuit_path"] = row["original_circuit_path"]
            frame.at[idx, "error"] = bool(proxy_row["error"])
            rerouted_rows += 1
            continue

        if explicit_error or long_runtime:
            parent_chain = f"{row['original_circuit']}__{row['opt_1']}"
            if parent_chain not in single_by_chain.index:
                raise KeyError(f"Missing single-step parent row for {parent_chain}")
            parent_row = single_by_chain.loc[parent_chain]
            if isinstance(parent_row, pd.DataFrame):
                raise ValueError(f"Multiple parent rows found for {parent_chain}")

            for column in final_columns + ["artifact_path"]:
                frame.at[idx, column] = parent_row[column]

            frame.at[idx, "parent_circuit"] = parent_chain
            frame.at[idx, "parent_circuit_path"] = parent_row["artifact_path"]
            frame.at[idx, "total_runtime_seconds"] = parent_row["total_runtime_seconds"]
            frame.at[idx, "current_step_runtime"] = 0.0
            frame.at[idx, "error"] = True
            rolled_back_rows += 1

    missing_artifact_mask = frame["artifact_path"].isna()
    frame.loc[missing_artifact_mask, "artifact_path"] = frame.loc[missing_artifact_mask, "original_circuit_path"]
    frame.attrs["rerouted_rows"] = rerouted_rows
    frame.attrs["rolled_back_rows"] = rolled_back_rows
    return frame


def verify_source_master_outputs(
    cleaned_single_master: pd.DataFrame,
    cleaned_two_step_master: pd.DataFrame,
    combined_master: pd.DataFrame,
    runtime_threshold: float,
) -> pd.DataFrame:
    assert cleaned_single_master["opt_chain"].is_unique, "single-step opt_chain must be unique"
    assert combined_master["opt_chain"].is_unique, "combined opt_chain must be unique"
    assert not combined_master["artifact_path"].isna().any(), "artifact_path must be non-null"
    assert not combined_master["original_circuit_path"].isna().any(), "original_circuit_path must be non-null"
    assert not (combined_master["current_step_runtime"] >= runtime_threshold).any(), "long runtime rows remain"

    single_counts = cleaned_single_master.groupby("original_circuit")["opt_chain"].nunique()
    assert (
        single_counts == len(OPTIMIZERS)
    ).all(), "each original circuit must have 4 single-step rows"

    combined_counts = combined_master.groupby("original_circuit")["opt_chain"].nunique()
    expected_combined = len(OPTIMIZERS) + len(OPTIMIZERS) * len(OPTIMIZERS)
    assert (
        combined_counts == expected_combined
    ).all(), "each original circuit must have 20 unique rows"

    failed_single_mask = cleaned_single_master["error"].fillna(False)
    for final_column, initial_column in final_initial_pairs(cleaned_single_master):
        assert (
            cleaned_single_master.loc[failed_single_mask, final_column]
            == cleaned_single_master.loc[failed_single_mask, initial_column]
        ).all(), f"single failure mismatch in {final_column}"

    assert (cleaned_single_master.loc[failed_single_mask, "current_step_runtime"] == 0).all()
    assert (cleaned_single_master.loc[failed_single_mask, "total_runtime_seconds"] == 0).all()

    single_lookup = cleaned_single_master.set_index(["original_circuit", "opt_1"])
    failed_single_keys = set(
        cleaned_single_master.loc[cleaned_single_master["error"], ["original_circuit", "opt_1"]]
        .itertuples(index=False, name=None)
    )
    child_of_failed_parent_mask = cleaned_two_step_master.apply(
        lambda row: (row["original_circuit"], row["opt_1"]) in failed_single_keys,
        axis=1,
    )
    for _, row in cleaned_two_step_master.loc[child_of_failed_parent_mask].iterrows():
        proxy_row = single_lookup.loc[(row["original_circuit"], row["opt_2"])]
        assert row["final_depth"] == proxy_row["final_depth"]
        assert row["current_step_runtime"] == proxy_row["current_step_runtime"]
        assert row["total_runtime_seconds"] == proxy_row["total_runtime_seconds"]
        assert bool(row["error"]) is bool(proxy_row["error"])
        assert row["parent_circuit"] == row["original_circuit"]

    return pd.DataFrame(
        [
            {
                "single_rows": len(cleaned_single_master),
                "two_step_rows": len(cleaned_two_step_master),
                "combined_rows": len(combined_master),
                "failed_single_rows": int(cleaned_single_master["error"].sum()),
                "failed_two_step_rows": int(cleaned_two_step_master["error"].sum()),
                "rerouted_two_step_rows": int(cleaned_two_step_master.attrs.get("rerouted_rows", 0)),
                "rolled_back_two_step_rows": int(cleaned_two_step_master.attrs.get("rolled_back_rows", 0)),
            }
        ]
    )


def build_source_result(config: SourceConfig) -> SourceBuildResult:
    single_circuit_df, single_opt_runs = load_single_source_data(config)
    original_circuit_df = build_original_circuits(single_circuit_df)

    single_master = build_single_master(single_circuit_df, single_opt_runs)
    cleaned_single_master = normalize_single_failures(single_master, RUNTIME_THRESHOLD_SECONDS)

    two_step_circuit_df, two_step_opt_runs = load_two_step_source_data(config)
    raw_two_step_master = build_two_step_master(
        two_step_circuit_df,
        two_step_opt_runs,
        original_circuit_df,
        cleaned_single_master,
    )
    two_step_with_synthetics = add_missing_children_for_failed_singles(
        raw_two_step_master,
        cleaned_single_master,
    )
    cleaned_two_step_master = normalize_two_step_failures(
        two_step_with_synthetics,
        cleaned_single_master,
        RUNTIME_THRESHOLD_SECONDS,
    )

    combined_master = pd.concat([cleaned_single_master, cleaned_two_step_master], ignore_index=True)
    combined_master["id"] = range(len(combined_master))
    summary = verify_source_master_outputs(
        cleaned_single_master,
        cleaned_two_step_master,
        combined_master,
        RUNTIME_THRESHOLD_SECONDS,
    )

    for frame in (
        original_circuit_df,
        two_step_circuit_df,
        cleaned_single_master,
        cleaned_two_step_master,
        combined_master,
    ):
        frame["_dataset_source"] = config.name

    return SourceBuildResult(
        config=config,
        original_circuit_df=original_circuit_df,
        two_step_circuit_df=two_step_circuit_df,
        single_master=cleaned_single_master,
        two_step_master=cleaned_two_step_master,
        combined_master=combined_master,
        summary=summary,
    )


def gate_count_sum(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    columns = [f"{prefix}{gate}_gate_count" for gate in ALLOWED_GATES]
    return frame[columns].fillna(0).sum(axis=1)


def identify_invalid_original_circuits(cleaned_df: pd.DataFrame) -> dict[str, set[str]]:
    final_allowed_sum = gate_count_sum(cleaned_df)
    initial_allowed_sum = gate_count_sum(cleaned_df, prefix="initial_")

    final_total = pd.to_numeric(cleaned_df["loaded_total_gate_count"], errors="coerce")
    initial_total = pd.to_numeric(cleaned_df["initial_total_gates"], errors="coerce")
    initial_two_qubit = pd.to_numeric(cleaned_df["initial_two_qubit_gates"], errors="coerce")

    final_gate_violation = cleaned_df.loc[
        ~np.isclose(final_total.fillna(-1), final_allowed_sum.fillna(-2)),
        "original_circuit",
    ]
    initial_gate_violation = cleaned_df.loc[
        ~np.isclose(initial_total.fillna(-1), initial_allowed_sum.fillna(-2)),
        "original_circuit",
    ]
    zero_two_qubit = cleaned_df.loc[initial_two_qubit <= 0, "original_circuit"]

    return {
        "disallowed_final_gates": set(final_gate_violation.dropna().astype(str)),
        "disallowed_initial_gates": set(initial_gate_violation.dropna().astype(str)),
        "zero_initial_two_qubit_gates": set(zero_two_qubit.dropna().astype(str)),
    }


def canonical_initial_metrics(filtered_cleaned_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "original_circuit",
        "initial_num_qubits",
        "initial_depth",
        "initial_two_qubit_gates",
        "initial_total_gates",
        "initial_two_qubit_depth",
        "initial_gate_density",
        "initial_two_qubit_ratio",
        "original_circuit_path",
        "original_category",
        "source_repository",
        "_dataset_source",
    ]
    available_columns = [column for column in columns if column in filtered_cleaned_df.columns]
    return (
        filtered_cleaned_df[available_columns]
        .drop_duplicates(subset=["original_circuit"])
        .set_index("original_circuit")
    )


def align_loaded_deltas_with_canonical_metrics(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    aligned = frame.copy()
    reference = reference_frame if reference_frame is not None else aligned

    loaded_total = pd.to_numeric(aligned["loaded_total_gate_count"], errors="coerce")
    loaded_two_qubit = pd.to_numeric(aligned["loaded_two_qubit_gate_count"], errors="coerce")
    final_total = pd.to_numeric(reference["final_total_gates"], errors="coerce")
    final_two_qubit = pd.to_numeric(reference["final_two_qubit_gates"], errors="coerce")

    aligned["loaded_total_gate_delta_vs_final"] = loaded_total - final_total
    aligned["loaded_two_qubit_gate_delta_vs_final"] = loaded_two_qubit - final_two_qubit
    aligned["loaded_counts_match_final"] = (
        np.isclose(aligned["loaded_total_gate_delta_vs_final"].fillna(np.inf), 0.0)
        & np.isclose(aligned["loaded_two_qubit_gate_delta_vs_final"].fillna(np.inf), 0.0)
    ).astype(float)
    return aligned


def write_feature_outputs(
    raw_master_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    input_csv_label: str,
) -> None:
    FEATURE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURE_OUTPUT_CSV, index=False)
    cleaned_df.to_csv(CLEANED_MASTER_OUTPUT, index=False)
    cleaned_df.to_csv(COMBINED_MASTER_OUTPUT, index=False)

    summary = {
        "input_csv": input_csv_label,
        "output_csv": str(FEATURE_OUTPUT_CSV),
        "cleaned_master_output_csv": str(CLEANED_MASTER_OUTPUT),
        "canonical_master_output_csv": str(COMBINED_MASTER_OUTPUT),
        "rows": int(len(features_df)),
        "unique_original_circuits": int(cleaned_df["original_circuit"].nunique()),
        "load_sources": {
            str(key): int(value)
            for key, value in features_df["feature_load_source"].value_counts(dropna=False).items()
        },
        "load_error_count": int(features_df["feature_load_error"].fillna("").ne("").sum()),
        "loaded_total_gate_count_mismatch_count": int(
            cleaned_df["loaded_total_gate_delta_vs_final"].fillna(0).ne(0).sum()
        ),
        "loaded_two_qubit_gate_count_mismatch_count": int(
            cleaned_df["loaded_two_qubit_gate_delta_vs_final"].fillna(0).ne(0).sum()
        ),
        "loaded_counts_match_final_count": int(
            cleaned_df["loaded_counts_match_final"].fillna(0).eq(1).sum()
        ),
        "columns": list(cleaned_df.columns),
    }
    FEATURE_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_source_outputs(
    result: SourceBuildResult,
    source_cleaned_df: pd.DataFrame,
    retained_originals: set[str],
    initial_metrics: pd.DataFrame,
) -> None:
    config = result.config
    config.source_dir.mkdir(parents=True, exist_ok=True)
    config.parse_dir.mkdir(parents=True, exist_ok=True)
    original_output_columns = [
        column for column in result.original_circuit_df.columns if column != "_dataset_source"
    ]
    if "source_repository" not in original_output_columns:
        original_output_columns.append("source_repository")
    single_output_columns = [
        column for column in result.single_master.columns if column != "_dataset_source"
    ]
    if "source_repository" not in single_output_columns:
        single_output_columns.append("source_repository")
    combined_output_columns = [
        column for column in result.combined_master.columns if column != "_dataset_source"
    ]
    if "source_repository" not in combined_output_columns:
        combined_output_columns.append("source_repository")
    two_step_circuit_columns = [
        column for column in result.two_step_circuit_df.columns if column != "_dataset_source"
    ]
    if "source_repository" not in two_step_circuit_columns:
        two_step_circuit_columns.append("source_repository")

    original_rows = (
        result.original_circuit_df.loc[result.original_circuit_df["original_circuit"].isin(retained_originals)]
        .copy()
        .sort_values("name", kind="mergesort")
        .reset_index(drop=True)
    )
    if not original_rows.empty:
        metric_map = initial_metrics.loc[original_rows["original_circuit"]]
        original_rows["num_qubits"] = metric_map["initial_num_qubits"].to_numpy()
        original_rows["initial_depth"] = metric_map["initial_depth"].to_numpy()
        original_rows["initial_two_qubit_gates"] = metric_map["initial_two_qubit_gates"].to_numpy()
        original_rows["initial_total_gates"] = metric_map["initial_total_gates"].to_numpy()
        original_rows["initial_two_qubit_depth"] = metric_map["initial_two_qubit_depth"].to_numpy()
        original_rows["gate_density"] = metric_map["initial_gate_density"].to_numpy()
        original_rows["two_qubit_ratio"] = metric_map["initial_two_qubit_ratio"].to_numpy()
        original_rows["qasm_path"] = metric_map["original_circuit_path"].to_numpy()
        original_rows["category"] = metric_map["original_category"].to_numpy()
        original_rows["source_repository"] = metric_map["source_repository"].to_numpy()
    elif "source_repository" not in original_rows.columns:
        original_rows["source_repository"] = pd.Series(dtype=object)

    source_cleaned_df = source_cleaned_df.reset_index(drop=True)
    source_combined_df = source_cleaned_df[combined_output_columns].copy()
    source_combined_df["id"] = range(len(source_combined_df))

    single_master = source_cleaned_df.loc[source_cleaned_df["opt_level"] == 1, single_output_columns].copy()

    error_df = single_master.loc[single_master["error"].fillna(False)].copy()

    two_step_circuits = (
        result.two_step_circuit_df.loc[result.two_step_circuit_df["original_circuit"].isin(retained_originals)]
        [[column for column in two_step_circuit_columns if column != "source_repository"]]
        .reset_index(drop=True)
    )
    if not two_step_circuits.empty:
        original_lookup = (
            source_cleaned_df[["original_circuit", "source_repository"]]
            .drop_duplicates(subset=["original_circuit"])
            .set_index("original_circuit")["source_repository"]
        )
        two_step_circuits["source_repository"] = (
            two_step_circuits["original_circuit"].map(original_lookup)
        )

    original_rows[original_output_columns].to_csv(config.original_circuits_csv, index=False)
    single_master.to_csv(config.single_master_csv, index=False)
    source_combined_df.to_csv(config.combined_master_csv, index=False)
    error_df.to_csv(config.error_csv, index=False)
    two_step_circuits.to_csv(config.one_opt_circuits_csv, index=False)


def write_build_summary(
    source_results: list[SourceBuildResult],
    invalid_originals_by_reason: dict[str, set[str]],
    retained_originals: set[str],
    filtered_cleaned_df: pd.DataFrame,
) -> None:
    summary = {
        "sources": {
            result.config.name: {
                "source_summary": result.summary.to_dict(orient="records")[0],
                "original_circuits_before_filter": int(result.combined_master["original_circuit"].nunique()),
                "rows_before_filter": int(len(result.combined_master)),
            }
            for result in source_results
        },
        "diagnostics": {
            reason: {
                "count": int(len(names)),
                "original_circuits": sorted(names),
            }
            for reason, names in invalid_originals_by_reason.items()
        },
        "retained_original_circuits": int(len(retained_originals)),
        "retained_rows": int(len(filtered_cleaned_df)),
        "retained_rows_per_original": (
            float(
                safe_divide(
                    float(len(filtered_cleaned_df)),
                    float(len(retained_originals)),
                )
            )
            if retained_originals
            else 0.0
        ),
        "retained_categories": {
            str(key): int(value)
            for key, value in filtered_cleaned_df["original_category"].value_counts().items()
        },
    }
    BUILD_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source masters and canonical combined master with extracted "
            "features, gate-set filtering, and zero-two-qubit filtering."
        )
    )
    parser.add_argument(
        "--source",
        choices=("all", "combined_fixed"),
        default="all",
        help="Build one source only, or the full combined dataset.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print feature-extraction progress every N rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--use-qiskit-loader",
        action="store_true",
        help="Use Qiskit QASM loading for feature extraction instead of the text parser.",
    )
    return parser.parse_args()


def selected_source_configs(source_arg: str) -> list[SourceConfig]:
    if source_arg == "all":
        return [COMBINED_FIXED_CONFIG]
    return [SOURCE_CONFIGS[source_arg]]


def main() -> None:
    args = parse_args()
    configs = selected_source_configs(args.source)

    source_results = [build_source_result(config) for config in configs]
    raw_combined_df = pd.concat([result.combined_master for result in source_results], ignore_index=True)
    raw_combined_df = apply_category_mapping(raw_combined_df)

    print(
        f"starting canonical feature extraction for {len(raw_combined_df)} rows "
        f"across {raw_combined_df['original_circuit'].nunique()} circuits",
        flush=True,
    )
    cleaned_df, features_df = build_cleaned_master_with_features(
        raw_combined_df,
        repo_root=REPO_ROOT,
        allow_fallback=False,
        strict_artifact_path=True,
        progress_every=args.progress_every,
        use_qiskit_loader=args.use_qiskit_loader,
    )

    invalid_originals_by_reason = identify_invalid_original_circuits(cleaned_df)
    invalid_originals = set(invalid_originals_by_reason["zero_initial_two_qubit_gates"])
    retained_originals = set(cleaned_df["original_circuit"]) - invalid_originals

    filtered_raw_df = (
        raw_combined_df.loc[raw_combined_df["original_circuit"].isin(retained_originals)].copy().reset_index(drop=True)
    )
    filtered_cleaned_df = (
        cleaned_df.loc[cleaned_df["original_circuit"].isin(retained_originals)].copy().reset_index(drop=True)
    )
    filtered_features_df = (
        features_df.loc[features_df["original_circuit"].isin(retained_originals)].copy().reset_index(drop=True)
    )

    filtered_cleaned_df = align_loaded_deltas_with_canonical_metrics(filtered_cleaned_df)
    filtered_features_df = align_loaded_deltas_with_canonical_metrics(
        filtered_features_df,
        reference_frame=filtered_cleaned_df,
    )

    filtered_cleaned_df["id"] = range(len(filtered_cleaned_df))
    filtered_raw_df["id"] = range(len(filtered_raw_df))

    if not filtered_cleaned_df.empty:
        combined_counts = filtered_cleaned_df.groupby("original_circuit")["opt_chain"].nunique()
        expected_combined = len(OPTIMIZERS) + len(OPTIMIZERS) * len(OPTIMIZERS)
        assert (combined_counts == expected_combined).all(), "filtered output lost optimizer rows"
        assert (filtered_cleaned_df["initial_two_qubit_gates"] > 0).all(), "zero-two-qubit circuits remain"

    initial_metrics = canonical_initial_metrics(filtered_cleaned_df)
    for result in source_results:
        source_rows = filtered_cleaned_df.loc[
            filtered_cleaned_df["_dataset_source"] == result.config.name
        ].copy()
        source_retained_originals = set(source_rows["original_circuit"])
        write_source_outputs(result, source_rows, source_retained_originals, initial_metrics)

    if args.source == "all":
        write_feature_outputs(
            raw_master_df=filtered_raw_df,
            cleaned_df=filtered_cleaned_df.drop(columns=["_dataset_source"], errors="ignore"),
            features_df=filtered_features_df.drop(columns=["_dataset_source"], errors="ignore"),
            input_csv_label="generated_by_build_combined_master.py",
        )

    write_build_summary(
        source_results=source_results,
        invalid_originals_by_reason=invalid_originals_by_reason,
        retained_originals=retained_originals,
        filtered_cleaned_df=filtered_cleaned_df,
    )

    removed_count = len(invalid_originals)
    print(f"retained {len(retained_originals)} original circuits", flush=True)
    print(f"removed {removed_count} original circuits during filtering", flush=True)
    for reason, names in invalid_originals_by_reason.items():
        print(f"  {reason}: {len(names)}", flush=True)

    if args.source == "all":
        print(f"wrote {COMBINED_MASTER_OUTPUT}", flush=True)
        print(f"wrote {CLEANED_MASTER_OUTPUT}", flush=True)
        print(f"wrote {FEATURE_OUTPUT_CSV}", flush=True)
        print(f"wrote {FEATURE_SUMMARY_JSON}", flush=True)
    print(f"wrote {BUILD_SUMMARY_JSON}", flush=True)
    for result in source_results:
        print(f"wrote {result.config.original_circuits_csv}", flush=True)
        print(f"wrote {result.config.single_master_csv}", flush=True)
        print(f"wrote {result.config.combined_master_csv}", flush=True)
        print(f"wrote {result.config.error_csv}", flush=True)
        print(f"wrote {result.config.one_opt_circuits_csv}", flush=True)


if __name__ == "__main__":
    main()
