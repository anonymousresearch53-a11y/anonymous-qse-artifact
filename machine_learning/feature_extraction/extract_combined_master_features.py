#!/usr/bin/env python3
"""Extract circuit and optimizer-history features from combined_master.csv."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from qiskit import QuantumCircuit, qasm2

try:
    from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS
except ImportError:
    LEGACY_CUSTOM_INSTRUCTIONS = ()


OPTIMIZERS = (
    "qiskit_standard",
    "tket",
    "wisq_rules",
    "wisq_bqskit",
)

GATE_NAMES = ("rz", "sx", "x", "cx")
NON_GATE_OPS = {"barrier", "measure", "delay"}
QASM_HEADER_PREFIXES = ("openqasm", "include", "creg", "opaque")
QREG_RE = re.compile(r"^\s*qreg\s+([A-Za-z_]\w*)\[(\d+)\]\s*;")
QUBIT_REF_RE = re.compile(r"\b([A-Za-z_]\w*)\[(\d+)\]")
QASM_OPERATION_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(?:\([^;]*\))?\s*(.*?)\s*;\s*$"
)

METADATA_COLUMNS = (
    "id",
    "original_circuit_id",
    "original_circuit",
    "original_category",
    "source_repository",
    "chain_name",
    "opt_chain",
    "opt_level",
    "opt_1",
    "opt_2",
    "opt_3",
    "artifact_path",
)

LOAD_COLUMNS = (
    "feature_circuit_path",
    "feature_load_source",
    "feature_load_error",
)

INITIAL_LOAD_COLUMNS = (
    "initial_feature_circuit_path",
    "initial_feature_load_source",
    "initial_feature_load_error",
)

QASM_CHECK_COLUMNS = (
    "loaded_num_qubits",
    "loaded_depth",
    "loaded_total_gate_count",
    "loaded_two_qubit_gate_count",
    "loaded_two_qubit_depth",
    "loaded_total_gate_delta_vs_final",
    "loaded_two_qubit_gate_delta_vs_final",
    "loaded_counts_match_final",
)

GATE_FEATURE_COLUMNS = tuple(
    name
    for gate_name in GATE_NAMES
    for name in (f"{gate_name}_gate_count", f"{gate_name}_gate_fraction")
)

DERIVED_RATIO_COLUMNS = (
    "two_qubit_gate_fraction",
    "depth_per_qubit",
    "gates_per_qubit",
    "two_qubit_gates_per_qubit",
)

CURRENT_STATE_COLUMNS = (
    "depth_reduction_so_far",
    "depth_reduction_so_far_fraction",
    "two_qubit_gate_reduction_so_far",
    "two_qubit_gate_reduction_so_far_fraction",
    "runtime_so_far_seconds",
    "current_step_runtime_seconds",
)

ACTION_HISTORY_LABEL_COLUMNS = ("last_optimizer",)

ACTION_HISTORY_COLUMNS = (
    "num_optimizers_used",
    "num_unique_optimizers_used",
    *(f"used_{optimizer}" for optimizer in OPTIMIZERS),
    *(f"last_optimizer_is_{optimizer}" for optimizer in OPTIMIZERS),
)

INTERACTION_COLUMNS = (
    "num_interaction_edges",
    "avg_interaction_degree",
    "max_interaction_degree",
    "active_interaction_qubits",
    "active_interaction_qubit_fraction",
)

STATE_FEATURE_COLUMNS = (
    *GATE_FEATURE_COLUMNS,
    *DERIVED_RATIO_COLUMNS,
    *INTERACTION_COLUMNS,
)

INITIAL_STATE_FEATURE_COLUMNS = tuple(
    f"initial_{column}" for column in STATE_FEATURE_COLUMNS
)

FEATURE_COLUMNS = (
    *STATE_FEATURE_COLUMNS,
    *INITIAL_STATE_FEATURE_COLUMNS,
    *CURRENT_STATE_COLUMNS,
    *ACTION_HISTORY_COLUMNS,
)


def configure_feature_schema(
    optimizer_names: tuple[str, ...] | None = None,
    gate_names: tuple[str, ...] | None = None,
) -> None:
    global OPTIMIZERS, GATE_NAMES, GATE_FEATURE_COLUMNS
    global ACTION_HISTORY_COLUMNS, STATE_FEATURE_COLUMNS
    global INITIAL_STATE_FEATURE_COLUMNS, FEATURE_COLUMNS

    if optimizer_names is not None:
        OPTIMIZERS = optimizer_names
    if gate_names is not None:
        GATE_NAMES = gate_names

    GATE_FEATURE_COLUMNS = tuple(
        name
        for gate_name in GATE_NAMES
        for name in (f"{gate_name}_gate_count", f"{gate_name}_gate_fraction")
    )
    ACTION_HISTORY_COLUMNS = (
        "num_optimizers_used",
        "num_unique_optimizers_used",
        *(f"used_{optimizer}" for optimizer in OPTIMIZERS),
        *(f"last_optimizer_is_{optimizer}" for optimizer in OPTIMIZERS),
    )
    STATE_FEATURE_COLUMNS = (
        *GATE_FEATURE_COLUMNS,
        *DERIVED_RATIO_COLUMNS,
        *INTERACTION_COLUMNS,
    )
    INITIAL_STATE_FEATURE_COLUMNS = tuple(
        f"initial_{column}" for column in STATE_FEATURE_COLUMNS
    )
    FEATURE_COLUMNS = (
        *STATE_FEATURE_COLUMNS,
        *INITIAL_STATE_FEATURE_COLUMNS,
        *CURRENT_STATE_COLUMNS,
        *ACTION_HISTORY_COLUMNS,
    )


def find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "dbs" / "combined_master.csv").exists():
            return candidate
    return Path.cwd().resolve()


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
DEFAULT_INPUT = REPO_ROOT / "dbs" / "combined_master.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "combined_master_features.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "combined_master_features_summary.json"
DEFAULT_CLEANED_MASTER_OUTPUT = SCRIPT_DIR / "combined_master_cleaned_with_features.csv"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def row_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None:
        return 0.0
    try:
        if math.isnan(denominator) or denominator == 0:
            return 0.0
    except TypeError:
        return 0.0
    return numerator / denominator


def derived_ratio_features_from_metrics(
    num_qubits: float,
    depth: float,
    total_gates: float,
    two_qubit_gates: float,
) -> dict[str, float]:
    return {
        "two_qubit_gate_fraction": safe_divide(two_qubit_gates, total_gates),
        "depth_per_qubit": safe_divide(depth, num_qubits),
        "gates_per_qubit": safe_divide(total_gates, num_qubits),
        "two_qubit_gates_per_qubit": safe_divide(two_qubit_gates, num_qubits),
    }


def final_ratio_features_from_row(row: pd.Series) -> dict[str, float]:
    return derived_ratio_features_from_metrics(
        num_qubits=row_float(row, "final_num_qubits"),
        depth=row_float(row, "final_depth"),
        total_gates=row_float(row, "final_total_gates"),
        two_qubit_gates=row_float(row, "final_two_qubit_gates"),
    )


def initial_ratio_features_from_row(row: pd.Series) -> dict[str, float]:
    return derived_ratio_features_from_metrics(
        num_qubits=row_float(row, "initial_num_qubits"),
        depth=row_float(row, "initial_depth"),
        total_gates=row_float(row, "initial_total_gates"),
        two_qubit_gates=row_float(row, "initial_two_qubit_gates"),
    )


def structural_features_from_circuit_and_ratios(
    circuit_features: dict[str, float],
    ratio_features: dict[str, float],
) -> dict[str, float]:
    features = {
        column: circuit_features.get(column, math.nan)
        for column in (*GATE_FEATURE_COLUMNS, *INTERACTION_COLUMNS)
    }
    features.update(ratio_features)
    return features


def prefix_features(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{column}": value for column, value in features.items()}


def resolve_existing_path(path_value: Any, repo_root: Path) -> Path | None:
    path_text = clean_text(path_value)
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo_root / path

    if path.is_file():
        return path.resolve()
    return None


def select_circuit_path(
    row: pd.Series, repo_root: Path, allow_fallback: bool
) -> tuple[str, str, str]:
    candidates = [("artifact_path", row.get("artifact_path"))]
    if allow_fallback:
        candidates.extend(
            [
                ("parent_circuit_path", row.get("parent_circuit_path")),
                ("original_circuit_path", row.get("original_circuit_path")),
            ]
        )

    tried_sources = []
    for source, value in candidates:
        path = resolve_existing_path(value, repo_root)
        if path is not None:
            return str(path), source, ""

        value_text = clean_text(value)
        if value_text:
            tried_sources.append(source)

    if tried_sources:
        return "", "missing", f"No existing QASM file found from {tried_sources}"
    return "", "missing", "No circuit path available"


def strip_qasm_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def qasm_qubit_indices(
    arg_text: str,
    register_offsets: dict[str, int],
    register_sizes: dict[str, int],
) -> list[int]:
    qargs: list[int] = []
    for register, index_text in QUBIT_REF_RE.findall(arg_text):
        index = int(index_text)
        if register in register_offsets:
            qargs.append(register_offsets[register] + index)
        else:
            qargs.append(index)

    if qargs:
        return qargs

    for token in (part.strip() for part in arg_text.split(",")):
        if token in register_offsets:
            offset = register_offsets[token]
            size = register_sizes[token]
            qargs.extend(range(offset, offset + size))

    return qargs


def features_from_counts(
    num_qubits: int,
    depth: int,
    total_gate_count: int,
    two_qubit_gate_count: int,
    two_qubit_depth: int,
    operation_counts: Counter[str],
    interaction_edges: set[tuple[int, int]],
) -> dict[str, float]:
    degrees = [0] * num_qubits
    for left, right in interaction_edges:
        if 0 <= left < num_qubits and 0 <= right < num_qubits:
            degrees[left] += 1
            degrees[right] += 1

    active_qubits = sum(1 for degree in degrees if degree > 0)

    features: dict[str, float] = {
        "loaded_num_qubits": float(num_qubits),
        "loaded_depth": float(depth),
        "loaded_total_gate_count": float(total_gate_count),
        "loaded_two_qubit_gate_count": float(two_qubit_gate_count),
        "loaded_two_qubit_depth": float(two_qubit_depth),
        "num_interaction_edges": float(len(interaction_edges)),
        "avg_interaction_degree": safe_divide(float(sum(degrees)), float(num_qubits)),
        "max_interaction_degree": float(max(degrees) if degrees else 0),
        "active_interaction_qubits": float(active_qubits),
        "active_interaction_qubit_fraction": safe_divide(active_qubits, num_qubits),
    }

    for gate_name in GATE_NAMES:
        count = float(operation_counts.get(gate_name, 0))
        features[f"{gate_name}_gate_count"] = count
        features[f"{gate_name}_gate_fraction"] = safe_divide(count, total_gate_count)

    return features


def extract_circuit_features_from_qasm_text(path: str) -> dict[str, float]:
    register_offsets: dict[str, int] = {}
    register_sizes: dict[str, int] = {}
    operation_counts: Counter[str] = Counter()
    interaction_edges: set[tuple[int, int]] = set()
    num_qubits = 0
    max_seen_qubit = -1
    total_gate_count = 0
    two_qubit_gate_count = 0
    depth = 0
    two_qubit_depth = 0
    qubit_depths: list[int] = []
    two_qubit_depths: list[int] = []
    in_gate_definition = False
    brace_depth = 0

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = strip_qasm_comment(raw_line)
            if not line:
                continue

            lower_line = line.lower()
            if in_gate_definition:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    in_gate_definition = False
                continue

            if lower_line.startswith("gate "):
                brace_depth = line.count("{") - line.count("}")
                in_gate_definition = brace_depth > 0 or "{" not in line
                continue

            qreg_match = QREG_RE.match(line)
            if qreg_match:
                register = qreg_match.group(1)
                size = int(qreg_match.group(2))
                register_offsets[register] = num_qubits
                register_sizes[register] = size
                num_qubits += size
                qubit_depths.extend([0] * size)
                two_qubit_depths.extend([0] * size)
                continue

            if lower_line.startswith(QASM_HEADER_PREFIXES):
                continue

            if lower_line.startswith("if ") or lower_line.startswith("if("):
                if ")" not in line:
                    continue
                line = line.split(")", 1)[1].strip()
                lower_line = line.lower()

            operation_match = QASM_OPERATION_RE.match(line)
            if not operation_match:
                continue

            operation_name = operation_match.group(1).lower()
            if operation_name in NON_GATE_OPS or operation_name == "reset":
                continue

            qargs = qasm_qubit_indices(
                operation_match.group(2), register_offsets, register_sizes
            )
            if not qargs:
                continue

            max_seen_qubit = max(max_seen_qubit, *qargs)
            operation_counts[operation_name] += 1
            total_gate_count += 1

            unique_qargs = sorted(set(qargs))
            if unique_qargs:
                op_depth = max(qubit_depths[index] for index in unique_qargs) + 1
                for index in unique_qargs:
                    qubit_depths[index] = op_depth
                depth = max(depth, op_depth)

            if len(qargs) >= 2:
                two_qubit_gate_count += 1
                interaction_edges.update(combinations(unique_qargs, 2))
                op_two_qubit_depth = (
                    max(two_qubit_depths[index] for index in unique_qargs) + 1
                )
                for index in unique_qargs:
                    two_qubit_depths[index] = op_two_qubit_depth
                two_qubit_depth = max(two_qubit_depth, op_two_qubit_depth)

    if num_qubits == 0 and max_seen_qubit >= 0:
        num_qubits = max_seen_qubit + 1
        qubit_depths.extend([0] * (num_qubits - len(qubit_depths)))
        two_qubit_depths.extend([0] * (num_qubits - len(two_qubit_depths)))

    return features_from_counts(
        num_qubits=num_qubits,
        depth=depth,
        total_gate_count=total_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        two_qubit_depth=two_qubit_depth,
        operation_counts=operation_counts,
        interaction_edges=interaction_edges,
    )


def load_qasm_circuit(path: str) -> QuantumCircuit:
    try:
        return qasm2.load(path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as qasm2_error:
        try:
            return QuantumCircuit.from_qasm_file(path)
        except Exception as fallback_error:
            message = f"qasm2.load failed: {qasm2_error}; from_qasm_file failed: {fallback_error}"
            raise RuntimeError(message) from fallback_error


def instruction_operation(instruction: Any) -> Any:
    if hasattr(instruction, "operation"):
        return instruction.operation
    return instruction[0]


def instruction_qubits(instruction: Any) -> list[Any]:
    if hasattr(instruction, "qubits"):
        return list(instruction.qubits)
    return list(instruction[1])


def instruction_name(instruction: Any) -> str:
    operation = instruction_operation(instruction)
    return clean_text(getattr(operation, "name", operation)).lower()


def is_two_qubit_gate(instruction: Any) -> bool:
    name = instruction_name(instruction)
    return name not in NON_GATE_OPS and len(instruction_qubits(instruction)) >= 2


def is_real_gate(instruction: Any) -> bool:
    name = instruction_name(instruction)
    return name not in NON_GATE_OPS and name != "reset"


def qubit_index(circuit: QuantumCircuit, qubit: Any) -> int:
    try:
        return int(circuit.find_bit(qubit).index)
    except Exception:
        return int(circuit.qubits.index(qubit))


def empty_circuit_features() -> dict[str, float]:
    columns = (*QASM_CHECK_COLUMNS, *GATE_FEATURE_COLUMNS, *INTERACTION_COLUMNS)
    return {column: math.nan for column in columns}


def extract_circuit_features_from_qiskit(path: str) -> dict[str, float]:
    circuit = load_qasm_circuit(path)
    operation_counts: Counter[str] = Counter()
    interaction_edges: set[tuple[int, int]] = set()
    two_qubit_gate_count = 0

    for instruction in circuit.data:
        name = instruction_name(instruction)
        operation_counts[name] += 1

        if name in NON_GATE_OPS:
            continue

        qargs = instruction_qubits(instruction)
        if len(qargs) < 2:
            continue

        two_qubit_gate_count += 1
        qindices = sorted(qubit_index(circuit, qubit) for qubit in qargs)
        interaction_edges.update(combinations(qindices, 2))

    num_qubits = int(circuit.num_qubits)
    depth = int(circuit.depth(filter_function=is_real_gate) or 0)
    total_gate_count = sum(1 for instruction in circuit.data if is_real_gate(instruction))
    two_qubit_depth = int(circuit.depth(filter_function=is_two_qubit_gate) or 0)

    return features_from_counts(
        num_qubits=num_qubits,
        depth=depth,
        total_gate_count=total_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        two_qubit_depth=two_qubit_depth,
        operation_counts=operation_counts,
        interaction_edges=interaction_edges,
    )


def extract_circuit_features(path: str, use_qiskit_loader: bool) -> dict[str, float]:
    if use_qiskit_loader:
        return extract_circuit_features_from_qiskit(path)
    return extract_circuit_features_from_qasm_text(path)


def optimizer_sequence(row: pd.Series) -> list[str]:
    sequence = [clean_text(row.get(column)) for column in ("opt_1", "opt_2", "opt_3")]
    sequence = [optimizer for optimizer in sequence if optimizer]
    if sequence:
        return sequence

    for column in ("chain_name", "opt_chain"):
        parts = clean_text(row.get(column)).split("__")
        sequence = [part for part in parts if part in OPTIMIZERS]
        if sequence:
            return sequence

    return []


def extract_row_features(row: pd.Series) -> dict[str, float | str]:
    final_depth = row_float(row, "final_depth")
    final_two_qubit_gates = row_float(row, "final_two_qubit_gates")
    initial_depth = row_float(row, "initial_depth")
    initial_two_qubit_gates = row_float(row, "initial_two_qubit_gates")

    features: dict[str, float | str] = final_ratio_features_from_row(row)
    depth_reduction = initial_depth - final_depth
    two_qubit_reduction = initial_two_qubit_gates - final_two_qubit_gates
    sequence = optimizer_sequence(row)
    last_optimizer = sequence[-1] if sequence else ""

    features.update(
        {
            "depth_reduction_so_far": depth_reduction,
            "depth_reduction_so_far_fraction": safe_divide(
                depth_reduction, initial_depth
            ),
            "two_qubit_gate_reduction_so_far": two_qubit_reduction,
            "two_qubit_gate_reduction_so_far_fraction": safe_divide(
                two_qubit_reduction, initial_two_qubit_gates
            ),
            "runtime_so_far_seconds": row_float(row, "total_runtime_seconds"),
            "current_step_runtime_seconds": row_float(row, "current_step_runtime"),
            "last_optimizer": last_optimizer,
            "num_optimizers_used": float(len(sequence)),
            "num_unique_optimizers_used": float(len(set(sequence))),
        }
    )

    for optimizer in OPTIMIZERS:
        features[f"used_{optimizer}"] = float(optimizer in sequence)
        features[f"last_optimizer_is_{optimizer}"] = float(last_optimizer == optimizer)

    return features


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def extract_qasm_check_features(
    row: pd.Series, circuit_features: dict[str, float]
) -> dict[str, float]:
    loaded_total = circuit_features.get("loaded_total_gate_count", math.nan)
    loaded_two_qubit = circuit_features.get("loaded_two_qubit_gate_count", math.nan)
    final_total = row_float(row, "final_total_gates", math.nan)
    final_two_qubit = row_float(row, "final_two_qubit_gates", math.nan)

    total_delta = (
        float(loaded_total) - final_total
        if is_finite_number(loaded_total) and is_finite_number(final_total)
        else math.nan
    )
    two_qubit_delta = (
        float(loaded_two_qubit) - final_two_qubit
        if is_finite_number(loaded_two_qubit) and is_finite_number(final_two_qubit)
        else math.nan
    )

    return {
        "loaded_total_gate_delta_vs_final": total_delta,
        "loaded_two_qubit_gate_delta_vs_final": two_qubit_delta,
        "loaded_counts_match_final": float(total_delta == 0 and two_qubit_delta == 0),
    }


def get_cached_circuit_features(
    path: str,
    cache: dict[str, tuple[dict[str, float], str]],
    use_qiskit_loader: bool,
) -> tuple[dict[str, float], str]:
    if path not in cache:
        try:
            cache[path] = (extract_circuit_features(path, use_qiskit_loader), "")
        except Exception as error:
            cache[path] = (empty_circuit_features(), str(error))
    return cache[path]


def metric_value(features: dict[str, float], name: str) -> float:
    value = features.get(name, math.nan)
    return float(value) if is_finite_number(value) else math.nan


def maybe_print_progress(position: int, total: int, progress_every: int) -> None:
    if progress_every <= 0:
        return
    if position != total and position % progress_every != 0:
        return

    percent = safe_divide(float(position), float(total)) * 100.0 if total else 100.0
    print(f"[progress] {position}/{total} rows ({percent:.1f}%)", flush=True)


def apply_metric_features_to_row(
    row: dict[str, Any],
    prefix: str,
    features: dict[str, float],
) -> None:
    num_qubits = metric_value(features, "loaded_num_qubits")
    depth = metric_value(features, "loaded_depth")
    total_gates = metric_value(features, "loaded_total_gate_count")
    two_qubit_gates = metric_value(features, "loaded_two_qubit_gate_count")
    two_qubit_depth = metric_value(features, "loaded_two_qubit_depth")

    row[f"{prefix}_num_qubits"] = num_qubits
    row[f"{prefix}_depth"] = depth
    row[f"{prefix}_two_qubit_gates"] = two_qubit_gates
    row[f"{prefix}_total_gates"] = total_gates
    row[f"{prefix}_two_qubit_depth"] = two_qubit_depth

    if prefix == "initial":
        row["initial_gate_density"] = safe_divide(total_gates, num_qubits)
        row["initial_two_qubit_ratio"] = safe_divide(two_qubit_gates, total_gates)


def select_original_circuit_path(row: pd.Series, repo_root: Path) -> tuple[str, str, str]:
    path = resolve_existing_path(row.get("original_circuit_path"), repo_root)
    if path is not None:
        return str(path), "original_circuit_path", ""

    value_text = clean_text(row.get("original_circuit_path"))
    if value_text:
        return "", "missing", "No existing QASM file found from original_circuit_path"
    return "", "missing", "No original_circuit_path available"


def feature_record_from_cleaned_row(
    cleaned_row: pd.Series,
    metadata_columns: list[str],
    final_path: str,
    final_load_source: str,
    final_load_error: str,
    final_features: dict[str, float],
    original_path: str,
    original_load_source: str,
    original_load_error: str,
    original_features: dict[str, float],
    source_row: pd.Series,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        column: cleaned_row.get(column) for column in metadata_columns
    }
    record.update(
        {
            "feature_circuit_path": final_path,
            "feature_load_source": final_load_source,
            "feature_load_error": final_load_error,
            "initial_feature_circuit_path": original_path,
            "initial_feature_load_source": original_load_source,
            "initial_feature_load_error": original_load_error,
        }
    )
    final_state_features = structural_features_from_circuit_and_ratios(
        final_features,
        final_ratio_features_from_row(cleaned_row),
    )
    initial_state_features = structural_features_from_circuit_and_ratios(
        original_features,
        initial_ratio_features_from_row(cleaned_row),
    )
    record.update(final_features)
    record.update(final_state_features)
    record.update(prefix_features(initial_state_features, "initial"))
    record.update(extract_row_features(cleaned_row))
    record.update(extract_qasm_check_features(source_row, final_features))
    return record


def build_cleaned_master_with_features(
    df: pd.DataFrame,
    repo_root: Path,
    allow_fallback: bool,
    strict_artifact_path: bool,
    progress_every: int,
    use_qiskit_loader: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    circuit_feature_cache: dict[str, tuple[dict[str, float], str]] = {}
    metadata_columns = [column for column in METADATA_COLUMNS if column in df.columns]
    total_rows = len(df)

    for position, (_, source_row) in enumerate(df.iterrows(), start=1):
        cleaned_record = source_row.to_dict()

        original_path, original_source, original_error = select_original_circuit_path(
            source_row, repo_root
        )
        if original_path:
            original_features, cached_original_error = get_cached_circuit_features(
                original_path, circuit_feature_cache, use_qiskit_loader
            )
            original_error = cached_original_error
            if not original_error:
                apply_metric_features_to_row(cleaned_record, "initial", original_features)
        else:
            original_features = empty_circuit_features()

        final_path, final_source, final_error = select_circuit_path(
            source_row, repo_root, allow_fallback
        )
        if strict_artifact_path and final_error:
            circuit_name = clean_text(source_row.get("original_circuit")) or "<unknown>"
            chain_name = clean_text(source_row.get("chain_name")) or "<unknown>"
            artifact_path = clean_text(source_row.get("artifact_path")) or "<missing>"
            raise RuntimeError(
                "Strict artifact path validation failed for "
                f"original_circuit={circuit_name}, chain_name={chain_name}, "
                f"artifact_path={artifact_path}: {final_error}"
            )
        if final_path:
            final_features, cached_final_error = get_cached_circuit_features(
                final_path, circuit_feature_cache, use_qiskit_loader
            )
            final_error = cached_final_error
            if strict_artifact_path and final_error:
                circuit_name = clean_text(source_row.get("original_circuit")) or "<unknown>"
                chain_name = clean_text(source_row.get("chain_name")) or "<unknown>"
                raise RuntimeError(
                    "Strict artifact path parsing failed for "
                    f"original_circuit={circuit_name}, chain_name={chain_name}, "
                    f"artifact_path={final_path}: {final_error}"
                )
            if not final_error:
                apply_metric_features_to_row(cleaned_record, "final", final_features)
        else:
            final_features = empty_circuit_features()

        cleaned_row = pd.Series(cleaned_record)
        feature_record = feature_record_from_cleaned_row(
            cleaned_row=cleaned_row,
            metadata_columns=metadata_columns,
            final_path=final_path,
            final_load_source=final_source,
            final_load_error=final_error,
            final_features=final_features,
            original_path=original_path,
            original_load_source=original_source,
            original_load_error=original_error,
            original_features=original_features,
            source_row=source_row,
        )
        cleaned_record.update(
            {
                column: value
                for column, value in feature_record.items()
                if column not in cleaned_record
            }
        )

        cleaned_records.append(cleaned_record)
        feature_records.append(feature_record)

        maybe_print_progress(position, total_rows, progress_every)

    return pd.DataFrame(cleaned_records), pd.DataFrame(feature_records)


def extract_features(
    df: pd.DataFrame,
    repo_root: Path,
    allow_fallback: bool,
    progress_every: int,
    use_qiskit_loader: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    circuit_feature_cache: dict[str, tuple[dict[str, float], str]] = {}
    metadata_columns = [column for column in METADATA_COLUMNS if column in df.columns]
    total_rows = len(df)

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        circuit_path, load_source, load_error = select_circuit_path(
            row, repo_root, allow_fallback
        )

        if circuit_path:
            if circuit_path not in circuit_feature_cache:
                try:
                    circuit_feature_cache[circuit_path] = (
                        extract_circuit_features(circuit_path, use_qiskit_loader),
                        "",
                    )
                except Exception as error:
                    circuit_feature_cache[circuit_path] = (
                        empty_circuit_features(),
                        str(error),
                    )
            circuit_features, cached_load_error = circuit_feature_cache[circuit_path]
            load_error = cached_load_error
        else:
            circuit_features = empty_circuit_features()

        record: dict[str, Any] = {column: row.get(column) for column in metadata_columns}
        record.update(
            {
                "feature_circuit_path": circuit_path,
                "feature_load_source": load_source,
                "feature_load_error": load_error,
            }
        )
        record.update(circuit_features)
        record.update(extract_row_features(row))
        record.update(extract_qasm_check_features(row, circuit_features))
        records.append(record)

        maybe_print_progress(position, total_rows, progress_every)

    return pd.DataFrame(records)


def write_outputs(
    source_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_csv: Path,
    summary_json: Path,
    merged_output_csv: Path | None,
    cleaned_master_output_csv: Path | None,
    input_csv: Path,
    use_qiskit_loader: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_csv, index=False)

    if merged_output_csv is not None:
        merged_output_csv.parent.mkdir(parents=True, exist_ok=True)
        appended_columns = [
            column for column in features_df.columns if column not in source_df.columns
        ]
        merged_df = pd.concat(
            [source_df.reset_index(drop=True), features_df[appended_columns]],
            axis=1,
        )
        merged_df.to_csv(merged_output_csv, index=False)

    if cleaned_master_output_csv is not None:
        cleaned_master_output_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(cleaned_master_output_csv, index=False)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "merged_output_csv": str(merged_output_csv) if merged_output_csv else None,
        "cleaned_master_output_csv": (
            str(cleaned_master_output_csv) if cleaned_master_output_csv else None
        ),
        "circuit_parser": "qiskit" if use_qiskit_loader else "qasm_text",
        "metric_definition": (
            "real gate operations only; excludes barrier, measure, delay, and reset"
        ),
        "rows": int(len(features_df)),
        "metadata_columns": [column for column in METADATA_COLUMNS if column in features_df.columns],
        "load_columns": list(LOAD_COLUMNS),
        "initial_load_columns": list(INITIAL_LOAD_COLUMNS),
        "qasm_check_columns": list(QASM_CHECK_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "fraction_note": (
            "Gate and ratio fractions are 0 to 1. Reduction fractions are signed "
            "fractional changes and can be negative when a metric gets worse."
        ),
        "unique_feature_circuit_paths": int(
            features_df["feature_circuit_path"].replace("", pd.NA).nunique()
        ),
        "load_sources": {
            str(key): int(value)
            for key, value in features_df["feature_load_source"].value_counts(dropna=False).items()
        },
        "load_error_count": int(features_df["feature_load_error"].fillna("").ne("").sum()),
        "loaded_total_gate_count_mismatch_count": int(
            features_df["loaded_total_gate_delta_vs_final"].fillna(0).ne(0).sum()
        ),
        "loaded_two_qubit_gate_count_mismatch_count": int(
            features_df["loaded_two_qubit_gate_delta_vs_final"].fillna(0).ne(0).sum()
        ),
        "loaded_counts_match_final_count": int(
            features_df["loaded_counts_match_final"].fillna(0).eq(1).sum()
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate RL/model features from dbs/combined_master.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Optional CSV with the original master columns plus the extracted columns.",
    )
    parser.add_argument(
        "--cleaned-master-output",
        type=Path,
        default=DEFAULT_CLEANED_MASTER_OUTPUT,
        help=(
            "CSV containing all master columns with recomputed metrics plus extracted "
            "features. Use an empty string to skip."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--strict-artifact-path",
        action="store_true",
        help="Use only artifact_path. By default, missing artifacts fall back to parent/original paths.",
    )
    parser.add_argument(
        "--use-qiskit-loader",
        action="store_true",
        help="Parse QASM with Qiskit instead of the faster text parser.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--optimizer-names",
        nargs="+",
        default=None,
        help="Optimizer labels used in action-history features.",
    )
    parser.add_argument(
        "--gate-names",
        nargs="+",
        default=None,
        help="Gate names used in gate-type features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_feature_schema(
        optimizer_names=(
            tuple(args.optimizer_names) if args.optimizer_names is not None else None
        ),
        gate_names=tuple(args.gate_names) if args.gate_names is not None else None,
    )
    input_csv = args.input.expanduser().resolve()
    output_csv = args.output.expanduser().resolve()
    summary_json = args.summary_output.expanduser().resolve()
    merged_output_csv = (
        args.merged_output.expanduser().resolve() if args.merged_output else None
    )
    cleaned_master_output_csv = (
        args.cleaned_master_output.expanduser().resolve()
        if clean_text(args.cleaned_master_output)
        else None
    )
    repo_root = find_repo_root(input_csv.parent)

    source_df = pd.read_csv(input_csv)
    if args.limit is not None:
        source_df = source_df.head(args.limit).copy()

    print(
        f"starting feature extraction for {len(source_df)} rows from {input_csv}",
        flush=True,
    )

    cleaned_df, features_df = build_cleaned_master_with_features(
        source_df,
        repo_root=repo_root,
        allow_fallback=not args.strict_artifact_path,
        strict_artifact_path=args.strict_artifact_path,
        progress_every=args.progress_every,
        use_qiskit_loader=args.use_qiskit_loader,
    )
    write_outputs(
        source_df=source_df,
        cleaned_df=cleaned_df,
        features_df=features_df,
        output_csv=output_csv,
        summary_json=summary_json,
        merged_output_csv=merged_output_csv,
        cleaned_master_output_csv=cleaned_master_output_csv,
        input_csv=input_csv,
        use_qiskit_loader=args.use_qiskit_loader,
    )

    load_errors = int(features_df["feature_load_error"].fillna("").ne("").sum())
    print(f"wrote {len(features_df)} rows to {output_csv}", flush=True)
    print(f"wrote summary to {summary_json}", flush=True)
    if merged_output_csv:
        print(f"wrote merged data to {merged_output_csv}", flush=True)
    if cleaned_master_output_csv:
        print(f"wrote cleaned master data to {cleaned_master_output_csv}", flush=True)
    print(f"load errors: {load_errors}", flush=True)


if __name__ == "__main__":
    main()
