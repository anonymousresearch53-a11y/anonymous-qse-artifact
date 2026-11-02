"""Benchpress circuit importer for trajectory database.

This module provides functionality to import quantum circuits from the
Qiskit Benchpress repository for use in RL training experiments.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from qiskit import qasm2
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

from ..transpilers import analyze_circuit
from .database import CircuitRecord, TrajectoryDatabase

# Benchpress repository URL
BENCHPRESS_REPO_URL = "https://github.com/Qiskit/benchpress.git"

# Known circuit categories in Benchpress
BENCHPRESS_CATEGORIES: list[str] = [
    "qft",
    "qaoa",
    "clifford",
    "qv",
    "bigint",
    "dtc",
    "feynman",
    "square-heisenberg",
    "qasmbench-small",
    "qasmbench-medium",
    "qasmbench-large",
]


@dataclass
class CircuitInfo:
    """Information about a discovered circuit."""

    name: str
    category: str
    source: str
    qasm_path: Path
    num_qubits: int


def _discover_qasm_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Discover QASM files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of QASM file paths
    """
    if not directory.exists():
        return []

    if recursive:
        return list(directory.rglob("*.qasm"))
    return list(directory.glob("*.qasm"))


def _get_circuit_num_qubits(qasm_path: Path) -> int | None:
    """Get the number of qubits from a QASM file.

    Args:
        qasm_path: Path to QASM file

    Returns:
        Number of qubits, or None if parsing fails
    """
    try:
        circuit = qasm2.load(
            str(qasm_path), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS
        )
        return circuit.num_qubits
    except Exception:
        return None


def _infer_category_from_path(qasm_path: Path, base_path: Path) -> str:
    """Infer circuit category from file path.

    Args:
        qasm_path: Path to QASM file
        base_path: Base directory to compute relative path from

    Returns:
        Inferred category name
    """
    try:
        rel_path = qasm_path.relative_to(base_path)
        parts = rel_path.parts

        # Check if any path component matches a known category
        for part in parts:
            part_lower = part.lower()
            for category in BENCHPRESS_CATEGORIES:
                if category.lower() in part_lower:
                    return category

        # Fall back to first directory name
        if len(parts) > 1:
            return parts[0].lower()

        return "unknown"
    except ValueError:
        return "unknown"


def _artifact_name_from_rel_path(rel_path: Path) -> str:
    """Create a stable circuit name from an artifact relative path."""
    rel_no_suffix = rel_path.with_suffix("")
    name = rel_no_suffix.as_posix().replace("/", "__").replace(" ", "_")
    name = name.replace(".", "_")
    return f"artifact_{name}"


# Known circuit name prefixes mapping to categories for artifact inference
_CATEGORY_PREFIXES: list[tuple[str, str]] = [
    ("qft", "qft"),
    ("qaoa", "qaoa"),
    ("qv_", "qv"),
    ("clifford", "clifford"),
    ("dtc_", "dtc"),
    ("square_heisenberg", "square-heisenberg"),
    ("efficient_su2", "qasmbench-small"),
    ("real_amplitudes", "qasmbench-small"),
    ("grover", "qasmbench-small"),
    ("hamiltonian", "qasmbench-small"),
]


def _infer_category_from_circuit_name(circuit_name: str) -> str:
    """Infer circuit category from the original circuit name.

    Uses known name prefixes to map circuit names to categories.

    Args:
        circuit_name: Original circuit name (parent directory in artifacts)

    Returns:
        Inferred category name
    """
    lower = circuit_name.lower()
    for prefix, category in _CATEGORY_PREFIXES:
        if lower.startswith(prefix):
            return category
    return "unknown"


class BenchpressImporter:
    """Importer for Qiskit Benchpress circuits."""

    def __init__(
        self,
        cache_dir: Path | str = "data/benchpress_circuits",
    ):
        """Initialize the Benchpress importer.

        Args:
            cache_dir: Directory to cache cloned repository
        """
        self.cache_dir = Path(cache_dir)
        self.repo_path = self.cache_dir / "benchpress"

    def clone_or_update_repo(
        self,
        force_update: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        """Clone or update the Benchpress repository.

        Args:
            force_update: Force update even if repo exists
            progress_callback: Callback for progress messages

        Returns:
            Path to the repository
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.repo_path.exists():
            if force_update:
                log("Updating Benchpress repository...")
                subprocess.run(
                    ["git", "pull"],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                )
                log("Repository updated")
            else:
                log("Using cached Benchpress repository")
        else:
            log(f"Cloning Benchpress repository to {self.repo_path}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", BENCHPRESS_REPO_URL, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
            log("Repository cloned")

        return self.repo_path

    def discover_circuits(
        self,
        categories: Sequence[str] | None = None,
        max_qubits: int = 20,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[CircuitInfo]:
        """Discover circuits in the Benchpress repository.

        Args:
            categories: List of categories to include (None = all)
            max_qubits: Maximum number of qubits to include
            progress_callback: Callback for progress messages

        Returns:
            List of CircuitInfo objects
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        if not self.repo_path.exists():
            self.clone_or_update_repo(progress_callback=progress_callback)

        # Look for circuits in common locations
        circuit_dirs = [
            self.repo_path / "benchpress" / "circuits",
            self.repo_path / "circuits",
            self.repo_path / "qasm",
            self.repo_path / "benchmarks",
        ]

        # Find all QASM files
        qasm_files: list[Path] = []
        for circuit_dir in circuit_dirs:
            qasm_files.extend(_discover_qasm_files(circuit_dir))

        log(f"Found {len(qasm_files)} QASM files")

        # Filter and collect circuit info
        circuits: list[CircuitInfo] = []
        filtered_categories = set(categories) if categories else None

        for qasm_path in qasm_files:
            # Infer category
            category = _infer_category_from_path(qasm_path, self.repo_path)

            # Filter by category
            if filtered_categories and category not in filtered_categories:
                continue

            # Get number of qubits
            num_qubits = _get_circuit_num_qubits(qasm_path)
            if num_qubits is None:
                log(f"  Skipping {qasm_path.name}: failed to parse")
                continue

            # Filter by qubit count
            if num_qubits > max_qubits:
                log(f"  Skipping {qasm_path.name}: {num_qubits} qubits > {max_qubits}")
                continue

            # Create unique name
            name = f"benchpress_{category}_{qasm_path.stem}"

            circuits.append(
                CircuitInfo(
                    name=name,
                    category=category,
                    source="benchpress",
                    qasm_path=qasm_path,
                    num_qubits=num_qubits,
                )
            )

        log(f"Discovered {len(circuits)} circuits within qubit limit")
        return circuits

    def import_to_database(
        self,
        db: TrajectoryDatabase,
        categories: Sequence[str] | None = None,
        max_qubits: int = 20,
        skip_existing: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Import Benchpress circuits to the trajectory database.

        Args:
            db: TrajectoryDatabase to import into
            categories: List of categories to include (None = all)
            max_qubits: Maximum number of qubits
            skip_existing: Skip circuits already in database
            progress_callback: Callback for progress messages

        Returns:
            Number of circuits imported
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Discover circuits
        circuits = self.discover_circuits(
            categories=categories,
            max_qubits=max_qubits,
            progress_callback=progress_callback,
        )

        imported = 0
        skipped = 0

        for circuit_info in circuits:
            # Check if already exists
            if skip_existing:
                existing = db.get_circuit_by_name(circuit_info.name)
                if existing is not None:
                    skipped += 1
                    continue

            # Load and analyze circuit
            try:
                circuit = qasm2.load(
                    str(circuit_info.qasm_path),
                    custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
                )
                metrics = analyze_circuit(circuit)
            except Exception as e:
                log(f"  Error loading {circuit_info.name}: {e}")
                continue

            # Compute derived features
            gate_density = metrics.total_gates / circuit_info.num_qubits if circuit_info.num_qubits > 0 else 0.0
            two_qubit_ratio = metrics.two_qubit_gates / metrics.total_gates if metrics.total_gates > 0 else 0.0

            # Insert into database
            record = CircuitRecord(
                id=None,
                name=circuit_info.name,
                category=circuit_info.category,
                source=circuit_info.source,
                qasm_path=str(circuit_info.qasm_path),
                num_qubits=circuit_info.num_qubits,
                initial_depth=metrics.depth,
                initial_two_qubit_gates=metrics.two_qubit_gates,
                initial_two_qubit_depth=metrics.two_qubit_depth,
                initial_total_gates=metrics.total_gates,
                gate_density=gate_density,
                two_qubit_ratio=two_qubit_ratio,
            )

            db.insert_circuit(record)
            imported += 1

        log(f"Imported {imported} circuits ({skipped} skipped as existing)")
        return imported


class LocalCircuitImporter:
    """Importer for local QASM circuit files."""

    def __init__(self, circuits_dir: Path | str):
        """Initialize the local circuit importer.

        Args:
            circuits_dir: Directory containing QASM files
        """
        self.circuits_dir = Path(circuits_dir)

    def discover_circuits(
        self,
        category: str = "local",
        max_qubits: int = 20,
        recursive: bool = True,
    ) -> list[CircuitInfo]:
        """Discover circuits in the local directory.

        Args:
            category: Category to assign to discovered circuits
            max_qubits: Maximum number of qubits
            recursive: Search recursively

        Returns:
            List of CircuitInfo objects
        """
        qasm_files = _discover_qasm_files(self.circuits_dir, recursive=recursive)

        circuits: list[CircuitInfo] = []
        for qasm_path in qasm_files:
            num_qubits = _get_circuit_num_qubits(qasm_path)
            if num_qubits is None or num_qubits > max_qubits:
                continue

            name = f"local_{qasm_path.stem}"
            circuits.append(
                CircuitInfo(
                    name=name,
                    category=category,
                    source="local",
                    qasm_path=qasm_path,
                    num_qubits=num_qubits,
                )
            )

        return circuits

    def import_to_database(
        self,
        db: TrajectoryDatabase,
        category: str = "local",
        max_qubits: int = 20,
        skip_existing: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Import local circuits to the trajectory database.

        Args:
            db: TrajectoryDatabase to import into
            category: Category to assign
            max_qubits: Maximum number of qubits
            skip_existing: Skip circuits already in database
            progress_callback: Callback for progress messages

        Returns:
            Number of circuits imported
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        circuits = self.discover_circuits(category=category, max_qubits=max_qubits)
        log(f"Found {len(circuits)} local circuits")

        imported = 0
        for circuit_info in circuits:
            if skip_existing:
                existing = db.get_circuit_by_name(circuit_info.name)
                if existing is not None:
                    continue

            try:
                circuit = qasm2.load(
                    str(circuit_info.qasm_path),
                    custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
                )
                metrics = analyze_circuit(circuit)
            except Exception as e:
                log(f"  Error loading {circuit_info.name}: {e}")
                continue

            gate_density = metrics.total_gates / circuit_info.num_qubits if circuit_info.num_qubits > 0 else 0.0
            two_qubit_ratio = metrics.two_qubit_gates / metrics.total_gates if metrics.total_gates > 0 else 0.0

            record = CircuitRecord(
                id=None,
                name=circuit_info.name,
                category=circuit_info.category,
                source=circuit_info.source,
                qasm_path=str(circuit_info.qasm_path),
                num_qubits=circuit_info.num_qubits,
                initial_depth=metrics.depth,
                initial_two_qubit_gates=metrics.two_qubit_gates,
                initial_two_qubit_depth=metrics.two_qubit_depth,
                initial_total_gates=metrics.total_gates,
                gate_density=gate_density,
                two_qubit_ratio=two_qubit_ratio,
            )

            db.insert_circuit(record)
            imported += 1

        log(f"Imported {imported} local circuits")
        return imported


class ArtifactCircuitImporter:
    """Importer for optimized circuit artifacts (step-1 outputs)."""

    def __init__(
        self,
        artifacts_dir: Path | str,
        step1_db_path: Path | str | None = None,
    ):
        """Initialize the artifact importer.

        Args:
            artifacts_dir: Directory containing optimized artifact QASM files
            step1_db_path: Optional path to step-1 trajectory database for
                category lookup. When provided, the original circuit's category
                is looked up from the step-1 DB. When absent, categories are
                inferred from circuit name prefixes.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._step1_db: TrajectoryDatabase | None = None
        self._category_cache: dict[str, str] = {}
        if step1_db_path is not None:
            self._step1_db = TrajectoryDatabase(Path(step1_db_path))

    def _resolve_category(
        self, original_circuit_name: str, fallback: str = "unknown"
    ) -> str:
        """Resolve the category for an artifact circuit.

        Looks up the original circuit name in the step-1 DB first, then
        falls back to name-prefix inference.

        Args:
            original_circuit_name: Parent directory name (original circuit)
            fallback: Fallback category if lookup fails

        Returns:
            Category string
        """
        if original_circuit_name in self._category_cache:
            return self._category_cache[original_circuit_name]

        category = fallback

        # Try step-1 DB lookup
        if self._step1_db is not None:
            record = self._step1_db.get_circuit_by_name(original_circuit_name)
            if record is not None:
                category = record.category
                self._category_cache[original_circuit_name] = category
                return category

        # Fall back to name-prefix inference
        inferred = _infer_category_from_circuit_name(original_circuit_name)
        if inferred != "unknown":
            category = inferred

        self._category_cache[original_circuit_name] = category
        return category

    def discover_circuits(
        self,
        category: str | None = None,
        max_qubits: int = 20,
        recursive: bool = True,
    ) -> list[CircuitInfo]:
        """Discover artifact circuits in the artifact directory.

        Args:
            category: Category override for all circuits (None = auto-detect
                from step-1 DB or name prefixes)
            max_qubits: Maximum number of qubits
            recursive: Search recursively

        Returns:
            List of CircuitInfo objects
        """
        qasm_files = _discover_qasm_files(self.artifacts_dir, recursive=recursive)

        circuits: list[CircuitInfo] = []
        for qasm_path in qasm_files:
            num_qubits = _get_circuit_num_qubits(qasm_path)
            if num_qubits is None or num_qubits > max_qubits:
                continue

            try:
                rel_path = qasm_path.relative_to(self.artifacts_dir)
            except ValueError:
                rel_path = Path(qasm_path.name)

            # Determine category: override or auto-detect from parent dir
            if category is not None:
                circuit_category = category
            else:
                # Parent directory is the original circuit name
                original_name = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
                circuit_category = self._resolve_category(original_name)

            name = _artifact_name_from_rel_path(rel_path)
            circuits.append(
                CircuitInfo(
                    name=name,
                    category=circuit_category,
                    source="artifact",
                    qasm_path=qasm_path,
                    num_qubits=num_qubits,
                )
            )

        return circuits

    def import_to_database(
        self,
        db: TrajectoryDatabase,
        category: str | None = None,
        max_qubits: int = 20,
        skip_existing: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Import artifact circuits to the trajectory database.

        Args:
            db: TrajectoryDatabase to import into
            category: Category override (None = auto-detect from step-1 DB
                or name prefixes)
            max_qubits: Maximum number of qubits
            skip_existing: Skip circuits already in database
            progress_callback: Callback for progress messages

        Returns:
            Number of circuits imported
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        circuits = self.discover_circuits(category=category, max_qubits=max_qubits)
        log(f"Found {len(circuits)} artifact circuits")

        imported = 0
        for circuit_info in circuits:
            if skip_existing:
                existing = db.get_circuit_by_name(circuit_info.name)
                if existing is not None:
                    continue

            try:
                circuit = qasm2.load(
                    str(circuit_info.qasm_path),
                    custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
                )
                metrics = analyze_circuit(circuit)
            except Exception as e:
                log(f"  Error loading {circuit_info.name}: {e}")
                continue

            gate_density = metrics.total_gates / circuit_info.num_qubits if circuit_info.num_qubits > 0 else 0.0
            two_qubit_ratio = metrics.two_qubit_gates / metrics.total_gates if metrics.total_gates > 0 else 0.0

            record = CircuitRecord(
                id=None,
                name=circuit_info.name,
                category=circuit_info.category,
                source=circuit_info.source,
                qasm_path=str(circuit_info.qasm_path),
                num_qubits=circuit_info.num_qubits,
                initial_depth=metrics.depth,
                initial_two_qubit_gates=metrics.two_qubit_gates,
                initial_two_qubit_depth=metrics.two_qubit_depth,
                initial_total_gates=metrics.total_gates,
                gate_density=gate_density,
                two_qubit_ratio=two_qubit_ratio,
            )

            db.insert_circuit(record)
            imported += 1

        log(f"Imported {imported} artifact circuits")
        return imported


def import_from_metadata_json(
    db: TrajectoryDatabase,
    metadata_path: Path | str,
    source: str = "local",
    skip_existing: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> int:
    """Import circuits from a metadata.json file (existing benchmark format).

    Args:
        db: TrajectoryDatabase to import into
        metadata_path: Path to metadata.json
        source: Source name to use
        skip_existing: Skip circuits already in database
        progress_callback: Callback for progress messages

    Returns:
        Number of circuits imported
    """

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    metadata_path = Path(metadata_path)
    base_dir = metadata_path.parent

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    imported = 0
    for entry in metadata.get("circuits", []):
        name = entry["name"]

        if skip_existing:
            existing = db.get_circuit_by_name(name)
            if existing is not None:
                continue

        qasm_path = base_dir / entry["file"]
        metrics = entry["metrics"]
        num_qubits = entry["num_qubits"]
        tags = entry.get("tags", [])
        category = tags[0] if tags else "local"

        gate_density = metrics["total_gates"] / num_qubits if num_qubits > 0 else 0.0
        two_qubit_ratio = metrics["two_qubit_gates"] / metrics["total_gates"] if metrics["total_gates"] > 0 else 0.0

        record = CircuitRecord(
            id=None,
            name=name,
            category=category,
            source=source,
            qasm_path=str(qasm_path),
            num_qubits=num_qubits,
            initial_depth=metrics["depth"],
            initial_two_qubit_gates=metrics["two_qubit_gates"],
            initial_two_qubit_depth=metrics["two_qubit_depth"],
            initial_total_gates=metrics["total_gates"],
            gate_density=gate_density,
            two_qubit_ratio=two_qubit_ratio,
        )

        db.insert_circuit(record)
        imported += 1

    log(f"Imported {imported} circuits from {metadata_path}")
    return imported


def import_from_artifacts_dir(
    db: TrajectoryDatabase,
    artifacts_dir: Path | str,
    category: str | None = None,
    max_qubits: int = 20,
    skip_existing: bool = True,
    step1_db_path: Path | str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> int:
    """Import circuits from optimized artifact outputs.

    Args:
        db: TrajectoryDatabase to import into
        artifacts_dir: Directory containing optimized QASM artifacts
        category: Category override (None = auto-detect from step-1 DB
            or name prefixes)
        max_qubits: Maximum number of qubits
        skip_existing: Skip circuits already in database
        step1_db_path: Optional path to step-1 DB for category lookup
        progress_callback: Callback for progress messages

    Returns:
        Number of circuits imported
    """
    importer = ArtifactCircuitImporter(
        artifacts_dir, step1_db_path=step1_db_path
    )
    try:
        return importer.import_to_database(
            db,
            category=category,
            max_qubits=max_qubits,
            skip_existing=skip_existing,
            progress_callback=progress_callback,
        )
    finally:
        if importer._step1_db is not None:
            importer._step1_db.close()
