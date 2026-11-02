"""SQLite-backed trajectory database for offline RL training.

This module provides storage for (state, action, reward, next_state, done) tuples
collected from quantum circuit optimization experiments.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# SQL schema for the trajectory database
SCHEMA_SQL = """
-- circuits: benchmark circuit registry with computed features
CREATE TABLE IF NOT EXISTS circuits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'benchpress', 'local', etc.
    qasm_path TEXT,
    num_qubits INTEGER NOT NULL,
    initial_depth INTEGER NOT NULL,
    initial_two_qubit_gates INTEGER NOT NULL,
    initial_two_qubit_depth INTEGER NOT NULL,
    initial_total_gates INTEGER NOT NULL,
    -- Derived features for RL state
    gate_density REAL NOT NULL,  -- total_gates / num_qubits
    two_qubit_ratio REAL NOT NULL,  -- two_qubit_gates / total_gates (0 if total=0)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_circuits_category ON circuits(category);
CREATE INDEX IF NOT EXISTS idx_circuits_source ON circuits(source);
CREATE INDEX IF NOT EXISTS idx_circuits_num_qubits ON circuits(num_qubits);

-- optimizers: optimizer configurations (action space)
CREATE TABLE IF NOT EXISTS optimizers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    runner_type TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '{}',
    description TEXT
);

-- optimization_runs: single optimizer results (for analysis)
CREATE TABLE IF NOT EXISTS optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id INTEGER NOT NULL REFERENCES circuits(id),
    optimizer_id INTEGER NOT NULL REFERENCES optimizers(id),
    -- Input metrics
    input_depth INTEGER NOT NULL,
    input_two_qubit_gates INTEGER NOT NULL,
    input_two_qubit_depth INTEGER NOT NULL,
    input_total_gates INTEGER NOT NULL,
    -- Output metrics
    output_depth INTEGER NOT NULL,
    output_two_qubit_gates INTEGER NOT NULL,
    output_two_qubit_depth INTEGER NOT NULL,
    output_total_gates INTEGER NOT NULL,
    -- Execution info
    duration_seconds REAL NOT NULL,
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT,
    artifact_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_circuit ON optimization_runs(circuit_id);
CREATE INDEX IF NOT EXISTS idx_runs_optimizer ON optimization_runs(optimizer_id);

-- trajectories: optimization chains (episodes)
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id INTEGER NOT NULL REFERENCES circuits(id),
    chain_name TEXT NOT NULL,
    num_steps INTEGER NOT NULL,
    -- Initial and final metrics for the episode
    initial_depth INTEGER NOT NULL,
    initial_two_qubit_gates INTEGER NOT NULL,
    initial_two_qubit_depth INTEGER NOT NULL,
    initial_total_gates INTEGER NOT NULL,
    final_depth INTEGER NOT NULL,
    final_two_qubit_gates INTEGER NOT NULL,
    final_two_qubit_depth INTEGER NOT NULL,
    final_total_gates INTEGER NOT NULL,
    -- Episode totals
    total_duration_seconds REAL NOT NULL,
    total_reward REAL NOT NULL,
    improvement_percentage REAL NOT NULL,  -- % reduction in 2Q gates
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trajectories_circuit ON trajectories(circuit_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_chain_name ON trajectories(chain_name);

-- trajectory_steps: (s, a, r, s', done) tuples for RL
CREATE TABLE IF NOT EXISTS trajectory_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    step_index INTEGER NOT NULL,
    optimizer_id INTEGER NOT NULL REFERENCES optimizers(id),
    -- State s (before action)
    state_depth INTEGER NOT NULL,
    state_two_qubit_gates INTEGER NOT NULL,
    state_two_qubit_depth INTEGER NOT NULL,
    state_total_gates INTEGER NOT NULL,
    state_num_qubits INTEGER NOT NULL,
    state_gate_density REAL NOT NULL,
    state_two_qubit_ratio REAL NOT NULL,
    state_steps_taken INTEGER NOT NULL,
    state_time_budget_remaining REAL NOT NULL,
    state_category_json TEXT NOT NULL,  -- one-hot encoding as JSON array
    -- Next state s' (after action)
    next_state_depth INTEGER NOT NULL,
    next_state_two_qubit_gates INTEGER NOT NULL,
    next_state_two_qubit_depth INTEGER NOT NULL,
    next_state_total_gates INTEGER NOT NULL,
    next_state_gate_density REAL NOT NULL,
    next_state_two_qubit_ratio REAL NOT NULL,
    next_state_steps_taken INTEGER NOT NULL,
    next_state_time_budget_remaining REAL NOT NULL,
    -- Rewards (multiple variants for offline experimentation)
    reward_improvement_only REAL NOT NULL,
    reward_efficiency REAL NOT NULL,
    reward_multi_objective REAL NOT NULL,
    reward_sparse_final REAL NOT NULL,
    reward_category_relative REAL NOT NULL DEFAULT 0.0,
    reward_efficiency_normalized REAL NOT NULL DEFAULT 0.0,
    -- Episode info
    done INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trajectory_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON trajectory_steps(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_steps_optimizer ON trajectory_steps(optimizer_id);
CREATE INDEX IF NOT EXISTS idx_steps_done ON trajectory_steps(done);
"""


@dataclass
class CircuitRecord:
    """Record for a benchmark circuit."""

    id: int | None
    name: str
    category: str
    source: str
    qasm_path: str | None
    num_qubits: int
    initial_depth: int
    initial_two_qubit_gates: int
    initial_two_qubit_depth: int
    initial_total_gates: int
    gate_density: float
    two_qubit_ratio: float


@dataclass
class OptimizerRecord:
    """Record for an optimizer configuration."""

    id: int | None
    name: str
    runner_type: str
    options: dict[str, Any]
    description: str | None = None


@dataclass
class TrajectoryStepRecord:
    """Record for a single (s, a, r, s', done) tuple."""

    trajectory_id: int
    step_index: int
    optimizer_id: int
    # State s
    state_depth: int
    state_two_qubit_gates: int
    state_two_qubit_depth: int
    state_total_gates: int
    state_num_qubits: int
    state_gate_density: float
    state_two_qubit_ratio: float
    state_steps_taken: int
    state_time_budget_remaining: float
    state_category: list[float]  # one-hot encoding
    # Next state s'
    next_state_depth: int
    next_state_two_qubit_gates: int
    next_state_two_qubit_depth: int
    next_state_total_gates: int
    next_state_gate_density: float
    next_state_two_qubit_ratio: float
    next_state_steps_taken: int
    next_state_time_budget_remaining: float
    # Rewards
    reward_improvement_only: float
    reward_efficiency: float
    reward_multi_objective: float
    reward_sparse_final: float
    reward_category_relative: float = 0.0
    reward_efficiency_normalized: float = 0.0
    # Episode info
    done: bool = False
    duration_seconds: float = 0.0


@dataclass
class SARSTuple:
    """State-Action-Reward-NextState tuple for RL training."""

    state: np.ndarray
    action: int  # optimizer_id
    reward: float
    next_state: np.ndarray
    done: bool


class TrajectoryDatabase:
    """SQLite-backed database for storing RL trajectories."""

    def __init__(self, db_path: Path | str):
        """Initialize the trajectory database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # Thread-local storage for connections
        self._write_lock = threading.RLock()
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # Allow cross-thread access (for safety)
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrent performance
            self._local.conn.execute("PRAGMA journal_mode = WAL")
            # Set busy timeout for handling lock contention
            self._local.conn.execute("PRAGMA busy_timeout = 5000")
        return self._local.conn

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(SCHEMA_SQL)
        optimization_run_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(optimization_runs)").fetchall()
        }
        if "metadata_json" not in optimization_run_columns:
            conn.execute(
                "ALTER TABLE optimization_runs "
                "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        conn.commit()

    def close(self) -> None:
        """Close the thread-local database connection.
        
        Note: This only closes the connection for the current thread.
        Other thread connections will be cleaned up when those threads exit.
        """
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self) -> "TrajectoryDatabase":
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    # --- Circuit CRUD ---

    def insert_circuit(self, circuit: CircuitRecord) -> int:
        """Insert a circuit record and return its ID."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO circuits (
                    name, category, source, qasm_path, num_qubits,
                    initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                    initial_total_gates, gate_density, two_qubit_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    circuit.name,
                    circuit.category,
                    circuit.source,
                    circuit.qasm_path,
                    circuit.num_qubits,
                    circuit.initial_depth,
                    circuit.initial_two_qubit_gates,
                    circuit.initial_two_qubit_depth,
                    circuit.initial_total_gates,
                    circuit.gate_density,
                    circuit.two_qubit_ratio,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_circuit_by_name(self, name: str) -> CircuitRecord | None:
        """Get a circuit by name."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM circuits WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return CircuitRecord(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            source=row["source"],
            qasm_path=row["qasm_path"],
            num_qubits=row["num_qubits"],
            initial_depth=row["initial_depth"],
            initial_two_qubit_gates=row["initial_two_qubit_gates"],
            initial_two_qubit_depth=row["initial_two_qubit_depth"],
            initial_total_gates=row["initial_total_gates"],
            gate_density=row["gate_density"],
            two_qubit_ratio=row["two_qubit_ratio"],
        )

    def get_circuit_by_id(self, circuit_id: int) -> CircuitRecord | None:
        """Get a circuit by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM circuits WHERE id = ?", (circuit_id,)
        ).fetchone()
        if row is None:
            return None
        return CircuitRecord(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            source=row["source"],
            qasm_path=row["qasm_path"],
            num_qubits=row["num_qubits"],
            initial_depth=row["initial_depth"],
            initial_two_qubit_gates=row["initial_two_qubit_gates"],
            initial_two_qubit_depth=row["initial_two_qubit_depth"],
            initial_total_gates=row["initial_total_gates"],
            gate_density=row["gate_density"],
            two_qubit_ratio=row["two_qubit_ratio"],
        )

    def list_circuits(
        self,
        category: str | None = None,
        source: str | None = None,
        max_qubits: int | None = None,
    ) -> list[CircuitRecord]:
        """List circuits with optional filters."""
        conn = self._get_connection()
        query = "SELECT * FROM circuits WHERE 1=1"
        params: list[Any] = []

        if category is not None:
            query += " AND category = ?"
            params.append(category)
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        if max_qubits is not None:
            query += " AND num_qubits <= ?"
            params.append(max_qubits)

        query += " ORDER BY category, name"

        rows = conn.execute(query, params).fetchall()
        return [
            CircuitRecord(
                id=row["id"],
                name=row["name"],
                category=row["category"],
                source=row["source"],
                qasm_path=row["qasm_path"],
                num_qubits=row["num_qubits"],
                initial_depth=row["initial_depth"],
                initial_two_qubit_gates=row["initial_two_qubit_gates"],
                initial_two_qubit_depth=row["initial_two_qubit_depth"],
                initial_total_gates=row["initial_total_gates"],
                gate_density=row["gate_density"],
                two_qubit_ratio=row["two_qubit_ratio"],
            )
            for row in rows
        ]

    # --- Optimizer CRUD ---

    def insert_optimizer(self, optimizer: OptimizerRecord) -> int:
        """Insert an optimizer record and return its ID."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO optimizers (name, runner_type, options_json, description)
                VALUES (?, ?, ?, ?)
                """,
                (
                    optimizer.name,
                    optimizer.runner_type,
                    json.dumps(optimizer.options),
                    optimizer.description,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_optimizer_by_name(self, name: str) -> OptimizerRecord | None:
        """Get an optimizer by name."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM optimizers WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return OptimizerRecord(
            id=row["id"],
            name=row["name"],
            runner_type=row["runner_type"],
            options=json.loads(row["options_json"]),
            description=row["description"],
        )

    def list_optimizers(self) -> list[OptimizerRecord]:
        """List all optimizers."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM optimizers ORDER BY name").fetchall()
        return [
            OptimizerRecord(
                id=row["id"],
                name=row["name"],
                runner_type=row["runner_type"],
                options=json.loads(row["options_json"]),
                description=row["description"],
            )
            for row in rows
        ]

    def get_or_create_optimizer(self, optimizer: OptimizerRecord) -> int:
        """Get optimizer ID, refreshing its configuration when it already exists."""
        existing = self.get_optimizer_by_name(optimizer.name)
        if existing is not None:
            conn = self._get_connection()
            with self._write_lock:
                conn.execute(
                    """
                    UPDATE optimizers
                    SET runner_type = ?, options_json = ?, description = ?
                    WHERE id = ?
                    """,
                    (
                        optimizer.runner_type,
                        json.dumps(optimizer.options),
                        optimizer.description,
                        existing.id,
                    ),
                )
                conn.commit()
            return existing.id or 0
        return self.insert_optimizer(optimizer)

    # --- Optimization Run CRUD ---

    def insert_optimization_run(
        self,
        circuit_id: int,
        optimizer_id: int,
        input_depth: int,
        input_two_qubit_gates: int,
        input_two_qubit_depth: int,
        input_total_gates: int,
        output_depth: int,
        output_two_qubit_gates: int,
        output_two_qubit_depth: int,
        output_total_gates: int,
        duration_seconds: float,
        success: bool = True,
        error_message: str | None = None,
        artifact_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert an optimization run record and return its ID."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO optimization_runs (
                    circuit_id, optimizer_id,
                    input_depth, input_two_qubit_gates, input_two_qubit_depth, input_total_gates,
                    output_depth, output_two_qubit_gates, output_two_qubit_depth, output_total_gates,
                    duration_seconds, success, error_message, artifact_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    circuit_id,
                    optimizer_id,
                    input_depth,
                    input_two_qubit_gates,
                    input_two_qubit_depth,
                    input_total_gates,
                    output_depth,
                    output_two_qubit_gates,
                    output_two_qubit_depth,
                    output_total_gates,
                    duration_seconds,
                    1 if success else 0,
                    error_message,
                    artifact_path,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def run_exists(self, circuit_id: int, optimizer_id: int) -> bool:
        """Check if an optimization run already exists for this circuit/optimizer pair.
        
        Args:
            circuit_id: ID of the circuit
            optimizer_id: ID of the optimizer
            
        Returns:
            True if a run exists, False otherwise
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT 1 FROM optimization_runs WHERE circuit_id = ? AND optimizer_id = ?",
            (circuit_id, optimizer_id),
        ).fetchone()
        return row is not None

    def count_optimization_runs(self) -> int:
        """Count total number of optimization runs."""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM optimization_runs").fetchone()
        return row["cnt"] if row else 0

    # --- Trajectory CRUD ---

    def insert_trajectory(
        self,
        circuit_id: int,
        chain_name: str,
        num_steps: int,
        initial_depth: int,
        initial_two_qubit_gates: int,
        initial_two_qubit_depth: int,
        initial_total_gates: int,
        final_depth: int,
        final_two_qubit_gates: int,
        final_two_qubit_depth: int,
        final_total_gates: int,
        total_duration_seconds: float,
        total_reward: float,
        improvement_percentage: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a trajectory (episode) record and return its ID."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO trajectories (
                    circuit_id, chain_name, num_steps,
                    initial_depth, initial_two_qubit_gates, initial_two_qubit_depth, initial_total_gates,
                    final_depth, final_two_qubit_gates, final_two_qubit_depth, final_total_gates,
                    total_duration_seconds, total_reward, improvement_percentage, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    circuit_id,
                    chain_name,
                    num_steps,
                    initial_depth,
                    initial_two_qubit_gates,
                    initial_two_qubit_depth,
                    initial_total_gates,
                    final_depth,
                    final_two_qubit_gates,
                    final_two_qubit_depth,
                    final_total_gates,
                    total_duration_seconds,
                    total_reward,
                    improvement_percentage,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def insert_trajectory_step(self, step: TrajectoryStepRecord) -> int:
        """Insert a trajectory step (s, a, r, s', done) record."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO trajectory_steps (
                    trajectory_id, step_index, optimizer_id,
                    state_depth, state_two_qubit_gates, state_two_qubit_depth, state_total_gates,
                    state_num_qubits, state_gate_density, state_two_qubit_ratio,
                    state_steps_taken, state_time_budget_remaining, state_category_json,
                    next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                    next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                    next_state_steps_taken, next_state_time_budget_remaining,
                    reward_improvement_only, reward_efficiency, reward_multi_objective,
                    reward_sparse_final, reward_category_relative, reward_efficiency_normalized,
                    done, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.trajectory_id,
                    step.step_index,
                    step.optimizer_id,
                    step.state_depth,
                    step.state_two_qubit_gates,
                    step.state_two_qubit_depth,
                    step.state_total_gates,
                    step.state_num_qubits,
                    step.state_gate_density,
                    step.state_two_qubit_ratio,
                    step.state_steps_taken,
                    step.state_time_budget_remaining,
                    json.dumps(step.state_category),
                    step.next_state_depth,
                    step.next_state_two_qubit_gates,
                    step.next_state_two_qubit_depth,
                    step.next_state_total_gates,
                    step.next_state_gate_density,
                    step.next_state_two_qubit_ratio,
                    step.next_state_steps_taken,
                    step.next_state_time_budget_remaining,
                    step.reward_improvement_only,
                    step.reward_efficiency,
                    step.reward_multi_objective,
                    step.reward_sparse_final,
                    step.reward_category_relative,
                    step.reward_efficiency_normalized,
                    1 if step.done else 0,
                    step.duration_seconds,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    # --- Query Methods ---

    def count_trajectories(self) -> int:
        """Count total number of trajectories."""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM trajectories").fetchone()
        return row["cnt"] if row else 0

    def count_trajectory_steps(self) -> int:
        """Count total number of trajectory steps."""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM trajectory_steps").fetchone()
        return row["cnt"] if row else 0

    def trajectory_exists(self, circuit_id: int, chain_name: str) -> bool:
        """Check if a trajectory already exists for this circuit/chain combination."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT 1 FROM trajectories WHERE circuit_id = ? AND chain_name = ?",
            (circuit_id, chain_name),
        ).fetchone()
        return row is not None

    # --- RL Training Interface ---

    def sample_batch(
        self,
        batch_size: int,
        reward_type: str = "reward_efficiency",
        seed: int | None = None,
    ) -> list[SARSTuple]:
        """Sample a random batch of (s, a, r, s', done) tuples for training.

        Args:
            batch_size: Number of tuples to sample
            reward_type: Which reward variant to use
            seed: Random seed for reproducibility

        Returns:
            List of SARSTuple objects
        """
        if seed is not None:
            np.random.seed(seed)

        conn = self._get_connection()

        # Get total count
        total = self.count_trajectory_steps()
        if total == 0:
            return []

        # Sample random indices
        sample_size = min(batch_size, total)
        indices = np.random.choice(total, size=sample_size, replace=False)

        # Fetch the sampled rows (using OFFSET is inefficient but works for prototype)
        # For production, consider using rowid-based sampling
        tuples: list[SARSTuple] = []
        for idx in sorted(indices):
            row = conn.execute(
                f"""
                SELECT
                    state_depth, state_two_qubit_gates, state_two_qubit_depth, state_total_gates,
                    state_num_qubits, state_gate_density, state_two_qubit_ratio,
                    state_steps_taken, state_time_budget_remaining, state_category_json,
                    next_state_depth, next_state_two_qubit_gates, next_state_two_qubit_depth,
                    next_state_total_gates, next_state_gate_density, next_state_two_qubit_ratio,
                    next_state_steps_taken, next_state_time_budget_remaining,
                    optimizer_id, {reward_type}, done
                FROM trajectory_steps
                ORDER BY id
                LIMIT 1 OFFSET ?
                """,
                (int(idx),),
            ).fetchone()

            if row is None:
                continue

            state_category = json.loads(row["state_category_json"])
            state = np.array(
                [
                    row["state_depth"],
                    row["state_two_qubit_gates"],
                    row["state_two_qubit_depth"],
                    row["state_total_gates"],
                    row["state_num_qubits"],
                    row["state_gate_density"],
                    row["state_two_qubit_ratio"],
                    row["state_steps_taken"],
                    row["state_time_budget_remaining"],
                ]
                + state_category,
                dtype=np.float32,
            )

            next_state = np.array(
                [
                    row["next_state_depth"],
                    row["next_state_two_qubit_gates"],
                    row["next_state_two_qubit_depth"],
                    row["next_state_total_gates"],
                    row["state_num_qubits"],  # num_qubits doesn't change
                    row["next_state_gate_density"],
                    row["next_state_two_qubit_ratio"],
                    row["next_state_steps_taken"],
                    row["next_state_time_budget_remaining"],
                ]
                + state_category,  # category doesn't change
                dtype=np.float32,
            )

            tuples.append(
                SARSTuple(
                    state=state,
                    action=row["optimizer_id"],
                    reward=row[reward_type],
                    next_state=next_state,
                    done=bool(row["done"]),
                )
            )

        return tuples

    def get_sars_tuples(
        self,
        reward_type: str = "reward_efficiency",
        circuit_ids: Sequence[int] | None = None,
    ) -> Iterator[SARSTuple]:
        """Iterate over all (state, action, reward, next_state, done) tuples.

        Args:
            reward_type: Which reward variant to use
            circuit_ids: Optional list of circuit IDs to filter by

        Yields:
            SARSTuple objects
        """
        conn = self._get_connection()

        query = f"""
            SELECT
                ts.state_depth, ts.state_two_qubit_gates, ts.state_two_qubit_depth,
                ts.state_total_gates, ts.state_num_qubits, ts.state_gate_density,
                ts.state_two_qubit_ratio, ts.state_steps_taken, ts.state_time_budget_remaining,
                ts.state_category_json,
                ts.next_state_depth, ts.next_state_two_qubit_gates, ts.next_state_two_qubit_depth,
                ts.next_state_total_gates, ts.next_state_gate_density, ts.next_state_two_qubit_ratio,
                ts.next_state_steps_taken, ts.next_state_time_budget_remaining,
                ts.optimizer_id, ts.{reward_type}, ts.done
            FROM trajectory_steps ts
            JOIN trajectories t ON ts.trajectory_id = t.id
        """

        params: list[Any] = []
        if circuit_ids is not None:
            placeholders = ",".join("?" for _ in circuit_ids)
            query += f" WHERE t.circuit_id IN ({placeholders})"
            params.extend(circuit_ids)

        query += " ORDER BY ts.trajectory_id, ts.step_index"

        cursor = conn.execute(query, params)

        for row in cursor:
            state_category = json.loads(row["state_category_json"])
            state = np.array(
                [
                    row["state_depth"],
                    row["state_two_qubit_gates"],
                    row["state_two_qubit_depth"],
                    row["state_total_gates"],
                    row["state_num_qubits"],
                    row["state_gate_density"],
                    row["state_two_qubit_ratio"],
                    row["state_steps_taken"],
                    row["state_time_budget_remaining"],
                ]
                + state_category,
                dtype=np.float32,
            )

            next_state = np.array(
                [
                    row["next_state_depth"],
                    row["next_state_two_qubit_gates"],
                    row["next_state_two_qubit_depth"],
                    row["next_state_total_gates"],
                    row["state_num_qubits"],
                    row["next_state_gate_density"],
                    row["next_state_two_qubit_ratio"],
                    row["next_state_steps_taken"],
                    row["next_state_time_budget_remaining"],
                ]
                + state_category,
                dtype=np.float32,
            )

            yield SARSTuple(
                state=state,
                action=row["optimizer_id"],
                reward=row[reward_type],
                next_state=next_state,
                done=bool(row["done"]),
            )

    def export_to_d4rl_format(
        self,
        reward_type: str = "reward_efficiency",
    ) -> dict[str, np.ndarray]:
        """Export trajectories to D4RL-compatible format.

        Returns a dictionary with:
            - observations: (N, state_dim) array
            - actions: (N,) array of action indices
            - rewards: (N,) array
            - next_observations: (N, state_dim) array
            - terminals: (N,) boolean array
        """
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        terminals: list[bool] = []

        for sars in self.get_sars_tuples(reward_type=reward_type):
            observations.append(sars.state)
            actions.append(sars.action)
            rewards.append(sars.reward)
            next_observations.append(sars.next_state)
            terminals.append(sars.done)

        if not observations:
            # Return empty arrays with correct dtype
            return {
                "observations": np.array([], dtype=np.float32),
                "actions": np.array([], dtype=np.int64),
                "rewards": np.array([], dtype=np.float32),
                "next_observations": np.array([], dtype=np.float32),
                "terminals": np.array([], dtype=bool),
            }

        return {
            "observations": np.stack(observations),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_observations": np.stack(next_observations),
            "terminals": np.array(terminals, dtype=bool),
        }

    # --- Statistics ---

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()

        stats: dict[str, Any] = {}

        # Counts
        stats["num_circuits"] = conn.execute(
            "SELECT COUNT(*) FROM circuits"
        ).fetchone()[0]
        stats["num_optimizers"] = conn.execute(
            "SELECT COUNT(*) FROM optimizers"
        ).fetchone()[0]
        stats["num_optimization_runs"] = conn.execute(
            "SELECT COUNT(*) FROM optimization_runs"
        ).fetchone()[0]
        stats["num_trajectories"] = self.count_trajectories()
        stats["num_trajectory_steps"] = self.count_trajectory_steps()

        # Categories
        rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM circuits GROUP BY category"
        ).fetchall()
        stats["circuits_by_category"] = {row["category"]: row["cnt"] for row in rows}

        # Trajectory stats
        if stats["num_trajectories"] > 0:
            row = conn.execute(
                """
                SELECT
                    AVG(improvement_percentage) as avg_improvement,
                    MAX(improvement_percentage) as max_improvement,
                    AVG(total_reward) as avg_reward,
                    AVG(num_steps) as avg_steps
                FROM trajectories
                """
            ).fetchone()
            stats["avg_improvement_percentage"] = row["avg_improvement"]
            stats["max_improvement_percentage"] = row["max_improvement"]
            stats["avg_total_reward"] = row["avg_reward"]
            stats["avg_trajectory_length"] = row["avg_steps"]

        return stats
