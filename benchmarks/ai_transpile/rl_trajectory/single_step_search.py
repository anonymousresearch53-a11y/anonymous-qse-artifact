"""Single-step grid search runner for exhaustive optimizer evaluation.

This module provides infrastructure to run exhaustive single-step optimization
across all circuits and optimizers, with async execution and concurrency control.

Key features:
- Async execution with semaphore-based concurrency control
- WISQ+BQSKit limited to 1 concurrent instance
- Resumable execution (skips completed runs)
- Artifact storage for output circuits
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from ..chain_executor import ChainStep, execute_chain
from ..transpilers import CircuitMetrics, analyze_circuit
from .database import CircuitRecord, OptimizerRecord, TrajectoryDatabase
from .grid_search import (
    ALL_OPTIMIZER_CONFIGS,
    OPTIMIZER_CONFIGS,
)

# Suppress warnings in the current process (subprocesses need PYTHONWARNINGS env var)
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message=".*sample.*too small.*",
)

# Suppress verbose INFO logging from qiskit_ibm_transpiler (disrupts progress display)
logging.getLogger("qiskit_ibm_transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit_ibm_transpiler").propagate = False
logging.getLogger("qiskit_ibm_transpiler").handlers = []


@dataclass
class SingleStepConfig:
    """Configuration for single-step grid search.
    
    Attributes:
        database_path: Path to the SQLite trajectory database
        max_qubits: Maximum number of qubits for circuits to include
        categories: Circuit categories to include (None = all)
        optimizers: List of optimizer names to use (None = all)
        rerun_optimizers: Optimizers to force rerun even if results exist (None = none)
        wisq_bqskit_timeout: Timeout in seconds for WISQ+BQSKit (default 5 min)
        optimizer_timeout_seconds: Hard timeout for each optimizer run (0 = disabled)
        max_concurrent_fast: Max concurrent runs for fast optimizers
        max_concurrent_wisq_rules: Max concurrent runs for wisq_rules
        max_concurrent_wisq_bqskit: Max concurrent runs for wisq_bqskit (should be 1)
        artifact_dir: Directory to save output circuit artifacts
        save_artifacts: Whether to save output circuits
    """
    
    database_path: Path = field(default_factory=lambda: Path("data/trajectories.db"))
    max_qubits: int = 20
    categories: list[str] | None = None
    optimizers: list[str] | None = None
    rerun_optimizers: list[str] | None = None  # Force rerun these even if results exist
    wisq_bqskit_timeout: int = 300  # 5 minutes
    optimizer_timeout_seconds: int = 900  # 15 minutes
    max_concurrent_fast: int = 4  # tket, qiskit_ai, qiskit_standard
    max_concurrent_wisq_rules: int = 2
    max_concurrent_wisq_bqskit: int = 1  # CRITICAL: Only 1 at a time
    artifact_dir: Path = field(default_factory=lambda: Path("data/artifacts"))
    save_artifacts: bool = True


@dataclass
class SingleStepResult:
    """Result of a single optimization run."""
    
    circuit_id: int
    optimizer_id: int
    circuit_name: str
    optimizer_name: str
    input_metrics: CircuitMetrics
    output_metrics: CircuitMetrics
    duration_seconds: float
    success: bool
    error_message: str | None = None
    artifact_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def improvement_percentage(self) -> float:
        """Calculate improvement in 2-qubit gate count."""
        if self.input_metrics.two_qubit_gates == 0:
            return 0.0
        return 100.0 * (
            self.input_metrics.two_qubit_gates - self.output_metrics.two_qubit_gates
        ) / self.input_metrics.two_qubit_gates


@dataclass
class SingleStepProgress:
    """Progress information for single-step grid search."""
    
    total_runs: int
    completed_runs: int
    skipped_runs: int
    failed_runs: int
    current_circuit: str
    current_optimizer: str
    elapsed_seconds: float
    
    @property
    def percent_complete(self) -> float:
        """Percentage of total runs completed (including skipped)."""
        if self.total_runs == 0:
            return 100.0
        return 100.0 * (self.completed_runs + self.skipped_runs) / self.total_runs


@dataclass
class SingleStepReport:
    """Report summarizing single-step grid search results."""
    
    total_circuits: int
    total_optimizers: int
    total_runs: int
    completed_runs: int
    skipped_runs: int
    failed_runs: int
    total_duration_seconds: float
    best_by_optimizer: dict[str, dict[str, Any]]
    failures: list[dict[str, Any]]
    equivalence_counts: dict[str, int] = field(default_factory=dict)


class OptimizerRunAborted(Exception):
    """Raised when an optimizer run is aborted by user interrupt."""


# Categorize optimizers by resource requirements
FAST_OPTIMIZERS = {
    "tket",
    "qiskit_ai",
    "qiskit_standard",
    "tket_hardware",
    "qiskit_ai_hardware",
    "qiskit_standard_hardware",
}
WISQ_RULES_OPTIMIZER = "wisq_rules"
WISQ_BQSKIT_OPTIMIZER = "wisq_bqskit"
WISQ_RULES_OPTIMIZERS = {WISQ_RULES_OPTIMIZER, "wisq_rules_hardware"}
WISQ_BQSKIT_OPTIMIZERS = {WISQ_BQSKIT_OPTIMIZER, "wisq_bqskit_hardware"}


def _start_process_group() -> None:
    """Start a new process group so timeout cleanup can kill child subprocesses."""
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass


def _terminate_process(process: mp.Process) -> None:
    """Terminate a child process, escalating to kill if it ignores SIGTERM."""
    if process.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    process.terminate()

    process.join(timeout=5)
    if not process.is_alive():
        return

    if process.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    process.kill()
    process.join(timeout=5)


def _kill_process_nowait(process: mp.Process) -> None:
    """Best-effort immediate kill for an already-aborting run."""
    if process.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass


def _send_process_payload(result_conn: Any, status: str, payload: Any) -> None:
    """Send a child-process result if the parent is still listening."""
    try:
        result_conn.send((status, payload))
    except (BrokenPipeError, EOFError, OSError):
        pass


def _run_single_step_process_entry(
    config: SingleStepConfig,
    circuit_record: CircuitRecord,
    optimizer_name: str,
    output_dir: Path,
    result_conn: Any,
) -> None:
    """Child-process entrypoint for one optimizer run."""
    _start_process_group()
    runner: AsyncSingleStepRunner | None = None
    try:
        runner = AsyncSingleStepRunner(config)
        result = runner._run_single_step_sync(circuit_record, optimizer_name, output_dir)
        _send_process_payload(result_conn, "result", result)
    except BaseException as exc:
        _send_process_payload(result_conn, "error", f"{type(exc).__name__}: {exc}")
    finally:
        if runner is not None:
            runner.close()
        result_conn.close()


class OptimizersProgressTracker:
    """Multi-progress bar tracker for optimizer execution.
    
    Displays overall progress and per-optimizer progress bars with detailed metrics:
    - Completed/Total circuits
    - Failed count
    - Currently running count
    - Success rate
    """
    
    def __init__(
        self,
        optimizer_names: list[str],
        optimizer_totals: dict[str, int],
        log_file: Path | None = None,
    ):
        """Initialize progress tracker.
        
        Args:
            optimizer_names: List of optimizer names to track
            optimizer_totals: Mapping of optimizer name to total circuit count
            log_file: Optional path to a clean plain-text log file.
                When provided, log messages and periodic summaries are
                written here without any ANSI escape sequences.
        """
        self.optimizer_names = optimizer_names
        self.optimizer_totals = optimizer_totals
        
        # Per-optimizer state
        self._completed: dict[str, int] = {name: 0 for name in optimizer_names}
        self._failed: dict[str, int] = {name: 0 for name in optimizer_names}
        self._running: dict[str, int] = {name: 0 for name in optimizer_names}
        self._skipped: dict[str, int] = {name: 0 for name in optimizer_names}
        self._durations: dict[str, list[float]] = {name: [] for name in optimizer_names}
        self._start_time: float = 0.0
        
        # Overall state
        self._total_completed = 0
        self._total_failed = 0
        self._total_running = 0
        self._total_skipped = 0
        
        # Clean file logging (no ANSI)
        self._log_file: Path | None = log_file
        self._log_fh: Any = None  # file handle, opened in __enter__
        self._log_event_count: int = 0  # counter for periodic summaries
        self._LOG_SUMMARY_INTERVAL: int = 50  # write summary every N events
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Rich progress
        self._progress: Progress | None = None
        self._overall_task: TaskID | None = None
        self._optimizer_tasks: dict[str, TaskID] = {}
    
    def __enter__(self) -> "OptimizersProgressTracker":
        """Start progress display and open log file."""
        from rich.console import Console

        self._start_time = time.perf_counter()

        # Open clean log file if requested
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self._log_file, "a", encoding="utf-8")  # noqa: SIM115
            total_runs = sum(self.optimizer_totals.values())
            self._write_log(
                f"=== Grid Search Started ===\n"
                f"Optimizers: {', '.join(self.optimizer_names)}\n"
                f"Total runs: {total_runs}\n"
            )
        
        # Create console - let Rich auto-detect terminal capabilities
        console = Console()
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("│"),
            TextColumn("[yellow]Failed: {task.fields[failed]}"),
            TextColumn("│"),
            TextColumn("[cyan]Running: {task.fields[running]}"),
            TextColumn("│"),
            TextColumn("[green]Success: {task.fields[success_rate]:.1f}%"),
            TextColumn("│"),
            TextColumn("[magenta]{task.fields[eta]}"),
            TextColumn("│"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=15,  # Balanced: smooth updates without overlap
        )
        self._progress.start()
        
        # Calculate total runs
        total_runs = sum(self.optimizer_totals.values())
        
        # Add overall progress bar
        self._overall_task = self._progress.add_task(
            "[bold cyan]Overall Progress",
            total=total_runs,
            failed=0,
            running=0,
            success_rate=0.0,
            eta="ETA: --",
        )

        # Add per-optimizer progress bars
        for name in self.optimizer_names:
            total = self.optimizer_totals[name]
            task_id = self._progress.add_task(
                f"  ├─ {name:15s}",
                total=total,
                failed=0,
                running=0,
                success_rate=0.0,
                eta="ETA: --",
            )
            self._optimizer_tasks[name] = task_id
        
        return self
    
    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Stop progress display and close log file."""
        if self._progress is not None:
            self._progress.stop()
        # Write final summary and close log file
        if self._log_fh is not None:
            self._write_log_summary(final=True)
            self._log_fh.close()
            self._log_fh = None
    
    async def start_task(self, optimizer_name: str) -> None:
        """Mark a task as started (running)."""
        async with self._lock:
            self._running[optimizer_name] += 1
            self._total_running += 1
            self._update_display(optimizer_name)
    
    async def complete_task(self, optimizer_name: str, success: bool, duration: float = 0.0) -> None:
        """Mark a task as completed."""
        async with self._lock:
            self._running[optimizer_name] -= 1
            self._total_running -= 1

            if success:
                self._completed[optimizer_name] += 1
                self._total_completed += 1
            else:
                self._failed[optimizer_name] += 1
                self._total_failed += 1

            if duration > 0:
                self._durations[optimizer_name].append(duration)

            self._update_display(optimizer_name)
            self._maybe_write_periodic_summary()
    
    async def skip_task(self, optimizer_name: str) -> None:
        """Mark a task as skipped (already completed)."""
        async with self._lock:
            self._skipped[optimizer_name] += 1
            self._total_skipped += 1
            self._update_display(optimizer_name)
            self._maybe_write_periodic_summary()
    
    def log(self, message: str, style: str | None = None) -> None:
        """Print a message above the progress bars and to the log file.
        
        Args:
            message: The message to print
            style: Optional Rich style (e.g., "bold green", "red", "dim")
        """
        if self._progress is not None:
            if style:
                self._progress.console.print(f"[{style}]{message}[/{style}]")
            else:
                self._progress.console.print(message)
        # Also write plain text to log file
        self._write_log(message)
    
    def _write_log(self, message: str) -> None:
        """Write a plain-text message to the log file (no ANSI)."""
        if self._log_fh is not None:
            import datetime

            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self._log_fh.write(f"[{timestamp}] {message}\n")
            self._log_fh.flush()
    
    def _write_log_summary(self, final: bool = False) -> None:
        """Write a periodic or final summary to the log file."""
        if self._log_fh is None:
            return
        
        total_done = self._total_completed + self._total_skipped + self._total_failed
        total_runs = sum(self.optimizer_totals.values())
        pct = 100.0 * total_done / total_runs if total_runs > 0 else 0.0
        
        label = "FINAL SUMMARY" if final else "PROGRESS"
        lines = [
            f"--- {label} ---",
            f"  Overall: {total_done}/{total_runs} ({pct:.1f}%)",
            f"  Completed: {self._total_completed}  Skipped: {self._total_skipped}"
            f"  Failed: {self._total_failed}  Running: {self._total_running}",
        ]
        for name in self.optimizer_names:
            done = self._completed[name] + self._skipped[name] + self._failed[name]
            total = self.optimizer_totals[name]
            eta = self._compute_eta(name)
            lines.append(
                f"  {name:20s}: {done}/{total}"
                f"  (ok={self._completed[name]} skip={self._skipped[name]}"
                f" fail={self._failed[name]} run={self._running[name]}"
                f" {eta})"
            )
        
        self._write_log("\n".join(lines))
    
    def _maybe_write_periodic_summary(self) -> None:
        """Write a periodic summary to the log file every N events."""
        self._log_event_count += 1
        if self._log_event_count % self._LOG_SUMMARY_INTERVAL == 0:
            self._write_log_summary()
    
    def _compute_eta(self, optimizer_name: str) -> str:
        """Compute wall-clock ETA for an optimizer using observed throughput.

        Uses actual completion rate (runs/second) which naturally accounts
        for concurrency — if 2 runs execute in parallel, throughput doubles
        and ETA halves compared to a naive duration-based estimate.
        """
        elapsed = time.perf_counter() - self._start_time
        done = self._completed[optimizer_name] + self._failed[optimizer_name]
        if done == 0 or elapsed <= 0:
            return "ETA: --"
        throughput = done / elapsed  # completions per second
        remaining = self.optimizer_totals[optimizer_name] - done - self._skipped[optimizer_name]
        if remaining <= 0:
            return "ETA: 0s"
        eta_seconds = remaining / throughput
        return self._format_eta(eta_seconds)

    def _compute_eta_seconds(self, optimizer_name: str) -> float | None:
        """Compute wall-clock ETA in seconds for an optimizer, or None if unknown."""
        elapsed = time.perf_counter() - self._start_time
        done = self._completed[optimizer_name] + self._failed[optimizer_name]
        if done == 0 or elapsed <= 0:
            return None
        throughput = done / elapsed
        remaining = self.optimizer_totals[optimizer_name] - done - self._skipped[optimizer_name]
        if remaining <= 0:
            return 0.0
        return remaining / throughput

    def _compute_overall_eta(self) -> str:
        """Compute overall ETA as the max across all optimizers (bottleneck)."""
        max_eta_seconds = 0.0
        any_data = False
        for name in self.optimizer_names:
            eta = self._compute_eta_seconds(name)
            if eta is not None:
                any_data = True
                if eta > max_eta_seconds:
                    max_eta_seconds = eta
        if not any_data:
            return "ETA: --"
        return self._format_eta(max_eta_seconds)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format an ETA in seconds to a human-readable string."""
        if seconds < 60:
            return f"ETA: {seconds:.0f}s"
        elif seconds < 3600:
            return f"ETA: {seconds / 60:.0f}m"
        else:
            return f"ETA: {seconds / 3600:.1f}h"

    def _update_display(self, optimizer_name: str) -> None:
        """Update progress bars (must be called with lock held)."""
        if self._progress is None:
            return
        
        # Update optimizer bar
        task_id = self._optimizer_tasks[optimizer_name]
        completed = self._completed[optimizer_name]
        skipped = self._skipped[optimizer_name]
        failed = self._failed[optimizer_name]
        running = self._running[optimizer_name]
        
        # Calculate success rate (skipped tasks were previously successful)
        total_done = completed + skipped + failed
        success_rate = 100.0 * (completed + skipped) / total_done if total_done > 0 else 0.0
        
        self._progress.update(
            task_id,
            completed=completed + skipped + failed,
            failed=failed,
            running=running,
            success_rate=success_rate,
            eta=self._compute_eta(optimizer_name),
        )

        # Update overall bar
        total_overall_done = self._total_completed + self._total_skipped + self._total_failed
        overall_success_rate = (
            100.0 * (self._total_completed + self._total_skipped) / total_overall_done
            if total_overall_done > 0 else 0.0
        )

        self._progress.update(
            self._overall_task,
            completed=self._total_completed + self._total_skipped + self._total_failed,
            failed=self._total_failed,
            running=self._total_running,
            success_rate=overall_success_rate,
            eta=self._compute_overall_eta(),
        )


class AsyncSingleStepRunner:
    """Async runner for single-step grid search with concurrency control."""
    
    def __init__(
        self,
        config: SingleStepConfig,
        progress_callback: Callable[[SingleStepProgress], None] | None = None,
        progress_tracker: OptimizersProgressTracker | None = None,
    ):
        """Initialize the async single-step runner.
        
        Args:
            config: Single-step search configuration
            progress_callback: Callback for progress updates (deprecated, use progress_tracker)
            progress_tracker: Progress tracker for multi-bar display
        """
        self.config = config
        self.progress_callback = progress_callback
        self.progress_tracker = progress_tracker
        self.db = TrajectoryDatabase(config.database_path)
        
        # Semaphores will be created in async context
        self._sem_fast: asyncio.Semaphore | None = None
        self._sem_wisq_rules: asyncio.Semaphore | None = None
        self._sem_wisq_bqskit: asyncio.Semaphore | None = None
        
        # Thread pool for CPU-bound optimization work
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_fast + 
                        config.max_concurrent_wisq_rules + 
                        config.max_concurrent_wisq_bqskit
        )
        self._executor_shutdown = False
        self._executor_lock = threading.Lock()
        self._abort_event = threading.Event()
        self._active_processes: set[mp.Process] = set()
        self._active_processes_lock = threading.Lock()
        
        # Ensure artifact directory exists
        if config.save_artifacts:
            config.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure optimizers are registered
        self._ensure_optimizers_registered()

    def _register_process(self, process: mp.Process) -> None:
        """Track an active optimizer child process for interrupt cleanup."""
        with self._active_processes_lock:
            self._active_processes.add(process)

    def _unregister_process(self, process: mp.Process) -> None:
        """Stop tracking an optimizer child process after it exits."""
        with self._active_processes_lock:
            self._active_processes.discard(process)

    def _snapshot_active_processes(self) -> list[mp.Process]:
        """Return a stable snapshot of active child processes."""
        with self._active_processes_lock:
            return list(self._active_processes)

    def _shutdown_executor(self, *, wait: bool, cancel_futures: bool) -> None:
        """Shut down the executor once, using interrupt-aware options."""
        with self._executor_lock:
            if self._executor_shutdown:
                return
            self._executor_shutdown = True
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def abort(self) -> None:
        """Abort the run and kill all active optimizer child processes."""
        already_aborting = self._abort_event.is_set()
        self._abort_event.set()

        for process in self._snapshot_active_processes():
            if already_aborting:
                _kill_process_nowait(process)
            else:
                _terminate_process(process)
            self._unregister_process(process)

        self._shutdown_executor(wait=False, cancel_futures=True)
    
    def _ensure_optimizers_registered(self) -> None:
        """Ensure all configured optimizers are in the database."""
        optimizer_names = self.config.optimizers or list(OPTIMIZER_CONFIGS.keys())
        
        for name in optimizer_names:
            if name not in ALL_OPTIMIZER_CONFIGS:
                raise ValueError(f"Unknown optimizer: {name}")
            
            opt_config = ALL_OPTIMIZER_CONFIGS[name]
            options = self._optimizer_options(name)
            
            optimizer = OptimizerRecord(
                id=None,
                name=name,
                runner_type=opt_config["runner_type"],
                options=options,
                description=opt_config.get("description"),
            )
            self.db.get_or_create_optimizer(optimizer)

    def _optimizer_options(self, optimizer_name: str) -> dict[str, Any]:
        options = dict(ALL_OPTIMIZER_CONFIGS[optimizer_name]["options"])
        if optimizer_name in WISQ_BQSKIT_OPTIMIZERS:
            options["opt_timeout"] = self.config.wisq_bqskit_timeout
        return options
    
    def _get_semaphore(self, optimizer_name: str) -> asyncio.Semaphore:
        """Get the appropriate semaphore for an optimizer."""
        if optimizer_name in WISQ_BQSKIT_OPTIMIZERS:
            assert self._sem_wisq_bqskit is not None
            return self._sem_wisq_bqskit
        elif optimizer_name in WISQ_RULES_OPTIMIZERS:
            assert self._sem_wisq_rules is not None
            return self._sem_wisq_rules
        else:
            assert self._sem_fast is not None
            return self._sem_fast
    
    def _load_circuit(self, circuit_record: CircuitRecord) -> QuantumCircuit | None:
        """Load a circuit from its QASM path."""
        if circuit_record.qasm_path is None:
            return None
        try:
            return qasm2.load(
                circuit_record.qasm_path,
                custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
            )
        except Exception:
            return None

    @staticmethod
    def _save_artifact(
        circuit: QuantumCircuit,
        artifact_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        artifact_qasm = qasm2.dumps(circuit.decompose(gates_to_decompose=["swap"]))
        artifact_path.write_text(artifact_qasm)
        layout = metadata.get("layout")
        if not layout:
            return
        layout_path = artifact_path.with_suffix(".layout.json")
        layout_path.write_text(json.dumps({
            "schema_version": 1,
            "artifact_qasm_sha256": hashlib.sha256(artifact_qasm.encode()).hexdigest(),
            "layout": layout,
        }, indent=2))
        metadata["layout_artifact_path"] = str(layout_path)
    
    def _run_single_step_sync(
        self,
        circuit_record: CircuitRecord,
        optimizer_name: str,
        output_dir: Path,
    ) -> SingleStepResult:
        """Run a single optimization step synchronously.
        
        This method is called in a thread pool executor.
        """
        # Suppress verbose qiskit logging (must be set after module import)
        for logger_name in ["qiskit_ibm_transpiler", "qiskit"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            logger.handlers = []
        
        # Create thread-local database connection for SQLite thread safety
        thread_db = TrajectoryDatabase(self.config.database_path)
        optimizer = thread_db.get_optimizer_by_name(optimizer_name)
        if optimizer is None or optimizer.id is None:
            return SingleStepResult(
                circuit_id=circuit_record.id or 0,
                optimizer_id=0,
                circuit_name=circuit_record.name,
                optimizer_name=optimizer_name,
                input_metrics=CircuitMetrics(0, 0, 0, 0),
                output_metrics=CircuitMetrics(0, 0, 0, 0),
                duration_seconds=0.0,
                success=False,
                error_message=f"Optimizer not found: {optimizer_name}",
            )
        
        # Load circuit
        circuit = self._load_circuit(circuit_record)
        if circuit is None:
            return SingleStepResult(
                circuit_id=circuit_record.id or 0,
                optimizer_id=optimizer.id,
                circuit_name=circuit_record.name,
                optimizer_name=optimizer_name,
                input_metrics=CircuitMetrics(0, 0, 0, 0),
                output_metrics=CircuitMetrics(0, 0, 0, 0),
                duration_seconds=0.0,
                success=False,
                error_message=f"Failed to load circuit: {circuit_record.qasm_path}",
            )
        
        input_metrics = analyze_circuit(circuit)
        
        # Get optimizer options
        opt_config = ALL_OPTIMIZER_CONFIGS[optimizer_name]
        options = self._optimizer_options(optimizer_name)
        
        step = ChainStep(
            runner_type=opt_config["runner_type"],
            options=options,
            name=optimizer_name,
        )
        
        # Retry for transient errors (e.g. Rust "Already borrowed" in qiskit-ibm-transpiler,
        # numpy scalar conversion bugs). 5 attempts with exponential backoff.
        max_attempts = 5
        last_error: Exception | None = None

        RETRIABLE_ERRORS = (
            "Already borrowed",
            "only length-1 arrays can be converted to Python scalars",
        )

        try:
            for attempt in range(max_attempts):
                try:
                    start_time = time.perf_counter()
                    result = execute_chain(
                        circuit.copy() if attempt > 0 else circuit,
                        steps=[step],
                        chain_name=optimizer_name,
                        output_dir=output_dir,
                        save_intermediates=self.config.save_artifacts,
                    )
                    duration = time.perf_counter() - start_time

                    if not result.step_results:
                        return SingleStepResult(
                            circuit_id=circuit_record.id or 0,
                            optimizer_id=optimizer.id,
                            circuit_name=circuit_record.name,
                            optimizer_name=optimizer_name,
                            input_metrics=CircuitMetrics(0, 0, 0, 0),
                            output_metrics=CircuitMetrics(0, 0, 0, 0),
                            duration_seconds=duration,
                            success=False,
                            error_message="No result returned from optimizer",
                        )

                    step_result = result.step_results[0]
                    output_metrics = step_result.output_metrics

                    metadata = dict(step_result.transpiled.metadata)
                    artifact_path: Path | None = None
                    if self.config.save_artifacts:
                        try:
                            artifact_dir = self.config.artifact_dir / circuit_record.name
                            artifact_dir.mkdir(parents=True, exist_ok=True)
                            artifact_path = artifact_dir / f"{optimizer_name}.qasm"
                            self._save_artifact(
                                step_result.transpiled.circuit, artifact_path, metadata
                            )
                        except Exception:
                            artifact_path = None  # Artifact save failed, but optimization succeeded

                    equivalence_failed = bool(metadata.get("equivalence_failed", False))
                    equivalence_error = metadata.get("equivalence_error")

                    return SingleStepResult(
                        circuit_id=circuit_record.id or 0,
                        optimizer_id=optimizer.id,
                        circuit_name=circuit_record.name,
                        optimizer_name=optimizer_name,
                        input_metrics=input_metrics,
                        output_metrics=output_metrics,
                        duration_seconds=duration,
                        success=not equivalence_failed,
                        error_message=(
                            str(equivalence_error) if equivalence_failed and equivalence_error else None
                        ),
                        artifact_path=artifact_path,
                        metadata=metadata,
                    )

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if attempt < max_attempts - 1 and any(msg in err_str for msg in RETRIABLE_ERRORS):
                        backoff = 0.5 * (2 ** attempt)  # 0.5s, 1.0s, 2.0s, 4.0s, 8.0s
                        logging.getLogger(__name__).info(
                            "Retry %d/%d for %s [%s] - %s (backoff %.1fs)",
                            attempt + 1, max_attempts, circuit_record.name, optimizer_name,
                            err_str[:80], backoff,
                        )
                        time.sleep(backoff)
                        continue
                    break

            return SingleStepResult(
                circuit_id=circuit_record.id or 0,
                optimizer_id=optimizer.id,
                circuit_name=circuit_record.name,
                optimizer_name=optimizer_name,
                input_metrics=input_metrics,
                output_metrics=input_metrics,
                duration_seconds=0.0,
                success=False,
                error_message=str(last_error),
            )
        finally:
            thread_db.close()

    def _failure_result(
        self,
        circuit_record: CircuitRecord,
        optimizer_name: str,
        error_message: str,
        duration_seconds: float = 0.0,
    ) -> SingleStepResult:
        """Create a failed result for process-level optimizer failures."""
        optimizer = self.db.get_optimizer_by_name(optimizer_name)
        optimizer_id = optimizer.id if optimizer and optimizer.id is not None else 0
        input_metrics = CircuitMetrics(0, 0, 0, 0)
        circuit = self._load_circuit(circuit_record)
        if circuit is not None:
            input_metrics = analyze_circuit(circuit)

        return SingleStepResult(
            circuit_id=circuit_record.id or 0,
            optimizer_id=optimizer_id,
            circuit_name=circuit_record.name,
            optimizer_name=optimizer_name,
            input_metrics=input_metrics,
            output_metrics=input_metrics,
            duration_seconds=duration_seconds,
            success=False,
            error_message=error_message,
        )

    def _run_single_step_process_with_timeout(
        self,
        circuit_record: CircuitRecord,
        optimizer_name: str,
        output_dir: Path,
    ) -> SingleStepResult:
        """Run one optimizer in a child process with a hard timeout."""
        timeout_seconds = self.config.optimizer_timeout_seconds
        if timeout_seconds <= 0:
            if self._abort_event.is_set():
                raise OptimizerRunAborted("Optimizer run aborted")
            return self._run_single_step_sync(circuit_record, optimizer_name, output_dir)

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_run_single_step_process_entry,
            args=(self.config, circuit_record, optimizer_name, output_dir, child_conn),
            daemon=False,
        )

        start_time = time.perf_counter()
        try:
            if self._abort_event.is_set():
                raise OptimizerRunAborted("Optimizer run aborted")

            process.start()
            self._register_process(process)
            child_conn.close()

            if self._abort_event.is_set():
                _terminate_process(process)
                raise OptimizerRunAborted("Optimizer run aborted")

            process.join(timeout=timeout_seconds)

            if self._abort_event.is_set():
                if process.is_alive():
                    _terminate_process(process)
                raise OptimizerRunAborted("Optimizer run aborted")

            if process.is_alive():
                _terminate_process(process)
                return self._failure_result(
                    circuit_record,
                    optimizer_name,
                    f"Optimizer timed out after {timeout_seconds} seconds",
                    duration_seconds=float(timeout_seconds),
                )

            duration = time.perf_counter() - start_time
            if parent_conn.poll():
                status, payload = parent_conn.recv()
                if status == "result":
                    return payload
                return self._failure_result(
                    circuit_record,
                    optimizer_name,
                    f"Optimizer process failed: {payload}",
                    duration_seconds=duration,
                )

            return self._failure_result(
                circuit_record,
                optimizer_name,
                f"Optimizer process exited without a result (exit code {process.exitcode})",
                duration_seconds=duration,
            )
        finally:
            parent_conn.close()
            self._unregister_process(process)
            if process.is_alive():
                _terminate_process(process)
    
    async def _run_single_step_async(
        self,
        circuit_record: CircuitRecord,
        optimizer_name: str,
    ) -> SingleStepResult:
        """Run a single optimization step asynchronously with semaphore control."""
        sem = self._get_semaphore(optimizer_name)
        
        async with sem:
            # Create output directory
            with tempfile.TemporaryDirectory(prefix=f"opt_{circuit_record.name}_") as tmp_dir:
                output_dir = Path(tmp_dir)
                
                # Run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._run_single_step_process_with_timeout,
                    circuit_record,
                    optimizer_name,
                    output_dir,
                )
                
                return result
    
    def _record_result(self, result: SingleStepResult) -> None:
        """Record a single-step result to the database."""
        self.db.insert_optimization_run(
            circuit_id=result.circuit_id,
            optimizer_id=result.optimizer_id,
            input_depth=result.input_metrics.depth,
            input_two_qubit_gates=result.input_metrics.two_qubit_gates,
            input_two_qubit_depth=result.input_metrics.two_qubit_depth,
            input_total_gates=result.input_metrics.total_gates,
            output_depth=result.output_metrics.depth,
            output_two_qubit_gates=result.output_metrics.two_qubit_gates,
            output_two_qubit_depth=result.output_metrics.two_qubit_depth,
            output_total_gates=result.output_metrics.total_gates,
            duration_seconds=result.duration_seconds,
            success=result.success,
            error_message=result.error_message,
            artifact_path=str(result.artifact_path) if result.artifact_path else None,
            metadata=result.metadata,
        )
    
    async def run_exhaustive_search(
        self,
        resume: bool = True,
    ) -> SingleStepReport:
        """Run exhaustive single-step search across all circuits and optimizers.
        
        Args:
            resume: Skip already-completed runs
            
        Returns:
            SingleStepReport with results summary
        """
        start_time = time.perf_counter()
        
        # Create semaphores in async context
        self._sem_fast = asyncio.Semaphore(self.config.max_concurrent_fast)
        self._sem_wisq_rules = asyncio.Semaphore(self.config.max_concurrent_wisq_rules)
        self._sem_wisq_bqskit = asyncio.Semaphore(self.config.max_concurrent_wisq_bqskit)
        
        # Get circuits
        circuits = self.db.list_circuits(max_qubits=self.config.max_qubits)
        if self.config.categories:
            circuits = [c for c in circuits if c.category in self.config.categories]
        
        # Get optimizer names
        optimizer_names = self.config.optimizers or list(OPTIMIZER_CONFIGS.keys())
        
        # Get optimizer IDs
        optimizers = [self.db.get_optimizer_by_name(name) for name in optimizer_names]
        optimizers = [o for o in optimizers if o is not None]
        
        if not circuits:
            return SingleStepReport(
                total_circuits=0,
                total_optimizers=len(optimizers),
                total_runs=0,
                completed_runs=0,
                skipped_runs=0,
                failed_runs=0,
                total_duration_seconds=0.0,
                best_by_optimizer={},
                failures=[],
                equivalence_counts={"verified": 0, "skipped": 0, "failed": 0},
            )
        
        # Build work queue and track per-optimizer totals
        work_items: list[tuple[CircuitRecord, str]] = []
        skipped_by_optimizer: dict[str, int] = {name: 0 for name in optimizer_names}
        optimizer_totals: dict[str, int] = {name: 0 for name in optimizer_names}
        
        for circuit in circuits:
            if circuit.id is None:
                continue
            for opt in optimizers:
                if opt.id is None:
                    continue
                
                optimizer_totals[opt.name] += 1
                
                # Check if already completed (skip unless optimizer is in rerun list)
                should_rerun = (
                    self.config.rerun_optimizers is not None
                    and opt.name in self.config.rerun_optimizers
                )
                if resume and not should_rerun and self.db.run_exists(circuit.id, opt.id):
                    skipped_by_optimizer[opt.name] += 1
                    if self.progress_tracker:
                        await self.progress_tracker.skip_task(opt.name)
                    continue
                
                work_items.append((circuit, opt.name))
        
        total_skipped = sum(skipped_by_optimizer.values())
        total_runs = len(work_items) + total_skipped
        completed_runs = 0
        failed_runs = 0
        failures: list[dict[str, Any]] = []
        best_by_optimizer: dict[str, dict[str, Any]] = {}
        equivalence_counts = {"verified": 0, "skipped": 0, "failed": 0}
        
        # Create all tasks upfront for concurrent execution
        # Wrap tasks to track start/complete with progress tracker
        async def run_with_progress(circuit: CircuitRecord, optimizer_name: str) -> SingleStepResult:
            if self.progress_tracker:
                await self.progress_tracker.start_task(optimizer_name)
            
            result = await self._run_single_step_async(circuit, optimizer_name)
            
            if self.progress_tracker:
                await self.progress_tracker.complete_task(
                    optimizer_name, result.success, duration=result.duration_seconds
                )
                # Log completion above progress bars
                if result.success:
                    improvement = result.improvement_percentage
                    if improvement > 0:
                        self.progress_tracker.log(
                            f"✓ {circuit.name} [{optimizer_name}] "
                            f"-{improvement:.1f}% 2Q gates",
                            style="green"
                        )
                    elif improvement < 0:
                        self.progress_tracker.log(
                            f"✓ {circuit.name} [{optimizer_name}] "
                            f"worse by {-improvement:.1f}% 2Q gates",
                            style="red"
                        )
                    else:
                        self.progress_tracker.log(
                            f"✓ {circuit.name} [{optimizer_name}] "
                            f"no improvement",
                            style="dim"
                        )
                else:
                    error_msg = (result.error_message or "Unknown error")[:50]
                    self.progress_tracker.log(
                        f"✗ {circuit.name} [{optimizer_name}] {error_msg}",
                        style="red"
                    )
            
            return result
        
        tasks = [
            asyncio.create_task(run_with_progress(circuit, optimizer_name))
            for circuit, optimizer_name in work_items
        ]

        try:
            # Process results as they complete
            for task in asyncio.as_completed(tasks):
                result = await task
                
                # Update progress (legacy callback)
                if self.progress_callback:
                    elapsed = time.perf_counter() - start_time
                    self.progress_callback(SingleStepProgress(
                        total_runs=total_runs,
                        completed_runs=completed_runs + total_skipped,
                        skipped_runs=total_skipped,
                        failed_runs=failed_runs,
                        current_circuit=result.circuit_name,
                        current_optimizer=result.optimizer_name,
                        elapsed_seconds=elapsed,
                    ))

                # Record result
                self._record_result(result)

                status = str(result.metadata.get("equivalence_status", "not_checked"))
                if status in equivalence_counts:
                    equivalence_counts[status] += 1

                if result.success:
                    completed_runs += 1

                    # Track best by optimizer
                    improvement = result.improvement_percentage
                    optimizer_name = result.optimizer_name
                    if optimizer_name not in best_by_optimizer:
                        best_by_optimizer[optimizer_name] = {
                            "circuit": result.circuit_name,
                            "improvement": improvement,
                        }
                    elif improvement > best_by_optimizer[optimizer_name]["improvement"]:
                        best_by_optimizer[optimizer_name] = {
                            "circuit": result.circuit_name,
                            "improvement": improvement,
                        }
                else:
                    failed_runs += 1
                    failures.append({
                        "circuit": result.circuit_name,
                        "optimizer": result.optimizer_name,
                        "error": result.error_message,
                    })
        except (asyncio.CancelledError, KeyboardInterrupt, OptimizerRunAborted):
            for task in tasks:
                task.cancel()
            self.abort()
            if tasks:
                await asyncio.wait(tasks, timeout=2)
            raise
        
        total_duration = time.perf_counter() - start_time
        
        return SingleStepReport(
            total_circuits=len(circuits),
            total_optimizers=len(optimizers),
            total_runs=total_runs,
            completed_runs=completed_runs,
            skipped_runs=total_skipped,
            failed_runs=failed_runs,
            total_duration_seconds=total_duration,
            best_by_optimizer=best_by_optimizer,
            failures=failures,
            equivalence_counts=equivalence_counts,
        )
    
    def run_sync(self, resume: bool = True) -> SingleStepReport:
        """Synchronous wrapper for run_exhaustive_search."""
        try:
            return asyncio.run(self.run_exhaustive_search(resume=resume))
        except KeyboardInterrupt:
            self.abort()
            raise
        except OptimizerRunAborted as exc:
            self.abort()
            raise KeyboardInterrupt from exc
    
    def close(self) -> None:
        """Close the database connection and executor."""
        if self._abort_event.is_set():
            self._shutdown_executor(wait=False, cancel_futures=True)
        else:
            self._shutdown_executor(wait=True, cancel_futures=False)
        self.db.close()
    
    def __enter__(self) -> "AsyncSingleStepRunner":
        return self
    
    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()


def run_single_step_grid_search(
    database_path: Path | str,
    categories: Sequence[str] | None = None,
    optimizers: Sequence[str] | None = None,
    rerun_optimizers: Sequence[str] | None = None,
    max_qubits: int = 20,
    wisq_bqskit_timeout: int = 300,
    optimizer_timeout_seconds: int = 900,
    save_artifacts: bool = True,
    artifact_dir: Path | str | None = None,
    progress_callback: Callable[[SingleStepProgress], None] | None = None,
    progress_tracker: OptimizersProgressTracker | None = None,
    resume: bool = True,
) -> SingleStepReport:
    """Run a single-step grid search with the given configuration.
    
    This is a convenience function for running single-step grid search.
    
    Args:
        database_path: Path to trajectory database
        categories: Categories to include (None = all)
        optimizers: Optimizers to use (None = all)
        rerun_optimizers: Optimizers to force rerun even if results exist (None = none)
        max_qubits: Maximum qubit count
        wisq_bqskit_timeout: Timeout for WISQ+BQSKit in seconds
        optimizer_timeout_seconds: Hard timeout for each optimizer run in seconds
        save_artifacts: Whether to save output circuits
        artifact_dir: Directory for artifacts (None = default)
        progress_callback: Progress callback (deprecated, use progress_tracker)
        progress_tracker: Progress tracker for multi-bar display
        resume: Skip already-completed runs
        
    Returns:
        SingleStepReport
    """
    config = SingleStepConfig(
        database_path=Path(database_path),
        max_qubits=max_qubits,
        categories=list(categories) if categories else None,
        optimizers=list(optimizers) if optimizers else None,
        rerun_optimizers=list(rerun_optimizers) if rerun_optimizers else None,
        wisq_bqskit_timeout=wisq_bqskit_timeout,
        optimizer_timeout_seconds=optimizer_timeout_seconds,
        save_artifacts=save_artifacts,
        artifact_dir=Path(artifact_dir) if artifact_dir else Path("data/artifacts"),
    )
    
    with AsyncSingleStepRunner(
        config, 
        progress_callback=progress_callback,
        progress_tracker=progress_tracker,
    ) as runner:
        return runner.run_sync(resume=resume)
