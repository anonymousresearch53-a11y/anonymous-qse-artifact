from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import yaml
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

try:
    from .chain_executor import create_chain_from_config, execute_chain
    from .circuit_comparison import compare_against_baseline
    from .transpilers import (
        QiskitAIRunnerConfig,
        QiskitStandardConfig,
        TKETConfig,
        TranspiledCircuit,
        VOQCConfig,
        WisqConfig,
        run_tket,
        run_voqc,
        run_wisq_opt,
        transpile_with_qiskit_ai,
        transpile_with_qiskit_standard,
    )
except ImportError:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from benchmarks.ai_transpile.chain_executor import (
        create_chain_from_config,
        execute_chain,
    )
    from benchmarks.ai_transpile.circuit_comparison import compare_against_baseline
    from benchmarks.ai_transpile.transpilers import (
        QiskitAIRunnerConfig,
        QiskitStandardConfig,
        TKETConfig,
        TranspiledCircuit,
        VOQCConfig,
        WisqConfig,
        run_tket,
        run_voqc,
        run_wisq_opt,
        transpile_with_qiskit_ai,
        transpile_with_qiskit_standard,
    )


# Global lock to ensure only one resynthesis server runs at a time
# This prevents port conflicts and BQSKit worker spawning race conditions
RESYNTHESIS_SERVER_LOCK = threading.Lock()

# Type variable for generic retry function
T = TypeVar("T")

# Default number of BQSKit workers for WISQ resynthesis
# 12 workers provides good parallelism while leaving headroom for system processes
DEFAULT_BQSKIT_NUM_WORKERS = 4



class RunnerTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RunnerTimeoutError("Runner timed out")


def run_with_timeout(func, timeout_seconds: int):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        return func()
    finally:
        signal.alarm(0)


def configure_bqskit_workers(num_workers: int | None = None, worker_fraction: float | None = None) -> int:
    """
    Configure the number of BQSKit workers to prevent system resource exhaustion.
    
    This function sets the BQSKIT_NUM_WORKERS environment variable to limit the number
    of parallel worker processes spawned by BQSKit during resynthesis. This prevents
    the system from becoming unresponsive during heavy optimization workloads.
    
    Args:
        num_workers: Explicit number of workers to use. If None, uses default (12 workers)
                     unless worker_fraction is specified or BQSKIT_NUM_WORKERS env var is set.
        worker_fraction: Fraction of available CPU cores to use (0.0-1.0).
                        If specified, overrides the default fixed worker count.
    
    Returns:
        The configured number of workers.
    
    Configuration priority (highest to lowest):
        1. Explicit num_workers parameter
        2. Explicit worker_fraction parameter
        3. Existing BQSKIT_NUM_WORKERS environment variable
        4. DEFAULT_BQSKIT_NUM_WORKERS (12)
    """
    # If already set in environment and no explicit override, use existing value
    # Only use existing env var if neither num_workers nor worker_fraction was explicitly provided
    if num_workers is None and worker_fraction is None and "BQSKIT_NUM_WORKERS" in os.environ:
        existing = int(os.environ["BQSKIT_NUM_WORKERS"])
        print(f"      Using existing BQSKIT_NUM_WORKERS={existing}", flush=True)
        return existing
    
    if num_workers is not None:
        workers = num_workers
    elif worker_fraction is not None:
        # Calculate based on fraction of available cores
        total_cores = multiprocessing.cpu_count()
        workers = max(1, int(total_cores * worker_fraction))
    else:
        # Use the default fixed worker count
        workers = DEFAULT_BQSKIT_NUM_WORKERS
    
    os.environ["BQSKIT_NUM_WORKERS"] = str(workers)
    print(f"      Configured BQSKIT_NUM_WORKERS={workers} (of {multiprocessing.cpu_count()} cores)", flush=True)
    return workers


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available (not in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False


def wait_for_port_cleanup(port: int, max_wait: float = 5.0, check_interval: float = 0.5) -> bool:
    """
    Wait for a port to become available.
    
    Args:
        port: Port number to check
        max_wait: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        True if port became available, False if timeout
    """
    elapsed = 0.0
    while elapsed < max_wait:
        if is_port_available(port):
            return True
        time.sleep(check_interval)
        elapsed += check_interval
    return False


def retry_on_failure(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    retry_exceptions: tuple = (KeyError, OSError, RuntimeError),
) -> Callable[..., T]:
    """
    Decorator/wrapper to retry a function on transient failures with exponential backoff.
    
    Args:
        func: Function to wrap
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each failure
        retry_exceptions: Tuple of exception types to retry on
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs) -> T:
        delay = initial_delay
        last_exception: BaseException | None = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e
                if attempt < max_attempts:
                    print(
                        f"      ⚠ Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(f"      Retrying in {delay:.1f}s...", flush=True)
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    print(
                        f"      ✗ All {max_attempts} attempts failed. Last error: {type(e).__name__}: {e}",
                        flush=True,
                    )
        
        # If we get here, all attempts failed (last_exception is guaranteed to be set)
        assert last_exception is not None
        raise last_exception
    
    return wrapper


@dataclass(frozen=True)
class CircuitConfig:
    name: str
    path: Path
    gate_set: str
    tags: Sequence[str]


@dataclass(frozen=True)
class RunnerSpec:
    name: str
    type: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    metadata: Mapping[str, Any]
    circuits: Sequence[CircuitConfig]
    runners: Sequence[RunnerSpec]
    metrics: Sequence[str]

    @property
    def output_dir(self) -> Path:
        default_dir = self.metadata.get("default_output_dir", "reports/circuit_benchmark")
        return Path(default_dir)

    @property
    def job_info(self) -> str:
        return str(self.metadata.get("job_info", "bench"))


def _find_project_root(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    while True:
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _discover_project_root(config_path: Path) -> Path:
    project_root = _find_project_root(config_path)
    if project_root:
        return project_root

    module_root = _find_project_root(Path(__file__).resolve())
    if module_root:
        return module_root

    parents = config_path.parents
    if len(parents) >= 3:
        return parents[2]
    return config_path.parent


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    project_root = _discover_project_root(config_path)

    payload = yaml.safe_load(config_path.read_text())
    circuits = [
        CircuitConfig(
            name=entry["name"],
            path=(project_root / entry["path"]).resolve(),
            gate_set=entry.get("gate_set", "IBMN"),
            tags=tuple(entry.get("tags", ())),
        )
        for entry in payload["circuits"]
    ]
    runners = [
        RunnerSpec(
            name=entry["name"],
            type=entry["type"],
            options={key: value for key, value in entry.items() if key not in {"name", "type"}},
        )
        for entry in payload["runners"]
    ]
    return ExperimentConfig(
        metadata=payload.get("metadata", {}),
        circuits=circuits,
        runners=runners,
        metrics=tuple(payload.get("metrics", ())),
    )


from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

def _load_circuit(path: Path) -> QuantumCircuit:
    return qasm2.loads(
        Path(path).read_text(),
        custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
    )


def _result_record(
    circuit_name: str,
    runner_name: str,
    transpiled: TranspiledCircuit,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "circuit": circuit_name,
        "runner": runner_name,
        "optimizer": transpiled.optimizer,
        "label": transpiled.label,
        "metrics": asdict(transpiled.metrics),
        "metadata": dict(transpiled.metadata),
    }
    if transpiled.artifact_path:
        record["artifact_path"] = str(transpiled.artifact_path)
    return record


def _run_qiskit_ai_runner(
    circuit: QuantumCircuit,
    spec: RunnerSpec,
) -> list[TranspiledCircuit]:
    options = spec.options
    config = QiskitAIRunnerConfig(
        optimization_levels=tuple(options.get("optimization_levels", (1, 2, 3))),
        iterations_per_level=int(options.get("iterations_per_level", 3)),
        layout_mode=str(options.get("layout_mode", "optimize")),
    )
    return transpile_with_qiskit_ai(circuit.copy(), config=config)


def _run_qiskit_standard_runner(
    circuit: QuantumCircuit,
    spec: RunnerSpec,
) -> list[TranspiledCircuit]:
    """Run standard Qiskit transpiler."""
    options = spec.options
    config = QiskitStandardConfig(
        optimization_levels=tuple(options.get("optimization_levels", (1, 2, 3))),
    )
    return transpile_with_qiskit_standard(circuit.copy(), config=config)


def _run_wisq_runner_impl(
    circuit_path: Path,
    circuit_gate_set: str,
    experiment_job_info: str,
    spec: RunnerSpec,
    output_root: Path,
) -> list[TranspiledCircuit]:
    """Internal implementation of WISQ runner (wrapped by _run_wisq_runner for retry logic)."""
    options = spec.options
    job_info = f"{experiment_job_info}_{spec.name}"
    runner_output = Path(options.get("output_dir", output_root / "wisq" / spec.name))
    print(
        f"      WISQ config: gate_set={options.get('target_gateset', circuit_gate_set)}, "
        f"objective={options.get('optimization_objective', 'TWO_Q')}, "
        f"timeout={options.get('opt_timeout', 600)}s, "
        f"approx_epsilon={options.get('approx_epsilon', 1e-10)}",
        flush=True,
    )
    advanced_args = options.get("advanced_args")
    wisq_config = WisqConfig(
        target_gateset=str(options.get("target_gateset", circuit_gate_set)),
        objective=str(options.get("optimization_objective", "TWO_Q")),
        timeout_seconds=int(options.get("opt_timeout", 600)),
        approximation_epsilon=float(options.get("approx_epsilon", 1e-10)),
        output_dir=runner_output,
        job_info=job_info,
        advanced_args=None if advanced_args is None else dict(advanced_args),
    )
    print("      Starting WISQ optimization...", flush=True)
    result = run_wisq_opt(circuit_path, config=wisq_config)
    print("      WISQ optimization completed", flush=True)
    return [result]


def _run_wisq_runner(
    circuit_path: Path,
    circuit_gate_set: str,
    experiment_job_info: str,
    spec: RunnerSpec,
    output_root: Path,
) -> list[TranspiledCircuit]:
    """
    Run WISQ optimizer with serialization lock and retry logic.
    
    This ensures only one resynthesis server runs at a time, preventing
    port conflicts and BQSKit worker spawning race conditions.
    """
    # Acquire the global lock to serialize resynthesis server usage
    lock_acquired = RESYNTHESIS_SERVER_LOCK.acquire(timeout=600)  # 10 minute timeout
    if not lock_acquired:
        raise RuntimeError(
            "Failed to acquire resynthesis server lock within 10 minutes. "
            "Another WISQ job may be stuck."
        )
    
    try:
        # Configure BQSKit worker count to prevent resource exhaustion
        # This limits parallel workers to leave headroom for SSH and system processes
        options = spec.options
        num_workers = options.get("bqskit_num_workers")
        worker_fraction = options.get("bqskit_worker_fraction")
        configure_bqskit_workers(
            num_workers=int(num_workers) if num_workers is not None else None,
            worker_fraction=float(worker_fraction) if worker_fraction is not None else None,
        )
        
        # Ensure port 8080 is available before starting
        # (in case previous run didn't clean up properly)
        if not is_port_available(8080):
            print(
                "      ⚠ Port 8080 in use, waiting for cleanup...",
                flush=True,
            )
            if not wait_for_port_cleanup(8080, max_wait=10.0):
                print(
                    "      ⚠ Port 8080 still in use after 10s, proceeding anyway...",
                    flush=True,
                )
        
        # Wrap the implementation with retry logic
        retry_wrapper = retry_on_failure(
            lambda: _run_wisq_runner_impl(
                circuit_path, circuit_gate_set, experiment_job_info, spec, output_root
            ),
            max_attempts=3,
            initial_delay=5.0,
            backoff_factor=2.0,
            retry_exceptions=(KeyError, OSError, RuntimeError),
        )
        
        result = retry_wrapper()
        
        # Wait briefly after completion to ensure server cleanup
        time.sleep(2.0)
        
        return result
    finally:
        # Always release the lock, even if an exception occurred
        RESYNTHESIS_SERVER_LOCK.release()
        print("      Released resynthesis server lock", flush=True)


def _run_tket_runner(
    circuit_path: Path,
    circuit_gate_set: str,
    experiment_job_info: str,
    spec: RunnerSpec,
    output_root: Path,
) -> list[TranspiledCircuit]:
    """Run TKET (pytket) optimizer."""
    options = spec.options
    job_info = f"{experiment_job_info}_{spec.name}"
    runner_output = Path(options.get("output_dir", output_root / "tket" / spec.name))
    gate_set = str(options.get("gate_set", circuit_gate_set))
    print(
        f"      TKET config: gate_set={gate_set}",
        flush=True,
    )
    tket_config = TKETConfig(
        gate_set=gate_set,
        output_dir=runner_output,
        job_info=job_info,
    )
    print("      Starting TKET optimization...", flush=True)
    result = run_tket(circuit_path, config=tket_config)
    print("      TKET optimization completed", flush=True)
    return [result]


def _run_voqc_runner(
    circuit_path: Path,
    circuit_gate_set: str,
    experiment_job_info: str,
    spec: RunnerSpec,
    output_root: Path,
) -> list[TranspiledCircuit]:
    """Run VOQC (Verified Optimizer for Quantum Circuits)."""
    options = spec.options
    job_info = f"{experiment_job_info}_{spec.name}"
    runner_output = Path(options.get("output_dir", output_root / "voqc" / spec.name))
    optimization_method = str(options.get("optimization_method", "nam"))
    print(
        f"      VOQC config: optimization_method={optimization_method}",
        flush=True,
    )
    voqc_config = VOQCConfig(
        optimization_method=optimization_method,
        output_dir=runner_output,
        job_info=job_info,
    )
    print("      Starting VOQC optimization...", flush=True)
    result = run_voqc(circuit_path, config=voqc_config)
    print("      VOQC optimization completed", flush=True)
    return [result]


def _run_chain_runner(
    circuit: QuantumCircuit,
    circuit_path: Path,
    experiment_job_info: str,
    spec: RunnerSpec,
    output_root: Path,
    circuit_name: str,
) -> list[TranspiledCircuit]:
    """Run a chain of optimizers sequentially.

    The chain runner executes multiple optimization steps in sequence,
    feeding the output of each step into the next.

    Args:
        circuit: Input quantum circuit
        circuit_path: Path to the circuit QASM file
        experiment_job_info: Job info prefix for naming
        spec: Runner specification with 'steps' list in options
        output_root: Root directory for outputs

    Returns:
        List containing a single TranspiledCircuit representing the final result
    """
    options = spec.options
    steps_config = options.get("steps", [])

    if not steps_config:
        raise ValueError(f"Chain runner '{spec.name}' must have 'steps' defined in options")

    # Create chain steps from configuration
    steps = create_chain_from_config(steps_config)

    chain_name = f"{experiment_job_info}_{circuit_name}_{spec.name}"

    base_output_dir = Path(
        options.get(
            "output_dir",
            output_root / "chains" / spec.name,
        )
    )

    chain_output_dir = base_output_dir / circuit_name
    print(f"      Chain config: {len(steps)} steps", flush=True)
    for i, step in enumerate(steps):
        print(f"        Step {i + 1}: {step.step_name} ({step.runner_type})", flush=True)

    print("      Starting chain execution...", flush=True)
    chain_result = execute_chain(
        circuit=circuit,
        steps=steps,
        chain_name=chain_name,
        output_dir=chain_output_dir,
        save_intermediates=options.get("save_intermediates", True),
    )
    print(f"      Chain completed in {chain_result.total_duration_seconds:.2f}s", flush=True)

    # Convert chain result to TranspiledCircuit format
    # Include chain-specific metadata
    chain_metadata: dict[str, Any] = {
        "chain_name": chain_result.chain_name,
        "num_steps": len(steps),
        "steps": [
            {
                "name": s.step_name,
                "runner_type": s.runner_type,
            }
            for s in steps
        ],
        "step_durations": [sr.duration_seconds for sr in chain_result.step_results],
        "step_improvements": [],
        "total_duration_seconds": chain_result.total_duration_seconds,
    }

    # Calculate per-step improvements
    for sr in chain_result.step_results:
        if sr.input_metrics.two_qubit_gates > 0:
            improvement = (
                100.0
                * (sr.input_metrics.two_qubit_gates - sr.output_metrics.two_qubit_gates)
                / sr.input_metrics.two_qubit_gates
            )
        else:
            improvement = 0.0
        chain_metadata["step_improvements"].append(improvement)

    # Create a label that describes the chain
    step_names = "_then_".join(s.step_name for s in steps)
    label = f"chain_{step_names}"

    # Find artifact path (last step's output)
    artifact_path = None
    if chain_result.step_results:
        artifact_path = chain_result.step_results[-1].artifact_path

    result = TranspiledCircuit(
        optimizer="chain",
        label=label,
        circuit=chain_result.final_circuit,
        metrics=chain_result.final_metrics,
        artifact_path=artifact_path,
        metadata=chain_metadata,
    )

    return [result]


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path | None = None,
    skip_runners: Iterable[str] | None = None,
    compare_against_baseline_runner: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    output_root = Path(output_dir) if output_dir else config.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    skip = set(skip_runners or ())

    # Store baseline circuits for comparison
    baseline_circuits: dict[str, QuantumCircuit] = {}

    for circuit_cfg in config.circuits:
        print(f"Processing circuit: {circuit_cfg.name}", flush=True)
        circuit = _load_circuit(circuit_cfg.path)
        circuit_results: list[dict[str, Any]] = []

        for runner in config.runners:
            if runner.name in skip:
                print(f"  Skipping runner: {runner.name}", flush=True)
                failures.append(
                    {
                        "circuit": circuit_cfg.name,
                        "runner": runner.name,
                        "error": "skipped",
                    }
                )
                continue
            print(f"  Running runner: {runner.name} (type: {runner.type})", flush=True)
            try:
                timeout_seconds = int(config.metadata.get("runner_timeout_seconds", 300))
                if runner.type == "qiskit_ai":
                    print("    Executing Qiskit AI transpiler...", flush=True)

                    variants = run_with_timeout(
                        lambda: _run_qiskit_ai_runner(
                            circuit,
                            runner,
                        ),
                        timeout_seconds,
                    )

                    print(f"    Completed Qiskit AI: {len(variants)} variants", flush=True)
                elif runner.type == "qiskit_standard":
                    print("    Executing standard Qiskit transpiler...", flush=True)
                    variants = _run_qiskit_standard_runner(circuit, runner)
                    print(f"    Completed standard Qiskit: {len(variants)} variants", flush=True)
                elif runner.type == "wisq":
                    print("    Executing WISQ (GUOQ via wisq)...", flush=True)
                    variants = _run_wisq_runner(
                        circuit_cfg.path,
                        circuit_cfg.gate_set,
                        config.job_info,
                        runner,
                        output_root,
                    )
                    print(f"    Completed WISQ: {len(variants)} variants", flush=True)
                elif runner.type == "tket":
                    print("    Executing TKET (pytket) optimizer...", flush=True)
                    variants = _run_tket_runner(
                        circuit_cfg.path,
                        circuit_cfg.gate_set,
                        config.job_info,
                        runner,
                        output_root,
                    )
                    print(f"    Completed TKET: {len(variants)} variants", flush=True)
                elif runner.type == "voqc":
                    print("    Executing VOQC (verified optimizer)...", flush=True)
                    variants = _run_voqc_runner(
                        circuit_cfg.path,
                        circuit_cfg.gate_set,
                        config.job_info,
                        runner,
                        output_root,
                    )
                    print(f"    Completed VOQC: {len(variants)} variants", flush=True)
                elif runner.type == "chain":
                    print("    Executing chain optimizer...", flush=True)

                    variants = run_with_timeout(
                        lambda: _run_chain_runner(
                            circuit,
                            circuit_cfg.path,
                            config.job_info,
                            runner,
                            output_root,
                            circuit_cfg.name,
                        ),
                        timeout_seconds,
                    )

                    print(f"    Completed chain: {len(variants)} variants", flush=True)
                else:
                    raise ValueError(f"Unsupported runner type '{runner.type}'")

                for variant in variants:
                    # Always save Qiskit AI circuits to files so they can be compared later
                    variant_to_save = variant
                    if variant.artifact_path is None:
                        # Save circuits to files for later comparison
                        circuit_output_dir = output_root / "circuits" / circuit_cfg.name / runner.name
                        circuit_output_dir.mkdir(parents=True, exist_ok=True)
                        circuit_file = circuit_output_dir / f"{variant.label}.qasm"
                        # Decompose swap gates to ensure compatibility with QASM parsers
                        circuit_to_save = variant.circuit.copy()
                        circuit_to_save = circuit_to_save.decompose(gates_to_decompose=["swap"])
                        circuit_file.write_text(qasm2.dumps(circuit_to_save))
                        # Create new variant with artifact_path
                        from benchmarks.ai_transpile.transpilers import TranspiledCircuit

                        variant_to_save = TranspiledCircuit(
                            optimizer=variant.optimizer,
                            label=variant.label,
                            circuit=variant.circuit,
                            metrics=variant.metrics,
                            artifact_path=circuit_file,
                            metadata=variant.metadata,
                        )

                    record = _result_record(circuit_cfg.name, runner.name, variant_to_save)
                    records.append(record)
                    circuit_results.append(record)

                    # Store baseline circuit if this is the baseline runner
                    if (
                        compare_against_baseline_runner
                        and runner.name == compare_against_baseline_runner
                        and circuit_cfg.name not in baseline_circuits
                    ):
                        baseline_circuits[circuit_cfg.name] = variant.circuit

            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR in {runner.name}: {exc}", flush=True)
                failures.append(
                    {
                        "circuit": circuit_cfg.name,
                        "runner": runner.name,
                        "error": str(exc),
                    }
                )

        # Perform comparisons if baseline runner is specified
        if compare_against_baseline_runner and circuit_cfg.name in baseline_circuits:
            baseline = baseline_circuits[circuit_cfg.name]
            for result in circuit_results:
                if result["runner"] != compare_against_baseline_runner:
                    # Load the optimized circuit
                    if "artifact_path" in result:
                        try:
                            optimized = qasm2.loads(Path(result["artifact_path"]).read_text())
                            comparison = compare_against_baseline(baseline, optimized)
                            comparisons.append(
                                {
                                    "circuit": circuit_cfg.name,
                                    "baseline_runner": compare_against_baseline_runner,
                                    "optimized_runner": result["runner"],
                                    "optimized_label": result["label"],
                                    **comparison,
                                }
                            )
                        except Exception as exc:  # noqa: BLE001
                            comparisons.append(
                                {
                                    "circuit": circuit_cfg.name,
                                    "baseline_runner": compare_against_baseline_runner,
                                    "optimized_runner": result["runner"],
                                    "optimized_label": result["label"],
                                    "equivalent": False,
                                    "error": f"Failed to load circuit: {exc}",
                                }
                            )

    report = {
        "metadata": {
            "job_info": config.job_info,
            "output_dir": str(output_root),
            "num_results": len(records),
        },
        "results": records,
        "failures": failures,
    }

    if comparisons:
        report["comparisons"] = comparisons

    # Save latest results
    report_path = output_root / "latest_results.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    # Also save timestamped backup to prevent overwrites
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_root / f"results_{timestamp}.json"
    backup_path.write_text(json.dumps(report, indent=2))
    
    report["report_path"] = str(report_path)
    report["backup_path"] = str(backup_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Circuit Benchmark experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/ai_transpile/circuit_benchmark.yaml"),
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where aggregated results should be saved.",
    )
    parser.add_argument(
        "--skip-runner",
        action="append",
        default=[],
        help="Runner names to skip (can pass multiple times).",
    )
    parser.add_argument(
        "--compare-against",
        type=str,
        default=None,
        help="Runner name to use as baseline for equivalence comparisons.",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)
    report = run_experiment(
        experiment_config,
        output_dir=args.output,
        skip_runners=args.skip_runner,
        compare_against_baseline_runner=args.compare_against,
    )
    print(f"Wrote {report['metadata']['num_results']} results to {report['report_path']}")
    if report["failures"]:
        print("Failures detected:")
        for failure in report["failures"]:
            print(f" - {failure['circuit']} / {failure['runner']}: {failure['error']}")


if __name__ == "__main__":
    main()
