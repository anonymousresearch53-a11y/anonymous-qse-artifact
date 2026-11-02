from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGates, SabreLayout, SabreSwap

try:  # pragma: no cover - optional dependency
    from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
    from qiskit_ibm_transpiler.ai.routing import AIRouting
    from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
    _IBM_TRANSPILER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CollectLinearFunctions = None  # type: ignore
    AIRouting = None  # type: ignore
    AILinearFunctionSynthesis = None  # type: ignore
    _IBM_TRANSPILER_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from wisq import optimize as wisq_optimize
except ImportError:  # pragma: no cover - optional dependency
    wisq_optimize = None  # type: ignore

# Check if TKET sub-environment is available (pytket requires networkx>=2.8.8,
# which conflicts with qiskit-ibm-ai-local-transpiler's networkx==2.8.5)
try:  # pragma: no cover - optional dependency
    from benchmarks.tket_runner import verify_tket_environment
    verify_tket_environment()
    PYTKET_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PYTKET_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from pyvoqc.qiskit.voqc_pass import voqc_pass_manager  # type: ignore[unresolved-import]

    PYVOQC_AVAILABLE = True
    PYVOQC_ERROR = None
except ImportError as e:  # pragma: no cover - optional dependency
    PYVOQC_AVAILABLE = False
    PYVOQC_ERROR = str(e)
    voqc_pass_manager = None  # type: ignore

BENCHMARK_ROOT = Path(__file__).resolve().parent
DEFAULT_WISQ_OUTPUT = Path(__file__).resolve().parents[2] / "reports" / "wisq_runs"


@dataclass(frozen=True)
class CircuitMetrics:
    depth: int
    two_qubit_gates: int
    two_qubit_depth: int
    total_gates: int


@dataclass(frozen=True)
class TranspiledCircuit:
    optimizer: str
    label: str
    circuit: QuantumCircuit
    metrics: CircuitMetrics
    artifact_path: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCircuit:
    name: str
    description: str
    tags: Sequence[str]
    num_qubits: int
    metrics: CircuitMetrics
    qasm_path: Path

    def load(self) -> QuantumCircuit:
        return qasm2.loads(self.qasm_path.read_text())


def load_benchmark_circuits(root: Path | None = None) -> dict[str, BenchmarkCircuit]:
    base_path = root or BENCHMARK_ROOT
    metadata = json.loads((base_path / "metadata.json").read_text(encoding="utf-8"))
    circuits: dict[str, BenchmarkCircuit] = {}
    for entry in metadata["circuits"]:
        metrics_dict = entry["metrics"]
        metrics = CircuitMetrics(
            depth=metrics_dict["depth"],
            two_qubit_gates=metrics_dict["two_qubit_gates"],
            two_qubit_depth=metrics_dict["two_qubit_depth"],
            total_gates=metrics_dict["total_gates"],
        )
        circuits[entry["name"]] = BenchmarkCircuit(
            name=entry["name"],
            description=entry["description"],
            tags=tuple(entry["tags"]),
            num_qubits=entry["num_qubits"],
            metrics=metrics,
            qasm_path=base_path / entry["file"],
        )
    return circuits


def get_benchmark_circuit(name: str, root: Path | None = None) -> BenchmarkCircuit:
    circuits = load_benchmark_circuits(root=root)
    if name not in circuits:
        raise KeyError(f"Unknown benchmark circuit '{name}'. Known circuits: {', '.join(sorted(circuits))}")
    return circuits[name]


def _count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if len(instruction.qubits) >= 2)


def _two_qubit_depth(circuit: QuantumCircuit) -> int:
    swap_free = circuit.decompose(gates_to_decompose=["swap"])
    return swap_free.depth(lambda instruction: len(instruction.qubits) >= 2)


def analyze_circuit(circuit: QuantumCircuit) -> CircuitMetrics:
    return CircuitMetrics(
        depth=circuit.depth(),
        two_qubit_gates=_count_two_qubit_gates(circuit),
        two_qubit_depth=_two_qubit_depth(circuit),
        total_gates=circuit.size(),
    )


def _ring_coupling_map(num_qubits: int) -> CouplingMap:
    edges = [(idx, (idx + 1) % num_qubits) for idx in range(num_qubits)]
    edges += [((idx + 1) % num_qubits, idx) for idx in range(num_qubits)]
    return CouplingMap(edges)


@dataclass(frozen=True)
class QiskitAIRunnerConfig:
    optimization_levels: Sequence[int] = (1, 2, 3)
    iterations_per_level: int = 3
    coupling_map: CouplingMap | None = None
    layout_mode: str = "optimize"


def transpile_with_qiskit_ai(
    circuit: QuantumCircuit,
    config: QiskitAIRunnerConfig | None = None,
) -> list[TranspiledCircuit]:
    cfg = config or QiskitAIRunnerConfig()
    coupling_map = cfg.coupling_map or _ring_coupling_map(circuit.num_qubits)

    collect_lfs = CollectLinearFunctions(
        do_commutative_analysis=True,
        min_block_size=0,
        max_block_size=2,
        collect_from_back=False,
    )
    ai_lf_synth = AILinearFunctionSynthesis(
        coupling_map=list(coupling_map.get_edges()),
        replace_only_if_better=True,
    )

    results: list[TranspiledCircuit] = []
    sabre_pm = PassManager([SabreLayout(coupling_map=coupling_map), SabreSwap(coupling_map=coupling_map)])
    routed_baseline = sabre_pm.run(circuit)
    results.append(
        TranspiledCircuit(
            optimizer="qiskit_ai",
            label="sabre_routed",
            circuit=routed_baseline,
            metrics=analyze_circuit(routed_baseline),
            metadata={"variant": "sabre", "optimization_level": 0, "iteration": 0},
        )
    )

    for level in cfg.optimization_levels:
        for iteration in range(cfg.iterations_per_level):
            pass_manager = PassManager(
                [
                    SabreLayout(coupling_map=coupling_map),
                    AIRouting(coupling_map=coupling_map, optimization_level=level, layout_mode=cfg.layout_mode),
                    collect_lfs,
                    ai_lf_synth,
                    Optimize1qGates(),
                ]
            )
            optimized = pass_manager.run(circuit)
            results.append(
                TranspiledCircuit(
                    optimizer="qiskit_ai",
                    label=f"ai_level_{level}_iter_{iteration + 1}",
                    circuit=optimized,
                    metrics=analyze_circuit(optimized),
                    metadata={"variant": "ai_transpiler", "optimization_level": level, "iteration": iteration + 1},
                )
            )

    return results


@dataclass(frozen=True)
class QiskitStandardConfig:
    optimization_levels: Sequence[int] = (1, 2, 3)
    coupling_map: CouplingMap | None = None


def transpile_with_qiskit_standard(
    circuit: QuantumCircuit,
    config: QiskitStandardConfig | None = None,
) -> list[TranspiledCircuit]:
    """Run standard Qiskit transpiler at different optimization levels."""
    cfg = config or QiskitStandardConfig()
    coupling_map = cfg.coupling_map or _ring_coupling_map(circuit.num_qubits)

    # Define IBMN basis gates
    basis_gates = ["id", "rz", "sx", "x", "cx"]

    results: list[TranspiledCircuit] = []
    for level in cfg.optimization_levels:
        start_time = time.perf_counter()
        optimized = transpile(
            circuit,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=level,
        )
        duration = time.perf_counter() - start_time

        results.append(
            TranspiledCircuit(
                optimizer="qiskit_standard",
                label=f"qiskit_opt_level_{level}",
                circuit=optimized,
                metrics=analyze_circuit(optimized),
                metadata={
                    "variant": "standard_transpiler",
                    "optimization_level": level,
                    "duration_seconds": duration,
                },
            )
        )

    return results


@dataclass(frozen=True)
class WisqConfig:
    target_gateset: str = "IBMN"
    objective: str = "TWO_Q"
    timeout_seconds: int = 600
    approximation_epsilon: float = 1e-10
    output_dir: Path = field(default=DEFAULT_WISQ_OUTPUT)
    job_info: str = "wisq"
    advanced_args: Mapping[str, str | None] | None = None

    def output_file_for(self, circuit_path: Path) -> Path:
        file_name = f"{circuit_path.stem}_{self.job_info}.qasm"
        return self.output_dir / file_name


def run_wisq_opt(circuit_path: Path, config: WisqConfig | None = None) -> TranspiledCircuit:
    if wisq_optimize is None:
        raise ImportError("wisq is not installed. Install it via `uv pip install wisq` or add it to your environment.")

    cfg = config or WisqConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_file_for(circuit_path)
    advanced_args = None
    if cfg.advanced_args:
        advanced_args = {str(key): (None if value is None else str(value)) for key, value in cfg.advanced_args.items()}

    start_time = time.perf_counter()
    wisq_optimize(
        input_path=str(circuit_path),
        output_path=str(output_path),
        target_gateset=cfg.target_gateset,
        optimization_objective=cfg.objective,
        timeout=cfg.timeout_seconds,
        approximation_epsilon=cfg.approximation_epsilon,
        advanced_args=advanced_args or {},
    )
    duration = time.perf_counter() - start_time

    optimized_circuit = QuantumCircuit.from_qasm_file(str(output_path))
    return TranspiledCircuit(
        optimizer="wisq",
        label=f"wisq_{cfg.objective.lower()}",
        circuit=optimized_circuit,
        metrics=analyze_circuit(optimized_circuit),
        artifact_path=output_path,
        metadata={
            "gate_set": cfg.target_gateset,
            "objective": cfg.objective,
            "duration_seconds": duration,
            "approximation_epsilon": cfg.approximation_epsilon,
            "advanced_args": advanced_args or {},
        },
    )


# Gate set mappings for TKET
TKET_GATE_SET_MAP: dict[str, list[str]] = {
    "ibm": ["u1", "u2", "u3", "cx"],
    "ibm_new": ["cx", "rz", "sx", "x"],
    "IBMN": ["cx", "rz", "sx", "x"],
    "nam": ["cx", "rz", "h", "x"],
}

# OpType mapping - defined as strings that will be used in the TKET sub-environment
TKET_OPTYPE_MAP: dict[str, str] = {
    "u1": "U1",
    "u2": "U2",
    "u3": "U3",
    "cx": "CX",
    "rz": "Rz",
    "sx": "SX",
    "x": "X",
    "h": "H",
    "rx": "Rx",
    "ry": "Ry",
}


@dataclass(frozen=True)
class TKETConfig:
    """Configuration for TKET (pytket) optimizer."""

    gate_set: str = "IBMN"
    output_dir: Path = field(default_factory=lambda: Path("reports/tket_runs"))
    job_info: str = "tket"

    def output_file_for(self, circuit_path: Path) -> Path:
        file_name = f"{circuit_path.stem}_{self.job_info}.qasm"
        return self.output_dir / file_name


def _build_tket_optimization_script(circuit_qasm: str, gate_names: list[str]) -> str:
    """
    Build TKET optimization script to run in sub-environment.

    Args:
        circuit_qasm: The input circuit as QASM string
        gate_names: List of target gate names (e.g., ["cx", "rz", "sx", "x"])

    Returns:
        Python script string to execute in TKET environment
    """
    # Escape the QASM string for safe embedding in triple double quotes.
    # In Python triple-quoted strings, \" is a valid escape sequence producing ",
    # and \\ produces a single \. So:
    #   - Replace \ with \\ (so \ in input becomes \\ in source, parsed back to \)
    #   - Replace """ with \"\"\" (so """ in input becomes \"\"\" in source, parsed back to """)
    escaped_qasm = circuit_qasm.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    gate_names_str = str(gate_names)

    return f'''
import json
import time
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
from pytket.passes import AutoRebase, DecomposeBoxes, FullPeepholeOptimise, RemoveRedundancies, SequencePass
from pytket.circuit import OpType

# Gate set mapping
TKET_OPTYPE_MAP = {{
    "u1": OpType.U1,
    "u2": OpType.U2,
    "u3": OpType.U3,
    "cx": OpType.CX,
    "rz": OpType.Rz,
    "sx": OpType.SX,
    "x": OpType.X,
    "h": OpType.H,
    "rx": OpType.Rx,
    "ry": OpType.Ry,
}}

# Load circuit from QASM
qasm_str = """{escaped_qasm}"""
tket_circuit = circuit_from_qasm_str(qasm_str)

# Get target gates
gate_names = {gate_names_str}
target_gates = {{TKET_OPTYPE_MAP[g] for g in gate_names if g in TKET_OPTYPE_MAP}}

# Build optimization pass sequence
seq_pass = SequencePass([
    DecomposeBoxes(),
    FullPeepholeOptimise(),
    RemoveRedundancies(),
    AutoRebase(target_gates),
])

# Run optimization
start_time = time.perf_counter()
seq_pass.apply(tket_circuit)
duration = time.perf_counter() - start_time

# Convert to QASM and output as JSON
output_qasm = circuit_to_qasm_str(tket_circuit)
result = {{
    "qasm": output_qasm,
    "duration": duration
}}
print(json.dumps(result))
'''


def run_tket(circuit_path: Path, config: TKETConfig | None = None) -> TranspiledCircuit:
    """Run TKET (pytket) optimizer on a circuit using isolated sub-environment.

    Uses FullPeepholeOptimise followed by gate set rebasing.
    PyTKET runs in .venv-tket to avoid networkx conflicts with qiskit-ibm-ai-local-transpiler.
    """
    if not PYTKET_AVAILABLE:
        raise ImportError(
            "TKET environment not found. Set it up via `./scripts/setup_tket_env.sh` "
            "or manually create .venv-tket with pytket installed."
        )

    cfg = config or TKETConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_file_for(circuit_path)

    # Read input circuit as QASM
    circuit_qasm = circuit_path.read_text()

    # Get target gate set configuration
    gate_set_name = cfg.gate_set.lower() if cfg.gate_set not in TKET_GATE_SET_MAP else cfg.gate_set
    gate_names = TKET_GATE_SET_MAP.get(gate_set_name, TKET_GATE_SET_MAP.get("IBMN", []))

    # Build TKET optimization script from template
    # Note: We build the script inline to inject the QASM and gate names as literal strings
    # The template in tket_optimize_template.py serves as documentation and for direct usage
    script = _build_tket_optimization_script(circuit_qasm, gate_names)

    # Execute in TKET sub-environment
    from benchmarks.tket_runner import run_tket_script
    
    result = run_tket_script(script, capture_output=True)
    
    # Parse JSON result
    import json
    result_data = json.loads(result.stdout.strip())
    optimized_qasm = result_data["qasm"]
    duration = result_data["duration"]

    # Convert to Qiskit circuit
    optimized_circuit = QuantumCircuit.from_qasm_str(optimized_qasm)

    # Save output
    output_path.write_text(qasm2.dumps(optimized_circuit))

    return TranspiledCircuit(
        optimizer="tket",
        label=f"tket_{cfg.gate_set.lower()}",
        circuit=optimized_circuit,
        metrics=analyze_circuit(optimized_circuit),
        artifact_path=output_path,
        metadata={
            "gate_set": cfg.gate_set,
            "duration_seconds": duration,
        },
    )


# VOQC optimization method mappings
VOQC_OPTIMIZATION_METHODS: dict[str, list[str]] = {
    "nam": ["optimize_nam"],
    "ibm": ["optimize_nam", "optimize_ibm"],
    "default": [],
}


@dataclass(frozen=True)
class VOQCConfig:
    """Configuration for VOQC (Verified Optimizer for Quantum Circuits)."""

    optimization_method: str = "nam"
    output_dir: Path = field(default_factory=lambda: Path("reports/voqc_runs"))
    job_info: str = "voqc"

    def output_file_for(self, circuit_path: Path) -> Path:
        file_name = f"{circuit_path.stem}_{self.job_info}.qasm"
        return self.output_dir / file_name


def run_voqc(circuit_path: Path, config: VOQCConfig | None = None) -> TranspiledCircuit:
    """Run VOQC (Verified Optimizer for Quantum Circuits) on a circuit.

    VOQC is a formally verified optimizer that guarantees semantic preservation.
    Note: VOQC doesn't support sx or rxx gates.
    """
    if not PYVOQC_AVAILABLE or voqc_pass_manager is None:
        error_msg = (
            "pyvoqc is not available. Install it via `uv pip install pyvoqc` "
            "(requires OCaml voqc library from opam)."
        )
        if PYVOQC_ERROR:
            error_msg += f" Import error: {PYVOQC_ERROR}"
        raise ImportError(error_msg)

    cfg = config or VOQCConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_file_for(circuit_path)

    # Load circuit
    circuit = QuantumCircuit.from_qasm_file(str(circuit_path))

    # Get optimization passes for the method
    post_opts = VOQC_OPTIMIZATION_METHODS.get(cfg.optimization_method, VOQC_OPTIMIZATION_METHODS["default"])

    # Create VOQC pass manager (voqc_pass_manager guaranteed non-None after guard above)
    vpm = voqc_pass_manager(post_opts=post_opts)

    start_time = time.perf_counter()
    optimized_circuit = vpm.run(circuit)
    duration = time.perf_counter() - start_time

    # Save output
    output_path.write_text(qasm2.dumps(optimized_circuit))

    return TranspiledCircuit(
        optimizer="voqc",
        label=f"voqc_{cfg.optimization_method}",
        circuit=optimized_circuit,
        metrics=analyze_circuit(optimized_circuit),
        artifact_path=output_path,
        metadata={
            "optimization_method": cfg.optimization_method,
            "duration_seconds": duration,
            "post_opts": post_opts,
        },
    )
