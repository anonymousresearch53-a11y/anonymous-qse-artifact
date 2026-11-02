"""Script to compare quantum circuits from circuit benchmark results or directly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

try:
    from benchmarks.ai_transpile.circuit_comparison import (
        compare_against_baseline,
        compare_circuits,
    )
except ImportError:
    # Add project root to path if running as script
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from benchmarks.ai_transpile.circuit_comparison import (
        compare_against_baseline,
        compare_circuits,
    )


def load_circuit_from_path(path: Path) -> QuantumCircuit:
    """Load a circuit from a file path (supports .qasm files)."""
    if path.suffix == ".qasm":
        # Use qasm2.load() for file paths - it handles includes properly
        return qasm2.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def compare_from_results_json(results_path: Path, circuit_name: str | None = None) -> None:
    """Compare circuits from a circuit benchmark results JSON file.

    For each circuit, compares all runner outputs against the baseline (first result).
    """
    results = json.loads(results_path.read_text())

    # Group results by circuit name
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for result in results.get("results", []):
        circ_name = result["circuit"]
        if circ_name not in by_circuit:
            by_circuit[circ_name] = []
        by_circuit[circ_name].append(result)

    # Filter to specific circuit if requested
    if circuit_name:
        if circuit_name not in by_circuit:
            print(f"Error: Circuit '{circuit_name}' not found in results")
            return
        by_circuit = {circuit_name: by_circuit[circuit_name]}

    # Compare each circuit's results
    for circ_name, circ_results in by_circuit.items():
        print(f"\n{'='*60}")
        print(f"Circuit: {circ_name}")
        print(f"{'='*60}")

        if len(circ_results) < 2:
            print(f"  Only {len(circ_results)} result(s) found - need at least 2 to compare")
            continue

        # Find baseline: prefer result with artifact_path, otherwise use first
        baseline_result: dict[str, Any] | None = None
        baseline_circuit: QuantumCircuit | None = None
        
        # First, try to find a result with artifact_path
        for result in circ_results:
            if "artifact_path" in result:
                baseline_result = result
                try:
                    baseline_circuit = load_circuit_from_path(Path(result["artifact_path"]))
                    break
                except Exception as exc:
                    print(f"  Warning: Could not load baseline from {result['artifact_path']}: {exc}")
                    continue
        
        # If no artifact_path found, try to load original circuit from benchmark metadata
        if baseline_circuit is None:
            try:
                # Try to find the original circuit file from benchmark metadata
                benchmark_root = Path(__file__).resolve().parents[1] / "benchmarks" / "ai_transpile"
                metadata_path = benchmark_root / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    for circ_entry in metadata.get("circuits", []):
                        if circ_entry["name"] == circ_name:
                            original_path = benchmark_root / circ_entry["file"]
                            if original_path.exists():
                                baseline_circuit = load_circuit_from_path(original_path)
                                baseline_result = {
                                    "runner": "original",
                                    "label": "original_circuit",
                                    "metrics": circ_entry.get("metrics", {}),
                                }
                                print("  Using original circuit from benchmark as baseline")
                                break
            except Exception as exc:
                print(f"  Could not load original circuit: {exc}")
        
        if baseline_circuit is None:
            print("  No baseline circuit available (no artifact_paths and original not found)")
            print("  Available results:")
            for result in circ_results:
                has_artifact = "artifact_path" in result
                print(f"    - {result['runner']}/{result['label']} (artifact_path: {has_artifact})")
            continue

        if baseline_result is None:
            print("  Baseline metadata is unavailable even though a circuit was loaded. Skipping circuit.")
            continue

        baseline_label = f"{baseline_result['runner']}/{baseline_result['label']}"
        print(f"\nBaseline: {baseline_label}")
        print(f"  Metrics: depth={baseline_result['metrics']['depth']}, "
              f"2q_gates={baseline_result['metrics']['two_qubit_gates']}")

        # Compare each other result against baseline
        for result in circ_results:
            # Skip if this is the baseline itself
            if result == baseline_result:
                continue
                
            runner_label = f"{result['runner']}/{result['label']}"
            print(f"\nComparing against: {runner_label}")
            print(f"  Metrics: depth={result['metrics']['depth']}, "
                  f"2q_gates={result['metrics']['two_qubit_gates']}")

            # Try to load circuit
            optimized_circuit = None
            if "artifact_path" in result:
                try:
                    optimized_circuit = load_circuit_from_path(Path(result["artifact_path"]))
                except Exception as exc:
                    print(f"  Could not load circuit from artifact_path: {exc}")
                    continue
            else:
                print("  No artifact_path - cannot compare (Qiskit AI circuits need to be saved)")
                print("  Tip: Use --compare-against when running circuit_benchmark_runner.py to save circuits")
                continue

            try:
                comparison = compare_against_baseline(baseline_circuit, optimized_circuit)

                print(f"  Equivalent: {comparison['equivalent']}")
                print(f"  Method: {comparison['method']}")
                if comparison.get("fidelity"):
                    print(f"  Fidelity: {comparison['fidelity']:.6f}")
                if comparison.get("error"):
                    print(f"  Error: {comparison['error']}")

                # Show metric differences
                baseline_metrics = baseline_result["metrics"]
                result_metrics = result["metrics"]
                print("  Metric deltas (vs baseline):")
                print(f"    depth: {result_metrics['depth'] - baseline_metrics['depth']:+d}")
                print(f"    2q_gates: {result_metrics['two_qubit_gates'] - baseline_metrics['two_qubit_gates']:+d}")
                print(f"    2q_depth: {result_metrics['two_qubit_depth'] - baseline_metrics['two_qubit_depth']:+d}")
                print(f"    total_gates: {result_metrics['total_gates'] - baseline_metrics['total_gates']:+d}")

            except Exception as exc:
                print(f"  Comparison failed: {exc}")


def compare_two_files(
    file1: Path,
    file2: Path,
    method: Literal["auto", "qcec", "operator", "statevector"] = "auto",
) -> None:
    """Compare two circuit files directly."""
    print("Comparing circuits:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")

    try:
        circuit1 = load_circuit_from_path(file1)
        circuit2 = load_circuit_from_path(file2)

        print(f"\nCircuit 1: {circuit1.num_qubits} qubits, {circuit1.size()} gates, depth {circuit1.depth()}")
        print(f"Circuit 2: {circuit2.num_qubits} qubits, {circuit2.size()} gates, depth {circuit2.depth()}")

        comparison = compare_circuits(circuit1, circuit2, method=method)

        print("\nComparison Results:")
        print(f"  Equivalent: {comparison.equivalent}")
        print(f"  Method: {comparison.method}")
        if comparison.fidelity is not None:
            print(f"  Fidelity: {comparison.fidelity:.6f}")
        if comparison.error:
            print(f"  Error: {comparison.error}")
        if comparison.details:
            print(f"  Details: {comparison.details}")

    except Exception as exc:
        print(f"Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare quantum circuits from circuit benchmark results or directly from files."
    )
    parser.add_argument(
        "input1",
        type=Path,
        help="First circuit file (.qasm) or circuit benchmark results JSON",
    )
    parser.add_argument(
        "input2",
        type=Path,
        nargs="?",
        help="Second circuit file (.qasm) - required if input1 is not a JSON file",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        help="Specific circuit name to compare (only used with results JSON)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "qcec", "operator", "statevector"],
        default="auto",
        help="Comparison method to use",
    )

    args = parser.parse_args()

    # Check if input1 is a JSON file (circuit benchmark results)
    if args.input1.suffix == ".json":
        compare_from_results_json(args.input1, circuit_name=args.circuit)
    elif args.input2:
        # Compare two files directly
        method_choice = cast(Literal["auto", "qcec", "operator", "statevector"], args.method)
        compare_two_files(args.input1, args.input2, method=method_choice)
    else:
        parser.error("If input1 is not a JSON file, input2 is required")


if __name__ == "__main__":
    main()
