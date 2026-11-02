#!/usr/bin/env python3
"""Verify semantic equivalence for the chain pilot circuits cited in the paper.

Compares original input circuits against:
1. Best single-optimizer outputs
2. Best chain outputs (for QFT_8)

Uses the circuit_comparison module with QCEC → operator → statevector fallback.

Note: WISQ writes intermediate solutions as tmp files. The "main" output file
is the LAST solution written, not necessarily the best. The framework reports
metrics from the best intermediate. We must use the tmp file with the correct
gate count.
"""

import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import SXGate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.ai_transpile.circuit_comparison import compare_circuits  # noqa: E402

# Define custom gates for QASM 2.0 loading (sx is not in qelib1.inc)
CUSTOM_INSTRUCTIONS = [
    qasm2.CustomInstruction("sx", 0, 1, constructor=SXGate),
]


def load_qasm(path: str | Path) -> QuantumCircuit:
    """Load a QASM file into a QuantumCircuit, handling sx gate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QASM file not found: {path}")
    try:
        return qasm2.load(
            str(path),
            custom_instructions=CUSTOM_INSTRUCTIONS,
        )
    except Exception:
        # Fallback: try QuantumCircuit.from_qasm_file
        return QuantumCircuit.from_qasm_file(str(path))


def count_2q_gates(circuit: QuantumCircuit) -> int:
    """Count two-qubit gates in a circuit."""
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def check_pair(name: str, original_path: str, optimized_path: str) -> dict:
    """Check equivalence of an original/optimized pair."""
    print(f"\n{'='*60}")
    print(f"Checking: {name}")
    print(f"  Original:  {original_path}")
    print(f"  Optimized: {optimized_path}")
    
    try:
        original = load_qasm(original_path)
        optimized = load_qasm(optimized_path)
    except Exception as e:
        print(f"  ERROR loading circuits: {e}")
        return {"name": name, "status": "load_error", "error": str(e)}
    
    orig_2q = count_2q_gates(original)
    opt_2q = count_2q_gates(optimized)
    print(f"  Original:  {original.num_qubits} qubits, {original.size()} gates, {orig_2q} 2Q")
    print(f"  Optimized: {optimized.num_qubits} qubits, {optimized.size()} gates, {opt_2q} 2Q")
    
    # Try equivalence check
    result = compare_circuits(original, optimized, method="auto")
    
    print(f"  Method:     {result.method}")
    print(f"  Equivalent: {result.equivalent}")
    if result.fidelity is not None:
        print(f"  Fidelity:   {result.fidelity:.15f}")
    if result.error:
        print(f"  Error:      {result.error}")
    if result.details:
        print(f"  Details:    {result.details}")
    
    return {
        "name": name,
        "status": "checked",
        "method": result.method,
        "equivalent": result.equivalent,
        "fidelity": result.fidelity,
        "error": result.error,
        "details": result.details,
        "original_qubits": original.num_qubits,
        "original_gates": original.size(),
        "original_2q_gates": orig_2q,
        "optimized_qubits": optimized.num_qubits,
        "optimized_gates": optimized.size(),
        "optimized_2q_gates": opt_2q,
    }


def main():
    base = project_root
    reports = base / "reports" / "chain_experiment"
    qasm_dir = base / "benchmarks" / "ai_transpile" / "qasm"
    
    checks = []
    
    # ================================================================
    # QFT_8: 8 qubits
    # ================================================================
    
    # Best single: wisq_bqskit (43 2Q gates)
    checks.append(check_pair(
        "qft_8: original vs wisq_bqskit_baseline (best single, 43 2Q)",
        qasm_dir / "qft_8.qasm",
        reports / "baselines" / "wisq_bqskit" / "qft_8_chain_experiment_wisq_bqskit_baseline.qasm",
    ))
    
    # Best chain: tket → wisq_rules (35 2Q gates)
    # NOTE: WISQ saves intermediate solutions as tmp files. The main file is the
    # LAST solution written (which may be worse). We use the tmp file with 35 CX gates.
    chain_35_path = (
        reports / "chains" / "tket_then_wisq_rules" / "step_1_wisq_rules"
        / "tmpdr7294rp_chain_experiment_tket_then_wisq_rules_step1_wisq_rules.qasm"
    )
    checks.append(check_pair(
        "qft_8: original vs tket→wisq_rules chain (best chain, 35 2Q)",
        qasm_dir / "qft_8.qasm",
        chain_35_path,
    ))
    
    # TKET baseline (intermediate step of the chain)
    checks.append(check_pair(
        "qft_8: original vs tket_baseline (56 2Q)",
        qasm_dir / "qft_8.qasm",
        reports / "tket" / "tket_baseline" / "qft_8_chain_experiment_tket_baseline.qasm",
    ))
    
    # WISQ rules baseline
    checks.append(check_pair(
        "qft_8: original vs wisq_rules_baseline (44 2Q)",
        qasm_dir / "qft_8.qasm",
        reports / "baselines" / "wisq_rules" / "qft_8_chain_experiment_wisq_rules_baseline.qasm",
    ))
    
    # ================================================================
    # EFFICIENT_SU2_12: 12 qubits
    # ================================================================
    
    checks.append(check_pair(
        "efficient_su2_12: original vs wisq_rules_baseline (best single, 11 2Q)",
        qasm_dir / "efficient_su2_12.qasm",
        reports / "baselines" / "wisq_rules" / "efficient_su2_12_chain_experiment_wisq_rules_baseline.qasm",
    ))
    
    # wisq_bqskit baseline
    checks.append(check_pair(
        "efficient_su2_12: original vs wisq_bqskit_baseline (23 2Q)",
        qasm_dir / "efficient_su2_12.qasm",
        reports / "baselines" / "wisq_bqskit" / "efficient_su2_12_chain_experiment_wisq_bqskit_baseline.qasm",
    ))
    
    # ================================================================
    # REAL_AMPLITUDES_8_R2: 8 qubits
    # ================================================================
    
    checks.append(check_pair(
        "real_amplitudes_8_r2: original vs wisq_bqskit_baseline (best single, 6 2Q)",
        qasm_dir / "real_amplitudes_8_r2.qasm",
        reports / "baselines" / "wisq_bqskit" / "real_amplitudes_8_r2_chain_experiment_wisq_bqskit_baseline.qasm",
    ))
    
    # WISQ rules baseline
    checks.append(check_pair(
        "real_amplitudes_8_r2: original vs wisq_rules_baseline (14 2Q)",
        qasm_dir / "real_amplitudes_8_r2.qasm",
        reports / "baselines" / "wisq_rules" / "real_amplitudes_8_r2_chain_experiment_wisq_rules_baseline.qasm",
    ))
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("EQUIVALENCE CHECK SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    errors = 0
    
    for c in checks:
        if c.get("status") == "load_error":
            icon = "?"
            errors += 1
        elif c.get("equivalent"):
            icon = "✓"
            passed += 1
        else:
            icon = "✗"
            failed += 1
        
        method = c.get("method", "N/A")
        fidelity_str = f", fidelity={c['fidelity']:.15f}" if c.get("fidelity") is not None else ""
        error_str = f" ERROR: {c.get('error')}" if c.get("error") and not c.get("equivalent") else ""
        print(f"  {icon} {c['name']}")
        print(f"    method={method}{fidelity_str}{error_str}")
    
    print(f"\nPassed: {passed}, Failed: {failed}, Errors: {errors}")
    
    # Save results
    output_path = base / "reports" / "chain_experiment" / "equivalence_check_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(checks, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
