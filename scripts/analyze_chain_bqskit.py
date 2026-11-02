#!/usr/bin/env python3
"""Detailed analysis of BQSKit resynthesis chain experiments.

This analyzes the BQSKit-based chains to understand when heavy resynthesis helps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def analyze_bqskit_results(results_data: dict[str, Any]) -> None:
    """Analyze BQSKit resynthesis results."""
    results = results_data["results"]
    
    # Group by circuit
    circuits: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        circuit = r["circuit"]
        if circuit not in circuits:
            circuits[circuit] = []
        circuits[circuit].append(r)
    
    print("="*80)
    print("BQSKIT RESYNTHESIS CHAIN ANALYSIS")
    print("="*80)
    
    for circuit_name in sorted(circuits.keys()):
        circuit_results = circuits[circuit_name]
        
        print(f"\n{'='*80}")
        print(f"Circuit: {circuit_name}")
        print(f"{'='*80}")
        
        # Find baseline
        baseline = next((r for r in circuit_results if not r["optimizer"] == "chain"), None)
        chains = [r for r in circuit_results if r["optimizer"] == "chain"]
        
        if baseline:
            print("\nBaseline (wisq_bqskit_baseline):")
            print(f"  2Q Gates: {baseline['metrics']['two_qubit_gates']}")
            print(f"  Depth: {baseline['metrics']['depth']}")
            print(f"  Duration: {baseline['metadata'].get('duration_seconds', 0):.1f}s")
        
        print("\nChain Results:")
        print(f"{'Chain Name':<40} {'2Q Gates':>10} {'Depth':>8} {'Duration':>10} {'vs Baseline':>12}")
        print("-"*80)
        
        for chain in sorted(chains, key=lambda x: x['metrics']['two_qubit_gates']):
            name = chain['runner']
            gates = chain['metrics']['two_qubit_gates']
            depth = chain['metrics']['depth']
            duration = chain['metadata'].get('total_duration_seconds', 0)
            
            if baseline:
                baseline_gates = baseline['metrics']['two_qubit_gates']
                if baseline_gates > 0:
                    improvement = (baseline_gates - gates) / baseline_gates * 100
                    imp_str = f"{improvement:+.1f}%"
                else:
                    imp_str = "N/A"
            else:
                imp_str = "N/A"
            
            print(f"{name:<40} {gates:>10} {depth:>8} {duration:>9.1f}s {imp_str:>12}")
        
        # Key insights
        print(f"\nKey Insights for {circuit_name}:")
        
        best_chain = min(chains, key=lambda x: x['metrics']['two_qubit_gates'])
        
        if baseline:
            baseline_gates = baseline['metrics']['two_qubit_gates']
            best_gates = best_chain['metrics']['two_qubit_gates']
            
            if best_gates < baseline_gates:
                improvement = (baseline_gates - best_gates) / baseline_gates * 100
                print(f"  ✓ Best chain improves by {improvement:.1f}% over baseline")
                print(f"    ({best_chain['runner']}: {best_gates} gates vs {baseline_gates} baseline)")
            elif best_gates == baseline_gates:
                print(f"  = Chains match baseline performance ({baseline_gates} gates)")
            else:
                degradation = (best_gates - baseline_gates) / baseline_gates * 100
                print(f"  ✗ All chains degrade vs baseline (worst: {degradation:.1f}%)")
        
        # Time analysis
        if baseline:
            baseline_time = baseline['metadata'].get('duration_seconds', 0)
            fastest_chain = min(chains, key=lambda x: x['metadata'].get('total_duration_seconds', float('inf')))
            slowest_chain = max(chains, key=lambda x: x['metadata'].get('total_duration_seconds', 0))
            
            fastest_time = fastest_chain['metadata'].get('total_duration_seconds', 0)
            slowest_time = slowest_chain['metadata'].get('total_duration_seconds', 0)
            
            print("\n  Time Analysis:")
            print(f"    Baseline: {baseline_time:.1f}s")
            print(f"    Fastest chain: {fastest_time:.1f}s ({fastest_chain['runner']})")
            print(f"    Slowest chain: {slowest_time:.1f}s ({slowest_chain['runner']})")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_circuits = len(circuits)
    improvements = 0
    ties = 0
    degradations = 0
    
    for circuit_name, circuit_results in circuits.items():
        baseline = next((r for r in circuit_results if r["optimizer"] != "chain"), None)
        chains = [r for r in circuit_results if r["optimizer"] == "chain"]
        
        if baseline and chains:
            best_chain = min(chains, key=lambda x: x['metrics']['two_qubit_gates'])
            baseline_gates = baseline['metrics']['two_qubit_gates']
            best_gates = best_chain['metrics']['two_qubit_gates']
            
            if best_gates < baseline_gates:
                improvements += 1
            elif best_gates == baseline_gates:
                ties += 1
            else:
                degradations += 1
    
    print("\nChain vs Baseline Performance:")
    print(f"  Improvements: {improvements}/{total_circuits} circuits")
    print(f"  Ties:         {ties}/{total_circuits} circuits")
    print(f"  Degradations: {degradations}/{total_circuits} circuits")
    
    print("\nRecommendations:")
    if improvements > 0:
        print("  • Chains CAN improve over individual BQSKit resynthesis")
        print("  • Use chains for circuits where preprocessing helps resynthesis")
    if ties > 0:
        print("  • For some circuits, BQSKit alone is sufficient")
    if degradations > 0:
        print("  • Avoid chains that add steps without benefit")
        print("  • TKET after BQSKit may not always help")


def main() -> None:
    results_path = Path("reports/chain_experiment/latest_results.json")
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    results_data = load_results(results_path)
    analyze_bqskit_results(results_data)


if __name__ == "__main__":
    main()


