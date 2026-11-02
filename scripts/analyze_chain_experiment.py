#!/usr/bin/env python3
"""Analyze chain experiment results and compare chains vs individual optimizers.

Usage:
    uv run python scripts/analyze_chain_experiment.py [--results PATH]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OptimizerResult:
    """Result from a single optimizer run."""

    circuit: str
    runner: str
    optimizer: str
    two_qubit_gates: int
    depth: int
    total_gates: int
    duration: float
    is_chain: bool
    chain_steps: list[str] | None = None


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def parse_results(results: list[dict[str, Any]]) -> list[OptimizerResult]:
    """Parse raw results into structured format."""
    parsed = []
    for r in results:
        is_chain = r["optimizer"] == "chain"
        
        # Get duration
        if is_chain:
            duration = r["metadata"].get("total_duration_seconds", 0)
            chain_steps = [s["name"] for s in r["metadata"].get("steps", [])]
        else:
            duration = r["metadata"].get("duration_seconds", 0)
            chain_steps = None
        
        parsed.append(OptimizerResult(
            circuit=r["circuit"],
            runner=r["runner"],
            optimizer=r["optimizer"],
            two_qubit_gates=r["metrics"]["two_qubit_gates"],
            depth=r["metrics"]["depth"],
            total_gates=r["metrics"]["total_gates"],
            duration=duration,
            is_chain=is_chain,
            chain_steps=chain_steps,
        ))
    return parsed


def analyze_circuit(circuit: str, results: list[OptimizerResult]) -> dict[str, Any]:
    """Analyze results for a single circuit."""
    circuit_results = [r for r in results if r.circuit == circuit]
    
    # Separate baselines and chains
    baselines = [r for r in circuit_results if not r.is_chain]
    chains = [r for r in circuit_results if r.is_chain]
    
    # Find best baseline
    best_baseline = min(baselines, key=lambda r: r.two_qubit_gates)
    
    # Find best chain
    best_chain = min(chains, key=lambda r: r.two_qubit_gates) if chains else None
    
    # Find best overall
    best_overall = min(circuit_results, key=lambda r: r.two_qubit_gates)
    
    return {
        "circuit": circuit,
        "baselines": baselines,
        "chains": chains,
        "best_baseline": best_baseline,
        "best_chain": best_chain,
        "best_overall": best_overall,
    }


def print_comparison_table(analysis: dict[str, Any]) -> None:
    """Print a comparison table for a circuit."""
    circuit = analysis["circuit"]
    baselines = analysis["baselines"]
    chains = analysis["chains"]
    best_baseline = analysis["best_baseline"]
    
    print(f"\n{'='*80}")
    print(f"Circuit: {circuit}")
    print(f"{'='*80}")
    
    # Baselines table
    print(f"\n{'INDIVIDUAL BASELINES':^80}")
    print(f"{'-'*80}")
    print(f"{'Runner':<30} {'2Q Gates':>10} {'Depth':>10} {'Duration':>12}")
    print(f"{'-'*80}")
    
    for r in sorted(baselines, key=lambda x: x.two_qubit_gates):
        marker = " *" if r == best_baseline else ""
        print(f"{r.runner:<30} {r.two_qubit_gates:>10} {r.depth:>10} {r.duration:>11.2f}s{marker}")
    
    # Chains table
    print(f"\n{'CHAIN RESULTS':^80}")
    print(f"{'-'*80}")
    print(f"{'Chain':<30} {'2Q Gates':>10} {'Depth':>10} {'Duration':>12} {'vs Best':>10}")
    print(f"{'-'*80}")
    
    for r in sorted(chains, key=lambda x: x.two_qubit_gates):
        improvement = ((best_baseline.two_qubit_gates - r.two_qubit_gates) / 
                      best_baseline.two_qubit_gates * 100) if best_baseline.two_qubit_gates > 0 else 0
        imp_str = f"{improvement:+.1f}%" if improvement != 0 else "0%"
        print(f"{r.runner:<30} {r.two_qubit_gates:>10} {r.depth:>10} {r.duration:>11.2f}s {imp_str:>10}")
    
    # Summary
    best_chain = analysis["best_chain"]
    best_overall = analysis["best_overall"]
    
    print(f"\n{'SUMMARY':^80}")
    print(f"{'-'*80}")
    print(f"Best baseline:  {best_baseline.runner} ({best_baseline.two_qubit_gates} 2Q gates)")
    if best_chain:
        print(f"Best chain:     {best_chain.runner} ({best_chain.two_qubit_gates} 2Q gates)")
        if best_chain.two_qubit_gates < best_baseline.two_qubit_gates:
            diff = best_baseline.two_qubit_gates - best_chain.two_qubit_gates
            improvement = diff / best_baseline.two_qubit_gates * 100
            print(f"Chain improves: {improvement:.1f}% over best baseline")
        elif best_chain.two_qubit_gates > best_baseline.two_qubit_gates:
            diff = best_chain.two_qubit_gates - best_baseline.two_qubit_gates
            degradation = diff / best_baseline.two_qubit_gates * 100
            print(f"Chain degrades: {degradation:.1f}% vs best baseline")
        else:
            print("Chain matches best baseline")
    print(f"Best overall:   {best_overall.runner} ({best_overall.two_qubit_gates} 2Q gates)")


def print_summary(all_analyses: list[dict[str, Any]]) -> None:
    """Print overall summary."""
    print(f"\n{'='*80}")
    print(f"{'OVERALL SUMMARY':^80}")
    print(f"{'='*80}")
    
    chain_wins = 0
    baseline_wins = 0
    ties = 0
    
    for analysis in all_analyses:
        best_baseline = analysis["best_baseline"]
        best_chain = analysis["best_chain"]
        
        if best_chain:
            if best_chain.two_qubit_gates < best_baseline.two_qubit_gates:
                chain_wins += 1
            elif best_chain.two_qubit_gates > best_baseline.two_qubit_gates:
                baseline_wins += 1
            else:
                ties += 1
    
    total = len(all_analyses)
    print(f"\nChain beats best baseline: {chain_wins}/{total} circuits")
    print(f"Baseline beats best chain: {baseline_wins}/{total} circuits")
    print(f"Ties:                      {ties}/{total} circuits")
    
    # Best chain per circuit
    print(f"\n{'Best Optimization Strategy per Circuit':^80}")
    print(f"{'-'*80}")
    print(f"{'Circuit':<25} {'Best Strategy':<35} {'2Q Gates':>10}")
    print(f"{'-'*80}")
    
    for analysis in all_analyses:
        best = analysis["best_overall"]
        print(f"{analysis['circuit']:<25} {best.runner:<35} {best.two_qubit_gates:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chain experiment results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/chain_experiment/latest_results.json"),
        help="Path to results JSON file",
    )
    args = parser.parse_args()
    
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        return
    
    # Load and parse results
    raw_results = load_results(args.results)
    results = parse_results(raw_results)
    
    # Get unique circuits
    circuits = sorted(set(r.circuit for r in results))
    
    # Analyze each circuit
    all_analyses = []
    for circuit in circuits:
        analysis = analyze_circuit(circuit, results)
        all_analyses.append(analysis)
        print_comparison_table(analysis)
    
    # Print overall summary
    print_summary(all_analyses)


if __name__ == "__main__":
    main()

