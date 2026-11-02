#!/usr/bin/env python3
"""Analyze chain experiments for synergies between optimization techniques.

Focus: Identify combinations where chaining produces BETTER results than any individual optimizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    """Structured result data."""
    circuit: str
    runner: str
    optimizer: str
    gates: int
    depth: int
    duration: float
    is_chain: bool
    steps: list[str] | None = None


def load_and_parse(path: Path) -> dict[str, list[Result]]:
    """Load results and group by circuit."""
    data = json.loads(path.read_text())
    
    circuits: dict[str, list[Result]] = {}
    
    for r in data["results"]:
        circuit = r["circuit"]
        is_chain = r["optimizer"] == "chain"
        
        duration = (
            r["metadata"].get("total_duration_seconds", 0) if is_chain
            else r["metadata"].get("duration_seconds", 0)
        )
        
        steps = None
        if is_chain:
            steps = [s["name"] for s in r["metadata"].get("steps", [])]
        
        result = Result(
            circuit=circuit,
            runner=r["runner"],
            optimizer=r["optimizer"],
            gates=r["metrics"]["two_qubit_gates"],
            depth=r["metrics"]["depth"],
            duration=duration,
            is_chain=is_chain,
            steps=steps,
        )
        
        if circuit not in circuits:
            circuits[circuit] = []
        circuits[circuit].append(result)
    
    return circuits


def analyze_synergies(circuits: dict[str, list[Result]]) -> None:
    """Find synergies where chains beat individual optimizers."""
    
    print("="*80)
    print("OPTIMIZATION SYNERGY ANALYSIS")
    print("="*80)
    print("\nGoal: Identify combinations where A + B > max(A, B)")
    print("i.e., chains that produce BETTER results than any individual optimizer")
    
    total_synergies = 0
    
    for circuit_name in sorted(circuits.keys()):
        results = circuits[circuit_name]
        
        baselines = [r for r in results if not r.is_chain]
        chains = [r for r in results if r.is_chain]
        
        # Find best baseline
        best_baseline = min(baselines, key=lambda x: x.gates)
        
        # Find chains that beat best baseline
        better_chains = [c for c in chains if c.gates < best_baseline.gates]
        
        print(f"\n{'='*80}")
        print(f"Circuit: {circuit_name}")
        print(f"{'='*80}")
        
        print(f"\nBest Individual Baseline: {best_baseline.runner}")
        print(f"  Gates: {best_baseline.gates}")
        print(f"  Duration: {best_baseline.duration:.1f}s")
        
        if better_chains:
            print(f"\n✓ SYNERGY FOUND! {len(better_chains)} chain(s) beat best baseline:")
            total_synergies += len(better_chains)
            
            for chain in sorted(better_chains, key=lambda x: x.gates):
                improvement = (best_baseline.gates - chain.gates) / best_baseline.gates * 100
                steps_str = " → ".join(chain.steps) if chain.steps else "unknown"
                
                print(f"\n  Chain: {steps_str}")
                print(f"    Gates: {chain.gates} ({improvement:+.1f}% vs baseline)")
                print(f"    Duration: {chain.duration:.1f}s")
                print(f"    Time overhead: {chain.duration - best_baseline.duration:.1f}s")
                
                # Calculate efficiency
                time_overhead = chain.duration - best_baseline.duration
                gates_saved = best_baseline.gates - chain.gates
                if time_overhead > 0:
                    efficiency = gates_saved / (time_overhead / 60)  # gates saved per minute
                    print(f"    Efficiency: {efficiency:.2f} gates saved per minute overhead")
        else:
            print("\n✗ No synergy: Best chain matches or is worse than baseline")
            best_chain = min(chains, key=lambda x: x.gates)
            if best_chain.gates == best_baseline.gates:
                print(f"  (Chains match baseline at {best_baseline.gates} gates)")
            else:
                degradation = (best_chain.gates - best_baseline.gates) / best_baseline.gates * 100
                print(f"  (Best chain has {best_chain.gates} gates, {degradation:.1f}% worse)")
        
        # Show all results for context
        print(f"\nAll Results for {circuit_name}:")
        print(f"{'Type':<10} {'Gates':>6} {'Duration':>10} {'Name/Steps'}")
        print("-"*80)
        
        all_results = sorted(results, key=lambda x: x.gates)
        for r in all_results:
            type_str = "CHAIN" if r.is_chain else "BASELINE"
            name_str = " → ".join(r.steps) if r.steps else r.runner
            marker = " ★" if r.gates == min(all_results, key=lambda x: x.gates).gates else ""
            print(f"{type_str:<10} {r.gates:>6} {r.duration:>9.1f}s {name_str}{marker}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_circuits = len(circuits)
    circuits_with_synergy = sum(
        1 for circuit_name in circuits
        if any(
            c.gates < min((r for r in circuits[circuit_name] if not r.is_chain), key=lambda x: x.gates).gates
            for c in circuits[circuit_name] if c.is_chain
        )
    )
    
    print(f"\nCircuits with synergy: {circuits_with_synergy}/{total_circuits}")
    print(f"Total synergistic chains found: {total_synergies}")
    
    # Key insights
    print(f"\n{'KEY INSIGHTS':^80}")
    print("-"*80)
    
    if circuits_with_synergy > 0:
        print("\n✓ Synergies EXIST between optimization techniques!")
        print("\nMost effective synergistic patterns:")
        
        # Analyze which patterns work
        synergy_patterns = []
        for circuit_name in circuits:
            results = circuits[circuit_name]
            baselines = [r for r in results if not r.is_chain]
            chains = [r for r in results if r.is_chain]
            best_baseline = min(baselines, key=lambda x: x.gates)
            
            for chain in chains:
                if chain.gates < best_baseline.gates:
                    improvement = (best_baseline.gates - chain.gates) / best_baseline.gates * 100
                    pattern = " → ".join(chain.steps) if chain.steps else chain.runner
                    synergy_patterns.append((pattern, improvement, circuit_name))
        
        # Show top patterns
        for pattern, improvement, circuit in sorted(synergy_patterns, key=lambda x: -x[1]):
            print(f"  • {pattern}")
            print(f"    Circuit: {circuit}, Improvement: {improvement:+.1f}%")
    else:
        print("\n✗ No clear synergies found.")
        print("  Individual optimizers perform as well or better than chains.")


def main() -> None:
    results_path = Path("reports/chain_experiment/latest_results.json")
    
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return
    
    circuits = load_and_parse(results_path)
    analyze_synergies(circuits)


if __name__ == "__main__":
    main()

