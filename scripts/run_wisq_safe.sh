#!/usr/bin/env bash
#
# Resource-limited wrapper for running WISQ/GUOQ resynthesis optimizations
#
# This script wraps wisq execution with resource limits to prevent the system
# from becoming unresponsive during heavy optimization workloads. It:
#   - Limits BQSKit workers to 15% of available CPU cores (prevents memory exhaustion)
#   - Runs the process with lower CPU priority (nice)
#   - Runs the process with idle I/O priority (ionice)
#
# Usage:
#   ./scripts/run_wisq_safe.sh <command>
#   ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
#   ./scripts/run_wisq_safe.sh uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py
#
# Environment variables:
#   BQSKIT_NUM_WORKERS     - Number of BQSKit workers (default: 15% of CPU cores)
#   BQSKIT_WORKER_FRACTION - Fraction of CPU cores to use (default: 0.15)
#   WISQ_NICE_LEVEL        - Nice level for CPU priority (default: 15)
#   WISQ_IONICE_CLASS      - I/O scheduling class: 1=realtime, 2=best-effort, 3=idle (default: 3)
#
# Examples:
#   # Run with defaults (15% cores ~8 workers, nice 15, idle I/O)
#   ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
#
#   # Run with custom worker count
#   BQSKIT_NUM_WORKERS=4 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm -ap 1e-10
#
#   # Run with 25% of CPU cores
#   BQSKIT_WORKER_FRACTION=0.25 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm -ap 1e-10
#
#   # Run with higher priority (less nice, best-effort I/O)
#   WISQ_NICE_LEVEL=5 WISQ_IONICE_CLASS=2 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm

set -euo pipefail

# BQSKit / NumPy BLAS thread control — prevents OOM from thread-local matrix buffer explosion
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Get the number of CPU cores
get_cpu_count() {
    if command -v nproc &> /dev/null; then
        nproc
    elif [ -f /proc/cpuinfo ]; then
        grep -c ^processor /proc/cpuinfo
    else
        # Fallback to 4 if we can't detect
        echo 4
    fi
}

# Calculate number of workers based on fraction
calculate_workers() {
    local total_cores
    local fraction
    local workers
    
    total_cores=$(get_cpu_count)
    fraction="${BQSKIT_WORKER_FRACTION:-0.15}"
    
    # Calculate workers as integer (bash doesn't do floating point)
    # Use awk for floating point calculation
    workers=$(awk "BEGIN {printf \"%.0f\", $total_cores * $fraction}")
    
    # Ensure at least 1 worker
    if [ "$workers" -lt 1 ]; then
        workers=1
    fi
    
    echo "$workers"
}

# Configure BQSKit workers if not already set
if [ -z "${BQSKIT_NUM_WORKERS:-}" ]; then
    export BQSKIT_NUM_WORKERS
    BQSKIT_NUM_WORKERS=$(calculate_workers)
    echo ">>> Setting BQSKIT_NUM_WORKERS=$BQSKIT_NUM_WORKERS (of $(get_cpu_count) cores)"
else
    echo ">>> Using existing BQSKIT_NUM_WORKERS=$BQSKIT_NUM_WORKERS"
fi

# Configure nice level (lower priority = higher nice value)
NICE_LEVEL="${WISQ_NICE_LEVEL:-15}"

# Configure ionice class (3 = idle, only runs when system is idle)
IONICE_CLASS="${WISQ_IONICE_CLASS:-3}"

# Check if ionice is available
if command -v ionice &> /dev/null; then
    echo ">>> Running with nice level $NICE_LEVEL and ionice class $IONICE_CLASS (idle)"
    exec nice -n "$NICE_LEVEL" ionice -c "$IONICE_CLASS" "$@"
else
    echo ">>> ionice not available, running with nice level $NICE_LEVEL only"
    exec nice -n "$NICE_LEVEL" "$@"
fi


