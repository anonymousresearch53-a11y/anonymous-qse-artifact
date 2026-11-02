#!/usr/bin/env bash
#
# Strict resource-limited wrapper using systemd-run for WISQ/GUOQ resynthesis
#
# This script provides HARD resource limits using Linux cgroups via systemd-run.
# Unlike nice/ionice which only lower priority, systemd-run enforces actual caps
# that cannot be exceeded, guaranteeing system responsiveness.
#
# Prerequisites:
#   - systemd (standard on most modern Linux distributions)
#   - User must be in a systemd user session (check with: systemctl --user status)
#
# Usage:
#   ./scripts/run_wisq_systemd.sh <command>
#   ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
#   ./scripts/run_wisq_systemd.sh uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py
#
# Environment variables:
#   WISQ_CPU_QUOTA         - CPU quota as percentage (100% = 1 core, 200% = 2 cores)
#                            Default: 200% (2 CPU cores worth)
#   WISQ_MEMORY_MAX        - Maximum memory (e.g., 8G, 4G, 16G)
#                            Default: 8G
#   WISQ_MEMORY_SWAP_MAX   - Maximum swap usage (0 disables swap to prevent thrashing)
#                            Default: 0
#   BQSKIT_NUM_WORKERS     - Number of BQSKit workers (default: calculated from CPU_QUOTA)
#   BQSKIT_WORKER_FRACTION - Fraction of CPU cores to use (default: 0.5)
#
# Examples:
#   # Run with defaults (2 cores, 8GB RAM, no swap)
#   ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
#
#   # Run with 4 cores and 16GB RAM
#   WISQ_CPU_QUOTA=400 WISQ_MEMORY_MAX=16G ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm
#
#   # Run with very conservative limits (1 core, 4GB)
#   WISQ_CPU_QUOTA=100 WISQ_MEMORY_MAX=4G ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm
#
# Notes:
#   - If systemd user session is not available, falls back to run_wisq_safe.sh
#   - CPU quota is NOT the number of cores, but percentage of total CPU time
#     (100% = 1 core, 400% = 4 cores at 100% each)
#   - Setting WISQ_MEMORY_SWAP_MAX=0 prevents swap thrashing which can freeze systems

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default resource limits
CPU_QUOTA="${WISQ_CPU_QUOTA:-200}"  # 200% = 2 CPU cores
MEMORY_MAX="${WISQ_MEMORY_MAX:-8G}"
MEMORY_SWAP_MAX="${WISQ_MEMORY_SWAP_MAX:-0}"

# Check if systemd user session is available
check_systemd_user() {
    if ! command -v systemd-run &> /dev/null; then
        return 1
    fi
    
    # Check if we're in a systemd user session
    if ! systemctl --user status &> /dev/null; then
        return 1
    fi
    
    return 0
}

# Get the number of CPU cores
get_cpu_count() {
    if command -v nproc &> /dev/null; then
        nproc
    elif [ -f /proc/cpuinfo ]; then
        grep -c ^processor /proc/cpuinfo
    else
        echo 4
    fi
}

# Calculate workers based on CPU quota or fraction
calculate_workers() {
    local total_cores
    local workers
    
    total_cores=$(get_cpu_count)
    
    if [ -n "${BQSKIT_NUM_WORKERS:-}" ]; then
        echo "$BQSKIT_NUM_WORKERS"
        return
    fi
    
    if [ -n "${BQSKIT_WORKER_FRACTION:-}" ]; then
        workers=$(awk "BEGIN {printf \"%.0f\", $total_cores * $BQSKIT_WORKER_FRACTION}")
    else
        # Calculate based on CPU quota (100% = 1 core)
        workers=$(awk "BEGIN {printf \"%.0f\", $CPU_QUOTA / 100}")
    fi
    
    # Ensure at least 1 worker
    if [ "$workers" -lt 1 ]; then
        workers=1
    fi
    
    echo "$workers"
}

# Configure BQSKit workers
if [ -z "${BQSKIT_NUM_WORKERS:-}" ]; then
    export BQSKIT_NUM_WORKERS
    BQSKIT_NUM_WORKERS=$(calculate_workers)
fi

echo ">>> Resource limits configuration:"
echo "    CPU Quota: ${CPU_QUOTA}% (equivalent to ~$(awk "BEGIN {printf \"%.1f\", $CPU_QUOTA / 100}") cores)"
echo "    Memory Max: ${MEMORY_MAX}"
echo "    Swap Max: ${MEMORY_SWAP_MAX}"
echo "    BQSKit Workers: ${BQSKIT_NUM_WORKERS}"

if ! check_systemd_user; then
    echo ">>> WARNING: systemd user session not available"
    echo ">>> Falling back to run_wisq_safe.sh (nice/ionice only)"
    exec "$SCRIPT_DIR/run_wisq_safe.sh" "$@"
fi

echo ">>> Running with systemd-run resource limits..."

# Run with systemd-run resource limits
exec systemd-run --user --scope \
    -p CPUQuota="${CPU_QUOTA}%" \
    -p MemoryMax="${MEMORY_MAX}" \
    -p MemorySwapMax="${MEMORY_SWAP_MAX}" \
    --description="WISQ/GUOQ resynthesis optimization" \
    "$@"


