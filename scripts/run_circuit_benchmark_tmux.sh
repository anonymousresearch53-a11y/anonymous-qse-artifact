#!/usr/bin/env bash
#
# Tmux session manager for running circuit benchmark experiments in the background
#
# This script creates and manages tmux sessions for running circuit benchmark
# experiments. Sessions persist across SSH disconnections, allowing you to start
# experiments, disconnect, and reconnect later to check progress.
#
# Resource Limiting:
#   All benchmark sessions automatically use resource limiting via run_wisq_safe.sh:
#   - BQSKit workers limited to 50% of CPU cores (keeps system responsive)
#   - Process runs at nice level 15 (lower CPU priority)
#   - Process runs with idle I/O priority (ionice class 3)
#
#   Override with environment variables:
#     BQSKIT_NUM_WORKERS=8 ./scripts/run_circuit_benchmark_tmux.sh create
#     BQSKIT_WORKER_FRACTION=0.25 ./scripts/run_circuit_benchmark_tmux.sh create
#     WISQ_NICE_LEVEL=10 ./scripts/run_circuit_benchmark_tmux.sh create
#
# Usage:
#   ./scripts/run_circuit_benchmark_tmux.sh create                    # Create full session (default)
#   ./scripts/run_circuit_benchmark_tmux.sh create --skip-full          # Create no-resynth session instead
#   ./scripts/run_circuit_benchmark_tmux.sh create --only-no-resynth   # Only create no-resynth session
#   ./scripts/run_circuit_benchmark_tmux.sh create --only-full         # Only create full session (same as default)
#   ./scripts/run_circuit_benchmark_tmux.sh create --force             # Recreate sessions (kill existing first)
#   ./scripts/run_circuit_benchmark_tmux.sh status                    # Check session status
#   ./scripts/run_circuit_benchmark_tmux.sh attach <name>             # Attach to a session
#   ./scripts/run_circuit_benchmark_tmux.sh kill <name>                # Kill a session
#   ./scripts/run_circuit_benchmark_tmux.sh kill-all                  # Kill all sessions
#   ./scripts/run_circuit_benchmark_tmux.sh list                      # List all sessions

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Session names
SESSION_NO_RESYNTH="circuit-bench-no-resynth"
SESSION_FULL="circuit-bench-full"

# Output directories
OUTPUT_NO_RESYNTH="$REPO_ROOT/reports/circuit_benchmark/no_resynth"
OUTPUT_FULL="$REPO_ROOT/reports/circuit_benchmark/full"

# Log directory
LOG_DIR="$REPO_ROOT/reports/circuit_benchmark/logs"
mkdir -p "$LOG_DIR"

# Config file
CONFIG_FILE="$REPO_ROOT/benchmarks/ai_transpile/circuit_benchmark.yaml"

# Function to check if tmux is available
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        echo "Error: tmux is not installed or not in PATH"
        echo "Install with: sudo apt install tmux  # or your package manager"
        exit 1
    fi
}

# Function to check if a session exists
session_exists() {
    local session_name="$1"
    tmux has-session -t "$session_name" 2>/dev/null
}

# Function to create a session
create_session() {
    local session_name="$1"
    local output_dir="$2"
    local skip_runners=("${@:3}")
    local log_file="$LOG_DIR/${session_name}.log"

    if session_exists "$session_name"; then
        echo "⚠ Session '$session_name' already exists. Skipping creation."
        echo "   Use 'attach $session_name' to connect, or 'kill $session_name' to remove it first."
        return 1
    fi

    echo "Creating session: $session_name"
    echo "  Output directory: $output_dir"
    echo "  Log file: $log_file"
    if [ ${#skip_runners[@]} -gt 0 ]; then
        echo "  Skipping runners: ${skip_runners[*]}"
    fi

    # Build the command using the resource-limited wrapper
    # This prevents the system from becoming unresponsive during heavy WISQ optimization
    local safe_wrapper="$REPO_ROOT/scripts/run_wisq_safe.sh"
    local cmd="$safe_wrapper uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py"
    cmd="$cmd --config '$CONFIG_FILE'"
    cmd="$cmd --output '$output_dir'"
    
    for runner in "${skip_runners[@]}"; do
        cmd="$cmd --skip-runner '$runner'"
    done

    # Create detached tmux session
    tmux new-session -d -s "$session_name" -c "$REPO_ROOT" \
        "echo '=== Circuit Benchmark Experiment ===' | tee -a '$log_file'; \
         echo 'Session: $session_name' | tee -a '$log_file'; \
         echo 'Started: $(date)' | tee -a '$log_file'; \
         echo 'Output directory: $output_dir' | tee -a '$log_file'; \
         echo 'Resource limiting: nice=15, ionice=idle, workers=50%%' | tee -a '$log_file'; \
         echo '' | tee -a '$log_file'; \
         $cmd 2>&1 | tee -a '$log_file'; \
         echo '' | tee -a '$log_file'; \
         echo 'Completed: $(date)' | tee -a '$log_file'; \
         echo 'Press Enter to close this window...'; \
         read"

    echo "✓ Session '$session_name' created successfully"
    echo "  Attach with: $0 attach $session_name"
    echo "  View logs: tail -f $log_file"
}

# Function to list all sessions
list_sessions() {
    echo "Circuit Benchmark Tmux Sessions:"
    echo "================================="
    
    local sessions=("$SESSION_NO_RESYNTH" "$SESSION_FULL")
    local found=0
    
    for session in "${sessions[@]}"; do
        if session_exists "$session"; then
            found=1
            local log_file="$LOG_DIR/${session}.log"
            local last_line=""
            if [ -f "$log_file" ]; then
                last_line=$(tail -n 1 "$log_file" 2>/dev/null || echo "")
            fi
            
            echo ""
            echo "Session: $session"
            echo "  Status: Running"
            echo "  Log: $log_file"
            if [ -n "$last_line" ]; then
                echo "  Last log: ${last_line:0:80}..."
            fi
            echo "  Attach: $0 attach $session"
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo ""
        echo "No active sessions found."
        echo "Create sessions with: $0 create"
    fi
}

# Function to show status
show_status() {
    echo "Circuit Benchmark Experiment Status"
    echo "===================================="
    echo ""
    
    list_sessions
    
    echo ""
    echo "Available sessions:"
    echo "  - $SESSION_NO_RESYNTH: Runs without resynthesis (skips wisq_bqskit)"
    echo "  - $SESSION_FULL: Runs with all optimizers including resynthesis"
    echo ""
    echo "Usage:"
    echo "  $0 create [flags]  # Create sessions (default: full session only, use flags to customize)"
    echo "  $0 attach <name>   # Attach to a session"
    echo "  $0 kill <name>     # Kill a session"
    echo "  $0 kill-all        # Kill all sessions"
    echo "  $0 list            # List active sessions"
    echo ""
    echo "Create flags: --skip-no-resynth, --skip-full, --only-no-resynth, --only-full, --force"
}

# Function to attach to a session
attach_session() {
    local session_name="$1"
    
    if ! session_exists "$session_name"; then
        echo "Error: Session '$session_name' does not exist"
        echo "Create it with: $0 create"
        exit 1
    fi
    
    echo "Attaching to session: $session_name"
    echo "Press Ctrl+B then D to detach without killing the session"
    echo ""
    sleep 1
    tmux attach-session -t "$session_name"
}

# Function to kill a session
kill_session() {
    local session_name="$1"
    
    if ! session_exists "$session_name"; then
        echo "Error: Session '$session_name' does not exist"
        exit 1
    fi
    
    echo "Killing session: $session_name"
    tmux kill-session -t "$session_name"
    echo "✓ Session '$session_name' killed"
}

# Function to kill all sessions
kill_all_sessions() {
    local sessions=("$SESSION_NO_RESYNTH" "$SESSION_FULL")
    local killed=0
    
    for session in "${sessions[@]}"; do
        if session_exists "$session"; then
            echo "Killing session: $session"
            tmux kill-session -t "$session"
            killed=1
        fi
    done
    
    if [ $killed -eq 0 ]; then
        echo "No active sessions to kill"
    else
        echo "✓ All sessions killed"
    fi
}

# Function to create sessions based on flags
create_sessions() {
    local skip_no_resynth=0
    local skip_full=0
    local only_no_resynth=0
    local only_full=0
    local force=0
    
    # Parse flags
    shift  # Remove 'create' command
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-no-resynth)
                skip_no_resynth=1
                shift
                ;;
            --skip-full)
                skip_full=1
                shift
                ;;
            --only-no-resynth)
                only_no_resynth=1
                shift
                ;;
            --only-full)
                only_full=1
                shift
                ;;
            --force)
                force=1
                shift
                ;;
            *)
                echo "Error: Unknown flag '$1'"
                echo "Available flags: --skip-no-resynth, --skip-full, --only-no-resynth, --only-full, --force"
                exit 1
                ;;
        esac
    done
    
    # Validate flags (only-* flags are mutually exclusive)
    if [ $only_no_resynth -eq 1 ] && [ $only_full -eq 1 ]; then
        echo "Error: --only-no-resynth and --only-full cannot be used together"
        exit 1
    fi
    
    # Determine which sessions to create
    # Default: create only full session
    local create_no_resynth=0
    local create_full=1
    
    if [ $only_no_resynth -eq 1 ]; then
        create_no_resynth=1
        create_full=0
    elif [ $only_full -eq 1 ]; then
        create_no_resynth=0
        create_full=1
    else
        # If skip flags are provided, adjust accordingly
        if [ $skip_full -eq 1 ]; then
            create_full=0
            # If skipping full, create no-resynth instead (unless also skipped)
            if [ $skip_no_resynth -eq 0 ]; then
                create_no_resynth=1
            fi
        fi
        if [ $skip_no_resynth -eq 1 ]; then
            create_no_resynth=0
        fi
    fi
    
    # Validate that at least one session will be created
    if [ $create_no_resynth -eq 0 ] && [ $create_full -eq 0 ]; then
        echo "Error: Cannot skip both sessions. At least one session must be created."
        exit 1
    fi
    
    # Kill existing sessions if --force is set
    if [ $force -eq 1 ]; then
        if [ $create_no_resynth -eq 1 ] && session_exists "$SESSION_NO_RESYNTH"; then
            echo "Force flag set: killing existing session '$SESSION_NO_RESYNTH'"
            tmux kill-session -t "$SESSION_NO_RESYNTH" 2>/dev/null || true
        fi
        if [ $create_full -eq 1 ] && session_exists "$SESSION_FULL"; then
            echo "Force flag set: killing existing session '$SESSION_FULL'"
            tmux kill-session -t "$SESSION_FULL" 2>/dev/null || true
        fi
        if [ $create_no_resynth -eq 1 ] || [ $create_full -eq 1 ]; then
            echo ""
        fi
    fi
    
    echo "Creating circuit benchmark experiment sessions..."
    echo ""
    
    local created=0
    
    if [ $create_no_resynth -eq 1 ]; then
        create_session "$SESSION_NO_RESYNTH" "$OUTPUT_NO_RESYNTH" "wisq_bqskit"
        created=1
    fi
    
    if [ $create_full -eq 1 ]; then
        if [ $created -eq 1 ]; then
            echo ""
        fi
        create_session "$SESSION_FULL" "$OUTPUT_FULL"
        created=1
    fi
    
    if [ $created -eq 0 ]; then
        echo "No sessions to create (all skipped or filtered out)"
        exit 1
    fi
    
    echo ""
    echo "✓ Sessions created"
    echo ""
    echo "To check status: $0 status"
    echo "To attach to a session: $0 attach <session-name>"
}

# Main command dispatcher
main() {
    check_tmux
    
    case "${1:-}" in
        create)
            create_sessions "$@"
            ;;
        status)
            show_status
            ;;
        list)
            list_sessions
            ;;
        attach)
            if [ -z "${2:-}" ]; then
                echo "Error: Session name required"
                echo "Usage: $0 attach <session-name>"
                echo "Available sessions: $SESSION_NO_RESYNTH, $SESSION_FULL"
                exit 1
            fi
            attach_session "$2"
            ;;
        kill)
            if [ -z "${2:-}" ]; then
                echo "Error: Session name required"
                echo "Usage: $0 kill <session-name>"
                echo "Available sessions: $SESSION_NO_RESYNTH, $SESSION_FULL"
                exit 1
            fi
            kill_session "$2"
            ;;
        kill-all)
            kill_all_sessions
            ;;
        *)
            echo "Circuit Benchmark Tmux Session Manager"
            echo "======================================"
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  create [flags]  Create experiment sessions (default: full session only)"
            echo "  status          Show status of all sessions"
            echo "  list            List active sessions"
            echo "  attach <name>   Attach to a session (detach with Ctrl+B then D)"
            echo "  kill <name>     Kill a specific session"
            echo "  kill-all        Kill all sessions"
            echo ""
            echo "Create flags:"
            echo "  --skip-no-resynth  Skip the no-resynth session"
            echo "  --skip-full        Skip the full session"
            echo "  --only-no-resynth  Only create the no-resynth session"
            echo "  --only-full        Only create the full session"
            echo "  --force            Kill existing sessions before creating (recreate)"
            echo ""
            echo "Session names:"
            echo "  - $SESSION_NO_RESYNTH: Without resynthesis (faster)"
            echo "  - $SESSION_FULL: With all optimizers (slower, more thorough)"
            echo ""
            echo "Examples:"
            echo "  $0 create                           # Create full session (default)"
            echo "  $0 create --skip-full               # Create no-resynth session instead"
            echo "  $0 create --only-no-resynth         # Create only no-resynth session"
            echo "  $0 create --only-full               # Create only full session (same as default)"
            echo "  $0 create --force                   # Recreate session (kill existing first)"
            echo "  $0 attach $SESSION_NO_RESYNTH       # Attach to a session"
            echo "  $0 status                           # Check status"
            exit 1
            ;;
    esac
}

main "$@"

