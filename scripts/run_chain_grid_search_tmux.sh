#!/usr/bin/env bash
#
# Tmux session manager for running chain grid search (step-2 from artifacts)
#
# This script creates and manages a tmux session for running multi-step
# optimizer chains starting from step-1 optimized artifacts.
#
# Usage:
#   ./scripts/run_chain_grid_search_tmux.sh create
#   ./scripts/run_chain_grid_search_tmux.sh create --attach
#   ./scripts/run_chain_grid_search_tmux.sh create --force
#   ./scripts/run_chain_grid_search_tmux.sh create -- --max-chain-length 2
#   ./scripts/run_chain_grid_search_tmux.sh status
#   ./scripts/run_chain_grid_search_tmux.sh attach
#   ./scripts/run_chain_grid_search_tmux.sh kill
#   ./scripts/run_chain_grid_search_tmux.sh logs
#
# Override defaults with environment variables:
#   DATABASE_PATH, ARTIFACT_INPUT_DIR, ARTIFACT_CATEGORY,
#   MAX_CHAIN_LENGTH, MAX_QUBITS, TIME_BUDGET

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Session name
SESSION_NAME="chain-grid-search-step2"

# Output directories
OUTPUT_DIR="$REPO_ROOT/reports/chain_grid_search"

# Log directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Inputs and configuration (overridable)
DATABASE_PATH="${DATABASE_PATH:-$REPO_ROOT/data/trajectories_step2.db}"
ARTIFACT_INPUT_DIR="${ARTIFACT_INPUT_DIR:-$REPO_ROOT/data/artifacts}"
ARTIFACT_CATEGORY="${ARTIFACT_CATEGORY:-artifact_step1}"
MAX_CHAIN_LENGTH="${MAX_CHAIN_LENGTH:-1}"
MAX_QUBITS="${MAX_QUBITS:-20}"
TIME_BUDGET="${TIME_BUDGET:-300}"

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
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Function to create the session
create_session() {
    local force=0
    local attach=0
    local extra_args=()

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force)
                force=1
                shift
                ;;
            --attach|-a)
                attach=1
                shift
                ;;
            --)
                shift
                extra_args=("$@")
                break
                ;;
            *)
                echo "Error: Unknown flag '$1'"
                echo "Available flags: --force, --attach, --"
                exit 1
                ;;
        esac
    done

    # Kill existing session if --force
    if [ $force -eq 1 ] && session_exists; then
        echo "Force flag set: killing existing session '$SESSION_NAME'"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        sleep 1
    fi

    if session_exists; then
        echo "Session '$SESSION_NAME' already exists."
        echo "Use 'attach' to connect, or 'kill' to remove it first."
        echo "Or use 'create --force' to recreate it."
        return 1
    fi

    local timestamp=$(date +%Y%m%d-%H%M%S)
    local log_file="$LOG_DIR/${SESSION_NAME}-${timestamp}.log"
    local latest_symlink="$LOG_DIR/${SESSION_NAME}-latest.log"

    echo "Creating session: $SESSION_NAME"
    echo "  Database: $DATABASE_PATH"
    echo "  Artifact input: $ARTIFACT_INPUT_DIR"
    echo "  Artifact category: $ARTIFACT_CATEGORY"
    echo "  Max chain length: $MAX_CHAIN_LENGTH"
    echo "  Max qubits: $MAX_QUBITS"
    echo "  Time budget: $TIME_BUDGET"
    echo "  Log file: $log_file"

    # Build the command using the resource-limited wrapper
    local safe_wrapper="$REPO_ROOT/scripts/run_wisq_safe.sh"
    local cmd="$safe_wrapper uv run python scripts/run_grid_search.py"
    cmd="$cmd --database '$DATABASE_PATH'"
    cmd="$cmd --import-artifacts '$ARTIFACT_INPUT_DIR'"
    cmd="$cmd --artifact-category '$ARTIFACT_CATEGORY'"
    cmd="$cmd --max-chain-length '$MAX_CHAIN_LENGTH'"
    cmd="$cmd --max-qubits '$MAX_QUBITS'"
    cmd="$cmd --time-budget '$TIME_BUDGET'"
    cmd="$cmd --resume"

    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd="$cmd ${extra_args[*]}"
    fi

    # Create directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$DATABASE_PATH")"

    # Create detached tmux session
    # Note: We use tmux's built-in logging instead of tee to avoid breaking progress bars
    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
        "echo '=== Chain Grid Search (Step-2) ==='; \
         echo 'Session: $SESSION_NAME'; \
         echo 'Started: $(date)'; \
         echo 'Database: $DATABASE_PATH'; \
         echo 'Artifact input: $ARTIFACT_INPUT_DIR'; \
         echo 'Artifact category: $ARTIFACT_CATEGORY'; \
         echo 'Max chain length: $MAX_CHAIN_LENGTH'; \
         echo 'Max qubits: $MAX_QUBITS'; \
         echo 'Time budget: $TIME_BUDGET'; \
         echo 'Log file: $log_file'; \
         echo ''; \
         $cmd 2>&1; \
         exit_code=\$?; \
         echo ''; \
         echo 'Completed: $(date)'; \
         echo 'Exit code: '\$exit_code; \
         echo 'Press Enter to close this window...'; \
         read"

    # Enable tmux logging for this session
    tmux pipe-pane -t "$SESSION_NAME" -o "cat >> '$log_file'"

    # Create symlink to latest log file
    ln -sf "$(basename "$log_file")" "$latest_symlink"

    echo ""
    echo "Session '$SESSION_NAME' created successfully!"

    # Attach if requested
    if [ $attach -eq 1 ]; then
        echo "Attaching to session..."
        echo "Press Ctrl+B then D to detach without killing the session"
        echo ""
        sleep 1
        tmux attach-session -t "$SESSION_NAME"
    else
        echo ""
        echo "Commands:"
        echo "  Attach:    $0 attach"
        echo "  Status:    $0 status"
        echo "  Logs:      $0 logs"
        echo "  Kill:      $0 kill"
        echo ""
        echo "Detach from session: Press Ctrl+B then D"
    fi
}

# Function to show status
show_status() {
    echo "Chain Grid Search Status"
    echo "========================="
    echo ""

    if session_exists; then
        local log_file="$LOG_DIR/${SESSION_NAME}-latest.log"
        echo "Session: $SESSION_NAME"
        echo "Status: RUNNING"
        echo "Log file: $log_file"

        if [ -f "$log_file" ]; then
            echo ""
            echo "Last 5 lines of log:"
            tail -n 5 "$log_file" 2>/dev/null | sed 's/^/  /'
        fi

        # Show database stats if exists
        if [ -f "$DATABASE_PATH" ]; then
            echo ""
            echo "Database statistics:"
            uv run python scripts/inspect_db.py --database "$DATABASE_PATH" --tables 2>/dev/null | sed 's/^/  /' || true
        fi
    else
        echo "Session: $SESSION_NAME"
        echo "Status: NOT RUNNING"
        echo ""
        echo "Create session with: $0 create"
    fi
}

# Function to attach to session
attach_session() {
    if ! session_exists; then
        echo "Error: Session '$SESSION_NAME' does not exist"
        echo "Create it with: $0 create"
        exit 1
    fi

    echo "Attaching to session: $SESSION_NAME"
    echo "Press Ctrl+B then D to detach without killing the session"
    echo ""
    sleep 1
    tmux attach-session -t "$SESSION_NAME"
}

# Function to kill session
kill_session() {
    if ! session_exists; then
        echo "Error: Session '$SESSION_NAME' does not exist"
        exit 1
    fi

    echo "Killing session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    echo "Session killed"
}

# Function to show logs
show_logs() {
    local log_file="$LOG_DIR/${SESSION_NAME}-latest.log"

    if [ ! -f "$log_file" ]; then
        echo "Log file not found: $log_file"
        echo "Session may not have been started yet."
        exit 1
    fi

    echo "Tailing log file: $log_file"
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f "$log_file"
}

# Function to show database info
show_db() {
    if [ ! -f "$DATABASE_PATH" ]; then
        echo "Database not found: $DATABASE_PATH"
        echo "Run 'create' to initialize and start the search."
        exit 1
    fi

    echo "Database: $DATABASE_PATH"
    echo ""
    uv run python scripts/inspect_db.py --database "$DATABASE_PATH" --tables --optimizers
}

# Main command dispatcher
main() {
    check_tmux

    case "${1:-}" in
        create)
            shift
            create_session "$@"
            ;;
        status)
            show_status
            ;;
        attach)
            attach_session
            ;;
        kill)
            kill_session
            ;;
        logs)
            show_logs
            ;;
        db)
            show_db
            ;;
        *)
            echo "Chain Grid Search Tmux Session Manager"
            echo "======================================"
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  create [flags]    Create and start the chain grid search session"
            echo "                      --attach, -a            Attach to session after creating"
            echo "                      --force                 Kill existing session first"
            echo "                      --                      Pass remaining args to run_grid_search.py"
            echo "  status            Show status of the session"
            echo "  attach            Attach to the session (detach with Ctrl+B D)"
            echo "  kill              Kill the session"
            echo "  logs              Tail the log file"
            echo "  db                Show database statistics"
            echo ""
            echo "Examples:"
            echo "  $0 create                                      # Start new search (detached)"
            echo "  $0 create --attach                             # Start and attach immediately"
            echo "  $0 create --force                              # Restart (kills existing session)"
            echo "  $0 create -- --max-chain-length 2              # Run longer chains"
            echo "  $0 attach                                      # View live output"
            echo "  $0 status                                      # Quick status check"
            echo "  $0 logs                                        # Follow log file"
            echo ""
            echo "The search will:"
            echo "  - Import artifacts from: $ARTIFACT_INPUT_DIR"
            echo "  - Store results in: $DATABASE_PATH"
            echo "  - Log to: $LOG_DIR/"
            echo ""
            echo "Session persists across SSH disconnections."
            exit 1
            ;;
    esac
}

main "$@"
