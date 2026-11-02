#!/usr/bin/env bash
#
# Tmux session manager for running single-step grid search in the background
#
# This script creates and manages a tmux session for running single-step grid search
# experiments. Sessions persist across SSH disconnections, allowing you to start
# experiments, disconnect, and reconnect later to check progress.
#
# Resource Limiting:
#   The benchmark session uses resource limiting via run_wisq_safe.sh:
#   - BQSKit workers limited to 50% of CPU cores (keeps system responsive)
#   - Process runs at nice level 15 (lower CPU priority)
#   - Process runs with idle I/O priority (ionice class 3)
#
#   Override with environment variables:
#     BQSKIT_NUM_WORKERS=8 ./scripts/run_single_step_grid_search_tmux.sh create
#     BQSKIT_WORKER_FRACTION=0.25 ./scripts/run_single_step_grid_search_tmux.sh create
#
# Usage:
#   ./scripts/run_single_step_grid_search_tmux.sh create           # Create and start search
#   ./scripts/run_single_step_grid_search_tmux.sh create --attach  # Create and attach immediately
#   ./scripts/run_single_step_grid_search_tmux.sh create --force   # Recreate (kill existing first)
#   ./scripts/run_single_step_grid_search_tmux.sh create --rerun   # Rerun all optimizers (keeps history)
#   ./scripts/run_single_step_grid_search_tmux.sh create --rerun-optimizers wisq_rules wisq_bqskit
#   ./scripts/run_single_step_grid_search_tmux.sh status           # Check session status
#   ./scripts/run_single_step_grid_search_tmux.sh attach           # Attach to session
#   ./scripts/run_single_step_grid_search_tmux.sh kill             # Kill session
#   ./scripts/run_single_step_grid_search_tmux.sh logs             # Tail the log file

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Session name
SESSION_NAME="single-step-grid-search"

# Output directories
OUTPUT_DIR="$REPO_ROOT/reports/single_step_search"
ARTIFACT_DIR="$REPO_ROOT/data/artifacts"

# Log directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Database path
DATABASE_PATH="$REPO_ROOT/data/trajectories.db"

# Metadata path for circuit import
METADATA_PATH="$REPO_ROOT/benchmarks/ai_transpile/metadata.json"

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
    local rerun=0
    local rerun_optimizers=""
    
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
            --rerun)
                rerun=1
                shift
                ;;
            --rerun-optimizers)
                shift
                # Collect all following args until next flag or end
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    rerun_optimizers="$rerun_optimizers $1"
                    shift
                done
                ;;
            *)
                echo "Error: Unknown flag '$1'"
                echo "Available flags: --force, --attach, --rerun, --rerun-optimizers <names...>"
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
    echo "  Artifacts: $ARTIFACT_DIR"
    echo "  Log file: $log_file"
    
    # Build the command using the resource-limited wrapper
    local safe_wrapper="$REPO_ROOT/scripts/run_wisq_safe.sh"
    local cmd="$safe_wrapper uv run python scripts/run_single_step_grid_search.py"
    cmd="$cmd --database '$DATABASE_PATH'"
    cmd="$cmd --import-metadata '$METADATA_PATH'"
    cmd="$cmd --artifact-dir '$ARTIFACT_DIR'"
    cmd="$cmd --resume"
    
    # Add rerun flags if specified
    if [ $rerun -eq 1 ]; then
        cmd="$cmd --rerun"
    elif [ -n "$rerun_optimizers" ]; then
        cmd="$cmd --rerun-optimizers$rerun_optimizers"
    fi
    
    # Create directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$ARTIFACT_DIR"
    mkdir -p "$(dirname "$DATABASE_PATH")"
    
    # Create detached tmux session
    # Note: We use tmux's built-in logging instead of tee to avoid breaking progress bars
    # PYTHONWARNINGS suppresses BQSKit's SmallSampleWarning which breaks Rich progress bars
    # Format: action:message:category:module:line
    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
        "export PYTHONWARNINGS='ignore:One or more sample arguments is too small'; \
         echo '=== Single-Step Grid Search ==='; \
         echo 'Session: $SESSION_NAME'; \
         echo 'Started: $(date)'; \
         echo 'Database: $DATABASE_PATH'; \
         echo 'Artifacts: $ARTIFACT_DIR'; \
         echo 'Resource limiting: nice=15, ionice=idle, workers=50%'; \
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
    echo "Single-Step Grid Search Status"
    echo "==============================="
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
            echo "Single-Step Grid Search Tmux Session Manager"
            echo "============================================="
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  create [flags]    Create and start the grid search session"
            echo "                      --attach, -a            Attach to session after creating"
            echo "                      --force                 Kill existing session first"
            echo "                      --rerun                 Rerun all optimizers (keeps history)"
            echo "                      --rerun-optimizers ...  Rerun specific optimizers (keeps history)"
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
            echo "  $0 create --rerun --attach                     # Rerun all, keeping history"
            echo "  $0 create --rerun-optimizers wisq_rules --attach  # Rerun specific optimizer"
            echo "  $0 attach                                      # View live output"
            echo "  $0 status                                      # Quick status check"
            echo "  $0 logs                                        # Follow log file"
            echo ""
            echo "The search will:"
            echo "  - Import circuits from: $METADATA_PATH"
            echo "  - Store results in: $DATABASE_PATH"
            echo "  - Save artifacts to: $ARTIFACT_DIR"
            echo "  - Log to: $LOG_DIR/"
            echo ""
            echo "Session persists across SSH disconnections."
            exit 1
            ;;
    esac
}

main "$@"
