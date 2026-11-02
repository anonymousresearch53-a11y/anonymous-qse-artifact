#!/usr/bin/env bash
#
# Tmux session manager for running the conservative online RL pilot overnight.
#
# Workflow:
#   1. Baseline evaluation on original circuits
#   2. Conservative online rollout collection into a separate DB
#   3. Mixed offline+online fine-tuning from the provided checkpoint
#   4. Post-fine-tune evaluation on original circuits
#
# Usage:
#   ./scripts/run_online_rl_pilot_tmux.sh create --checkpoint data/rl_checkpoints/<ckpt>
#   ./scripts/run_online_rl_pilot_tmux.sh create --checkpoint ... --attach
#   ./scripts/run_online_rl_pilot_tmux.sh create --checkpoint ... --force
#   ./scripts/run_online_rl_pilot_tmux.sh create --checkpoint ... --start-from finetune
#   ./scripts/run_online_rl_pilot_tmux.sh status
#   ./scripts/run_online_rl_pilot_tmux.sh attach
#   ./scripts/run_online_rl_pilot_tmux.sh logs
#   ./scripts/run_online_rl_pilot_tmux.sh kill
#
# Environment overrides:
#   SESSION_NAME, CONFIG_PATH, OFFLINE_DB, ONLINE_DB, RESULTS_DIR,
#   CHECKPOINT_OUTPUT_DIR, ROLLOUT_LIMIT, ROLLOUT_STEPS, ROLLOUT_CIRCUITS,
#   EVAL_MAX_CIRCUITS, EVAL_ROLLOUT_STEPS, FINE_TUNE_EPOCHS,
#   ONLINE_MIX_WEIGHT, BQSKIT_NUM_WORKERS, BQSKIT_WORKER_FRACTION

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_NAME="${SESSION_NAME:-online-rl-pilot}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/cql_online_pilot.yaml}"
OFFLINE_DB="${OFFLINE_DB:-$REPO_ROOT/data/trajectories_combined.db}"
ONLINE_DB="${ONLINE_DB:-$REPO_ROOT/data/trajectories_online.db}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/online_rl_pilot}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-$REPO_ROOT/data/rl_checkpoints}"
REPORTS_DIR="$REPO_ROOT/reports/online_rl_pilot"
LOG_DIR="$REPORTS_DIR/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-3}"
ROLLOUT_CIRCUITS="${ROLLOUT_CIRCUITS:-original}"
EVAL_MAX_CIRCUITS="${EVAL_MAX_CIRCUITS:-50}"
EVAL_ROLLOUT_STEPS="${EVAL_ROLLOUT_STEPS:-3}"
FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-50}"
ONLINE_MIX_WEIGHT="${ONLINE_MIX_WEIGHT:-4}"
EVAL_CIRCUITS="${EVAL_CIRCUITS:-original}"

check_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is not installed or not in PATH"
        exit 1
    fi
}

session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

usage() {
    sed -n '2,26p' "$0"
}

is_valid_checkpoint_dir() {
    local path="$1"
    [[ -d "$path" && -f "$path/model.pt" && -f "$path/config.yaml" && -f "$path/normalization.json" ]]
}

resolve_checkpoint_dir() {
    local input_path="$1"
    local abs_path
    abs_path="$(cd "$(dirname "$input_path")" && pwd)/$(basename "$input_path")"

    if is_valid_checkpoint_dir "$abs_path"; then
        printf '%s\n' "$abs_path"
        return 0
    fi

    if [[ -L "$abs_path" && -d "$abs_path" ]] && is_valid_checkpoint_dir "$abs_path"; then
        printf '%s\n' "$abs_path"
        return 0
    fi

    if [[ -d "$abs_path" ]]; then
        local candidate=""
        for link_name in cql_latest latest bc_latest iql_latest dt_latest; do
            if [[ -e "$abs_path/$link_name" ]] && is_valid_checkpoint_dir "$abs_path/$link_name"; then
                candidate="$abs_path/$link_name"
                break
            fi
        done
        if [[ -n "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi

        candidate="$(find "$abs_path" -mindepth 1 -maxdepth 1 -type d \
            -exec test -f '{}/model.pt' -a -f '{}/config.yaml' -a -f '{}/normalization.json' ';' \
            -printf '%f\t%p\n' | \
            awk -F'\t' 'match($1, /_([0-9]+)$/, parts) { print parts[1] "\t" $2 }' | \
            sort -n | tail -n 1 | cut -f2-)"
        if [[ -z "$candidate" ]]; then
            candidate="$(find "$abs_path" -mindepth 1 -maxdepth 1 -type d \
                -exec test -f '{}/model.pt' -a -f '{}/config.yaml' -a -f '{}/normalization.json' ';' \
                -print | xargs -r ls -td 2>/dev/null | head -n 1)"
        fi
        if [[ -n "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi

        local parent_dir
        parent_dir="$(dirname "$abs_path")"
        for link_name in cql_latest latest bc_latest iql_latest dt_latest; do
            if [[ -e "$parent_dir/$link_name" ]] && is_valid_checkpoint_dir "$parent_dir/$link_name"; then
                printf '%s\n' "$parent_dir/$link_name"
                return 0
            fi
        done
    fi

    return 1
}

create_session() {
    local force=0
    local attach=0
    local checkpoint=""
    local requested_checkpoint=""
    local start_from="baseline"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint)
                checkpoint="$2"
                shift 2
                ;;
            --start-from)
                start_from="$2"
                shift 2
                ;;
            --force)
                force=1
                shift
                ;;
            --attach|-a)
                attach=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Error: Unknown flag '$1'"
                echo "Required: --checkpoint <path>"
                exit 1
                ;;
        esac
    done

    if [[ -z "$checkpoint" ]]; then
        echo "Error: --checkpoint is required"
        exit 1
    fi
    case "$start_from" in
        baseline|rollout|finetune|posteval)
            ;;
        *)
            echo "Error: --start-from must be one of: baseline, rollout, finetune, posteval"
            exit 1
            ;;
    esac

    requested_checkpoint="$checkpoint"
    if ! checkpoint="$(resolve_checkpoint_dir "$checkpoint")"; then
        local requested
        requested="$(cd "$(dirname "$requested_checkpoint")" 2>/dev/null && pwd)/$(basename "$requested_checkpoint")"
        echo "Error: could not resolve a valid checkpoint directory from: ${requested:-$requested_checkpoint}"
        echo "Expected one of:"
        echo "  1. A checkpoint directory containing model.pt/config.yaml/normalization.json"
        echo "  2. A symlink to such a directory"
        echo "  3. A parent directory containing latest or *_latest symlinks"
        echo "  4. A parent directory containing timestamped checkpoint subdirectories"
        exit 1
    fi

    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Error: config file not found: $CONFIG_PATH"
        exit 1
    fi
    if [[ ! -f "$OFFLINE_DB" ]]; then
        echo "Error: offline DB not found: $OFFLINE_DB"
        exit 1
    fi

    if [[ $force -eq 1 ]] && session_exists; then
        echo "Force flag set: killing existing session '$SESSION_NAME'"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        sleep 1
    fi

    if session_exists; then
        echo "Session '$SESSION_NAME' already exists."
        echo "Use '$0 attach' or '$0 kill'."
        return 1
    fi

    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    local log_file="$LOG_DIR/${SESSION_NAME}-${timestamp}.log"
    local latest_symlink="$LOG_DIR/${SESSION_NAME}-latest.log"
    local safe_wrapper="$REPO_ROOT/scripts/run_wisq_safe.sh"

    echo "Creating session: $SESSION_NAME"
    echo "  Checkpoint:      $checkpoint"
    echo "  Config:          $CONFIG_PATH"
    echo "  Offline DB:      $OFFLINE_DB"
    echo "  Online DB:       $ONLINE_DB"
    echo "  Results dir:     $RESULTS_DIR"
    echo "  Rollout limit:   $ROLLOUT_LIMIT"
    echo "  Rollout steps:   $ROLLOUT_STEPS"
    echo "  Fine-tune epochs:$FINE_TUNE_EPOCHS"
    echo "  Start from:      $start_from"
    echo "  Log file:        $log_file"

    mkdir -p "$REPORTS_DIR" "$RESULTS_DIR" "$CHECKPOINT_OUTPUT_DIR"

    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
        "set -euo pipefail; \
         export PYTHONWARNINGS='ignore:One or more sample arguments is too small'; \
         CHECKPOINT='$checkpoint'; \
         CONFIG_PATH='$CONFIG_PATH'; \
         OFFLINE_DB='$OFFLINE_DB'; \
         ONLINE_DB='$ONLINE_DB'; \
         RESULTS_DIR='$RESULTS_DIR'; \
         CHECKPOINT_OUTPUT_DIR='$CHECKPOINT_OUTPUT_DIR'; \
         SAFE_WRAPPER='$safe_wrapper'; \
         START_FROM='$start_from'; \
         PRE_EVAL_JSON='${RESULTS_DIR}/${timestamp}-pre-eval.json'; \
         ROLLOUT_JSON='${RESULTS_DIR}/${timestamp}-rollout.json'; \
         POST_EVAL_JSON='${RESULTS_DIR}/${timestamp}-post-eval.json'; \
         echo '=== Online RL Pilot ==='; \
         echo 'Session: $SESSION_NAME'; \
         echo 'Started: $(date)'; \
         echo 'Checkpoint: '$checkpoint; \
         echo 'Offline DB: '$OFFLINE_DB; \
         echo 'Online DB: '$ONLINE_DB; \
         echo 'Results dir: '$RESULTS_DIR; \
         echo 'Start from: '$start_from; \
         echo 'Log file: $log_file'; \
         echo ''; \
         if [[ \"$start_from\" == 'baseline' ]]; then \
             echo '--- [1/4] Baseline evaluation ---'; \
             \"$safe_wrapper\" uv run python scripts/evaluate_policy.py \
                 --checkpoint \"$checkpoint\" \
                 --database \"$OFFLINE_DB\" \
                 --split test \
                 --circuits \"$EVAL_CIRCUITS\" \
                 --online \
                 --rollout-steps \"$EVAL_ROLLOUT_STEPS\" \
                 --online-max-circuits \"$EVAL_MAX_CIRCUITS\" \
                 --output \"$RESULTS_DIR/${timestamp}-pre-eval.json\"; \
             echo ''; \
         fi; \
         if [[ \"$start_from\" == 'baseline' || \"$start_from\" == 'rollout' ]]; then \
             echo '--- [2/4] Online rollout collection ---'; \
             \"$safe_wrapper\" uv run python scripts/rollout_policy.py \
                 --checkpoint \"$checkpoint\" \
                 --database \"$OFFLINE_DB\" \
                 --output-db \"$ONLINE_DB\" \
                 --circuits \"$ROLLOUT_CIRCUITS\" \
                 --limit \"$ROLLOUT_LIMIT\" \
                 --max-steps \"$ROLLOUT_STEPS\" \
                 --output \"$RESULTS_DIR/${timestamp}-rollout.json\"; \
             echo ''; \
         fi; \
         if [[ \"$start_from\" == 'baseline' || \"$start_from\" == 'rollout' || \"$start_from\" == 'finetune' ]]; then \
             if [[ ! -f \"$ONLINE_DB\" ]]; then \
                 echo 'Error: online DB not found for fine-tuning: '$ONLINE_DB; \
                 exit 1; \
             fi; \
             echo '--- [3/4] Mixed replay fine-tuning ---'; \
             uv run python scripts/fine_tune_online_rl.py \
                 --checkpoint \"$checkpoint\" \
                 --offline-db \"$OFFLINE_DB\" \
                 --online-db \"$ONLINE_DB\" \
                 --output-dir \"$CHECKPOINT_OUTPUT_DIR\" \
                 --num-epochs \"$FINE_TUNE_EPOCHS\" \
                 --online-mix-weight \"$ONLINE_MIX_WEIGHT\" \
                 --eval-circuits \"$EVAL_CIRCUITS\"; \
         fi; \
         FINE_TUNED_CHECKPOINT=''; \
         if [[ \"$start_from\" == 'baseline' || \"$start_from\" == 'rollout' || \"$start_from\" == 'finetune' ]]; then \
             FINE_TUNED_CHECKPOINT=\$(ls -td \"$CHECKPOINT_OUTPUT_DIR\"/*_online_ft_* 2>/dev/null | head -n 1); \
         else \
             FINE_TUNED_CHECKPOINT=\$(ls -td \"$CHECKPOINT_OUTPUT_DIR\"/*_online_ft_* 2>/dev/null | head -n 1); \
         fi; \
         if [[ -z \"\$FINE_TUNED_CHECKPOINT\" ]]; then \
             echo 'Error: no fine-tuned checkpoint found in '$CHECKPOINT_OUTPUT_DIR; \
             exit 1; \
         fi; \
         echo 'Fine-tuned checkpoint: '\$FINE_TUNED_CHECKPOINT; \
         echo ''; \
         echo '--- [4/4] Post fine-tune evaluation ---'; \
         \"$safe_wrapper\" uv run python scripts/evaluate_policy.py \
             --checkpoint \"\$FINE_TUNED_CHECKPOINT\" \
             --database \"$OFFLINE_DB\" \
             --split test \
             --circuits \"$EVAL_CIRCUITS\" \
             --online \
             --rollout-steps \"$EVAL_ROLLOUT_STEPS\" \
             --online-max-circuits \"$EVAL_MAX_CIRCUITS\" \
             --output \"$RESULTS_DIR/${timestamp}-post-eval.json\"; \
         exit_code=\$?; \
         echo ''; \
         echo 'Completed: $(date)'; \
         echo 'Exit code: '\$exit_code; \
         echo 'Press Enter to close this window...'; \
         read"

    tmux pipe-pane -t "$SESSION_NAME" -o "cat >> '$log_file'"
    ln -sf "$(basename "$log_file")" "$latest_symlink"

    echo
    echo "Session '$SESSION_NAME' created successfully."
    if [[ $attach -eq 1 ]]; then
        echo "Attaching..."
        sleep 1
        tmux attach-session -t "$SESSION_NAME"
    else
        echo "Commands:"
        echo "  Attach: $0 attach"
        echo "  Status: $0 status"
        echo "  Logs:   $0 logs"
        echo "  Kill:   $0 kill"
    fi
}

show_status() {
    echo "Online RL Pilot Status"
    echo "======================"
    echo
    if session_exists; then
        local log_file="$LOG_DIR/${SESSION_NAME}-latest.log"
        echo "Session: $SESSION_NAME"
        echo "Status: RUNNING"
        echo "Log file: $log_file"
        if [[ -f "$log_file" ]]; then
            echo
            echo "Last 10 lines of log:"
            tail -n 10 "$log_file" | sed 's/^/  /'
        fi
    else
        echo "Session: $SESSION_NAME"
        echo "Status: NOT RUNNING"
        echo
        echo "Create session with: $0 create --checkpoint <path>"
    fi
}

attach_session() {
    if ! session_exists; then
        echo "Error: Session '$SESSION_NAME' does not exist"
        exit 1
    fi
    tmux attach-session -t "$SESSION_NAME"
}

kill_session() {
    if ! session_exists; then
        echo "Session '$SESSION_NAME' does not exist"
        return 0
    fi
    tmux kill-session -t "$SESSION_NAME"
    echo "Killed session '$SESSION_NAME'"
}

show_logs() {
    local log_file="$LOG_DIR/${SESSION_NAME}-latest.log"
    if [[ ! -f "$log_file" ]]; then
        echo "No log file found: $log_file"
        exit 1
    fi
    tail -f "$log_file"
}

main() {
    check_tmux
    local command="${1:-}"
    case "$command" in
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
        --help|-h|help|"")
            usage
            ;;
        *)
            echo "Error: Unknown command '$command'"
            usage
            exit 1
            ;;
    esac
}

main "$@"
