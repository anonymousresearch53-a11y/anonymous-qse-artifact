#!/usr/bin/env bash
set -euo pipefail

REPLICATES="${1:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${JAVA_HOME:-}" ]]; then
  if [[ -d "/usr/lib/jvm/java-21-openjdk-amd64" ]]; then
    export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
  elif [[ -d "$HOME/.local/share/java/jdk-21.0.5+11" ]]; then
    export JAVA_HOME="$HOME/.local/share/java/jdk-21.0.5+11"
  fi
fi
if [[ -n "${JAVA_HOME:-}" ]]; then
  export PATH="$JAVA_HOME/bin:$PATH"
fi

mkdir -p data/confirmatory logs/confirmatory

FAST_CONCURRENCY="${FAST_CONCURRENCY:-4}"
WISQ_RULES_CONCURRENCY="${WISQ_RULES_CONCURRENCY:-2}"
WISQ_BQSKIT_CONCURRENCY="${WISQ_BQSKIT_CONCURRENCY:-1}"
MAX_QUBITS="${MAX_QUBITS:-100}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-full_unmapped}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_sql_summary() {
  local db="$1"
  uv run python scripts/inspect_db.py --database "$db" --query "SELECT source, category, COUNT(*) FROM circuits GROUP BY source, category ORDER BY 3 DESC" || true
  uv run python scripts/inspect_db.py --database "$db" --query "SELECT o.name, COUNT(*) AS runs FROM optimization_runs r JOIN optimizers o ON r.optimizer_id = o.id GROUP BY o.name ORDER BY o.name" || true
  uv run python scripts/inspect_db.py --database "$db" --query "SELECT o.name, COUNT(*) AS failed FROM optimization_runs r JOIN optimizers o ON r.optimizer_id = o.id WHERE r.success = 0 GROUP BY o.name ORDER BY o.name" || true
}

for rep in $(seq 1 "$REPLICATES"); do
  DB="data/confirmatory/${RUN_NAME_PREFIX}_r${rep}.db"
  RUNNER_LOG="logs/confirmatory/${RUN_NAME_PREFIX}_r${rep}.runner.log"
  STDOUT_LOG="logs/confirmatory/${RUN_NAME_PREFIX}_r${rep}.stdout.log"

  log "=== Starting replicate ${rep}/${REPLICATES} ==="
  log "Database: $DB"
  log "Runner log: $RUNNER_LOG"
  log "Stdout log: $STDOUT_LOG"

  if [[ -e "$DB" ]]; then
    if [[ "$FORCE_CLEAN" == "1" ]]; then
      log "Removing existing DB for clean rerun: $DB"
      rm -f "$DB"
    else
      log "DB already exists: $DB"
      log "Refusing to continue without FORCE_CLEAN=1 to avoid mixing runs."
      exit 1
    fi
  fi

  : > "$STDOUT_LOG"

  {
    log "Importing GUOQ original circuits"
    uv run python scripts/import_guoq_circuits.py \
      --source benchmarks/ai_transpile/qasm/guoq_ibmnew \
      --database "$DB" \
      --category guoq_ibmnew

    log "Importing Benchpress preprocessed circuits"
    uv run python scripts/import_benchpress_preprocessed.py \
      --database "$DB" \
      --output-dir benchmarks/ai_transpile/qasm/benchpress_ibmn

    log "Launching clean optimization-only single-step rerun"
    CMD=(uv run python scripts/run_single_step_grid_search.py
      --database "$DB"
      --sources guoq benchpress
      --resume
      --no-artifacts
      --max-concurrent-fast "$FAST_CONCURRENCY"
      --max-concurrent-wisq-rules "$WISQ_RULES_CONCURRENCY"
      --max-concurrent-wisq-bqskit "$WISQ_BQSKIT_CONCURRENCY"
      --log-file "$RUNNER_LOG")

    if [[ -n "$MAX_QUBITS" ]]; then
      CMD+=(--max-qubits "$MAX_QUBITS")
    fi

    "${CMD[@]}"

    log "Replicate ${rep} complete. Summary follows."
    run_sql_summary "$DB"
  } 2>&1 | tee -a "$STDOUT_LOG"

done

log "All confirmatory replicates completed successfully."
