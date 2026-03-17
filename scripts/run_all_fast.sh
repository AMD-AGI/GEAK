#!/bin/bash
# Run ALL ablation experiments sequentially with isolated output.
# Uses geak --kernel-url directly (not geak-orchestrate).
#
# Usage: nohup bash scripts/run_all_fast.sh &> /home/sapmajum/fast_experiments.log &
#   Or:  bash scripts/run_all_fast.sh          (foreground)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/home/sapmajum/fast_experiments.log"

log() { echo "[$(date -u '+%H:%M:%S')] $1" | tee -a "$LOG"; }

EXPERIMENTS=(exp0 exp2 exp3 exp4 exp8 exp_lancedb)

wait_for() {
    local exp="$1"
    local container="geak-fast-$exp"
    log "Waiting for $exp..."
    while true; do
        if grep -q "^COMPLETE:" "/home/sapmajum/geak-experiments/$exp/logs/master.log" 2>/dev/null; then
            log "$exp COMPLETED"
            cat "/home/sapmajum/geak-experiments/$exp/logs/master.log" | tee -a "$LOG"
            return 0
        fi
        if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            log "$exp container stopped unexpectedly"
            cat "/home/sapmajum/geak-experiments/$exp/logs/master.log" 2>/dev/null | tee -a "$LOG" || true
            return 1
        fi
        sleep 60
    done
}

log "=========================================="
log "GEAK Fast Ablation Suite (11 experiments)"
log "=========================================="

for exp in "${EXPERIMENTS[@]}"; do
    if grep -q "^COMPLETE:" "/home/sapmajum/geak-experiments/$exp/logs/master.log" 2>/dev/null; then
        log "SKIP $exp (already complete)"
        continue
    fi

    log "=== LAUNCHING $exp ==="
    bash "$SCRIPT_DIR/run_fast_ablation.sh" "$exp" 2>&1 | tee -a "$LOG"
    wait_for "$exp"

    # Stop container to free resources
    docker rm -f "geak-fast-$exp" 2>/dev/null || true

    # Disk cleanup
    find "/home/sapmajum/geak-experiments/$exp" -name "worktrees" -type d -exec rm -rf {} + 2>/dev/null || true
    find "/home/sapmajum/geak-experiments/$exp" -path "*/.git/objects" -type d -exec rm -rf {} + 2>/dev/null || true
    docker image prune -f 2>/dev/null > /dev/null

    log "--- $exp done, disk: $(df -h / | tail -1 | awk '{print $4}') free ---"
    log ""
done

log "=========================================="
log "ALL EXPERIMENTS COMPLETE"
log "Results: /home/sapmajum/geak-experiments/*/experiment_summary.json"
log "=========================================="
