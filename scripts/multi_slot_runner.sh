#!/bin/bash
# Master experiment sequencer: waits for slot idle, runs next queued kernel.
# Args: slot queue_file
# queue_file format: LEVEL:KERNEL:MEM_MODE  (one per line)
SLOT="$1"
QUEUE_FILE="$2"
LOGDIR="/data/sapmajum/triton_runs"
QLOG="$LOGDIR/master_${SLOT}.log"
AKA="/home/sapmajum/work/repos/AgentKernelArena/tasks/triton2triton/geak_eval"

echo "[$(date -Is)] START master queue for $SLOT, queue=$QUEUE_FILE" >> "$QLOG"

slot_idle() {
    # Return 0 if slot has no active geak/test_kernel_harness
    local busy=$(docker exec "$SLOT" bash -c "ps -ef | grep -cE 'bin/geak -t|test_kernel_harness' 2>/dev/null" 2>/dev/null | grep -v "^$" | tail -1)
    [ -z "$busy" ] || [ "$busy" -le "2" ]
}

ensure_slot_up() {
    local status=$(docker inspect -f '{{.State.Status}}' "$SLOT" 2>/dev/null)
    if [ "$status" != "running" ]; then
        docker start "$SLOT" >> "$QLOG" 2>&1
        sleep 10
    fi
}

run_kernel() {
    local LEVEL="$1"
    local K="$2"
    local MEM_MODE="$3"
    local TS=$(date +"%Y%m%d_%H%M%S")
    local LOGFILE="$LOGDIR/${K}_canonical-rocm700_mem${MEM_MODE}_${TS}.log"

    echo "[$(date -Is)] $SLOT -> $LEVEL/$K mem=$MEM_MODE log=$(basename $LOGFILE)" >> "$QLOG"

    # Mem env — supports ablation modes:
    #   off       : all memory disabled (GEAK_MEMORY_DISABLE=1)
    #   on        : full stack (KB experiences + RAG + injected fast_rms seeds)
    #   on-norag  : KB experiences only, RAG hook disabled (isolates seed contribution)
    local MEM_ENV_ARGS=()
    # GEAK_SAVE_TO_KNOWLEDGE_BASE=1 enables auto-ingestion of run winners
    # into knowledge_base.json via consolidation.py — only set for mem=on
    # variants so we accumulate insights from successful runs without
    # contaminating the baseline mem=off measurements.
    if [ "$MEM_MODE" = "off" ]; then
        MEM_ENV_ARGS=(-e "GEAK_MEMORY_DISABLE=1")
    elif [ "$MEM_MODE" = "on-norag" ]; then
        MEM_ENV_ARGS=(-e "GEAK_USE_KNOWLEDGE_BASE=1" -e "GEAK_RAG_HOOK_DISABLE=1" -e "GEAK_SAVE_TO_KNOWLEDGE_BASE=1")
    else
        MEM_ENV_ARGS=(-e "GEAK_USE_KNOWLEDGE_BASE=1" -e "GEAK_SAVE_TO_KNOWLEDGE_BASE=1")
    fi

    # Backup the previous run's winner patch + final_report into a per-run
    # snapshot dir BEFORE clearing /outputs so we don't lose KB seed material.
    local PREV_SNAP_BASE="/data/sapmajum/triton_runs/winner_snapshots"
    mkdir -p "$PREV_SNAP_BASE"
    local PREV_SNAP="$PREV_SNAP_BASE/${SLOT}_${K}_$(date +%Y%m%d_%H%M%S)"
    docker exec "$SLOT" bash -c "
        if [ -d /workspace/outputs/$K ]; then
            mkdir -p /tmp/wsnap_${K}
            cp -r /workspace/outputs/$K/final_report.json /tmp/wsnap_${K}/ 2>/dev/null
            cp -r /workspace/outputs/$K/geak_agent.log /tmp/wsnap_${K}/ 2>/dev/null
            cp -r /workspace/outputs/$K/baseline_metrics.json /tmp/wsnap_${K}/ 2>/dev/null
            # Best patch path is in final_report.json under 'best_patch'; extract & copy
            BP=\$(python3 -c \"import json,sys; print(json.load(open('/workspace/outputs/$K/final_report.json')).get('best_patch',''))\" 2>/dev/null)
            if [ -n \"\$BP\" ] && [ -f \"\$BP\" ]; then
                cp \"\$BP\" /tmp/wsnap_${K}/best_patch.diff 2>/dev/null
            fi
            true
        fi
    " 2>/dev/null
    docker cp "$SLOT:/tmp/wsnap_${K}" "$PREV_SNAP" 2>/dev/null
    docker exec "$SLOT" rm -rf "/tmp/wsnap_${K}" 2>/dev/null

    # Clear prior outputs + Triton JIT cache + torch extensions cache to avoid
    # cache contamination from previous kernel runs (baselines can drift
    # otherwise: prior-run-optimized code gets reused).
    # Clear BOTH possible output dirs: /outputs (legacy) and /workspace/outputs
    # (where geak actually writes per "outputs/$K" relative path with cwd=/workspace).
    # Otherwise prior-run strategy directories pollute the new run's
    # evaluate_round_best (it picks up best_results.json from leftovers).
    docker exec "$SLOT" rm -rf "/outputs/$K" "/workspace/outputs/$K" 2>/dev/null
    docker exec "$SLOT" bash -c "
        rm -rf /root/.triton/cache 2>/dev/null
        rm -rf /root/.cache/torch_extensions 2>/dev/null
        rm -rf ~/.triton/cache 2>/dev/null
        rm -rf ~/.cache/torch_extensions 2>/dev/null
        true
    " 2>/dev/null
    # Warm triton cache via a one-shot benchmark on the CURRENT kernel source.
    docker exec "$SLOT" bash -c "cd '$AKA/$LEVEL/$K' 2>/dev/null && timeout 180 python3 test_kernel_harness.py --benchmark >/dev/null 2>&1; true"

    # Run geak
    docker exec "${MEM_ENV_ARGS[@]}" "$SLOT" bash -lc "
        geak -t 'Optimize the kernel at $AKA/$LEVEL/$K/kernel.py. Use the test harness at $AKA/$LEVEL/$K/test_kernel_harness.py. Use GPUs 0-3. The output directory should be in outputs/$K.'
    " > "$LOGFILE" 2>&1
    local rc=$?
    echo "[$(date -Is)] $SLOT $K mem=$MEM_MODE exit=$rc" >> "$QLOG"

    # Snapshot outputs
    local SNAP="/data/sapmajum/triton_runs/canonical_snapshots/${SLOT}_${K}_mem${MEM_MODE}_${TS}"
    mkdir -p "$SNAP"
    docker cp "$SLOT:/workspace/outputs/$K" "$SNAP/" 2>/dev/null \
        || docker cp "$SLOT:/outputs/$K" "$SNAP/" 2>/dev/null

    return $rc
}

# Wait for slot to be idle before starting our queue (current run may still be going)
echo "[$(date -Is)] $SLOT: waiting for idle..." >> "$QLOG"
while ! slot_idle; do sleep 60; done
echo "[$(date -Is)] $SLOT: idle reached, starting queue" >> "$QLOG"

# Process queue
while IFS=: read -r LEVEL K MEM; do
    # Skip empty/whitespace-only lines (no K, just blanks)
    [ -z "$K" ] && [ -z "$LEVEL" ] && continue
    # Skip commented lines: LEVEL may be '# some text', or '#' alone
    case "${LEVEL# }" in
        '#'*) continue ;;
    esac
    # Strip leading whitespace from all three fields
    LEVEL="${LEVEL# }"
    K="${K# }"
    MEM="${MEM# }"
    [ -z "$K" ] && continue
    [ "${K:0:1}" = "#" ] && continue
    ensure_slot_up
    run_kernel "$LEVEL" "$K" "$MEM"
done < "$QUEUE_FILE"

echo "[$(date -Is)] DONE master queue for $SLOT" >> "$QLOG"
