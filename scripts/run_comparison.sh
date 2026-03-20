#!/usr/bin/env bash
set -euo pipefail

LOGDIR="/home/sdubagun/work/repos/GEAK/outputs/comparison_logs"
mkdir -p "$LOGDIR"

TS=$(date +%Y%m%d_%H%M%S)
AIG_EVAL="/home/sdubagun/work/repos/AIG-Eval"
GEAK="/home/sdubagun/work/repos/GEAK"
API_KEY="471c248fdb454e8b96173c8d25b03593"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/master_${TS}.log"; }

# ============================================================================
# Step 1: OpenEvolve on fused_rms_fp8
# ============================================================================
log "=== OpenEvolve: fused_rms_fp8 ==="
docker exec geak-v2-sdubagun bash -c "
  export AMD_API_KEY='${API_KEY}'
  export OPENAI_API_KEY='${API_KEY}'
  cd ${AIG_EVAL}
  mkdir -p logs
  ./run_geak_v2.sh openevolve fused_rms_fp8
" 2>&1 | tee "$LOGDIR/oe_fused_rms_fp8_${TS}.log"
log "OpenEvolve fused_rms_fp8 done."
sleep 10

# ============================================================================
# Step 2: OpenEvolve on fused_qkv_rope
# ============================================================================
log "=== OpenEvolve: fused_qkv_rope ==="
docker exec geak-v2-sdubagun bash -c "
  export AMD_API_KEY='${API_KEY}'
  export OPENAI_API_KEY='${API_KEY}'
  cd ${AIG_EVAL}
  mkdir -p logs
  ./run_geak_v2.sh openevolve fused_qkv_rope
" 2>&1 | tee "$LOGDIR/oe_fused_qkv_rope_${TS}.log"
log "OpenEvolve fused_qkv_rope done."
sleep 10

# ============================================================================
# Step 3: geak-orchestrate on fused_rms_fp8
# ============================================================================
log "=== geak-orchestrate: fused_rms_fp8 ==="
docker exec geak-agent-sdubagun geak-orchestrate \
    --preprocess-dir outputs/eval_3kernels/fused_rms_fp8/homo_multi \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --max-rounds 2 \
    --model claude-opus-4-6 \
    2>&1 | tee "$LOGDIR/geak_fused_rms_fp8_${TS}.log"
log "geak-orchestrate fused_rms_fp8 done."
sleep 10

# ============================================================================
# Step 4: geak-orchestrate on fused_qkv_rope
# ============================================================================
log "=== geak-orchestrate: fused_qkv_rope ==="
docker exec geak-agent-sdubagun geak-orchestrate \
    --preprocess-dir outputs/eval_3kernels/fused_qkv_rope/homo_multi \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --max-rounds 2 \
    --model claude-opus-4-6 \
    2>&1 | tee "$LOGDIR/geak_fused_qkv_rope_${TS}.log"
log "geak-orchestrate fused_qkv_rope done."

# ============================================================================
# Summary
# ============================================================================
log "=== ALL RUNS COMPLETE ==="
log "Logs in: $LOGDIR"
