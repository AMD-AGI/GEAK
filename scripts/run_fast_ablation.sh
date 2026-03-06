#!/bin/bash
# Run a single ablation experiment using geak --kernel-url directly.
# Each experiment gets an ISOLATED output directory -- no shared state.
#
# Usage: bash scripts/run_fast_ablation.sh <experiment_id>
# Example: bash scripts/run_fast_ablation.sh exp0

set -e

EXP_ID="${1:?Usage: $0 <experiment_id>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPERIMENTS_DIR="/home/sapmajum/geak-experiments"
OUTPUT_DIR="$EXPERIMENTS_DIR/$EXP_ID"
IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
CONTAINER="geak-fast-$EXP_ID"

KEY1="${AMD_LLM_API_KEY:-fa273d4402b74a9c830c9e9fc4ebfb54}"
KEY2="${AMD_LLM_API_KEY_2:-471c248fdb454e8b96173c8d25b03593}"

KERNEL_NAMES=(rope fused_rms_fp8 topk nsa_backward device_segmented_reduce device_merge_sort)
KERNEL_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/rope/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/nsa_backward/kernel.py"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_segmented_reduce.hpp#L42"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_merge_sort.hpp#L46"
)

declare -A EXP_CONFIGS
EXP_CONFIGS[exp0]="GEAK_MEMORY_DISABLE=1"
EXP_CONFIGS[exp2]="GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp3]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_FORCE_REFERENCE=1"
EXP_CONFIGS[exp4]="GEAK_MEMORY_NO_WORKING=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp7]="GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp10]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_SAGE=0"
EXP_CONFIGS[exp11]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_CONFIDENCE=0"
EXP_CONFIGS[exp12]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_ANTIFIXATION=0"
EXP_CONFIGS[exp_jsonl]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=jsonl"
EXP_CONFIGS[exp_lancedb]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=lancedb"

EXP_ENV="${EXP_CONFIGS[$EXP_ID]}"
if [ -z "$EXP_ENV" ]; then
    echo "ERROR: Unknown experiment '$EXP_ID'"
    echo "Available: ${!EXP_CONFIGS[@]}"
    exit 1
fi

echo "=== Experiment: $EXP_ID ==="
echo "Config: $EXP_ENV"
echo "Output: $OUTPUT_DIR"
echo "Kernels: ${KERNEL_NAMES[*]}"

mkdir -p "$OUTPUT_DIR/logs"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    -v "$REPO_DIR:/workspace" \
    -v /home/sapmajum/AIG-Eval:/workspace/AIG-Eval \
    -v "$OUTPUT_DIR:/output" \
    -e AMD_LLM_API_KEY="$KEY1" \
    -e AMD_LLM_API_KEY_2="$KEY2" \
    -e HIP_FORCE_DEV_KERNARG=1 \
    "$IMAGE" \
    sleep infinity

sleep 3

docker exec "$CONTAINER" bash -c '
pip install -e /workspace 2>&1 | tail -1
if ! python3 -c "from metrix import Metrix" 2>/dev/null; then
    git clone --depth 1 https://github.com/AMDResearch/intellikit.git /tmp/intellikit 2>&1 | tail -1
    cd /tmp/intellikit/metrix && pip install -e . 2>&1 | tail -1 && cd /workspace
fi
pip install -e /workspace/mcp_tools/metrix-mcp 2>&1 | tail -1
pip install -e /workspace/mcp_tools/profiler-mcp 2>&1 | tail -1
case "'"$EXP_ID"'" in
    exp_lancedb) pip install lancedb pyarrow 2>&1 | tail -1 ;;
    exp_mem0)    pip install mem0ai 2>&1 | tail -1 ;;
esac
find /workspace/src -name "*.pyc" -delete 2>/dev/null
find /workspace/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
mkdir -p /workspace/.geak_resolved
git config --global --add safe.directory "*"
git config --global user.email "geak@local"
git config --global user.name "geak"
# Create a proper git repo from host-mounted AIG-Eval (host has no .git)
if [ ! -d "/workspace/.geak_resolved/AMD-AGI_AIG-Eval/.git" ]; then
    rm -rf /workspace/.geak_resolved/AMD-AGI_AIG-Eval
    cp -a /workspace/AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval
    cd /workspace/.geak_resolved/AMD-AGI_AIG-Eval
    git init -b geak-eval-kernels && git add -A && git commit -q -m "init from host"
    echo "AIG-Eval git repo created ($(git rev-parse --short HEAD))"
    cd /workspace
fi
mkdir -p /root/.config/mini-swe-agent
cat > /root/.config/mini-swe-agent/.env << ENVEOF
MODEL=openai/amd-llama-4-maverick
OPENAI_API_KEY=$AMD_LLM_API_KEY
OPENAI_BASE_URL=https://api.amd.com/llm/v1
MSWEA_CONFIGURED=true
MSWEA_MODEL_NAME=openai/amd-llama-4-maverick
ENVEOF
echo "Setup complete"
'

echo "=== Verifying memory code ==="
docker exec "$CONTAINER" python3 -c "
import inspect
from minisweagent.run.orchestrator import _run_llm_steps, _auto_finalize
from minisweagent.run.pipeline_helpers import inject_pipeline_context
s1 = inspect.getsource(_run_llm_steps)
s2 = inspect.getsource(_auto_finalize)
s3 = inspect.getsource(inject_pipeline_context)
ok = True
if 'working_memory' not in s1: print('FAIL: working_memory missing'); ok = False
if 'optimization_technique' in s2: print('FAIL: bad kwargs in auto_finalize'); ok = False
if 'assemble_memory_context' not in s3: print('FAIL: memory missing from pipeline_helpers'); ok = False
from metrix import Metrix; print('Metrix: OK')
if ok: print('All memory code verified OK')
" 2>&1 || { echo "VERIFICATION FAILED"; exit 1; }

# Build the kernel arrays as shell-safe strings for docker exec
NAMES_STR=$(printf '"%s" ' "${KERNEL_NAMES[@]}")
URLS_STR=$(printf '"%s" ' "${KERNEL_URLS[@]}")

docker exec -d "$CONTAINER" bash -c '
cd /workspace
KEY1="'"$KEY1"'"
KEY2="'"$KEY2"'"
EXP_ID="'"$EXP_ID"'"
EXP_ENV="'"$EXP_ENV"'"

KERNEL_NAMES=('"$NAMES_STR"')
KERNEL_URLS=('"$URLS_STR"')

BASE="/output"
LOG="$BASE/logs"
mkdir -p "$LOG"

echo "=== $EXP_ID ===" > "$LOG/master.log"
echo "Config: $EXP_ENV" >> "$LOG/master.log"
echo "Kernels: ${KERNEL_NAMES[*]}" >> "$LOG/master.log"
echo "Started: $(date)" >> "$LOG/master.log"

SEQ=1
for i in "${!KERNEL_NAMES[@]}"; do
    T="${KERNEL_NAMES[$i]}"
    U="${KERNEL_URLS[$i]}"
    [ $((SEQ % 2)) -eq 1 ] && K="$KEY1" || K="$KEY2"

    # Clean git state between kernels so worktrees are fresh
    (cd /workspace/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null && git clean -fd && git checkout . ) 2>/dev/null || true
    rm -rf /workspace/.geak_resolved/AMD-AGI_AIG-Eval/.git/worktrees/* 2>/dev/null || true

    # Link per-output .geak_resolved to global git repo so resolver finds it
    mkdir -p "$BASE/$T/.geak_resolved"
    ln -sfn /workspace/.geak_resolved/AMD-AGI_AIG-Eval "$BASE/$T/.geak_resolved/AMD-AGI_AIG-Eval"

    ST=$(date +%s)
    echo "" >> "$LOG/master.log"
    echo "[$SEQ/${#KERNEL_NAMES[@]}] $(date +%H:%M:%S) START $T" >> "$LOG/master.log"

    env $EXP_ENV AMD_LLM_API_KEY="$K" HIP_VISIBLE_DEVICES="" GEAK_AGENT_STEP_LIMIT=100 \
    geak --kernel-url "$U" --gpu-ids 0,1,2,3,4,5,6,7 \
        --patch-output "$BASE/$T" --exit-immediately --yolo --heterogeneous \
        > "$LOG/${T}.log" 2>&1

    ET=$(date +%s); EL=$(( ET - ST ))

    BSP=0; BPSZ=0; BTASK=""
    for f in $(find "$BASE/$T/results" -maxdepth 3 -name "best_results.json" 2>/dev/null); do
        SP=$(python3 -c "import json; print(json.load(open(\"$f\")).get(\"best_patch_speedup\", 0))" 2>/dev/null || echo 0)
        PSZ=$(python3 -c "import json,os; pf=json.load(open(\"$f\")).get(\"best_patch_file\",\"\"); print(os.path.getsize(pf) if pf and os.path.isfile(pf) else 0)" 2>/dev/null || echo 0)
        BT=$(python3 -c "print(1 if float(\"${SP:-0}\") > float(\"$BSP\") and int(\"${PSZ:-0}\") > 0 else 0)" 2>/dev/null || echo 0)
        if [ "$BT" = "1" ]; then BSP="$SP"; BPSZ="$PSZ"; BTASK="$(dirname "$f")"; fi
    done
    RDS=$(ls -d "$BASE/$T/results/round_"* 2>/dev/null | wc -l)
    NE=$(find "$BASE/$T" -name "patch_*.patch" -size +0c 2>/dev/null | wc -l)
    echo "[$SEQ/${#KERNEL_NAMES[@]}] DONE $T -> ${BSP}x (${EL}s R=$RDS ne=$NE patch=${BPSZ}b)" >> "$LOG/master.log"

    # Cleanup worktrees and git objects to save disk
    find "$BASE/$T" -name "worktrees" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE/$T" -path "*/.git/objects" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE/$T" -name "*.pyc" -delete 2>/dev/null || true
    find /workspace -name ".rocprofv3" -type d -exec rm -rf {} + 2>/dev/null || true

    python3 -c "
import json, os
result = {\"kernel\": \"$T\", \"speedup\": float(\"$BSP\"), \"patch_size_bytes\": int(\"$BPSZ\"),
          \"rounds\": int(\"$RDS\"), \"non_empty_patches\": int(\"$NE\"),
          \"wall_time_sec\": int(\"$EL\"), \"task_dir\": \"$BTASK\"}
os.makedirs(\"$BASE/results_json\", exist_ok=True)
with open(\"$BASE/results_json/${T}.json\", \"w\") as f:
    json.dump(result, f, indent=2)
" 2>/dev/null || true

    SEQ=$((SEQ + 1))
done

echo "" >> "$LOG/master.log"
echo "COMPLETE: $(date)" >> "$LOG/master.log"

python3 -c "
import json, os, glob, math
results_dir = \"$BASE/results_json\"
all_results = []
for f in sorted(glob.glob(os.path.join(results_dir, \"*.json\"))):
    with open(f) as fh:
        all_results.append(json.load(fh))
summary = {\"experiment\": \"$EXP_ID\", \"config\": \"$EXP_ENV\", \"kernels\": all_results}
speedups = [r[\"speedup\"] for r in all_results if r[\"speedup\"] > 0]
if speedups:
    summary[\"geo_mean\"] = round(math.exp(sum(math.log(s) for s in speedups) / len(speedups)), 4)
with open(\"$BASE/experiment_summary.json\", \"w\") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
" 2>/dev/null || true
'

echo ""
echo "=== Experiment $EXP_ID launched ==="
echo "Monitor: docker exec geak-fast-$EXP_ID cat /output/logs/master.log"
echo "Output: $OUTPUT_DIR"
