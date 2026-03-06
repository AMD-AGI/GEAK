#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="/home/sapmajum/geak-reports"
EXP_ID="${1:?Usage: $0 <experiment_id>}"

KEY1="${AMD_LLM_API_KEY:-fa273d4402b74a9c830c9e9fc4ebfb54}"
KEY2="${AMD_LLM_API_KEY_2:-471c248fdb454e8b96173c8d25b03593}"
CONTAINER="geak-exp-${EXP_ID}"
IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"

declare -A EXP_CONFIGS
EXP_CONFIGS[exp0]="GEAK_MEMORY_DISABLE=1"
EXP_CONFIGS[exp1]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_NO_REME=1 GEAK_MEMORY_NO_PRINCIPLES=1 GEAK_MEMORY_NO_PROFILE_SIM=1 GEAK_MEMORY_NO_SAGE=1 GEAK_MEMORY_NO_CONFIDENCE=1 GEAK_MEMORY_NO_ANTIFIXATION=1 GEAK_MEMORY_NO_RECONCILIATION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp2]="GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp3]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_FORCE_REFERENCE=1"
EXP_CONFIGS[exp4]="GEAK_MEMORY_NO_WORKING=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp5]="GEAK_MEMORY_NO_STRATEGY_GRAPH=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp6]="GEAK_MEMORY_NO_CONSOLIDATION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp7]="GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp9]="GEAK_MEMORY_BUDGET=999999"
EXP_CONFIGS[exp10]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_SAGE=0"
EXP_CONFIGS[exp11]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_CONFIDENCE=0"
EXP_CONFIGS[exp12]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_ANTIFIXATION=0"
EXP_CONFIGS[exp_sqlite]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=sqlite"
EXP_CONFIGS[exp_jsonl]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=jsonl"
EXP_CONFIGS[exp_mem0]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=mem0"
EXP_CONFIGS[exp_memgraph]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=memgraph"
EXP_CONFIGS[exp_lancedb]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=lancedb"
EXP_CONFIGS[exp_falkordb]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=falkordb"
EXP_CONFIGS[exp_redis]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=redis"
EXP_CONFIGS[exp_b250]="GEAK_MEMORY_BUDGET=250"
EXP_CONFIGS[exp_b500]="GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp_b1000]="GEAK_MEMORY_BUDGET=1000"
EXP_CONFIGS[exp_bunlim]="GEAK_MEMORY_BUDGET=999999"

EXP_ENV="${EXP_CONFIGS[$EXP_ID]}"
if [ -z "$EXP_ENV" ]; then
    echo "ERROR: Unknown experiment ID '$EXP_ID'"
    echo "Available: ${!EXP_CONFIGS[@]}"
    exit 1
fi

echo "=== Experiment: $EXP_ID ==="
echo "Config: $EXP_ENV"
echo "Container: $CONTAINER"
echo "Report: $REPORT_DIR/REPORT_${EXP_ID}.md"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    -v "$REPO_DIR:/workspace" \
    -v /home/sapmajum/AIG-Eval:/workspace/AIG-Eval \
    -e AMD_LLM_API_KEY="$KEY1" \
    -e AMD_LLM_API_KEY_2="$KEY2" \
    -e HIP_FORCE_DEV_KERNARG=1 \
    "$IMAGE" \
    sleep infinity

sleep 3

docker exec "$CONTAINER" bash -c '
pip install -e /workspace 2>&1 | tail -1
# Install Metrix profiler (from AMDResearch/intellikit) -- required for bottleneck classification
if ! python3 -c "from metrix import Metrix" 2>/dev/null; then
    git clone --depth 1 https://github.com/AMDResearch/intellikit.git /tmp/intellikit 2>&1 | tail -1
    cd /tmp/intellikit/metrix && pip install -e . 2>&1 | tail -1 && cd /workspace
fi
pip install -e /workspace/mcp_tools/metrix-mcp 2>&1 | tail -1
pip install -e /workspace/mcp_tools/profiler-mcp 2>&1 | tail -1
# Install backend-specific dependencies based on experiment config
case "'"$EXP_ID"'" in
    exp_lancedb) pip install lancedb pyarrow 2>&1 | tail -1 ;;
    exp_mem0)    pip install mem0ai 2>&1 | tail -1 ;;
    exp_redis)   pip install redis 2>&1 | tail -1 ;;
esac
mkdir -p /workspace/.geak_resolved
if [ ! -d "/workspace/.geak_resolved/AMD-AGI_AIG-Eval" ]; then
    ln -sf /workspace/AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null || \
    cp -a /workspace/AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null || true
fi
if [ ! -d "/workspace/.geak_resolved/ROCm_rocPRIM" ]; then
    git clone --depth 1 -b develop_deprecated https://github.com/ROCm/rocPRIM.git /workspace/.geak_resolved/ROCm_rocPRIM 2>&1 | tail -1
fi
git config --global --add safe.directory "*"
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

TRITON_NAMES=(rope fused_qkv_rope fused_rms_fp8 gemm ff_backward topk nsa_forward nsa_backward)
TRITON_URLS=("https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/rope/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_qkv_rope/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/gemm/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/ff_backward/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/nsa_forward/kernel.py" "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/nsa_backward/kernel.py")

HIP_NAMES=(device_segmented_reduce warp_reduce device_nth_element block_radix_rank device_binary_search device_merge_sort block_run_length_decode device_partial_sort)
HIP_URLS=("https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_segmented_reduce.hpp#L42" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/warp/warp_reduce.hpp#L41" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_nth_element.hpp#L42" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/block/block_radix_rank.hpp#L52" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_binary_search.hpp#L39" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_merge_sort.hpp#L46" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/block/block_run_length_decode.hpp#L126" "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_partial_sort.hpp#L49")

# Use Triton-only by default (HIP kernels consistently produce 0x)
if [ "${GEAK_ALL_KERNELS:-0}" = "1" ]; then
    ALL_NAMES=("${TRITON_NAMES[@]}" "${HIP_NAMES[@]}")
    ALL_URLS=("${TRITON_URLS[@]}" "${HIP_URLS[@]}")
else
    ALL_NAMES=("${TRITON_NAMES[@]}")
    ALL_URLS=("${TRITON_URLS[@]}")
fi

docker exec -d "$CONTAINER" bash -c '
cd /workspace
KEY1="'"$KEY1"'"
KEY2="'"$KEY2"'"
EXP_ID="'"$EXP_ID"'"
EXP_ENV="'"$EXP_ENV"'"

ALL_NAMES=('"$(printf '"%s" ' "${ALL_NAMES[@]}")"')
ALL_URLS=('"$(printf '"%s" ' "${ALL_URLS[@]}")"')

BASE="/workspace/patches/$EXP_ID"
LOG="$BASE/logs"
mkdir -p "$LOG"

echo "=== $EXP_ID ===" > "$LOG/master.log"
echo "Config: $EXP_ENV" >> "$LOG/master.log"
echo "Canonical: 8 GPUs, 100 steps, 2 rounds + early stop" >> "$LOG/master.log"
echo "Started: $(date)" >> "$LOG/master.log"

SEQ=1
for i in "${!ALL_NAMES[@]}"; do
    T="${ALL_NAMES[$i]}"; U="${ALL_URLS[$i]}"
    [ $((SEQ % 2)) -eq 1 ] && K="$KEY1" || K="$KEY2"

    (cd /workspace/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null && git clean -fd && git checkout . ) 2>/dev/null || true
    (cd /workspace/.geak_resolved/ROCm_rocPRIM 2>/dev/null && git clean -fd && git checkout . ) 2>/dev/null || true
    rm -rf /workspace/.geak_resolved/AMD-AGI_AIG-Eval/.git/worktrees/* 2>/dev/null || true
    rm -rf /workspace/.geak_resolved/ROCm_rocPRIM/.git/worktrees/* 2>/dev/null || true

    mkdir -p "$BASE/$T/.geak_resolved"
    if echo "$U" | grep -q "AIG-Eval"; then
        ln -sf /workspace/.geak_resolved/AMD-AGI_AIG-Eval "$BASE/$T/.geak_resolved/AMD-AGI_AIG-Eval" 2>/dev/null || true
    else
        ln -sf /workspace/.geak_resolved/ROCm_rocPRIM "$BASE/$T/.geak_resolved/ROCm_rocPRIM" 2>/dev/null || true
    fi

    ST=$(date +%s)
    echo "" >> "$LOG/master.log"
    echo "[$SEQ/16] $(date +%H:%M:%S) START $T" >> "$LOG/master.log"

    env $EXP_ENV AMD_LLM_API_KEY="$K" HIP_VISIBLE_DEVICES="" GEAK_AGENT_STEP_LIMIT=100 \
    geak --kernel-url "$U" --gpu-ids 0,1,2,3,4,5,6,7 \
        --patch-output "$BASE/$T" --exit-immediately --yolo \
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
    echo "[$SEQ/16] DONE $T -> ${BSP}x (${EL}s R=$RDS ne=$NE patch=${BPSZ}b)" >> "$LOG/master.log"

    # Cleanup worktrees and git objects to save disk (keeps results, patches, logs)
    find "$BASE/$T" -name "worktrees" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE/$T" -path "*/.git/objects" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE/$T" -name "*.pyc" -delete 2>/dev/null || true

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
import json, os, glob
results_dir = \"$BASE/results_json\"
all_results = []
for f in sorted(glob.glob(os.path.join(results_dir, \"*.json\"))):
    with open(f) as fh:
        all_results.append(json.load(fh))
summary = {\"experiment\": \"$EXP_ID\", \"config\": \"$EXP_ENV\", \"kernels\": all_results}
speedups = [r[\"speedup\"] for r in all_results if r[\"speedup\"] > 0]
if speedups:
    from functools import reduce
    import math
    summary[\"geo_mean\"] = round(math.exp(sum(math.log(s) for s in speedups) / len(speedups)), 4)
with open(\"$BASE/experiment_summary.json\", \"w\") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
" 2>/dev/null || true
'

echo ""
echo "=== Experiment $EXP_ID launched ==="
echo "Monitor: docker exec $CONTAINER cat /workspace/patches/$EXP_ID/logs/master.log"
echo "Results: docker exec $CONTAINER cat /workspace/patches/$EXP_ID/experiment_summary.json"
