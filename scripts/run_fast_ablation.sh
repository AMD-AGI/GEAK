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

KEY1="${AMD_LLM_API_KEY:-}"
KEY2="${AMD_LLM_API_KEY_2:-${AMD_LLM_API_KEY:-}}"
if [ -z "$KEY1" ]; then
    echo "ERROR: AMD_LLM_API_KEY must be set in the environment."
    exit 1
fi
BASE_GEAK_MODEL="${GEAK_MODEL:-claude-opus-4.6}"
OPT_AGENT_ENSEMBLE="${GEAK_MODEL_ENSEMBLE:-gpt-5.2,claude-opus-4.6}"
GEAK_MAX_ROUNDS_OVERRIDE="${GEAK_MAX_ROUNDS:-}"
GEAK_AGENT_STEP_LIMIT_OVERRIDE="${GEAK_AGENT_STEP_LIMIT:-200}"
DEFAULT_MEMORY_ENV="GEAK_MEMORY_NO_CROSSSESSION=${GEAK_MEMORY_NO_CROSSSESSION:-1} GEAK_MEMORY_NO_REME=${GEAK_MEMORY_NO_REME:-1} GEAK_MEMORY_NO_PRINCIPLES=${GEAK_MEMORY_NO_PRINCIPLES:-1}"

TRITON_NAMES=(rope fused_rms_fp8 topk nsa_backward)
TRITON_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/rope/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/nsa_backward/kernel.py"
)

QUICK_VALIDATION_NAMES=(fused_rms_fp8 topk device_segmented_reduce)
QUICK_VALIDATION_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_segmented_reduce.hpp#L42"
)

QUICK_VALIDATION_PLUS_BINARY_NAMES=(fused_rms_fp8 topk device_segmented_reduce device_binary_search)
QUICK_VALIDATION_PLUS_BINARY_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_segmented_reduce.hpp#L42"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_binary_search.hpp#L39"
)

SMOKE_PAIR_NAMES=(fused_rms_fp8 device_binary_search)
SMOKE_PAIR_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_binary_search.hpp#L39"
)

TRITON_SMOKE_NAMES=(fused_rms_fp8)
TRITON_SMOKE_URLS=(
    "https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
)

HIP_SMOKE_NAMES=(device_binary_search)
HIP_SMOKE_URLS=(
    "https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_binary_search.hpp#L39"
)

case "${GEAK_KERNEL_SET:-default}" in
    default)
        KERNEL_NAMES=("${TRITON_NAMES[@]}")
        KERNEL_URLS=("${TRITON_URLS[@]}")
        ;;
    quick_validation)
        KERNEL_NAMES=("${QUICK_VALIDATION_NAMES[@]}")
        KERNEL_URLS=("${QUICK_VALIDATION_URLS[@]}")
        ;;
    quick_validation_plus_binary)
        KERNEL_NAMES=("${QUICK_VALIDATION_PLUS_BINARY_NAMES[@]}")
        KERNEL_URLS=("${QUICK_VALIDATION_PLUS_BINARY_URLS[@]}")
        ;;
    smoke_pair)
        KERNEL_NAMES=("${SMOKE_PAIR_NAMES[@]}")
        KERNEL_URLS=("${SMOKE_PAIR_URLS[@]}")
        ;;
    smoke_triton)
        KERNEL_NAMES=("${TRITON_SMOKE_NAMES[@]}")
        KERNEL_URLS=("${TRITON_SMOKE_URLS[@]}")
        ;;
    smoke_hip)
        KERNEL_NAMES=("${HIP_SMOKE_NAMES[@]}")
        KERNEL_URLS=("${HIP_SMOKE_URLS[@]}")
        ;;
    *)
        echo "ERROR: Unknown GEAK_KERNEL_SET='${GEAK_KERNEL_SET}'"
        echo "Available: default quick_validation quick_validation_plus_binary smoke_pair smoke_triton smoke_hip"
        exit 1
        ;;
esac

declare -A EXP_CONFIGS
EXP_CONFIGS[exp0]="GEAK_MEMORY_DISABLE=1"
EXP_CONFIGS[exp2]="GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp3]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_FORCE_REFERENCE=1"
EXP_CONFIGS[exp4]="GEAK_MEMORY_NO_WORKING=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp7]="GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8_qv]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8_qv_nogpu]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8_qv_nogpu_clean]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp8_qv_nogpu_verify]="GEAK_MEMORY_NO_CROSSSESSION=1 GEAK_MEMORY_NO_GPU=1 GEAK_MEMORY_BUDGET=500"
EXP_CONFIGS[exp10]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_SAGE=0"
EXP_CONFIGS[exp11]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_CONFIDENCE=0"
EXP_CONFIGS[exp12]="GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_NO_ANTIFIXATION=0"
EXP_CONFIGS[exp_jsonl]="GEAK_MEMORY_NO_CROSSSESSION=0 GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=jsonl"
EXP_CONFIGS[exp_lancedb]="GEAK_MEMORY_NO_CROSSSESSION=0 GEAK_MEMORY_BUDGET=500 GEAK_MEMORY_BACKEND=lancedb"

EXP_CONFIG="${EXP_CONFIGS[$EXP_ID]}"
if [ -z "$EXP_CONFIG" ]; then
    echo "ERROR: Unknown experiment '$EXP_ID'"
    echo "Available: ${!EXP_CONFIGS[@]}"
    exit 1
fi
EXP_ENV="$DEFAULT_MEMORY_ENV $EXP_CONFIG"

echo "=== Experiment: $EXP_ID ==="
echo "Config: $EXP_ENV"
echo "Output: $OUTPUT_DIR"
echo "Kernel set: ${GEAK_KERNEL_SET:-default}"
echo "Kernels: ${KERNEL_NAMES[*]}"
echo "Base GEAK model: $BASE_GEAK_MODEL"
echo "Optimization ensemble: $OPT_AGENT_ENSEMBLE"
if [ -n "$GEAK_MAX_ROUNDS_OVERRIDE" ]; then
    echo "GEAK max rounds override: $GEAK_MAX_ROUNDS_OVERRIDE"
fi
echo "GEAK agent step limit: $GEAK_AGENT_STEP_LIMIT_OVERRIDE"

mkdir -p "$OUTPUT_DIR/logs"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    -v "$REPO_DIR:/workspace" \
    -v /home/sapmajum/AIG-Eval:/workspace/AIG-Eval:ro \
    -v /home/sapmajum/.cursor:/home/sapmajum/.cursor \
    -v "$EXPERIMENTS_DIR:/geak-experiments" \
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
    [ -d /opt/intellikit/metrix ] || git clone --depth 1 https://github.com/AMDResearch/intellikit.git /opt/intellikit 2>&1 | tail -1
    pip uninstall -y metrix >/dev/null 2>&1 || true
    pip install /opt/intellikit/metrix 2>&1 | tail -1
fi
pip install -e /workspace/mcp_tools/metrix-mcp 2>&1 | tail -1
pip install -e /workspace/mcp_tools/profiler-mcp 2>&1 | tail -1
python3 - <<'PY'
from pathlib import Path

try:
    import metrix.profiler.rocprof_wrapper as rw

    path = Path(rw.__file__)
    text = path.read_text()
    changed = False

    if "import shlex\n" not in text:
        text = text.replace("import tempfile\n", "import tempfile\nimport shlex\n")
        changed = True

    old = "            prof_cmd.extend(command.split())\n"
    new = "            prof_cmd.extend(shlex.split(command))\n"
    if old in text:
        text = text.replace(old, new)
        changed = True

    if changed:
        path.write_text(text)
        print(f"Patched Metrix rocprof wrapper: {path}")
    else:
        print(f"Metrix rocprof wrapper already patched: {path}")
except Exception as exc:
    print(f"WARNING: failed to patch Metrix rocprof wrapper: {exc}")
PY
case "'"$EXP_ID"'" in
    exp_lancedb) pip install lancedb pyarrow 2>&1 | tail -1 ;;
    exp_mem0)    pip install mem0ai 2>&1 | tail -1 ;;
esac
find /workspace/src -name "*.pyc" -delete 2>/dev/null
find /workspace/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
mkdir -p /workspace/.geak_seed /workspace/.geak_pristine /workspace/.geak_resolved /output/logs
mkdir -p /geak-experiments/_testcase_cache
SEED_CACHE=/workspace/.geak_seed/AMD-AGI_AIG-Eval
ACTIVE_RESOLVER=/workspace/.geak_resolved/AMD-AGI_AIG-Eval
SEED_SOURCE=""
rm -rf "$SEED_CACHE"
if [ -d "$ACTIVE_RESOLVER" ]; then
    cp -a --no-preserve=ownership "$ACTIVE_RESOLVER" "$SEED_CACHE"
    SEED_SOURCE="$ACTIVE_RESOLVER"
else
    cp -a --no-preserve=ownership /workspace/AIG-Eval "$SEED_CACHE"
    SEED_SOURCE="/workspace/AIG-Eval"
fi
rm -rf "$SEED_CACHE/.git"
cd "$SEED_CACHE"
git init -b geak-eval-kernels >/dev/null
git add -A
GIT_AUTHOR_NAME="geak" GIT_AUTHOR_EMAIL="geak@local" \
GIT_COMMITTER_NAME="geak" GIT_COMMITTER_EMAIL="geak@local" \
git commit -q -m "pristine experiment seed"
rm -rf /workspace/.geak_pristine/AMD-AGI_AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval
cp -a --no-preserve=ownership "$SEED_CACHE" /workspace/.geak_pristine/AMD-AGI_AIG-Eval
cd /workspace/.geak_pristine/AMD-AGI_AIG-Eval
SEED_COMMIT=$(git -c safe.directory=/workspace/.geak_pristine/AMD-AGI_AIG-Eval rev-parse --short HEAD 2>/dev/null || echo "")
SEED_TREE=$(git -c safe.directory=/workspace/.geak_pristine/AMD-AGI_AIG-Eval rev-parse HEAD^{tree} 2>/dev/null || echo "")
cd /workspace
cp -a --no-preserve=ownership /workspace/.geak_pristine/AMD-AGI_AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval
python3 - <<PY
import json
from pathlib import Path

payload = {
    "seed_source": "$SEED_SOURCE",
    "seed_cache": "$SEED_CACHE",
    "seed_repo": "/workspace/.geak_pristine/AMD-AGI_AIG-Eval",
    "resolved_repo": "/workspace/.geak_resolved/AMD-AGI_AIG-Eval",
    "seed_commit": "$SEED_COMMIT",
    "seed_tree": "$SEED_TREE",
    "source_mount": "/workspace/AIG-Eval",
    "source_mount_read_only": True,
}
Path("/output/logs/source_seed.json").write_text(json.dumps(payload, indent=2))
print("AIG-Eval pristine seed prepared (%s tree=%s)" % (payload["seed_commit"], payload["seed_tree"]))
PY
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
BASE_GEAK_MODEL="'"$BASE_GEAK_MODEL"'"
OPT_AGENT_ENSEMBLE="'"$OPT_AGENT_ENSEMBLE"'"
DEFAULT_MEMORY_ENV="'"$DEFAULT_MEMORY_ENV"'"
GEAK_MAX_ROUNDS_OVERRIDE="'"$GEAK_MAX_ROUNDS_OVERRIDE"'"
GEAK_AGENT_STEP_LIMIT_OVERRIDE="'"$GEAK_AGENT_STEP_LIMIT_OVERRIDE"'"

KERNEL_NAMES=('"$NAMES_STR"')
KERNEL_URLS=('"$URLS_STR"')

BASE="/output"

# Start every rerun from a fully clean experiment output tree.
shopt -s dotglob nullglob
rm -rf "$BASE"/*
shopt -u dotglob nullglob

LOG="$BASE/logs"
mkdir -p "$LOG"
SEED_TREE=$(git -c safe.directory=/workspace/.geak_pristine/AMD-AGI_AIG-Eval -C /workspace/.geak_pristine/AMD-AGI_AIG-Eval rev-parse HEAD^{tree} 2>/dev/null || echo "")
SEED_COMMIT=$(git -c safe.directory=/workspace/.geak_pristine/AMD-AGI_AIG-Eval -C /workspace/.geak_pristine/AMD-AGI_AIG-Eval rev-parse --short HEAD 2>/dev/null || echo "")
python3 - <<PY
import json
from pathlib import Path

payload = {
    "seed_repo": "/workspace/.geak_pristine/AMD-AGI_AIG-Eval",
    "resolved_repo": "/workspace/.geak_resolved/AMD-AGI_AIG-Eval",
    "seed_commit": "$SEED_COMMIT",
    "seed_tree": "$SEED_TREE",
    "source_mount": "/workspace/AIG-Eval",
    "source_mount_read_only": True,
}
Path("$LOG/source_seed.json").write_text(json.dumps(payload, indent=2))
PY

echo "=== $EXP_ID ===" > "$LOG/master.log"
echo "Config: $EXP_ENV" >> "$LOG/master.log"
echo "Kernels: ${KERNEL_NAMES[*]}" >> "$LOG/master.log"
echo "Base GEAK model: $BASE_GEAK_MODEL" >> "$LOG/master.log"
echo "Optimization ensemble: $OPT_AGENT_ENSEMBLE" >> "$LOG/master.log"
echo "Memory defaults: $DEFAULT_MEMORY_ENV" >> "$LOG/master.log"
echo "Canonical testcase cache: /geak-experiments/_testcase_cache" >> "$LOG/master.log"
if [ -n "$GEAK_MAX_ROUNDS_OVERRIDE" ]; then
    echo "GEAK max rounds override: $GEAK_MAX_ROUNDS_OVERRIDE" >> "$LOG/master.log"
fi
echo "GEAK agent step limit: $GEAK_AGENT_STEP_LIMIT_OVERRIDE" >> "$LOG/master.log"
echo "Source seed: ${SEED_COMMIT} tree=${SEED_TREE}" >> "$LOG/master.log"
echo "Started: $(date)" >> "$LOG/master.log"

SEQ=1
for i in "${!KERNEL_NAMES[@]}"; do
    T="${KERNEL_NAMES[$i]}"
    U="${KERNEL_URLS[$i]}"
    [ $((SEQ % 2)) -eq 1 ] && K="$KEY1" || K="$KEY2"

    # Rebuild the active resolver repo from the pristine experiment seed so
    # every kernel starts from the same clean baseline source tree.
    rm -rf /workspace/.geak_resolved/AMD-AGI_AIG-Eval
    cp -a --no-preserve=ownership /workspace/.geak_pristine/AMD-AGI_AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval
    rm -rf /workspace/.geak_resolved/AMD-AGI_AIG-Eval/.git/worktrees/* 2>/dev/null || true
    SEED_TREE=$(git -c safe.directory=/workspace/.geak_pristine/AMD-AGI_AIG-Eval -C /workspace/.geak_pristine/AMD-AGI_AIG-Eval rev-parse HEAD^{tree} 2>/dev/null || echo "")
    ACTIVE_TREE=$(git -c safe.directory=/workspace/.geak_resolved/AMD-AGI_AIG-Eval -C /workspace/.geak_resolved/AMD-AGI_AIG-Eval rev-parse HEAD^{tree} 2>/dev/null || echo "")
    if [ -z "$SEED_TREE" ] || [ "$SEED_TREE" != "$ACTIVE_TREE" ]; then
        echo "[$SEQ/${#KERNEL_NAMES[@]}] ERROR $T -> clean baseline reset failed" >> "$LOG/master.log"
        exit 1
    fi

    # Start each kernel from a truly clean output tree so stale task files,
    # best_results.json, and patch logs cannot bleed into a new run.
    rm -rf "$BASE/$T"
    # Link per-output .geak_resolved to global git repo so resolver finds it
    mkdir -p "$BASE/$T/.geak_resolved"
    ln -sfn /workspace/.geak_resolved/AMD-AGI_AIG-Eval "$BASE/$T/.geak_resolved/AMD-AGI_AIG-Eval"

    ST=$(date +%s)
    echo "" >> "$LOG/master.log"
    echo "[$SEQ/${#KERNEL_NAMES[@]}] $(date +%H:%M:%S) START $T" >> "$LOG/master.log"

    EXTRA_ENV=()
    if [ -n "$GEAK_MAX_ROUNDS_OVERRIDE" ]; then
        EXTRA_ENV+=("GEAK_MAX_ROUNDS=$GEAK_MAX_ROUNDS_OVERRIDE")
    fi

    env $EXP_ENV AMD_LLM_API_KEY="$K" HIP_VISIBLE_DEVICES="" GEAK_AGENT_STEP_LIMIT="$GEAK_AGENT_STEP_LIMIT_OVERRIDE" \
    GEAK_MODEL="$BASE_GEAK_MODEL" \
    GEAK_MODEL_ENSEMBLE="$OPT_AGENT_ENSEMBLE" \
    GEAK_TESTCASE_CACHE_DIR="/geak-experiments/_testcase_cache" \
    GEAK_EXCLUDED_AGENTS=openevolve \
    "${EXTRA_ENV[@]}" \
    geak --kernel-url "$U" --gpu-ids 0,1,2,3,4,5,6,7 \
        --patch-output "$BASE/$T" --exit-immediately --yolo --heterogeneous \
        > "$LOG/${T}.log" 2>&1

    ET=$(date +%s); EL=$(( ET - ST ))

    BSP=0; BPSZ=0; BTASK=""
    FINAL_REPORT="$BASE/$T/final_report.json"
    if [ -f "$FINAL_REPORT" ]; then
        BSP=$(python3 - <<PY 2>/dev/null || echo 0
import json
from pathlib import Path

path = Path("$FINAL_REPORT")
report = json.loads(path.read_text())
speedup = report.get("verified_speedup")
if speedup is None:
    round_eval = report.get("round_evaluation", {})
    full_benchmark = round_eval.get("full_benchmark", {}) if isinstance(round_eval, dict) else {}
    speedup = full_benchmark.get("verified_speedup", round_eval.get("benchmark_speedup"))
if speedup is None:
    total = str(report.get("total_speedup", "")).strip()
    if total.endswith("%"):
        speedup = 1.0 + float(total.rstrip("%")) / 100.0
    elif total.lower().endswith("x"):
        speedup = float(total[:-1])
    elif total:
        speedup = float(total)
    else:
        speedup = 0.0
print(f"{max(1.0, float(speedup)):.6f}")
PY
)
        BEST_PATCH=$(python3 - <<PY 2>/dev/null || echo ""
import json
from pathlib import Path
path = Path("$FINAL_REPORT")
report = json.loads(path.read_text())
print(report.get("best_patch", ""))
PY
)
        if [ -n "$BEST_PATCH" ] && [ -f "$BEST_PATCH" ]; then
            BPSZ=$(python3 - <<PY 2>/dev/null || echo 0
from pathlib import Path
print(Path("$BEST_PATCH").stat().st_size if Path("$BEST_PATCH").is_file() else 0)
PY
)
            BTASK="$(dirname "$BEST_PATCH")"
        fi
    fi
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
