# GEAK Adaptive Mixed Mode — Full Documentation

**Branch:** `mixed-optimizer` (pushed to `origin/mixed-optimizer`)
**Base:** `gwiab-scheduler` at commit `f2f0e1b`
**Commit:** `1433875` — "Adaptive K allocation for mixed mode dispatch"
**Repo:** `git@github.com:AMD-AGI/GEAK.git`
**Local clone:** `/data/sapmajum/GEAK_scheduler`

---

## 1. Problem Statement

GEAK has three pipeline modes for multi-GPU kernel optimization:

| Mode | K (planned slots) | N-K (fixed slots) | Behavior |
|------|-------------------|-------------------|----------|
| **fixed** | 0 | N | All workers get identical canonical prompt — pure brute-force exploration |
| **planned** | N | 0 | All workers get distinct LLM-generated strategies — pure diversity |
| **mixed** | K | N-K | Some planned, some fixed — hybrid |

The old mixed mode used a **static** split: `K = N // 2` (half each). This is suboptimal:
- If planned strategies consistently outperform fixed, we waste half the GPU slots
- If fixed outperforms planned (e.g., on simple kernels), we waste the other half
- No learning across rounds — round 5 makes the same allocation as round 1

---

## 2. Solution: Adaptive K Allocation

**Core idea:** After each round, measure which source (planned vs fixed) produced better speedups. Allocate proportionally in the next round.

### Algorithm

```
Round 1: K = N // 2 (equal split, no history)

Round R > 1:
  planned_avg = mean(speedup of all kind="planned" tasks across rounds 1..R-1)
  fixed_avg   = mean(speedup of all kind="fixed" tasks across rounds 1..R-1)

  K = round(N * planned_avg / (planned_avg + fixed_avg))
  K = clamp(K, 1, N-1)  ← exploration floor
```

### Properties

- **Proportional:** Better source gets more slots, proportional to its advantage
- **Cumulative:** Uses ALL historical data, not just last round (more stable)
- **Exploration floor:** Both sources always get at least 1 slot (never starves either)
- **Backward compatible:** fixed mode (K=0) and planned mode (K=N) unchanged
- **Graceful degradation:** If no per-task data exists (old eval format), falls back to N//2

### Example

With N=4 GPUs:
```
Round 1: K=2 (2 planned, 2 fixed)
  → planned tasks: 2.0x, 1.5x (avg=1.75)
  → fixed tasks:   1.1x, 1.0x (avg=1.05)

Round 2: K = round(4 * 1.75 / 2.80) = round(2.5) = 2
  → Still 2/2, but if planned pulls further ahead...

Round 3: (planned_avg=2.5, fixed_avg=1.05)
  K = round(4 * 2.5 / 3.55) = round(2.82) = 3
  → 3 planned, 1 fixed — shifted toward planned

Round 4: (planned_avg=2.75, fixed_avg=1.0)
  K = round(4 * 2.75 / 3.75) = round(2.93) = 3
  → Stays 3/1, clamped by exploration floor (min 1 fixed)
```

---

## 3. Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UNIFIED ROUND LOOP                          │
│                     (unified.py:_run_unified_loop)                  │
│                                                                     │
│  for round_num in 1..max_rounds:                                    │
│                                                                     │
│    ┌──────────┐     ┌────────────┐     ┌─────────┐     ┌────────┐  │
│    │   PLAN   │────▶│   SELECT   │────▶│  WRITE  │────▶│EXECUTE │  │
│    │TaskPlanner│    │ Dispatcher │    │task files│    │ staged │  │
│    │.build_pool│    │  .select() │    │  (.md)  │    │dispatch│  │
│    └──────────┘     └────────────┘     └─────────┘     └────────┘  │
│         │                │                                    │     │
│    M candidates     K planned +              per-task best_results  │
│    (pool)           (N-K) fixed              + patches on disk      │
│                     (DispatchPlan)                             │     │
│                          ▲                                    ▼     │
│                          │              ┌──────────┐               │
│                          │              │ EVALUATE │               │
│                    round_evals ◀────────│ evaluate │               │
│                    (per_task with        │ round_best│               │
│                     kind + speedup)     └──────────┘               │
│                                              │                      │
│                                      round_N_evaluation.json        │
│                                      (includes per_task array)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| TaskPlanner | `run/planner/task_planner.py` | LLM generates M candidate strategies; always injects canonical fixed entry |
| CandidatePool | `run/planner/candidate_pool.py` | Pool of M candidates with `.planned`, `.fixed` filtering |
| **Dispatcher** | `run/dispatcher/selector.py` | **THE adaptive seam** — picks K planned + (N-K) fixed |
| Writer | `run/dispatcher/writer.py` | Writes DispatchPlan as `.md` task files (YAML frontmatter + markdown body) |
| Staged dispatch | `run/dispatch.py` | Priority-staged execution with early exit on improvement |
| Evaluation | `run/postprocess/evaluation.py` | FULL_BENCHMARK verification, per-task outcome tagging |
| Unified loop | `run/unified.py` | Orchestrates PLAN→SELECT→WRITE→EXECUTE→EVALUATE per round |

### Data Flow for `kind`

```
CandidateTask.kind ──▶ DispatchPlanItem.kind ──▶ task file YAML (kind: planned/fixed)
                                                        │
                                                        ▼
                                              task_file_to_agent_task()
                                                  cfg["kind"] = meta["kind"]  ← NEW
                                                        │
                                                        ▼
                                              results/{label}/best_results.json
                                                        │
                                              evaluate_round_best() scans results
                                              _resolve_task_kind() reads YAML ← NEW
                                                        │
                                                        ▼
                                              round_N_evaluation.json
                                                { ..., "per_task": [
                                                    {"label": "...", "kind": "planned", "speedup": 1.5},
                                                    {"label": "...", "kind": "fixed", "speedup": 1.1}
                                                ]}
                                                        │
                                                        ▼
                                              Dispatcher._k_for_mode(round_evals=...)
                                              → adaptive K for next round
```

---

## 4. Code Changes (4 files)

### 4.1 `src/minisweagent/run/dispatcher/selector.py`

**Full file after changes (172 lines):**

The `Dispatcher.select()` now accepts `round_evals` and passes it to `_k_for_mode()`.

`_k_for_mode()` was replaced from a one-liner:
```python
# OLD:
return {"fixed": 0, "planned": n, "mixed": n // 2}.get(mode, n // 2)
```
to the adaptive algorithm that:
1. Returns 0 for fixed, N for planned (unchanged)
2. For mixed: scans `per_task` entries across all `round_evals`, computes per-source average speedup, allocates K proportionally, clamps to [1, N-1]

### 4.2 `src/minisweagent/run/postprocess/evaluation.py`

Three additions:
1. **`_resolve_task_kind(task_files_dir, label)`** — looks up `kind` from the `.md` task file matching a given label
2. In `evaluate_round_best()`: each candidate now includes `"kind"` resolved from the task file; `round_eval["per_task"]` is populated before any return path
3. In `write_eval_results()`: imports `PerTaskOutcome`, converts `round_eval["per_task"]` into typed objects on the returned `RoundEvaluation`

### 4.3 `src/minisweagent/run/dispatch.py`

One-line change at line 317:
```python
# OLD:
for _passthrough_key in ("baseline_metrics", "benchmark_baseline"):
# NEW:
for _passthrough_key in ("baseline_metrics", "benchmark_baseline", "kind"):
```

### 4.4 `src/minisweagent/run/unified.py`

One-line change at line 581:
```python
# OLD:
plan = dispatcher.select(pool, mode, n_workers)
# NEW:
plan = dispatcher.select(pool, mode, n_workers, round_evals=round_evals)
```

---

## 5. Setup on a New Node

### 5.1 Clone the branch

```bash
cd /data/$USER
git clone -b mixed-optimizer git@github.com:AMD-AGI/GEAK.git GEAK_mixed
cd GEAK_mixed
```

### 5.2 Container requirements

GEAK runs inside Docker containers with ROCm and GPU access. You need two containers:

**Triton container** (for Triton kernels):
```bash
# Example: create or reuse a container with ROCm + Triton support
# The parity test used: geak_gwiab_triton (image: geak-agent:latest)
docker run -d --name geak_mixed_triton \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /data:/data \
  -e HSA_XNACK=1 \
  geak-agent:latest sleep infinity
```

**HIP container** (for HIP/CUDA kernels):
```bash
# The parity test used: geak_gwiab_hip (image: rocm/pytorch:rocm7.1.1)
docker run -d --name geak_mixed_hip \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /data:/data \
  rocm/pytorch:rocm7.1.1 sleep infinity
```

### 5.3 Copy GEAK into containers

```bash
GEAK_SRC="/data/$USER/GEAK_mixed"

# Triton container
docker exec geak_mixed_triton bash -c "rm -rf /workspace/src && mkdir -p /workspace"
docker cp "$GEAK_SRC/src" geak_mixed_triton:/workspace/src
docker exec geak_mixed_triton pip install -e /workspace 2>/dev/null || true

# HIP container
docker exec geak_mixed_hip bash -c "rm -rf /workspace/src && mkdir -p /workspace"
docker cp "$GEAK_SRC/src" geak_mixed_hip:/workspace/src
docker exec geak_mixed_hip pip install -e /workspace 2>/dev/null || true
```

### 5.4 API key

```bash
export AMD_LLM_API_KEY="7d3c15d3142b4d1492859da87f16d9fc"
```

The model is `claude-opus-4.6` via AMD's LLM gateway.

### 5.5 Kernel test data (AKA)

The parity test kernels are in the AKA (Automated Kernel Archive):
```
/data/sapmajum/parity_test/runs_20260514/AKA_triton/tasks/triton2triton/geak_eval/
/data/sapmajum/parity_test/runs_20260514/AKA_hip/tasks/hip2hip/others/
```

If running on a different node, you'll need to either:
- Copy these directories to the new node
- Or clone AKA fresh from the repository

---

## 6. Test Plan: Mixed Mode vs Planned vs Fixed

### 6.1 Goal

Run the same 30 kernels (18 Triton + 12 HIP) in **mixed mode** and compare with the existing **planned** (Triton) and **fixed** (HIP) baselines from the May 14 parity test.

### 6.2 Previous baseline results

#### Triton kernels (planned mode, May 14)

| Kernel | Level | Final Speedup | Rounds |
|--------|-------|---------------|--------|
| fused_append_shared_experts | L1 | 2.99x | 5 |
| llama_ff_triton | L1 | 5.03x | 5 |
| mla_decode | L1 | 1.18x | 2 |
| moe_routing_sigmoid_top1 | L1 | 1.26x | 5 |
| refk_fp8_blockwise_mm | L1 | 1.00x | 4 |
| refk_identity | L1 | 1.02x | 4 |
| fast_rms_layernorm | L2 | 10.37x | 5 |
| ff_backward | L2 | 1.00x | 2 |
| lean_atten_paged | L2 | 1.00x | 5 |
| topk | L2 | 1.21x | 5 |
| fused_moe_mxfp4 | L3 | 1.06x | 2 |
| fused_mxfp4_quant_moe_sort | L3 | 1.84x | 2 |
| fused_qk_rope_cache_mla | L3 | 8.25x | 5 |
| fused_qkv_rope | L3 | 3.93x | 4 |
| fused_rms_fp8 | L3 | 2.31x | 3 |
| gemm | L3 | 3.13x | 5 |
| gemm_a16w16_atomic | L3 | — | 0 (preflight fail) |
| gemm_a16wfp4 | L3 | 1.01x | 5 |

#### HIP kernels (fixed mode, May 14)

| Kernel | Final Speedup |
|--------|---------------|
| roiaware_pool3d | 27.01x |
| roipoint_pool3d | 14.76x |
| assign_score_withk | 2.81x |
| knn | 2.54x |
| three_nn | 1.98x |
| three_interpolate | 1.80x |
| gather_points | 1.46x |
| silu | 1.27x (incomplete) |
| furthest_point_sample | 1.13x |
| ball_query | 1.18x |
| matrix_multiplication | 1.09x |
| points_in_boxes | 1.03x |

### 6.3 Run commands

**Mixed mode** is the hardcoded default in `mini.py` (`pipeline_mode = "mixed"`). The key env var is `GEAK_PIPELINE_MODE` which overrides it from the launch script.

#### Single kernel test (quick validation):

```bash
# Pick a well-known kernel for quick validation
KERNEL_DIR="/path/to/AKA_triton/tasks/triton2triton/geak_eval/L1/llama_ff_triton"
OUT_DIR="/data/$USER/mixed_test/triton_llama_ff_triton"

docker exec \
  -e "GEAK_MODEL=claude-opus-4.6" \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_PIPELINE_MODE=mixed" \
  geak_mixed_triton bash -c \
  "cd /workspace && geak -t 'Optimize the Triton kernel at ${KERNEL_DIR}/kernel.py. Use the test harness at ${KERNEL_DIR}/test_kernel_harness.py.' \
   --num-parallel 4 --gpu-ids 0,1,2,3 --mode full -y --exit-immediately" \
  > "$OUT_DIR/run.log" 2>&1
```

#### Full batch script template:

```bash
#!/usr/bin/env bash
# Mixed mode parity test: 18 Triton + 12 HIP
set -euo pipefail

AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-7d3c15d3142b4d1492859da87f16d9fc}"
BATCH_DIR="/data/$USER/mixed_test"
MODEL="claude-opus-4.6"

TRITON_CONTAINER="geak_mixed_triton"
HIP_CONTAINER="geak_mixed_hip"

# Point these to your AKA directories
AKA_TRITON="/path/to/AKA_triton/tasks/triton2triton/geak_eval"
AKA_HIP="/path/to/AKA_hip/tasks/hip2hip/others"

RESULTS_CSV="$BATCH_DIR/results.csv"
mkdir -p "$BATCH_DIR"
echo "name,language,level,mode,verified_speedups,final_speedup,status,elapsed_s" > "$RESULTS_CSV"

TRITON_KERNELS=(
  "fused_append_shared_experts|L1|fused_append_shared_experts"
  "llama_ff_triton|L1|llama_ff_triton"
  "mla_decode|L1|mla_decode"
  "moe_routing_sigmoid_top1|L1|moe_routing_sigmoid_top1"
  "refk_fp8_blockwise_mm|L1|refk_fp8_blockwise_mm"
  "refk_identity|L1|refk_identity"
  "fast_rms_layernorm|L2|fast_rms_layernorm"
  "ff_backward|L2|ff_backward"
  "lean_atten_paged|L2|lean_atten_paged"
  "topk|L2|topk"
  "fused_moe_mxfp4|L3|fused_moe_mxfp4"
  "fused_mxfp4_quant_moe_sort|L3|fused_mxfp4_quant_moe_sort"
  "fused_qk_rope_cache_mla|L3|fused_qk_rope_cache_mla"
  "fused_qkv_rope|L3|fused_qkv_rope"
  "fused_rms_fp8|L3|fused_rms_fp8"
  "gemm|L3|gemm"
  "gemm_a16w16_atomic|L3|gemm_a16w16_atomic"
  "gemm_a16wfp4|L3|gemm_a16wfp4"
)

HIP_KERNELS=(
  "assign_score_withk" "ball_query" "furthest_point_sample"
  "gather_points" "knn" "matrix_multiplication"
  "points_in_boxes" "roiaware_pool3d" "roipoint_pool3d"
  "silu" "three_interpolate" "three_nn"
)

run_triton_one() {
  local NAME=$1 LEVEL=$2 DIRNAME=$3
  local KERNEL_DIR="${AKA_TRITON}/${LEVEL}/${DIRNAME}"
  local OUT_DIR="$BATCH_DIR/triton_${NAME}"
  [[ -f "$OUT_DIR/out/final_report.json" ]] && return 0
  rm -rf "$OUT_DIR" 2>/dev/null; mkdir -p "$OUT_DIR"

  local T0=$(date +%s)
  docker exec \
    -e "GEAK_MODEL=$MODEL" \
    -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
    -e "GEAK_PIPELINE_MODE=mixed" \
    "$TRITON_CONTAINER" bash -c \
    "cd /workspace && geak -t 'Optimize the Triton kernel at ${KERNEL_DIR}/kernel.py. Use the test harness at ${KERNEL_DIR}/test_kernel_harness.py.' \
     --num-parallel 4 --gpu-ids 0,1,2,3 --mode full -y --exit-immediately" \
    > "$OUT_DIR/run.log" 2>&1 || true

  local ELAPSED=$(($(date +%s) - T0))
  # Extract results (same helper as parity test)
  local FINAL="" VS="" STATUS="done"
  [[ -f "$OUT_DIR/out/final_report.json" ]] && \
    FINAL=$(python3 -c "import json; d=json.load(open('$OUT_DIR/out/final_report.json')); print(d.get('total_speedup') or d.get('best_speedup') or '')" 2>/dev/null) || STATUS="no_final_report"
  VS=$(grep -oE "Verified speedup: [0-9.]+x" "$OUT_DIR/run.log" 2>/dev/null | tr '\n' ' ') || true

  echo "${NAME},triton,${LEVEL},mixed,\"${VS}\",${FINAL},${STATUS},${ELAPSED}" >> "$RESULTS_CSV"
}

run_hip_one() {
  local NAME=$1
  local REPO_DIR="${AKA_HIP}/${NAME}"
  local OUT_DIR="$BATCH_DIR/hip_${NAME}"
  [[ -f "$OUT_DIR/out/final_report.json" ]] && return 0
  rm -rf "$OUT_DIR" 2>/dev/null; mkdir -p "$OUT_DIR"

  local KERNEL_PATH=$(find "$REPO_DIR" \( -name '*.hip' \) ! -name '*_ref.hip' ! -path '*/build/*' -print 2>/dev/null | head -1)
  local T0=$(date +%s)
  docker exec \
    -e "GEAK_MODEL=$MODEL" \
    -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
    -e "GEAK_PIPELINE_MODE=mixed" \
    "$HIP_CONTAINER" bash -c \
    "cd /workspace && geak -t 'Optimize the HIP kernel at ${KERNEL_PATH}.' \
     --num-parallel 4 --gpu-ids 4,5,6,7 --mode full -y --exit-immediately" \
    > "$OUT_DIR/run.log" 2>&1 || true

  local ELAPSED=$(($(date +%s) - T0))
  local FINAL="" VS="" STATUS="done"
  [[ -f "$OUT_DIR/out/final_report.json" ]] && \
    FINAL=$(python3 -c "import json; d=json.load(open('$OUT_DIR/out/final_report.json')); print(d.get('total_speedup') or d.get('best_speedup') or '')" 2>/dev/null) || STATUS="no_final_report"
  VS=$(grep -oE "Verified speedup: [0-9.]+x" "$OUT_DIR/run.log" 2>/dev/null | tr '\n' ' ') || true

  echo "${NAME},hip,,mixed,\"${VS}\",${FINAL},${STATUS},${ELAPSED}" >> "$RESULTS_CSV"
}

# Run Triton sequentially on GPUs 0-3
for entry in "${TRITON_KERNELS[@]}"; do
  IFS='|' read -r NAME LEVEL DIRNAME <<< "$entry"
  run_triton_one "$NAME" "$LEVEL" "$DIRNAME"
done &

# Run HIP sequentially on GPUs 4-7 (in parallel with Triton)
for NAME in "${HIP_KERNELS[@]}"; do
  run_hip_one "$NAME"
done &

wait
echo "All runs complete. Results: $RESULTS_CSV"
```

### 6.4 What to look for in logs

The adaptive K is logged by the Dispatcher. Grep for:
```bash
grep "Dispatcher: mode=mixed" run.log
```

Expected output progression:
```
Dispatcher: mode=mixed N=4 K=2 → 2 planned + 2 fill    # Round 1: equal split
Dispatcher: mode=mixed N=4 K=3 → 3 planned + 1 fill    # Round 2+: shifted toward planned
```

Also check per-task outcomes in evaluation JSONs:
```bash
cat out/round_1_evaluation.json | python3 -m json.tool | grep -A3 "per_task"
```

### 6.5 Comparison methodology

After runs complete, compare the `results.csv` against the May 14 baselines above. Key metrics:
- **Win rate:** How many kernels does mixed beat planned (for Triton) or fixed (for HIP)?
- **Geometric mean speedup:** Overall performance across the kernel set
- **Worst-case regression:** Any kernel where mixed is significantly worse?

---

## 7. Important Notes

1. **`GEAK_PIPELINE_MODE=mixed`** must be set in the container env. Without it, `mini.py` defaults to `pipeline_mode = "mixed"` (hardcoded at line 972), so it should work by default — but set it explicitly for clarity.

2. **The adaptive behavior only matters for N >= 3.** With N=2 (2 GPUs), it falls back to K=1 (one of each), same as the old static split. Use N=4 to see the adaptation in action.

3. **Round 1 is always equal split.** The adaptation only kicks in from round 2 onward, after per-task outcomes are available.

4. **Existing test suite:** 132/132 dispatcher/evaluation/pipeline tests pass. Only failure is a pre-existing litellm import error (unrelated).

5. **No config knobs.** The exploration floor (min 1 slot each) is hardcoded. To change it, modify `_k_for_mode` in `selector.py`.

---

## 8. File Reference

| File | Path | Lines changed |
|------|------|---------------|
| Dispatcher | `src/minisweagent/run/dispatcher/selector.py` | 91-133 (adaptive _k_for_mode) |
| Evaluation | `src/minisweagent/run/postprocess/evaluation.py` | 85-95 (_resolve_task_kind), 908+ (per_task population) |
| Dispatch | `src/minisweagent/run/dispatch.py` | 317 (kind passthrough) |
| Unified loop | `src/minisweagent/run/unified.py` | 581 (round_evals threading) |
| Pipeline types | `src/minisweagent/run/pipeline_types.py` | 111-134 (PerTaskOutcome — already existed, now populated) |
| Candidate pool | `src/minisweagent/run/planner/candidate_pool.py` | Unchanged (kind field already existed) |
| Task planner | `src/minisweagent/run/planner/task_planner.py` | Unchanged (canonical fixed injection already existed) |
| Writer | `src/minisweagent/run/dispatcher/writer.py` | Unchanged (kind already written to YAML) |
| Task file I/O | `src/minisweagent/run/task_file.py` | Unchanged |
