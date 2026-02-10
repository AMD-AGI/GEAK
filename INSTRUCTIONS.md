# GEAK Agent Pipeline Instructions

READ THIS FILE COMPLETELY before starting any work. It contains the exact
commands, rules, and patterns for every tool in the pipeline.

All paths below use `KERNEL_DIR` as a placeholder. Replace it with the
actual kernel directory (e.g., `/workspace/AIG-Eval/tasks/geak_eval/gemm`).

---

## Tool CLI Reference

### kernel-profile (Metrix hardware profiler)
```
kernel-profile [-h] [--gpu-devices GPU_DEVICES]
               [--replays REPLAYS] [--auto-select] [--quick]
               command

positional arguments:
  command               Command to profile (e.g., "python3 kernel.py --profile")

options:
  --gpu-devices GPU_DEVICES   GPU device ID(s): "0" or "0,1,2" (default: 3)
  --replays REPLAYS           Number of profiling replays (default: 3)
  --auto-select               Automatically select main kernel
  --quick                     Fast profiling (3 metrics, 1 pass)
```

### run_openevolve.py (evolutionary kernel optimizer)
```
python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py [-h]
        [--iterations ITERATIONS] [--gpu GPU] [--output OUTPUT]
        [--config CONFIG] [--api-key API_KEY] [--skip-profiling]
        [--commandment COMMANDMENT] [--baseline-metrics BASELINE_METRICS]
        kernel_path

positional arguments:
  kernel_path                   Path to the kernel file to optimise

options:
  --iterations N, -n N          Max evolution iterations (default: 10)
  --gpu GPU, -g GPU             GPU device ID (default: 0)
  --output OUTPUT, -o OUTPUT    Output directory (default: <kernel_dir>/optimization_output)
  --config CONFIG, -c CONFIG    Path to OpenEvolve config.yaml
  --api-key API_KEY             LLM API key (default: from AMD_LLM_API_KEY env)
  --skip-profiling              Skip Metrix baseline profiling
  --commandment COMMANDMENT     Path to pre-built COMMANDMENT.md (skips auto-build)
  --baseline-metrics BASELINE_METRICS  Path to baseline_metrics.json
```

IMPORTANT: The output flag is `--output` (or `-o`), NOT `--output-dir`.

---

## 1. DISCOVER: Analyze the Kernel

Read `kernel.py` and identify:
- Triton JIT functions (`@triton.jit`)
- Python wrappers (`triton_op`, `torch_op`)
- Evaluation configs (`EVAL_CONFIGS`)
- Whether `--profile` flag is supported
- Supported activations, data types, etc.

Quick discovery command:
```bash
python3 -c "
from geak_agent.mcp_tools.discovery import discover
result = discover(workspace='KERNEL_DIR')
print(f'Kernels: {len(result.kernels)}')
print(f'Tests: {len(result.tests)}')
print(f'Benchmarks: {len(result.benchmarks)}')
"
```

Also run the kernel evaluation to verify correctness:
```bash
cd KERNEL_DIR && python3 kernel.py
```

---

## 2. PROFILING: kernel-profile (Metrix)

kernel-profile is a hardware profiler.  It can be used at **any stage**:
baseline measurement, post-optimisation validation, or ad-hoc investigation.
OpenEvolve also invokes it during evolution via the COMMANDMENT PROFILE section.

### Running the profiler
```bash
kernel-profile "python3 KERNEL_DIR/kernel.py --profile" \
  --gpu-devices 0 --replays 5
```

The profiler reports **every** GPU kernel it observes during the run, not
just the one you intend to optimise.  The output will include framework
overhead (PyTorch internals, memory copies, etc.) alongside the actual
compute kernels.

### YOUR job: choose which kernels matter

Read the profiler output carefully.  Based on the optimisation task:

1. Identify which kernel(s) are the target of optimisation.
2. Decide whether the task involves a single kernel or a group that must
   be considered together (e.g. a fused operation that dispatches multiple
   GPU kernels).
3. Ignore framework overhead unless the task explicitly concerns it.

This decision cannot be automated — it depends on the task context.

### Saving baseline_metrics.json (pre-built COMMANDMENT mode only)

Once you know which kernels to include, use `geakagent.baseline_metrics`
to format them into the JSON that `run_openevolve.py --baseline-metrics`
expects.  You must tell it **exactly** which kernels to include.

```bash
mkdir -p KERNEL_DIR/optimization_output

# Step 1: Profile and save the raw profiler output
python3 -c "
from geak_agent.mcp_tools.metrix import MetrixTool
from geakagent.baseline_metrics import list_kernels
import json

tool = MetrixTool(gpu_devices='0')
result = tool.profile(command='python3 KERNEL_DIR/kernel.py --profile', auto_select=False, num_replays=5, quick=False)
with open('KERNEL_DIR/optimization_output/profiler_output.json', 'w') as f:
    json.dump(result, f, indent=2)

# Print all kernels so you can decide which are relevant
for i, k in enumerate(list_kernels(result)):
    print(f'[{i}] {k[\"duration_us\"]:>10.2f} µs  {k[\"bottleneck\"]:<10}  {k[\"name\"]}')
"

# Step 2: Build baseline_metrics.json from the kernels YOU chose
#   --kernels "name1,name2"   select by exact name
#   --indices 0,2             select by index from the listing above
#   --all                     use every kernel (when only the relevant ones are present)
python3 -m geakagent.baseline_metrics build \
  KERNEL_DIR/optimization_output/profiler_output.json \
  --kernels "topk_stage1,topk_stage2" \
  -o KERNEL_DIR/optimization_output/baseline_metrics.json
```

Or equivalently from Python:
```python
from geakagent.baseline_metrics import build_baseline_metrics
baseline = build_baseline_metrics(result, kernel_names=["topk_stage1", "topk_stage2"])
# or: build_baseline_metrics(result, kernel_indices=[0, 2])
# or: build_baseline_metrics(result, include_all=True)
```

When multiple kernels are selected:
- `duration_us` is **summed** (total wall-time of the group).
- Other hardware metrics are **duration-weighted averages**.
- `bottleneck` and `observations` come from the dominant (longest) kernel.

### Key Metrics
- `duration_us` — kernel execution time in microseconds (PRIMARY metric for scoring)
- `memory.hbm_bandwidth_utilization` — HBM bandwidth usage (%)
- `memory.l2_hit_rate` — L2 cache hit rate (%)
- `memory.coalescing_efficiency` — memory access pattern quality (%)
- Bottleneck classification: memory-bound, compute-bound, latency-bound, etc.

### Profiling after optimisation

After OpenEvolve completes, profile the best kernel to verify the improvement:
```bash
kernel-profile "python3 KERNEL_DIR/optimization_output/best_kernel.py --profile" \
  --gpu-devices 0 --replays 5
```
Compare with the baseline to confirm the speedup is real and not an artefact.

---

## 3. OPTIMIZATION: Run OpenEvolve

There are two modes. Choose ONE.

### Option A: Auto-build mode (RECOMMENDED for standard AIG-Eval kernels)

Use this when the kernel has `triton_op()`, `torch_op()`, `EVAL_CONFIGS`, and
a `--profile` flag. This covers all kernels in `AIG-Eval/tasks/geak_eval/`.

```bash
cd KERNEL_DIR && python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py \
  kernel.py \
  --iterations 10 \
  --gpu 0 \
  --output optimization_output
```

What auto-build does for you:
1. Detects `triton_op`, `torch_op`, `EVAL_CONFIGS`, `--profile` in kernel.py
2. Builds SETUP, CORRECTNESS, and PROFILE commands automatically
3. Validates all commands on the baseline kernel first
4. Writes a frozen COMMANDMENT.md
5. Profiles baseline with Metrix to get baseline_metrics.json
6. Runs OpenEvolve evolutionary optimization

You do NOT need to create COMMANDMENT.md or baseline_metrics.json — it's all automatic.

### Option B: Pre-built COMMANDMENT mode (for non-standard / custom kernels)

Use this when the kernel does NOT follow the standard AIG-Eval interface, or
you need custom correctness checking or profiling commands.

**Step 1:** Profile baseline (see Section 2 above) and save baseline_metrics.json

**Step 2:** Write COMMANDMENT.md (see Section 4 below for format rules)

**Step 3:** Run OpenEvolve with pre-built files:
```bash
cd KERNEL_DIR && python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py \
  kernel.py \
  --iterations 10 \
  --gpu 0 \
  --output optimization_output \
  --commandment optimization_output/COMMANDMENT.md \
  --baseline-metrics optimization_output/baseline_metrics.json
```

### OpenEvolve's profiling freedom

OpenEvolve invokes `kernel-profile` during every candidate evaluation via the
COMMANDMENT PROFILE section.  This is how it scores each candidate against the
baseline.  Do NOT restrict OpenEvolve's ability to profile — the COMMANDMENT
PROFILE section must always include the full profiling command.  OpenEvolve
decides when and how often to profile; the baseline_metrics.json is only the
starting reference point.

### OpenEvolve Output Files
- `optimization_output/best_kernel.py` — the best optimized kernel found
- `optimization_output/openevolve_result.json` — final results with best score
- `optimization_output/progress.log` — iteration-by-iteration progress
- `optimization_output/COMMANDMENT.md` — the frozen evaluation contract
- `optimization_output/evals/` — per-candidate evaluation directories

---

## 4. COMMANDMENT.md Format (CRITICAL RULES)

COMMANDMENT.md is the contract between the agent and OpenEvolve's evaluator.
If you use auto-build mode (Option A), you do NOT need to write this file.
Only read this section if you are using pre-built mode (Option B).

### Environment Variables Set Automatically by the Evaluator
These are available in every command — do NOT set them yourself:
- `${GEAK_WORK_DIR}` — the eval temp directory. The candidate kernel.py is ALREADY here.
- `${GEAK_GPU_DEVICE}` — the GPU device ID for this evaluation
- `${GEAK_KERNEL_DIR}` — the original kernel directory

### CRITICAL RULES
1. Only three section headers are recognized: `## SETUP`, `## CORRECTNESS`, `## PROFILE`
2. Any other `##` header ends the current section (content after it is ignored)
3. **NEVER put a `cp` command in SETUP** — OpenEvolve writes kernel.py into `${GEAK_WORK_DIR}` automatically before running any section
4. Always use `${GEAK_WORK_DIR}/kernel.py` to reference the candidate kernel
5. Always use `${GEAK_GPU_DEVICE}` instead of hardcoded GPU IDs
6. Include TWO warm-up runs before actual profiling (Triton JIT compilation + GPU power ramp)
7. Lines starting with `#` (comments) and empty lines are skipped
8. Lines starting with ``` are skipped (don't wrap commands in code fences)
9. Commands run with `cwd=${GEAK_WORK_DIR}`

### Template (replace KERNEL_DIR with actual path)

```
## SETUP
export HIP_VISIBLE_DEVICES=${GEAK_GPU_DEVICE}
export PYTHONPATH=${GEAK_WORK_DIR}:${PYTHONPATH}

## CORRECTNESS
python3 /workspace/geak-oe/examples/geak_eval/correctness_check.py --baseline KERNEL_DIR/kernel.py --generated ${GEAK_WORK_DIR}/kernel.py

## PROFILE
python3 ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
python3 ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
kernel-profile "python3 ${GEAK_WORK_DIR}/kernel.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
```

### Common Mistakes to Avoid
- Adding `cp $GEAK_CANDIDATE_PATH ...` in SETUP — this variable does not exist
- Hardcoding GPU IDs (use `${GEAK_GPU_DEVICE}`)
- Referencing `${GEAK_WORK_DIR}` as the optimization_output dir — it's actually a per-eval temp dir
- Wrapping commands in markdown code fences inside the COMMANDMENT file
- Adding sections like `## SCORING` or `## BASELINE METRICS` — they end the PROFILE section
- Using `--output-dir` instead of `--output` for run_openevolve.py

---

## 5. Saving Final Results

After OpenEvolve completes:
```bash
cd KERNEL_DIR
cp optimization_output/best_kernel.py kernel_optimized.py
cat optimization_output/openevolve_result.json
```

---

## 6. Environment Reference

- `AMD_LLM_API_KEY` — required for LLM calls (already set in container)
- `HIP_VISIBLE_DEVICES` — GPU selection (set by COMMANDMENT automatically)
- `GEAK_OE_ROOT` — OpenEvolve root (default: /workspace/geak-oe)
- Correctness checker: `/workspace/geak-oe/examples/geak_eval/correctness_check.py`
- OpenEvolve runner: `/workspace/geak-oe/examples/geak_eval/run_openevolve.py`
- kernel-profile: `/opt/venv/bin/kernel-profile`
