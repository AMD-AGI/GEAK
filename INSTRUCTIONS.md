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

### 1a. DISCOVER: Review Pre-Scanned Discovery Results

**Test discovery was already run by the pre-agent pipeline.** The results
are included in your task context (look for "Discovered Tests" and "Kernel
Analysis" sections).  You do NOT need to re-run discovery manually.

The pre-scan found:
- Kernel type, language, and build info
- Existing test files ranked by confidence with suggested commands
- Existing benchmark files
- Extracted test patterns (tolerances, input shapes, dtypes, import patterns)

**Review the discovery results in your task context:**
1. If a **test harness was already created** by the pre-agent pipeline,
   use it as-is.  Do NOT recreate it.  The path will be noted in your task.
2. If discovery found high-confidence existing tests (confidence > 0.5),
   **read the test file** and reuse its reference implementations, input
   patterns, tolerances, and import patterns.
3. If no pre-built harness exists and discovery found nothing, proceed to
   section 1b to create one from scratch.

Also run the kernel evaluation to verify correctness:
```bash
cd KERNEL_DIR && python3 kernel.py
```

**If you need to re-run discovery manually** (e.g., the pre-scan results are
missing or the kernel path changed), use:
```bash
PYTHONPATH=/workspace:/workspace/src:$PYTHONPATH python3 -c "
from minisweagent.tools.discovery import discover
from pathlib import Path
result = discover(workspace='KERNEL_DIR', kernel_path=Path('KERNEL_DIR/kernel.py'), interactive=False)
print(f'Kernels: {len(result.kernels)}')
print(f'Tests: {len(result.tests)}')
for t in result.tests[:5]:
    print(f'  Test: {t.file_path} (confidence: {t.confidence:.1f})')
    print(f'    Command: {t.command}')
"
```

### 1b. DISCOVER: Build a Test Harness (for non-standard kernels)

**Check first:** If the pre-agent pipeline (UnitTestAgent) already created a
test harness, use it as-is.  The harness is an immutable evaluation contract
— do NOT modify it.  Skip to section 1c.

When no pre-built harness exists, no suitable existing tests are found, or
the kernel file does NOT have a built-in `--profile` flag or standard
`triton_op`/`torch_op` interface, create a **test harness** — a small Python
script that imports the kernel, creates test inputs, and provides
`--correctness`, `--profile`, and `--benchmark` modes.

**If discovery found existing test files**, read them first and reuse:
- Their reference implementations for correctness checking
- Their input generation patterns (shapes, dtypes, edge cases)
- Their tolerance values (atol, rtol)
- Their import patterns (how they import the kernel)

**Common pitfalls to avoid when writing test harnesses:**

1. **Import the kernel via the package path, NOT `importlib.util`.**
   Triton kernels often have deep import chains (e.g.,
   `from aiter.ops.triton._triton_kernels.rope.rope import ...`).
   Using `spec.loader.exec_module` breaks these because the parent package
   isn't initialised.  Instead, add the repo root to `sys.path` and use a
   normal `import` or `from ... import`:
   ```python
   import sys
   sys.path.insert(0, '/path/to/repo/root')
   from aiter.ops.triton.rope.rope import rope_fwd, RotateStyle
   ```

2. **PYTHONPATH must be set BEFORE the process starts, not inline.**
   `kernel-profile` passes the command to `rocprofv3` which uses `execvpe`.
   Inline `PYTHONPATH=... python3 ...` won't work.  Instead, either:
   - Set `PYTHONPATH` in the COMMANDMENT `## SETUP` section, or
   - Create a small wrapper shell script:
   ```bash
   #!/bin/bash
   export PYTHONPATH=/path/to/repo:$PYTHONPATH
   python3 /path/to/test_harness.py "$@"
   ```

3. **Use a fixed random seed** (`torch.manual_seed(42)`) so that correctness
   checks compare deterministic outputs.

4. **Use these FIXED tensor sizes for all tests and profiling.**
   The baseline and every OpenEvolve evaluation MUST use identical input
   dimensions, otherwise speedup numbers are meaningless.  Use these
   standard sizes (large enough to saturate the GPU):
   - **Attention/RoPE kernels:** `S=2048, B=4, H=32, D=128` (fp16)
   - **GEMM kernels:** `M=1024, N=1024, K=1024` (fp16)
   - **Elementwise/pointwise:** at least 16M elements
   Hardcode these in the test harness — do NOT let them vary between runs.

5. **Use `torch.testing.assert_close`** for correctness, NOT manual
   `torch.allclose` with always-pass fallbacks.

6. **The `--profile` mode should run the kernel once** (with minimal setup)
   so that `kernel-profile` / `rocprofv3` captures exactly the kernel(s)
   you care about.  Avoid running benchmarks or loops in profile mode.

7. **Keep the harness file OUTSIDE the kernel directory** or in a fixed
   location that won't be overwritten by OpenEvolve's candidate files.

8. **Generate tensors on CPU, then move to GPU.**
   In `--profile` mode, `rocprofv3` captures ALL GPU kernels — including
   random number generation from `torch.randn(..., device='cuda')`.  This
   pollutes the profiler trace with unrelated kernels.  Instead:
   ```python
   # WRONG — launches GPU RNG kernel that shows up in profiler
   x = torch.randn(S, B, H, D, dtype=torch.float16, device='cuda')
   # CORRECT — RNG on CPU, only the target kernel appears in profiler
   x = torch.randn(S, B, H, D, dtype=torch.float16, device='cpu').to('cuda')
   ```

**Language-specific test harness notes:**

When the kernel is NOT a Python/Triton kernel, the test harness approach varies:

- **HIP/CUDA kernels (.cu, .hip, .cpp):** The test harness should still be a
  Python script that calls the kernel via its pybind11 binding (e.g.,
  `torch.ops.aiter.my_kernel(...)`) or via ctypes. If no Python binding exists,
  create a C++ test that compiles with `hipcc` and outputs timing to stdout.
  Use `--correctness` and `--profile` flags.

- **Composable Kernel (CK):** CK kernels are template-heavy C++. After editing
  template parameters, rebuild with `hipcc` or `cmake`. The test harness should
  import the rebuilt module and call the kernel.

- **Assembly (HSACO):** HSACO binaries are precompiled. You cannot edit the
  assembly. The test harness should test the Python wrapper's launch config
  (grid dims, block dims, shared memory size).

### 1c. DISCOVER: Identify the optimisation target file

**CRITICAL:** When the target kernel file is a **wrapper** that imports the
actual `@triton.jit` kernel from a different file, you MUST optimise the
**inner kernel file** instead of the wrapper.

**Signs of a wrapper:**
- The file imports `@triton.jit` functions from another module (e.g.,
  `from aiter.ops.triton._triton_kernels.rope.rope import _rope_kernel_sbhd_fwd`)
- The file only sets launch parameters (`BLOCK_S`, `num_warps`, `grid`)
- The actual compute logic (`tl.load`, `tl.store`, arithmetic, memory
  access patterns) lives in the imported file

**Why this matters:** Tuning launch parameters alone (BLOCK_S, num_warps,
waves_per_eu) yields limited improvement.  The real optimisation
opportunities (memory coalescing, vectorisation, shared memory usage,
algorithmic changes) are in the `@triton.jit` kernel implementation.

**What to do — DIRECT EDITING (no OpenEvolve):**

When directly optimising (not using OpenEvolve), you MUST edit BOTH files:
1. The **wrapper file** — for launch parameters (BLOCK_S, num_warps, grid)
2. The **inner kernel file** — for algorithmic changes (memory access
   patterns, shared memory staging, vectorisation, tl.load/tl.store patterns)

Edit the inner kernel file directly with sed or cat.  After each edit, run
the test harness for correctness, then re-profile.  Iterate.

Focus on the inner kernel for the biggest gains:
- Improve memory coalescing (the biggest source of latency)
- Add shared memory (LDS) staging for strided access patterns
- Use `tl.trans()` to transpose data in registers
- Vectorise loads/stores for better bandwidth
- Change the loop structure for better pipelining

**CRITICAL — COMMANDMENT.md rules (violating these causes silent failure):**

> 1. MUST use EXACTLY these section headers: `## SETUP`, `## CORRECTNESS`, `## PROFILE`.
>    Any other header (e.g., `## Test Command`, `## Benchmark`) will be **SILENTLY IGNORED**
>    by the OpenEvolve evaluator, resulting in 0 iterations and no speedup.
> 2. Commands must NOT start with shell built-ins (`cd`, `source`, `export`).
>    `rocprofv3` uses `os.execvpe()`, not a shell. Use absolute paths or `bash -c "..."`.
> 3. Commands must NOT use inline env var prefixes like `HIP_VISIBLE_DEVICES=1 python3 ...`.
>    `rocprofv3` treats `HIP_VISIBLE_DEVICES=1` as the executable name and crashes with
>    `FileNotFoundError`. Set env vars in a wrapper script created in `## SETUP`.
> 4. Do NOT set or export `HIP_VISIBLE_DEVICES` — it is ALREADY SET in the environment
>    by the scheduler. Use `${GEAK_GPU_DEVICE}` if you need the GPU ID.
> 5. Each section must contain at least one executable command.

**What to do — OpenEvolve mode:**

1. Trace the import chain to find the file containing the `@triton.jit`
   function that matches the kernel name from profiling
2. Pass that **inner kernel file** to `run_openevolve.py` as `kernel_path`
3. In COMMANDMENT, create a **wrapper shell script** in `## SETUP` that
   sets PYTHONPATH and runs the test harness.  Then use that wrapper in
   `## CORRECTNESS` and `## PROFILE` instead of calling python3 directly.

   **WHY A WRAPPER SCRIPT IS REQUIRED:** The COMMANDMENT evaluator runs
   each command as a separate subprocess.  `export PYTHONPATH=...` in one
   command does NOT persist to subsequent commands.  A wrapper script
   solves this by setting the environment inside the same process that
   runs python3.

4. In the SETUP section:
   a. Create the package directory structure inside `${GEAK_WORK_DIR}`
   b. Copy the candidate to the correct package path within that structure
   c. Create `__init__.py` files for each package level
   d. Write a wrapper script that sets PYTHONPATH with `${GEAK_WORK_DIR}`
      first (so the mutated candidate shadows the original)
   e. Use `printf` on a single line (NOT a heredoc — the COMMANDMENT parser
      splits lines into separate commands)

5. In CORRECTNESS and PROFILE sections, call the wrapper script instead
   of `python3` directly.

**Example COMMANDMENT** for a kernel at
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`:

```
## SETUP
mkdir -p ${GEAK_WORK_DIR}/aiter/ops/triton/_triton_kernels/gemm/basic
cp ${GEAK_WORK_DIR}/gemm_a16w16.py ${GEAK_WORK_DIR}/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py
touch ${GEAK_WORK_DIR}/aiter/__init__.py ${GEAK_WORK_DIR}/aiter/ops/__init__.py ${GEAK_WORK_DIR}/aiter/ops/triton/__init__.py ${GEAK_WORK_DIR}/aiter/ops/triton/_triton_kernels/__init__.py ${GEAK_WORK_DIR}/aiter/ops/triton/_triton_kernels/gemm/__init__.py ${GEAK_WORK_DIR}/aiter/ops/triton/_triton_kernels/gemm/basic/__init__.py
printf '#!/bin/bash\nexport PYTHONPATH=%s:/workspace/.geak_resolved/ROCm_aiter:${PYTHONPATH}\nexport HIP_VISIBLE_DEVICES=%s\npython3 /path/to/test_harness.py "$@"\n' "${GEAK_WORK_DIR}" "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run_harness.sh && chmod +x ${GEAK_WORK_DIR}/run_harness.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run_harness.sh --correctness

## PROFILE
${GEAK_WORK_DIR}/run_harness.sh --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run_harness.sh --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run_harness.sh --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
```

**Key rules:**
- The candidate keeps its original basename (e.g., `gemm_a16w16.py`,
  NOT `kernel.py`)
- PYTHONPATH puts `${GEAK_WORK_DIR}` FIRST so the shadow takes priority
- The `printf` must be a SINGLE LINE (no heredocs in COMMANDMENT)
- Each GPU evaluation gets its own `${GEAK_WORK_DIR}` — no race conditions

---

## 2. PROFILING: kernel-profile (Metrix)

kernel-profile is a hardware profiler.  It can be used at **any stage**:
baseline measurement, post-optimisation validation, or ad-hoc investigation.
OpenEvolve also invokes it during evolution via the COMMANDMENT PROFILE section.

### Running the profiler

**CRITICAL: Always warm up before profiling.** Triton kernels are JIT-compiled
on first invocation.  If you profile a cold run, the measured duration will
include compilation overhead and be much slower than steady-state performance.
The COMMANDMENT.md template includes two warm-up runs before the actual
`kernel-profile` call — your baseline profiling MUST do the same, otherwise
the baseline duration will be inflated and all speedup numbers will be
meaningless (comparing cold baseline vs warm evaluation).

```bash
# Warm up (JIT compile + GPU power ramp) — MUST match COMMANDMENT warm-up
python3 KERNEL_DIR/kernel.py --profile > /dev/null 2>&1 || true
python3 KERNEL_DIR/kernel.py --profile > /dev/null 2>&1 || true
# Now profile (kernel is already compiled and cached)
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

Once you know which kernels to include, use `minisweagent.baseline_metrics`
to format them into the JSON that `run_openevolve.py --baseline-metrics`
expects.  You must tell it **exactly** which kernels to include.

```bash
mkdir -p KERNEL_DIR/optimization_output

# Step 0: Warm up (JIT compile + GPU power ramp) — MUST do before profiling!
python3 KERNEL_DIR/kernel.py --profile > /dev/null 2>&1 || true
python3 KERNEL_DIR/kernel.py --profile > /dev/null 2>&1 || true

# Step 1: Profile and save the raw profiler output (kernel is now warm)
python3 -c "
from geak_agent.mcp_tools.metrix import MetrixTool
from minisweagent.baseline_metrics import list_kernels
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
python3 -m minisweagent.baseline_metrics build \
  KERNEL_DIR/optimization_output/profiler_output.json \
  --kernels "topk_stage1,topk_stage2" \
  -o KERNEL_DIR/optimization_output/baseline_metrics.json
```

Or equivalently from Python:
```python
from minisweagent.baseline_metrics import build_baseline_metrics
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

### Monitoring OpenEvolve in Real-Time

OpenEvolve runs as a subprocess and its stdout is buffered until completion.
To see live progress, **start a background `tail -f` on the progress log
BEFORE launching OpenEvolve**, then run the optimizer:

```bash
# Step 1: Start the progress monitor in the background
OUTPUT_DIR=KERNEL_DIR/optimization_output
mkdir -p ${OUTPUT_DIR}
tail -f ${OUTPUT_DIR}/progress.log 2>/dev/null &
TAIL_PID=$!

# Step 2: Run OpenEvolve (stdout will be buffered, but progress.log updates live)
python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py \
  KERNEL_DIR/kernel.py \
  --iterations 10 --gpu 0 --output ${OUTPUT_DIR} \
  --commandment ${OUTPUT_DIR}/COMMANDMENT.md \
  --baseline-metrics ${OUTPUT_DIR}/baseline_metrics.json

# Step 3: Clean up the tail process
kill $TAIL_PID 2>/dev/null
```

This prints live updates like:
```
ITERATION 3  (186.2s)
  Island 0: 5 programs, best=1.2806
  Island 1: 3 programs, best=1.3072
  *** OVERALL BEST SPEEDUP: 1.3072x ***
```

For detailed logs (LLM calls, per-candidate scores, errors):
```bash
tail -f ${OUTPUT_DIR}/openevolve.log
```

### OpenEvolve Output Files
- `optimization_output/best_kernel.py` — the best optimized kernel found
- `optimization_output/openevolve_result.json` — final results with best score
- `optimization_output/progress.log` — iteration-by-iteration progress (tail -f friendly)
- `optimization_output/openevolve.log` — detailed log (LLM calls, eval results, errors)
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
3. **NEVER copy the candidate INTO `${GEAK_WORK_DIR}` in SETUP** — OpenEvolve writes kernel.py there automatically.  However, you SHOULD use `cp` to place the candidate at the correct import path when optimising an inner kernel file (see Section 1c).
4. Always use `${GEAK_WORK_DIR}/kernel.py` to reference the candidate kernel
5. Always use `${GEAK_GPU_DEVICE}` instead of hardcoded GPU IDs
6. Include TWO warm-up runs before actual profiling (Triton JIT compilation + GPU power ramp). This MUST match the warm-up used during baseline profiling — otherwise speedup numbers will be inflated.
7. Lines starting with `#` (comments) and empty lines are skipped
8. Lines starting with ``` are skipped (don't wrap commands in code fences)
9. Commands run with `cwd=${GEAK_WORK_DIR}`

### Template (replace KERNEL_DIR with actual path)

```
## SETUP
printf '#!/bin/bash\nexport HIP_VISIBLE_DEVICES=%s\nexport PYTHONPATH=%s:${PYTHONPATH}\nexec python3 "$@"\n' "${GEAK_GPU_DEVICE}" "${GEAK_WORK_DIR}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run.sh /workspace/geak-oe/examples/geak_eval/correctness_check.py --baseline KERNEL_DIR/kernel.py --generated ${GEAK_WORK_DIR}/kernel.py

## PROFILE
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
```

**WHY a wrapper script:** All COMMANDMENT commands are run via `os.execvpe()`,
not a shell.  Shell built-ins (`export`, `cd`, `source`) and inline env var
prefixes (`VAR=val cmd`) crash with `FileNotFoundError`.  A wrapper script
sets the environment inside the same process that runs python3.

### Common Mistakes to Avoid
- Adding `cp $GEAK_CANDIDATE_PATH ...` in SETUP — this variable does not exist
- Hardcoding GPU IDs (use `${GEAK_GPU_DEVICE}`)
- Referencing `${GEAK_WORK_DIR}` as the optimization_output dir — it's actually a per-eval temp dir
- Wrapping commands in markdown code fences inside the COMMANDMENT file
- Adding sections like `## SCORING` or `## BASELINE METRICS` — they end the PROFILE section
- Using `--output-dir` instead of `--output` for run_openevolve.py
- Using inline env vars: `HIP_VISIBLE_DEVICES=1 python3 ...` — crashes `rocprofv3`
- Using bare `export`, `cd`, or `source` as command prefixes — use a wrapper script
- Setting `HIP_VISIBLE_DEVICES` at all — it is already set by the scheduler

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
