---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

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
  command               Command to profile (e.g., "./build/bin/test_kernel --profile")

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

Read the kernel source file(s) (`.cpp`, `.hpp`) and identify:
- CK or CK-tile device instances and kernel entry points
- Template parameters (tile sizes, block sizes, pipeline stages, data types)
- The build system (CMake targets, `hipcc` commands, or Makefile rules)
- Host-side launch wrappers (grid dims, block dims, shared memory)
- Data types, layouts (Row/Col), and operation fusions (bias, activation, etc.)
- Whether a test executable or Python binding already exists

### 1a. DISCOVER: Review Pre-Scanned Discovery Results

**Test discovery was already run by the pre-agent pipeline.** The results
are included in your task context (look for "Discovered Tests" and "Kernel
Analysis" sections).  You do NOT need to re-run discovery manually.

The pre-scan found:
- Kernel type (CK / CK-tile), language (C++/HIP), and build info
- Build system details (CMake targets, hipcc flags)
- Existing test/example executables ranked by confidence with suggested commands
- Existing benchmark executables
- Extracted test patterns (tolerances, input shapes, dtypes, template parameters)

**Review the discovery results in your task context:**
1. If a **test harness was already created** by the pre-agent pipeline,
   use it as-is.  Do NOT recreate it.  The path will be noted in your task.
2. If discovery found high-confidence existing tests (confidence > 0.5),
   **read the test file** and reuse its reference implementations, input
   patterns, tolerances, and build commands.
3. If no pre-built harness exists and discovery found nothing, proceed to
   section 1b to create one from scratch.

Also build and run the kernel to verify it compiles and produces correct output:
```bash
cd KERNEL_DIR && cmake --build build --target <target> -j$(nproc) && ./build/<test_executable>
```

**If you need to re-run discovery manually** (e.g., the pre-scan results are
missing or the kernel path changed), use:
```bash
PYTHONPATH=/workspace:/workspace/src:$PYTHONPATH python3 -c "
from minisweagent.tools.discovery import discover
from pathlib import Path
result = discover(workspace='KERNEL_DIR', kernel_path=Path('KERNEL_DIR/kernel.cpp'), interactive=False)
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

<<<<<<< HEAD
When no pre-built harness exists, no suitable existing tests are found, or
the kernel file does NOT have a built-in `--profile` flag or standard
`triton_op`/`torch_op` interface, create a **test harness** — a small Python
script that imports the kernel, creates test inputs, and provides
`--correctness`, `--profile`, `--benchmark`, `--full-benchmark`, and
`--iterations N` modes.
=======
When no pre-built harness exists or no suitable existing tests are found,
create a **test harness**.  For CK/CK-tile kernels this is typically a
Python script that builds the C++ kernel, runs the compiled executable, and
parses its output.  The harness must provide `--correctness`, `--profile`,
`--benchmark`, and `--full-benchmark` modes.
>>>>>>> 8e7d360 (Change generation instructions to be CK-tile specific)

**If discovery found existing test/example executables**, read them first
and reuse:
- Their reference implementations for correctness checking
- Their input generation patterns (shapes, dtypes, layouts)
- Their tolerance values
- Their build commands and CMake targets

**Common pitfalls to avoid when writing test harnesses for CK/CK-tile:**

1. **Always rebuild after editing.**
   CK/CK-tile kernels are compiled C++.  Every edit to template parameters,
   tile sizes, or kernel logic requires a rebuild.  Use incremental builds:
   ```bash
   cmake --build build --target <target> -j$(nproc)
   ```
   If no `build/` directory exists, configure first:
   ```bash
   mkdir -p build && cd build && cmake -G Ninja .. && ninja -j$(nproc) <target>
   ```
   Prefer Ninja over Make for faster dependency resolution.
   **NEVER delete the build directory** — reuse existing build artifacts.

2. **Understand CK/CK-tile template structure.**
   CK kernels are heavily templated.  Key template parameters include:
   - **Tile sizes:** `MPerBlock`, `NPerBlock`, `KPerBlock`,
     `MPerXDL`, `NPerXDL`, `MXdlPerWave`, `NXdlPerWave`
   - **Pipeline stages:** `LoopScheduler`, `PipelineVersion`
   - **Data types:** `F16`, `BF16`, `F32`, `I8`, etc.
   - **Layouts:** `Row`, `Col` (or `tensor_layout::gemm::RowMajor`, etc.)
   - **Operations:** fused activations (ReLU, GELU), bias, residual add
   CK-tile uses a similar but distinct API with `tile_program`,
   `tile_partition`, and explicit pipeline definitions.

3. **Use a fixed random seed** for host-side input generation so that
   correctness checks are deterministic across runs.

4. **Extract shapes from discovered test files, not hardcoded defaults.**
   The harness must define three shape lists at the top of the script:
   - `ALL_SHAPES`: every unique shape from the discovered test files,
     sorted by total element count.
   - `HARNESS_SHAPES` (20-25): uniformly sampled from ALL_SHAPES. If
     ALL_SHAPES has ≤25 entries, HARNESS_SHAPES = ALL_SHAPES.
   - `PROFILE_SHAPES` (5): evenly-spaced from ALL_SHAPES, prevents OOM.

   Shape routing by CLI mode:
   - `--profile`        → PROFILE_SHAPES (5 shapes)
   - `--benchmark`      → HARNESS_SHAPES (20-25 shapes)
   - `--correctness`    → HARNESS_SHAPES
   - `--full-benchmark` → ALL_SHAPES (every discovered shape)

   The harness must also accept `--iterations N` (default 20) to override
   the number of benchmark iterations for both `--benchmark` and
   `--full-benchmark`.  If the flag is not passed, the harness should read
   `GEAK_BENCHMARK_ITERATIONS` from the environment as a fallback.
   The pipeline sets `GEAK_BENCHMARK_EXTRA_ARGS` to `--iterations 50`
   during evaluation to reduce GPU timing noise.

   If the kernel does NOT have discovered test files, fall back to these
   standard sizes (large enough to saturate the GPU):
   - **GEMM kernels:** `M=1024, N=1024, K=1024` (fp16)
   - **Convolution kernels:** `N=128, C=256, H=28, W=28, K=256, R=3, S=3`
   - **Attention kernels:** `S=2048, B=4, H=32, D=128` (fp16)
   - **Elementwise/pointwise:** at least 16M elements

5. **Validate correctness against a reference implementation.**
   Compare the CK/CK-tile kernel output against a host-side (CPU) or
   known-good GPU reference (e.g., rocBLAS for GEMM, PyTorch for
   convolutions).  Use element-wise comparison with appropriate tolerances
   for the data type (fp16 typically needs `atol=1e-2, rtol=1e-2`).

6. **The `--profile` mode should run the kernel once** (with minimal setup)
   so that `kernel-profile` / `rocprofv3` captures exactly the kernel(s)
   you care about.  Avoid running benchmarks or loops in profile mode.
   **CRITICAL: `--profile` must use ONLY PROFILE_SHAPES (5 shapes) to
   prevent OOM.**

7. **Keep the harness file OUTSIDE the kernel directory** or in a fixed
   location that won't be overwritten by OpenEvolve's candidate files.

8. **Minimize profiler noise.**
   In `--profile` mode, `rocprofv3` captures ALL GPU kernels.  Perform
   all input allocation and initialization before the profiled region.
   If the harness calls the executable, ensure the executable separates
   setup (memory allocation, data init) from the timed kernel launch.

9. **Handle compilation errors gracefully.**
   CK template errors can be verbose.  If the build fails, the harness
   should exit with a non-zero code and print the first ~50 lines of
   compiler output for diagnosis.  Common C++ template errors include:
   - Invalid tile size combinations (tile must divide problem size)
   - Unsupported data type / layout combinations
   - Missing specializations for fused operations

### 1c. DISCOVER: Identify the optimisation target file

**CRITICAL:** CK/CK-tile projects often separate the kernel **device
instance** (template specialization) from the **host launcher** and the
**kernel implementation headers**.  You must identify which file(s) to
optimise.

**Typical CK/CK-tile file structure:**
- **Device instance file** (`.cpp`): Contains the explicit template
  instantiation with specific tile sizes, data types, and layouts.  This
  is where `DeviceOp` template parameters are set (e.g.,
  `DeviceGroupedConvFwdMultipleABD_Xdl<...>`).
- **Kernel implementation headers** (`.hpp`): Contain the actual compute
  logic — the `GridwiseGemm`, `BlockGemm`, or `tile_program` code that
  defines how data moves through LDS, VGPR, and MFMA instructions.
- **Host launcher** (`.cpp`): Sets up problem dimensions, allocates
  memory, and launches the kernel.  Usually not the optimisation target.

**What to optimise:**

The primary optimisation target is usually the **device instance file**
where template parameters are set.  This is where you tune:
- **Tile sizes:** `MPerBlock`, `NPerBlock`, `KPerBlock` — controls how
  work is partitioned across workgroups
- **XDL parameters:** `MPerXDL`, `NPerXDL`, `MXdlPerWave`,
  `NXdlPerWave` — controls MFMA instruction mapping
- **Pipeline configuration:** `LoopScheduler`, `PipelineVersion`,
  `BlockTransferSrcScalarPerVector`, etc.
- **Prefetch stages:** number of software pipeline stages

For deeper algorithmic changes, you may also need to edit the kernel
**implementation headers** (e.g., custom fusions, memory access patterns,
LDS layout changes).

**What to do — DIRECT EDITING (no OpenEvolve):**

When directly optimising (not using OpenEvolve), you MUST:
1. Identify the **device instance file** — the `.cpp` file with the
   template instantiation that sets tile sizes and data types
2. Edit template parameters to explore different tile configurations
3. Rebuild with `cmake --build build --target <target> -j$(nproc)`
4. Run the test harness for correctness, then re-profile
5. Iterate on parameters until performance improves

Focus on these optimisation levers for the biggest gains:
- **Tile size tuning** — match tile sizes to the GPU's MFMA instruction
  capabilities and wave occupancy constraints
- **LDS usage optimisation** — balance LDS capacity against occupancy;
  larger tiles use more LDS but may reduce waves per CU
- **Prefetch depth** — increase pipeline stages to hide global memory
  latency (but increases register pressure)
- **Vector load widths** — use wider vector loads
  (`BlockTransferSrcScalarPerVector`) to maximise HBM bandwidth
- **Memory access patterns** — ensure coalesced global memory access
  and bank-conflict-free LDS access
- **Occupancy tuning** — balance between larger tiles (more compute per
  wave) and more concurrent waves per CU

**CRITICAL — COMMANDMENT.md rules (violating these causes silent failure):**

> 1. MUST use EXACTLY these section headers: `## SETUP`, `## CORRECTNESS`, `## PROFILE`,
>    `## BENCHMARK`, `## FULL_BENCHMARK`.
>    Any other header will be flagged as an error by `validate_commandment`.
> 2. Commands must NOT start with shell built-ins (`cd`, `source`, `export`).
>    `rocprofv3` uses `os.execvpe()`, not a shell. Use absolute paths or `bash -c "..."`.
> 3. Commands must NOT use inline env var prefixes like `HIP_VISIBLE_DEVICES=1 python3 ...`.
>    `rocprofv3` treats `HIP_VISIBLE_DEVICES=1` as the executable name and crashes with
>    `FileNotFoundError`. Set env vars in a wrapper script created in `## SETUP`.
> 4. Do NOT set or export `HIP_VISIBLE_DEVICES` — it is ALREADY SET in the environment
>    by the scheduler. Use `${GEAK_GPU_DEVICE}` if you need the GPU ID.
> 5. Each section must contain at least one executable command.

**What to do — OpenEvolve mode:**

1. Identify the **device instance file** (the `.cpp` file containing the
   template instantiation with tile sizes and data type parameters)
2. Pass that file to `run_openevolve.py` as `kernel_path`
3. In COMMANDMENT, the `## SETUP` section must:
   a. Copy the mutated candidate `.cpp` file to the correct source
      location in the build tree
   b. Rebuild the target with `cmake --build` or `hipcc`
   c. Create a wrapper script for running the test harness

   **WHY A WRAPPER SCRIPT IS REQUIRED:** The COMMANDMENT evaluator runs
   each command as a separate subprocess.  `export` in one command does
   NOT persist to subsequent commands.  A wrapper script sets the
   environment inside the same process that runs the test.

4. In the SETUP section:
   a. Copy the candidate `.cpp` to the correct source path in the project
   b. Rebuild incrementally: `cmake --build <build_dir> --target <target> -j$(nproc)`
   c. Write a wrapper script that sets `HIP_VISIBLE_DEVICES` and runs
      the test harness or compiled executable
   d. Use `printf` on a single line (NOT a heredoc — the COMMANDMENT parser
      splits lines into separate commands)

5. In CORRECTNESS and PROFILE sections, call the wrapper script or
   the built executable.

**Example COMMANDMENT** for a CK kernel at
`library/src/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.cpp`:

```
## SETUP
cp ${GEAK_WORK_DIR}/kernel.cpp /workspace/composable_kernel/library/src/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.cpp
cmake --build /workspace/composable_kernel/build --target test_grouped_conv_fwd -j$(nproc)
printf '#!/bin/bash\nexport HIP_VISIBLE_DEVICES=%s\n/workspace/composable_kernel/build/bin/test_grouped_conv_fwd "$@"\n' "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run_harness.sh && chmod +x ${GEAK_WORK_DIR}/run_harness.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run_harness.sh --correctness

## PROFILE
${GEAK_WORK_DIR}/run_harness.sh --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run_harness.sh --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run_harness.sh --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
```

**Key rules:**
- The candidate `.cpp` is copied to the correct source tree path before
  rebuilding — incremental builds only recompile the changed file
- The `printf` must be a SINGLE LINE (no heredocs in COMMANDMENT)
- Each GPU evaluation gets its own `${GEAK_WORK_DIR}` — no race conditions
- Build errors (invalid template parameters) cause the SETUP step to
  fail, which correctly rejects that candidate

---

## 2. PROFILING: kernel-profile (Metrix)

kernel-profile is a hardware profiler.  It can be used at **any stage**:
baseline measurement, post-optimisation validation, or ad-hoc investigation.
OpenEvolve also invokes it during evolution via the COMMANDMENT PROFILE section.

### Running the profiler

**CRITICAL: Always warm up before profiling.** Even though CK/CK-tile
kernels are pre-compiled (no JIT overhead), the GPU needs a warm-up run to
stabilise clock frequencies and power states.  The COMMANDMENT.md template
includes two warm-up runs before the actual `kernel-profile` call — your
baseline profiling MUST do the same, otherwise the baseline duration will
be inflated and all speedup numbers will be meaningless.

```bash
# Warm up (GPU power ramp) — MUST match COMMANDMENT warm-up
KERNEL_DIR/build/bin/test_kernel --profile > /dev/null 2>&1 || true
KERNEL_DIR/build/bin/test_kernel --profile > /dev/null 2>&1 || true
# Now profile (GPU clocks are stable)
kernel-profile "KERNEL_DIR/build/bin/test_kernel --profile" \
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

# Step 0: Warm up (GPU power ramp) — MUST do before profiling!
KERNEL_DIR/build/bin/test_kernel --profile > /dev/null 2>&1 || true
KERNEL_DIR/build/bin/test_kernel --profile > /dev/null 2>&1 || true

# Step 1: Profile and save the raw profiler output (GPU is now warm)
python3 -c "
from metrix_mcp.core import MetrixTool
from minisweagent.baseline_metrics import list_kernels
import json

tool = MetrixTool(gpu_devices='0')
result = tool.profile(command='KERNEL_DIR/build/bin/test_kernel --profile', auto_select=False, num_replays=5, quick=False)
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
  --kernels "gridwise_gemm,block_gemm_pipeline" \
  -o KERNEL_DIR/optimization_output/baseline_metrics.json
```

Or equivalently from Python:
```python
from minisweagent.baseline_metrics import build_baseline_metrics
baseline = build_baseline_metrics(result, kernel_names=["gridwise_gemm", "block_gemm_pipeline"])
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

After OpenEvolve completes, rebuild with the best kernel source and profile
to verify the improvement:
```bash
# Copy the best candidate back to the source tree and rebuild
cp KERNEL_DIR/optimization_output/best_kernel.cpp /path/to/source/tree/kernel_instance.cpp
cmake --build KERNEL_DIR/build --target <target> -j$(nproc)
# Profile the rebuilt kernel
kernel-profile "KERNEL_DIR/build/bin/test_kernel --profile" \
  --gpu-devices 0 --replays 5
```
Compare with the baseline to confirm the speedup is real and not an artefact.

### Apples-to-apples speedup comparison (CRITICAL)

The test harness has two benchmark modes that use **different** shape sets:
- `--benchmark` uses HARNESS_SHAPES (20-25 sampled shapes)
- `--full-benchmark` uses ALL_SHAPES (every discovered shape)

**You MUST compare matching modes.** Comparing `--full-benchmark` baseline
against `--benchmark` iteration results (or vice versa) produces meaningless
speedup numbers because the shape mix is different.

**Baseline setup:** Run BOTH modes on the unmodified kernel and record both
results separately:
```bash
# Reduced baseline (for comparing during iterations)
python3 test_harness.py --benchmark > baseline_benchmark.txt
# Full baseline (for start/end comparison)
python3 test_harness.py --full-benchmark > baseline_full_benchmark.txt
```

**During optimization iterations:** Use `--benchmark` (reduced) and compare
against the **reduced baseline** only.

**At the end of optimization:** Run `--full-benchmark` on the best kernel
and compare against the **full baseline** to report the final speedup.

**Summary table:**

| When                     | Run mode           | Compare against          |
|--------------------------|--------------------|--------------------------|
| Baseline (start)         | --benchmark AND --full-benchmark | (record both)  |
| Each iteration           | --benchmark        | baseline --benchmark     |
| Final result             | --full-benchmark   | baseline --full-benchmark|

Never mix modes in a comparison. If you only have a `--full-benchmark`
baseline, re-run `--benchmark` on the original kernel before comparing
with iteration results.

---

## 3. OPTIMIZATION: Run OpenEvolve

CK/CK-tile kernels use the **pre-built COMMANDMENT mode** because they
require a C++ build step that auto-build cannot infer.

### Workflow for CK/CK-tile kernels

**Step 1:** Build the baseline and profile it (see Section 2 above).
Save `baseline_metrics.json`.

**Step 2:** Write `COMMANDMENT.md` (see Section 4 below for format rules).
The COMMANDMENT must include:
- `## SETUP`: Copy the mutated `.cpp` candidate to the source tree and
  rebuild with `cmake --build` or `hipcc`
- `## CORRECTNESS`: Run the correctness test
- `## PROFILE`: Run the kernel for profiling

**Step 3:** Run OpenEvolve with pre-built files:
```bash
cd KERNEL_DIR && python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py \
  kernel_instance.cpp \
  --iterations 10 \
  --gpu 0 \
  --output optimization_output \
  --commandment optimization_output/COMMANDMENT.md \
  --baseline-metrics optimization_output/baseline_metrics.json
```

<<<<<<< HEAD
### OpenEvolve's evaluation sections
=======
**Note on kernel_path:** Pass the `.cpp` device instance file (the file
containing template parameter instantiations) as `kernel_path`.  OpenEvolve
will mutate this file's template parameters across iterations.

IMPORTANT: The output flag is `--output` (or `-o`), NOT `--output-dir`.

### OpenEvolve's profiling freedom
>>>>>>> 8e7d360 (Change generation instructions to be CK-tile specific)

OpenEvolve reads the `## BENCHMARK` section from COMMANDMENT.md for per-iteration
fitness evaluation.  This runs the harness with `--benchmark` (wall-clock latency
on 20-25 HARNESS_SHAPES) and produces a speedup ratio against the baseline.

The `## PROFILE` section (deep hardware analysis via Metrix) is NOT run per-iteration
by OpenEvolve.  It remains in COMMANDMENT for the orchestrator's per-round evaluation
and for on-demand `profile_kernel` calls by agents.

**geak-oe repo change required**: In `run_openevolve.py`, the evaluator must read
`## BENCHMARK` (not `## PROFILE`) from COMMANDMENT.md and parse wall-clock median
latency from stdout for the fitness function (`baseline_latency / candidate_latency`).

**Build failures are expected:** When OpenEvolve mutates template parameters,
some combinations will produce invalid C++ (e.g., tile sizes that exceed LDS
capacity or unsupported type combinations).  These candidates will fail at
the SETUP build step, which is the correct behaviour — they are automatically
rejected and the evolution continues.

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
  KERNEL_DIR/kernel_instance.cpp \
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
- `optimization_output/best_kernel.cpp` — the best optimized kernel source found
- `optimization_output/openevolve_result.json` — final results with best score
- `optimization_output/progress.log` — iteration-by-iteration progress (tail -f friendly)
- `optimization_output/openevolve.log` — detailed log (LLM calls, eval results, errors)
- `optimization_output/COMMANDMENT.md` — the frozen evaluation contract
- `optimization_output/evals/` — per-candidate evaluation directories

---

## 4. COMMANDMENT.md Format (CRITICAL RULES)

COMMANDMENT.md is the contract between the agent and OpenEvolve's evaluator.
CK/CK-tile kernels always use the pre-built COMMANDMENT mode because a C++
build step is required.

### Environment Variables Set Automatically by the Evaluator
These are available in every command — do NOT set them yourself:
- `${GEAK_WORK_DIR}` — the eval temp directory. The candidate `.cpp` file is ALREADY here as `kernel.cpp`.
- `${GEAK_GPU_DEVICE}` — the GPU device ID for this evaluation
- `${GEAK_KERNEL_DIR}` — the original kernel directory

### CRITICAL RULES
<<<<<<< HEAD
1. Five section headers are recognized: `## SETUP`, `## CORRECTNESS`, `## PROFILE`, `## BENCHMARK`, `## FULL_BENCHMARK`
2. Any other `##` header is flagged as an error by `validate_commandment`
3. **NEVER copy the candidate INTO `${GEAK_WORK_DIR}` in SETUP** — OpenEvolve writes kernel.py there automatically.  However, you SHOULD use `cp` to place the candidate at the correct import path when optimising an inner kernel file (see Section 1c).
4. Always use `${GEAK_WORK_DIR}/kernel.py` to reference the candidate kernel
5. Always use `${GEAK_GPU_DEVICE}` instead of hardcoded GPU IDs
6. Include TWO warm-up runs before actual profiling (Triton JIT compilation + GPU power ramp). This MUST match the warm-up used during baseline profiling — otherwise speedup numbers will be inflated.
7. Lines starting with `#` (comments) and empty lines are skipped
8. Lines starting with ``` are skipped (don't wrap commands in code fences)
9. Commands run with `cwd=${GEAK_WORK_DIR}`
=======
1. Only three section headers are recognized: `## SETUP`, `## CORRECTNESS`, `## PROFILE`
2. Any other `##` header ends the current section (content after it is ignored)
3. The candidate `.cpp` is already in `${GEAK_WORK_DIR}/kernel.cpp` — you MUST `cp` it to the correct source tree location before rebuilding
4. Always use `${GEAK_GPU_DEVICE}` instead of hardcoded GPU IDs
5. Include TWO warm-up runs before actual profiling (GPU power ramp). This MUST match the warm-up used during baseline profiling — otherwise speedup numbers will be inflated.
6. Lines starting with `#` (comments) and empty lines are skipped
7. Lines starting with ``` are skipped (don't wrap commands in code fences)
8. Commands run with `cwd=${GEAK_WORK_DIR}`
9. The SETUP section MUST include the `cmake --build` or `hipcc` command — if the build fails (invalid template parameters), the candidate is correctly rejected
>>>>>>> 8e7d360 (Change generation instructions to be CK-tile specific)

### Template (replace KERNEL_DIR and paths with actual values)

```
## SETUP
cp ${GEAK_WORK_DIR}/kernel.cpp /workspace/composable_kernel/library/src/tensor_operation_instance/gpu/<subdir>/kernel_instance.cpp
cmake --build /workspace/composable_kernel/build --target <cmake_target> -j$(nproc)
printf '#!/bin/bash\nexport HIP_VISIBLE_DEVICES=%s\n/workspace/composable_kernel/build/bin/<test_executable> "$@"\n' "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run.sh --correctness

## PROFILE
<<<<<<< HEAD
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5

## BENCHMARK
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --benchmark ${GEAK_BENCHMARK_EXTRA_ARGS:-}

## FULL_BENCHMARK
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --full-benchmark ${GEAK_BENCHMARK_EXTRA_ARGS:-}
=======
${GEAK_WORK_DIR}/run.sh --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run.sh --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run.sh --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
>>>>>>> 8e7d360 (Change generation instructions to be CK-tile specific)
```

**WHY a wrapper script:** All COMMANDMENT commands are run via `os.execvpe()`,
not a shell.  Shell built-ins (`export`, `cd`, `source`) and inline env var
prefixes (`VAR=val cmd`) crash with `FileNotFoundError`.  A wrapper script
sets the environment inside the same process that runs the executable.

### Common Mistakes to Avoid
- Forgetting to `cp` the candidate to the source tree before building — the build will use the old source
- Hardcoding GPU IDs (use `${GEAK_GPU_DEVICE}`)
- Referencing `${GEAK_WORK_DIR}` as the optimization_output dir — it's actually a per-eval temp dir
- Wrapping commands in markdown code fences inside the COMMANDMENT file
- Adding unrecognized sections like `## SCORING` or `## BASELINE METRICS`
- Using `--output-dir` instead of `--output` for run_openevolve.py
- Using inline env vars: `HIP_VISIBLE_DEVICES=1 ./test_kernel ...` — crashes `rocprofv3`
- Using bare `export`, `cd`, or `source` as command prefixes — use a wrapper script
- Setting `HIP_VISIBLE_DEVICES` at all — it is already set by the scheduler
- Deleting the build directory (`rm -rf build`) — always use incremental builds
- Not using `-j$(nproc)` for parallel builds — single-threaded CK builds are extremely slow

---

## 5. Saving Final Results

After OpenEvolve completes:
```bash
cd KERNEL_DIR
cp optimization_output/best_kernel.cpp kernel_optimized.cpp
cat optimization_output/openevolve_result.json
```

To apply the optimised kernel back to the source tree:
```bash
cp kernel_optimized.cpp /path/to/source/tree/kernel_instance.cpp
cmake --build build --target <target> -j$(nproc)
```

---

## 6. Environment Reference

### Variables set automatically by the pipeline

- `HIP_VISIBLE_DEVICES` — GPU selection (set by COMMANDMENT / scheduler)
- `GEAK_WORK_DIR` — per-evaluation temp directory (candidate kernel lives here)
- `GEAK_GPU_DEVICE` — GPU device ID for this evaluation
- `GEAK_REPO_ROOT` — original repository root
- `GEAK_HARNESS` — absolute path to the test harness script
- `GEAK_BENCHMARK_EXTRA_ARGS` — extra CLI args appended to `--benchmark` and
  `--full-benchmark` invocations (default: `--iterations 50`).  Controls the
  number of benchmark iterations used by preprocessing baselines, agent
  benchmarks, and orchestrator evaluations to ensure consistency.
- `GEAK_BENCHMARK_ITERATIONS` — alternative fallback read by the harness
  itself when `--iterations` is not passed on the command line.

### User-configurable

- `AMD_LLM_API_KEY` — required for LLM calls (already set in container)
- `GEAK_OE_ROOT` — OpenEvolve root (default: /workspace/geak-oe)
- `GEAK_MAX_ROUNDS` — maximum orchestration rounds (default: 5)
- `GEAK_ALLOWED_AGENTS` — comma-separated allowlist of agent types
- `GEAK_EXCLUDED_AGENTS` — comma-separated blocklist of agent types

### Tool paths

- Correctness checker: `/workspace/geak-oe/examples/geak_eval/correctness_check.py`
- OpenEvolve runner: `/workspace/geak-oe/examples/geak_eval/run_openevolve.py`
- kernel-profile: `/opt/venv/bin/kernel-profile`

### Patch exclusions

When generating patches, the following patterns are excluded from `git diff`
to prevent binary artifacts and build cache from polluting patches:
`traj.json`, `*.log`, `.rocprofv3/`, `__pycache__/`, `*.pyc`,
`.pytest_cache/`, `*.egg-info/`, `*.so`, `.geak_resolved/`.
