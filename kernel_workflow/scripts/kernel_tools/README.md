# Shared kernel analysis and screening tools

These CLIs provide evidence for GEAK agents working on AMD Triton/Gluon kernels.
They do not select optimization tactics. The agent combines SKU/roofline, profile and
ISA/IR evidence, changes the kernel and verifies it through the existing workflow.

| Tool | Purpose | Runtime needs |
| --- | --- | --- |
| `asm_loop_audit.py` | Matrix/VALU/memory/wait mix, hot loop, register and LDS context | Python; AMD assembly or disassembly |
| `deep_mfma_analysis.py` | Matrix bursts and instructions strictly between bursts | Python; AMD assembly |
| `asm_schedule_viz.py` | Plot instruction order by class | Above + `matplotlib` when rendering |
| `mfma_efficiency.py` | Observed matrix cadence and intervening issue intervals | Python; already decoded wave/code JSON |
| `ab_bench.py` | Interleaved, controlled A/B screening | Kernel's runtime + GEAK `harness_lib` |
| `create_harness.py` | Generate an editable search harness skeleton | Python to generate; PyTorch/kernel runtime to execute |
| `parse_correctness.py` | Adapt GEAK rc/JSON or legacy oracle text | Python |

`amd_artifacts.py` and `amd_resources.py` share kernel identity and resource parsing;
`harness_stub_env.py` supports CPU-only skeleton tests and is not a benchmark runtime.
Use Python 3.10+. Static tools have no GPU requirement. Install optional rendering with
`python -m pip install matplotlib` in the analysis environment. No matrix-layout,
rocprof, ATT decoder or new mandatory GEAK dependency is required by these CLIs.
The wave reader consumes existing decoded artifacts; collection stays in the profiling
workflow. Native roofline/calibration remains owned by TraceLens.

## Typical calls

Run from the GEAK checkout (replace the example paths and kernel name):

```bash
python kernel_workflow/scripts/kernel_tools/asm_loop_audit.py kernel.s --kernel my_kernel
python kernel_workflow/scripts/kernel_tools/deep_mfma_analysis.py kernel.s --kernel my_kernel
python kernel_workflow/scripts/kernel_tools/asm_schedule_viz.py kernel.s --kernel my_kernel --out schedule.png
python kernel_workflow/scripts/kernel_tools/create_harness.py --kernel-path kernel.py --kernel-type triton --output harness.py
python kernel_workflow/scripts/kernel_tools/ab_bench.py --module adapter.py --baseline base --control auto --cells 6 --permute --json screen.json
python kernel_workflow/scripts/kernel_tools/parse_correctness.py --log unittest.log --parse geak --exit-code 0 --out correctness.json
```

Pass the **actual captured unittest exit code** in the last command. A selected kernel
must match its descriptor and compilation metadata. Ambiguous multi-kernel inputs require
`--kernel`; a unique substring is accepted. `--loop-label` pins the loop within that kernel.
The selector handles LLVM assembly descriptors/YAML and named objdump function bodies;
unrecognizable stripped excerpts cannot supply a trusted kernel identity.

## GEAK contracts

- **Screening:** A/B calls [`harness_lib.time_op`](../../../e2e_workflow/scripts/harness_lib.py).
  GEAK verify executes the task's frozen COMMANDMENT commands for final acceptance.
  E2e-extracted UTs use `measure_legs` internally; other tasks retain their existing runner.
  A shared timer does not automatically align deployment graph/compile mode, workload
  weights, process isolation, the complete correctness suite or acceptance thresholds.
  Pass the desired `--min-improve-pct` explicitly; the local default is 2%.
- **A/B adapters:** `variants()` constructs zero-argument callables. Provide numeric error
  metrics through `oracle(name)` or tensors through `outputs(name)` + `reference(name)`.
  The latter uses `harness_lib.correct`, including matching infinities; NaN/error metrics
  cannot pass. Failed variants are excluded. Missing evidence permits diagnostics only.
  `fingerprint` and `toolchain` hooks guard artifact collisions and compiler mixing.
  Variant names must be non-empty strings; `__control__` is reserved for internal controls.
- **A/B JSON v2:** `validity.valid` must be true before reading any `screen=pass`.
  Both windows retain their own controls, per-cell receipts and iteration effort.
  Bad timers/priming, missing correctness, incomplete rotation, hot-cache mode and
  unstable reversed windows invalidate screening. Exit 0 means valid screening, not a
  speedup or final acceptance; exit 2 means invalid screening/input; exit 1 means a run
  failure. `median` is the ranking key; `min` replaces the old descriptive `stable_min`.
- **Skeletons:** Reuse an existing task runner or immutable UT first. Generate only an
  optional search scaffold; `--output` refuses existing files and symlinks. Fill shapes,
  launch and reference TODOs. Gluon alone gets a layout factory.
  The requested benchmark mode must have cases; empty cases emit
  `UT_HARNESS_INCOMPLETE`/exit 3. Correctness uses `check_correct_multi`; benchmark output
  includes one `GEAK_RESULT_LATENCY_MS` per case and a separately named geomean.
  It does not import workload weights or deployment graph/compile settings, or generate
  random/replay checks. Preserve those requirements and the frozen oracle in the task harness.
  `--sweep-one` is an optional adapter for the legacy `PA_*` sweep protocol; thread its
  knobs into the implementation before using it. It is not automatic kernel generation.
- **Correctness adapter:** GEAK rc 0/1/2/3 maps to pass/fail/environment/incomplete;
  invalid error metrics or explicit failing cases, including children of summary objects,
  cannot be hidden by rc 0. Null error
  metrics remain valid for nonnumeric checks. Legacy text formats are compatibility inputs.
- **Evidence limits:** Static instruction order does not measure overlap. The visualizer's
  legacy `--cycles` option uses heuristic weights, not measured or arch-specific cycles.
  The wave reader cannot infer stall causality or chip utilization from issue intervals.
  Dtype-only hardware constants cannot establish a shape-specific ideal cadence; supply
  `--ideal-cadence` only with evidence for the actual instruction and architecture.

The existing [Gluon script paths](../../../perf_knowledge/expert_skills/skills/gluon_authoring/scripts/)
forward to these shared tools. This layer adds no automatic Claude Code routing; skill
selection and invocation guidance belong to the skill/workflow integration layer.

## Offline smoke

```bash
python -m pytest -q kernel_workflow/scripts/tests/test_kernel_tools_bench.py \
  kernel_workflow/scripts/tests/test_kernel_tools_isa.py \
  kernel_workflow/scripts/tests/test_kernel_tools_regressions.py \
  kernel_workflow/scripts/tests/test_shared_amd_tools.py
```

These tests run without GPU, network or profiling installations. Rendering is exercised
by the visualizer selftest when matplotlib is installed. GPU correctness/performance and
real decoded-trace validation must be recorded separately from these offline checks.
