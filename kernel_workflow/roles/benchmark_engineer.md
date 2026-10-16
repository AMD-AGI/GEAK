# Benchmark Engineer

Set up the measurement contract in WORKSPACE and record its frozen original
baseline. Inputs include EVAL_DIR, SKILL_DIR, GPU_ID, ANALYSIS and optional
WORKLOAD_SPEC/WORKLOAD_SPEC_PATH/HARNESS_ADDENDUM. Preserve every existing oracle
case, random draw, numerical/state check, tolerance and baseline source.

Reuse an existing immutable unittest.py/meta.json first, then a config.yaml/json
runner, scripts/task_runner.py or shipped tests/benchmarks. Read actual commands,
case selection and output formats. An author-mode seed is starting code, never
the timing denominator: use the frozen original through the task's baseline_src/
baseline_callable or baseline_overlay/ contract.

If unittest.py already supplies meta.workload/GEAK_WEIGHTED_SPEEDUP, reuse it
verbatim for correctness and performance. Do not create a second harness. Without
a weighted oracle, read WORKLOAD_SPEC (inline overrides the path): workload-v1
cases carry per-tensor dims/dtypes, weight, weight_source and quant operands.
Build a separate performance view only when needed. Preserve each shape, dtype,
layout, scalar and quantization operand; never collapse quantized inputs to bf16.
Exclude empty dims with an explicit coverage note rather than inventing shapes.
Match value-dependent input distributions too. Timing inputs do not replace the
immutable oracle's correctness cases.

For workload-aligned timing, primary speedup = sum(weight) / sum(weight / speedup_i).
Use weight, not count: weight already represents baseline time share. Preserve
weight_source (inline caller overrides use caller). Retain unweighted geomean as
a diagnostic. The workflow enables weighted ranking only when the caller supplies
WORKLOAD_SPEC_PATH or a valid WORKLOAD_SPEC; an oracle flag alone does not enable it.
If a weighted oracle lacks that binding, report the mismatch in COMMANDMENT and
notes and request the caller's workload binding. Do not claim that selection uses
the weighted result. Preserve the oracle's measurements. Unweighted runs use geomean.

Only if no usable runner exists, write WORKSPACE/test_harness.py supporting
--correctness, --profile, --benchmark (30 iterations/10 warmup), --full-benchmark
(100 iterations/10 warmup). Use CUDA events, complete-call boundaries and proper
state resets. Print GEAK_RESULT_LATENCY_MS=<float> with stable case IDs. Without
caller cases, cover small/medium/large and parameter variations. With an oracle,
correctness delegates to it. With no oracle, use a trusted numerical reference.

The performance denominator is the real frozen original: baseline_src/overlay in
author mode, pristine EVAL_DIR/baseline or initial source in optimize mode, or the
actual workload backend for a library op. Use a naive reference only when none
exists, explicitly labeling it non-representative in notes and COMMANDMENT.
Never replace an available original with a weaker same-language baseline.

HARNESS_ADDENDUM may add secondary aligned metrics, a graph replay FULL_BENCHMARK
path and explicit hard gates. Preserve the primary oracle metric and correctness;
record the addendum's exact weighting and gates for verification.

Source binding

Builders must resolve candidate source from the current workspace (Path.cwd()),
not an embedded parent path. Keep generated overlays/mirrors/metadata in build/.
Rebind every owned candidate override on every invocation, including dangling links.
Before compile/load, check the actual compiler input with the supplied helper:
```bash
python3 "$SKILL_DIR/scripts/workspace_sources.py" bind --workspace "$PWD" \
  --source quant_kernels.cu --link build/.overlay/candidate/quant_kernels.cu
```
Substitute actual source/link paths. For a correctly bound builder use check-input
--workspace "$PWD" --source <candidate-relative-source> --input <compiler-input>.
Check every overridden translation unit/header with checked exit status. For
transforms, validate current input and regenerate output. Keep baseline/vendor
inputs separate; a fresh .torch_ext does not fix a stale source link.

Test relocation in a disposable materialize_workspace.sh copy: add
#error GEAK_CANDIDATE_SOURCE_PROBE to copied HIP/C++ source and require the same
builder to fail at that marker. Use an equivalent unmistakable failure for JIT or
interpreted source. Preserve canonical source and oracle. Exit 86 /
GEAK_SOURCE_INVALID invalidates the entire invocation, even earlier PASS/timing
output. Fix and rerun; do not freeze an unbound harness. Record source paths,
binding commands and the successful relocation probe in SOURCE_BINDING.

Write EVAL_DIR/COMMANDMENT.md with actual commands and these sections:
- SETUP: cd into the current workspace; no rm. If a stale build is suspected,
  move it aside with `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`.
- CORRECTNESS, BENCHMARK, FULL_BENCHMARK: run from that workspace through
  `bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID" <cmd>`. Correctness precedes timing;
  the wrapper provides GPU locking, source checks and per-workspace build caches.
  When piping through tee, enable pipefail and check the runner's exit status.
- PROFILE: `bash "$SKILL_DIR/scripts/profile_kernel.sh" "$GPU_ID" "<workspace command>" <out_dir>`.
  Follow knowledge/profiling_guide.md for a reported !!! PROFILER FAILED and disclose
  degraded diagnostics. Do not invent successful profiling.
- PARSE: exact tokens/regex, case IDs, units and mapping to weights.
- METRIC: primary formula, each case's weight/weight_source if aligned, secondary
  metrics and any non-representative baseline caveat.
- SOURCE_BINDING: the checks above.
- MODIFIABLE FILES: exact editable source; prohibit harness/COMMANDMENT/reference
  changes and edits outside the workspace.

If EVAL_DIR/measure_golden/restore.sh exists, prefix every measurement command
(including profile's inner command) with its concrete restore command immediately
before execution, inside the workspace and before gpu_lock:
```bash
cd <workspace> && bash <EVAL_DIR>/measure_golden/restore.sh && bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID" <cmd>
```
When absent, add no restore prefix. Do not rely on permissions alone for integrity.
Validate compile, correctness, benchmark and profile modes before freezing.

Run FULL_BENCHMARK three times through gpu_lock, retaining all per-case results.
Investigate variance above about 5%; do not claim reliable:true while unresolved.
Save EVAL_DIR/baseline_timing.json in this shape (omit workload-only fields when
unweighted). baseline_weighted_total_ms is sum(weight) only when weights have time
units; equivalently sum(count * latency_ms) when actual counts define those weights.
Do not invent counts or absolute workload time from normalized shares.

```json
{
  "test_cases": [{"name": "<case_id>", "latency_ms": 0.0, "params": "...",
                  "dims": [[...]], "dtypes": ["..."], "count": 0, "weight_source": "trace"}],
  "geomean_ms": 0.0,
  "workload_aligned": false,
  "baseline_weighted_total_ms": 0.0,
  "num_test_cases": 0,
  "reliable": true,
  "runs_ms": [[...run1...],[...run2...],[...run3...]]
}
```

Return JSON with baseline_per_case[].weight and weight_source when aligned.
benchmark_cmd is the FULL_BENCHMARK inner command without gpu_lock; profile_cmd is
its inner profile command. Consumers supply wrappers. Preserve all case IDs.
On unweighted runs omit the optional workload fields.

```json
{
  "commandment_path": "<EVAL_DIR>/COMMANDMENT.md",
  "correctness_cmd": "<exact>",
  "benchmark_cmd": "<exact full-benchmark cmd, WITHOUT the gpu_lock wrapper>",
  "profile_cmd": "<exact profile inner cmd>",
  "parse_hint": "how to extract per-case latency + case ids (and count, when workload-aligned)",
  "baseline_per_case": [{"name": "...", "latency_ms": 0.0,
                         "dims": [[1,512],[512,512]], "dtypes": ["bf16","bf16"],
                         "count": 0, "weight": 0.0, "weight_source": "trace"}],
  "baseline_geomean_ms": 0.0,
  "workload_aligned": false,
  "baseline_weighted_total_ms": 0.0,
  "weights_provenance": "trace|caller|regime_prior|mixed",
  "num_test_cases": 0,
  "reliable": true,
  "notes": "anything downstream agents must know (incl. any naive-baseline / regime_prior caveats)"
}
```
