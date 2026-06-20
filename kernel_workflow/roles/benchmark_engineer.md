# Benchmark Engineer — Measurement Contract Setup

You build the immutable measurement infrastructure that EVERY other agent must use. Reliability of
the whole workflow depends on this being correct and stable. Operate on the canonical `WORKSPACE`.

## Inputs
`WORKSPACE`, `EVAL_DIR`, `SKILL_DIR`, `GPU_ID`, and `ANALYSIS` (kernel type, files, existing tests).

**DEEP-MODE harness refinement (act ONLY if `HARNESS_ADDENDUM` is in your inputs; otherwise ignore —
a normal run never passes it).** The IMMUTABLE oracle (`unittest.py`/`meta.json`/`reference_io.pt`:
correctness, golden output, tolerance, frozen baseline) is **NEVER modified or re-weighted** — it stays
the source of truth. `HARNESS_ADDENDUM` only refines the PERFORMANCE view so the isolated target predicts
end-to-end: Read it and, in the COMMANDMENT you build, (a) report a SECONDARY e2e-aligned geomean that
weights cases per the addendum (e.g. weight the decode M-buckets that dominate serving) ALONGSIDE the
unweighted oracle geomean, (b) if the addendum specifies a cudagraph capture/replay measurement wrapper,
add it as the FULL_BENCHMARK timing path (so a kernel that only wins eager is exposed), and (c) record the
addendum's hard constraint gates (decode-no-regress, memory-footprint cap, cudagraph-safe) as explicit
PASS/FAIL checks the verify step will enforce. Never let the addendum relax a correctness check.

## Steps

### 1. Discover existing infrastructure (prefer reusing it)
Look for, in order:
- **Author mode**: if the workspace holds an IMMUTABLE `unittest.py` + `meta.json` (the op task dir's
  oracle, copied in read-only by the Director's author-mode setup), THAT is the runner — reuse it
  verbatim. It already does correctness-vs-oracle + per-case timing in the canonical print shape. Do
  NOT write a new harness and do NOT modify it; just point the COMMANDMENT's CORRECTNESS/BENCHMARK at
  `python3 unittest.py` (via gpu_lock) and record the authored baseline from its output.
- `config.yaml` / `config.json` declaring `compile_command` / `correctness_command` /
  `performance_command` (common in GEAK kernels).
- `scripts/task_runner.py` with `compile|correctness|performance` modes.
- `test_*.py` / `*_test.py` / `bench*.py`.

If a runner with compile/correctness/performance exists, USE IT — do not invent a new harness. Read
it to learn the exact commands and the per-case output format it prints (e.g. lines like
`Perf: <ms> ms (<case_id>)`, or `GEAK_RESULT_LATENCY_MS=<ms>`, or a JSON performance report).

### 2. Create a harness only if none exists
If there is no usable runner, write `WORKSPACE/test_harness.py` supporting `--correctness`,
`--profile` (minimal allocations for profiler attach), `--benchmark` (30 iters/10 warmup),
`--full-benchmark` (100 iters/10 warmup). Use CUDA events for timing. Print one line per case:
`GEAK_RESULT_LATENCY_MS=<float>` plus a case id. Correctness must compare to a trusted reference
(PyTorch/naive) with appropriate tolerance. Cover small/medium/large + parameter variations.

### 3. Validate every mode actually runs
Run compile (if any), correctness, benchmark, profile once each (correctness/benchmark via
`gpu_lock.sh $GPU_ID`). Fix anything that errors before continuing.

### 4. Write the COMMANDMENT
Write `EVAL_DIR/COMMANDMENT.md` — the immutable contract. Fill in the EXACT commands discovered/
created. **Run EVERY GPU command (correctness / benchmark / full-benchmark / profile) through
`bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID ...` from inside the workspace dir** — the wrapper not
only serializes GPU access but also (a) isolates the torch cpp_extension build cache per workspace
(`TORCH_EXTENSIONS_DIR=$PWD/.torch_ext`) and (b) compiles only for the local GPU arch. Both are
essential: without (a), parallel engineers compiling `torch.utils.cpp_extension.load(name=...)`
share ONE global cache → they serialize on a single lock and can benchmark each other's `.so`;
without (b) every compile builds ~9 architectures. These are generic to any torch HIP extension.

The COMMANDMENT MUST contain, with concrete commands (not placeholders):
- `SETUP` — `cd <workspace>`. Do NOT use `rm` anywhere in the COMMANDMENT (it triggers an approval
  prompt that blocks autonomous/background runs). Each workspace is already a fresh artifact-free copy
  (build/__pycache__/*.so/.torch_ext excluded at copy time), so there is nothing stale to clear; ninja
  keeps the isolated `.torch_ext/` in sync with sources automatically. If you ever suspect a stale build
  (e.g. after editing headers), MOVE it aside instead of deleting:
  `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true` (a fresh `.torch_ext` rebuilds).
  So `SETUP` is just `cd <workspace>` (plus the env exports below) — no deletion.
- `CORRECTNESS` — wrapped: `cd <workspace> && bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <correctness cmd>`.
- `BENCHMARK` — wrapped in gpu_lock (quick measurement).
- `FULL_BENCHMARK` — wrapped in gpu_lock (authoritative).
- `PROFILE` — `bash $SKILL_DIR/scripts/profile_kernel.sh $GPU_ID "<cmd that cd's into the workspace>" <out_dir>`.
  If the report shows a `!!! PROFILER FAILED` block, follow the fault-tolerance ladder in
  `knowledge/profiling_guide.md` (override the named env var with the corrected flag, or degrade and say so).
- `PARSE` — a one-paragraph description of how to extract per-case latency from the output (the
  exact token/regex and the case-id mapping), so verify/profile engineers parse identically.
- `MODIFIABLE FILES` and the rules (never modify harness/COMMANDMENT/files outside the workspace;
  always run correctness before benchmark; always invoke via gpu_lock from the workspace; benchmark
  output is the source of truth).

### 5. Record baseline + check reliability
Run the FULL benchmark **3 times** via gpu_lock. Confirm per-case results are within ~5% across
runs. If variance is high, investigate (GPU busy? clocks? other procs on this GPU?) and re-run.
Save `EVAL_DIR/baseline_timing.json`:
```json
{
  "test_cases": [{"name": "<case_id>", "latency_ms": 0.0, "params": "..."}],
  "geomean_ms": 0.0,
  "num_test_cases": 0,
  "reliable": true,
  "runs_ms": [[...run1...],[...run2...],[...run3...]]
}
```

## Return JSON
```json
{
  "commandment_path": "<EVAL_DIR>/COMMANDMENT.md",
  "correctness_cmd": "<exact>",
  "benchmark_cmd": "<exact full-benchmark cmd, WITHOUT the gpu_lock wrapper>",
  "profile_cmd": "<exact profile inner cmd>",
  "parse_hint": "how to extract per-case latency + case ids",
  "baseline_per_case": [{"name": "...", "latency_ms": 0.0}],
  "baseline_geomean_ms": 0.0,
  "num_test_cases": 0,
  "reliable": true,
  "notes": "anything downstream agents must know"
}
```
