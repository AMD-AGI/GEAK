# Benchmark Engineer — Measurement Contract Setup

You build the immutable measurement infrastructure that EVERY other agent must use. Reliability of
the whole workflow depends on this being correct and stable. Operate on the canonical `WORKSPACE`.

## Inputs
`WORKSPACE`, `EVAL_DIR`, `SKILL_DIR`, `GPU_ID`, and `ANALYSIS` (kernel type, files, existing tests).

**WORKLOAD ALIGNMENT.** The real-workload shape/dtype distribution is handled by the immutable
`unittest.py` oracle itself — the Kernel Extractor bakes the weighted cases (`meta.workload.cases[]`)
and the time-weighted metric into it. So in the common (e2e-fed) path you do NOTHING special:

- **If the task dir's `unittest.py` is ALREADY workload-weighted** (it prints `GEAK_WEIGHTED_SPEEDUP`
  / `meta.json` has a `workload` key), it is the SINGLE harness for BOTH correctness and the weighted
  perf metric. **REUSE it verbatim, do NOT author a separate performance harness.** Point the
  COMMANDMENT's CORRECTNESS/BENCHMARK/FULL_BENCHMARK/PROFILE at `python3 unittest.py`, and record the
  PRIMARY metric as its `GEAK_WEIGHTED_SPEEDUP = Σ_i weight_i / Σ_i (weight_i / speedup_i)` (the
  unweighted geomean is a secondary diagnostic). Operands/regime are already in-regime in the oracle —
  you do not rebuild them.
- **Only if NO weighted oracle exists but a caller passes `WORKLOAD_SPEC_PATH`/`WORKLOAD_SPEC` inline**
  (a standalone single-kernel run, not an e2e extraction) do you build the weighted perf harness
  yourself (Step 2). Read it (the `workload-v1` schema: `cases[]` each
  `{dims:[[…per-tensor shape…]], dtypes:[…], weight, weight_source, quant}`; `WORKLOAD_SPEC` inline
  overrides the path, `weight_source` becomes `caller`). Then:
  - Benchmark EXACTLY these (dims, dtypes) cases (one harness case each, each tensor with its own shape
    AND dtype + the case's `quant` operands — fp8+scales etc., NOT collapsed to bf16); random values
    (perf is value-independent). A case with empty `dims` cannot be benchmarked — exclude it, say so in
    `notes`, never invent a shape.
  - PRIMARY metric = the time-weighted ratio-of-sums `Σ_i weight_i / Σ_i (weight_i / speedup_i)` using
    each case's `weight` (do NOT use `count`; `weight` already folds frequency × per-call cost).
  - Baseline must be IN-REGIME (the live quantized GEMM / fp8-KV attention / torch.compile-fused path),
    never an unquantized or unfused-eager strawman.
- **If neither** → benchmark the harness's own default cases unweighted (normal run, unchanged).

**CORRECTNESS IS DECOUPLED AND UNCHANGED** in all cases: it runs against the IMMUTABLE frozen oracle
(`unittest.py`/`reference_io.pt`) on its own recorded golden shapes — never re-weighted, replaced, or
relaxed. Random-valued workload-shape inputs are for timing only.

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
  verbatim. It already does correctness-vs-oracle + a random-input parity check vs the frozen online
  baseline + per-case timing in the canonical print shape. Do
  NOT write a new harness and do NOT modify it; just point the COMMANDMENT's CORRECTNESS/BENCHMARK at
  `python3 unittest.py` (via gpu_lock) and record its output. **The `baseline_ms` it prints is the FROZEN
  REAL ONLINE kernel** (`meta.baseline_callable` / `baseline_src/`) — that is the speedup denominator,
  regardless of `TARGET_LANGUAGE`. The authored impl's own timing is the SEED's `optimized_ms` (typically
  slower than the online kernel, i.e. `seed_speedup < 1×`, which is fine); NEVER re-point the denominator
  at the authored same-language scaffold.
- `config.yaml` / `config.json` declaring `compile_command` / `correctness_command` /
  `performance_command` (common in GEAK kernels).
- `scripts/task_runner.py` with `compile|correctness|performance` modes.
- `test_*.py` / `*_test.py` / `bench*.py`.

If a runner with compile/correctness/performance exists, USE IT — do not invent a new harness. Read
it to learn the exact commands and the per-case output format it prints (e.g. lines like
`Perf: <ms> ms (<case_id>)`, or `GEAK_RESULT_LATENCY_MS=<ms>`, or a JSON performance report).

**Workload-weighted oracle present (the e2e-fed path)**: if `unittest.py` already prints
`GEAK_WEIGHTED_SPEEDUP` (extractor baked `meta.workload`), it IS the perf harness too — reuse it for
correctness AND the weighted metric; do NOT author `test_harness.py`. Skip Step 2.

### 2. Create the (performance) harness — only when NOT already covered by a weighted oracle
Write `WORKSPACE/test_harness.py` when there is no usable runner, OR (even if a runner exists) when a
`WORKLOAD_SPEC` is supplied inline AND the oracle `unittest.py` is NOT already workload-weighted — in
that latter case it is the PERFORMANCE harness only; correctness stays on the oracle. Support
`--correctness`, `--profile` (minimal allocations for profiler attach), `--benchmark` (30 iters/10
warmup), `--full-benchmark` (100 iters/10 warmup). Use CUDA events for timing. Print one line per case:
`GEAK_RESULT_LATENCY_MS=<float>` plus a case id.

**Cases:**
- WORKLOAD_SPEC present → one case per spec case, inputs built with each tensor's own `dims`+`dtype`
  (+ scalar params + `quant` operands), random values. Emit the per-case `weight` (and `weight_source`)
  so the parser can compute the time-weighted metric. Exclude empty-`dims` cases (say so in `notes`).
- No WORKLOAD_SPEC → cover small/medium/large + parameter variations (unweighted, as before).

**Baseline (perf reference) — use the ORIGINAL implementation, never an LLM naive reimplementation.**
The speedup denominator must be the real workload code, otherwise "2× over naive torch" can be slower
than production. In order of preference: (a) **author mode: the frozen REAL ONLINE kernel in
`baseline_src/` (via `meta.baseline_callable`)** — the authored from-scratch impl in the target language
is the optimize loop's CODE SEED, NOT the denominator, so a naive-HIP seed is timed against the live
online Triton kernel, never against itself; (b) the pristine original in `EVAL_DIR/baseline` / the
workspace's initial commit (optimize mode always has this); (c) for a library op with no editable
source, the actual default backend the workload uses (e.g. the default GEMM/attention call), as the
extractor's GEMM oracle already does. Only if NONE exists, fall back to a naive PyTorch reference
and FLAG it in `notes` + the COMMANDMENT as a non-representative baseline.

For `--correctness` in the no-runner case (no oracle at all), compare to a trusted reference
(PyTorch/naive) with appropriate tolerance. When the oracle exists, `--correctness` just defers to it.

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
- `METRIC` — define the PRIMARY speedup the optimize loop is judged on:
  - **No WORKLOAD_SPEC**: unweighted geomean of per-case speedups (unchanged default).
  - **WORKLOAD_SPEC present**: the **time-weighted ratio-of-sums**
    `speedup = Σ_i weight_i / Σ_i (weight_i / speedup_i)` (PRIMARY), and ALSO report the unweighted
    geomean as a secondary diagnostic. List each case's `weight` and `weight_source` so every
    downstream agent computes the SAME number. State that this primary number is what the round winner
    gate and the final result use. If the baseline is the flagged naive fallback, say so here.
- `MODIFIABLE FILES` and the rules (never modify harness/COMMANDMENT/files outside the workspace;
  always run correctness before benchmark; always invoke via gpu_lock from the workspace; benchmark
  output is the source of truth).

### 5. Record baseline + check reliability
Run the FULL benchmark **3 times** via gpu_lock. Confirm per-case results are within ~5% across
runs. If variance is high, investigate (GPU busy? clocks? other procs on this GPU?) and re-run.
Save `EVAL_DIR/baseline_timing.json` (the `count`/`dims`/`dtypes`/`weight_source` fields appear only
when a WORKLOAD_SPEC drove the cases; `baseline_weighted_total_ms = Σ count_i·latency_i`):
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

## Return JSON
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
When `workload_aligned` is true, `baseline_per_case[].count` is the coefficient the time-weighted
metric uses, and `weight = count·latency_ms` is the case's time share. On an unweighted run omit the
workload fields entirely (output is identical to before).
