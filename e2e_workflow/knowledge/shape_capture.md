# Shape & I/O Capture — Turning a Hot Kernel into a Standalone Unittest

This is the **Kernel Extractor's** playbook. Its output is a self-contained kernel task dir that the
UNCHANGED single-kernel `kernel_workflow` consumes — same contract as a hand-written kernel task. The
critical property: the unittest must replay the kernel's REAL serving shapes and check correctness
against a recorded I/O oracle, and it must be **immutable during optimization** (anti-cheating).

## What the kernel layer expects (the task-dir contract)
The single-kernel workflow takes `args.kernel_path` = a directory containing the kernel source +
wrapper + a unittest that (1) optionally builds, (2) runs, (3) checks correctness, (4) reports
per-case speedup. The Extractor must produce exactly this shape so the kernel layer runs unmodified:
```
<task_dir>/
  kernel_src...        # the extracted source (copied from sglang/aiter overlay subtree, editable)
  reference_io.pt      # recorded inputs + golden outputs (the oracle) — READ-ONLY for optimizers
  unittest.py          # builds(opt)/runs/checks-correctness/times speedup; IMMUTABLE during opt
  meta.json            # kernel name, source path in sglang, shapes, dtypes, backend, regime
```

## Step 1 — Capture real shapes AND reference I/O from the live server
Profiles give shapes, but optimization needs the actual tensor VALUES (the oracle) so a backend swap
can be proven numerically correct. Capture both with one hook, driven by the SAME bench workload the
Profiler used (so shapes match the regime).

`scripts/capture_shapes.py` installs a wrapper around the target callable (the Triton entry fn or the
python op that dispatches the kernel) via the overlay monkeypatch mechanism
([[sglang_internals]] §3b), runs a SHORT bounded window of the bench, and for the first N distinct
input-shape signatures records `(args, kwargs) -> output` to `reference_io.pt`. Key rules:
- **Detach + clone to CPU** (or keep on-device but snapshot) so later in-place ops can't corrupt the
  oracle. Record dtype, device, stride/contiguity, and any non-tensor scalar args (seeds, scales).
- **Distinct-shape dedup**: one record per shape signature, up to `--max-cases` (default 5). For a
  kernel serving both prefill (large M) and decode (small M) regimes, this naturally captures BOTH →
  the unittest gets multi-case coverage and the kernel squad can build regime-specific variants.
- **Determinism**: capture with temp=0 so re-running the reference is reproducible.
- Bound the window (`--num-steps`) so capture is fast and the file stays small.

### Step 1b — Graph-hidden shapes: MEASURE the m_buckets with the per-shape probe
`capture_shapes.py` above works when the kernel actually runs through Python during the capture
window. But most GEMMs / decode-path kernels run inside a **CUDA/HIP graph**: on graph *replay* the
kernel is NOT dispatched through Python, so the profiler AND the plain capture see `dims=[]` for the
decode calls. The only Python-visible calls are the graph *capture* pass (a few per size bucket) —
not the real serving frequency. Historically the extractor then fell back to an **inferred**
`M ≈ conc` guess (e.g. gpt-oss `M=64/256`), which is a gamble.

To get the REAL per-shape distribution, run the server with **CUDA graph OFF** (`--enforce-eager`) so
every decode step goes through Python, and use the per-shape probe engine `capture_shapes_probe.py`
(a sibling of `capture_shapes.py`, purpose-built for this):
- **No `max_cases` cap** — accumulates the FULL per-shape call-count distribution, not the first 5.
- **Does NOT snapshot tensors** — records only `dims`/`dtype`/`count` (+ optional cuda.Event GPU
  time), so it is light enough to sit on the hot path and never dumps `reference_io.pt` (correctness
  oracle still comes from Step 1 with the unmodified kernel).
- Output → `probe_postprocess.py` → `probe_to_mbuckets.py` yields
  `{decode_m_buckets, prefill_m_buckets}` to MERGE into meta.json, replacing the `M≈conc` guess.
  Everything downstream (`attribute_weights`, `unittest.py`) is unchanged — only the values improve.

Use Step 1b ONLY for graph-hidden kernels (Step 1 returned empty decode dims). Non-graph kernels keep
the Step-1 captured shapes. See `roles/kernel_extractor.md` step 2b for the exact commands.

## Step 2 — Emit an immutable, general unittest
The unittest must be backend-agnostic: it loads `reference_io.pt`, calls whatever the CURRENT kernel
entry point is on the recorded inputs, compares to the golden output (tolerance per dtype:
bf16/fp16 → `rtol=2e-2, atol=2e-2` typical; fp8 looser; fp32 tight), and times baseline-vs-current.
Because it pins inputs+oracle and never imports a specific backend by name, it transparently judges a
Triton / HIP / CK / aiter / asm reimplementation — the optimizer just has to make the entry point
fast AND match the oracle. This is what makes the unittest "general" per the spec.

Anti-cheating (inherited from the single-kernel COMMANDMENT contract):
- The optimizer MUST NOT edit `unittest.py` or `reference_io.pt`. The Extractor records a checksum in
  `meta.json`; the e2e Integrator/Validator re-checks it before trusting any speedup.
- Correctness is judged ONLY against the recorded oracle, not against a re-run of the same code path
  (which would let a no-op "pass").

## Step 3 — Build (optional) + speedup contract
Some kernels are pure Python/Triton (no build step) — `meta.json.build=false`. Others (HIP/CK/asm
candidates) need a compile — `meta.json.build=true` with a build command. The unittest's speedup
number is per-case `baseline_ms / optimized_ms`, geomean over cases — identical to the single-kernel
workflow so the kernel layer's Director/verify_engineer math is unchanged.

## Step 4 — Hand off to the kernel layer, then overlay back
- Extractor returns the `task_dir`; the orchestration calls `kernel_workflow.js` with
  `kernel_path=task_dir`. That recursive run does the real multi-backend optimization + verification
  and returns a `final_patch.diff` against the extracted source.
- The e2e Integrator maps that patch onto the sglang overlay subtree ([[sglang_internals]] §3),
  relaunches a warm server, and validates END-TO-END throughput + output parity. A kernel win is
  accepted into e2e ONLY if (a) the isolated unittest speedup is real, (b) Amdahl says it can move
  the needle, and (c) the measured e2e throughput delta exceeds the noise band.

## Common pitfalls
- **Wrong entry granularity**: hooking too deep (a single Triton `@jit`) misses host-side reshape
  cost; hooking too shallow (a whole layer) makes the unittest non-portable. Hook the smallest
  callable that owns the kernel's inputs+outputs as plain tensors.
- **Shape drift**: capture with the EXACT ISL/OSL/concurrency of the throughput bench, after warmup,
  or the unittest optimizes the wrong regime.
- **Hidden state**: kernels reading global config / KV cache need those captured as inputs too, or
  the oracle won't reproduce. Record everything the callable reads.
- **In-place outputs**: if the kernel writes into a passed-in buffer, snapshot the buffer BEFORE the
  call as input and AFTER as the oracle output.

### Per-shape probe (Step 1b) hard pitfalls — learned the hard way
- **Lazy hook, never eager-import**: the probe overlay MUST NOT `import` a heavy lib (e.g. aiter) at
  interpreter start — eager-loading it on the EngineCore handshake path stalls startup and the server
  never becomes healthy. Use a passive meta-path finder that wraps the target only once vLLM itself
  imports it. (`capture_shapes_probe.py` already does this.)
- **SIGTERM, not atexit**: vLLM tears down its worker processes with SIGTERM (and overrides the
  SIGTERM handler), so `atexit` flush does NOT run → data lost. The probe must flush periodically (a
  daemon snapshot every few seconds), independent of the exit path.
- **Scan kwargs too**: some ops are called entirely by keyword (e.g. `unified_attention(q=..,k=..,v=..)`).
  A positional-only scan records `dims=[]`. Scan both args and kwargs.
- **JITFunction guard**: a Triton `@jit` kernel is called as `fn[grid](...)`; wrapping it with a plain
  function breaks the `[grid]` launch syntax and crashes the forward. Detect JITFunction and skip it
  (it's usually a tiny op anyway). `capture_shapes_probe.py` guards this.
- **enforce-eager is mandatory for the probe**: with CUDA graph ON the probe only sees the capture-pass
  calls (a handful per bucket), which is NOT the real serving count. Always probe with `--enforce-eager`.
