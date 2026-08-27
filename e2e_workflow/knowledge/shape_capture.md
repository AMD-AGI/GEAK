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
  unittest.py          # builds(opt)/runs/checks-correctness/times speedup; IMMUTABLE during opt
  meta.json            # kernel name, source path in sglang, shapes, dtypes, backend, regime
```

## Step 1 — Capture real shapes from the live server
Profiles give shapes, but they are aggregated and lossy: optimization needs the exact operand
signature (dims, dtypes, layout/contiguity, non-tensor scalars, and which REGIMES actually occur).
Capture that with one hook, driven by the SAME bench workload the Profiler used (so shapes match the
regime).

🔴 **Capture records NO tensors.** There is no `reference_io.pt`. A stored golden is redundant
(`baseline_overlay/` already IS a runnable reference, and must exist anyway as the timing denominator),
costs hundreds of MB–GB per task, and is a failure mode of its own: it is only valid while the operands
reproduce bit-for-bit, so a box or torch-build change becomes a hard failure. Correctness is LIVE parity
against the baseline leg (see Step 2).

`scripts/capture_shapes.py` installs a wrapper around the target callable (the Triton entry fn or the
python op that dispatches the kernel) via the overlay monkeypatch mechanism
([[sglang_internals]] §3b), runs a SHORT bounded window of the bench, and for the first N distinct
input-shape signatures records the SHAPE/DTYPE spec of `(args, kwargs)` into `meta.json`. Key rules:
- **Record the spec, never the values**: dims, dtype, device, stride/contiguity, and any non-tensor
  scalar args (seeds, scales). Nothing that scales with tensor size is written, so a capture dir is a
  few KB of JSON no matter how large the MoE operands are — and there is nothing for a later in-place
  op to corrupt.
- **Distinct-shape dedup**: one record per shape signature, up to `--max-cases` (default 5). For a
  kernel serving both prefill (large M) and decode (small M) regimes, this naturally captures BOTH →
  the unittest gets multi-case coverage and the kernel squad can build regime-specific variants.
- **Determinism**: capture with temp=0 so the shape mix is reproducible.
- Bound the window (`--num-steps`) so capture is fast.
- **CUDA-graph replay is not capturable** — the hook records eager calls only. A regime that only ever
  runs captured must be reconstructed from the launch flags (`regime.json`), not from the hook.

## Step 2 — Emit an immutable, general unittest
The unittest must be backend-agnostic: `cases.py:random_shapes` rebuilds FRESH in-regime operands at
the captured dims from a seeded generator, the BASELINE leg runs them under `baseline_overlay/` and
records its outputs (`h.baseline_random_outputs`), and the candidate is compared against those
outputs on every draw (tolerance per dtype: bf16/fp16 → `rtol=2e-2, atol=2e-2` typical; fp8 looser;
fp32 tight). Route everything through `h.run_correctness(...)`, which builds the eager cases via
`h.live_oracle_cases` and also runs the sequence + fail-closed CUDA-graph-replay legs.
Because it pins the SHAPES and reaches the op only through `meta.target_callable`, it transparently
judges a Triton / HIP / CK / aiter / asm reimplementation — the optimizer just has to make the entry
point fast AND match the live baseline.

Live parity also covers MANY value draws instead of the one a recorded golden could pin.

Anti-cheating (inherited from the single-kernel COMMANDMENT contract):
- The optimizer MUST NOT edit `unittest.py`, `cases.py`, `harness_lib.py` or `meta.json`. The Extractor
  records checksums in `meta.json`; the e2e Integrator/Validator re-checks them before trusting any
  speedup.
- Correctness is judged against the BASELINE LEG, never against a re-run of the candidate's own code
  path (which would let a no-op "pass"). `h.assert_legs_differ` refuses to measure if the two legs
  resolve to the same code, and `assert_independent_outputs` fails a candidate that returns a
  shared/persistent buffer.

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
