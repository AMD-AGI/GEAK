# Shape & I/O Capture — Turning a Hot Kernel into a Standalone Unittest

This is the **Kernel Extractor's** playbook. Its output is a self-contained kernel task dir that the
UNCHANGED single-kernel `kernel_workflow` consumes — same contract as a hand-written kernel task. The
critical property: the unittest must replay the kernel's REAL serving shapes and check correctness
against a recorded I/O oracle, and it must be **immutable during optimization** (anti-cheating).

## What the kernel layer expects (the task-dir contract)
The single-kernel workflow takes `args.kernel_path` = the task directory. For `PHASE=extract`,
follow `roles/kernel_extractor.md` for the complete contract and commands:
```
<task_dir>/
  kernel_src/          # one defining source file; the only optimizer-writable directory
  baseline_overlay/   # frozen current serving overlay, including accepted changes
  baseline_ref/*.orig # unimportable source copies for diffing
  cases.py           # shared call, timing_cases, random_shapes, eager_cases
  reference_io.pt     # recorded inputs + golden outputs; immutable during optimization
  harness_lib.py      # vendored shared correctness/timing helpers
  leg_runner.py       # vendored runner shared by both timing legs
  overlay_setup.py    # vendored candidate-overlay builder
  unittest.py        # builds, checks correctness and times speedup; immutable during optimization
  meta.json           # target_callable, candidate_bind, regime, shapes, checksums, workload
```
Copy the defining file, never a package subtree. Declare `candidate_bind` as a module replacement
or a rebind of the public callable. Both timing legs run the same `cases.py` under different
overlays; `cases.call` resolves `meta.target_callable`, never `kernel_src` directly.
The operation-comparison `extract_op` phase has its own artifact contract in the role file.

## Step 1 — Capture real shapes AND reference I/O from the live server
Profiles give shapes, but optimization needs the actual tensor VALUES (the oracle) so a backend swap
can be proven numerically correct. Capture both with one hook, driven by the SAME bench workload the
Profiler used (so shapes match the regime).

`scripts/capture_shapes.py` wraps a safe Python launcher or whole-operation dispatcher via
`scripts/overlay_setup.py add-capture` (see `sglang_internals.md`). Capture on a copy of the current
accepted overlay. Run a bounded window of the bench and record `(args, kwargs) -> output` to
`reference_io.pt` for distinct input-shape signatures. Key rules:
- **Detach + clone to CPU** (or keep on-device but snapshot) so later in-place ops can't corrupt the
  oracle. Record dtype, device, stride/contiguity, and any non-tensor scalar args (seeds, scales).
- **Coverage**: deduplication does not guarantee both regimes. Verify every served regime and required
  shape bucket; exclude unserved regimes. Missing value-dependent operands/layouts invalidate the oracle.
- **Determinism**: capture with temp=0 so re-running the reference is reproducible.
- Bound the window with `REPEATS`/`CAPTURE_MAX` and the supplied storage limits, never smaller workload
  dimensions. Run `kernel_selection.py` across all process/rank traces to verify the profiled GPU symbol,
  promote the selected oracle and reclaim process-local captures. Return its verdict verbatim.

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
- Use `h.run_correctness` for captured-oracle and random-value parity at fixed online shapes. Generate
  random references in the frozen baseline subprocess, never with the candidate. Supply the required
  graph replay bundle and check state across real call sequences; missing coverage is a harness defect.

## Step 3 — Build (optional) + speedup contract
Some kernels are pure Python/Triton (no build step) — `meta.json.build=false`. Others (HIP/CK/asm
candidates) need a compile — `meta.json.build=true` with a build command. Use `h.measure_legs` for
matched baseline/candidate graph/compile timing and source-resolution checks. Report each case's
`baseline_ms / optimized_ms` and geomean. When workload metadata is present, the primary metric comes
from `h.serving_weighted_speedup`: measured baseline latency times analytic serving calls, restricted
to served regimes. Profile weights are priors, not the final within-operation metric.

## Step 4 — Hand off to the kernel layer, then overlay back
- Extractor returns the `task_dir`; the orchestration calls `kernel_workflow.js` with
  `kernel_path=task_dir`. That recursive run does the real multi-backend optimization + verification
  and returns a `final_patch.diff` against the extracted source.
- The e2e Integrator deploys that patch using the task's `candidate_bind` on the accepted overlay,
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
