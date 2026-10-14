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
