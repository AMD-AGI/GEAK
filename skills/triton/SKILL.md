---
name: triton
description: >
  Use when generating a fixed test harness for a Triton (@triton.jit) GPU
  kernel under the v3 GEAK preprocess pipeline. Covers harness CLI contract,
  Triton-specific entry-point detection, three-tier shape lists,
  --iterations argparse rule, and the GPU-RNG-pollution pitfall that
  rocprofv3 punishes.
---

# Triton — Harness-Generation KB

This knowledge base is loaded into the v3 `harness-generator` subagent
when the active `KernelLanguage` resolves to Triton. Read it carefully
before writing the harness — most pitfalls below have already burned
real evaluation runs.

## Task → doc routing

| What you are about to do | Read first |
|---|---|
| Build the harness file (CLI flags, shape lists, sampling) | `docs/triton_harness_writing.md` |
| Detect the kernel entry point / launcher | `docs/triton_idioms.md` |
| Decide test tolerances or what dtype to keep per tensor | `docs/triton_harness_writing.md` (Dtype preservation section) |

---

## Common tips (language-agnostic — duplicated per language by design)

The three rules below are the same on every language; each language KB
keeps its own copy so the KB stays self-contained when injected into a
child subagent prompt.

### 1. USER TASK CONTEXT block is HIGHEST PRIORITY

When the task body contains a `USER TASK CONTEXT` block delimited by
lines of `=` characters, that block is the AUTHORITATIVE shape / dtype /
template-args contract. It overrides any discovered benchmark file's
default sweeps.

```
HIGHEST PRIORITY (overrides any later "Shape source priority" rule)
- If the task message contains a `USER TASK CONTEXT` block delimited by
  `================================================================`, treat
  THAT block as the AUTHORITATIVE shape / dtype / template-args contract.
- When the user task lists explicit `(m, n)` shape tuples, `input_shapes`
  / `output_shapes` arrays, dtypes, or template specializations
  (e.g. `<Add=true, Quant=true>`), use THOSE values for `ALL_SHAPES` in
  the harness. Build the per-cell `(m, n, dtype, ...)` stream EXACTLY
  as the user task specifies.
- Do NOT mirror op-test default sweeps (e.g. `m ∈ {1, 32, 64, ..., 163840}`
  `n ∈ {1024, 4096, 6400, 8192}`) from a discovered benchmark file when
  the user task supplies an explicit production-shape contract, EVEN IF
  discovery flagged a benchmark file as "for this kernel".
- The discovered benchmark file (if any) is then a REFERENCE for kernel
  invocation pattern only (helper functions to build inputs, import
  paths, build steps, correctness reference impl). Borrow those, but do
  NOT borrow its `m`/`n`/`dtype` value lists.
- Write `user_task:production` to `harness_shapes_source.txt` instead of
  the benchmark file path. This signals to the post-UTA shape fixer
  agent that the user task is the canonical source.
- The order of precedence is:
    1. USER TASK CONTEXT block (this rule)
    2. Discovered benchmark file shapes (the legacy rule below)
    3. Discovered test file shapes
    4. Shapes inferred from kernel source
```

### 2. `--iterations` argparse rule (STRONGLY RECOMMENDED)

The harness MUST accept `--iterations N` so the orchestrator can pass
`--iterations 50` (or whatever the workload needs) during evaluation
without crashing argparse. Preferred form, both channels (CLI + env):

```python
parser.add_argument("--iterations", type=int, default=None)
args = parser.parse_args()
ITERATIONS = args.iterations if args.iterations is not None \
    else int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
```

If you omit the flag, GEAK detects this at validation time, emits a
WARNING, and silently skips passing the flag on the harness CLI so
argparse does not crash. The orchestrator then controls iteration counts
via the `GEAK_BENCHMARK_ITERATIONS` env var, so your harness MUST read
that env var as a fallback (or live with its hardcoded default).

### 3. Generate tensors on CPU, then move to GPU

In `--profile` mode, `rocprofv3` captures ALL GPU kernels — including
random number generation from `torch.randn(..., device='cuda')`. This
pollutes the profiler trace with unrelated kernels. Instead:

```python
# WRONG — launches GPU RNG kernel that shows up in profiler
x = torch.randn(S, B, H, D, dtype=torch.float16, device='cuda')
# CORRECT — RNG on CPU, only the target kernel appears in profiler
x = torch.randn(S, B, H, D, dtype=torch.float16, device='cpu').to('cuda')
```

This rule is enforced by `harness-verifier` as a fatal Phase-1 static
check on any function whose name contains `profile`. Do not work around
it.

---

## Reference documentation

- `docs/triton_harness_writing.md` — pitfalls 1, 2, 4, 6, 8 from the
  legacy `INSTRUCTIONS.md`, the three-tier shape lists with OOM
  rationale, wrapper-vs-inner-kernel detection, default fallback shapes,
  per-kernel-type Triton testing bullets.
- `docs/triton_idioms.md` — entry-point detection (`@triton.jit`
  decorator), kernel-shape skeleton, launcher-shape skeleton, common
  optimisations.
