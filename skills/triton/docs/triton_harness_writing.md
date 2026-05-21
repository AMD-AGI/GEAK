# Triton — Harness Writing Deep Dive

This document concentrates the Triton-specific harness pitfalls scattered
across the legacy `src/minisweagent/run/preprocess/INSTRUCTIONS.md`
(section 1b "Common pitfalls" 1, 2, 4, 6, 8) and the per-kernel-type
notes from `src/minisweagent/run/preprocess/unit_test_agent.py`
(`_LANGUAGE_GUIDANCE["triton"]`).

---

## Per-kernel-type Triton testing bullets

Lifted from `unit_test_agent.py::_LANGUAGE_GUIDANCE["triton"]`:

- This is a Triton kernel (JIT-compiled Python). No build step needed.
- Import the kernel via its Python package path (do NOT use `importlib.util`).
- Use `torch.testing.assert_close` for correctness validation.
- Use `triton.testing.do_bench` or `torch.cuda.Event` for benchmarking.
- Set `PYTHONPATH` before the process starts if the package is not installed.
- Use fixed random seed (`torch.manual_seed(42)`) and fixed tensor sizes.

---

## Pitfall 1 — Import via the package path, NOT `importlib.util`

Triton kernels often have deep import chains (e.g.
`from aiter.ops.triton._triton_kernels.rope.rope import ...`).
Using `spec.loader.exec_module` breaks these because the parent package
is not initialised. Instead, add the repo root to `sys.path` and use a
normal `import` or `from ... import`:

```python
import sys
sys.path.insert(0, '/path/to/repo/root')
from aiter.ops.triton.rope.rope import rope_fwd, RotateStyle
```

---

## Pitfall 2 — `PYTHONPATH` must be set BEFORE the process starts, not inline

`kernel-profile` passes the command to `rocprofv3` which uses `execvpe`.
Inline `PYTHONPATH=... python3 ...` will NOT work. Instead, either:

- Set `PYTHONPATH` in the COMMANDMENT `## SETUP` section, OR
- Create a small wrapper shell script:

```bash
#!/bin/bash
export PYTHONPATH=/path/to/repo:$PYTHONPATH
python3 /path/to/test_harness.py "$@"
```

---

## Pitfall 4 — Three-tier shape lists (ALL / HARNESS / PROFILE)

Extract shapes from discovered test files, NOT hardcoded defaults. The
harness must define three shape lists at the top of the script:

- **`ALL_SHAPES`** — every unique shape from the discovered test files,
  sorted by total element count.
- **`HARNESS_SHAPES`** (20-25) — uniformly sampled from `ALL_SHAPES`. If
  `ALL_SHAPES` has ≤25 entries, `HARNESS_SHAPES = ALL_SHAPES`.
- **`PROFILE_SHAPES`** (5) — evenly-spaced from `ALL_SHAPES`. Prevents OOM
  during profiling, where `rocprofv3` keeps every kernel launch alive
  for replay.

### Shape routing by CLI mode

| Flag | Shape set | Why |
|---|---|---|
| `--profile` | `PROFILE_SHAPES` (5) | Prevents OOM. |
| `--benchmark` | `HARNESS_SHAPES` (20-25) | Reduced wall-clock per iteration. |
| `--correctness` | `HARNESS_SHAPES` | Same sample mix as `--benchmark` so coverage matches. |
| `--full-benchmark` | `ALL_SHAPES` | Full per-shape report for final comparison. |

### `--iterations N` override

The harness must accept `--iterations N` (default 20) to override the
number of benchmark iterations for both `--benchmark` and
`--full-benchmark`. If the flag is not passed, the harness should read
`GEAK_BENCHMARK_ITERATIONS` from the environment as a fallback. The
pipeline sets `GEAK_BENCHMARK_EXTRA_ARGS` to `--iterations 50` during
evaluation to reduce GPU timing noise.

### Default fallback shapes (when discovery returns nothing)

If the kernel does NOT have discovered test files, fall back to these
standard sizes (large enough to saturate the GPU):

- **Attention / RoPE kernels:** `S=2048, B=4, H=32, D=128` (fp16)
- **GEMM kernels:** `M=1024, N=1024, K=1024` (fp16)
- **Elementwise / pointwise:** at least 16M elements

---

## Pitfall 6 — `--profile` mode runs the kernel ONCE

The `--profile` mode should run the kernel once (with minimal setup)
so that `kernel-profile` / `rocprofv3` captures exactly the kernel(s)
you care about. Avoid running benchmarks or loops in profile mode.

**CRITICAL:** `--profile` must use ONLY `PROFILE_SHAPES` (5 shapes) to
prevent OOM.

---

## Pitfall 8 — Generate tensors on CPU, then move to GPU

In `--profile` mode, `rocprofv3` captures ALL GPU kernels — including
random number generation from `torch.randn(..., device='cuda')`. This
pollutes the profiler trace with unrelated kernels. Instead:

```python
# WRONG — launches GPU RNG kernel that shows up in profiler
x = torch.randn(S, B, H, D, dtype=torch.float16, device='cuda')
# CORRECT — RNG on CPU, only the target kernel appears in profiler
x = torch.randn(S, B, H, D, dtype=torch.float16, device='cpu').to('cuda')
```

`harness-verifier` enforces this as a fatal Phase-1 static check on any
function whose name contains `profile` — it matches the regex
`torch\.(randn?|empty|zeros|ones|full)\([^)]*device\s*=\s*["']cuda["']`
inside the body. Fix is always `device="cpu"` then `.to("cuda")`.

---

## Dtype preservation (per-tensor execution contract)

From `INSTRUCTIONS.md` pitfall 9 and `shape_fixer_agent.py` SYSTEM_PROMPT:

Preserve the source benchmark/test's full execution contract, not just
shapes:

- Keep per-tensor semantics independent: **dtype**, **device**, **layout**,
  **contiguity**, **index dtypes**, **auxiliary buffers / caches / scales**,
  and any helper-side preprocessing.
- Do NOT normalise every tensor to the main activation dtype just because
  the benchmark uses that dtype for `query`, `key`, or other primary
  activations.
- Tolerances: the legacy Triton builder hints suggest
  `torch.allclose(candidate, reference, atol=1e-4, rtol=1e-4)` unless
  the user test specifies tighter tolerances. Prefer
  `torch.testing.assert_close` over manual `torch.allclose` with
  always-pass fallbacks.

---

## Wrapper-vs-inner-kernel detection

(Section 1c of `INSTRUCTIONS.md`.)

**CRITICAL:** When the target kernel file is a **wrapper** that imports
the actual `@triton.jit` kernel from a different file, you MUST optimise
the **inner kernel file** instead of the wrapper.

### Signs of a wrapper

- The file imports `@triton.jit` functions from another module (e.g.
  `from aiter.ops.triton._triton_kernels.rope.rope import _rope_kernel_sbhd_fwd`)
- The file only sets launch parameters (`BLOCK_S`, `num_warps`, `grid`)
- The actual compute logic (`tl.load`, `tl.store`, arithmetic, memory
  access patterns) lives in the imported file

### Why this matters

Tuning launch parameters alone (`BLOCK_S`, `num_warps`, `waves_per_eu`)
yields limited improvement. The real optimisation opportunities (memory
coalescing, vectorisation, shared memory usage, algorithmic changes) are
in the `@triton.jit` kernel implementation.

When the kernel file is a wrapper, the harness still imports through the
wrapper's public function (so reruns stay stable across the wrapper and
the inner kernel mutations); the wrapper file path is what the
orchestrator passes around, while the inner file is where edits land.

---

## General harness rules (Triton-flavoured)

- Use `torch.testing.assert_close` for correctness, NOT manual
  `torch.allclose` with always-pass fallbacks.
- Use a fixed random seed (`torch.manual_seed(42)`) so that correctness
  checks compare deterministic outputs.
- Keep the harness file OUTSIDE the kernel directory or in a fixed
  location that won't be overwritten by OpenEvolve's candidate files.
- GPU event-based timing ONLY:
  `torch.cuda.Event(enable_timing=True)` or `triton.testing.do_bench`.
  NEVER `time.time()` or wall-clock.
- WARMUP = 50, ITERATIONS resolved from `--iterations` then
  `GEAK_BENCHMARK_ITERATIONS` (default 200). Report MEDIAN per shape.

---

## GEAK_SHAPES_USED (determinism contract)

In ALL modes, after the loop, print:

```
GEAK_SHAPES_USED=<list of config indices>
```

…where each index is the 0-based position in the full config list.
Example: `GEAK_SHAPES_USED=[0, 3, 7, 12, 24]`.

Use the config INDEX, not the config values. This avoids repr
differences (enums, torch dtypes, string formats). Two independent runs
must choose the EXACT same cases for `--benchmark`, `--correctness`, and
`--profile`. Equivalent config values are NOT sufficient if a different
ordered full case stream causes `_pick()` to choose a different subset.
Preserve the source file's ordered full case stream exactly.
