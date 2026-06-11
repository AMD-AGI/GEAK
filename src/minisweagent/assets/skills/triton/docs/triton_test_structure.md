# Triton — Test / Harness Structure Reference

> **GEAK contract precedence.** This file documents AKA-style tactical
> patterns observed in `AgentKernelArena/tasks/triton2triton/` (and
> `tasks/hip2hip/`, since several patterns generalise across both
> backends). HIP-specific patterns — pybind11 + `forward` entry,
> separate build dirs for ref / opt extension loading, `hipEventRecord`
> for standalone HIP binaries, the MMCV `autograd.Function` wrapper —
> live in [`skills/hip/docs/hip_test_structure.md`](../../hip/docs/hip_test_structure.md).
> As with that file, when AKA conventions conflict with GEAK's harness
> contract (4 CLI modes, `--correctness` / `--profile` / `--benchmark`
> / `--full-benchmark`, `GEAK_RESULT_GEOMEAN_SPEEDUP=<n>` stdout marker
> for speedup-verify, geometric-mean speedup, `WARMUP=50`, 3-tier
> shape lists `ALL_SHAPES` / `HARNESS_SHAPES` / `PROFILE_SHAPES`),
> **GEAK's contract WINS**. The patterns below are **TACTICAL** —
> they describe internal helper-function shape, timing protocol,
> golden-data conventions, and anti-patterns; they are NOT a
> replacement contract.

Read the language-agnostic quality requirements in the harness-generator's
`## Consistency & Robustness Requirements` block first. The rules below
are additive and describe HOW to satisfy that contract for Triton (and
any reference path expressed as a PyTorch module).

The rule numbering matches `hip_test_structure.md` so the same rule
points at the same pattern across both languages.

---

## Rule 7: paired `cuda.Event` timing protocol

**What.** Time kernel launches with paired
`torch.cuda.Event(enable_timing=True)` events. Run warmup before the
timed loop, call `torch.cuda.synchronize()` immediately before
`start.record()` and again after `end.record()`, then read the
GPU-measured interval via `start.elapsed_time(end)`.

**Why.** This is identical to the HIP protocol: `time.time()` measures
CPU-side launch queuing, not the kernel runtime; same-stream events are
serialised by the GPU and yield accurate ms intervals. Triton kernels
launch on the same `torch.cuda` stream as eager PyTorch ops, so the
`torch.cuda.Event` pair applies unchanged. `triton.testing.do_bench` is
a higher-level wrapper that uses the same underlying protocol — feel
free to use it when the harness only needs latency, not stage-by-stage
timing.

**How.**

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:144-160
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(n_warmup):
    kernel(*inputs, fn=triton_fn)

torch.cuda.synchronize()
start.record()
for _ in range(n_iter):
    kernel(*inputs, fn=triton_fn)
end.record()
torch.cuda.synchronize()

avg_time_ms = start.elapsed_time(end) / n_iter
```

This already aligns with GEAK's `WARMUP=50` + `ITERATIONS=200` contract;
the AKA citation is provenance for the protocol shape.

---

## Rule 12: PyTorch module + functional + `fn=` injection pattern

**What.** Express the reference implementation as a `torch.nn.Module`
whose `forward(self, *args, fn=module_fn)` defers to an injectable
`fn`. The default `fn` is the canonical PyTorch implementation;
passing `fn=triton_fn` swaps in the Triton kernel without changing any
other code.

**Why.** Lets the same harness drive both **PyTorch-eager vs Triton**
and **ref-Triton vs opt-Triton** comparisons by parameterising the
kernel call at the wrapper boundary. Less idiomatic for pure-Triton
benchmarks than for HIP-extension-loaded modules (Triton kernels are
typically called directly rather than through a Module), but valid
whenever the reference is expressed as a PyTorch module — common in
rocmbench-style suites where the baseline is an `nn.Module`.

**How.**

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:8-15
class SimpleMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return a.matmul(b + b)
```

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_functional/py_3267_SimpleMatmulModule_func.py:20-25
class SimpleMatmulModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, fn=module_fn):
        return fn(a, b)  # fn=module_fn for ref; fn=triton_fn for Triton
```

This is **tactical**: GEAK's harness contract still owns the 4 CLI
modes and the shape sweep. Use `fn=` injection inside the
correctness / benchmark loops, not as a substitute for the modes.

---

## Rule 13: `get_inputs()` / `get_init_inputs()` convention

**What.** Module-scope generators that yield test inputs as positional
argument lists. `get_inputs()` yields multiple shape / dtype
configurations; `get_init_inputs()` returns the `(args, kwargs)` pair
used to construct the Module.

**Why.** Centralising the test-input source at module scope makes the
harness a thin wrapper: iterate `get_inputs()`, call the Module with
each yielded list, no per-shape construction inside the timing loop.
The pattern shows up unchanged in AKA `triton2triton/rocmbench/` tests
because the harness contract there assumes the same yield-list
protocol.

**How.**

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:17-52
def get_inputs():
    configs = [
        ([4, 4], [4, 4]),
        ([16, 16], [16, 16]),
        ([4, 16, 16], [4, 16, 16]),
        # ...one tuple per (a_shape, b_shape) test case
    ]
    for a_shape, b_shape in configs:
        a = torch.rand(a_shape, dtype=torch.float32)
        b = torch.rand(b_shape, dtype=torch.float32)
        yield [a, b]

def get_init_inputs():
    return [[], {}]  # (args, kwargs) for Module(...)
```

GEAK harnesses still emit `GEAK_SHAPES_USED=[i, j, k, ...]` (config
indices, not config values) from each mode — `get_inputs()` ordering
**is** the index source.

---

## Rule 14: golden `.pt` save / load convention

**What.** Persist reference inputs and expected outputs to `.pt` files
using `torch.save({"tensor": ..., "requires_grad": ...}, path)`. Load
them back with `torch.load(path, map_location=device, weights_only=True)`.

**Why.** Universal across backends. Pinned golden data makes the
correctness check byte-stable across machines / drivers.
`weights_only=True` is the modern (post-2.4) default-safe `torch.load`
mode that avoids the arbitrary-code-execution path; `map_location`
lets the same `.pt` file replay on whichever GPU is visible. Saving
as a dict preserves autograd flags that get stripped by a plain
`torch.save(tensor)`.

**How.**

```python
# AKA: tasks/hip2hip/others/ball_query/test_ball_query.py:49-50, 80-87
# save side
torch.save(
    {"tensor": xyz.detach(), "requires_grad": xyz.requires_grad},
    os.path.join(save_dir, "xyz.pt"),
)
# load side (inputs)
xyz_data = torch.load(os.path.join(save_dir, "xyz.pt"), map_location=device)
xyz = xyz_data["tensor"].to(device).requires_grad_(xyz_data["requires_grad"])

# for expected outputs without autograd flags, weights_only=True is fine
expected = torch.load(path, map_location="cpu", weights_only=True)
```

---

## Rule 17: anti-patterns

These are **don'ts** observed in failed harnesses; each one breaks
correctness or timing for the same underlying reason.

- **Do not use `time.time()` (or `time.perf_counter()`) for kernel
  timing.** CPU-side timers measure launch queuing on async streams;
  the kernel may not have started — let alone finished — by the time
  `t1 - t0` is recorded. Pair `torch.cuda.Event` with explicit
  `synchronize` calls (Rule 7), or use `triton.testing.do_bench` which
  wraps the same protocol.

- **Do not allocate test tensors with `device='cuda'` inside the timed
  region.** `torch.randn(..., device='cuda')` launches a GPU RNG
  kernel that `rocprofv3` records and that contributes to the
  elapsed-time window. Generate on CPU, then `.to('cuda')` outside the
  timing bracket. This is also enforced by GEAK's harness-verifier
  Phase 1 static check on any function whose name contains `profile`.

- **Do not conflate "compilation success" with "module loadability".**
  For Triton this manifests as `@triton.jit`-decorated functions
  reporting `success` from a dry-run compile while a launch-time
  argument-type mismatch surfaces only on the first real invocation.
  Always exercise the kernel on a small canonical input immediately
  after import and surface a `HARNESS_ERROR:` line on failure — never
  `try / except: pass`. (The HIP-side analogue is the
  `cpp_extension.load(...)` symbol-missing trap; see
  `hip_test_structure.md` Rule 17.)
