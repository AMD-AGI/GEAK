# HIP — Test / Harness Structure Reference

> **GEAK contract precedence.** This file documents AKA-style tactical
> patterns for HIP test/harness construction observed in
> `AgentKernelArena/tasks/hip2hip/`. When these conventions conflict
> with GEAK's existing harness contract (4 CLI modes, `--correctness`
> / `--profile` / `--benchmark` / `--full-benchmark`,
> `GEAK_RESULT_GEOMEAN_SPEEDUP=<n>` stdout marker for speedup-verify,
> geometric-mean speedup, `WARMUP=50`, 3-tier shape lists
> `ALL_SHAPES` / `HARNESS_SHAPES` / `PROFILE_SHAPES`), **GEAK's contract
> WINS**. The patterns below are **TACTICAL** — they describe the
> shape of internal helper functions, build-and-load mechanics,
> golden-data conventions, and anti-patterns. They are NOT a
> replacement contract.

Read the language-agnostic quality requirements in the harness-generator's
`## Consistency & Robustness Requirements` block first. The rules below
are additive: they describe HOW to satisfy that contract for HIP
specifically.

---

## Rule 5: pybind11 + `forward` entry pattern

**What.** When loading a HIP source as a PyTorch C++ extension via
`torch.utils.cpp_extension.load(...)`, expose the kernel entry point with
the canonical name `forward` from a `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`
block. The Python side then accesses `<ext>.forward` directly.

**Why.** `cpp_extension.load(...)` returns a module whose attributes are
the names bound by `PYBIND11_MODULE`. Standardising on `forward` lets
runners assume the same attribute regardless of which kernel they
load — no per-kernel attribute lookup, no string-keyed dispatch table.

**How.**

```cpp
// AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/hip/hip_3267_SimpleMatmulModule.hip:139-141
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "<kernel> HIP forward");
}
```

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/utils.py:58-66
from torch.utils.cpp_extension import load

hip_ext = load(name=kernel_name, sources=[f"{code_dir}/{hip_src}"], verbose=True)
hip_fn = hip_ext.forward
```

---

## Rule 6: separate build directories when loading two pybind extensions

**What.** When the harness loads a reference kernel and an optimized
kernel back-to-back via `cpp_extension.load(...)`, give each a
**distinct module name** (`<name>_ref` vs `<name>`) AND a **distinct
build directory**.

**Why.** pybind11's module registry is global within the process. If
two `load(...)` calls share a module name, the second call silently
returns the cached first module — the optimized kernel is never
loaded, the reference is timed twice, and the apparent speedup is
≈ 1.0. Distinct build dirs prevent object-file collisions even when
the source filename happens to match.

**How.**

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:199-214
ref_hip_dir = os.path.join(build_dir, "hip_ref")
opt_hip_dir = os.path.join(build_dir, "hip_opt")
os.makedirs(ref_hip_dir, exist_ok=True)
os.makedirs(opt_hip_dir, exist_ok=True)

ref_module_name = kernel_name + "_ref"
opt_module_name = kernel_name
ref_fn = load_hip_kernel(ref_module_name, ref_hip_dir, ref_hip_file_name)
opt_fn = load_hip_kernel(opt_module_name, opt_hip_dir, opt_hip_file_name)
```

---

## Rule 7: paired `cuda.Event` timing protocol

**What.** Time kernel launches with paired `torch.cuda.Event(enable_timing=True)`
events, with warmup before the timed loop and explicit `torch.cuda.synchronize()`
around `record` calls. (HIP's PyTorch backend reuses the `torch.cuda.*` API.)

**Why.** `time.time()` and any wall-clock approach measure CPU-side launch
queuing, not the kernel runtime. Events queued on the same stream are
serialised by the GPU and `start.elapsed_time(end)` returns the GPU-measured
interval in ms. Missing the leading `synchronize()` lets the warmup tail
leak into the first timed sample.

**How.**

```python
# AKA: tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:144-160
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(n_warmup):
    kernel(*inputs, fn=hip_fn)

torch.cuda.synchronize()
start.record()
for _ in range(n_iter):
    kernel(*inputs, fn=hip_fn)
end.record()
torch.cuda.synchronize()

avg_time_ms = start.elapsed_time(end) / n_iter
```

This already aligns with GEAK's `WARMUP=50` + `ITERATIONS=200` contract;
the AKA citation is provenance for the protocol shape itself.

---

## Rule 8: HIP standalone-binary timing with `hipEventRecord`

**What.** When the harness shells out to a `make`-built binary instead
of loading the kernel as a Python extension, time the kernel inside the
C++ binary using `hipEventCreate` / `hipEventRecord` /
`hipEventSynchronize` / `hipEventElapsedTime`, then expose the result on
stdout for the Python wrapper to parse (see Rule 15).

**Why.** Standalone-binary kernels (raw `hipcc` + `make` builds) have
no Python-side kernel handle, so the `torch.cuda.Event` protocol from
Rule 7 does not apply. The HIP C API exposes the same paired-event
semantics; using it inside the binary keeps the timing source of
truth co-located with the launch site instead of measuring
subprocess startup + kernel together.

**How.**

```cpp
// AKA: tasks/hip2hip/others/silu/silu.hip:71-79
hipEvent_t s, t;
HIP_CHECK(hipEventCreate(&s));
HIP_CHECK(hipEventCreate(&t));
for (int i = 0; i < warmup; ++i) launch();
HIP_CHECK(hipDeviceSynchronize());
HIP_CHECK(hipEventRecord(s));
for (int i = 0; i < iters; ++i) launch();
HIP_CHECK(hipEventRecord(t));
HIP_CHECK(hipEventSynchronize(t));
float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, s, t));
```

---

## Rule 12: PyTorch module + functional + `fn=` injection pattern

**What.** Express the reference implementation as a `torch.nn.Module`
whose `forward(self, a, b, fn=module_fn)` defers to an injectable `fn`.
The default `fn` is the canonical PyTorch implementation; passing
`fn=hip_fn` swaps in the HIP kernel without changing any other code.

**Why.** Lets the same harness drive both **PyTorch-eager vs HIP**
and **ref-HIP vs opt-HIP** comparisons by parameterising the kernel
call at the wrapper boundary. No code duplication for the swap. Pairs
naturally with Rule 7's `kernel(*inputs, fn=hip_fn)` shape and Rule 6's
ref/opt module separation.

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
        return fn(a, b)  # fn=module_fn for ref; fn=hip_fn for HIP
```

This is **tactical**: GEAK's harness contract still owns the 4 CLI
modes and the shape sweep. Use the `fn=` injection inside the
correctness / benchmark loops, not as a substitute for the modes.

---

## Rule 13: `get_inputs()` / `get_init_inputs()` convention

**What.** Module-scope generators that yield test inputs, returned as
positional argument lists. `get_inputs()` yields multiple shape /
dtype configurations; `get_init_inputs()` returns the `(args, kwargs)`
pair used to construct the Module.

**Why.** Centralising the test-input source at module scope makes the
harness a thin wrapper: iterate `get_inputs()`, call the Module with
each yielded list, no per-shape construction inside the timing loop.
Mirrors the shape-source-priority rule in the harness-generator
prompt (Section "Shape source priority").

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

**Why.** Pinned golden data makes the correctness check byte-stable
across machines / drivers. `weights_only=True` is the modern (post-2.4)
default-safe `torch.load` mode and avoids the
arbitrary-code-execution path; `map_location` lets the same `.pt`
file replay on whichever GPU is visible. Saving as a dict
(`{"tensor": ..., "requires_grad": ...}`) preserves autograd flags that
get stripped by a plain `torch.save(tensor)`.

**How.**

```python
# AKA: tasks/hip2hip/others/ball_query/test_ball_query.py:49-50, 80-87
# save side
torch.save(
    {"tensor": xyz.detach(), "requires_grad": xyz.requires_grad},
    os.path.join(save_dir, "xyz.pt"),
)
# load side
xyz_data = torch.load(os.path.join(save_dir, "xyz.pt"), map_location=device)
xyz = xyz_data["tensor"].to(device).requires_grad_(xyz_data["requires_grad"])

# for expected outputs, weights_only=True is fine (no autograd flag)
expected_idx = torch.load(path, map_location="cpu", weights_only=True)
```

---

## Rule 15: C++ / HIP standalone argv parsing + stdout markers

**What.** For `make`-built standalone HIP binaries, parse shape flags
from `argv` and emit two canonical stdout lines so a Python wrapper
can pick up the result with a regex:

- `Check: ... -> PASS` (or `FAIL`)
- `Perf: <X.XXX> us/launch`

**Why.** GEAK's harness still owns the 4-mode CLI contract on the
Python side. A wrapped HIP binary is then a **THIN subprocess** whose
only job is to print these two markers; the Python harness re-emits
them in GEAK's stdout format (`GEAK_RESULT_LATENCY_MS=<float>`).
Without a fixed marker the wrapper has to grep loose log text and
breaks the moment the kernel adds a debug print.

**How.**

```cpp
// AKA: tasks/hip2hip/others/silu/silu.hip:82-91, 117, 124
int main(int argc, char** argv) {
    int64_t B = 4096, H = 6400;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--B" && i + 1 < argc) B = std::atoll(argv[++i]);
        else if (std::string(argv[i]) == "--H" && i + 1 < argc) H = std::atoll(argv[++i]);
    }
    // ... launch + verify + time_kernel_ms() ...
    printf("Check: max_abs=%.4g  max_rel=%.4g  -> %s\n", max_abs, max_rel, ok ? "PASS" : "FAIL");
    printf("Perf: %.3f us/launch | ~BW: %.1f GB/s\n", us, gbs);
}
```

---

## Rule 16: MMCV `autograd.Function` wrapper

**What.** OpenMMLab-style HIP kernels expose their kernel through a
`torch.autograd.Function` subclass. `forward(ctx, ...)` asserts
`is_contiguous()` on each input, bound-checks the radii / sample
counts, allocates the output on `torch.cuda.*Tensor`, calls the
pybind extension's wrapper function, and `mark_non_differentiable`s
integer outputs. `backward(ctx, ...)` typically returns `None` for
each input (no gradient).

**Why.** A wide class of MMCV / mmpretrain / mmdetection kernels use
this wrapper as the user-facing entry. Recognising the shape lets the
harness import the wrapper directly (no need to call into the pybind
module by hand) and inherits the wrapper's input validation.

**How.**

```python
# AKA: tasks/hip2hip/others/ball_query/ball_query_wrapper.py:8-46
class BallQuery(Function):
    @staticmethod
    def forward(ctx, min_radius, max_radius, sample_num, xyz, center_xyz):
        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert min_radius < max_radius

        B, N, _ = xyz.size()
        npoint = center_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, sample_num).zero_()

        ball_query_ext.ball_query_wrapper(
            B, N, npoint, min_radius, max_radius, sample_num,
            center_xyz, xyz, idx,
        )
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

ball_query = BallQuery.apply
```

The harness then calls `ball_query(...)` directly — no pybind
plumbing in the harness file.

---

## Rule 17: anti-patterns

These are **don'ts** observed in failed harnesses; each one breaks
correctness or timing for the same underlying reason.

- **Do not use `time.time()` (or `time.perf_counter()`) for kernel timing.**
  CPU-side timers measure launch queuing on async streams; the kernel may
  not have started — let alone finished — by the time `t1 - t0` is recorded.
  Always pair `torch.cuda.Event` (Rule 7) or `hipEventRecord` (Rule 8) with
  an explicit synchronize.

- **Do not allocate test tensors with `device='cuda'` inside the timed
  region.** `torch.randn(..., device='cuda')` launches a GPU RNG kernel
  that `rocprofv3` records and that contributes to the elapsed-time
  window. Generate on CPU, then `.to('cuda')` outside the timing
  bracket. This is also enforced by GEAK's harness-verifier Phase 1
  static check on any function whose name contains `profile`.

- **Do not conflate "compilation success" with "module loadability".**
  `cpp_extension.load(...)` returning a non-`None` value only means the
  shared object built and dlopened. The kernel symbol the harness wants
  to call may still be missing, mis-named, or shadowed by a cached
  earlier build (Rule 6). Always exercise `<ext>.forward(...)` on a
  small canonical input immediately after load and surface a
  `HARNESS_ERROR:` line on failure — never `try/except: pass`.
