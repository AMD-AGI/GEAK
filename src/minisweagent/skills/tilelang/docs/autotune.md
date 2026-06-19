---
title: TileLang autotuning on MI300X
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16, fp8]
regimes: [both]
status: competitive
updated: 2026-06-19
sources:
  - https://github.com/tile-ai/tilelang
  - upstream tilelang 0.1.11 (examples/amd/example_amd_flash_attn_fwd.py, examples/gemm/example_gemm_autotune.py, tilelang/autotuner/)
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
---

# TileLang autotuning

## TL;DR
TileLang pairs `@tilelang.autotune` with `@tilelang.jit`: you define a candidate config space
(typically `itertools.product` of tile/thread/stage params), and the tuner JIT-compiles and times each,
caching inputs. There are two upstream styles: the **decorator** style (`@tilelang.autotune(configs=...)`
on the jitted kernel, as in the AMD FA example) and the **`AutoTuner.from_kernel(...)`** builder style
(as in the GEMM example). Autotuning is the main reason TileLang beats Triton on attention without hand
tuning. See [primitives.md](primitives.md) for the knobs being swept.

## Style A — decorator (upstream AMD FlashAttention)
```python
@tilelang.autotune(configs=get_configs(), cache_input_tensors=True, supply_prog=supply_tensors_gpu)
@tilelang.jit(out_idx=[3])
def fast_flashattn(batch, heads, seq_len, dim, is_causal, groups,
                   block_M, block_N, num_split_q, threads, num_stages,
                   enable_rasterization, k_pack, panel_size,
                   qk_coalesced_width, v_coalesced_width):
    @T.prim_func
    def main(...): ...
    return main

kernel = fast_flashattn(batch, heads, seq_len, dim, is_causal, groups=groups)
print(kernel.config)            # winning config
```
- `configs=get_configs()` — list of candidate dicts (the search space).
- `cache_input_tensors=True` — reuse GPU input tensors across trials (cheap re-timing).
- `supply_prog=...` — a function that allocates the input tensors on GPU for timing.
- `@tilelang.jit(out_idx=[3])` — JIT each variant; `out_idx` marks which arg(s) are outputs.

## Style B — builder (upstream GEMM)
```python
autotuner = (AutoTuner.from_kernel(kernel=kernel_fn, configs=get_configs(M, N, K))
             .set_compile_args(out_idx=[-1], target="auto")
             .set_profile_args(supply_type=tl.TensorSupplyType.Auto,
                               ref_prog=ref_program, skip_check=False, backend="event"))
result = autotuner.run(warmup=3, rep=20)
kernel = result.kernel; print(result.config)
```

## The real swept space (from upstream examples)
**FlashAttention-fwd (`examples/amd/example_amd_flash_attn_fwd.py`, CDNA3 branch):**
| param | upstream CDNA3 values | role |
|---|---|---|
| `block_M`, `block_N` | `[64, 128, 256]` | tile sizes |
| `num_split_q` | `[64, 128, 256]` | persist/split Q across blocks (occupancy on MI300X's 304 CUs) |
| `threads` | `[128, 256]` | block size |
| `num_stages` | `[0, 1]` | `T.Pipelined` prefetch depth (LDS-bounded at 64 KB) |
| `k_pack` | `[2]` | K per MFMA operand (RDNA/WMMA needs `[1]`) |
| `enable_rasterization` | `[True]` | threadblock swizzle on |
| `panel_size` | `[7, 8]` | `T.use_swizzle` panel width |
| `qk_coalesced_width` | `[8]` | vectorized QK global-load width |
| `v_coalesced_width` | `[4]` | vectorized V global-load width |

RDNA branch clamps `block_M/N ∈ [16,32]`, `threads ∈ [32,64]`, `num_stages=[0]`, `k_pack=[1]`.

**GEMM (`examples/gemm/example_gemm_autotune.py`):**
`block_M,block_N ∈ [64,128,256]`, `block_K ∈ [32,64]`, `num_stages ∈ [0,1,2,3]`,
`thread_num ∈ [128,256]`, `enable_rasteration ∈ [True,False]` (note upstream's `enable_rasteration`
spelling in the GEMM example's config dict). A roller (`MatmulTemplate.recommend_hints`) can replace the
fixed grid with device-aware TensorCore tilings via `with_roller=True`.

> Note: the ROCm blog quotes an "optimal `block_M=128, block_N=32, threads=512`" for an earlier kernel.
> The current upstream AMD example does **not** sweep those values — trust the upstream config space
> above and re-tune for your shape; treat the blog number as a historical data point, not a target.

## Building a config list
```python
import itertools
configs = [dict(block_M=m, block_N=n, num_stages=s, threads=t)
           for m, n, s, t in itertools.product(block_M, block_N, num_stages, threads)]
```
Keep candidate lists small and physically valid (respect 64 KB LDS and VGPR budgets) so the product stays
in the ~100s. Drop configs that overrun LDS before they reach the compiler.

## Pitfalls
- An uncapped `itertools.product` explodes; bound each list to plausible values for the shape.
- A config that overruns LDS/VGPR fails to compile or spills — the tuner should skip it, a manual pin can
  break.
- The autotuned config is **shape-specific** (b/h/s/d, M/N/K) and **build-specific** (ROCm/TileLang
  version) — re-tune per serving shape and after upgrades; never ship a frozen config as portable.
- Tiny default search spaces miss the optimum on unusual shapes — widen deliberately.
- The GEMM example uses `enable_rasteration` (misspelled) as a dict key for back-compat; match whatever
  key the kernel signature expects.

## Verify
- Print the winning config (`kernel.config` / `result.config`) and confirm it is LDS/VGPR-valid.
- `profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)` then `profiler.do_bench()` vs the
  reference at the exact shape (≥3 warm repeats, median).

## Sources
- upstream `examples/amd/example_amd_flash_attn_fwd.py` — decorator autotune, exact FA config space (CDNA3 + RDNA branches), `kernel.config`.
- upstream `examples/gemm/example_gemm_autotune.py` — `AutoTuner.from_kernel` builder, GEMM config space, roller hints.
- tile-ai/tilelang (`tilelang/autotuner/`): https://github.com/tile-ai/tilelang
- TileLang FlashAttention on MI300X (ROCm Blog — historical 128/32/512 figure): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
