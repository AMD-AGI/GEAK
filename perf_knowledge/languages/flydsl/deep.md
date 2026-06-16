---
title: FlyDSL — deep dive (FLIR, ROCDL surface, compile flow)
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - /sgl-workspace/aiter/aiter/ops/flydsl/kernels/splitk_hgemm.py
  - /opt/venv/lib/python3.10/site-packages/flydsl/expr/rocdl/
---

# FlyDSL — deep dive

## 1. FLIR — Flexible Layout IR
FlyDSL's core IR is **FLIR**, a **layout algebra inspired by CuTe**: tensors carry a **(Shape, Stride)**
descriptor, and the DSL composes layouts for tiling, swizzling, and vectorization. This is what lets a
FlyDSL kernel reason about "block → warp → thread → individual MFMA fragment" index math in Python and
still lower to tight ROCDL. The MLIR pipeline runs canonicalization + CSE and a **GPU-to-ROCDL**
lowering to produce AMDGCN. (Per AMD's Kimi-K2.5 write-up.)

## 2. The DSL surface (what you import)
From a real kernel (`kernels/splitk_hgemm.py`):
```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, const_expr, rocdl, vector
from flydsl.expr.typing import T                      # T.f32, T.i16, T.vec(n, T.f32), T.f32x4
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
```
- `arith`, `vector`, `math` — element/vector ops; `vector.bitcast(T.vec(4,T.i16), frag)` etc.
- `gpu` — `block_id`, `thread_id`, barriers (block/grid identifiers).
- `scf` / `range_constexpr` / `const_expr` — structured control flow and compile-time loops.
- `rocdl` — **the hardware intrinsics** (see §3).
- `SmemAllocator` — explicit LDS allocation (`allocator = SmemAllocator(None, arch=GPU_ARCH,
  global_sym_name="smem")`), returns `SmemPtr` you index with swizzled offsets.
- `T` (typing) — MLIR types incl. vector fragment types matching MFMA lane layouts.

## 3. The ROCDL intrinsic surface (the reason to use FlyDSL)
`flydsl.expr.rocdl` exposes the CDNA hardware ops 1:1. The families that matter for kernel perf:

**MFMA / SMFMAC (matrix core):**
`mfma_f32_16x16x16f16`, `mfma_f32_16x16x16bf16_1k`, `mfma_f32_16x16x32_{f16,bf16}`,
`mfma_f32_16x16x32_{fp8_fp8,fp8_bf8,bf8_bf8}`, `mfma_i32_16x16x32_i8`, the 32x32 variants, and
CDNA4 **block-scaled** `mfma_scale_f32_16x16x128_f8f6f4` / `mfma_scale_f32_32x32x64_f8f6f4` (MXFP8/6/4),
plus the sparse `smfmac_*`. Example wrapper from the source:
```python
class WmmaHalf_m16n16k16:
    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a = vector.bitcast(T.vec(4, T.i16), a_frag)
            b = vector.bitcast(T.vec(4, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, c_frag, 0, 0, 0])
        return rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0])
```
The `[a, b, c, cbsz, abid, blgp]` operand list mirrors the raw MFMA builtin (broadcast controls = 0
for standard GEMM). Fragment shapes (`A=4, B=4, C=4` values/lane for 16×16×16; `A=8,B=8,C=4` for
×32) are the wavefront-distributed MFMA layout.

**Memory (direct-to-LDS / async copy):**
`raw_ptr_buffer_load_lds`, `raw_ptr_buffer_load_async_lds`, `global_load_lds`, `global_load_async_lds`,
`global_load_async_to_lds_b{8,32,64,128}`, `cluster_load_async_to_lds_b*`, `load_to_lds`,
`lds_transpose_load`, `tensor_load_to_lds` / `tensor_store_from_lds`. These move global→LDS bypassing
VGPRs. In `splitk_hgemm.py` the load is issued as:
```python
lds_addr_ = rocdl.readfirstlane(lds_addr)             # scalarize the LDS base
rocdl.raw_ptr_buffer_load_lds(...)                    # global -> LDS, no VGPR staging
```

**Instruction scheduling (hand-built software pipeline):**
`sched_mfma(n)`, `sched_vmem(n)`, `sched_dsrd(n)` (DS read), `sched_dswr(n)` (DS write),
`sched_barrier(mask)`, `sched_group_barrier(...)`, `s_setprio`. The split-K HGEMM interleaves them to
keep the matrix core fed while loads are in flight:
```python
rocdl.sched_vmem(ldg_.consume(1))         # 1 global load
rocdl.sched_mfma(mfma_.consume(avg_mfma_count))  # N MFMAs
rocdl.sched_dswr(1)                        # 1 LDS write (stage next tile)
rocdl.sched_barrier(0)                     # hard scheduling fence
```
These are the FlyDSL equivalent of the HIP `__builtin_amdgcn_sched_group_barrier`/`iglp_opt` machinery
(see HIP [intrinsics.md](../hip_cpp/intrinsics.md)).

**Sync / cross-lane:**
`s_waitcnt`, `s_wait_asynccnt`, `s_barrier` (+ named-barrier variants `s_barrier_signal/_wait`),
`ds_bpermute`, `ds_swizzle`, `readfirstlane`, `wait_asyncmark`.

## 4. LDS allocation & swizzle
LDS is managed explicitly via `SmemAllocator`, and bank conflicts are avoided with the standard
XOR swizzle the source defines:
```python
def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)
```
Applied at both the LDS write (staging from global) and the LDS read (feeding MFMA). LDS budget is
arch-aware: `addressable_lds_bytes_for_gfx` → 65536 (gfx942) / 163840 (gfx950); aiter's
`_estimate_hgemm_lds_bytes` checks `(stages·tile_m·tile_k + stages·tile_n·tile_k)·2B` against
`get_shared_memory_per_block()` before compiling.

## 5. Split-K accumulation (global semaphore)
Split-K HGEMM uses a **global semaphore** + signal-state ring (not just atomics): aiter maintains
`SPLIT_K_GLOBAL_SEMAPHORE` (`int32[3·128]`) per stream and an `OnlineScheduler`
(release/consume signals) so the split-K partials are reduced deterministically. `SPLIT_K_COUNTER_MAX_LEN
= 128` caps `ceil(M/tile_m)·(N/tile_n)` output tiles for split-K>1.

## 6. Compile / JIT flow
`compile_flydsl_hgemm_kernel(...)` (aiter `kernels/hgemm_dispatch.py`) → `compile_hgemm_kernel(...)`
(`splitk_hgemm.py`, `@lru_cache(1024)`) builds the FLIR/MLIR module, runs the FlyDSL compiler
(`flydsl.compiler`), and returns an executable launched via `tensor_shim._run_compiled(kernel, out, a,
b, bias, m, semaphore, signal_state, stream)`. The fixed-at-compile facts in the current kernel:
`FIXED_STAGE = 2`, `c_to_lds = False`, `async_copy` forced from arch, MFMA warp atom = **16×16**
(`tile_m % (block_m_warps·16) == 0`, `tile_n % (block_n_warps·16) == 0`).

## 7. Numerics
HGEMM: fp16/bf16 in, **fp32 MFMA accumulate**, fp16/bf16 out — parity with library up to tiling
rounding. `bf16` path bitcasts fragments to `i16` for `mfma_*bf16_1k`. Scaling (`scale_a/b/c`) is
**not** supported by `flydsl_hgemm` (asserts in `tuned_gemm.flydsl_gemm`); fp8/int8 scaled GEMM goes
through `flydsl_preshuffle_gemm_a8` (separate path, `x_scale`/`w_scale`).

## Sources
- FLIR / (Shape,Stride) / GPU-to-ROCDL pipeline / instruction-level control: https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- DSL body (imports, mfma wrappers, swizzle_xor16, sched_*, raw_ptr_buffer_load_lds, split-K semaphore): ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/kernels/splitk_hgemm.py
- ROCDL op list (mfma/smfmac/scale_f8f6f4, async-to-LDS, sched_*, ds_bpermute): flydsl 0.1.5 @ /opt/venv/lib/python3.10/site-packages/flydsl/expr/rocdl/
- LDS budget / arch gating / lds estimate: ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/{utils.py,gemm_kernels.py}
