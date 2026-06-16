---
title: FlyDSL — usage patterns (HGEMM, preshuffle, split-K, MoE)
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py
  - /sgl-workspace/aiter/aiter/tuned_gemm.py
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
---

# FlyDSL — usage patterns

All signatures verified against the on-box aiter source. Two ways to reach FlyDSL: the **direct API**
(`aiter.ops.flydsl.flydsl_hgemm`) for authoring/benchmarking, and the **tuned-GEMM dispatch**
(`aiter.tuned_gemm`, `flydsl` libtype) which production uses transparently.

## 1. Direct HGEMM (bf16/fp16)
```python
from aiter.ops.flydsl import flydsl_hgemm
# C[M,N] = A[M,K] @ B[N,K]^T  (B is row-major [N,K], the weight layout)
out = flydsl_hgemm(
    a, b, out=None, bias=None,
    tile_m=128, tile_n=128, tile_k=64,     # tile_k must be %32==0 and >=32
    split_k=1,
    block_m_warps=1, block_n_warps=4,      # warp grid; tile_m %(bmw*16)==0, tile_n %(bnw*16)==0
    b_preshuffle=True,                     # default True -> b must be pre-shuffled (see §3)
    b_to_lds=False,                        # mutually exclusive with b_preshuffle
    async_copy=False,                      # forced from arch (gfx950 True); validated, can't override
    stages=2,                              # FIXED at 2 in current kernel
)
```
Constraints enforced by `_validate_hgemm_tiling`: `N % tile_n == 0` and `N >= tile_n`;
`(K/split_k) % tile_k == 0` and `>= tile_k`; `tile_m*tile_k`, `tile_n*tile_k`, `tile_m*tile_n` each
divisible by `ldg_vec_size(8) · block_m_warps·block_n_warps·64`; estimated LDS ≤ arch limit. Inputs
forced contiguous.

## 2. Tuned-GEMM dispatch (production seam)
`aiter.tuned_gemm.gemm_a16w16(A, B, ...)` looks up `(cu_num, padded_M, N, K, bias, dtype, otype,
scaleAB, bpreshuffle)` in the committed config CSV. If the row's `libtype == "flydsl"`, it parses the
`kernelName` (which **encodes every knob**, see knobs.md) and calls `flydsl_gemm` →
`flydsl_hgemm(...)`. Other libtypes: `hipblaslt`, `asm`, `skinny`, `torch`, `triton`(stub).
```
# kernel name grammar (gemm_kernels._HGEMM_KERNEL_RE):
flydsl_gemm{stage}_a{dt}_w{dt}_{out}_t{M}x{N}x{K}_split_k{S}_block_m_warp{bmw}_block_n_warp{bnw}
  _async_copy{T/F}_b_to_lds{T/F}_b_preshuffle{T/F}_c_to_lds{T/F}[_small_m[_nr{}][_pn{}][_wpe{}][_ur{}]]_{gfx}
```
> The config CSV is **aiter-build-specific** — do not copy a tuned table across ROCm/aiter versions
> (sourcing rule #2). `is_flydsl_available()` falls back to CK/CKTile if FlyDSL isn't installed.

## 3. b_preshuffle (the weight layout that matters)
`b_preshuffle=True` (the default) expects the weight already laid out for the matrix core:
```python
from aiter.ops.shuffle import shuffle_weight
b_sh = shuffle_weight(b, layout=(16*pack_n, 16))   # _get_flydsl_shuffle_layout(pack_n); pack_n=1
out = flydsl_hgemm(a, b_sh, b_preshuffle=True)
# or let the API shuffle once:
out = flydsl_hgemm(a, b, b_preshuffle=True, auto_shuffle_b=True)
```
Preshuffle removes the in-kernel transpose/relayout of B (the MFMA wants a specific fragment order).
`b_preshuffle=True` requires `b_to_lds=False`. For weights you reuse across many tokens (LLM serving),
shuffle once at load time.

## 4. Split-K for skinny/decode shapes
Small M·N, large K → use `split_k` so the K reduction spreads across more CUs (same idea as Triton
SPLIT_K). aiter's `_hgemm_split_k_options(k, tile_k)` only offers split_k that divide K cleanly and
leave 2–8 block-K loops. Split-K>1 uses the global semaphore reduction (deep.md §5) and caps output
tiles at `SPLIT_K_COUNTER_MAX_LEN=128`.
```python
out = flydsl_hgemm(a, b_sh, tile_m=16, tile_n=128, tile_k=64, split_k=8)   # decode-ish
```

## 5. Small-M family (decode GEMV-like)
For tiny M, the **small_m** kernel family (`kernels/small_m_hgemm.py`) fixes `tile_m=16`,
`block_m_warps=1`, `b_preshuffle=False`, **bf16 only**, and exposes extra knobs:
`n_tile_repeat`, `persistent_n_tiles`, `waves_per_eu`, `b_to_lds_unroll`. Selected automatically when
`m,n,k` are passed to `get_flydsl_splitk_hgemm_kernels` (it merges `iter_small_m_registry_configs`).

## 6. fp8 / int8 preshuffle GEMM (scaled)
```python
from aiter.ops.flydsl import flydsl_preshuffle_gemm_a8
flydsl_preshuffle_gemm_a8(XQ, WQ, x_scale, w_scale, Out,
    tile_m, tile_n, tile_k,
    lds_stage=2, use_cshuffle_epilog=0, use_async_copy=0, waves_per_eu=0)
# XQ: fp8 or int8; Out: bf16/fp16. N % tile_n == 0, K % tile_k == 0.
```
This is the **scaled** path (`flydsl_hgemm` itself asserts no scaling). `use_cshuffle_epilog` keeps the
result in MFMA layout through the epilogue (the FlyDSL analogue of Triton `OPTIMIZE_EPILOGUE`).

## 7. Fused MoE (the Kimi-K2.5 win)
`flydsl_moe_stage1` / `flydsl_moe_stage2` (`moe_kernels.py`, bodies in `kernels/moe_gemm_2stage.py` and
`mixed_moe_gemm_2stage.py`) implement the **2-stage grouped GEMM** for fused MoE — the kernel AMD
rewrote in FlyDSL for Kimi-K2.5 (the fused MoE was 87–90% of GPU time). Supports W4A16 / W4A16↔BF16
hybrid (`FLYDSL_W4A16_HYBRID=w2_bf16`) and fp4/fp8 output (kernel-name suffixes `_fp4`/`_fp8`/`_sbm{N}`).
Measured (vendor, AMD blog, MI300X, ROCm 7.2.0, aiter 0.1.5.post5.dev409, 2026-03-24): FlyDSL fused-MoE
**0.13 ms** (512 tok) / **0.60 ms** (2048) / **2.25 ms** (4096) / **8.68 ms** (16384) bf16; W4A16
**0.11 / 0.69 / 2.42 / 9.77 ms** — CK reported gpu_fault/unsupported on the large W4A16 case.

## 8. How to bench an authored FlyDSL kernel
1. Pick a tile config that passes `_validate_hgemm_tiling` for your `(M,N,K)`.
2. Preshuffle B once; warm up (kernels are `@lru_cache`-compiled on first call).
3. Median of ≥3 warm repeats vs `hipblaslt` default and vs the tuned aiter row.
4. e2e-gate via `tuned_gemm` (set the config CSV row's libtype to `flydsl`), not just isolated TFLOPS.

## Sources
- flydsl_hgemm / preshuffle / kernel-name grammar / split-K options: ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/gemm_kernels.py
- tuned_gemm flydsl libtype dispatch & solMap: ROCm/aiter@/sgl-workspace/aiter:aiter/tuned_gemm.py
- small-M / MoE kernel families: ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/{kernels/small_m_hgemm.py,moe_kernels.py}
- Kimi-K2.5 fused-MoE perf (vendor): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
