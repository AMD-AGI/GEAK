---
title: cast_fill_copy on Triton — SOTA card
kind: sota_card
operator: cast_fill_copy
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/sgl-project/sglang/pull/2601
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# cast_fill_copy × triton

## TL;DR
Triton is the SOTA way to author a **fused** cast/copy (e.g. cast-at-the-store of a norm/GEMM, or a
strided-gather-into-compute). For a *plain contiguous* copy/fill, prefer the runtime (`hipMemcpy`/
`hipMemset`) — a Triton kernel only earns its place when it **fuses** something. The fp8 cast must use the
**FNUZ** type on gfx942 (`tl.float8e4b8`), OCP on gfx950.

## SOTA implementation(s)
| impl | source | gens/dtypes | notes | when best |
|---|---|---|---|---|
| fused cast at store (norm/GEMM epilogue) | [../fusion.md](../fusion.md) | gfx942/950 | `OPTIMIZE_EPILOGUE=1`; fp8 in fp32 acc | quant path, the real win |
| strided-gather → contiguous (or → compute) | this card | gfx942/950 | LDS-tile if transpose-like | `.contiguous()` you can't elide |
| standalone vectorized cast/copy/fill | this card | gfx942/950 | matches HIP BW ceiling | when not fusable |

```python
@triton.jit
def cast_bf16_fp8(in_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK); m = offs < n
    x = tl.load(in_ptr + offs, mask=m).to(tl.float32) * scale
    y = x.to(tl.float8e4b8)                            # FNUZ on gfx942 (NOT float8e4nv = OCP)
    tl.store(out_ptr + offs, y, mask=m)
```

## Config space / knobs
- `BLOCK` a multiple of the **larger** vec width across in/out dtypes (so both sides 128-bit).
- `num_warps` 2/4, `num_stages=1`, `waves_per_eu=3/4` (hide HBM latency).
- `knobs.amd.use_buffer_ops=1` for masked tail.
- fp8 dtype: `tl.float8e4b8`/`tl.float8e5b16` (FNUZ, gfx942); OCP on gfx950.
- strided copy: if it's a transpose, tile through LDS (the GEMM swizzle applies); else just stride-index.

## Numerics / parity
copy/fill bit-exact; float→float RNE; **fp8: FNUZ on gfx942** (wrong dialect ≈ 2× off — #2601 normalize),
saturate to max-normal. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
The cast almost always rides a norm/GEMM kernel (don't ship it standalone). Standalone cast → torch custom
op (Inductor keeps opaque). For plain copy/fill, call `torch`/`hip` runtime, not Triton.

## Pitfalls & anti-patterns
- OCP `float8_e4m3fn` into a gfx942 store → wrong dialect / `Unsupported conversion`. Use fnuz.
- A standalone Triton copy/fill where `hipMemcpy`/`hipMemset` is already at peak — pointless kernel.
- `BLOCK` not a vec multiple of the wider dtype → narrow stores.
- Strided read in the fused kernel killing coalescing — sometimes elide-or-materialize is the real fix.

## How to verify
ISA shows `global_load/store_dwordx4`; GB/s vs ~4.3 TB/s; copy/fill bitwise vs torch; cast atol + fp8 task
gate; confirm the fp8 dialect for the arch.

## Alternatives / cross-links
[hip.md](hip.md) (runtime copy/fill, peak BW) · [pytorch_inductor.md](pytorch_inductor.md) (auto-elide/fuse)
· [../fusion.md](../fusion.md) · [`../../elementwise/backends/triton.md`](../../elementwise/backends/triton.md).

## Sources
- `OPTIMIZE_EPILOGUE`, ISA `dwordx4`: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- fnuz fp8 normalization (#2601): https://github.com/sgl-project/sglang/pull/2601
- supported_fp8_dtypes / knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
