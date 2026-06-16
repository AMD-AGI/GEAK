---
title: lm_head_logits on hip — SOTA card
kind: sota_card
operator: lm_head_logits
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/skinny_gemms.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# lm_head_logits × hip

## TL;DR
The editable HIP path for the head is **vLLM's own skinny GEMMs** in `csrc/rocm/skinny_gemms.cu`
(`wvSplitK`, `wvSplitKrc`, `LLMM1`, `wvSplitKQ` for fp8) — the same kernels used for decode linears, applied
to the head's small-M shape. These are a strong **decode** path (M=batch) and the natural Tier-C rewrite
seam when you want to hand-tune the `(M, N=V, K=d)` GEMM or fuse the soft_cap/argmax into the kernel.
Hand-HIP is the right tool only when you need that control; the default head GEMM is AITER/hipBLASLt.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `wvSplitK` / `wvSplitKrc` (split-K skinny) | `vllm-project/vllm@HEAD:csrc/rocm/skinny_gemms.cu` | gfx942/950, bf16/fp16 | decode skinny GEMM; can beat a generic library kernel at small M | decode head, AITER off, `VLLM_ROCM_USE_SKINNY_GEMM=1` |
| `LLMM1` (small-M matmul) | same | gfx942/950, bf16/fp16 | tiny-M path | very small batch decode |
| `wvSplitKQ` (fp8 split-K skinny) | same | gfx942/950, fp8_e4m3_fnuz | quantized head | memory-constrained large-vocab head (accuracy-gate) |
| hand-HIP fused GEMM+soft_cap+argmax | author ([[hip_cpp]]) | gfx942/950 | one-pass head for greedy (skip `[M,V]` materialize) | greedy serving, full control |

## Config space / knobs
- Engage existing kernels: `VLLM_ROCM_USE_SKINNY_GEMM=1` (default). Hand-tuning the `.cu`: MFMA
  `matrix_instr_nonkdim=16`, `__launch_bounds__` (waves/EU), **split-K factor** to fill 304 CUs for the
  small-M head, `__restrict__` + `global_load_dwordx4` for the `[V,d]` weight read (bandwidth-bound).
- Block = multiple of 64 (wave64); `-munsafe-fp-atomics` for the split-K atomic accumulate.
- For a fused-argmax kernel: a block-level argmax reduction (`atomicMin` on index for lowest-index
  tie-break) over the `V` dimension — avoids materializing/all-gathering full logits for greedy.

## Numerics / parity
fp32 accumulate, **fp32 logits**. split-K atomic order is non-deterministic → tiny FP variance; for greedy
ensure the argmax tie-break is lowest-index and **re-check temp=0 parity** after a split-K change. fp8 head
= fnuz on gfx942 (off-by-2× if read as OCP) → task-accuracy gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`csrc/rocm/skinny_gemms.cu` + `csrc/rocm/torch_bindings.cpp` (`rocm_ops.def("wvSplitK", ...)`) is the
registration surface; editing requires a **vLLM rebuild**. Selected via `VLLM_ROCM_USE_SKINNY_GEMM` when
AITER linear is off. Verify: rocprofv3 shows `wvSplitK*`/`LLMM1` for the `N=V` GEMM.

## Pitfalls & anti-patterns
- `warpSize==32` grid math (it's 64 on CDNA).
- No split-K at tiny M → grid underfills 304 CUs (`[V,d]` read serializes).
- Skinny kernels are decode-only; a prefill-shaped head won't benefit.
- Editing `.cu` needs a rebuild; not a runtime swap.
- fp8 head accuracy (fnuz dialect) — gate it.

## How to verify
rocprofv3 confirms `wvSplitK*`/`LLMM1` ran (not a Triton/AITER fallback); isolated skinny-GEMM bench at the
served decode batch; greedy/temp=0 parity after any split-K/kernel change.

## Alternatives / cross-links
[aiter.md](aiter.md) (live GEMM) · [vllm_kernels.md](vllm_kernels.md) (wiring + same kernels) ·
[triton.md](triton.md) · [../overview.md](../overview.md) · [[hip_cpp]] · [[skinny_gemv_decode]] ·
[[argmax_topk]].

## Sources
- vLLM ROCm skinny GEMM kernels (`wvSplitK`, `LLMM1`, `wvSplitKQ`): https://github.com/vllm-project/vllm/blob/main/csrc/rocm/skinny_gemms.cu
- HIP kernel language (wave64, `__launch_bounds__`, `__restrict__`): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Skinny/split-K decode GEMM, ≥1024 grid, `-munsafe-fp-atomics`: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
