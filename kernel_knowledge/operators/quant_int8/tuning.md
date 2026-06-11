---
title: quant_int8 — tuning
kind: operator_overview
operator: quant_int8
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8]
regimes: [both]
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/py_itfs_ck/smoothquant_kernels.cu
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# quant_int8 — tuning

> Like FP8, standalone INT8 quant is **bandwidth-bound** — one read of the bf16 activation, one write of
> the INT8 tensor + per-token scale (+ azp). The dominant lever is fusing the SmoothQuant scale and the
> cast into the producing RMSNorm/act ([[fusion.md]]).

## Performance model
`x[M,N]` bf16 → int8: traffic ≈ `2·M·N` read + `1·M·N` write + `M·(scale[,azp])`. Hit ~HBM peak. The
SmoothQuant per-channel scale `s_smooth[N]` is a small extra read broadcast over rows — negligible.

## Knobs by backend
### vLLM HIP (`int8_quant_kernels.cu`)
- one block per token row (`token_idx`), threads stride `hidden_size`; vectorize the cast.
- symmetric vs azp path (`azp` optional arg) — azp adds a min-reduce alongside the amax.
- `float_to_int8_rn` uses `nearbyint` (FE_TONEAREST) — don't change the rounding mode.

### aiter smoothquant (`csrc/py_itfs_ck/smoothquant_kernels.cu`, CK-backed)
- `smooth_per_token_scaled_quant_kernel` template: `block_size`, `thread_data_size` (16), optional
  `transpose_out_dim01`, smooth-scale map/hash (`has_smscale_map`) for MoE per-expert scales.
- MoE variants (`moe_smooth_per_token_scaled_quant_v1/v2`) fold the expert routing into the quant.

### Triton (`moe_op_gemm_int8_smoothquant.py`)
- the int8 smoothquant is **fused into the MoE GEMM** — `get_kernel_config(m,n,k,routing_data)` picks
  `BLOCK_M/N/K`; a gluon gfx942 variant exists. `can_overflow_int32` guards the INT32 accumulator.

## Decode vs prefill
- prefill: bandwidth-bound; maximize vector width + occupancy, grid ≥ 1024.
- decode: launch-bound → fuse into norm; the standalone INT8 quant launch dominates.

## SmoothQuant α tuning
α (migration strength, ≈0.5) is an **offline calibration** knob, not a kernel knob — it trades activation
vs weight quantization difficulty. Tune α per-model on a calibration set (AutoSmoothQuant in Quark does
this per-layer); the kernel just applies the resulting `s_smooth`.

## Sources
- vLLM int8 kernel structure (token row, azp, nearbyint): `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- aiter CK smoothquant kernel template (block/thread sizes, MoE map): `ROCm/aiter@a6bb49937:csrc/py_itfs_ck/smoothquant_kernels.cu`, `csrc/kernels/quant_kernels.cu`.
- ≥1024 WG, vectorization: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
