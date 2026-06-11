---
title: fused_allreduce_rmsnorm on HIP — SOTA card
kind: sota_card
operator: fused_allreduce_rmsnorm
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/custom_all_reduce.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/rmsnorm_kernels.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# fused_allreduce_rmsnorm × HIP

## TL;DR
> HIP is where the fused comm+norm kernel is authored — fold the RMSNorm (+residual add, +fp8 quant) into
> the **custom all-reduce epilogue** (`custom_all_reduce.cu`) or the SP kernel's reduce path so the
> activation is read/written once. Reach for HIP to own the AR algorithm + norm fusion + quant in a single
> kernel for a specific layer/dtype.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| custom-AR epilogue + RMSNorm | `csrc/kernels/custom_all_reduce.cu` + `rmsnorm_kernels.cu` | gfx942/950; bf16/fp16/fp8 | inherits custom-AR small-msg win | decode TP AR+norm in one kernel |
| HIP fused add+rmsnorm+fp8 quant | `csrc/kernels/rmsnorm_quant_kernels.cu`, `gated_rmsnorm_quant_kernels.cu` | gfx942/950 | — | the norm+quant epilogue |
| qknorm+AR (asm/HIP) | aiter `qknorm_allreduce_fusion_kernel_2stage` | gfx942/950 | ~10–15% prefill TPS TP=2/4 (AMD-reported) | QK-norm+AR sibling template |

## Config space / knobs
- AR: 1-shot/2-shot xGMI P2P (IPC buffers); thread the norm into the epilogue after the reduce completes.
- RMSNorm: one block per row, hidden in LDS, fp32 reduce (wave64), `global_load_dwordx4`, `__restrict__`.
- fp8 quant: fold into the same pass (`-munsafe-fp-atomics` if atomic-accumulating); fnuz on gfx942.
- `__launch_bounds__` to cap VGPR; grid ≥1024 / CU-count.

## Numerics / parity
fp32 norm reduce + fp32 AR accumulate → parity-safe; fp8 quant gate; residual add before norm. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
aiter compiles the kernels JIT/AOT; the custom-AR + norm path engages via `SGLANG_USE_AITER_AR=1` /
`VLLM_ROCM_USE_AITER`. Edit the `.cu` + rebuild for a custom fused kernel.

## Pitfalls & anti-patterns
- Norm before the AR completes (missing barrier) → wrong values; sequence the reduce then norm.
- Without `-munsafe-fp-atomics` the quant/accumulate falls to slow CAS.
- aiter custom AR segfaults (#1542) — stability gate.

## How to verify
Disassemble: confirm one read/write of the activation, `v_*` norm reduce, no spills; rocprof for the fused
kernel; numeric vs separate AR+norm; e2e parity/eval.

## Alternatives / cross-links
[aiter.md](aiter.md) · [mori_rccl.md](rccl.md) · [`languages/hip_cpp/`](../../../languages/hip_cpp/overview.md) ·
[overview.md](../overview.md).

## Sources
- on-box: `ROCm/aiter@a6bb49937:csrc/kernels/{custom_all_reduce,rmsnorm_kernels,rmsnorm_quant_kernels,gated_rmsnorm_quant_kernels}.cu`.
- HIP norm/atomics: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- AR segfault: https://github.com/ROCm/aiter/issues/1542
