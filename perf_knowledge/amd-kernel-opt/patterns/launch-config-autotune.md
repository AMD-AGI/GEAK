---
type: Optimization Pattern
title: Launch-config autotune (warps / nonkdim / kpack / stages / BLOCK_M)
description: Sweep the cheap host-visible launch parameters; they pin MFMA atom, occupancy, and pipelining at once.
tags: [domain-gemm, domain-moe, domain-attention, bottleneck-compute, bottleneck-occupancy, lever-config, gfx942]
bottleneck: compute / occupancy
lever_class: config-only
median_speedup: 1.05x-1.63x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
Compute- or occupancy-bound kernel where the body is fine but the launch config is
suboptimal. The cheapest first move on a new kernel — one config sweep counts as a single
attempt.

# Mechanism
Sweep, one variable at a time:
- `num_warps` (e.g. →1 for tiny-M skinny paths, →4 to raise occupancy)
- `nonkdim` / `kpack` (MFMA atom width & K-packing; `kpack=64/elem_bytes` for FP8/INT8, 4 for bf16/fp16)
- `num_stages` (1→2 enables LDS pipelining)
- `BLOCK_M` (256→128 launches more workgroups to cover CUs)

# Evidence
- [fused_moe_gptq_awq](/cases/fused-moe-gptq-awq.md) — `num_warps=1, nonkdim=16, kpack=2` → **1.63×** (campaign20)
- [_gemm_a8w8_blockscale_kernel](/cases/gemm-a8w8-blockscale.md) — `nonkdim=32, kpack=2` launch-proxy HIP opts → **1.55×**
- [MLA prefill](/cases/mla-prefill.md) — `BLOCK_M 256→128, num_stages 1→2, num_warps 8→4` → **1.21×**
- [fused_moe_gptq_awq (KernelForge)](/cases/fused-moe-gptq-awq.md) — `num_stages=2` gave most of the ~1.05–1.10×

# Caveats
- Tune on representative shapes; a config fit to one shape regresses others.
- Crossing 128 arch-VGPR drops a wave; crossing 256 spills — watch register pressure when
  widening tiles/warps.

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md (pos2, pos9)
2. spare_kernels/k04_fmha_prefill/reference_solution/OPT_NOTES.md
3. KernelForge/results/fused_moe_kernel_gptq_awq/tasks/cli/*/workspace/optimization_report.md
