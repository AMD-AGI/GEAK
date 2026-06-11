---
title: reduce_scatter on RCCL / MoRI-CCL — SOTA card
kind: sota_card
operator: reduce_scatter
backend: rccl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
---

# reduce_scatter × RCCL / MoRI-CCL

## TL;DR
> RCCL is the default reduce-scatter; MoRI-CCL is the latency-focused alternative. It is a reduction (fp32
> accumulate). The e2e value is the SP rewrite (RS + sharded norm + AG) + AsyncTP overlap. Same xGMI mesh
> and busbw class as [[allreduce]].

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| RCCL reduce-scatter (ring/tree) | `ROCm/rccl` | gfx942/950 | ~316–330 GB/s busbw class @ 8× MI300X | default SP/2-shot AR head |
| MoRI-CCL | `ROCm/mori` | gfx942/950 | latency-focused (vendor) | small/latency-sensitive RS |

## Config space / knobs
- `NCCL_MIN_NCHANNELS=112` (sub-island, A/B), `RCCL_MSCCLPP_THRESHOLD`, `HSA_ENABLE_SDMA=1`, `-G 1`,
  overlap (`TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`). SP rewrite is the headline. Full table:
  [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md).

## Numerics / parity
fp32 accumulate; reduction-order deltas benign; SP equivalence vs AR+norm (greedy parity). See
[numerics.md](../numerics.md).

## Integration (rebind seam)
torch distributed / framework RS → RCCL; SP rewrite is a compile-pass / layer change, not just env.

## Pitfalls & anti-patterns
- MIN_NCHANNELS disables the tuning model — A/B.
- RS on CUs vs SDMA for standalone — prefer SDMA.
- Off-island RS onto the NIC — tune RDMA.

## How to verify
`rccl-tests reduce_scatter_perf -b 8 -e 16G -f 2 -g 1 -G 1`; e2e SP tok/s; greedy parity.

## Alternatives / cross-links
[hip.md](hip.md) (fused SP kernel) · [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md) ·
[`backends/mori_rccl/overview.md`](../../../backends/mori_rccl/overview.md) · [overview.md](../overview.md).

## Sources
- RCCL env / algos: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- xGMI mesh: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
