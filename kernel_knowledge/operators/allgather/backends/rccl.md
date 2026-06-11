---
title: allgather on RCCL / MoRI-CCL — SOTA card
kind: sota_card
operator: allgather
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

# allgather × RCCL / MoRI-CCL

## TL;DR
> RCCL is the default all-gather; MoRI-CCL is the lightweight latency-focused alternative. All-gather is
> pure copy → the levers are algorithm-by-size, SDMA offload, and overlap with the adjacent GEMM. Same
> xGMI mesh and busbw class as [[allreduce]].

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| RCCL all-gather (ring/tree) | `ROCm/rccl` | gfx942/950 | ~316–330 GB/s busbw class @ 8× MI300X | default SP/TP gather |
| MoRI-CCL | `ROCm/mori` | gfx942/950 | latency-focused (vendor) | small/latency-sensitive gathers |
| sglang AITER AG | `SGLANG_USE_AITER_AG=1` | gfx942/950 | small-msg win | sglang AG path |

## Config space / knobs
- `HSA_ENABLE_SDMA=1` (SDMA offload), `NCCL_MIN_NCHANNELS=112` (sub-island, A/B), `RCCL_MSCCLPP_THRESHOLD`,
  `-G 1` graph capture, `SGLANG_USE_AITER_AG=1` (needs `SGLANG_USE_AITER=1`). Full table:
  [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md).

## Numerics / parity
Pure copy → parity-safe (no reduction). Only risk is shard layout. See [numerics.md](../numerics.md).

## Integration (rebind seam)
torch distributed / framework AG → RCCL automatically; tune via env. `SGLANG_USE_AITER_AG=1` for the AITER
AG kernel.

## Pitfalls & anti-patterns
- AG on CUs instead of SDMA steals GEMM cycles — `HSA_ENABLE_SDMA=1`.
- MIN_NCHANNELS disables the tuning model — A/B.
- Off-island AG falls onto the NIC — tune RDMA.

## How to verify
`rccl-tests all_gather_perf -b 8 -e 16G -f 2 -g 1 -G 1`; rocprof for SDMA usage + GEMM overlap; e2e SP tok/s.

## Alternatives / cross-links
[hip.md](hip.md) · [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md) ·
[`backends/mori_rccl/overview.md`](../../../backends/mori_rccl/overview.md) · [overview.md](../overview.md).

## Sources
- RCCL env / algos / SDMA: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- xGMI mesh: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
