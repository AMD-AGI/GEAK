---
title: reduce_scatter — tuning
kind: technique
operator: reduce_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/reduce_scatter.py
---

# reduce_scatter — tuning

## What you actually tune
Algorithm-by-size, overlap with the producing GEMM, and — the real win — the **SP rewrite** that fuses RS
with the norm/all-gather. Full RCCL knob table:
[`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).

## Levers
- **SP rewrite**: replace `all_reduce` with `reduce_scatter + local RMSNorm + all_gather`; the norm runs on
  the **sharded** tensor (1/P the work) and AsyncTP overlaps RS/AG with the GEMMs. The headline tuning
  decision for RS.
- **Channels (sub-island)**: `NCCL_MIN_NCHANNELS=112` (TP=2/4, A/B — bypasses tuning model).
- **MSCCL++**: `RCCL_MSCCLPP_THRESHOLD` for small-msg fast kernels.
- **Graph capture**: `-G 1`.
- **Overlap**: `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, `SGLANG_ROCM_USE_MULTI_STREAM=1`.
- **GPU-initiated (Iris)**: aiter Triton `reduce_scatter` (`_reduce_scatter_kernel`, `reduce_scatter`) for
  fusing RS into a Triton kernel; grid sized to CU count.

## xGMI facts
Same mesh as all-reduce: ~45–48 GB/s/link realized, slowest link caps it, use all 8 GPUs, ring for large
RS. As the 2-shot AR head, message size = the shard.

## Pitfalls
- RS on CUs vs SDMA — for a standalone RS prefer SDMA; the fused SP kernel uses CUs (it's reducing+norming).
- MIN_NCHANNELS disables the tuning model — A/B.
- Reduction order changes are benign but break byte parity — gate on greedy parity.

## How to verify
`rccl-tests reduce_scatter_perf -b 8 -e 16G -f 2 -g 1 -G 1`; rocprof for RS↔GEMM overlap; e2e SP tok/s;
greedy parity after the SP rewrite.

## Sources
- RCCL env / algos: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- aiter Triton reduce_scatter: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/reduce_scatter.py`.
- full knob table: [`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).
