---
title: all_to_all_dispatch_combine on MoRI — SOTA card
kind: sota_card
operator: all_to_all_dispatch_combine
backend: mori
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-09
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://github.com/ROCm/mori
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# all_to_all_dispatch_combine × MoRI

## TL;DR
**MoRI-EP is the SOTA EP all-to-all on Instinct** — first-party, HIP-graph-capturable, GPU-initiated comm,
co-designed with AITER FusedMoE, and the stack AMD uses in its own wide-EP DeepSeek deployments. Choose it
for production MoE EP on MI300X/MI355X; DeepEP-on-ROCm / UCCL-EP ([[backends/mori_rccl/deepep.md]]) are the
portable alternatives. Full backend detail in [[backends/mori_rccl/mori_ep.md]]. Its **quantized all-to-all**
(FP4 dispatch + FP8 combine) gives a **2.56× round-trip bandwidth reduction** (28672 → 11200 B/token); on
MI355X, MoRI+SGLang beats B200 SGLang by **1.25× tok/s/GPU** at iso-latency.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| MoRI-EP dispatch/combine | `ROCm/mori@v1.2.0` `docs/MORI-EP-GUIDE.md` | gfx942/950, FP8 dispatch/BF16 combine | MI300X+CX7 **307/330 GB/s** dispatch/combine; MI355X+AINIC **345/420 GB/s**, **31/36 µs** @128 tok; up to **82%** latency cut; **64-GPU** scaling (mori v1.2.0, 2026-06, AMD-reported) | production MoE EP |
| MoRI quantized all-to-all (FP4 dispatch + FP8 combine) | LMSYS MoRI blog | gfx950 | **2.56× round-trip BW reduction** (28672 → 11200 B/token); MoRI-EP combine (EP8, BF16, 4096 tok, hidden 7168) fp8_blockwise **~736 µs** vs BF16 **~907 µs**; adaptive InterNodeV1LL **1.52× dispatch / 1.82× combine** ≤256 tok/rank; MI355X+MoRI SGLang **1.25× tok/s/GPU vs B200** at iso-latency (LMSYS, 2026-05-28) | low-bit EP / MI355X serving |
| MoRI-EP in wide-EP DeepSeek | wide-EP blog | gfx942, EP16 | **32.3k in / 12.4k out tok/s per node** (32× MI300X, 2P2D, +AITER), ROCm 6.3.1, 2025-11 (AMD-reported) | scaled MoE serving |

## Config space / knobs
- `EpDispatchCombineKernelType`: `InterNodeV1`(throughput) / `InterNodeV1LL`(low-latency) / `AsyncLL`;
  auto-switch via `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`.
- `EpDispatchCombineConfig`: `block_num`(80), `warp_num_per_block`(8), `rdma_block_num`(0), `num_qp_per_pe`(1),
  `hidden_dim`(7168), `num_experts_per_token`(8), `max_num_inp_token_per_rank`(4096),
  `use_external_inp_buf`(True, zero-copy).
- Arch pin `MORI_GPU_ARCHS=gfx942|gfx950`; precompile `MORI_PRECOMPILE=1`.

## Numerics / parity
combine reduction order + FP8 dispatch → accuracy-gate (greedy/temp=0), not byte parity. See
[[operators/all_to_all_dispatch_combine/numerics.md]].

## Integration (rebind seam)
`mori.shmem.shmem_torch_process_group_init("default")`; AITER seam `aiter/moe_op/mori_all2all.py`
(`MoriAll2AllManager`); vLLM/SGLang `--all2all-backend mori_low_latency`. DeepEP-compatible 3D API needs
**`ENABLE_STANDARD_MOE_ADAPT=ON`** at build.

## Pitfalls & anti-patterns
- ⚠ JIT first-iteration cost (`~/.mori/jit/`) → `MORI_PRECOMPILE=1` + warm before timing.
- ⚠ Missing `ENABLE_STANDARD_MOE_ADAPT=ON` → no 3D DeepEP-compatible API (`RuntimeError`).
- ⚠ Dynamic EP input sizes vs HIP-graph static-shape → pad/static-ize to capture.
- ⚠ `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` **incompatible** with MoRI.
- ⚠ Expert load imbalance up to **2×** → EPLB-style frequency grouping.
- ⚠ `grid_size` should match the **304 CU count** (authored-kernel finding); MoRI defaults differ — tune.

## How to verify
`rccl-tests` fabric baseline (~316–330 GB/s) → rocprofv3 dispatch/combine present, overlapping grouped-GEMM,
no `hipMalloc` steady-state; banner `mori_low_latency`; numeric parity vs torch MoE.

## Alternatives / cross-links
[backends/aiter.md](aiter.md) (FusedMoE seam) · [backends/hip.md](hip.md) (authored a2a) ·
[[backends/mori_rccl/mori_ep.md]] · [[backends/mori_rccl/deepep.md]] · [[backends/mori_rccl/rccl_tuning.md]].

## Sources
- MoRI-EP guide / repo (API, kernel modes, bandwidth table, 82%, 64-GPU): https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md · https://github.com/ROCm/mori @ v1.2.0
- MoRI quantized A2A 2.56× BW (28672→11200 B/tok), fp8_blockwise combine ~736 vs BF16 ~907 µs, InterNodeV1LL 1.52×/1.82× ≤256 tok/rank, 1.25× tok/s/GPU vs B200: https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
- Wide-EP 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html (ROCm 6.3.1, 2025-11)
- grid_size=304 / authored a2a: https://gau-nernst.github.io/amd-a2a/
