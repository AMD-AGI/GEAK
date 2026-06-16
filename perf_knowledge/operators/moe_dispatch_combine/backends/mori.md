---
title: moe_dispatch_combine on MoRI-EP — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: mori
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-06-09
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://github.com/ROCm/mori
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# moe_dispatch_combine × MoRI-EP

## TL;DR
> **MoRI-EP is the SOTA EP dispatch/combine on Instinct** — AMD's native, HIP-graph-capturable, GPU-initiated
> all-to-all, co-designed with AITER FusedMoE and used in AMD's own Wide-EP DeepSeek deployments. Choose it for
> production MoE EP on MI300X/MI355X. It exploits the **fully-connected xGMI mesh** (every GPU pair has a direct
> P2P link) intra-node and RDMA inter-node, with five kernel types for different topology/latency regimes.
> Its **quantized all-to-all** (FP4 dispatch + FP8 combine) gives a **2.56× round-trip bandwidth reduction**
> (28672 → 11200 B/token), and MI355X+MoRI SGLang beats B200 SGLang by **1.25× tok/s/GPU** at iso-latency.
> DeepEP-on-ROCm / UCCL-EP are the portable alternatives ([`backends/mori_rccl/deepep.md`](../../../backends/mori_rccl/deepep.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| MoRI-EP `dispatch`/`combine` | `ROCm/mori` (`python/mori/ops/dispatch_combine.py`) | gfx942/950, fp8 dispatch + bf16 combine | **307 GB/s dispatch / 330 GB/s combine** (MI300X+CX7); **345 / 420 GB/s** (MI355X+AINIC); LL @128 tok **31 µs / 36 µs** — mori v1.2.0, EP8, 4096 tok / hidden 7168 / top-8, 2026-06 (vendor, mori guide) | production intra- & inter-node EP |
| MoRI-EP in Wide-EP (32× MI300X) | Wide-EP blog | gfx942 | **32.3k in / 12.4k out tok/s per node**, ROCm 6.3.1, 2025-11 (AMD-reported) | large distributed DeepSeek |
| MoRI quantized all-to-all (FP4 dispatch + FP8 combine) | LMSYS MoRI blog | gfx950 | **2.56× round-trip BW reduction** (28672 → 11200 B/token); MoRI-EP combine (EP8, BF16, 4096 tok, hidden 7168) fp8_blockwise **~736 µs** vs BF16 ref **~907 µs**; adaptive InterNodeV1LL **1.52× dispatch / 1.82× combine** at ≤256 tok/rank (LMSYS, 2026-05-28) | low-bit EP at scale |
| MoRI on MI355X (SGLang) | LMSYS/AMD TCO blog | gfx950 | MI355X+MoRI SGLang **1.25× tok/s/GPU vs B200 SGLang** at iso-latency; cost-competitive distributed DeepSeek vs H-class (vendor) | MI355X EP serving |
| hand-rolled a2a (reference point) | gau-nernst blog | gfx942 | **292 µs** dispatch+combine, grid=304 (1 block/CU), GPU-MODE comp, 2025-11 (community) | shows the achievable floor / xGMI P2P model |

## Config space / knobs (on-box `EpDispatchCombineConfig`, mori @ b8af93f)
| field | default | note |
|---|---|---|
| `kernel_type` | `IntraNode` | `IntraNode` (xGMI), `InterNode` (baseline/debug), `InterNodeV1` (throughput), `InterNodeV1LL` (low latency), `AsyncLL` (latency; only kernel with split `dispatch_recv`) |
| `block_num` | **80** | main-kernel GPU blocks (≈ kMaxBlocks elsewhere) |
| `warp_num_per_block` | **8** | warps/block |
| `gpu_per_node` | **8** | affects all kernel types (xGMI fan-out) |
| `rdma_block_num` | 0 | inter-node RDMA blocks |
| `num_qp_per_pe` | 1 | RDMA queue pairs per PE |
| `max_num_inp_token_per_rank` | — | per-rank input cap (memory) |
| `num_experts_per_token` | — | top-k |
| `use_external_inp_buf` | True | zero-copy when False |
| `quant_type` | `"none"` | or `"fp8_direct_cast"` (`EpDispatchCombineQuantType.Fp8DirectCast`) |

- **Auto kernel select**: `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` switches kernel type by token count / topology.
- **Layouts**: native 2D `[T,H]`; **DeepEP-compatible 3D** via `dispatch_standard_moe`/`combine_standard_moe`
  (needs build flag `ENABLE_STANDARD_MOE_ADAPT=ON`).
- **Split phases**: `dispatch_send` / `dispatch_recv` (AsyncLL only), `combine_send/recv` for overlap.
- **Arch/JIT**: `MORI_GPU_ARCHS=gfx942/gfx950`, `MORI_PRECOMPILE=1` (avoid first-iter JIT cost in `~/.mori/jit/`).

### API shape (on-box)
```python
op.dispatch(input, weights, scales, indices, block_num=-1, rdma_block_num=-1,
            warp_per_block=-1, call_local_expert_count=False)   # -1 ⇒ tuned launch params
op.combine(input, weights, indices, ..., use_external_inp_buf=-1, call_reset=False)
```
`-1` launch params trigger `_resolve_launch_params` (tuning rules per dtype/quant/token count). `combine` with
`use_external_inp_buf=0` ⇒ zero-copy.

## Numerics / parity
fp8 dispatch (`Fp8DirectCast`, quant gate) + bf16 combine; combine multiplies the **unbiased** routing weight;
static-pad tokens must be **masked** from the reduction. EP keeps each expert whole → GEMM math identical to
single-GPU (best fp8 accuracy vs TP, which adds a cross-rank down-proj reduce). Greedy/temp=0 parity vs torch
MoE. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- AITER: `MoriAll2AllManager` (aiter EP adapter; on-box smoke `op_tests/multigpu_tests/test_mori_all2all.py`).
- vLLM/SGLang: register MoRI as all2all backend (`--all2all-backend mori_low_latency`). Init via
  `mori.shmem.shmem_torch_process_group_init("default")`; `reset()` between iterations.
- ⚠ `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is **incompatible** with MoRI — shared-expert fusion is done
  MoRI-side in the Wide-EP path.

## Pitfalls & anti-patterns
- Forgetting `ENABLE_STANDARD_MOE_ADAPT=ON` → no 3D AITER-compatible API (`RuntimeError`).
- JIT first-iteration cost (`~/.mori/jit/`); `MORI_PRECOMPILE=1` + warm before timing.
- Dynamic EP shapes vs HIP-graph static requirement → pad/static-ize (and **mask** the pad in combine).
- Expert load imbalance up to ~2× → EPLB-style frequency grouping.
- `max_total_recv_tokens` cap exceeded → kernel **asserts** (the config docstring warns); size it to worst-case.
- Using `dispatch_recv` outside `AsyncLL` → only AsyncLL supports the split recv phase.

## How to verify
mori `tests/python/ops/test_dispatch_combine_intranode.py -q` (correctness),
`tests/python/ops/bench_dispatch_combine.py` (bandwidth vs the table above); rocprofv3 to confirm overlap with
the grouped GEMM and the correct kernel mode; round-trip + greedy parity.

## Worked example (EP8 DeepSeek-V3, single MI300X node)
EP=8, E=256 (32/rank), hidden 7168, top-8, 4096 tok/rank, fp8 dispatch.
1. `EpDispatchCombineConfig(kernel_type=IntraNode, block_num=80, warp_num_per_block=8, gpu_per_node=8,
   quant_type="fp8_direct_cast", num_experts_per_token=8, hidden_dim=7168)`.
2. `MORI_PRECOMPILE=1` warmup (avoid JIT in the timed loop).
3. `op.dispatch(x, w, scales, indices)` (fp8) → per-rank tokens; run [[fused_moe_grouped_gemm]];
   `op.combine(y, w, indices)` (bf16) → source ranks.
4. Verify: `bench_dispatch_combine.py` ≈ 307/330 GB/s; rocprof overlap with the GEMM; parity vs single-GPU.
Inter-node: switch to `InterNodeV1` (throughput) or `InterNodeV1LL`/`AsyncLL` (latency), set `rdma_block_num`.

## Alternatives / cross-links
[[moe_dispatch_combine]] · [hip.md](hip.md) · [aiter.md](aiter.md) (the EP seam) · [triton.md](triton.md) ·
[[fused_moe_grouped_gemm]] · [`backends/mori_rccl/mori_ep.md`](../../../backends/mori_rccl/mori_ep.md) ·
[`backends/mori_rccl/deepep.md`](../../../backends/mori_rccl/deepep.md) (portable) · [overview.md](../overview.md) ·
[numerics.md](../numerics.md).

## Sources
- MoRI-EP guide + bandwidth/latency table + kernel types: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md ; https://github.com/ROCm/mori
- on-box: `ROCm/mori@b8af93f:python/mori/ops/dispatch_combine.py` (`EpDispatchCombineConfig` defaults, `dispatch`/`combine`/`dispatch_send`/`dispatch_recv`/`combine`, `dispatch_standard_moe`/`combine_standard_moe`, `Fp8DirectCast`).
- Wide-EP 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- MI355X TCO / SGLang+MoRI (quantized A2A 2.56× BW 28672→11200 B/tok; fp8_blockwise combine ~736 vs BF16 ~907 µs; InterNodeV1LL 1.52×/1.82× ≤256 tok/rank; 1.25× tok/s/GPU vs B200): https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
- a2a reference point (292 µs, grid=304, xGMI P2P): https://gau-nernst.github.io/amd-a2a/
