---
title: moe_routing_topk on HIP/C++ — SOTA card
kind: sota_card
operator: moe_routing_topk
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/moe_align_block_size_kernels.cu
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
---

# moe_routing_topk × HIP/C++

## TL;DR
> HIP/C++ is where the **actual** router kernels live — both aiter's (`csrc/kernels/topk_softmax_kernels_
> group.cu`, `moe_align_block_size_kernels.cu`, `moe_fused_gate.cu`) and SGLang/vLLM's `moe_align_block_
> size`. Reach for HIP when you need DPP cross-lane reductions, exact LDS/XCD-aware grid control, or to
> author a kernel for an expert count/group config the library doesn't cover. This is the Tier-C edit seam.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| SGLang multi-block `moe_align_block_size` | yiakwy-xpu fork / SGLang PR #3613 | gfx942 (+MI100) | **7× MI300X, 10× MI100** vs prior; 41W→20W cycles; 52 VGPR / 5 kB LDS / 6.8% bank-conflict, 66 active CUs | the align&sort step, any E≤256 |
| aiter `topk_softmax_kernels_group.cu` (DPP) | `ROCm/aiter@a6bb49937` | gfx9, fp16/bf16/fp32 | 1.66× vs CK | grouped sigmoid select |
| aiter `moe_fused_gate.cu` | `ROCm/aiter@a6bb49937` | gfx942/950 | — | fused gate+select |

Recommend: don't re-author align&sort — consume the SGLang/aiter kernel. Author HIP only for an
unsupported group/expert config or a custom fusion.

## Config space / knobs
- **Grid (the dominant knob)**: < 8 blocks → pin to slowest XCD (XCD7); ≥ 8 → grid = multiple of **8**
  (num XCDs). Avoid `hipCooperativeLaunch` unless needed (L2 pressure on Die-Die exchange).
- **Block size**: tune down until **VGPR spills = 0** (tuned: 52 VGPR/wave). `__launch_bounds__` to cap.
- **LDS**: keep the per-CU histogram/scan in LDS (~5 kB), pad to avoid bank conflicts (achieved 6.8%).
- **Cross-lane**: DPP / `__builtin_amdgcn_ds_swizzle` / `__shfl` (wave64) for the group reduction instead
  of an LDS round-trip; honor `THREAD_PER_GRP=64/num_expert_group ∈ {2,4,8}`.

## Numerics / parity
fp32 accumulate; tie-breaks flip benignly. A hand-rolled kernel must match the reference scoring (sigmoid
vs softmax) and the **unbiased** weight rule (bias for selection only). See [numerics.md](../numerics.md).

## Integration (rebind seam)
- aiter: kernels compiled JIT/AOT into `aiter/jit/`; exposed via `aiter/ops/topk.py`, `moe_sorting.py`.
- SGLang/vLLM: `sgl-kernel` / `csrc/` op `moe_align_block_size` registered as a torch op; rebuild after
  editing the `.cu`. Autotune grid (XCD multiple), block size, LDS padding when rewriting.

## Pitfalls & anti-patterns
- Over-parallelizing a memory-bound router across all 8 XCDs → Die-Die comm dominates (MI100 beat MI300X
  in the author's test). Pin grid small for small T.
- `THREAD_PER_GRP` outside {2,4,8} → incomplete reduction (aiter #2153).
- Forgetting to hoist `torch.empty`/memset → caching-allocator malloc in the trace.

## How to verify
rocprof-compute: active CU count, LDS bank-conflict %, VGPR spills=0, grid multiple of 8. Isolated timing
of align&sort + gate; parity vs torch reference.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [`languages/hip_cpp/`](../../../languages/hip_cpp/overview.md) ·
[overview.md](../overview.md).

## Sources
- multi-block align&sort (grid/XCD rule, resource budget, 7×): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang ; https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- on-box HIP kernels: `ROCm/aiter@a6bb49937:csrc/kernels/{moe_align_block_size_kernels,topk_softmax_kernels_group,moe_fused_gate}.cu`.
