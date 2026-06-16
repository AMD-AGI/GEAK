---
title: sampling_topk_topp on hip — SOTA card
kind: sota_card
operator: sampling_topk_topp
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/cpp_itfs/sampling/sampling.cuh
  - https://flashinfer.ai/2025/03/10/sampling.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# sampling_topk_topp × hip

## TL;DR
The production sampler **is** a HIP kernel: aiter's `csrc/cpp_itfs/sampling/sampling.cuh` is a hand-written
HIP/hipCUB implementation of Dual-Pivot Rejection Sampling. So "hip" here is not a hypothetical port — it is
the **editable source** of the live path, and the right place for a Tier-C rewrite (custom filter, a
fused-from-logits variant, a min-p kernel aiter lacks, or a NaN-safe greedy). It uses hipCUB
`BlockScan`/`BlockReduce`, `hiprand` philox, `atomicMin` tie-break, one threadblock per request, and a
templated `DETERMINISTIC` prefix-sum.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `sampling.cuh` dual-pivot rejection sampler | `ROCm/aiter@a6bb49937:csrc/cpp_itfs/sampling/sampling.cuh` | gfx942/950, fp32 | sorting-free, no host sync; the live AMD path | top-k/top-p/joint decode sampling |
| hand-HIP fused `logits→temp→filter→Gumbel→token` | author ([[hip_cpp]]) | gfx942/950, fp32 | single kernel from logits (skip softmax materialize), NaN-safe greedy | a fusion/feature aiter lacks (min-p kernel, processed-logprobs) |

## Config space / knobs (kernel internals — Tier-C)
- **`BLOCK_THREADS=1024`** — one threadblock per request row; row processed in
  `ceil_div(V, BLOCK_THREADS·VEC_SIZE)` chunks. Block = multiple of 64 (wave64).
- **`VEC_SIZE`** — vectorized load width (`vec_t<float,VEC_SIZE>::cast_load`); widen for HBM efficiency
  over the `V`-row (the kernel is a streaming read).
- **`DETERMINISTIC` template** — `DeterministicInclusiveSum` (reproducible, slower) vs hipCUB `BlockScan`.
- **dual-pivot loop** — `pivot_0 = probs[sampled_id]`, `pivot_1 = (pivot_0+high)/2`; two BlockReduce sums
  (`>pivot_0`, `>pivot_1`) per round decide accept/narrow/raise; range shrinks ≥½ per round.
- **RNG** — `hiprand_init(philox_seed, row, philox_offset, &state)` (stateless, no host sync).
- **tie-break** — `atomicMin(&sampled_id, idx)` → lowest index.
- Compile: `--offload-arch=gfx942/gfx950`, hipCUB headers; JIT via aiter `compile_ops` on first use.
  `-Rpass-analysis=kernel-resource-usage` to watch VGPR/LDS (the SamplingTempStorage union sizes LDS).

## Numerics / parity
fp32; non-associative prefix-sum → `DETERMINISTIC` for reproducibility; lowest-index tie-break; philox
reproducible. Statistical equivalence (KL gate); greedy exact. All-NaN/-inf row → OOR token id (add a
NaN-guard if authoring; see [../numerics.md](../numerics.md)).

## Integration (rebind seam)
aiter JIT-compiles `sampling.cuh` into `module_*` and registers torch ops (`top_k_top_p_sampling_from_probs`
etc.). A custom HIP sampler is wired the same way (aiter `compile_ops` / a `torch.library` op) — editing the
`.cu`/`.cuh` requires recompiling the aiter module. Verify: rocprofv3 shows the sampling kernel + **no D2H
sync**.

## Pitfalls & anti-patterns
- `warpSize==32` assumptions (it's 64) in any cross-lane/block reduction sizing.
- LDS overflow from the `SamplingTempStorage` union at large `BLOCK_THREADS`/VEC_SIZE → check
  resource-usage (64 KB CDNA3 / 160 KB CDNA4).
- Skipping the deterministic-scan path when reproducibility is required (non-monotone CDF).
- No NaN-guard → OOR token id on fully-masked rows (the SGLang-observed failure).
- bf16 probs → biased thresholds; fp32.

## How to verify
Statistical KL gate vs sorted multinomial; greedy exact parity; fixed-seed reproducibility with
`DETERMINISTIC`; `-Rpass-analysis=kernel-resource-usage` for occupancy; rocprofv3 no D2H sync; latency vs
the PyTorch sort at `(M, V=128k–256k)`.

## Alternatives / cross-links
[aiter.md](aiter.md) (ships this kernel) · [vllm_kernels.md](vllm_kernels.md) · [triton.md](triton.md) ·
[../overview.md](../overview.md) · [[hip_cpp]] · [[cumsum_scan]] · [[argmax_topk]] · [[softmax]].

## Sources
- aiter HIP dual-pivot sampler (full algorithm, philox, atomicMin, DETERMINISTIC, BLOCK_THREADS=1024,
  VEC_SIZE, SamplingTempStorage): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/cpp_itfs/sampling/sampling.cuh` (on-box).
- Algorithm + numerics (non-monotone prefix-sum): https://flashinfer.ai/2025/03/10/sampling.html
- HIP kernel language (wave64, LDS, resource usage): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
