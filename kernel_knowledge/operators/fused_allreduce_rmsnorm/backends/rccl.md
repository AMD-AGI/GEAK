---
title: fused_allreduce_rmsnorm on RCCL/MoRI — SOTA card
kind: sota_card
operator: fused_allreduce_rmsnorm
backend: rccl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# fused_allreduce_rmsnorm × RCCL / MoRI

## TL;DR
> RCCL itself does **not** fuse the norm — it provides the all-reduce (or the reduce-scatter/all-gather for
> the SP rewrite), and the **norm runs as a separate kernel** unless you take the SP-rewrite (where the
> norm is folded between RCCL's RS and AG) or vLLM's AR-epilogue fusion (which fuses the norm onto the
> result, not into RCCL). So on the RCCL path, "fusion" means **SP-rewrite using RCCL RS/AG** + **overlap**,
> not a single RCCL kernel. For a truly fused single kernel use aiter ([aiter.md](aiter.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| RCCL AR + separate RMSNorm | `ROCm/rccl` + aiter/torch RMSNorm | gfx942/950 | ~316–330 GB/s AR busbw + a small norm | baseline; large msgs |
| SP rewrite on RCCL (RS + norm + AG) | RCCL RS/AG + local norm + AsyncTP | gfx942/950 | norm-on-shard + overlap | SP TP layers |
| MoRI-CCL AR + norm | `ROCm/mori` | gfx942/950 | latency-focused (vendor) | small/latency-sensitive |

## Config space / knobs
- SP rewrite + AsyncTP (overlap RS/AG with the GEMMs); `NCCL_MIN_NCHANNELS=112` (sub-island, A/B),
  `RCCL_MSCCLPP_THRESHOLD`, `-G 1`, `GPU_MAX_HW_QUEUES=2`. Full table:
  [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md).

## Numerics / parity
RCCL fp32 accumulate + fp32 norm reduce → parity-safe; SP equivalence (norm over hidden, shard the sequence);
fp8 quant gate if added. See [numerics.md](../numerics.md).

## Integration (rebind seam)
SP rewrite is a layer/compile change; vLLM AR-epilogue fusion (`enable_fi_allreduce_fusion`) fuses the norm
onto the AR result (the AR can be RCCL). For one fused kernel, switch to aiter.

## Pitfalls & anti-patterns
- Expecting RCCL to fuse the norm — it doesn't; the fusion is SP-rewrite or aiter-side.
- SP rewrite adds an AG — only wins with overlap + norm-on-shard.
- MIN_NCHANNELS disables the tuning model — A/B.

## How to verify
rcl-tests for the AR/RS/AG bandwidth; rocprof for norm↔comm overlap; e2e SP tok/s; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) (true fused kernel) · [hip.md](hip.md) ·
[`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md) · [overview.md](../overview.md).

## Sources
- RCCL env / algos: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- SP / AR-epilogue fusion passes: https://docs.vllm.ai/en/latest/design/fusions/
