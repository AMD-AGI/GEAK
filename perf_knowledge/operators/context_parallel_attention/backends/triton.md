---
title: context_parallel_attention on Triton — SOTA card
kind: sota_card
operator: context_parallel_attention
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# context_parallel_attention × Triton

## TL;DR
Triton provides the **portable CP kernel layer** on AMD: the local FA tile (`@triton.jit` FlashAttention
at `seq/cp`) plus the **LSE-merge** (`merge_attn_states`) that combines partial outputs across ranks. The
collective itself is RCCL or aiter's Triton comm primitives. There is **no single "ring attention kernel"**
— CP is the local FA kernel + an LSE-merge + an orchestration loop. This is the editable path; use it to
prototype CP or where aiter's CP path doesn't cover your model.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton local FA + LSE-merge | aiter Triton FA (`_triton_kernels/attention/`) + `merge_attn_states` (in MLA split, `attention/mla_decode.py`) | gfx942/950; bf16/fp16 | no public CP-scaling number on AMD — measure | the editable CP kernel layer |
| ring orchestration | framework loop (sglang prefill-CP proposal, zigzag) | gfx942/950 | maturing on ROCm | long-context prefill, PD-disagg |

> No hand-tuned CK/asm ring-attention kernel on AMD as of 2026-06; CP = portable Triton FA tile + LSE
> merge + collective.

## Config space / knobs
- Local tile: `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`, `waves_per_eu=2–3`,
  `schedule_hint=attention`, `knobs.amd.use_buffer_ops=ON`.
- CP-level: `cp` degree, zigzag causal balancing, overlap the next KV P2P with the current tile. See
  [tuning.md](../tuning.md).
- Collective: `allgather_reducescatter` on ROCm, not NCCL all-to-all.

## Numerics / parity
LSE-merge fp32 + associative; zigzag un-permutation; match single-GPU reference. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
The local FA `@triton.jit` and the merge are clean Python seams; the ring loop lives in the framework's CP
wrapper. Verify with `TRITON_PRINT_AUTOTUNING=1` and profile XGMI overlap.

## Pitfalls & anti-patterns
- **No overlap** (compute then communicate serially) → XGMI latency exposed, CP doesn't scale.
- **No zigzag** on causal masks → load imbalance, last rank dominates.
- `num_warps=8` on the local tile → spill.
- Treating CP as a single kernel — it's an orchestration of FA + merge + collective.

## How to verify
TTFT scales with `cp` at long context; CP output parity vs single-GPU at a fits-in-HBM seq; XGMI
utilization in `rocprofv3` shows transfer hidden under compute.

## Alternatives / cross-links
[overview.md](../overview.md) · [aiter.md](aiter.md) · [sglang_kernels.md](sglang_kernels.md) ·
languages: [[triton_amd]] · core: [[attention_prefill_fmha]] · collectives: [[allgather]] · [[reduce_scatter]].

## Sources
- aiter Triton FA + merge: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/attention/`, `attention/mla_decode.py` (on-box).
- SGLang zigzag ring attention: https://github.com/sgl-project/sglang/issues/22223
- AMD Triton knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
