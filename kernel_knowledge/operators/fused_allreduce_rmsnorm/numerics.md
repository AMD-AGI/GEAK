---
title: fused_allreduce_rmsnorm — numerics
kind: technique
operator: fused_allreduce_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/design/fusions/
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# fused_allreduce_rmsnorm — numerics & parity

## Fusion is math-preserving (the bf16/fp16 path)
Fusing AR+RMSNorm doesn't change the math: AR fp32-accumulates, RMSNorm fp32-reduces over the hidden dim,
γ-scales. The fused kernel must keep the **fp32 reduction** for the norm and the **fp32 accumulate** for the
AR — then it is parity-safe vs the unfused pair (modulo benign AR reduction-order deltas). Gate with
greedy/temp=0 parity, not byte match.

## SP-rewrite equivalence
The SP form (`RS + local RMSNorm + AG`) is equivalent to `AR + RMSNorm` **only if** the norm runs over the
**full hidden dim** — RMSNorm reduces over hidden, which is **not** the sharded dim (sequence is sharded),
so the local norm sees the complete hidden vector for its tokens. Verify the shard axis is the **sequence/
token** dim, not hidden; sharding hidden would break the norm reduction. Greedy parity vs the non-SP path
catches a wrong axis.

## fp8 static quant epilogue — the gate
The `...StaticQuantFP8` patterns and the fused SP kernel's `_quantize_fp8_stage` apply an **fp8 quant after
the norm**. This changes values → accuracy gate:
- fnuz on gfx942 (wrong dialect = 2× off); OCP on gfx950.
- static (per-tensor) scale must be calibrated; a stale scale clips/saturates.
- re-run a small eval when enabling the quant epilogue.

## Residual add ordering
`FusedAddRMSNorm` adds the residual **before** the norm (`norm(x + residual)`). Preserve that order; adding
after the norm is a real regression. The fused kernel folds the add into the same pass — verify it matches
the reference order.

## Verification recipe
1. Numeric: fused AR+RMSNorm output vs separate AR then RMSNorm (fp32 reference) — bf16 within tolerance.
2. SP equivalence: SP path vs non-SP — greedy parity (catches wrong shard axis / norm dim).
3. Quant gate: a small eval (gsm8k) when the fp8 static-quant epilogue is on.

## Sources
- fusion patterns (add/quant ordering): https://docs.vllm.ai/en/latest/design/fusions/
- fused SP kernel fp8 stage: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- cross-ref: [[rmsnorm]], [[fused_add_rmsnorm]], [[fused_norm_quant]] numerics.
