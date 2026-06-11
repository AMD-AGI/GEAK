---
title: linear_attention_gated_delta — tuning
kind: technique
operator: linear_attention_gated_delta
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/flash-linear-attention
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# linear_attention_gated_delta — tuning

## The one principle: keep the state resident
GDN is **memory-bandwidth bound on the state matrix S** (d_k×d_v per head), not compute-bound. A
decomposed pipeline that reads/writes S to HBM 5+ times per token is estimated **10–50× slower** than a
fused kernel that holds S in registers/LDS for the whole update. Everything below serves that.

## Chunked prefill knobs (aiter `gated_delta_rule/prefill/` — [[triton_amd]])
The prefill is several fused Triton kernels (on-box): `chunk.py`, `chunk_o.py`, `chunk_delta_h.py`,
`fused_cumsum_kkt.py`, `fused_solve_tril_recompute.py`, `fused_gdn_gating_prefill.py`,
`causal_conv1d_fwd_split_qkv.py`.
- **Chunk size C** (commonly 64): larger C → more intra-chunk parallelism (better matmul utilization) but
  more state to hold and a bigger triangular solve; smaller C → more sequential chunk steps. Sweep
  {32,64,128}.
- The intra-chunk math is a **`solve_tril`** (lower-triangular solve, WY representation) + **cumsum** of
  decays — keep these in LDS; they're the chunk's serial dependency.
- AMD Triton: `matrix_instr_nonkdim=16`, `num_warps=4` (avoid 8 → spill), `num_stages=1`,
  `waves_per_eu=2–3`, `knobs.amd.use_buffer_ops=ON`.
- Align block/state tiles so S (d_k×d_v) fits LDS budget (64 KB CDNA3 / 160 KB CDNA4).

## Decode knobs (aiter `gated_delta_rule/decode/` — on-box)
`fused_recurrent.py`, `fused_sigmoid_gating_recurrent.py`, `causal_conv1d_split_qkv.py`.
- One state update per token → **launch/occupancy bound** at small batch. Maximize grid (one program per
  (batch,head)); use `schedule_hint`/`memory-bound-attention`-style scheduling.
- The gating sigmoid + L2-norm + conv1d are fused into the recurrent kernel — do **not** split them out.
- `num_warps=2–4`, `waves_per_eu=3–4` (memory-bound).

## Causal conv1d pre-step
GDN applies a short causal conv1d on Q/K/V before the scan. FLA provides Triton conv1d (so
`causal-conv1d` C++ lib is not required). Fuse the conv with the QKV split (`fused_qkvzba_split.py`,
`causal_conv1d_split_qkv.py`) to avoid an HBM round-trip. See [[causal_conv1d]].

## Autotune sketch
Prefill: key on `(seq, head_dim_k, head_dim_v, chunk)`; sweep `chunk∈{32,64,128}`, `num_warps∈{4,8}`,
`waves_per_eu∈{2,3}`. Decode: key on `(batch, head_dim_k, head_dim_v)`; sweep `num_warps∈{2,4}`,
`waves_per_eu∈{3,4}`. Re-tune per d_k/d_v (state size dominates).

## Verify the fusion is real
Prefill time should scale ~O(T/C) (sublinear jumps at chunk boundaries), decode ~O(T). If decode latency
grows superlinearly with state size beyond bandwidth, S is spilling to HBM — check VGPR/LDS with
`AMDGCN_ENABLE_DUMP=1`.

## Sources
- FLA chunked GDN kernels + Triton conv1d: https://github.com/fla-org/flash-linear-attention
- aiter GDN prefill/decode kernel files: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/gated_delta_rule/` (on-box).
- AMD Triton knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Decomposed-path 10–50× penalty (memory-bound on state): FLA / Qwen3-Next analysis.
