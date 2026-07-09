---
kernel_class: fused_moe_grouped_gemm
gfx: gfx942
regime: bf16 unquantized MoE (Mixtral-8x7B E=8 topk=2 K=4096 I=14336), vLLM
confidence: 2
confirms: 1
last_seen: 2026-07-08
---

# aiter bf16 fused-MoE (rocm_aiter_ops.fused_moe) vs vLLM Triton fused_experts — gfx942

**Lever:** server env `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1` routes the unquantized
bf16 fused-MoE through aiter's asm `fmoe_g1u1` (weights shuffle-once at load via
`rocm_aiter_ops.shuffle_weights`, layout (16,16); quant_method=0, activation SILU). No source edit,
memory-neutral (shuffled bf16 same size). Correctness parity vs fp32 ref = expected_close
(max_rel ~0.006, well inside bf16 2e-2).

**Isolated bake-off (immutable oracle, GPU-pinned, steady-state = weights pre-shuffled):**
- prefill M=8192: **1.29x** (Triton 16.27ms → aiter 12.59ms) — big win on large-M.
- decode M=64: **0.95x** (0.836 → 0.883ms) — REGRESSES.
- decode M=1: **0.70x** (0.245 → 0.350ms) — REGRESSES hard.
- meta weight_norm-weighted = **1.1355x** (weights prefill 0.7 / dec64 0.24 / dec1 0.06).

**caution: the weighted number is prefill-biased and MISLEADING for decode-bound serving.** This
Mixtral e2e (osl=1024) is decode/TPOT-dominated; the meta floored decode via `--min-regime-share 0.3`
because the profiling window was prefill-biased. aiter's global MoE flag applies to BOTH regimes, so
it drags decode down where serving throughput actually lives. Classic "prefill-only iso win → e2e
regression" trap — MUST be e2e-gated on a decode-bound run, not accepted on the weighted iso alone.
The real win would need prefill→aiter / decode→Triton dispatch routing (Tier-C dispatcher rebind),
which the global env flag cannot express.

**e2e-transfer: UNKNOWN (isolated only, not yet e2e-gated).** Prior Mixtral MoE "wins" in this
program were unrequested quant rewrites (bf16→fp8 1.68x, int8 1.46x) that failed the parity/quant
gate — this bf16 lever is quant-free and parity-safe, but its decode regression is the open risk.

**flydsl fused-MoE on this box is an int4/quant path** (`fused_flydsl_moe.py` in_dtype="int4_bf16",
needs packed weights + scales) — NOT a bf16 candidate; do not put flydsl in a bf16 author_plan.
Tier-C author = optimize the editable Triton `fused_moe_kernel` (rewrite) to win decode too without
the aiter decode penalty; memory-neutral.
