---
id: flash_mla_tilelang_to_triton
title: "Flash MLA decode: port the TileLang FP8 core to Triton (gfx950/MI355X)"
kind: expert_skill
authors: [yueliu14]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  operator: mla_attention
  arch_class: [deepseek_mla]
  gens: [gfx950]                  # authored and validated on MI355X (gfx950, CDNA4) only
  dtypes: [fp8_e4m3_fnuz]        # KV cache input format; bf16 is only the internal Q/compute/output dtype
  regimes: [decode]
  from_backend: tilelang
  to_backend: triton
# ---- expected effect: the validation gate's pass criteria ----
expects:
  isolated_speedup_min: 2.0
  parity: required
# ---- enforcement: STRICT for flash_mla — the agent MUST follow this skill as a mandate ----
# (advisory_prior is overridden by mode:strict; see e2e_workflow.js expertSkillsBlock + the
#  _fragments/expert_skills.md "Strict-enforcement skills" section.)
enforcement:
  mode: strict
  unittest:
    source: skill                    # judge from THIS skill's scripts/test_triton_decode.py + docs/unit-test.md
    cases_immutable: true            # do NOT re-derive the case set; keep tolerances + capture-safety probe
    require: {capture_safe: 1, correctness: all}
  optimization:
    mandatory_specs:                 # implement each, or REJECT the round (unless skip is benchmark-justified)
    - fp8_fused_dequant              # P0
    - capture_safety_ensure_warmed   # #0
    - dual_scope_fused               # SPEC8
    - autotune_or_bucket_dispatch    # SPEC1
    - shape_specialized_constexpr    # SPEC9
    skip_requires_benchmark: true
    forbid_split_k_unless_active: true
  targets:
    per_op_e2e_delta_pct_min: 5
    campaign_e2e_pct_target: 100
    reweight_regime: decode
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use
The Flash MLA (multi-head latent attention) sparse-attention **decode** path is the bottleneck on a
DeepSeek-V4 / DS_v4-class model served on AMD **MI355X (gfx950, CDNA4)**,
the production KV cache is stored as **FP8** (MODEL1_FP8Sparse, E8M0 scales), and the live implementation
is a **TileLang** kernel. Port it to **Triton** with FP8 in-kernel dequant so it can replace tilelang in
the sglang serving path (`SGLANG_HACK_FLASHMLA_BACKEND=triton`) — an editable, autotunable core that the
workflow can tune further.

## Mechanism
Decode is memory-bandwidth-bound on the KV-latent read. The #1 lever is a **single fused kernel** that
reads raw FP8 bytes from the KV cache, dequantizes with E8M0 scales, reconstructs RoPE byte-pairs, and
computes attention — with **zero intermediate buffers**. A Triton kernel that only consumes
pre-dequantized bf16 cannot replace tilelang in production: the Python-side dequant adds a separate GPU
launch + a full HBM read+write (≈3× traffic) and doubles VRAM, so even a 16×-faster bf16 kernel loses
all gains at e2e. Beyond FP8 fusion, the transferable wins are MFMA tiling (multi-head packing,
8×64 accumulator splitting to avoid register spill on the 512-wide d_v, bf16 dot products),
**shape-gated Split-K** (conditional, never global — it under-fills or regresses the cases that don't
need it) with its companion specialized-combine / buffer-pool / dual-scope pieces, dual-scope fusion,
and `@triton.autotune` with config pruning + total-tokens bucketing.

## Procedure
This recipe is split into two verbatim documents kept **separate on purpose** — write the unit test /
harness first, then optimize against it:

1. **Unit test & harness** → [`docs/unit-test.md`](docs/unit-test.md) (+ [`scripts/`](scripts/)).
   Copy `scripts/` into the kernel task dir as `kernel_src/`. It provides FP8 KV-cache data generation
   (`lib.py`), the PyTorch golden reference (`ref.py`), FP8 quant/dequant (`quant.py`), the
   correctness+perf driver (`test_triton_decode.py`), and `kernelkit/` utilities. Follow the
   `geak_unittest.py` contract (call the FP8 entry point, pass `KVScope` objects, never pre-dequant or
   `torch.cat` the dual scopes, real DS_v4 serving shapes, FP8 tolerances atol/rtol=2e-2).

2. **Optimization roadmap + full implementation guide** → [`docs/optimize.md`](docs/optimize.md).
   The roadmap (P0–P3 priorities, SPECs) is up top; the full ~1900-line implementation guide with
   complete code (Strategy 1–12, memory layouts, reference formulas) is inlined below it in the same file.
   Implement in priority order: **P0** FP8 in-kernel dequant (fused gather+dequant+attention); **P1**
   multi-head packing + tl.dot MFMA, accumulator splitting, bf16 compute, shape-gated Split-K WITH its
   companion specialized split_k=2/4/8 combine kernels + `SplitKBufferPool` + dual-scope Split-K; **P2**
   dual-scope fusion, `@triton.autotune` + config pruning + total-tokens bucketing, QK hoisting; **P3**
   exp2 fast-math, topk_length early exit. Then SPEC 9–12 go beyond the expert reference (compile-time
   shape specialization, triton cache pre-warm, LDS reuse, AMD hw hints).

3. **Gate on e2e, not isolated geomean.** The unit-test data-gen makes the MLA kernel look like ~1% of
   the trace, but it is ~60% of e2e GPU time on the live path. Do NOT stop on an isolated-geomean plateau
   — keep landing the P2 pieces and gate the final decision on an e2e throughput measurement (overlay
   into the server, bench tok/s). Reference e2e (c32, ISL=8192, OSL=1024, TP=8): tilelang 333 tok/s →
   best P0+P1 Triton 647 tok/s (1.94×, beats the sglang expert kernel's 614 tok/s by +5.4%).

## Knobs & pitfalls
- **FP8 dequant correctness is fragile**: E8M0 scale is `exp2(uint8 - 127.0)` (NOT `x/127`); RoPE is
  `lo | (hi << 8)` low-byte-first; FP8 load needs `bitcast=True`; scale `other=127` (neutral 1.0) for
  invalid positions; int64 for all KV pointer arithmetic (cache > 2GB); clamp NaN after dequant.
- **Split-K MUST be shape-gated** — a global Split-K scored geomean 2.39 vs 3.05 without it; the
  shape-gated version scored 3.69 (same code, gating is the whole difference). Store partials bf16, keep
  PartM/PartL f32, `torch.empty` (never zeros).
- **Autotune**: use EXACTLY the 10 proven configs, `num_stages=1` always, key on
  `total_tokens_bucket` (power-of-2). Adding `BLOCK_H=128,BLOCK_N=128,num_warps=8` (or `BLOCK_H≥32` with
  `h_q≤64`) causes VGPR overflow → GPU SIGABRT in TP=8 serving. Minimize constexpr count (each bool
  constexpr doubles compiled binaries → NCCL heartbeat timeout at startup; pre-warm the triton cache).
- **AMD buffer_ops** with KV > 2GB uses int32 addressing → silent corruption; disable per-launch above
  the 2GB threshold. Cast `.to(tl.float32)` after every `tl.dot` on AMD MFMA.

## Do-no-harm notes
- Decode-regime skill only — do not apply the decode tiling to prefill MLA (different shape regime).
- Optimize for the REAL production shapes (main topk=128 + extra topk=1024 = 1152 total; h_q=128;
  d_qk=d_v=512), NOT artificial topk=16384 large-topk configs that never occur in real serving.
- If parity fails at any candidate config, drop that config — never trade correctness for speed.
- Split-K, wider autotune configs, and dual-scope split-K must stay OFF on the cases they regress
  (small/medium topk, large batch, launch-floor b≤2); the gate verifies per-case ms does not regress.

## Sources
- Authored from on-box work on MI355X (gfx950), DS_v4 decode serving, TP=8.
- Optimization roadmap + full implementation guide verbatim: [`docs/optimize.md`](docs/optimize.md).
- Unit-test infrastructure verbatim: [`docs/unit-test.md`](docs/unit-test.md) and [`scripts/`](scripts/).
- STATUS: draft — needs on-box validation. Run `_contribute/validate_skill.py
  flash_mla_tilelang_to_triton --emit-plan` then `--record` with the kernel_workflow isolated A/B eval
  dir (and note the e2e tok/s overlay result in this section).
