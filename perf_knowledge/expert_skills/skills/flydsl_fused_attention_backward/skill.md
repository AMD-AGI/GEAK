---
id: flydsl_fused_attention_backward
title: "Attention backward: author a fused multi-GEMM FlyDSL kernel (CDNA3/CDNA4)"
kind: expert_skill
authors: [GEAK Team]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  operator: [attention_prefill_fmha, gqa_mqa_attention, mla_attention]
  arch_class: ['*']
  gens: [gfx942, gfx950]
  dtypes: [bf16, fp16]
  regimes: [training]               # training-only: no inference kernel can match this selector
  from_backend: ""                  # any — asm/ck/triton/fa_rocm backward, or no backward at all
  to_backend: flydsl
  profile_signature:
    op_name_regex: fmha_bwd|mha_bwd|_bwd_kernel|flash.*bwd|attn.*backward
    min_pct_gpu: 15.0
# ---- expected effect: the validation gate's pass criteria ----
expects:
  isolated_speedup_min: 1.10        # general authoring bar; the measured instance reached 1.38 (see Sources)
  parity: required                  # cos >= 0.999 on every gradient vs an fp32 reference
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
An attention **backward** kernel is a top GPU-time entry in a training or fine-tuning run and the live
path is a poor fit for the shape. The two situations that make authoring worthwhile:

- **The vendor fast path does not accept your head dims and pads to reach them.** Padding costs work
  proportional to the dim ratio, and it can also cost you the fast *forward*: the aiter gfx942 v3
  backward dispatch is gated on `hdim_q == hdim_v` while its ASM forward requires `hdim_v == 128`, so an
  asymmetric-head-dim model has no configuration where both kernels are fast.
- **No tuned backward exists for the variant at all** (GQA/MQA group ratios, small head dims, unusual
  causal alignment), leaving a generic CK-tile or Triton fallback.

**Do not use it when a tuned asm/CK backward dispatches natively for your exact shape** — that path is
strong and this recipe is unlikely to beat it. Check the dispatch conditions first; that check is
minutes and decides whether the rest of the work has any headroom.

## Mechanism
The performance comes from structure, not from instruction-level tuning, and the binding constraint is
the **256 arch-VGPR cap** rather than the tile size: a fused backward carries several long-lived operand
sets *plus* accumulators across the whole loop, so most classical "fewer instructions" optimizations
lose to spill. The four structural decisions, in the order they must be made:

1. **Fuse all the GEMMs into one kernel.** Splitting one out forces the scores and their gradient to be
   recomputed there, because those intermediates are too large to spill. Per score element the fused
   chain is `2·(HD_QK + HD_V + HD_V + HD_QK + HD_QK)` FLOPs; evaluate the split's overhead for your head
   dims before choosing (~+33% at symmetric 128, more as the dims diverge).
2. **Choose the operand-swapped MFMA fragment orientation** so the score comes out as `S[qrow, kpos]` —
   already the A-operand layout that both dV and dK need, since both contract over q rows. The
   probabilities and their gradient then **never leave registers and never transit LDS**. Free if chosen
   first, a rewrite if retrofitted.
3. **Loop KV-outer.** KV rows become exclusive to a block, so dK/dV are stored directly with no atomics
   and no workspace, and only dQ accumulates. The q-outer orientation puts the larger tensors on atomics
   instead — compare the fp32 workspace volumes for your shape rather than assuming.
4. **Pick the wave count against the register cap, not by occupancy intuition.** Fewer waves gives each
   wave more registers but doubles its slice of the split axis, hence doubles the operand set that must
   live in arch VGPRs — against a cap that does **not** move with occupancy. This is why the
   fewer-waves direction can lose here while it wins on GEMM.

Full derivations, the geometry constraints and the anti-pattern table are in
[[languages/flydsl/authoring_attention_levers]]; this skill is the regulated procedure that applies them.

## Procedure
Build in dependency order; each phase gates the next. Add the structural optimisations **one at a time**
(step 5) or an unexplained regression becomes unattributable.

1. **fp32 reference first.** A varlen shape descriptor (seqlens, nheads, `HD_QK`, `HD_V`, softmax scale,
   `cu_seqlens`), an fp32 forward returning `(out, lse)` and backward returning every gradient plus `D`,
   and a `compare` reporting cosine / `max_err/scale` / `rms/scale`. Fix the **causal convention here**
   (bottom-right aligned: `delta = seqlen_k − seqlen_q`, keep `j <= i + delta`) — top-left alignment is a
   different problem that silently produces high-cosine garbage. Validate the reference itself against an
   independent forward + autograd; a subtly wrong reference certifies a matching bug in the kernel.
2. **Harness before kernel.** Build *all* variants up front, then time them round-robin in one process
   keeping the per-variant minimum ([[profiling/benchmarking_methodology]] §single-kernel variant
   sweeps). Include a smoke shape with at least one non-tile-multiple length so tail handling runs every
   time. Add an **ISA probe** now, not later — register allocation is the dominant failure mode and is
   invisible from timings.
3. **Preprocess kernel:** `D = rowsum(dO ⊙ O)` into a **head-major** `[nheads, seqlen]` fp32 tensor. The
   main kernel must index `D` and `lse` the same way; transposing this still yields plausible-looking
   gradients on uniform shapes, so gate it explicitly.
4. **Main kernel.** Derive all geometry from the tile shape and assert the divisibility constraints at
   build time (see the levers card). `lse` is natural-log with the scale folded in, as a
   Flash-Attention forward emits it, so the kernel computes `p = exp2(s·scale·log2e − lse·log2e)`.
   Required structure:
   - **Three LDS tiles, not five** — read the second orientation of each staged tile out of the single
     copy instead of materialising a transposed one ([[optimization/lds_and_bank_conflicts]]).
   - **The causal cost axis on the SLOWEST grid axis** (longest-job-first)
     ([[optimization/xcd_l2_locality]]).
   - **One scheduling region, two barriers** — double-buffer the staging and issue the next tile's loads
     inside this tile's MFMA chain with no barrier between them; keep the staging branch-free by clamping
     rather than guarding ([[optimization/memory_pipelining]]).
   - **Re-partition the waves** for the output that contracts over the split axis, instead of reducing
     partials across waves.
5. **Bring-up ladder**, layout → scheduling → dispatch, measuring each step, and dumping the ISA at
   every one. A step that improves instruction count but spills is a regression.
6. **Head-to-head** against the live path interleaved in one process — and assert the baseline is
   actually computing gradients rather than silently no-op'ing on an unsupported configuration.

## Knobs & pitfalls
Expose `qk_head_dim`, `v_head_dim`, `block_m`, `block_n`, `num_waves`, `dq_groups`, `transposed_tiles`,
`pipeline`, `schedule`, `sched_steps`, and the grid-order flag, with defaults at the measured optimum.
Keep the three ablation paths (`transposed_tiles=True`, `pipeline=False`, grid-order off) alive
permanently — they are what make the step-5 ladder re-checkable in one interleaved process on new
hardware, and they let a regression be bisected against a mechanism rather than a commit.

Gates, in order: parity on every gradient (`cos ≥ 0.999`, relative bound ~3e-2) → resolved-geometry echo
→ **arch VGPR / AGPR / `v_accvgpr` move count** → the dispatch-order ablation ratio against its scheduling
model → the MFMA counter matching the decomposition exactly.

**Judge the wave count from the ISA, not the clock.** In the measured instance the best 4-wave build
reported *lower* `private_seg_size` than the 8-wave build and was still 15% slower, because the spill
relocated into the accumulator file where it does not appear as scratch. Count `v_accvgpr` moves
([[optimization/occupancy_and_registers]]).

**Dead ends, built and measured — do not re-derive** (mechanisms in
[[languages/flydsl/authoring_attention_levers]] §anti-patterns): growing the KV tile beyond the point
where the operand set fits the arch-VGPR cap; halving the q tile to relieve it; fanning dQ atomics across
more slots; `sched_group_barrier` hints once the region is already branch-free and single; hoisting
invariant staging address math out of the hot loop; folding the loop to a single barrier by deferring the
dQ phase; splitting the longest blocks to collect a low occupancy figure; coalescing a once-per-block
prologue.

## Do-no-harm notes
- **Training-only selector.** `regimes: [training]` means no prefill or decode kernel can match this
  skill, so nothing on a serving path is affected by its presence. It also lands `validation.status:
  draft`, so it is not auto-applied even when `use_expert_skills` is on (itself default OFF).
- **Symmetric head dims are usually not worth authoring.** Where `hdim_q == hdim_v`, the tuned asm
  backward dispatches natively and is strong; the recipe should decline rather than compete.
- **The speedup bar is generic; the measured instance is not.** `isolated_speedup_min: 1.10` reflects the
  shape-independent structural wins. The 1.38× (per unit of useful work) in the instance below also
  includes the baseline's padding tax, which exists only on gfx942 with asymmetric head dims — aiter
  supports native asymmetric dims on gfx950. Do not carry the ratio across gens.
- **Re-derive the wave count per gen.** gfx950's larger register file is the one place the fewer-waves
  schedule may win; the gfx942 result is not evidence about gfx950 either way.
- **Confirm the compute partition mode (SPX/NPS1) before trusting any timing.** The grid-order lever
  depends on workgroups reaching CUs in x-fastest order with static XCD assignment; CPX changes that and
  the device-scope atomic behaviour. A dispatch-order ablation that comes out neutral when the model says
  otherwise is usually a partition-mode problem, not a kernel problem.
- **Non-causal removes the cost gradient** that longest-job-first exploits — re-derive the grid order
  rather than assuming it still wins.
- Budget the output-cast overhead as this recipe's own cost if the epilogue leaves gradients in fp32; the
  vendor path typically folds that cast into an existing pass.

## Sources
- Levers, derivations and anti-patterns (general): [[languages/flydsl/authoring_attention_levers]].
- Worked instance with per-step deltas, counter expectations and the full dead-end ledger — fused
  five-GEMM MLA backward, `HD_QK=192`/`HD_V=128`, bf16, causal, varlen, gfx942 MI300X/MI325X SPX/NPS1,
  ROCm 7.1.0: [[operators/mla_attention/backends/flydsl]]. Measured 715–745 µs
  vs 984 µs for the padded-ASM chain (−27.3% wall while executing 15.4% less work, so 1.38× per unit of
  useful work), `cos = 0.999999` on all three gradients, and 1.49×/1.79× e2e against the two aiter
  configurations.
- Wave-count counter-measurement against HipKittens §3.3.2: [[optimization/mfma_scheduling]] ·
  [[languages/hipkittens/perf_findings]].
- FlyDSL authoring surface: [[languages/flydsl/authoring_optimization]] (register-allocation gate) ·
  [[languages/flydsl/debugging]].
