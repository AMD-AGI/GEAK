---
key: bf16 unquantized fused-MoE expert GEMM pair (gate/up then down-proj) in Triton behind a Python launcher on gfx950/MI355, one launcher serving decode and prefill token buckets
type: lever
confidence: ★★
effect: 1.26x geomean isolated vs frozen baseline, mean of 3, parity clean on all cases; per-case 1.21x at decode M=64, 1.12x at prefill M=1080, 1.49x at prefill M=7218 — run-to-run spread stayed under half the smallest per-case gain
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both
description: Tune the per-M-bucket launch config first, then buy the tail by deleting satellite dispatches into grids that already exist: 1.26x on bf16 Triton fused-MoE
keywords: ['moe', 'grouped-gemm', 'dispatch-collapse', 'kernel-fusion', 'launch-overhead', 'm-bucket', 'prologue', 'bf16', 'triton', 'gfx950']
kernels: ['fused_moe_kernel', 'moe_align_block_size']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-17
levers: ['host.launch-config', 'host.dispatch-collapse', 'compute.epilogue-fusion']
---
# Delete the satellite dispatches once both MoE GEMMs sit at the HBM roof
- lever: On an untuned bf16 MoE expert-GEMM pair, take the per-M-bucket launch config first; once both GEMMs are at their bandwidth roof, the only live budget left is the non-GEMM satellites (align/scan, activation, zero-fill, reduce) — buy it by deleting dispatches, not by re-blocking them.
- apply: Ship a per-M-bucket config table before editing any kernel (the fallback heuristic ignored topk and left the group size at 1 at prefill); M-gate an epilogue that pairs the gate and up halves, giving that arm its own N block; then merge the prologue satellites into one scan kernel and fold each remaining elementwise satellite into a grid that ALREADY exists rather than into a fresh kernel.
- stack: total 1.26x geomean isolated (director-verified) = two rounds compounded
  - 1. per-M-bucket config table + M-gated gate/up epilogue pairing + a streaming cache hint on the wide-K weight operand — 1.21x cumulative (round 1, verified; the three lanes measured 1.16x / 1.10x / 1.07x standalone and the hand-merge beat all of them, since the conflict was which bucket picks which path)
  - 2. tail dispatch collapse, dispatch count 6 -> 4, non-GEMM share of device time ~12% -> ~7.6% — +4.2% on top of (1) (round 2, verified)
  - note: attribution is incremental in landing order; in round 2 both GEMMs were unchanged within noise, so all of its gain is deleted dispatches.
- verify: Paired same-session A/B against the frozen baseline (this box drifts 1-2% over tens of minutes, which manufactured and then erased a ~1% result), plus a dispatch count and per-dispatch duration from the trace to confirm a fold actually removed a launch.
- pitfall: A whole decode knob sweep read flat while the same knobs moved prefill → an earlier round's fused CLONE of the down-proj leg served that bucket and never received the constexprs → confirm from the trace which kernel runs in a bucket before believing a flat sweep.
- pitfall: A re-blocked satellite moving 4x the bytes measured the same duration → both satellites already sit on the box's per-dispatch floor → count dispatches, not CTAs or bytes; a fold into a NEW kernel netted about a fifth of what the same fold into an existing grid did.
- pitfall: Counting tokens per expert with an (E x N) comparison matrix spilled so hard decode went ~3x worse than the original baseline → per-element gather against an E-wide register vector → a histogram intrinsic, or a masked sum over a [SUB, E] tile with SUB <= 64.
- pitfall: Block-by-block comparison of the sorted token ids failed a correct re-implementation → within-expert order in the reference is atomic-arbitrary, and the decode shape (one expert per block) hides it → validate per expert: block purity, global permutation, expert-id and padded-count equality.
- pitfall: Two lanes stacked with clean text but one semantic conflict — both deleted the same zero-fill and folded it into a different existing kernel → single ownership → an explicit owner-selection block A/B'd on the merged tree, with the loser kept behind a default-off env flag.
- caution: Also verify the fold direction per bucket: folding the activation into the wide-K GEMM epilogue won at large M and lost at decode, so it shipped M-gated rather than global.
- source: run kernel_20_geak_0811_2h_kb_cleankb_2h, TechLead report + director validation, 2026-08-17
