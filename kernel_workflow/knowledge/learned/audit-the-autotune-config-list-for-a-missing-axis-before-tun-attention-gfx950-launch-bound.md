---
key: the whole tunable surface of a Triton AMD-MFMA chunked linear-attention forward on gfx950 — decorator Config axes plus backend compile options — with a launch-floored tiny grid alongside large-batch grids
type: lever
confidence: ★★
effect: run A — adding the MFMA instruction-size axis (with a wider K-tile) to a config list that omitted it beat the converged autotuner by 1.18x on the mid-size batch case and 1.14x on the largest (medians of >=3 sweeps each), VGPR 44 -> 28 lifting a 62.5% occupancy ceiling, ~1.0x on the launch-bound smallest case. run B (same class, later) — the same audit extended past the decorator: re-ranking at the TRUE grids instead of the tiny one carried num_warps 4 -> 1 plus nonkdim=16 for ~1.5x/~1.2x, and a backend compile option (denormal flush-to-zero) added +3-4.5% at EVERY grid with bit-identical output; a per-grid table beyond that had exactly one entry (num_warps 1 -> 4 at the launch-floored grid, +6.2%, and ~60% worse if globalised)
confirms_cited: 1
confirms_blind: 1
losses: 2
attempts: 6
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: audit-the-autotune-config-list-for-a-missing-axis-before-tun-attention-gfx950-launch-bound
description: Audit the whole tunable surface — omitted decorator axes, the grid the list was keyed to, backend compile options — before tuning in it: 1.18x, +3-4.5% more
keywords: ['autotune', 'config-sweep', 'mfma', 'occupancy', 'vgpr', 'launch-bound', 'interleaved-ab', 'num-warps', 'launch-meta', 'numerics', 'isa-check', 'measurement-method']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: launch-bound
layer: learned
levers: ['compute.launch-meta', 'compute.compile-options']
cost: L1
lifecycle: active
verified_on: 2026-08-12
roofline: launch/config-bound -> store-bandwidth-bound -> read-bound; occupancy is a lever only on the grid that underfills the machine (there ~1 wave per CTA over twice as many SIMDs), the saturated grids only fragment the store
---
# Audit the tunable surface before tuning inside it
- lever: treat the tuning SPACE as the object to audit, in three layers: (1) axes the @triton.autotune decorator omits (`matrix_instr_nonkdim` was absent from this class's vendored list entirely), (2) the grid the existing list was keyed/ranked on — here it was ranked for the tiny case and mis-served the real ones, (3) backend compile options passed at build time, which behave as extra config axes and are orthogonal to the table.
- apply: enumerate the axes present in the Config(...) entries, sweep the missing ones jointly with the present ones (nonkdim 16 vs 32, K-tile width), then re-rank the survivors at each grid you actually grade on; keep one config table only for axes that genuinely disagree across grids and leave the rest global.
- stack: total 2.1507x director-verified on the frozen baseline = three directions compounded, attribution incremental in landing order
  - 1. autotune re-rank at the true grids + host launch path (round 1, verified 1.5528x) — the bulk
  - 2. host launcher bypass integrated with a write-through store modifier (round 2, verified 2.029x cumulative; the two were measured standalone at 1.9243x and 1.136x and are orthogonal by construction) — +30.7% on top of (1)
  - 3. this card's lever, per-grid num_warps entry + the compile option (round 3, verified 2.1425x cumulative) — +5.6% on top of (1,2)
- verify: compare medians of >=3 independent sweeps against the unmodified decorator (an autotune sweep re-converges to different winners on identical source), settle the final ranking in the grader's own timing loop, and check the register count moved the way the occupancy story predicts.
- pitfall: two variants that were byte-identical machine code to the incumbent measured ~5% faster in all reps -> fixed-order interleaving gives whichever variant is timed first a position artifact -> rotate variant order per rep and carry one ISA-identical control to expose the residual.
- pitfall: several store/load policy knobs measured "within noise of the incumbent" and were recorded as a ceiling -> `eviction_policy` and `.ca` are not lowered at all on this backend (byte-identical disassembly) -> disassemble the two builds first; a knob that does not change codegen cannot be measured.
- pitfall: an out-of-harness sweep rig, an in-process probe and the grader disagreed on the same pair by 1.2% / 7.5% / 6.2% -> a long-lived probe process carries host pressure the grader does not -> use any external rig to RANK only.
- pitfall: a minimum-waves-per-EU request cost up to 2.2x -> the kernel was not register-capped, and at scale capping waves/EU removes the occupancy that hides store latency -> check a knob's default and whether the kernel is in that regime before funding a round on it.
- caution: the flush-to-zero compile option is a numerics-MODE change even when output is bit-identical (at these operand magnitudes no intermediate is ever denormal, so only the fixup code leaves) — also verify downstream policy allows it, since dropping it costs the 3-4.5%; and also verify a per-grid entry against the grid you will deploy on, since the one entry here is worth 6.2% locally and ~60% globally.
- caution: an earlier enumeration of this surface concluded no second knob had headroom; the later run found one in the backend-option layer it had not enumerated, so also verify which layers your audit actually covered before calling the surface closed.
- source: run kernel_20_geak_0808_4h 2026-08-08; run kernel_20_geak_0811_2h 2026-08-12 (directions r1_d0 and r3_d0, Director-verified)
