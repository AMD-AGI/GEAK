---
key: instruction-count reduction on an fp8 grouped GEMM already limited by the MFMA/global-load interlock, gfx950/MI355X, Triton block-pingpong schedule
type: anti-pattern
confidence: ★★
effect: ~1.00x or worse per-case across three independent directions whose intended change provably landed: -15.5% ds_read per MAC bought -0.59% / -0.08% time; -220 v_cndmask per wave with MFMA/LDS/load counts byte-identical paid -4.5%; recovering 12.5% masked prep lanes paid -1.07% to -1.64% against a +/-0.2% planted-null floor
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill
description: gfx950 fp8 grouped GEMM at MFMA/load interlock: cutting real work from a co-resident LDS/VALU pipe returned ~1.00x or worse, three ways.
keywords: ['moe-grouped-gemm', 'fp8-block-scale', 'gfx950', 'mfma-interlock', 'anti-pattern', 'occupancy', 'paired-ab-rig']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
roofline: LDS at 56% of speed-of-light co-resident with MFMA at 54%, i.e. not serialised in front of it
origin_kernels: ['fused_moe_kernel']
---
# Instruction cuts on a co-resident pipe do not convert
- lever: When the profile shows the secondary pipe co-resident with MFMA rather than serialised ahead of it, an axis that only removes instructions from that pipe is a cheap thing to price out early: on this shape it closed across three directions, so the round is better spent on the schedule itself.
- apply: Read the two pipes' speed-of-light fractions together before choosing the axis; if both sit near the same fraction, treat the critical path as the pipeline interlock and check whether a candidate perturbs the k-loop instruction MIX (waitcnt count moved 323 -> 336 here) rather than only shrinking it.
- verify: Confirm the intended cut actually landed in the compiled artifact (counter deltas plus a disassembly diff) BEFORE timing, then time it on a paired in-process rig with a planted known-null arm; a real counter win with a null or negative time delta is the signature of this axis.
- pitfall: Forcing the 32x32 MFMA shape killed the 16x16 form entirely and still bought nothing -> the win was booked on the counter, not the clock -> the counter and the timing must be reported as two separate claims.
- caution: Also verify this on your own shapes before writing the axis off: the finding is conditioned on a finely balanced block-pingpong schedule at high MFMA occupancy, and a schedule with an actually serialised LDS pipe would behave differently.
- source: run fused_moe_kernel-own16h, 2026-08-12, rounds 5-7 (all non-improving), director-validated run
