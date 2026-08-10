---
key: moe grouped gemm · gfx950 · large-batch
type: lever
confidence: ★★
effect: Standalone A/B on one compiled binary: +2.5% on the smallest case, +6.0% and +4.4% on the two larger cases (L2 hit 64.7 -> 85.1%, L2 miss traffic -58%, VALU/MFMA/wave counts byte-identical); +3.3% on the run geomean, on top of an already-retiled 1.75x kernel, and the last verified gain of a 7-round run.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950
last_seen: 2026-08-08
---
# Pad the grid to a multiple of NUM_XCD so the pid swizzle fires
- lever: On a multi-XCD CDNA part an XCD round-robin pid remap (pid -> (pid % NUM_XCD) * (grid / NUM_XCD) + pid / NUM_XCD) buys L2 locality only when the launch grid is a multiple of NUM_XCD; pad the grid up to that multiple rather than guarding the remap on divisibility.
- apply: Round the grid up to a multiple of NUM_XCD (8 on this part), keep the remap a bijection over the padded range, and apply it on EVERY path the kernel can take; padded programs exit on the existing bounds check before touching memory, so the pad is free.
- verify: Toggle the remap off in the SAME binary and diff L2 hit rate and L2 miss bytes: a real win shows up as locality at an identical instruction/VALU/MFMA/wave count. If those counters do not move, the remap never engaged.
- caution: Also verify the divisibility guard actually fired before recording any swizzle experiment as neutral - a knob that self-disables reads exactly like a measured negative, and here it hid a whole round's win. Also re-check any pid grouping already present: an L2-grouping knob worth 1.077x standalone went dead once the remap landed, and the two were anti-additive.
- source: run kernel_20_geak_0808_4h 2026-08-08
