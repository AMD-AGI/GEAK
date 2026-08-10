---
key: quantized gemm · gfx950 · compute-bound
type: lever
confidence: ★★
effect: +3.6% at M=64k from moving one load statement; +5.5% / +3.5% / +3.6% at M=2k / 32k / 64k from a whole-body sweep the next round; +1.0% more from epilogue order -- all at zero ISA-footprint change. 3 of 4 tries paid; the 4th cost 4-5% at M=2k on a body that had already been reordered.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: triton 3.6.0 / torch 2.11.0+gitd0c8b1f / gfx950 CDNA4
last_seen: 2026-08-10
---
# on a latency-bound GEMM, source statement order is a tunable: resident loads to the head, streaming loads to the tail
- lever: when the profile says dependency-wait rather than bandwidth, WHERE each load is issued in source order is worth several percent at zero register or LDS cost: classify every load stream as cache-RESIDENT (a few scalars that retire for free at the head of the queue) or STREAMING (a vector or tile whose latency can only hide behind the tile loads), then issue resident loads at the head of the loop body and streaming loads at the tail.
- apply: sweep 3-5 source positions per statement across the WHOLE body rather than one statement at a time -- a single-statement sweep cannot see the anti-symmetry between the two stream kinds; then try crossing load order against consume/dot order, independently worth +2.1% here; the same treatment of the epilogue (hoist addresses and masks, then convert-and-store each group immediately) paid again.
- verify: confirm instruction counts, shared-memory size and register count are unchanged so the delta is scheduling and not footprint, and adjudicate variants interleaved inside ONE measurement window against an unchanged control, since these effects are smaller than typical cross-window drift.
- caution: also verify the ordering in the exact body you ship: the optimum here did not transfer across tile shapes nor survive a change to warps/stages, and a reorder that adds one loop-carried value collapsed the pipeliner's multibuffering (shared 98304->16384, matrix ops 24->8, -35%) with no spill and a LOWER register count to warn you.
- source: run kernel_20_geak_0808_4h 2026-08-10
