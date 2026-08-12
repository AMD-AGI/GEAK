---
key: ttg.convert_layout plus LDS round-trips between the reduction and the epilogue store in a small-tile Triton top-k / routing kernel on gfx950
type: lever
confidence: ★★
effect: The patch that owned producer+consumer as one edit cut 599 -> 530 static instructions (-11.5%), ds_write 9->1, ds_read 3->1, s_barrier 3->1, local_alloc -> 0; it paid +7.2% (12/12 paired wins) on the only case with device time still above the host enqueue and +2.3% geomean, and ~0% on the two smaller launch-masked cases. Director-verified end state 2.351x geomean (2.33 / 2.46 / 2.26 from smallest to largest token count). Two earlier attempts at the same seam authored only at the consumer measured exactly 0%.
confirms_cited: 0
confirms_blind: 1
losses: 1
attempts: 6
toolchain: ROCm 7.2 / triton 3.6.0 / torch 2.11.0+gitd0c8b1f / gfx950 CDNA4
last_seen: 2026-08-10
name: reshape-a-tile-into-the-layout-s-own-factorisation-instead-o-topk-routing-gfx950-launch-bound
description: Reshape a tile into the layout's own factorisation instead of fighting convert_layout: -11.5% instructions, +7.2% on the one device-bound top-k case
keywords: ['cross-lane', 'lds', 'topk', 'launch-bound', 'isa-check', 'tile-shape', 'workgroup-size']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: topk_routing
regime: launch-bound
lifecycle: active
---
# Reshape a tile into the layout's own factorisation instead of fighting convert_layout
- lever: When a small-tile Triton kernel spends its instruction budget on ttg.convert_layout plus LDS round-trips (ds_write/ds_read/s_barrier pairs, a local_alloc) between a reduction and its epilogue store, treat the encoding mismatch itself as the target rather than the store width or the arithmetic. Reshaping the value into the layout's own factorisation so the wanted axis becomes a real tensor axis is what paid; rewriting at the consumer, and choosing the layout via pointer arithmetic at the producer, each measured 0% or negative.
- apply: Read the distributed encoding actually chosen (sizePerThread / threadsPerWarp / warpsPerCTA), reshape the tile into that same 3-D factorisation e.g. [BLOCK_M, LANES, REGS], run the reduce over the axis you want collapsed, and let the result's slice encoding BE the store encoding by construction. Author producer and consumer in ONE patch — this seam is not separable.
- verify: Diff static instruction count and the ds_*/s_barrier/local_alloc counts in the ISA before and after; a linear instructions-to-device-time model calibrated on this kernel held out of sample (its per-instruction cost predicted ~20% high, and -11.5% instructions gave -11.6% device time), so instruction count is a usable pre-compile design metric. Then confirm the convert_layouts became identity in the TTGIR rather than being re-materialised.
- pitfall: two rounds at this seam measured exactly 0% -> each authored only the consumer, leaving the producer's encoding in place so the convert_layout was re-materialised -> land producer and consumer as one patch.
- caution: Also verify the warp count the reshape assumes: with 128 threads the free within-warp lane reduce is what makes the reshape cheap, and at 256 threads the same reduce crosses warps through LDS — so re-check the compile knob you tuned in an earlier round, because it may have become a correctness-of-design constraint rather than a free parameter.
- source: run kb_on_0810 2026-08-10
