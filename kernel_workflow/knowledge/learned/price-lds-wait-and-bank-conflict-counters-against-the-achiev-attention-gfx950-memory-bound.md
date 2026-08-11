---
name: price-lds-wait-and-bank-conflict-counters-against-the-achiev-attention-gfx950-memory-bound
description: LDS-wait and bank-conflict counters read as headroom but were overlapped: all three edits aimed at them lost (-5.5%/-6.5%, ~1%, ~4.5% on the large case)
keywords: [counters, lds, occupancy, vgpr, roofline, measurement-method, attention, memory-bound, isa-check]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: LDS bank-conflict / occupancy restructures on a Triton grouped attention stage-1 kernel on gfx950 already near its achievable HBM roof with non-temporal loads in place
lifecycle: active
type: anti-pattern
confidence: ★★
effect: the profile looked like a textbook LDS problem - LDSBankConflict 10.75%, SQ_WAIT_INST_LDS ~2.35 per LDS instruction, MeanOccupancy 2.63 and ~18% apparent roofline headroom - and every edit aimed at it lost: forcing occupancy 3 via __launch_bounds__(256,3) reached 144 VGPR + 0 AGPR with zero spill and was uniformly slower (-5.5% small case, -6.5% large case), the free occupancy-2 point (160 VGPR + 16 AGPR = 176, LDS 16384B) being the sweet spot; removing a barrier that read as provably unnecessary regressed ~1% (it was load-bearing, suppressing concurrent shared-buffer bank contention); and short-circuiting an 8-scalar transpose readback to one vectorized read regressed the large case ~4.5%; the disconfirming number explaining all three is that with genuine non-temporal loads in place the large case already reads its 4.295 GB at ~80% of achievable HBM bandwidth, so the counters sit off the critical path - LDS traffic is fully overlapped by the HBM stream at occupancy 2 with ~17-wave residency, VALU:VMEM 11.6:1 and MemUnitStalled 0.30%
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign, 2026-08-11
last_seen: 2026-08-11
---
# Price LDS-wait and bank-conflict counters against the achievable-bandwidth roof before attacking them
- lever: When counters say LDS-wait or bank conflict but the kernel is already near its achievable bandwidth roof, treat those counters as overlapped rather than as headroom, and spend the round proving it cheaply instead of funding a restructure.
- apply: Two one-edit diagnostics settle it - delete a barrier you believe is redundant, and collapse a scalarised shared-memory access into a vector one; if both regress, the LDS path is hidden behind the memory stream and the residual gap is ramp-up plus the genuine bandwidth limit. Compute the VALU:VMEM issue ratio and read MemUnitStalled before writing any bandwidth or occupancy plan, since a high ratio with near-zero memory stall means an apparent roofline gap is a compute-tail artifact.
- pitfall: a spill-free forced-occupancy-3 build looked strictly better on paper -> AGPRs count toward the same budget and the naturally chosen occupancy-2 point was already the sweet spot -> read VGPR/AGPR from the ELF msgpack rather than from the profiler's register field, and treat occupancy as a measurement rather than an assumption.
- verify: Re-measure each one-edit diagnostic on both the small and the large case; a regression on the large case is the tell that the counter was overlapped.
- caution: Also verify the non-temporal / streaming load path is genuinely in place first - the overlap argument here depends on it, and the same counters can be real headroom on a kernel still short of its achievable roof.
- source: chuschen 16h time-budget campaign, 2026-08-11
