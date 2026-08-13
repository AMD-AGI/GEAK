---
key: the frozen QKV body of a paged attention kernel on gfx950 already at about 80% of achievable HBM bandwidth with non-temporal KV loads applied
type: anti-pattern
confidence: ★★
effect: closed axis over four directions at about 80% of achievable HBM bandwidth: forcing one more wave of occupancy cost about 6% on both the short and long-context cases, removing an apparently redundant barrier about 1%, vectorizing the transpose readback about 4.5% on the long-context case, and the reduce phase is only 3.6% (short) / 2% (long) of call time so its whole ceiling is about 1.037x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both
description: Near the achievable HBM ceiling, paged attention LDS-conflict/wait counters and occupancy are not levers: four body directions returned ~1.00x or regressed.
keywords: ['anti-pattern', 'closed-axis', 'roofline', 'hardware-counters', 'lds', 'bank-conflict', 'occupancy', 'waves-per-eu', 'split-kv', 'attention-decode', 'paged-kv', 'gfx950']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention_decode
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
roofline: memory-bound at roughly 0.8 of achievable HBM bandwidth before and after; the residual is ramp-up/down, not an attackable stall
levers: ['mem.lds-tiling', 'compute.occupancy', 'algo.split-kv-granularity']
---
# High LDS-wait counters next to a high roofline fraction can be a compute tail
- lever: Classify with the VALU:VMEM issue ratio and the memory-stall percentage before planning a bandwidth, LDS-padding or occupancy attack: a high fraction of achievable bandwidth alongside a near-zero memory-stall percentage means the residual is a compute/ramp tail rather than headroom.
- apply: Take register and LDS usage from the ELF metadata rather than the profiler's register counter, and take the reduce phase's share of call time from a trace before treating it as a target.
- verify: Re-time each candidate per case against the frozen baseline and expect the counters to stay high while the time does not move; that combination is the signature of an overlapped, off-critical-path cost.
- pitfall: LDS bank conflicts around 10.75% and multi-cycle LDS waits read as headroom -> at the free occupancy point with many co-resident waves the LDS traffic is fully overlapped by the KV stream -> both a barrier removal and a transpose-readback vectorization regressed.
Coarsening the split-KV partition size silently corrupted results (max abs diff around 3) -> the frozen body sizes a shared-memory logits array from that constant, so a larger partition overflows its inner dimension -> treat partition size as a body change with an LDS budget, not a config knob.
- caution: Also verify the free-occupancy register point before chasing another wave: here trimming accumulator registers to reach one more wave removed all spills and was still uniformly slower.
- source: run paged_attention_ragged-ch16h, 16h time-scaling campaign, 2026-08-12, directions r2_d0/r2_d1 and the deep-explore round
