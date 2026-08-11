---
key: paged/ragged attention inner body on gfx950 already at ~80% of achievable HBM bandwidth — deciding whether the residual headroom is an LDS/VALU tail worth attacking
type: anti-pattern
confidence: ★★
effect: Four disconfirming A/Bs, all at 1.0x or worse vs the frozen baseline: forcing occupancy 3 via launch_bounds -5.5% short-context / -6.5% long-context cases; removing a seemingly redundant reduction-phase barrier -1%; vectorizing an 8-scalar LDS transpose readback -4.5% on the long-context case; whole direction closed at 1.0x on all 9 cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: unclaimed-roofline-headroom-on-a-memory-bound-attention-body-attention-gfx950-mixed
description: Memory-bound label + ~20% roofline headroom did not mean attackable BW: VALU:VMEM 11.6:1, and LDS/occupancy attacks all regressed. Classify before planning.
keywords: ['roofline', 'memory-bound', 'lds-bank-conflict', 'occupancy', 'valu-vmem-ratio', 'attention', 'gfx950', 'profiling', 'anti-pattern']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention
regime: mixed
layer: learned
lifecycle: active
---
# unclaimed roofline headroom on a memory-bound attention body can be a compute-tail artifact
- lever: Before planning any bandwidth/occupancy work on an op the roofline calls memory-bound, classify with the VALU:VMEM instruction ratio and the memory-stall counter; at 11.6:1 with ~0.3% MemUnitStalled the residual headroom was a compute/latency tail, not attackable bandwidth.
- apply: Read register counts from the ELF msgpack note rather than the profiler's VGPR_Count field, and treat a free occupancy point (here 176 regs incl. AGPRs) as the sweet spot rather than a number to raise; the extra wave bought nothing even with zero spill.
- verify: For each candidate, run the isolated A/B per case — bank-conflict and LDS-wait counters were REAL here yet sat off the critical path, so the counter improving is not the gate; only the per-case ratio is.
- pitfall: an LDS-wait/bank-conflict counter looks alarming but every fix regresses -> the LDS traffic is fully overlapped by the concurrent HBM stream at low occupancy with many waves in flight -> stop the line and spend the round elsewhere.
- caution: A barrier that looks provably unnecessary may be load-bearing as bank-contention control — also verify by measuring its removal per case before treating it as a wasted stall.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, round 2 direction d1 + deep-explore direction d0, 2026-08-11
