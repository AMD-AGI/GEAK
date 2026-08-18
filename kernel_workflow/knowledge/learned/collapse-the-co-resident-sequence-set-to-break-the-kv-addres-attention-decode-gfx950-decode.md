---
key: paged bf16 KV decode attention in HIP C++ on gfx950/CDNA4 whose per-sequence KV span is a power of two, so many live sequences alias onto the same cache sets
type: lever
confidence: ★★
effect: 1.240x isolated standalone vs frozen baseline (1.2489x after integrating a second disjoint direction), every case up, largest on the long-context cases; cross-case HBM-read-volume spread narrowed 27% -> 7.4%; the kv-head-fastest variant of the same remap reached only 1.049x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode
description: Sequence-major workgroup dispatch collapses the co-resident sequence set and breaks the power-of-two KV base-address phase: 1.24x, read spread 27%->7.4%.
keywords: ['attention-decode', 'paged-attention', 'paged-kv', 'workgroup-mapping', 'l2-locality', 'co-residency', 'decode', 'memory-bound', 'gfx950', 'raw-hip']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
levers: ['mem.workgroup-mapping']
---
# Collapse the co-resident sequence set to break the KV address phase
- lever: When each sequence's KV span is a power of two, the base addresses of all live sequences share a cache-set phase; try reindexing the launch grid sequence-major so every workgroup of one sequence is consecutive, shrinking the set of sequences resident at once.
- apply: A pure blockIdx-decode change in the kernel prologue (a swizzle-mode switch over the (partition, kv_head, seq) decomposition); no math and no traffic change, so oracle parity is free and the diff is a handful of lines.
- verify: Interleaved per-case A/B against the frozen baseline plus the per-case HBM read-volume counter: the signature of the real mechanism is the cross-case read-volume spread collapsing, not just a faster geomean.
- pitfall: Making the kv-head dimension fastest-varying looked like the same idea and returned only 1.049x -> it packs 8 workgroups onto one page while leaving all 64 sequences live, so the aliasing set is unchanged -> the lever is how many sequences are co-resident, not the innermost dimension order.
Undoing the hardware XCD round-robin on top of the swizzle was 0.3% slower and matching the reduce grid's order to the writer's XCD was inside noise -> producer/consumer XCD affinity between the two dispatches is a separate, empty axis here.
Once the swizzle lands the residual inverts: the only non-power-of-two context became the case with the least remaining headroom (1.05x) while the aliased contexts caught up -> re-rank the per-case table before planning a follow-up aliasing round.
- caution: Also verify the remaining swizzle permutations per case rather than by geomean: a further rotation traded +0.7..+2.3% at short context for -1.3..-2.3% at long context for a net 0.997x, so the family can look alive in aggregate while no member wins everywhere.
- source: run mi355x_vllm_hip_paged_attention_decode-bmk7-12h, 2026-08-11..17, gfx950/MI355X, 13 rounds, direction r1_d0; director-validated geomean 1.365x, correctness 7/7
