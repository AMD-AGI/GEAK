---
key: reading per-case bandwidth on a repeated-launch quantize/cast harness on gfx950, where every case re-reads the same buffers and part of the traffic is cache-served
type: method
confidence: ★★
effect: Diagnostic, not a speedup by itself. Director-verified end state 4.16x geomean (3.40x on the small launch-bound case, 4.83x mid, 4.39x on the largest streaming case). The control is what separated real headroom from an artifact: the mid case's apparent bandwidth is partly cache-served, while the largest case sits at ~98% of achievable DRAM bandwidth at its non-resident footprint and only 0.9% off a zero-arithmetic probe of its own bytes. A 4.7% over-subscription of the last-level cache cost 0.06%, and the bypass-fraction sweep was a straight line with no interior optimum.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-10
name: price-cache-residency-with-a-footprint-control-before-believ-quantize-cast-gfx950-mixed
description: Run a footprint control before believing a per-case bandwidth number: it showed one quant case cache-served and the largest at ~98% of achievable DRAM BW
keywords: ['roofline', 'measurement-method', 'control-experiment', 'l2-locality', 'quant', 'memory-bound', 'counters', 'harness-artifact']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: mixed
lifecycle: active
---
# Price cache residency with a footprint control before believing a per-case bandwidth number
- lever: In a repeated-launch harness every case re-reads the same buffers, so a per-case bandwidth figure is a cache number, not a memory-system number. Before funding any 'this shape is N% below that shape's bandwidth' direction, or any cache-partitioning / thrash-mitigation direction, run a footprint control that varies the cyclic working set independently of the bytes moved per dispatch.
- apply: Allocate nsets distinct input/output buffer sets and dispatch round-robin over them, so bytes per launch depend only on the shape while the cyclic working set is nsets times larger; compare the rate at nsets=1 against a set size just under and just over the last-level cache. Quote every roofline percentage with the footprint it was measured at, and re-derive the achievable roof at that footprint rather than against the nameplate.
- verify: The control has bitten when the two shapes' rates converge (or the deficit vanishes) once both are non-resident; a shape whose number drops sharply as nsets grows was cache-served. Confirm any partitioning direction has room at all by first measuring the cost of a small deliberate over-subscription.
- caution: Also verify the residency story against the write side separately: a write set about the size of the last-level cache evicts itself, so an allocating store can lose to a non-allocating one at exactly the size where 'make it resident' sounds most attractive. Also verify a negative's scope before generalising it - a cache-hint result measured at one footprint here inverted past the cache.
- source: run kb_on_0810 2026-08-10
