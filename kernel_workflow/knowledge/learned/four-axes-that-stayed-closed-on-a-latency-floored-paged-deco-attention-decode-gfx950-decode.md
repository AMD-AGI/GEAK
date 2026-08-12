---
key: closed KV-load and host-dispatch axes on a latency-floored paged grouped-attention decode kernel in Triton on gfx950
type: anti-pattern
confidence: ★★
effect: four decode directions returned ~1.00x or worse against the frozen baseline: graph capture/replay left the tiny-grid case tied and regressed the mid/large cases to 0.82x/0.79x; a .cg cache modifier cost +12-20% on the large case; a manual double-buffer prefetch was ~3% worse there; loop-split ~4-5% worse — while the same run banked ~1.23x from launcher metadata
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode
description: On latency-floored paged attention decode at ~1 WG/CU, cache-modifier, sw-prefetch, loop-split and graph replay all measured <=1.00x.
keywords: ['anti-pattern', 'cache-modifier', 'software-prefetch', 'graph-replay', 'launch-overhead', 'attention-decode', 'paged-kv', 'latency-bound', 'gfx950']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
verified_on: 2026-08-12
---
# Four axes that stayed closed on a latency-floored paged decode kernel
- lever: When the profile says latency-floored at ~1 workgroup/CU and the ISA already shows 128-bit vector loads, the KV-load region and host dispatch are the axes most likely to return ~1.00x; spending the round on launcher metadata and occupancy paid instead.
- apply: Cheap triage before opening the load region: read the generated ISA for load width, count workgroups per CU, and check whether the pipeliner already overlaps the address chain — then a prefetch or cache-hint round is a measurement, not a plan.
- verify: Confirm closure the same way as a win: interleaved A/B against the frozen baseline, per case. A genuinely closed axis shows tied medians on every case, not a small positive on one.
- pitfall: a .cs cache modifier is a hard compile error on this Triton/arch pair -> the modifier is unsupported there -> sweep modifiers behind a try-compile instead of assuming the set
exp2 substitution in the logsumexp missed the 1e-2 max-relative parity gate -> rebasing error accumulates in the normalizer -> keep the reference exponential for that reduction
head-split and a block width unequal to the page size multiplied KV traffic and broke the scalar-page numeric gate -> more waves also spilled -> keep the block width tied to the page size
- caution: Also verify the structural ceiling before committing to a deep rewrite: a full body/softmax/split-KV restructure reached only ~1.37x/1.34x on the mid/large cases and did not beat the already-banked geomean.
- source: GEAK 16h per-kernel time-budget campaign, run _fwd_grouped_kernel_stage1-ch16h, 2026-08-12, 61 passes
