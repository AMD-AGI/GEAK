---
key: ragged/mixed-batch paged attention in HIP C++ on gfx950/CDNA4, where the bf16 KV stream is read exactly once and has no L2 reuse
type: lever
confidence: ★★
effect: 1.066x weighted geomean isolated vs frozen baseline, non-overlapping, bit-gate correct; every case improved: +2.6% on the long-context prefill-shaped case up to +7.6% on the short decode-shaped cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both
description: Read-once paged KV on gfx950: the shipped 16-byte non-temporal helper drops the nt bit; one 128-bit vector builtin load restores it for ~1.07x, all cases up.
keywords: ['attention-decode', 'paged-kv', 'non-temporal-loads', 'cache-modifier', 'kv-cache', 'memory-bound', 'isa-inspection', 'gfx950']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention_decode
regime: both
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-07-29
levers: ['mem.non-temporal-load']
---
# Get the nt bit onto KV loads by loading one native 128-bit vector
- lever: Issue a single non-temporal builtin load on a native 128-bit vector type (a 4-wide int ext_vector) for each KV chunk so the compiler emits one 16-byte global load carrying the nt modifier.
- apply: Loads only, no math change, so oracle parity comes for free; the edit is at the KV fetch site, replacing the multi-scalar 16-byte non-temporal helper.
- verify: Disassemble and confirm the 16-byte global load actually carries the nt modifier, then run the interleaved per-case A/B against the frozen baseline and require every case up, not just the geomean.
- pitfall: A helper whose name says non-temporal produced nothing -> on this arch it re-vectorizes four scalar nt loads and the nt modifier is dropped in the process -> read the emitted ISA rather than the source before concluding the hint does not pay.
- caution: Also verify the KV stream is genuinely read-once with no L2 reuse: on a sibling paged decode kernel the same nt bit flipped sign with streamed-set size (about +7% at one concurrency, about -7% at another), so uniform favorability here is a property of zero reuse.
- source: run paged_attention_ragged-ch16h, 2026-07-29, 16h time-scaling campaign, direction r1_d0 (non-temporal KV loads)
