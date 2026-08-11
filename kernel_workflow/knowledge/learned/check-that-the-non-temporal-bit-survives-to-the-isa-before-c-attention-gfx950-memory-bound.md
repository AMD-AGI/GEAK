---
name: check-that-the-non-temporal-bit-survives-to-the-isa-before-c-attention-gfx950-memory-bound
description: Grade a non-temporal load helper at the ISA level before crediting it: +6.6% geomean on a read-once KV stream, all 9 attention cases up
keywords: [cache-policy, streaming-operand, isa-check, memory-bound, attention, decode, l2-locality]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: non-temporal / cache-bypassing loads on a read-once KV stream in a memory-bound attention decode kernel on gfx950, graded from the disassembly rather than the source
lifecycle: active
type: lever
confidence: ★★
effect: The whole 15.76h campaign banked exactly one win, and this was it: 1.06636 cumulative geomean (49 independent re-validation passes re-measure it at 1.054-1.066, median ~1.061), all 9 cases up with no regression (+2.6-7.6%), per-case 1.0647 / 1.0739 / 1.0684 / 1.0758 / 1.0710 / 1.0731 / 1.0705 / 1.0745 on the eight small signature cases and only 1.0263 on the large case, roofline-emp 0.400 -> 0.460. The mechanism was an inspection finding, not a tuning one, and the same bit on a sibling decode kernel FLIPPED sign by streamed-set size (+7% at one context length, -7% at another); here the KV stream is read-once with zero L2 reuse, so it was uniformly favourable and needed no per-shape gating.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign, 2026-08-11
last_seen: 2026-08-11
---
# Check that the non-temporal bit survives to the ISA before crediting a streaming-load helper
- lever: On a kernel whose large operand stream is read once, cache-bypassing loads are worth a round, but grade the helper you use at the ISA level rather than at the source level: a vendor or in-tree 'nontemporal' wrapper can lower to ordinary loads and read as a null result for the whole idea. Getting the load width and the modifier in one instruction is the part that pays.
- apply: Replace the helper with a single __builtin_nontemporal_load on a native 128-bit vector type so one instruction carries both the width and the cache policy.
- verify: Disassemble and confirm the nt / cache-policy modifier is actually present on the widest load instruction (global_load_dwordx4 ... nt), not merely present in the source.
- pitfall: the whole direction read as a null result -> the shipped 16-byte non-temporal helper is a documented gfx950 no-op that re-vectorises four scalar nt loads and DROPS the nt bit -> emit the wide load yourself and re-check the ISA.
- caution: Decide the sign from L2 reuse first — also verify across context lengths on any operand that other blocks re-read, since such a stream can gain at one working-set size and lose at another; a read-once stream tends to gain everywhere.
- source: chuschen 16h time-budget campaign, 2026-08-11
