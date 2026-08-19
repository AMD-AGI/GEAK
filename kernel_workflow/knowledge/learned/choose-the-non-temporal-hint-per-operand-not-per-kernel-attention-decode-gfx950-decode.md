---
key: picking the non-temporal cache hint separately for K, V, Q and the partial-output stores inside one HIP C++ paged decode attention on gfx950/CDNA4
type: lever
confidence: ★★
effect: +6.5% isolated vs frozen baseline from marking only the K fetch non-temporal, carried entirely by the long-context cases (+10.7% and +10.6..+14.5%) with the three short-context cases flat inside the noise band; the blanket K+V form is a consistent loss at every context and nt on Q is -1.1%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode
description: Non-temporal is a per-operand call in paged decode: nt on the K stream is +6.5%; nt on the re-touched V tile, on Q and on the output stores lose.
keywords: ['attention-decode', 'paged-attention', 'paged-kv', 'non-temporal-loads', 'cache-modifier', 'kv-cache', 'isa-inspection', 'l2-residency', 'decode', 'gfx950']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
levers: ['mem.non-temporal-load']
origin_kernels: ['mi355x_vllm_hip_paged_attention_decode']
---
# Choose the non-temporal hint per operand, not per kernel
- lever: Split the cache-policy decision per operand instead of per kernel: mark non-temporal only the stream that is genuinely single-touch, and A/B each of K, V, Q and the partial-output stores on its own.
- apply: One builtin non-temporal load on a native 128-bit vector type at the single-touch fetch site, leaving the other operands on the default path; loads only, so parity is free.
- verify: Diff the emitted ISA against a non-temporal-free control to confirm exactly one wide global load gained the hint, then interleaved per-case A/B; a policy edit that changes one bit and shows no ISA delta is the silent no-op failure mode.
- pitfall: The in-tree 16-byte non-temporal helper bought nothing -> it lowers to four scalar loads and quadruples the request count -> a single builtin load on a 4-wide vector emits one wide global load carrying the hint.
The blanket KV form captured only about a third of the available win -> the V tile is re-touched across the lane and head-element loops while K is not -> reuse distance, not operand family, decides the hint.
Every write-side non-temporal configuration lost, and a hint-free inline-asm control was +0.06% -> the partial outputs are consumed hot by the very next dispatch -> keep the store path cacheable.
- caution: Also verify whether the hint can reach the last-level cache at all before budgeting a residency round: 13 fractional-residency sweep points all lost, monotone in the mixing fraction in both directions, which is consistent with the hint being an L2 replacement policy that does not suppress LLC allocation.
- source: run mi355x_vllm_hip_paged_attention_decode-bmk7-12h, 2026-08-11..17, gfx950/MI355X, directions r2_d1 / r3_d1 / r5_d0; director-validated geomean 1.365x, correctness 7/7
