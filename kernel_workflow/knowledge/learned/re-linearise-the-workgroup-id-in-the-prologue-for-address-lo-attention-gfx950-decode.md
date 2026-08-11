---
name: re-linearise-the-workgroup-id-in-the-prologue-for-address-lo-attention-gfx950-decode
description: Re-linearise the workgroup id so consecutive workgroups walk one request's contiguous KV: +3.1% on attention decode, scaling with workgroup count
keywords: [pid-remap, l2-locality, xcd, address-locality, decode, isa-check, bijection, interleaved-ab]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: program-id re-decode for KV address locality in a Triton grouped/MLA decode attention stage-1 kernel (q_len=1) on gfx950
lifecycle: active
type: lever
confidence: ★★
effect: director-verified 1.86x geomean end state (1.82 / 2.11 / 1.67 at batch 2 / 32 / 64, decode q_len=1); this lever's own A/B was +3.1% and scales with workgroup count - +1.2% at batch=2 (8 workgroups, the remap is nearly a no-op), +4.1% at batch=32, +3.1% at batch=64 - and a halves-probe put 100% of the batch=32 gain and 78% of the batch=64 gain on the ORDERING alone, only ~0.8% on the XCD partitioning it was designed around
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm7.2.3 / triton3.6.0 / torch2.11.0
source: run kernel_20_geak_0808_4h 2026-08-08
last_seen: 2026-08-10
---
# Re-linearise the workgroup id in the prologue for address locality
- lever: Permuting which unit runs which block changes no accumulation order, so this class of lever clears an effectively bit-parity correctness gate by construction; re-decode the linear program id so that consecutive workgroups walk one request's contiguous KV region instead of an N-strided sample of the address space.
- apply: Rewrite the pid -> (batch, head-block, split) decode in the prologue with every divisor folded to a compile-time constant; the loop body, the summation order and the output bytes all stay identical, so only the CU a block lands on changes.
- verify: Diff the inner-loop ISA before and after - instruction, memory-op, barrier and register counts should be unchanged, so the entire delta is placement - and prove the mapping is a bijection in Python across several grid shapes including non-divisible tails before spending any GPU time.
- pitfall: a semantic no-op measured slower than stock in 14 of 14 reps, enough to hide a real win -> a runtime integer divide left in the prologue cost ~1% by itself -> fold every divisor to a compile-time constant, and A/B an identity mapping written in the SAME arithmetic style as the control.
- caution: Also note that refinements past the first ordering choice (granularity, reversed pairing, interleaved streams) all measured flat or negative here.
- source: run kernel_20_geak_0808_4h 2026-08-08
