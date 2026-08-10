---
key: attention · gfx950 · decode
type: lever
confidence: ★★
effect: director-verified 1.86x geomean end state (1.82 / 2.11 / 1.67 at batch 2 / 32 / 64, decode q_len=1); this lever's own A/B was +3.1% and scales with workgroup count — +1.2% at batch=2 (8 workgroups, the remap is nearly a no-op), +4.1% at batch=32, +3.1% at batch=64 — and a halves-probe put 100% of the batch=32 gain and 78% of the batch=64 gain on the ORDERING alone, only ~0.8% on the XCD partitioning it was designed around
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm7.2.3 / triton3.6.0 / torch2.11.0
last_seen: 2026-08-08
---
# Re-linearise the workgroup id in the prologue for address locality
- lever: Permuting which unit runs which block changes no accumulation order, so this class of lever clears an effectively bit-parity correctness gate by construction; re-decode the linear program id so that consecutive workgroups walk one request's contiguous KV region instead of an N-strided sample of the address space.
- apply: Rewrite the pid -> (batch, head-block, split) decode in the prologue with every divisor folded to a compile-time constant; the loop body, the summation order and the output bytes all stay identical, so only the CU a block lands on changes.
- verify: Diff the inner-loop ISA before and after — instruction, memory-op, barrier and register counts should be unchanged, so the entire delta is placement — and prove the mapping is a bijection in Python across several grid shapes including non-divisible tails before spending any GPU time.
- caution: Also verify an identity mapping written in the SAME arithmetic style: a runtime integer divide in the prologue cost ~1% here and made a semantic no-op measure slower than stock in 14 of 14 reps, which is enough to hide a real win; and note that refinements past the first ordering choice (granularity, reversed pairing, interleaved streams) all measured flat or negative here.
- source: run kernel_20_geak_0808_4h 2026-08-08
