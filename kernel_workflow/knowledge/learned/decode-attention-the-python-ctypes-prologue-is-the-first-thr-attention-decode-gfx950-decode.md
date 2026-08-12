---
key: aiter-style JIT paged decode attention on gfx950 dispatched through a Python/ctypes wrapper, small-batch host-bound
type: lever
confidence: ★★
effect: host-only directions took the isolated cumulative from 1.00x to 3.15x vs the frozen baseline (non-overlapping, director re-verified over 3 runs); the gain is per-case lopsided: +26% on the 2-sequence case, fully hidden behind device time at 32 and 64 sequences
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode
description: Memoize the whole per-call Python/ctypes prologue of a JIT decode-attention wrapper: cumulative 1.00x->3.15x isolated, concentrated at small batch.
keywords: ['decode', 'paged-attention', 'launch-overhead', 'host-wrapper', 'small-batch', 'jit', 'memoization']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
---
# Decode attention: the Python/ctypes prologue is the first three rounds
- lever: - lever: when a decode attention op is dispatched through a JIT + ctypes wrapper, memoize the ENTIRE per-call prologue as one unit (resolved function handle, device props, derived ints, boxed argv) before touching the device.
- apply: - apply: module-scope caches keyed on shapes + partition size for the constant scale tensors and a pooled split-KV workspace; a pointer-identity launch memo; a per-case partition-size table. Each lever alone reads as noise; together they compound.
- verify: - verify: isolated A/B per case, plus a host-only timing that shows the prologue collapsing to a flat per-call constant; the win survives only where wall-clock is host-dominated, so expect it to vanish at the larger batch cases.
- pitfall: - pitfall: memo armed only on a cold cache -> it never engaged on the warm path the harness actually times -> arm it on the first call and every call after.
- pitfall: launch memo guarded on query-tensor identity alone -> stale results, parity collapsed from ~48 dB SNR to ~15 dB -> key the memo on every mutable pointer and shape it captures.
- caution: - caution: also verify a wrapper-level graph capture/replay arm before concluding the host floor is unreachable — here capture was correct and value-identical, but replay carried a flat per-call add that exceeded the eager path at every case measured.
- source: - source: kernel_workflow 16h campaign, run kernel_20_geak_0808_16h, 2026-08-12; director-validated geomean 3.98x, correctness pass
