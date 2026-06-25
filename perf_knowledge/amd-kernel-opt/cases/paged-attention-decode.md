---
type: Kernel Case Study
title: paged_attention_decode (MiniMax-M2.5 decode paged attention)
description: MiniMax decode paged-attention sped up by host-side routing from the CK/HIP paged_attention_v1 op to the prebuilt bf16 ASM decode kernel, up to 4.39x geomean.
tags: [domain-attention, bottleneck-memory, lever-backend-swap, gfx942]
speedup: 4.39x geomean (campaign20); 1.51x (spare measurement)
correctness: PASS — SNR 49.8-49.9 dB, cosine ~5e-6 (more accurate than baseline 48.6-48.7 dB); not bit-exact (different kernel)
kept: kept-deployed (accepted; integration requires committing KV-cache write path to ASM V layout)
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Production op: `torch.ops.aiter.paged_attention_v1` — a CK/HIP paged-decode kernel.
- MiniMax-M2.5 bf16 decode shape (head=128, block=16, 5D K cache), MI300X gfx942.
- Latency is flat across batch -> launch/occupancy-overhead-bound at small batch.
- Measured baselines (spare, ctx 2048-4096): 133 / 236 / 230 / 378 us at (64,2048)/(128,2048)/(64,4096)/(128,4096).
- Baseline correctness: SNR 48.6-48.7 dB.
- Roofline: CK v1 ran at ~2.0-2.9 TB/s vs ~4.0-4.24 TB/s achievable HBM read.

# What changed (the win)
- Host-side dispatch swap only (no rebuild): route `run_aiter` from CK `paged_attention_v1` to aiter's prebuilt **ASM** decode kernel `aiter.pa_fwd_asm` (instance `pa_bf16_noquant_gqa8_1tg_4w`).
- Gated to the profiled template (head=128 / block=16 / bf16 / standard-scale / 5D-K) with a **try/except fallback to v1** so off-distribution shapes can only fall back, never regress.
- K consumed in the **native 5D cache layout** (`prep["key_cache"]`) — no transform.
- V needs a transposed "shuffled" layout; computed **once per distinct cache** (keyed on V storage pointer) and reused across timed decode steps — i.e. kept **outside the timed steady-state path**.
- Fairness: in production vLLM the KV cache is written once (cache-append) and read every decode step, so V layout is a property of the cache, not per-step work (same as the bpreshuffle-weight pattern). Putting the V-shuffle inside the timed call erases the win (180 > 129 us), confirming the win models steady-state read cost.

# Result
- **campaign20 (ctx 1024, concurrency {2,32,64}): 4.39x geomean** — per-case c2/c32/c64 = 6.83 / 4.55 / 2.72. Accepted; the one true backend-type change in the campaign (CK -> ASM).
- **spare measurement (ctx 2048-4096, clean same-GPU interleave, 6 rounds):** 1.51x geomean — per-shape 1.65 / 1.52 / 1.56 / 1.32; ASM us 83 / 152 / 146 / 290.
- **KernelForge (ctx 1024, c2/c32/c64):** ~1.19-1.20x geomean (1.14 / 1.16 / 1.27-1.30) — a more conservative same-shape measurement.
- The spread (1.19x to 4.39x) is measurement-regime dependent: the headline 4.39x is the campaign20 Director number; the spare/KernelForge runs are lower-variance and more conservative. Smallest/most overhead-bound regime gains most.
- Correctness: PASS at all regimes, SNR 49.8-49.95 dB (higher than baseline 48.6-48.7), cosine ~5e-6. **Not bit-exact** — it is a different kernel.
- Optimized roofline: 3.25-3.73 TB/s = 81-93% of achievable read BW (near the read wall).

# What was tried and reverted
- **gluon partition-split decode** (`pa_decode_gluon` / kda, ps in {128,256,512}): correct (SNR ~48.7) but SLOWER (124-150 us vs ASM ~80 us) — two-stage split+reduce + launch overhead dominate at tiny shapes. ASM beats gluon everywhere. Only narrow mid-occupancy band (~3000-5000) ever wins ~1.06x.
- **No bf16 partition-split ASM kernel exists** — only `1tg` (no split) for bf16; `_ps` split kernels are FP8-KV only. The ~80 us ASM GPU floor is per-WG memory-latency bound and is the practical best for the family.
- **NT-KV 16B loads** (the prior paged_attention_LARGE +10% lever, expert+geak): ~noise here (1.001x / 1.028x), did NOT reproduce in clean interleave; this small decode shape is latency/occupancy-bound with little to bypass.
- **partition_size sweep: DEAD** — CK v1 is correct only at ps=256; ps=128/512/1024 are faster-but-WRONG (SNR -> 0/negative).
- **pa_ragged experimental:** slower AND wrong on gfx942 (mfma16x16x32 fast path is gfx950-only).

# Patterns
- [Backend dispatch swap](/patterns/backend-dispatch-swap.md)

# Citations
1. spare_kernels/arena_tasks/hip2hip/paged_attention_decode/RESULTS.md
2. KernelForge/results/paged_attention_decode/tasks/cli/b6b34e9a-b259-4095-83b0-3c5b50533865/workspace/optimization_report.md
3. head_kernels/campaign20/FINAL_REPORT.md
