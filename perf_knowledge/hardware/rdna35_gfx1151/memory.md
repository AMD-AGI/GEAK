---
title: RDNA3.5 / gfx1151 (Strix Halo) — memory hierarchy behaviour
kind: hardware
gens: [gfx1151]
dtypes: [fp16, bf16, int8, int4]
regimes: [both]
updated: 2026-11-06
sources:
  - ../../../kernel_workflow/knowledge/amd_ryzen.md
  - repro/ceilings.py
  - repro/bw.py
  - https://github.com/ROCm/hip-ep/pull/675
---

# RDNA3.5 / gfx1151 — memory hierarchy behaviour

> The hierarchy (2 MiB L2, 32 MiB Infinity Cache, BIOS-sliced LPDDR5X shared with the CPU) is in
> [`amd_ryzen.md` §0](../../../kernel_workflow/knowledge/amd_ryzen.md); measured bandwidths are in
> [peak_tables.md](peak_tables.md). This card is how that hierarchy behaves.

## TL;DR
> The Infinity Cache serves reads at **~4× DRAM** ([peak_tables.md](peak_tables.md)), which matters
> only for bandwidth-bound work. Two effects decide a kernel's time here more than on a discrete GPU:
> dirty lines left in the Infinity Cache are written back on the next kernel's time, and the CPU
> shares the same DRAM.

## Concepts
- **The cache step only matters in a window.** With the compute roof at ~36 TFLOP/s, the Infinity
  Cache and DRAM roofs cross it at ~39 and ~155 FLOP/byte. Only between those intensities does
  "does it fit in 32 MiB" change a bound / not-bound verdict; a 3072³ fp16 GEMM (~1000 FLOP/byte) is
  compute-bound either way.
- **Dirty lines are paid for by the next kernel.** A kernel that leaves ~25 MB of written data in the
  Infinity Cache hands its write-back to whatever runs next. `[measured]` In a Qwen3.8-27B decode
  step, the GEMM after a GatedDeltaNet state update (~25 MB fp32) ran at 129 GB/s; counting the
  write-back bytes puts it at the DRAM roof (serving trace; [repro/README.md](repro/README.md)).
- **Achievable is ~90% of the pin rate**, for reads and mixed traffic alike (peak_tables). A real
  KV-streaming decode kernel measured 222 GB/s (ROCm/hip-ep PR #675).
- **The CPU is on the same memory.** A busy CPU takes DRAM bandwidth from the GPU.

## The levers
1. Fit the hot working set under 32 MiB — after checking that the kernel's intensity is inside the
   window above.
2. Keep write-once output out of the Infinity Cache when nothing re-reads it. `[measured]` part of an
   add+rmsnorm fusion (1.81×) whose streaming shapes moved from ~81% to 99–100% of a 230 GB/s DRAM
   roof (GEAK campaign; [repro/README.md](repro/README.md)).
3. Count bytes against the fused minimum. `[measured]` a 3-dispatch row kernel ran at 83–101% of the
   roof on the bytes it moved and 25–40% on the bytes it needed; one-pass fusion gave softmax 2.21×
   (GEAK campaign; [repro/README.md](repro/README.md)).

## Pitfalls
- Rooflining against 256 GB/s; use the measured values in [peak_tables.md](peak_tables.md).
- Evicting with writes before a timed call: the write-back lands in the timed kernel and
  under-reports it. Evict by reading a buffer larger than the Infinity Cache, as
  [`repro/ceilings.py`](repro/ceilings.py) does.
- Timing a cache-resident kernel one call per event pair. A 24 MB read takes ~27 µs; time a batch of
  calls between one pair of events, as `ceilings.py` does.
- Counting only the bytes a bandwidth kernel reads when it also writes; `ceilings.py` writes one
  int32 per program for that reason.
- Timing a decode GEMM warm. Consecutive layers never reuse a weight in serving; cold vs warm is
  3–4× on real decode weights (serving trace; [repro/README.md](repro/README.md)).
- Reading a cache bandwidth from [`repro/bw.py`](repro/bw.py). It is host-timed, so its LLC-sized
  cases measure launch overhead.

## Verify
- [`repro/ceilings.py`](repro/ceilings.py) `bw` — cold DRAM and LLC-resident read.
- [`repro/bw.py`](repro/bw.py) — read-read-write streaming.
- `rocprofv3` with `GL2C_HIT` / `GL2C_MISS` for L2 traffic, within the counter budget in
  [pitfalls.md](pitfalls.md).

## Sources
- [`repro/ceilings.py`](repro/ceilings.py), [`repro/bw.py`](repro/bw.py) — Radeon 8060S, 2026-11-06.
- Write-back, cold/warm and fusion figures: gfx1151 serving traces and GEAK kernel campaigns
  (2026-09 to 2026-11); no script, see [repro/README.md](repro/README.md).
- ROCm/hip-ep PR #675: https://github.com/ROCm/hip-ep/pull/675
