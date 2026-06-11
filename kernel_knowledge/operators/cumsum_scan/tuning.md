---
title: cumsum_scan — tuning (Hillis-Steele vs Blelloch, wave-scan, chunking)
kind: operator_overview
operator: cumsum_scan
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://moderngpu.github.io/scan.html
  - https://srush.github.io/annotated-mamba/hard.html
  - https://github.com/proger/accelerated-scan
---

# cumsum_scan — tuning

A scan has a **log-depth dependency chain**, so unlike a flat elementwise/reduce it can't be pure
bandwidth — the levers are the **algorithm** (work vs depth) and the **memory hierarchy** the carries
travel through (lanes → LDS → global).

## 1. The two parallel-scan algorithms
| algorithm | work | depth | LDS access | use |
|---|---|---|---|---|
| **Hillis-Steele** (inclusive, "scan-then-propagate") | `O(N log N)` | `log N` (shallow) | every step reads/writes | small N (one wave/block), simplest |
| **Blelloch** (work-efficient, up-sweep + down-sweep) | `O(N)` | `2 log N` | two passes over the tree | large N, work-bound, the block/grid scan |

Rule of thumb: **Hillis-Steele within a wave/small block** (depth wins, work is tiny), **Blelloch for the
block/grid level** (work-efficiency wins when N is large). `accelerated-scan` extends Blelloch from the
warp level (shuffles) up to the block level (LDS).

## 2. The hardware hierarchy (AMD)
```
intra-wave scan (cross-lane __shfl_up, 64 lanes)  →  inter-wave carry (LDS)  →  inter-block carry (global, 2nd pass)
```
- **Wave scan**: `__shfl_up(v, off)` for `off = 1,2,4,...,32` (6 steps, **64 lanes**) gives a 64-element
  inclusive scan in registers — the fastest level (warp shuffles, no LDS). This is the inner primitive.
- **Block scan**: each wave scans its 64 lanes; the last lane's total goes to LDS; wave 0 scans those
  block-partials; each wave adds its exclusive carry. `block=256` → 4 waves → tiny LDS.
- **Grid scan (long axis)**: see chunking below.

## 3. Chunking a long axis (the 3-stage stitch)
A single block can't hold an arbitrarily long sequence (LDS/reg limit). Split the axis into blocks
(one program id each) and stitch:
1. **block-scan** each chunk locally + emit each chunk's total.
2. **scan the chunk-totals** (exclusive) → per-chunk carry-in.
3. **block-scan again** (or just add the carry) initialized with the carry-in.
The same kernel runs in stages 1 and 3 with a carry-in argument (Annotated-Mamba pattern). This is the
scan analogue of split reduction.

## 4. AMD-specific knobs
- **wave64 shuffle**: `__shfl_up` 6 steps (not 5) — a 32-lane scan loses half the wave.
- **block 256** (4 waves), grid sized so chunks cover the axis × rows; aim ≥1024 programs across rows for
  short-axis many-row cases (MoE: one program per row, axis tiny → Hillis-Steele in LDS, no chunking).
- **fp32 accumulate**; vectorize the **load and store** (scan reads *and* writes the full tensor, so
  128-bit both ways).
- **LDS for the carries** — tiny (one slot per wave); the bandwidth-bound part is still the global
  load/store, so keep those 128-bit and coalesced.

## 5. Recurrence as a pair-scan (SSM/gated-delta)
To scan `x[t] = a[t]·x[t-1] + b[t]`, scan **pairs** `(a, b)` with combine
`(a₁,b₁)⊕(a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)` (associative, **not commutative**). A common trick: pack the two
fp32 values into one int64 via bitcast so a single scan tensor carries both, then unpack. ⚠ The
non-commutativity hits Triton's `associative_scan` at seq ≥ 128 — see [numerics.md](numerics.md).

## 6. Triton specifics
`tl.cumsum(x, axis)` / `tl.associative_scan(x, axis, combine_fn)` lower to the wave-scan + LDS
automatically (the algorithm is the compiler's choice). Knobs: `BLOCK = next_pow2(chunk)`, `num_warps`
(how many waves combine), `num_stages=1`. For long axes, do the 3-stage stitch in the driver. See
[backends/triton.md](backends/triton.md).

## Sources
- Hillis-Steele vs Blelloch (work/depth, up-sweep/down-sweep), tree depth: https://moderngpu.github.io/scan.html
- chunked block-scan + carry stitch, wave/LDS hierarchy, pair-scan recurrence: https://srush.github.io/annotated-mamba/hard.html
- warp-shuffle → LDS hierarchical scan (Blelloch warp→block), first-order recurrence: https://github.com/proger/accelerated-scan
