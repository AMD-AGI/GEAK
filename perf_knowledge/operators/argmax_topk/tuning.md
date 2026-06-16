---
title: argmax_topk — tuning (val,idx reduce, split, bitonic top-k)
kind: operator_overview
operator: argmax_topk
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://moderngpu.github.io/scan.html
  - https://triton-lang.org/main/python-api/generated/triton.language.max.html
---

# argmax_topk — tuning

This is a reduction that carries a `(value, index)` pair, so it reuses the reduction machinery
([`../reduction/tuning.md`](../reduction/tuning.md)) with two differences: the combine compares values and
selects the winning index (with a tie rule), and **top-k** needs a partial sort.

## 1. argmax = `(value, index)` wave/LDS reduce
```
thread: each lane tracks its best (val, idx)  →  wave64 shuffle reduce on the pair  →  LDS combine
```
- The shuffle combine compares values; on equal values apply the **tie rule** (left-most = smaller index).
  Both operands must be shuffled (`__shfl_down` the val *and* the idx) — 6 steps over 64 lanes.
- Pack `(val, idx)` into one 64-bit word for a single shuffle when possible (e.g. fp32 val in high bits,
  int32 idx in low, with the bit-twiddle that makes integer compare == float compare) — halves shuffle
  traffic. Otherwise shuffle two values.
- LDS holds `nwaves` pairs; wave 0 reduces them with the same tie rule.

## 2. Split for greedy decode (few rows × huge vocab)
`[batch=1..8, vocab=128k]` → one row, huge axis → split the **vocab axis** across blocks, each emits its
local `(best_val, best_idx)`, then a tiny second kernel combines the per-block winners (with the tie rule).
The two-call deterministic path is preferred here — argmax has **no fp atomic**, and the result feeds
sampling (parity matters). This is the argmax analogue of split reduction.

## 3. Top-k strategies (pick by k)
| k | strategy | cost |
|---|---|---|
| **k=1** | argmax (above) | one reduce |
| **small k (≤32)** | iterative argmax: find max, mask it, repeat k times | `k ×` reduce — fine for k≤~16 |
| **medium k (≤256)** | per-thread local top-k heap → LDS merge → bitonic/merge top-k | one pass, more regs/LDS |
| **large k / full sort** | bitonic sort the (masked) candidates | `O(n log²n)` |
- Iterative argmax is simplest and correct (reuses the argmax path) but `O(k·n)` — only for small k.
- For sampling top-k (k often 40–100) a **threshold + count** approach (find the k-th value via a
  selection, then gather everything ≥ it) avoids a full sort. Watch ties at the threshold (can return
  >k or <k items — match torch's tie behavior; see [numerics.md](numerics.md)).

## 4. AMD knobs
- **wave64 shuffle** 6 steps; **block 256**; LDS tiny (pairs per wave) → occupancy set by VGPRs.
- **vectorize the logits load** (128-bit) — the bandwidth-bound part is reading `[batch, vocab]`.
- `waves_per_eu=3/4` to hide the logits-load latency.
- **fp32 compare** for bf16/fp16 logits (stable ordering; bf16 has many ties → tie rule matters more).
- Contiguous, hole-free shuffle masks reduce faster.

## 5. Triton specifics
`tl.argmax(x, axis, tie_break_left=True)` and `tl.max(x, axis, return_indices=True,
return_indices_tie_break_left=True)` lower to the pair reduce. `BLOCK = next_pow2(axis)`; for huge vocab,
chunk + 2-call combine in the driver. Top-k: iterative `tl.argmax`+mask, or a hand-written bitonic in the
kernel. See [backends/triton.md](backends/triton.md).

## Sources
- (val,idx) wave reduce, 64-bit shuffle, tie selection: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- selection/top-k as partial sort, bitonic: https://moderngpu.github.io/scan.html
- `tl.max(return_indices, tie_break_left)`: https://triton-lang.org/main/python-api/generated/triton.language.max.html
