---
title: allgather — numerics
kind: technique
operator: allgather
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
---

# allgather — numerics & parity

## Pure copy → parity-safe
All-gather performs **no reduction** — it concatenates shards. The output is **bitwise identical** across
RCCL / xGMI P2P / Iris backends and algorithms (ring/tree/1-shot), modulo the **layout/ordering** of the
gathered shards. There is no accumulation dtype, no rounding, no accuracy gate.

## The only real risk: layout / shard ordering
The single correctness concern is that each rank's shard lands at the **right offset** in the output. A
wrong rank→offset map (e.g. an off-by-one in the gather index, or a mismatched shard size on the last rank)
produces a structurally wrong tensor — caught by a simple structural test, not a tolerance.

## fp8 shards
If shards are fp8, all-gather just moves the bytes (no dequant) — still lossless **as a copy**. The fp8
dialect (fnuz on gfx942) only matters when the gathered tensor is later interpreted; the gather itself is
dialect-agnostic byte movement.

## Verification recipe
1. Structural: gather of `[r]*S` per rank must produce `[0,0,…,1,1,…,P-1,…]` at the right offsets.
2. e2e: any SP layout change → greedy parity (should be exact for the AG itself; any diff is a layout bug
   or a downstream reduction, not the gather).

## Sources
- RCCL all-gather semantics: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
