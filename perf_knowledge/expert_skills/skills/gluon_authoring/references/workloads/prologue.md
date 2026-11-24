# Prologue kernels — the per-token body with no hot loop

QK-norm, RoPE, paged KV-cache write, per-token quant, and their fusions: one token per program,
**no reduction loop to pipeline**. `hw_budget.py` has no byte/FLOP model for this archetype — declare
traffic with `--tensors`; FLOPs are near zero, so the byte side carries the bound.

**What likely acts** (structural reasoning — there are no loop iterations to overlap; a suggested
starting point, not a measured ranking): memory path (coalescing, `buffer_load`, per-lane width), VGPR budget / occupancy, and
the grid (rank, workgroup count, tokens or heads per program). **Little to act on:** pipelining /
prefetch / scheduling (no loop iterations, unless a variant handles several tokens per program); LDS
layout unless the variant stages through LDS. Record "pipeline inert" as a result.

Async direct-to-LDS rarely applies (per-lane width rule:
`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`).

## Vector width is arithmetic, not a knob

`bytes_per_lane = row_elements / 64 * dtype_bytes` (wave64); a larger `size_per_thread` only replicates.
Wider per-lane access comes from more rows per program (a grid change); fp8 halves `bytes_per_lane`.

## Rotary pairing has three layouts

Read the pairing convention off the reference implementation:

| Convention | Partner of `r` | Cost |
| --- | --- | --- |
| Interleaved | `r ^ 1` | in-register if each lane owns both halves: `reshape [...,2]` / `split` / `join` |
| Split-half / NeoX | `(r + width/2) % width` | lane-local only if the halves land in one lane; else a static-stride cross-lane exchange |
| Runtime pairing | data-dependent | a real `gl.gather` |

`gl.gather` silently falls back to
LDS sized by the whole source tensor:
`../gluon/layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`.

## The paged index chain

The block-table load is a dependency of the payload address, so it serializes within a token. Overlap
comes from other tokens: more waves in flight, or issuing token *t+1*'s index load before consuming
*t*'s payload; the explicit prefetch is worth building when the profile shows the index load exposed.

## The quant epilogue's scale layout is a contract

The scale tensor's layout is fixed by the downstream matrix consumer, not chosen locally. Derive it
with `gl.amd.cdna4.get_mfma_scale_layout` from the consumer's dot-operand layout
(`../gluon/layout-reference.md ## The Scale Operand's Layout For mfma_scaled`); hardware group = 32
(`get_mfma_scale_layout` asserts it on 3.8.0) — details in `moe.md ### Axis 2 — how the quantization reaches the`. gfx942: VALU
scale, FNUZ payload. Suggested: write the contract next to the kernel, and check it with real (non-uniform)
scales — uniform scales hide a layout mismatch.

## A 2D grid with a small axis 0: grid shape, not num_warps=1

Observed upstream on gfx950: one small `num_warps=1` kernel on a multi-block 2-D grid ran much slower than
the same kernel on a flattened 1-D grid (`../hardware/capability-matrix.md`). **Measured on gfx950 / Triton
3.8.0, the cause is the grid's shape:** the slowdown appeared when axis 0 was small and axis 1 large, and
went away with the large extent on axis 0 or with a flattened 1-D grid. With the large extent on axis 0,
1-D, 2-D and 3-D grids tied. It showed whenever the programs were tiny enough that the launch was
dispatch-rate bound — at `num_warps` 1, 2 and 4 alike; `num_warps=1` is where such programs are typical,
not the cause. On elementwise kernels with larger programs, grid rank moved time only slightly at any warp
count, following the program-to-tile traversal order. If a tiny-workgroup kernel is slow, put the large
extent on axis 0 (or linearize — a zero-resource probe), then A/B `num_warps` separately. A 1-D `grid=(1,)`
single-block kernel at `num_warps=1` is not this case.

**Before quoting a % of HBM peak:** the compute floor is ~0, so a `memory` verdict is a default, not
a finding; a per-token footprint often fits in LLC/MALL, so HBM may be the wrong denominator
(`../hardware/roofline-models.md`).
