# Workload: Reduction / Elementwise / Vector

Memory-bound tile patterns; plain Triton usually covers them, and an explicit layout pays when a
concrete coalescing, bank-conflict or vector-width mechanism needs one.

## Usual bound class

Typically memory-bound; the open question is **which** byte metric applies:
- **Decode-shape footprints (tens of KiB) stay in L2 / the memory-side cache when warm**: "% of HBM
  peak" is then inapplicable — such a kernel sits at the launch / dispatch floor (measured on gfx950), and a
  larger warm footprint can beat a floor built from probed HBM bandwidth (a harness that flushes caches
  between iterations brings HBM back). `--footprint-mb` against the memory-side cache decides it
  (`../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`;
  GEMM version: `gemm.md ## Usual bound class`).
- Once it reaches DRAM: compare against an in-shape probed ceiling (`probe.py measure`).
- **A graph removes the CPU launch cost, not the per-dispatch cost:** on gfx950 (Triton 3.8.0) each
  captured dispatch still has a measurable cost (`floor2 - floor1`, `gemm-small-m.md`), and fusing two dependent elementwise
  launches paid under a graph as well as eager, at small and large sizes. Measured fusion losses under a
  graph came from the fused kernel itself getting slower (a fused split-K reduce, a fused scan), not from
  fusing — so A/B the fused kernel under the graph (`../hardware/roofline-models.md ## Kernel time
  decomposition`). When the op is launch-floor-bound (wall ≈ graph nodes × a flat per-node cost),
  deleting a node paid (learned card `kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md`).

Buffer-op rules: `../gluon/memory-reference.md`.

## Group-wise quantization (absmax -> scale -> cast)

Production side lives here; consumption of scales is `../tile-programming/low-precision.md`;
quantize-once vs fused-per-tile dispatch is `gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound`.
Invariant: **`BLOCK % GROUP == 0`** (mask the ragged edge up to whole groups).

**Stage 1 — group absmax.** Reshape to `[rows, BLOCK/GROUP, GROUP]`, reduce the last axis with
`keep_dims=True`; clamp absmax away from zero. **After the cast, force padding to zero in a separate
`where`** (block-scaled formats cannot represent "absent").
- **Portable absmax (replaces PTX `max.NaN.xorsign.abs.f32`):** bitcast fp32 -> `uint32`, `& 0x7FFFFFFF`,
  integer max — NaN sorts above +inf, so it propagates.
- A second data pass is a residency question: group in registers = free; group spanning a K loop =
  second read or two kernels.

**Stage 2 — scale.** Hardware group on CDNA4 is **32** (`moe.md ### Axis 2 — how the quantization reaches the`);
128 is a software convention, NVFP4 is 16. A group-128 scale cannot feed `mfma_scaled` directly:
expand to 4x32 (exact only for power-of-two scales) or rescale the fp32 accumulator once per 128 of K.
- **e8m0** (`uint8`): `(bits & 0x7F800000) >> 23`; add the mantissa mask first to round up.
- **fp8 e4m3 scale** (NVFP4): reported (upstream, not re-checked here) to be rounded to nearest by the
  cast even under a ROUND_UP flag, so values can exceed max-finite.
- Keep the saturating conversion; clip explicitly with delayed/global scales. gfx942: FNUZ targets, VALU scale.

The scale **layout** depends on the MFMA non-K dimension (host shuffle and in-kernel un-shuffle must
be exact inverses), so it is not portable across instruction shapes. A standalone quantized-GEMM
benchmark with synthesized scales (`torch.randint` e8m0 near unity, as the upstream tutorial does)
excludes the quantizer (`downcast_to_mxfp` is a separate `@triton.jit` kernel or a fused epilogue) —
scope the comparison to the same span.

## Stage 3 — dequant + residual + norm

Argue the fusion in bytes: each unfused step is one HBM round trip of the activation.
- Accumulate in fp32 (promote once at the top); **add the residual before the variance reduction**, in fp32.
- A deliberate `.to(bf16).to(fp32)` is a correctness contract: re-insert it at the old boundary when fusing.
- `enable_fp_fusion` (default on) is module-wide; flipping it only tests whether a mismatch is contraction.

## Rank-3 reshape -> unplanned convert_layout through LDS

On the explicit-layout path the 2D->3D group reshape's broadcast can add an LDS `convert_layout`; the
two ways around it are `BLOCK == GROUP` or an authored rank-3 layout, visible in the TTGIR. In plain
Triton the reshape form is the usual spelling.

## Family signals

- **Elementwise:** tiny vector stages often lose to layout + launch overhead; a wave64 `BlockedLayout`
  matching the inferred TTGIR layout with generic `gl.load`/`gl.store` is the neutral baseline.
- **Softmax / reduction / norm:** one reduction axis; explicit identity and accumulator dtype;
  derive reduction state from the parent layout it broadcasts into. Pipeline / slicing layers have
  no MFMA loop to act on in pure memory-bound kernels; an LDS layout matters when a reused staging
  buffer shows bank conflicts.
