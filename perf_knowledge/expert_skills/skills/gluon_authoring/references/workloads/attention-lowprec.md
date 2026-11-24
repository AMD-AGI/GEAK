# Workload: low-precision attention (fp8 / fp4 scaled-MFMA)

For attention-shaped kernels whose matmuls run fp8 / fp4. `../tile-programming/low-precision.md`
covers the scale side path for GEMM (pure MFMA -> MFMA chain); `attention.md` +
`../tile-programming/llir-codesign.md` cover the softmax VALU budget for bf16. Here the scale path
and the softmax compete for the same issue slots, which neither page prices.

**Arch scope: gfx950 (CDNA4).** Scaled MFMA (`mfma_scaled`, the f8f6f4 shapes, E8M0 scales) is
CDNA4-only (CDNA3 ISA has no scaled MFMA). **gfx942 downgrade** (`index.md` list): fp8 runs
`cdna3.mfma` on FNUZ operands and a descale that is not folded into an operand upstream runs as
explicit VALU on the accumulator (row 3 of the table in section 2: pure softmax-competing demand); the
scaled-MFMA cause of the shared-read-bus hazard of section 3 is absent there.

## Start from what does transfer

- In Gluon on Triton 3.8.0 / gfx950, fp4 and the K=64/128 f8f6f4 shapes are reached through
  `mfma_scaled` (compile-checked: `gl.amd.cdna4.mfma` with fp8 operands at `[32,32,64]` / `[16,16,128]`
  fails lowering, `[32,32,16]` lowers; plain `tl.dot` reaches the unscaled `32x32x64` form)
  (`../gluon/matrix-reference.md ## Matrix-Family Details`); the scale/operand layout is the first
  design decision there (`../tile-programming/low-precision.md`).
- **Read the oracle type and threshold before building the pipeline:** estimate the fp8 error under
  the actual gate on the bf16 anchor. A tight cosine/angular threshold can reject an fp8 matmul that a
  loose abs/rel gate accepts, and a tight elementwise gate can reject one a cosine gate accepts (at
  common elementwise tolerances abs/rel was the stricter gate for an MXFP8 GEMM on gfx950 —
  `../tile-programming/low-precision.md`).
- **The structure is precision-independent:** matmuls, reduction axes, masks and split boundaries
  are settled on the bf16 anchor (`attention.md`).

## What does NOT transfer, and why

### 1. The budget's own constants move with the dtype

Interval moves with operand format (`../hardware/isa-mechanisms.md ## Matrix/VALU co-execution`); the
window of the scaled shapes is now measured there (gfx950: the scaled variant's head is one pass longer
than the unscaled shape's, ATT) — **not** interval minus the bf16 read phase. Calibration (`../hardware/isa-mechanisms.md ### Calibrating the window for a new shape`)
gives it; the bf16 balance does not transfer.

### 2. The scale side path is a third consumer of the same slots

| the scale path's work | class | where it lands |
| --- | --- | --- |
| global read of the scales | memory | memory region, with the other loads |
| staging store / transposed read to reach the scale layout | memory (shared) | memory region — shares the port with the operand reads, not free |
| per-block descale arithmetic outside the matrix op | VALU | a compute region: **demand**, competing with the softmax |

**A descale folded into the matrix op's scale operand costs nothing in the budget; the same descale
as explicit VALU is softmax-equivalent demand.** Check the lowered code — the source does not settle
it. This holds for descales expressible as the matrix op's E8M0 scale operand (power of two, per 32
elements; measured on a gfx950 / 3.8.0 fp8 `mfma_scaled` loop). An arbitrary fp32 per-block descale
cannot be folded that way: hoist it onto the operand loaded once upstream or into a once-per-output
epilogue rather than the per-output tile.

### 3. The wave-structure guidance actively conflicts — and attention sits on the conflict

Scaled MFMA adds a hidden scale-load window and paired SIMDs share LDS read bandwidth (ATT-measured on
gfx950), so two in-phase waves can go shared-read bound where one wave is not
(`../tile-programming/scheduling-model.md ### inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)`);
attention's softmax placement needs exactly two waves
(`attention.md ### Where the softmax goes, and why it rides with the matrix op`). **The choice is a
measurement:** on both the two-wave and the one-wave candidate, matrix busy (`MfmaUtil`) and the
shared-read stall with the bank-conflict counter at zero. The signal reproduced on a gfx950 / 3.8.0
fp8 `mfma_scaled` LDS-fed loop (two waves raised the shared-read wait at zero conflicts), yet there the
two-wave candidate was still faster — so a shared-read-bound reading suggests timing the one-wave /
hand-interleaved candidate, not dropping the two-wave one. Keep the scales register-resident. An AGPR-forcing rung cannot buy registers on an 8-warp
ping-pong: the CTA needs 2 waves/SIMD, so `N >= 256` fails at launch and a smaller N only re-splits the 256
budget (`../tile-programming/llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`).

### 4. fp8 KV with bf16 matmuls: dequantize on the LDS read (native operands not required)

A latent / MLA KV cache stored as fp8 with per-(row, column-group) float scales (DSA sparse MLA)
is a different kernel from the scaled-MFMA case above: the scale varies per key inside an operand,
and `v_cvt_scalef32_pk_bf16_fp8` takes only the exponent of its scale (gfx950 ISA pseudo-code:
`scale = exponent(S1.f32)`), so a non-power-of-two scale cannot ride the convert or an E8M0 operand.
What paid on Triton 3.8 / gfx950 (sparse-MLA prefill, two independent kernels): stage the **raw fp8
row** in LDS (async copy), convert to bf16 on each operand read, run bf16 MFMA, and fold the group
scales into the scores and into P. Scaled-fp8 QK on the raw K was a tie; fp8 PV through a hi/lo
split of P lost there (the split costs the VALU the removed converts saved). For this case the
"operands native" acceptance line below does not apply — accept bf16 dots fed from an fp8 LDS tile.
Levers: `attention.md ### Sparse MLA (DSA): measured levers`.

## Reading order

Oracle + scale layout (`../tile-programming/low-precision.md`) -> structure on a bf16 anchor
(`attention.md`) -> emitted MFMA shape/formats, interval, window (`../hardware/isa-mechanisms.md`) ->
place scale work by class and compute the budget incl. VALU descale
(`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`) -> measure section 3.

**The bf16 anchor and the scaled candidate run different matrix instructions.** `MfmaUtil` / `SQ_VALU_MFMA_BUSY_CYCLES` count
both the bf16 and the scaled MFMA on gfx950, but the op counters split by format (fp8 x fp4 ->
`SQ_INSTS_VALU_MFMA_MOPS_F8`, fp4 x fp4 -> `SQ_INSTS_VALU_MFMA_MOPS_F6F4`): compare anchor and
candidate on busy cycles, op counts only within one format
(`../hardware/optimization-gotchas.md`, row 2).

## Acceptance

- Numerics under the **task's own** oracle type, not a generic tolerance.
- Scaled matrix ops present in the lowered code with the correct format/scale encoding, operands
  native (no upcast back to f16 in the hot loop). Exception: per-group non-power-of-two KV scales
  (section 4), where bf16 dots on an fp8 LDS tile are the accepted end state.
- The two wave-structure readings of section 3, recorded separately.
- Anchor vs candidate matrix engagement compared on busy cycles, not across op counters.
- Timing A/B as the arbiter, with the budget and the lowered code agreeing about *why* it moved.
