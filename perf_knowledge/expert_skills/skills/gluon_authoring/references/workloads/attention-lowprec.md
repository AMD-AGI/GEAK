# Workload: low-precision attention (fp8 / fp4 scaled-MFMA)

Read this when an attention-shaped kernel — two matrix phases with vector work between them —
runs its matmuls in fp8 or fp4 through scaled MFMA. It is the intersection of two pages that
were written independently and do **not** compose cleanly:

- `../tile-programming/low-precision.md` is written for **GEMM**: a scale side path bolted onto a pure
  MFMA -> MFMA accumulator chain, where the only question is how to feed the scales cheaply.
- `attention.md` + `../tile-programming/llir-codesign.md` are written for **dense f16/bf16
  attention**, where the third instruction category is softmax VALU and the budget that decides
  everything is calibrated on one matrix shape.

A low-precision attention kernel has **both** a scale side path and a VALU-between-matmul stage,
and they compete for the same issue slots. Neither page prices that competition. This one states
what is known, what is not, and how to find out — it deliberately does not invent numbers.

**Arch scope: this page is gfx950 (CDNA4, MI350X / MI355X).** Scaled MFMA (`mfma_scaled`, the
f8f6f4 shapes, the E8M0 scale operand) exists only there. **gfx942 downgrade** (CDNA3, MI300X /
MI325X): there is no scaled matrix op and no fp4 matrix path; fp8 attention runs the regular
`cdna3.mfma` fp8 instruction on **FNUZ** operands (`e4m3fnuz` saturates at 240, not 448) and every
descale is explicit VALU on the accumulator. That is the third row of the table in section 2 below
for *all* of the scale work — pure softmax-competing demand — and the shared-read-bus hazard of
section 3 (a scaled-path effect) does not arise, so the two-wave structure of `attention.md` stands
without that measurement.

## Start from what does transfer

These carry over unchanged, and they are most of the work:

- **Scaled MFMA is mandatory, not an optimization.** Where there is no regular fp8/fp4 matrix
  intrinsic, the scaled form is the only matrix path. Get the scale/operand **layout** right
  before anything else — a layout mismatch is the top "compiles but wrong numbers" cause, and it
  is generated, not guessed (`../tile-programming/low-precision.md`).
- **The correctness gate can veto the whole direction.** Read the *type* of the oracle first: an
  angular/cosine gate can reject an fp8 matmul that passes an abs/rel gate, because reducing the
  matmul shifts the output direction more than its magnitude. Decide this **before** building the
  pipeline, not after (`../tile-programming/low-precision.md`).
- **The structural attention decisions are precision-independent.** Which matmuls exist, which
  reductions run on which axis, where the masks and split boundaries are — settle those on the
  f16 anchor and keep them (`attention.md`).

## What does NOT transfer, and why

### 1. The budget's own constants move with the dtype

The co-execution budget is `capacity = (MFMA in region) x window` against `demand = sum of class
issue costs`. **Both sides move when the matmul goes low-precision**, and not by the same factor:

- The **interval** of a scaled matrix op is operand-format dependent — all-f4 runs at the short
  interval and anything involving f8 at double it — so the same tile shape is two different
  budgets depending on the formats (`../hardware/isa-mechanisms.md ## Matrix/VALU co-execution`).
- The **window** for every scaled shape is currently `unknown-needs-probe`. It is **not** the
  interval minus the calibrated shape's read phase; that subtract-a-constant shortcut is already
  known false across shapes on this family.
- A dtype change usually also changes the tile shape and `BLOCK_K`, which changes how many matrix
  ops a region contains — i.e. `capacity` again, through a second independent route.

**So:** do not carry an f16 balance ratio into an fp8 kernel. Calibrate the window for the shape
you actually emit (`../hardware/isa-mechanisms.md ### Calibrating the window for a new shape`),
or record a scoped ceiling for budget-based work on that shape and use only the levers that
reduce demand without needing the arithmetic.

### 2. The scale side path is a third consumer of the same slots

In GEMM the scale path is judged on whether it feeds the matrix op in time. In attention it also
**competes with the softmax** for the shadow, and the page that describes it does not say which
region it lands in. Resolve this per kernel, because placement is the decision:

| the scale path's work | class | where it should land |
| --- | --- | --- |
| global read of the scales | memory | the memory region, with the other loads |
| staging store / transposed read to reach the scale layout | memory (shared) | the memory region — it shares an issue port with the operand reads, so it is *not* free there either |
| any per-block descale arithmetic applied outside the matrix op | VALU | a compute region — and it is then **demand**, competing with the softmax |

The trap is the third row. A descale that is folded into the matrix op's scale operand costs
nothing in the budget; the same descale applied as explicit vector math is softmax-equivalent
work and has to be counted. Check which one you actually emitted by reading the lowered code —
the source form does not settle it.

Apply the general cheapest-carrier rule here as the first lever: push the scale onto the operand
that is loaded once and upstream, rather than onto the per-output-tile result.

### 3. The wave-structure guidance actively conflicts — and attention sits on the conflict

This is the open problem, and it should be read as one rather than resolved by picking whichever
page you read last.

- `../tile-programming/scheduling-model.md` records a **scaled-MFMA vs shared-read-bus hazard**:
  on the scaled path the hardware decouples each matrix op into a hidden scale-load window plus
  the compute window, and paired SIMDs share one shared-read issue bus — so two in-phase waves
  can collide there and the scaled path can go shared-read-throughput-bound where the unscaled
  path does not. Its conclusion: the two-wave candidate is *not* automatically better for low
  precision; a single resident wave is structurally immune.
- `attention.md` needs **exactly that two-wave structure**: the whole reason the softmax rides
  with the matrix ops is that memory must issue from the *other* wave to pair with it. Drop to
  one wave and the co-execution premise is gone.

**These cannot both be satisfied by a general rule, so measure rather than choose.** The two
readings that separate them are cheap and they are *different* counters:

1. scaled-matrix issue efficiency, and
2. shared-read utilization,

taken **separately** on the two-wave and one-wave candidates. If the shared-read bus is the bound
class, the hazard is real for your shape and the one-wave / hand-interleaved candidate wins
despite losing the pairing. If it is not, the attention structure stands. Record which one you
measured — a low-precision attention kernel closed without this pair of readings has an
unexamined assumption in it either way.

Corollary worth stating: **do not stack the two-wave ping-pong with an AGPR-forcing rung by
default.** They solve different cadence problems, and the added register pressure can drop the
resident wave count from two to one — which destroys the ping-pong you were relying on.

## Reading order

1. Settle the oracle type and the scale/operand layout (`../tile-programming/low-precision.md`).
2. Settle the attention structure on an f16 anchor (`attention.md`).
3. Establish the matrix shape and operand formats the loop actually emits, then get the interval
   and calibrate (or scope out) the window (`../hardware/isa-mechanisms.md`).
4. Place the scale path's work by class, then compute the budget — including any descale that
   stayed as explicit vector math (`../tile-programming/llir-codesign.md ## Attention: the
   co-execution budget`).
5. Measure the wave-structure question above before committing to a ping-pong.

> **Steps 2 and 3 sit on opposite counter calibers, by construction.** The f16 anchor issues the
> regular matrix instruction; the accepted end state issues the scaled one with native operands.
> Those two do not report matrix engagement through the same counter
> (`gemm.md ## Preshuffled / block-scaled signals`, the dtype-aware counter bullet;
> `../hardware/optimization-gotchas.md` rows 2 and 5). Elsewhere that mismatch shows up between two
> dispatch entries a reader might compare by choice — here it is **between two prescribed steps of
> this page's own workflow**, so an anchor-to-candidate comparison of matrix occupancy reads as the
> engine going idle even when the candidate is strictly better. Carry the caliber with the number:
> a matrix-occupancy reading taken at step 2 is not comparable to one taken after step 3 unless
> both say which matrix op they were measuring.

## Acceptance

- Numerics under the **task's own** oracle type, not a generic tolerance.
- Scaled matrix ops present in the lowered code with the correct format/scale encoding, operands
  native (no upcast back to f16 in the hot loop).
- The two wave-structure readings above, recorded separately.
- Every matrix-occupancy reading recorded with the caliber it was taken under, so the anchor and
  the candidate are not compared across the boundary in the note above.
- Timing at the pinned comparator as the arbiter, with the budget and the lowered code agreeing
  about *why* it moved.
