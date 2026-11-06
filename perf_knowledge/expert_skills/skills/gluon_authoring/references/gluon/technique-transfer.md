# Carrying a technique from one kernel to the next

`../workloads/index.md` defines the pipeline column as **"the part that does not transfer between
archetypes."** That sentence names a complement it never writes down. This page is that
complement: for each backbone layer, what you can carry from a kernel you just finished onto the
next one **unchanged**, what carries only after you **recompute a quantity**, and what **reverses
sign** — where the same move that paid last time costs you this time.

**What this page is and is not.** Every cell here is a pointer to a conclusion that already stands
somewhere else in the pack, plus one sentence of *why it transfers* — because a transfer list
without the mechanism is the kind of list that silently goes stale. Nothing here is new, nothing
here is a measurement, and nothing here ranks levers or quotes a benefit size. The pack already
records that a census can improve while the time gets worse
(`../pitfalls/negative-patterns.md ## Instruction count is not the objective function`); the value of
knowing a technique transfers is that you skip re-deriving it, not that you skip measuring it.

**Not the same axis as `../hardware/optimization-gotchas.md`.** That page is eight *profile-reading*
traps — the same kernel, read wrongly. This one is about *two kernels* — a conclusion that was
correct on the first and is not portable to the second. Read that one at the start of a kernel;
read this one when you arrive with a finished kernel behind you.

## Layer 1.5 — scheduling model

| | |
| --- | --- |
| **Carries unchanged** | **Choose the model before you choose a depth, and treat the choice as pinning `warps/CTA`** (`../tile-programming/scheduling-model.md ## How to choose`). It transfers because the ordering is structural, not empirical: the model determines what the launch geometry has to be, so a depth chosen first is a depth chosen against a geometry you have not committed to yet. |
| **Recompute first** | **The unit is the region, not the kernel** (`../tile-programming/scheduling-model.md ## The unit is the region, not the kernel`). A kernel whose body was one region last time may be two this time, and the model is then answered per region — so the *answer* does not carry even though the procedure does. |
| **Reverses sign** | **Wave-level ping-pong's candidacy gate** (`../tile-programming/warp-pipeline.md ## Gate 1: is this kernel a candidate?`). It needs two interleavable dot clusters. A decode-shaped body has a degenerate q dimension and therefore no second cluster, so the move that paid on prefill has nothing to interleave here and spends the schedule for it anyway. |

## Layer 2 — memory path

| | |
| --- | --- |
| **Carries unchanged** | **The async direct-to-LDS path accepts only a small set of per-lane widths, and that is an applicability gate rather than a tuning preference** (`memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`). It transfers because the floor is a property of the copy instruction, not of your tile: below it the path is simply unavailable, on any kernel. |
| **Recompute first** | **Which legal width you are landing on, and how many elements that is** — the same width is a different element count per dtype, so the tile shape that hit a legal width for one dtype misses for a narrower one. It is a set, not a floor: overshooting lands you on an illegal width as easily as undershooting. |
| **Reverses sign** | **Porting a gfx950 async ring down to the gfx942 downgrade reverses its sign**: at CDNA3's 32-bit-per-thread width the async form measured slower than synchronous staging on every minor (`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`; row A1 of `pipeline/authored-overlap.md`). The mechanism is available and the code is correct; it is the width that makes the trade negative. |

## Layer 3 — LDS layout

| | |
| --- | --- |
| **Carries unchanged** | **Settle the conflict question at compile time before changing a layout** (`appendix-api.md ## The compile-time oracle worth knowing about`). It transfers because `gl.bank_conflicts` reads only two types — it needs no GPU, no run, and nothing about the kernel's shape history, so the procedure is identical on every kernel that has a distributed-to-shared access at all. |
| **Recompute first** | **The `dtype` you hand `compute_efficient_padded_shared_layout`** — element bit-width is the contract, and the function returns `None` rather than raising when your `k_width` / element width / instruction shape falls outside its coverage. Whether that `None` needs a branch is itself a per-kernel question: statically-pinned arguments constant-fold and a fallback arm is untestable dead code, while an argument that varies at compile time must be handled (`layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)`). And **recompute the occupancy tier**: padding buys conflict-freedom out of LDS capacity, which is its own divisor (`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`). |
| **Reverses sign** | **A narrower dtype tightens the per-lane vector-width ceiling instead of relaxing it** (`../workloads/prologue.md ## Vector width is arithmetic, not a knob`). The intuition that a smaller element gives you more room is the wrong direction, and it is the one that sends people to widen `size_per_thread` on exactly the kernel that cannot afford it. |

## Layer 4 — pipeline

The layer the pack declares non-transferable. What transfers is therefore the **method**, never the
structure.

| | |
| --- | --- |
| **Carries unchanged** | **The order to reach for pipeline mechanisms, and where re-injection sits in it**: hand-written first — register-level prefetch, an authored LDS ring, then `warp_pipeline_stage` and the scheduling-model choice (`../tile-programming/pipeline.md`, which defines the order once). Re-injection is the lowest rung: a diagnostic below the parity gate that sizes a `lost_pipeline` debt, or a last resort when the hand-written form cannot reach parity; its ceiling is plain parity, its numbers are `injected`, and it never applies to an incumbent Gluon kernel (`pipeline/reinjection.md`). The rule transfers because it is about where you are in the process, not about the kernel. |
| **Recompute first** | **The stage count, from this loop's dependencies** (`../tile-programming/warp-pipeline.md ### Deriving the stage count from the loop's dependencies`), and the drain that follows from it (`### How long the drain is, and why the skeleton's formula does not generalize`). Copying a depth across kernels is copying the one number that is derived entirely from the body you just replaced. |
| **Reverses sign** | **The presence of a `tt.dot` is not what makes re-injection the move** (`pipeline/reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`). A `tt.dot` only makes the loop a candidate for the pass; reaching for it on a kernel that never had a plain pipeline to lose is spending the mechanism on a debt that does not exist. |

## Layer 5 — slicing and registers

| | |
| --- | --- |
| **Carries unchanged** | **Ask which limiter is active before moving any occupancy lever, and accept "neither" as an answer** (`../tile-programming/slicing.md ### Occupancy budget (P8)`). It transfers because an empty limiter is a real, common result — and it is the result that makes every lever in this layer inert, whatever worked last time. |
| **Recompute first** | **The tier edges, on both divisors.** VGPR: `waves/SIMD = min(8, 512 // round_up(vgpr, 8))`, so 64 registers is the last cell of the 8-wave tier and 65 drops to 7 (`kernel_workflow/scripts/kernel_tools/amd_occupancy.py --arch gfx950`, and the worked form in `../workloads/linear-attention.md`). LDS: its own capacity-over-granule floor division (`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`). A saving that does not cross a divisor buys nothing on either. |
| **Reverses sign** | **On a dependency-chain-bound loop, raising occupancy is the wrong direction** — more waves do not shorten a chain. And the launch geometry is not free to raise either: `warps_per_cta` is a coupled layout rewrite rather than a knob (`../pitfalls/negative-patterns.md` ## `warps_per_cta` is not an independently tunable knob), and `num_warps=1` is frequently deliberate — on gfx950 a 2D grid combined with it is a recorded performance hazard (`../hardware/capability-matrix.md`). A generic "raise the warp count" recommendation should be rejected on this layer, not tried. |

## Layer 7 — matrix

| | |
| --- | --- |
| **Carries unchanged** | **The lowering order, and counting conversions in the steady state rather than in the prologue** (`matrix-reference.md ## Matrix Lowering Order`, `## Hot-Loop Conversion Counting`). Both transfer because they are properties of how the layout chain is resolved, which does not depend on the dtype or the shape you resolved it for. |
| **Recompute first** | **Which instruction this dtype *at this shape* is entitled to** — the entitlement is per `(version, M, N, K)`, not per dtype. On CDNA4 fp4 has no regular MFMA intrinsic at all, and fp8 has one at `[16, 16, 32]` / `[32, 32, 16]` but not at `[16, 16, 128]` / `[32, 32, 64]`; where the regular entry is missing the path is `mfma_scaled`, with the scale operand's layout derived from the matrix tile and therefore moving whenever that tile moves (`matrix-reference.md ## Matrix-Family Details`, `layout-reference.md ## The Scale Operand's Layout For mfma_scaled`). |
| **Reverses sign** | **Scaled fp8 does not move the generic MFMA counter**, so a `MfmaUtil` near zero means the opposite of what it meant on the bf16 kernel you came from — and the same flag is arch-scoped, because gfx942's non-scaled fp8 *does* hit it (`../hardware/optimization-gotchas.md`, row 2). |

## Cross-layer: the counter convention

| | |
| --- | --- |
| **Carries unchanged** | **State which counter convention a bound claim is made under, before making it** (`../hardware/bound-class-signals.md`, and the eight reversed-intuition rows in `../hardware/optimization-gotchas.md`). |
| **Recompute first** | **Matrix engagement is a union test, not a single counter** — generic MFMA utilisation *or* the scaled-fp8 pipe — so the expression you evaluate changes with the dtype even though the question does not. |
| **Reverses sign** | **Mixing two conventions inside one comparison does not add noise, it inverts the conclusion.** A before/after where the two sides are read under different conventions can show a regression for a change that improved the kernel, and the reverse. |

## Cross-layer: does a roofline apply

| | |
| --- | --- |
| **Carries unchanged** | **Ask whether a roofline applies to this kernel at all before quoting either floor** (`../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`, and C0 in `../method/profile.md ## 3.1 Required evidence`). It transfers because the question is about the measurement setup — workgroup count, working set, dispatches inside the timed region — none of which is a property of the algorithm. |
| **Recompute first** | **Which floor the residency voids.** A working set inside the memory-side LLC voids the **memory** floor only; the compute floor survives and becomes the binding one. The tell is a measured time *below* the memory floor — read it as the model announcing it does not apply, not as an impossibility. |
| **Reverses sign** | **A bound claim argued as a percentage of HBM peak is invalid end to end on an LLC-resident working set** — not weakened, invalid. Carrying that style of argument from a large-footprint kernel to a small one produces a confident wrong classification. |

## Cross-layer: exp and exp2

| | |
| --- | --- |
| **Carries unchanged** | **Fold a per-element scalar onto the earliest, cheapest level it can live at** (`../method/profile.md ### Reducing compute-class VALU (fold scalars off the tile)`). It transfers because it is an algebraic identity; the only question per kernel is which level is earliest. |
| **Recompute first** | **What the fold did to your statistics' units.** The softmax form moves the `log2e` onto `lse`, so the statistic the loop now carries is not in the base the unfolded version produced — every consumer of that statistic has to agree with the producer about which one it is. |
| **Reverses sign** | **Polynomial FMA emulation of `exp2` pays only when the transcendental unit itself is the bound** (`../method/profile.md ### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)`). It works by moving a fraction of the evaluations onto the FMA pipe so both run concurrently — so on a kernel that is bound on total VALU instead, it adds work to the pipe that is already the limiter. |

## Cross-layer: inline-asm sites

| | |
| --- | --- |
| **Carries unchanged** | **The classification itself.** A site's mechanism bin is read off two strings and its intent bin off one deletion test, and both are properties of the call rather than of the kernel (`inline-asm-reference.md ## Classifying a site: mechanism × intent`). So does the cost: the optimizer stops seeing through the block on every kernel, and durability is per class on every kernel. |
| **Recompute first** | **Every parameter.** `pack`, the constraint string's arity, the clobber list, sometimes the asm text and the comparison polarity can each be keyed on a tile constant — and occasionally three of them on the *same* constant in one call: asm text, physical-VGPR clobber list and an operand all moving together with the row count. Re-derive them against **this** tile; a transplanted site compiles and is wrong at a shape you did not run (`## Shape-keying: inline asm is a per-M specialization`). And re-check the pairing: a site that is one half of an issue/wait or save/restore pair carries no intent on its own, so carrying one half forward carries nothing. |
| **Reverses sign** | **"It worked in the last kernel" is evidence against, for the scheduling class specifically.** Those wins are pinned to one compiler version and evaporate on upgrade with nothing warning you — the asm still assembles (`## What it costs you, on every class`). Treat the previous kernel's placement as the hypothesis to disprove on this toolchain, not as the reason to paste it. |

## Using this page

Read the row for the layer you are about to touch, not the whole page. Then:

1. If the cell you need is in **Carries unchanged**, skip the derivation and go straight to the
   edit — but still verify the edit landed before reading a time
   (`../hardware/bound-class-signals.md ## Lever gating laws`).
2. If it is in **Recompute first**, the pointer tells you *which quantity*; recompute it on this
   kernel before writing anything.
3. If it is in **Reverses sign**, the previous kernel is evidence *against* the move here. Treat
   the last kernel's success as the hypothesis to disprove, not as the reason to proceed.
