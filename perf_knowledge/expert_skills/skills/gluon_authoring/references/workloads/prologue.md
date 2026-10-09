# Prologue kernels — the per-token body with no hot loop

The kernel that runs before attention: QK-norm, RoPE, the paged KV-cache write, per-token
quantization, and the fused combinations of those. One token's worth of work per program, a handful
of tensors, and — the defining property — **no reduction loop to pipeline**.

Reached from `index.md ## prologue`. That row is route-only: `hw_budget.py` has no byte/FLOP model
for this archetype, so declare the traffic with `--tensors` and expect ~0 useful FLOPs. Zero is a
real answer here; it says the compute floor cannot bind and the whole question is the byte side.

## Which layers act when there is no loop

The backbone (`../hardware/atlas.md` Table B) is written for a body with a hot loop, and the
common shorthand — "layers 2–5 have nothing to act on" — over-states it. Exactly one layer is
inert *for lack of a loop*, and it takes its dependent with it:

| Layer | Acts on a loop-free body? | Why |
| --- | --- | --- |
| 1 anchor | **yes** | Transcription is about explicit layouts, not about a loop. |
| 1.5 scheduling model | **mostly not** | All three models describe how a loop body's stages interleave. With no stages there is nothing to choose between, and warps/CTA falls out of the grid instead. |
| 2 memory path | **yes** | Coalescing, `buffer_load` vs generic load, and the per-lane vector width are decided per access, not per iteration. Only the *async direct-to-LDS* half of this layer needs a ring to pay for itself — and see the floor below. |
| 3 LDS layout | **usually not, for a different reason** | Not because there is no loop, but because these kernels typically stage nothing through LDS. No LDS traffic, no banks, no conflict to resolve. If your variant *does* stage through LDS, this layer is fully live. |
| 4 pipeline | **no** | Prefetch and double-buffering need iterations to overlap. This is the one layer the missing loop actually removes. |
| 4+ instruction scheduling | **no** | It paces what layer 4 created and never creates a structure itself, so it goes with layer 4. |
| 5 slicing / registers | **yes** | The VGPR budget and the occupancy tier are properties of the kernel, not of a loop. |
| 6 beyond the hot loop | **yes, and this is often where the answer is** | A launch-per-token grid is a layer-6 question: grid rank, workgroup count, and whether the launch geometry itself is the cost. |

So the reading order for this archetype is **2 → 5 → 6**, with 3 conditional on whether anything
reaches LDS. Proposing a layer-4 edit here is not a slow round, it is a round spent on a layer that
has no surface.

> **The async floor is a second, independent reason layer 2's ring half does not apply.**
> Direct-to-LDS async copy accepts only a small set of per-thread widths — 16 B or 4 B on gfx950
> (CDNA4); on the gfx942 (CDNA3) downgrade 4 B only, with `order=[1,0]` — and it is a set, not a floor
> (`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`).
> A per-token body usually owns only a few elements per lane, which is below that floor — the load
> is rejected at lowering, and the arithmetic in the next section says the floor is often
> unreachable by construction rather than by a bad layout choice.

## Vector width is arithmetic, not a knob

The per-lane extent is not free to choose. For one row of `row_elements` spread across a wave:

```text
elements_per_lane = row_elements / lanes_per_wave        # lanes_per_wave == 64 on CDNA
bytes_per_lane    = elements_per_lane * dtype_bytes
```

`lanes_per_wave` is fixed at 64 by the wave64 constraint
(`../gluon/layout-reference.md ## BlockedLayout Constraints (wave64)`:
`product(threads_per_warp) == 64`). So for a 256-element row at bf16, each lane owns 4 elements =
8 B, and **a 128-bit per-lane access is unreachable at that shape** — not because
`size_per_thread` was set timidly, but because there are not enough elements to go around.

Two consequences that run against the instinct:

- **A narrower dtype tightens this bound, it does not relax it.** Halving `dtype_bytes` halves
  `bytes_per_lane` for the same row. Moving a prologue from bf16 to fp8 moves it *away* from the
  wide-access and async-floor thresholds, not toward them.
- **Narrow per-lane access is not the same as uncoalesced access.** 64 lanes × 2 B is still 128
  contiguous bytes. Judge the coalescing on the wave's footprint; judge the instruction width on
  the lane's. They are different questions and only the second one is bounded by the row length.

The lever that does exist is **widening the row the tile covers** — giving each program more than
one token, or more than one head — which raises `row_elements` and therefore `elements_per_lane`.
That is a layer-6 change to the grid, which is another reason the reading order above ends there.

> **`size_per_thread` is coupled, so this is never a one-line edit.** Layer-3 and layer-2 decisions
> that were derived from the old per-lane extent move with it, and on this archetype the coupling
> is usually to the rotary partner index below: raising `size_per_thread` is exactly what stops a
> pairing from being lane-local. `../gluon/layout-reference.md ## BlockedLayout Constraints (wave64)`
> states the rule; the next section is the instance that bites here.

## Rotary pairing has three layouts

"Rotary pairing is a lane exchange" is true for some rotary kernels and false for others, and the
difference decides whether the layer-2/3 work is zero or substantial. Which one you have is a
property of the **reference implementation's pairing convention**, so read it off the source rather
than assuming:

| Convention | Partner of element `r` | What it costs |
| --- | --- | --- |
| **Interleaved** (adjacent even/odd) | `r ^ 1` | Cheapest. If each lane owns both halves of its pairs, the exchange is *in-register*: `reshape` to `[..., 2]`, `split`, `join` the halves in the other order. No cross-lane traffic at all. |
| **Split-half / NeoX** (first half pairs with second) | `(r + width/2) % width` | Lane-local only if `width/2` is a whole number of per-lane elements *and* the halves land in the same lane. Otherwise it is a cross-lane exchange with a regular, statically-known stride. |
| **Genuine runtime pairing** | data-dependent | A real `gl.gather`, with the layout conditions and the silent LDS fallback that come with it. |

For the interleaved case the shape APIs preserve or infer layout through the transform
(`../gluon/layout-reference.md ## Shape APIs And Layout Propagation`), and the result can be
re-parented with `gl.convert_layout(..., assert_trivial=True)` — which is the point of that
argument: it turns "I believe this conversion is free" into a compile-time failure if it is not
(`../gluon/layout-reference.md` ## `convert_layout` Decision Table). Claim the conversion is trivial
by asserting it, not by commenting it.

For the third case, check all three warp-local conditions before assuming the exchange is cheap
(`../gluon/layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`).
**Missing any one of them is silent and the cost is not a round trip** — the scratch requirement
goes from zero to the size of the whole source tensor and is charged against the kernel's
shared-memory budget, which means against occupancy, which means against the whole kernel and not
just the gather. And per the previous section, widening `size_per_thread` along the gather axis
grows the shuffle count faster than it grows the work, so the two levers pull against each other.

## The paged index chain

Paged KV-cache writes and paged reads load a block table, then use it to address the payload. The
structural fact is that **the index load is a dependency, not a payload**: the address arithmetic
for the payload cannot begin until the index arrives, so the two are serialized within one token's
work no matter how the loads are spelled.

That makes the only available overlap a cross-token one — issue the *next* token's index load
before consuming the *current* token's payload — which is a software-pipelining shape at the grid
level rather than at the loop level. Before building it, price it: an index chain is a small number
of bytes against a payload that is usually much larger, so the latency you can hide is bounded by
how much of the kernel is actually waiting on the index rather than on the payload. If the profile
does not show the index load exposed, this is not the lever.

The two levels of runtime-indexed access and which one applies are routed by
`../gluon/index.md`'s runtime-indexed-access row: register-level and layout-conditional
(`../gluon/layout-reference.md`) versus LDS-level (`../gluon/smem-lds-reference.md`).

## The quant epilogue's scale layout is a contract

When the prologue ends in a quantization that writes both a value tensor and a scale tensor, the
scale's layout is **not** a local choice — it is consumed by a downstream matrix instruction, and
that instruction has an opinion.

- **The hardware group size on CDNA4 is 32.** A group of 128 is a software convention the matrix
  instruction has no notion of, and NVFP4's is 16. The same trap is stated in `index.md ## norm`;
  it applies here for the same reason and with the same consequence.
- **The consumer's scale layout is derivable, so derive it rather than matching it by hand.**
  `../gluon/layout-reference.md ## The Scale Operand's Layout For mfma_scaled` covers
  `gl.amd.cdna4.get_mfma_scale_layout`, which takes the consumer's dot-operand layout and returns
  the layout the scales must be in. Its own assertion fixes the scale factor at 32, which is the
  bullet above expressed as an API contract rather than as prose.
- **gfx942 downgrade:** there is no scaled matrix consumer, so the downstream GEMM applies the
  scale as explicit VALU on its accumulator and the group size is a software contract between the
  two kernels only; the fp8 payload must be written in the FNUZ encoding (`e4m3fnuz` saturates at
  240, not 448), so a quantizer ported from gfx950 changes both its clamp and its bit pattern.
- **Write down the contract next to the kernel.** A prologue whose scale layout only works for one
  downstream instruction shape has a precondition, and the consumer is in a different file and
  usually a different launch. `../method/triage.md` when the gap is the language's rather
  than yours.

## Before recording a result here

- The compute floor is ~0 by construction, so a `memory` verdict is a **default, not a finding**.
  Say which side you measured, and check the working set against the LLC before quoting any
  fraction of peak bandwidth (`../hardware/roofline-models.md`) — a per-token body's footprint is
  frequently small enough that the HBM figure is not the right denominator.
- A 2D or higher grid combined with `num_warps=1` is a known performance hazard on this target and
  this archetype reaches it often; grid rank and warp count must be tested separately, not as one
  change (`../hardware/capability-matrix.md`).
- Layer 4 being inert is a **result to record**, not a gap to apologize for. The next round should
  not re-derive it.
