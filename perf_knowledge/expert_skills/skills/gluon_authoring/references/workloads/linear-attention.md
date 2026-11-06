# Chunked Linear Attention / Gated Recurrence (gfx950 / gfx942)

Gated DeltaNet, delta-rule and the linear-attention family: a recurrent state carried from chunk
to chunk instead of a score matrix normalized within one. This page exists because the attention
pages cannot host it — their variable set assumes a softmax between two matmuls, and this
archetype has neither the softmax nor the `[Br, Bc]` intermediate. Read
`intake.md ## 0. First: is it the archetype its name says?` before mapping it onto
attention's variables; it is a different archetype, not a different variable assignment.

**No upstream anchor.** Triton ships no linear-attention, chunked-recurrence or state-passing
example in its tutorials, examples, or tests on any of 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0. Everything
below is derived from mechanisms this skill already establishes elsewhere, and is marked where it
is. Nothing here has been run on hardware.

## The loop shape, and where the state lives

Per chunk `i`: two matmuls and one state update.

```text
intra-chunk : A_i   = Q_i @ K_i^T          (masked to the chunk)
cross-chunk : O_i  += Q_i @ S_{i-1}        (the recurrence's read)
state       : S_i   = K_i^T @ V_i + S_{i-1} * decay_i
```

**That is gated linear attention, and it is only half the family this page is named for.** A
delta rule carries a write strength `beta_i` and writes a **residual** rather than the value
itself — `u_i = beta_i (v_i - S_{i-1} k_i)` — and because `S` is updated inside the chunk, the
rows of a chunk are no longer independent of each other. Serialising them is the naive reading and
it is the one to avoid: substituting the recurrence into itself turns the whole chunk into one
linear solve,

```text
(I + T) U = beta (V - diag(p) S_in K^T),   T = tril(-beta_i (K K^T)_ij D_ij, strict)
```

and **the word that makes this usable is "strict".** The coupling only ever runs `j < i`, so `T` is
*strictly* lower triangular, so `I + T` is **unit** lower triangular: its diagonal is exactly 1, its
determinant is exactly 1, it is invertible for every input, and its inverse is unit lower triangular
too. **No numerical condition is being assumed** — not well-conditioning, not non-singularity, and
nothing about the decay values, which may be zero. It is a structural guarantee that falls out of
the index range, which is what makes replacing a serial chunk-length recurrence with a solve a safe
rewrite rather than a bet. Production does this and ships it; the file names it in its own
docstring, "the three nonzero blocks of the unit-triangular inverse". Everything this page says
below about matrix rates, state residency and chunk length applies to that solve as well — it is
where a large share of a delta-rule prefill kernel's work actually is.

The one structural fact that decides everything downstream: **`S` is shaped `[D_k, D_v]`, and
that shape has no chunk-length term in it.** The accumulator of a flash-attention body is
`[Br, D]` and shrinks when you shrink the query block; `S` does not. It is the same size at
chunk length 16 as at 128.

## The recurrence: fold the decay into the accumulator operand

This is the first thing to get right, and it fails in a way that survives a smoke test.

```text
correct : S = mfma(k_t, v, S * decay)        # decay applied to the operand the MFMA adds
wrong   : S = S * decay + mfma(k_t, v, S)    # counts S twice
```

It is the same defect, with the same signature, as the online-softmax rescale documented in
`../gluon/matrix-reference.md ## Accumulator + per-step rescale (online normalization)` — **it passes at a
single chunk and fails from two on**, because the first chunk's decay multiplies a zero state.
Read that section; the reasoning transfers unchanged, and a gated recurrence is the case where a
non-trivial decay arrives earliest.

## The affine prefix scan: the one lever that moves the chain length

Treat this as a primary structural technique for this archetype, at the same level as online
softmax is for attention — not as an exotic alternative. It is what online softmax is too: a
restatement of the recurrence that changes which quantity the loop is serial in.

**The structure is a mathematical identity, not a heuristic.** The state update in the loop shape
above is an affine map of the previous state:

```text
S_i = A_i * S_{i-1} + B_i        with  A_i = decay_i ,  B_i = K_i^T @ V_i
```

Affine maps compose, and the composition is associative:

```text
(A_2, B_2) . (A_1, B_1) = (A_2 * A_1 , A_2 * B_1 + B_2)
```

So the chain `S_0 -> S_1 -> ... -> S_n` *is* an associative scan over the per-chunk pairs, and its
serial depth drops from `O(chunks)` to `O(log chunks)`. Every other lever on this page — the state's
dtype, splitting `D_v`, pipeline depth — changes the cost of a step. **This is the only one that
changes how many steps are on the critical path.** That is the reason to know it; whether the chain
length is what is actually costing you is still a measurement, and if the loop is bandwidth-bound on
`K`/`V` it is not.

**The structure of `A_i` is a price, not an admission test.** The composition above needs
`A_2 * A_1` and `A_2 * B_1`. With a scalar or per-channel decay those are a multiply and a scaling
of the `[D_k, D_v]` increment — cheap, which is why the technique reads naturally in that case.
With a general matrix transition they are matmuls, one per combine. **That is a cost to weigh
against the three below, not a reason to stop reading**: a shipped gfx950 GDN prefill kernel builds
a dense 128x128 transform and composes it, and its profile's registered signature range proves the
file is served rather than vestigial. Its `A_i` is dense by construction — the delta correction
contributes a `(diag(q)K)^T (I+T)^{-1} beta diag(p)K` term that no choice of decay makes diagonal.
So price the composition by the structure of `A_i` and decide together with the two variables in
cost 2; do not gate on it.

Three costs, all of which land on things this page has already priced:

1. **The carried object doubles.** You carry `(A_i, B_i)` where you carried `S`. `B` is
   `[D_k, D_v]`-shaped, the same size as the state, so the register budget line in
   `## The state is an occupancy budget line; whether it is the limiter is a separate question`
   is the one that moves — and it moves in the direction that crosses a tier divisor, not away
   from one. Predict the new tier before building.
2. **Total work goes up while depth goes down.** A sequential chain performs one combine per chunk;
   a scan performs more (an up-sweep and a down-sweep, or a log-depth sweep that repeats work).
   You are buying parallelism with arithmetic, so it pays only where the machine was idle waiting
   on the chain — and that is **two variables, not a position on the decode-to-prefill axis.**
   Compute both before deciding:
   - **Is there a chain to compress?** Its length is the chunk count. A single-step form has one
     update and therefore nothing to shorten, so the technique does not apply there rather than
     applying especially well.
   - **Is the machine already full without it?** The independent work is the product of the batch,
     head and sequence-block extents that do *not* participate in the chain. Compare the resulting
     CTA count against the device's CU count.
   The two combine rather than reduce to one axis, and the same sequence length can land on either
   answer: a long chain with too few CTAs to fill the device is the case the scan is for, while the
   same chain at a batch large enough to saturate is one where it spends arithmetic on a machine
   that had no idle slot to buy. Predict the CTA count first, then the chain length.
3. **`gl.associative_scan` does not express it.** The builtin scans over the elements of a tile
   along an axis; here the scan element is a state *matrix* plus its decay, at chunk granularity.
   The association is yours to author across chunks (and possibly across workgroups or dispatches),
   which means you own the handoff — and a cross-dispatch handoff of running statistics is the same
   hazard class flagged in `index.md ## scan`, which labels scan a mechanism rather
   than an archetype for exactly this reason. Use the builtin for the *within-chunk* decay
   accumulation, as the primitives table below already does, and do not expect it to scale up to
   the chunk chain.

## The Independence rule is not what blocks you here

A loop-carried recurrence looks like it should defeat software pipelining, and the rule it looks
like it violates is `../tile-programming/pipeline.md ### Independence rule (correctness of pipelining)`.
It does not violate it. That rule constrains the **staging slots** — `DOT(k)` must not depend on
the same-slot `LR(k+1)` or `AC(k+2)` — and the recurrence lives on the accumulator side, not in
the prefetch buffers. Each chunk's `Q` / `K` / `V` / gate tiles are independent of every other
chunk's, which is exactly what the rule asks for.

The evidence that this is not a theoretical distinction: injecting plain's pipeliner into a body
with two chained dots *and* loop-carried accumulators does not get refused
(`../gluon/pipeline-reference.md ## And on attention — two dots chained through a softmax`). That is
evidence about legality, not a route recommendation — the overlap here is still hand-written first
(`## Getting the pipeline, and two ways of being misled about it`).

What the recurrence does block is narrower and already recorded: **symmetric ping-pong**. The
Improve step in `../tile-programming/pipeline.md` admits ping-pong "only where the data dependency
allows", and names the online-softmax recurrence as the case that blocks it. A gated state update
is the same dependency shape. Take the buffering depth; do not plan on the symmetric split.

## The state is an occupancy budget line; whether it is the limiter is a separate question

Two claims get merged into "occupancy is the constraint here", and they need different evidence.
Keep them apart:

- **A budget line is arithmetic, knowable before anything runs.** `S` sits in registers across the
  entire loop, alongside the prefetch buffers for the next chunk's operands. For a `[128, 128]` fp32
  state on a four-wave workgroup that is on the order of 64 VGPRs per lane, held for the whole
  kernel, before any staging. You compute this.
- **A limiter is a measurement.** Whether that budget line is what actually caps resident waves is
  the SPI occupancy-limiter reading, and it can come back **empty** — which means nothing was
  resource-capped and the kernel is not occupancy-bound at all
  (`../method/profile.md ### rocprof-compute per-round (aggregate SOL + memory chart + occupancy limiter)`).

**The reverse route matters as much as the forward one.** If the limiter is empty and matrix-engine
utilization is low, the fork in
`../hardware/bound-class-signals.md ## under-fill vs occupancy (the low-MfmaUtil fork)` sends you
the *opposite* way: the tile is too small, and the fix is to grow it and accept lower occupancy.
Relieving a state that was never the limiter costs a round and moves nothing — and on this workload
the state is conspicuous enough to be blamed by inspection.

### Which tier a state size lands in

CDNA occupancy does not degrade smoothly. It is a combined `arch_vgpr + accum_vgpr <= 512` per-SIMD
budget with an allocation granule of **8** and a cap of **8 waves/SIMD**, so
`waves/SIMD = min(8, 512 // round_up(next_free_vgpr, 8))`
(`../hardware/planning-constants.md ## VGPR / occupancy thresholds (CDNA combined budget)`, which
also tabulates the lower tiers). The boundary that matters at this page's state sizes:

| `next_free_vgpr` | waves/SIMD | |
| --- | --- | --- |
| 64 | 8 | the last slot in the top tier |
| 65 | **7** | one register over, one tier down |

`scripts/amd_occupancy.py --vgpr 65 --arch gfx950` prints that without a GPU, and
`--asm <kernel>.s` reads it from the KD together with LLVM's own `; Occupancy:` comment. Take
`next_free_vgpr` from the KD, not from a profiler's VGPR field alone.

**Do not read that `; Occupancy:` comment as the kernel's occupancy.** It is authoritative for the
**register term** and for nothing else: across six retained dumps at three distinct VGPR counts it
equals `floor(512 / VGPR)` exactly, with **no LDS term in it at all**, and the emitter prints its
own disqualifier four lines above the value —
`LDSByteSize: 0 bytes/workgroup (compile time only)`. A Gluon kernel allocates its LDS
**dynamically at launch**, so that field reads 0 and the emitted value can overstate the real
limit by 3x. On a kernel of this shape the hand derivation `min(VGPR-limited, LDS-limited)` is
**more correct, not less**. (Whether a *statically* allocated LDS makes the emitter include the
term is unverified — do not read this either way.) Owner of the full argument, including why
stamping your own derivation "unverified against the authoritative emitter" points the reader at
the wrong number: `../hardware/planning-constants.md ## The emitted ; Occupancy: N is a
register-term answer`.

**And the `next_free_vgpr` → waves/SIMD table in this section is in waves per SIMD, which is not
the residency that feeds bandwidth.** That is **workgroups per CU**, and a workgroup is
indivisible. Convert before spending a round — the conversion, why it deletes most of the ladder at
high warp counts, and the measured arm that cut two VGPR rungs for zero extra workgroups are owned
by
`../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`
(worked for a Gluon warp-count knob in `moe.md`, subsection *What raising `num_warps` does and
does not buy you*). Carry `num_warps` as an explicit column on any occupancy table this page's
tiers feed. The LDS term of that conversion is arch-dependent: 160 KiB per CU on gfx950, 64 KiB on
the gfx942 downgrade, so a state-plus-staging footprint that is VGPR-limited on gfx950 can be
LDS-limited on gfx942.

The consequence for a carried state: **halving its dtype is a tier change or it is nothing** — and
a tier change is a *permission*, not a prediction. A state already sitting at the top tier gains no
waves from going fp32 → bf16; one sitting just past a divisor gains a whole step, which still has
to survive the `wg/CU` conversion before it is a step in anything that feeds bandwidth. Work
out which side of a divisor you are on before spending the round, and predict the new tier rather
than reading it off afterwards.

> **Occupancy is one-sided here as everywhere: use a computed *loss* to veto, never a computed
> *gain* to motivate.** The measured CDNA4 points behind that rule, and why they must not be
> averaged into an exchange rate, are stated once in `moe.md`, subsection *What raising
> `num_warps` does and does not buy you*. What survives is the compile-side feasibility question: does the
> configuration fit, does it spill, and what is the **next** threshold that actually moves `wg/CU`.

> **On the state, that trade is gated by correctness before it is gated by tiers.** Every other
> lever on this page is free to be judged on time alone, because it cannot change the answer. The
> state's dtype can: `S` is carried across chunks, and in the single-step form it is carried across
> *tokens*, so a rounding error introduced into it is fed back into its own next update rather than
> being discarded at the end of a tile. That is the one place in this workload where reduced
> precision compounds along the generated sequence instead of staying bounded. Settle it against
> the correctness oracle's **type** first
> (`../tile-programming/low-precision.md`) — an angular/cosine gate can reject a state downcast
> that an abs/rel gate accepts, and reading the tier table before the oracle is how a round gets
> spent on a change that was never admissible.
>
> **The opposite error exists and reads as cleanup.** This page's risk is losing precision you
> needed. The mirror risk — staying in fp32 across a point where the reference materialized bf16,
> so the fused kernel is *more* accurate than the oracle and fails a tight one — is in
> `reduction-elementwise.md ## Stage 3 — dequant + residual + norm`. A narrowing round trip that
> looks like dead code may be the boundary contract; check the oracle before deleting one.

### The lever that does not apply here

Shrinking the query block relieves an attention accumulator; it does **not** touch `S`, because
`[D_k, D_v]` has no chunk-length term in it. The levers that do move it are the state's own dtype,
splitting `D_v` across workgroups, and accepting a shallower pipeline. Every register-buffered step
still has to pass the tri-lemma check in
`../tile-programming/slicing.md ### Occupancy budget (P8)` — predict waves/CU *before* keeping
the change, not after.

## Getting the pipeline, and two ways of being misled about it

- **`num_stages` is inert on the Gluon path on all four versions.** It is accepted, it lands in
  the IR, and nothing reads it, because `add_stages()` routes a Gluon kernel straight to
  `gluon_to_ttgir` while the passes that consume it live in `make_ttgir`
  (`../gluon/pipeline-reference.md ## Roll the loop (cut i-cache pressure)`). The failure is silent, so a
  null result from turning it up is **not** evidence that this loop resists pipelining. Depth
  comes from hand-authored staging, in the order `../tile-programming/pipeline.md` defines —
  register-level prefetch of the next chunk's `Q` / `K` / `V` / gate tiles, then an authored LDS ring
  (gfx950 `async_copy` + `commit_group` / `wait_group`; gfx942 downgrade: sync staging, async only
  32-bit with `order=[1,0]`), then `warp_pipeline_stage` where `num_warps >= 8`. Re-injecting plain's
  pipeliner is the lowest rung — a below-parity diagnostic or a last resort, its numbers labelled
  *injected*, never on an incumbent Gluon kernel. And there is a further direction the shared layer
  already documents: **single-wave ILP**, via a per-launch
  `llvm_fn_attrs` scheduler strategy paired with an unroll factor on the chunk loop
  (mechanism: `../tile-programming/llvm-fn-attrs.md`; placement in the layer loop:
  `../tile-programming/instruction-scheduling.md ## llvm_fn_attrs — the portable, per-compile scheduler strategy`).
  **Check the ILP-starvation gate before spending a round on it**: a chunk recurrence whose
  iterations form one serial dependence chain has no reordering slack to hand the scheduler, and
  there every strategy is neutral-to-negative. And the setting is **per launch, not per workload**
  — adjacent launches in one host function take different values, and a state-update kernel is not
  entitled to one because a sibling state-update kernel carries it. Neither the value nor the
  decision to use it transfers without its own assembly diff.
  **The two halves of that pairing sit at different levels, and conflating them wastes a round.**
  `llvm_fn_attrs` is a launch keyword; `loop_unroll_factor` is **not** — it is a keyword on the
  loop construct inside the kernel body, and no launch accepts it. The way a host-side decision
  reaches it is indirect: pass a `gl.constexpr` unroll parameter at the launch and spell the loop
  as `tl.range(..., loop_unroll_factor=UNROLL)` in the body. Go to
  `../gluon/pipeline-reference.md ## Roll the loop (cut i-cache pressure)` before writing either:
  the knob runs in two directions, and a chunk recurrence whose bound is **loaded from a
  sequence-offset tensor** is the one shape where raising it is not substitutable by
  `gl.static_range` — a runtime trip count has no compile-time form to unroll.
- **A minimal body does not settle it** (relevant when re-injection is used as the diagnostic or
  last resort). The same page records a case where the injection fired
  perfectly — op census identical to plain's — and still lost badly, because the staging layout it
  must fall back to cannot express what plain got. A state-update loop reads a V-like tile through
  exactly that path, so expect to have to measure it rather than infer it.
- **If you re-inject: `buffer_ops=True` conflicts with a body that already uses buffer ops.** The conversion it
  restores runs *after* the pipeliner, and it requires the body to have no `amdg.buffer_*` of its
  own left — including in early-exit branches. A state-update loop that loads or stores through
  `gl.amd.cdna3.buffer_load` / `buffer_store` fails legalization instead of pipelining. Two ways
  out, and pick deliberately: write both sides as `gl.load` / `gl.store` on the arm you intend to
  inject into, or keep the buffer ops and set `buffer_ops=False`. The rule and its exact failure
  strings are in `scripts/gluon_swp.py`.

## What the language gives you, on all four versions

Checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 — every primitive this workload needs is present on all
four, so there is no version gate on the workload itself:

| Need | API | Note |
| --- | --- | --- |
| the chunk loop | a plain Python `range` | Gluon has no `gl.range`; only `static_range` is exported. A plain `range` lowers to `scf.for` with the state as an `iter_args` value, which is what carries `S` |
| state seed | `gl.zeros(shape, dtype, layout=...)` | pass the layout explicitly, as everywhere else |
| within-chunk decay accumulation | `gl.associative_scan` | this is `tl_core.associative_scan` re-exported unchanged — **no Gluon-specific `layout=` kwarg**, unlike `gl.histogram`. Gluon exports no `cumsum`, so a running product/sum inside the chunk is built from this or from a reduction |
| the two matmuls and the state update | `gl.amd.cdna4.mfma` on gfx950 (`gl.amd.cdna3.mfma` on the gfx942 downgrade) | `S` is the accumulator operand, per the fold above |

Two version differences exist but sit beside the workload rather than in it: `gl.dot_fma` accepts
a rank-3 accumulator only from 3.8.0 — 3.6.0 / 3.7.0 / 3.7.1 unpack the accumulator shape as exactly
two values, so a batched call fails at trace time — and `loop_unroll_factor` only takes effect from
3.8.0. If a batched `dot_fma` is on the table, note that 3.8.0 also warns at compile time when the
joined `M x N x K` (times batch) exceeds 2^19, on the grounds that the FMA expansion compiles
slowly. A chunked recurrence reaches that product quickly, so read the warning as a reason to check
the tile rather than as noise.

## When the matrix core is still right

The reflex on seeing a recurrence is to reach for `gl.dot_fma`, and here that is usually wrong.
The M of both chunk matmuls is the **chunk length**, which is typically 16-64 and fills the MFMA
shape — this is not the M=1 decode regime that
`../gluon/matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)` describes.
Evaluate `dot_fma` only when the chunk length is at or below the instruction shape's M, and never
as a way around a layout that failed to compile.

### Which matrix rate the state's dtype puts you on

Choosing the matrix core still leaves a rate open, and on this workload the fp32 state picks the
slow one. `scripts/gfx950_isa.py facts <opcode>` gives dims and cycles per instruction, and
`2*M*N*K / cycles` turns them into a rate:

| Opcode | dims | cycles | FLOP/cyc | `C/D` VGPRs |
| --- | --- | --- | --- | --- |
| `V_MFMA_F32_16X16X4_F32` | 16x16x4 | 32 | **64** | 4 |
| `V_MFMA_F32_32X32X2_F32` | 32x32x2 | 64 | **64** | 16 |
| `V_MFMA_F32_16X16X32_BF16` | 16x16x32 | 16 | **1024** | 4 |

Two things follow, and the second is the one that gets tried anyway:

- **The fp32 matrix path runs at a sixteenth of the bf16 one.** An fp32 `S` used as the accumulator
  operand keeps the state update on that path. Whether that matters depends on where the bound is:
  while the chunk chain is the critical path the matrix rate is not what you are waiting on, but
  compressing the chain (`## The affine prefix scan: the one lever that moves the chain length`)
  can hand the bound to it. Re-read the bound after that change rather than assuming it stayed put.
- **gfx942 downgrade.** `gfx950_isa.py` is gfx950-only (and `V_MFMA_F32_16X16X32_BF16` is a CDNA4
  shape). On gfx942 the per-CU fp32 matrix rate is about the same as gfx950's while the bf16 rate is
  half (derive both from `perf_knowledge/hardware/data/sku.json` peaks over `cus × clock`), so the
  fp32-state penalty is roughly eight-fold there instead of sixteen-fold — still the slow path.
- **Widening the fp32 shape buys nothing.** Both fp32 shapes above sit at the same 64 FLOP/cyc, so
  `32x32x2` is not a faster tile — it is the same rate with a four-times larger accumulator
  (16 VGPRs against 4), which spends the budget line in `## The state is an occupancy budget line;
  whether it is the limiter is a separate question` for no rate in return. The only thing that
  moves this number is the dtype, and that is gated on the oracle first (the note in
  `### Which tier a state size lands in`).

## Variable-length sequences

When the chunk count is data-dependent, the three consequences in
`attention.md ### The dynamic-body variable (sparse / paged / varlen)` apply without
modification: the index chain is VALU work that has to be *placed* rather than tuned, the
static-unroll premise is gone so the remainder path is a different region structure from the main
body, and the budget becomes a distribution rather than a number. Read the acceptance signal on
the spread of worktile durations, not the mean.
