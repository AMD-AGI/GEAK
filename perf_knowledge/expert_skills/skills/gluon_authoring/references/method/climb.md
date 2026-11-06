# Climb — one coupled layer per round, against `champion_ms`

**What this stage decides.** Which layer of the backbone the next round works on, which mechanism inside
it, whether the round's change is kept, matured, or reverted, and — the part that is easy to get wrong
once you are inside the loop — **when to keep going, when to switch, and when a "ceiling" is not one.**
It is the only search this pack runs, and it is depth.

**When you are here.** On a **port** entry: after the anchor has been transcribed ([`transcribe.md`](transcribe.md))
and the transcription debt attributed and the parity gate disposed ([`recover.md`](recover.md)). On an
**incumbent** entry (the source was already explicit Gluon): immediately after the champion assertion
([`entry.md`](entry.md)) — transcribe / recover / parity are a defined no-op and this stage is the whole
run. Read `## Layer Backbone` at the **anchor**, before the first round; this stage re-reads it for the
order alone.

**Who climbs, in GEAK.** The deep_engineer running the single `deep_explore` direction: each climb round
here is one iteration of its own measure → profile → rewrite loop, recorded in its round log and
`checkpoint/` and closed through [`close.md`](close.md). The round's result is accepted by GEAK —
`verify_engineer` re-benchmarks it, the commit gate (`MIN_IMPROVE` = 2% over the cumulative best; a
measured noise band can only make that stricter) merges it, Director validates. The skill is advisory: it
names the layer, the reading behind the edit, and the evidence the round owes.

---

## Stage-Climb: one search kind, and it is depth

```text
per round:  profile -> analyze -> edit -> verify.  ONE attributable change, ONE layer.
            this IS `## 3. The round loop`, and it is the ONLY search this pack runs.
```

Stage-Gluon advances a **single evolving candidate** over the layer backbone. It is a sequential
hill-climb with a checkpoint and a backtrack — not a beam search and not a worker pool.

**No SWEEP, no BRANCH.** SWEEP and structural branching belong to the front end
([`front-end.md`](front-end.md)). This deep role preserves the inherited configuration and structure
while it climbs explicit layout, memory-path, pipeline, and register controls.

**WHICH layer, and in what order, is `## Layer Backbone` below** — what each layer decides, the reference
page that owns its mechanisms, the gating order, and the three pieces of evidence a layer is landed on.
It is read at the ANCHOR, not per round — layers 2–5 are visited twice in role, once to repay
transcription debt and once to climb — but this is the stage that consumes the ORDER, so re-read it once
when the climb opens. Two of its specifics are settled *before* layer 1 — layer 0 (parallel axis and
reduction landing) arrived decided in the champion, and the archetype row in `../workloads/index.md` that
`hw_budget.py --workload` already made you name says which layers have any surface at all on this kernel.
Climbing the wrong layer is not a slow round, it is a round spent where the archetype had nothing to act
on.

**Layer 4 is the one layer whose tooling the dials do not hand you, and above the parity gate the
re-injection tooling is not the climb at all.** [`profile.md`](profile.md)'s §3.1 names a tool for every
round-level reading; staging is a structural move rather than a dial, so it appears in none of them. The
climb's layer-4 move is **hand-written** staging in the order given under `### Layer 4 — the hand-written
pipeline order` below. What the shipped re-injection scripts cover is the **debt**, not the climb:
`scripts/pipeline_survey.py` classifies a plain-Triton kernel by the pipeline FORM it can exercise — only
the cross-iteration one is re-injectable, and it needs a `tl.range` whose `num_stages` resolves ≥ 2, so
run it before assuming the debt exists at all. `scripts/gluon_swp.py` runs plain's pipeliner over the
Gluon module in-process, without editing a Triton file; `scripts/patch_reinject.py` (and
`patch_async_reinject.py`) reach the same IR and their own docstrings name them the non-preferred route.
All of them reach *plain's* overlap and stop there, so they belong to [`recover.md`](recover.md) — as an
attribution **diagnostic** below the parity gate, or a **last resort** when hand-written staging cannot
reach parity — and their numbers are labelled **`injected`**, never a win. They are never used on an
incumbent (already-Gluon) kernel. Above the gate the win has to come from staging plain could not
express (`../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question
per tier`). A loop that never wanted a pipeline owes nothing here — settle that from the layer-4 cell
before any of it.

**The one genuinely exclusive choice, taken sequentially.** Layer 1.5's two non-base scheduling models are
mutually exclusive — compiler-interleaved (leans on AGPR) versus wave-ping-pong (keeps accumulators in
VGPR) — and they set warps/CTA, so they gate layers 4–5. Do **not** open two arms. Pick one **from the
profile**: the AGPR/VGPR split in A1 and the cadence verdict in B1 say which model the kernel's register
budget can actually sustain. Climb it to a wall, and only then, if the wall is the model's own
(accumulator pressure, or an interleave that will not materialize), go back and try the other. A
sequential re-try costs one direction's depth; two shallow arms cost the answer.

**Structure and resweep are not on this pack's menu.** If the inherited structure or configuration is
suspect, return `structure_suspect` or `resweep_request` with evidence in the deep_engineer's result and
stop that line. tech_lead decides whether to turn it into the next round's direction — a re-tune in GEAK's
own plain rounds, or a fresh `deep_explore` direction on a new structure (a tile retune belongs to GEAK's
tuning, not to this skill).

## Layer Backbone

Advance one coupled layer at a time; the *direction* of each layer is workload-agnostic, only the
specifics differ. Mechanism detail is lazy-loaded from the cited reference.

**This sits right after the budget because that is where its input comes from.** [`budget.md`](budget.md)'s
`hw_budget.py --workload` already made you name the archetype, and the archetype is what decides which of
the layers below has any surface on this kernel. Read this section before `## 2.` and before the first
round — not when the climb opens. The layers are visited twice in role (below), so the first read is at
the **anchor**; Stage-Climb re-reads it for the order alone.

Every layer below is an ISA-level claim: an explicit layout, a `ds_read` width, an async path, an MFMA
operand shape. `../hardware/isa-mechanisms.md` is where those mechanisms are stated for this
architecture. Read it when a layer's claim is about what the hardware DOES, not about what a counter
reported — that is the difference between a mechanism and a symptom, and this backbone is made of the
former. When the distilled value is not enough, **on gfx950 the primary source is already shipped**:
`kernel_workflow/scripts/kernel_tools/gfx950_isa.py` serves all 13 committed ISA chapters offline —
`search <term>` queries layout, encoding and facts at once; `layout`/`locate` give the lane/VGPR mapping
behind an explicit Gluon layout; `encoding <instr>` adds the chapter-12 pseudo-code; and `errata` names
the two traps that bite here — the 85 VOP3 opcodes the ISA prints 64 too high, and the `DS_READ_*_TR_*`
transpose loads whose lane mapping the document never states. (Reachable either directly or as
`layout_facts.py gfx950 <subcommand>`.) For other architectures (the gfx942 downgrade included),
`kernel_tools/hw_sources.sh` fetches the XML / calculator / PDF tier instead.

Layer 0 (structure: parallel axis and reduction landing) is **not here** — it is the front end's, and it
arrives already decided in the champion. This backbone starts at the anchor.

**The layer DIRECTIONS are workload-agnostic; the specifics are not, and two of them are decided before
layer 1.** Which tile the archetype wants, and which of the layers below has anything to act on, come from
`../workloads/index.md` keyed by the archetype `hw_budget.py --workload` already made you name — it prints
the citation next to the ceiling. Read that row before the anchor, because it is what tells you e.g. that
a `prologue` kernel has no hot loop, so layer 4 and its dependent 4+ have no surface while 2, 5 and 6
still do (`../workloads/prologue.md ## Which layers act when there is no loop`), that `moe` reaches the
small-M regime per expert rather than per launch, or that `linear_attention` is gated by
register-resident state and not by any capability cell. Getting the archetype wrong is not a slow round,
it is a whole run spent on the wrong layer.

| layer | what it decides | reference |
| --- | --- | --- |
| 1 | transcription anchor — explicit layouts, correct-but-slow | [`transcribe.md`](transcribe.md) |
| 1.5 | scheduling model, and there are **three**, not two: `authored_stage` (you place the commit groups — stock, one resident wave is enough, and the base the other two sit on), `compiler_interleave` (leans on AGPR; build-pinned passes), `inter_wave` wave-ping-pong via `warp_pipeline_stage` (keeps accumulators in VGPR; Gate 0 is **`num_warps >= 8`** — at 4 it is silently absent, not refused — plus 2 waves/SIMD **and** a stage body free of waits). Sets warps/CTA, so it gates layers 4–5. In 3.8.0 the upstream coexec scheduler (`TRITON_HIP_USE_COEXEC_SCHEDULER`, default on) is part of the `compiler_interleave` picture; fork-only env vars are labelled as such where they are cited | `../tile-programming/scheduling-model.md`; the wave-ping-pong mechanism, its compile-time constraints and version gates: `../tile-programming/warp-pipeline.md` |
| 2 | memory path: `buffer_load` → async direct-to-shared (gfx950: `async_copy` + `commit_group`/`wait_group`, 128- and 32-bit direct-to-LDS widths) | `../tile-programming/memory-path.md`. gfx942 downgrade: async direct-to-LDS is 32-bit only and the destination needs `order=[1,0]`; a padded destination fails translation, so build swizzled first |
| 3 | LDS layout: padding / swizzle for conflict-free `ds_read` (gfx950: 64 banks, `ds_read_b64_tr` available; gfx942: 32 banks, no `ds_read_tr`). It also **gates layer 2's async form and therefore layer 4's**: direct-to-LDS needs each lane to make exactly one access of a native width, so a distributed layout covering more than the tile makes two and the copy does not lower — with the same `failed to translate module to LLVM IR` text an absent op produces, which is how it once got recorded as an architecture verdict (`../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)`) | `../tile-programming/layout-recipes.md` |
| 4 | pipeline: prefetch + multi-buffer, **hand-written** (order below). `gluon_to_ttgir` runs nine passes and the pipeliner trio is not among them, and `num_stages` is **dead** on the Gluon path in 3.8.0 — no pass consumes it, so it is a budget parameter / champion record, never a knob carried over from plain. **First settle whether this loop wants a pipeline at all** — a tile that lives in registers, an absent or fully unrolled loop, or a handful of MFMA per iteration makes staging gain-zero and cost-positive (LDS squeezes occupancy, a barrier is pure latency), and the move is source-level reordering into the dependency shadow. **Author the staging** — per-tensor depth and staggered chains are what re-injection cannot express. Re-injecting plain's pipeliner is **not a climb move**: it is a below-gate attribution diagnostic or a last resort in [`recover.md`](recover.md), its numbers labelled `injected` | `../tile-programming/pipeline.md` (the single definition of the order); API router: `../gluon/pipeline-reference.md` |
| 4+ | instruction scheduling: pace what the overlap created — the empty-asm scheduling fence (tied identity), `s_nop` placement, and the per-compile `llvm_fn_attrs` strategy (3.8.0 only — a JIT compile option, **not** a Gluon mechanism, gated on the kernel being ILP-starved and decided per kernel: `../tile-programming/llvm-fn-attrs.md`). It schedules WITHIN a structure and never creates one. The first two are inline asm, and the fence is the most-misread construct on this layer: it is a **compiler** liveness/motion boundary, not a memory fence, so it never replaces a `gl.barrier()` or a `wait_group` (`../gluon/inline-asm-reference.md ## Class 2 — scheduling control`) | `../tile-programming/instruction-scheduling.md` |
| 5 | slicing / register / occupancy: fit the joint VGPR+AGPR budget, zero hot-loop spill. **When slicing is exhausted there is one lever left that this page does not own** — a value living longer than it needs to, which the allocator pays for in accumulation registers, in scratch, or in an occupancy step. It is the same empty tied identity as 4+ doing a different job: constraining the allocator rather than the schedule. The entry signal is four numbers off the static `.amdgcn` and needs no GPU; the site it produces emits **no instruction**, so no profile will ever point at it, and no-hotspot-here is the expected appearance rather than evidence of nothing to do | `../tile-programming/slicing.md`, then `../gluon/inline-asm/deciding.md` **L0.7** — which stays locked until **L0.0** has found no headroom left to spend, because the usual cause of pressure is headroom the kernel gave away somewhere else |
| 6 | beyond the hot loop: XCD-aware PID remap, workgroup grouping for L2 reuse | `../workloads/attention.md` |
| 7 | matrix engine + low precision: MFMA continuity, tile shape (16×16 MFMA first on gfx950), and which matrix op this dtype is entitled to **at this shape** — the entitlement is per `(version, M, N, K)`, never per dtype, so read the op the loop issues rather than inferring it from fp8/fp4. gfx950: scaled MFMA / MXFP, OCP fp8; gfx942 downgrade: no scaled MFMA, fp8 is FNUZ | `../tile-programming/low-precision.md` |

**The numbering is an index, not a work order.** Three of these layers gate another, and taking a gate
out of order does not cost a round — it voids the layer underneath it:

- **1.5 before 4 and 5.** It sets warps/CTA, and its two non-base models (`compiler_interleave`,
  `inter_wave`) are mutually exclusive, so a pipeline depth or a register budget sized under one is void
  under the other. Stage-Climb takes that choice **sequentially**, never as two arms.
- **3 before 2's async form, and therefore before 4.** The one-access-of-a-native-width rule in the
  layer-3 cell. Layout is not one of seven peers here: on CDNA4 it decides whether the memory path layer 4
  wants is reachable at all, and its failure mode is a lowering error that reads like an architecture
  verdict.
- **4 before 4+.** Instruction scheduling paces a structure and never creates one.

The rest are ordered by what the profile names. Whether layers 6 and 7 have any surface at all is an
archetype question answered by the `../workloads/index.md` row above, not a layer question.

Layers 2–5 are visited **twice in role, once in order**: below the parity gate their job is to close a
named transcription debt (Stage-Recover, [`recover.md`](recover.md)), above it to buy something plain
never had (Stage-Climb). Same mechanisms, different claim — record which one a round is making.

A layer is landed on **three pieces of evidence**: budget consistent, profile delta positive, IR/asm
signal confirmed. An enabling step may be provisionally accepted on a flat delta and is confirmed only
when the combination net-beats the checkpoint (`## Attributing a change that only creates headroom` in
[`close.md`](close.md)).

### Layer 4 — the hand-written pipeline order

The order to reach is defined once, in `../tile-programming/pipeline.md` (`## Authoring the overlap
yourself (the climb default on the Gluon path)`); this is its summary as the climb consumes it. Each rung
is entered only when the one before has been measured and has not closed the bucket it targets.

1. **Register-level prefetch.** Issue the next iteration's global loads into registers ahead of the MFMA
   block that consumes the current one. No LDS cost, no barrier; the cheapest overlap, and on a
   register-resident tile often the only one that pays.
2. **Authored LDS ring.** Double/multi-buffer with `allocate_shared_memory` plus barriers. **gfx950
   (CDNA4) path:** `async_copy` direct-to-LDS + `commit_group` / `wait_group`, per-tensor depth and
   staggered chains, 160 KiB/CU to spend. **gfx942 downgrade:** synchronous staging
   (`buffer_load` → `local_store`); async direct-to-LDS only at 32-bit with destination `order=[1,0]`,
   a padded destination fails translation (build swizzled first), and 64 KiB/CU caps the depth.
   Vetted skeletons: `scripts/pipeline_examples_cdna4.py` (main) and `pipeline_examples_cdna3.py`
   (downgrade example).
3. **`warp_pipeline_stage` + the scheduling-model choice (layer 1.5).** Wave-ping-pong / staged warp
   groups on top of the ring — Gate 0 `num_warps >= 8`, 2 waves/SIMD, a waitless stage body. Because it
   changes warps/CTA, choosing it re-sizes rungs 1–2: take it as the sequential model re-try described
   in Stage-Climb, not as a parallel arm.

Then 4+ paces the result. Re-injection of plain's auto-pipeliner (`gluon_swp` / `patch_reinject` /
`patch_async_reinject`) is **below every rung**: only an attribution diagnostic under the parity gate, or a
last resort when the hand-written ring cannot reach parity ([`recover.md`](recover.md)), always labelled
`injected`, never on an incumbent.

### Inline asm: gated, not ordered

**One mechanism cuts across every layer, and it is gated rather than ordered.** Where the ISA can express
something the Gluon language cannot — an ordering the abstraction does not offer, a machine-state bit, a
wave-collective, a whole publish/consume protocol — the single door is `gl.inline_asm_elementwise`,
present on all four checked versions. It has no layer of its own because it is reached *from* layers 2,
3, 4, 4+ and 7, and it is the only construct here whose acceptance evidence is the disassembly plus a
determinism race-test rather than a counter delta. Read
`../gluon/inline-asm/classifying.md ## Axis M — mechanism: what the call is made of` **before writing
one** and `../gluon/inline-asm-reference.md ## Traps, ranked by how likely you are to hit one` before
editing one someone else wrote. **Do not read it at all** when the language already has the spelling
(`../gluon/inline-asm-reference.md ## Before you reach for it: what the language already emits`), when
the kernel is a pure elementwise epilogue, or when the
only reason is that a production file you were reading had one — adoption is a count, not a
recommendation (`../workloads/index.md ## Before any compiler-invisible-ordering lever`).

## 2. Reversed-intuition traps — read this once

Ten places where a strong model's *default* read of the profile is wrong. Everything else in the
bound-class reasoning you already do unaided; **these you do not**. Deep detail:
`../hardware/optimization-gotchas.md`.

| # | The tempting-but-wrong move | What actually decides |
| --- | --- | --- |
| 1 | Low MFMA util → "occupancy wall", climb the occupancy ladder | **Occupancy adequacy**, not util. ≥2 waves and not the VGPR/LDS limiter = adequate ⇒ the matrix unit is *under-filled*: **grow the tile**, accept lower occupancy. And occupancy is **one-sided**: a computed *loss* vetoes a change that spends VGPR or LDS; a computed *gain* argues nothing — measured, +28.6 % computed bought **+0.16 %**. Never motivate a sweep by a predicted occupancy gain; if you time one, pre-register a direction **and a floor**. The labelled points in both directions, and why they are not an exchange rate: `../hardware/planning-constants.md ## Occupancy is a one-sided criterion` |
| 2 | narrow-precision matmul reads MFMA util or FLOPs ~0 → "not compute-bound" | **Two claims here; keep them apart.** (a) The FLOP rows are **one per dtype** and every inactive one reads exactly 0.00 — a row-selection artefact, not a fact about your kernel. Select the row matching the matrix op you can see in the disassembly. (b) The stronger claim — that the scaled-fp8 pipe misses the *generic* MFMA counters — **measured false on ROCm 7.0 / rocprofv3 1.0.0 / gfx950**: generic MFMA instruction count equalled the dtype-specific count exactly, and busy-cycles equalled that count times the documented per-instruction cost, so a low util there means what it says. Use the unified engagement test (it is correct either way), but **re-test before inheriting (b), and version-scope any "structurally blind" note you write** — counter coverage changes, and a stale blindness warning teaches readers to discard a reading that has since become correct |
| 3 | L2 hit low → "locality gap", do an XCD / GROUP_SIZE_M remap | L2-low **+ coalescing-low is a symptom** of the access, not a reuse gap. Coalesce/grow first; remap is inert until the access is already coalesced |
| 4 | Full-drain `lgkmcnt(0)` + low MFMA util → "bandwidth-bound", add prefetch | At equal registers and occupancy, low util with full drains and no memory-unit stall = a **schedule/overlap gap**, not bandwidth. Which move follows depends on the STAGE: below the parity gate it is the named `lost_pipeline` debt, repaid by hand-written staging in [`recover.md`](recover.md) (re-injecting the pipeliner only sizes that debt, or is the last resort, ceiling plain parity, labelled `injected`); above the gate, author the ring — that is what production is built on, and it is the only route that expresses per-tensor depth. A **natively authored** (incumbent) Gluon kernel never had a transcription debt, so only the authored ring is on the table. Not a ceiling either way |
| 5 | Low-precision GEMM → "we're at N% of the MFMA roofline" | VALU% ≫ MFMA% means a **per-element convert on the operand path** dominates, and the FLOP roofline has no axis for it. Int4 unpack is one form; the commoner AMD form is an **fp8→bf16 upcast feeding regular MFMA**, which reads as a dtype detail and costs a VALU op per element — it also flips row 2's caliber, so read the two together. Cheapen or delete the convert; do not chase peak MFMA |
| 6 | Any prefetch / deeper-pipeline lever "should help" | **Occupancy gates it.** Removing exposed serial latency pays only at low occupancy; at 2+ resident waves it is a wash. Bound-the-win first ([`profile.md`](profile.md) `## Bound-the-win probe (before investing in a latency-hiding / pipeline lever)`) |
| 7 | The scheduler toggle is "just another knob to try" | The **in-tree** toggle assumes a pure MFMA→MFMA accumulator chain: with VALU between matmuls it emits **invalid IR**, not a slowdown, so default-skip it for attention-shaped ops. But scope that to the toggle — a **region-classifying** scheduler (the 3.8.0 coexec scheduler is one) routes matrix+VALU regions to a second, co-execution model and is transactional. Which one you have decides the answer (`../tile-programming/llir-codesign.md`) |
| 8 | A compiler-pass lever regressed → "the compiler is the ceiling" | **A compiler-realized lever exploits structure, it never creates it.** Verify in the IR that the paired source structure is there. If it is not, this is a kernel bug, not a compiler miss |
| 9 | `s_nop` in the loop → "exposed hazard, add unroll / occupancy" | **Three origins the listing cannot tell apart.** True for one the **compiler** inserted — the tempting read. A couple a kernel **deliberately places** at a stage head are a *phase* lever — shifting one wave's LDS burst off the other wave's 3-source VALU, which contends in the register file, not at the issue port; phase effects are non-monotone: sweep, do not bisect. And a bare `s_nop` ahead of a DPP / `permlane` / cross-lane op is an ISA-mandated wait state a hand-written asm block supplies itself, so the discriminator is the *next instruction* and deleting it is a correctness bug, not a round (`../tile-programming/llir-codesign.md ## Phase pacing: moving a burst instead of removing work`) |
| 10 | A schedule tool changed nothing → "at the schedule ceiling" | Check two silent skips first: is the loop's **matrix shape in the tool's cost model** (an unpriced shape drops the region without a word), and was the **declaration satisfiable** (one op that never issues makes the solver abandon the whole region, which looks identical). Both leave the assembly looking un-scheduled — which is exactly what a real ceiling looks like (`../tile-programming/llir-codesign.md ## Applicability gate: shapes the tool does not model`) |

## 3. The round loop

Each round: **profile → analyze → edit → verify.** One change you can attribute. If you cannot say what
you expect to move and which number will show it, you are not ready to edit.

**Every decision this skill makes is profile-guided.** The required instrument set is
[`profile.md`](profile.md) `### 3.1 Required evidence — the four dials, every round`, and the four groups
A/B/C/D must all be read — it is not a menu. Without them you are guessing at a black box: you cannot see
the register wall (A), what the instruction stream is actually made of (B), whether you are
bandwidth-bound (C), or whether a change was real (D). Reasoning from only the instruments you happened
to read — while the deciding one sits dark — is the most common way an optimization goes confidently
wrong.

So, as a hard discipline:

- **Gather the evidence required by the current claim BEFORE forming a hypothesis.** Establish a complete
  profile for the anchor, then repeat it on material source/config/boundary changes, evidence conflicts,
  ambiguous bounds, and close claims. Record targeted evidence between those triggers; no edit may rely on
  a stale or unexplained reading. The every-round minimum is the re-profile in `## Non-stopping rules`.
- **Every analysis and every edit must name the specific reading (A1 / B2 / …) that motivates it.** "grow
  the tile" is not a plan; "B1 says cadence is issue-bound and A2 says the tile has spare registers, so
  grow it" is. A lever with no reading behind it is a guess — spend the round reading instead.
- **A missing number is reported as missing, never read as a zero.** Several readings have a
  plausible-looking wrong source that silently returns 0 (A3 is the sharpest such trap). A missing
  evidence layer does not produce "no answer" — it produces a *confident wrong one*, with
  plausible-looking levers attached. If a reading is unavailable, say so next to any conclusion that leans
  on it, and get it before you trust the conclusion.

**Config knobs are not rounds and they are not this pack's job** — the champion arrives with its config
already swept and pinned. Re-sweeping it here is both a re-run of settled work and a correctness hazard:
`BLOCK_*` is coupled to the recovered layout family (`warps_per_cta` ties mma / dot / shared / global), so
changing a tile means re-recovering the whole layout set. A tile retune is a `resweep_request` (returned to
tech_lead, which hands it to GEAK's plain tuning). Before trusting any version-sensitive knob, `scripts/probe_levers.py --all` (no GPU)
says whether it is `live`, a `dead-declaration`, or `absent` on **this** build.

## Non-stopping rules

- A kept win does **not** end the task. Refresh the evidence and pick the next open layer.
- One layer's win does **not** close another layer without a refreshed profile.
- **Re-profile every round.** Last round's numbers describe a kernel that no longer exists — an edit can
  move the bottleneck stack, and reasoning from a stale ranking is the most common way to spend a round
  on a resource that stopped mattering. This holds inside GEAK's loop too: GEAK driving the rounds does
  not waive the per-round evidence.
- **Continue on a scoped failure.** A compile / correctness / perf failure in one mechanism does not
  abort the direction: classify it ([`triage.md`](triage.md)) and descend the ladder below. For a runtime
  deadlock / crash / wrong result after a clean compile, run the async-handoff worksheet
  ([`triage.md`](triage.md)) before reclassifying.
- A finished stage, a measured lever, a passed diagnostic, or a degraded evidence layer is **not** a stop
  condition ([`close.md`](close.md) `## Stop conditions`).

## A direction matures over rounds — do not revert it on round one

A **structural** direction (defuse, grow-tile, packed-atomic, a pipeline rebuild) cannot win in a single
round: the enabling step usually costs something before the payoff arrives. Decide the direction's
iteration budget before you start, scaled to how much has to be rebuilt: a knob-shaped direction
converges in a few rounds, a structural one needs several more (its enabling step, then the layers that
pay it back). While it is inside that budget treat a flat or slightly negative step as **maturing, not
failed**. Finalize the revert only once the budget is spent and the direction still has not net-beaten
the checkpoint.

Name the direction and its class *before* editing, and count its rounds. Without that, a multi-round
direction looks like the same lever failing repeatedly and gets abandoned one round before it would have
paid. (GEAK's `max_no_improve` / `progress_delta` launch args are what let a maturing direction
survive the loop — [`index.md`](index.md) G0.)

## Fallback ladder (escalate scope only after the finer scope is exhausted)

1. **Next mechanism, same layer.** On a stall, or a crash-loop under threshold (same error <3×: roll back
   the newest change, re-read the source, change edit strategy, re-verify the environment), switch
   mechanism — layout vs config vs schedule. A regression that looks like a **resource cliff** is retried
   once *combined with a register-relief step* before being recorded negative; cliffs are frequently a
   pressure problem wearing another mechanism's clothes.

2. **Next open layer.** When a layer reaches a soft ceiling (≈3 profiles within 1% across ≥3 tuning
   steps), mark it closed or blocked **with the evidence** and move to the open layer with the most
   headroom.

3. **Re-rank the bottleneck stack — this is the rung people skip.** When the binding resource hits its
   floor, the bottleneck list is a **stack to descend, not a single target**. Re-read the ranked census and
   run the loop on #2 and #3 before declaring a ceiling or escalating. *A kernel at the roofline on
   resource #1 is now bound by #2.* An instruction audit that concludes "VALU-volume bound, irreducible"
   must still read the rest of the census: a large feed or accumulator-traffic bucket sitting in the same
   profile is the real #2, not a ceiling.

4. **Escalate the tier.** Only when every layer is closed or blocked **and every ranked resource has had
   its loop**, and a residual gap with explicit-control headroom remains: plain → Gluon, or Gluon → LLVM
   if sanctioned (`../tile-programming/compiler-contract.md`, `../tile-programming/llvm-codesign-handbook.md`).

5. **Deliver the direction's best and move on.** A direction with blocked layers but partial wins still
   delivers. Keep its best result and take the next-priority direction.

**Scoped-ceiling handoff.** If the blocker needs something outside your scope — a compiler change without
sanction, a build/version switch, an oracle/data/repo gate, a wrong result with no in-scope fix
([`triage.md`](triage.md)) — record the handoff, keep the best in-scope result as the fallback, and
continue any other open work. Run the cheapest falsifying probe **first**: "assumed infeasible" is not an
acceptable terminal state ([`index.md`](index.md) `## Non-Negotiable Rules`).

## Self-monitoring

Track these yourself; they tell you which rung of the ladder you are on.

| signal | what it means | response |
| --- | --- | --- |
| **stall** — several steps, no net improvement | the mechanism is not moving the bucket it targets | rung 1: switch mechanism |
| **soft ceiling** — ~3 profiles within 1% | this layer has converged | rung 2: close it with evidence, move on |
| **crash-loop** — same error ≥3× | the edit strategy is wrong, not the lever | roll back, re-read the source, change strategy |
| **resource cliff** — a large regression at one config step | usually register/LDS pressure crossing a threshold | retry once with a relief step before recording negative |
| **binding resource at its floor** | #1 is done; #2 is now binding | rung 3: re-rank. Do **not** call this a ceiling |

The budget counters behind `timeout` (warn / wrap-up / timeout at 70 / 85 / 95 % of N) are recorded per
[`records.md`](records.md) (`## 6. Self-Monitoring / Insight Buffer`).

**Verify the edit landed before you read the timing.** The lever's expected ISA or counter signal must
have moved. A flat result with an unconfirmed signal means "my change did not take effect", which is a
different problem from "this lever does not help here" — and `kernel_tools/probe.py` answers it in
seconds, without a profiler.

**When the kernel serves a range, keep or revert on the SERVED number.** The comparator already carries
one latency per served case — `per_case` on a swept `plain_best_config.json`, `served_range` on a
`plain_champion.json` — so a round has the per-shape effect of its edit without measuring anything extra.
Judge it with `scripts/served_envelope.py`: the served speedup is a weighted HARMONIC mean, because time
adds and rates do not, and an arithmetic mean of per-shape speedups overstates a win. Two consequences
worth stating, because a single-point round cannot see either:

- a round that improves the pinned case and regresses another may be a net loss on the served mix, and
  reading only the pinned case records it as a keep;
- `MV_k` — the served value per unit of time removed at case `k` — is what says which case the NEXT round
  should attack. Ranking cases by their raw microseconds answers "which is slowest", which is almost never
  the same question.

If the comparator carries no served table (the sweep declared no case order), say so in the round record
rather than quoting a single-point delta as if it were the served one.

**A delta near the noise floor is decided by how it was measured, not by its sign.** Two sequential runs
differ by drift, thermal state and whatever else held the box in between, so a 1–3% reading taken that
way is a coin flip wearing a decimal point. Two instruments, two jobs ([`benchmark-hygiene.md`](benchmark-hygiene.md)):

- **Screening (search only):** `scripts/ab_bench.py --permute` interleaves the sides in one session with a
  control arm and reports the paired spread. Reach for it whenever the delta is inside the band the
  sweep's own repeated readings showed — recorded in the comparator — to decide which candidate is worth
  confirming. Screens at that margin routinely fail confirmation, and one has come back with the sign
  reversed.
- **Acceptance (the keep):** GEAK's timing (`e2e_workflow/scripts/harness_lib.py`: CUDA events with
  per-sample sync, read-evict flush, median, a fresh process per leg, baseline measured in the same
  window). GEAK's `verify_engineer` / `measure_legs` produce the number and the commit gate requires
  `MIN_IMPROVE` = 2% over the cumulative best. A measured noise band wider than 2% makes the verdict
  stricter; it never relaxes it. Do not record any keep whose margin you would not bet the next ten rounds
  on.

## 5. AMD-only levers: from a bubble to a tactic

`../hardware/lever-cards.json` holds the levers that are **AMD-specific** — an ISA or microarchitectural
fact that does not exist on NVIDIA or means something different there — plus any lever that is the named
destination of a bucket the profiler actually reports. Generic GPU tactics are deliberately absent: you
already know those.

This pack ships the cards this tier can **express** (`expressible_in` includes `gluon`). A plain-only card
is not missing — it is the front end's, and it is already disposed in the champion.

**Index them by the bound YOUR profile named:**

```bash
python3 scripts/lever_index.py --bound <class> --arch gfx950 [--sub <sub>] [--occupancy-waves N]
```

Chip-absent levers are excluded (arch availability is measured fact), gating-law-forbidden ones are
flagged, and the rest carry the `gate` to check against your own read. Experience, not verdict.
`--list-bounds` prints the bound vocabulary. The gating laws themselves are
`../hardware/bound-class-signals.md ## Lever gating laws (single source of truth)`.

The rollup's bucket lines carry the cards registered against each bucket (`bubble_bucket`). That pointer
and `lever_index.py` are the routed paths into the catalogue. Use it **both ways**: before inventing a
lever, check the bucket; after inventing one, check it anyway — the catalogue records what has already
been tried on AMD, which is the one thing the profile in front of you cannot tell you.

A card that does not fit is a named N/A, not a spent round. **A ranked bucket with no card behind it is a
catalogue gap — report it.**

## Record what you decided

One entry per round: what you profiled, what the evidence said (the A1/B2/… readings named), what you
changed and why, which role the round plays (`recovery` against a named suspect, or `climb`), the measured
before/after, and keep-or-revert. Template: [`records.md`](records.md) (`## 1c. Per-round log entry —
written as you go, not reconstructed` and `## 5. Per-Layer Round Ledger (core)`). This log is the
deliverable a reviewer actually reads — the kernel shows *what* you ended with, the log shows whether the
result is attributable. The deep_engineer keeps it in its OUTPUT_DIR (`decision_log.md` or the round log
in `report.md`); the per-round summary rides GEAK's `worker_result` notes.

## Optional within-layer fan-out (`breadth_enabled: true` only)

**In this pack this section is inert** — it ships `breadth_enabled: false` (`scripts/pack_facts.json`),
and the track is one deep_engineer in one `deep_explore` direction ([`orchestration.md`](orchestration.md)
`## Parallelism: bounded measurements only, never a fan-out of the port`). An undecidable within-layer choice
is resolved sequentially in the same loop — narrow it with a compile-only probe (`probe.py`) and a screening
A/B (`ab_bench.py`) on the engineer's own GPU, then confirm the keep with acceptance timing. The coupled
backbone (memory → LDS → pipeline) is never split across engineers.

## Sources

Merged from: `references/phases/layer-loop.md` (all sections); `references/method-reference.md`
`## Layer Backbone`, `## 2. Reversed-intuition traps — read this once`, `## 3. The round loop`,
`### Stage-Climb: one search kind, and it is depth`, `## 5. AMD-only levers: from a bubble to a tactic`;
the compressed copies of the same sections in `tile-programming-gluon.md` (distinct facts unioned: the
one-sided occupancy measurement in trap 1, the `num_warps >= 8` silent-absence gate on layer 1.5, the
three-origin wording of trap 9); `tile-programming-gluon.md` "Standing references" paragraph on
`lever_index.py`.

Conflicts resolved (contract rules):
- **Pipeline priority.** Layer-4 cell, trap 4 and the Stage-Climb layer-4 paragraph previously routed a
  below-gate `lost_pipeline` reading to re-injection as *the* repayment. Rewritten: hand-written order
  (register prefetch → authored LDS ring → `warp_pipeline_stage` / scheduling model) is the climb and the
  recovery default; re-injection is demoted to a below-gate attribution diagnostic or last resort,
  labelled `injected`, never on an incumbent (detail in `recover.md`).
- **`num_stages`** declared dead on the Gluon path in 3.8.0 (budget parameter / champion record only).
- **Noise-floor keep.** `ab_bench.py` "interleaves the two sides in one session … turns it into a keep"
  rewritten: ab_bench is search/screening only; the keep comes from GEAK acceptance timing
  (`harness_lib`, fresh process per leg) and the 2% commit gate; a measured band may only tighten it.
- **Re-profile every round** stated as binding inside GEAK's loop too (drops the GEAK skill.md rule "no full
  profile before the port lands" for the climb).
- **Tile retune** = `resweep_request`, returned to tech_lead for GEAK's plain tuning.
- Run-mode split (GEAK-embedded vs the pack's own `toolctl` spine / `gluon-direction` agent) removed: the
  climb is the deep_engineer's loop inside GEAK's Optimize phase; the parallel within-layer screening option
  (one GPU per worker, geomean merge) dropped — this pack never fans out.
- gfx950-first facts added to layers 2, 3, 7 with gfx942 downgrade notes; coexec scheduler (3.8.0,
  default on) named on layer 1.5 / trap 7.

Dropped: nothing. Pointer paths updated: `failure-triage.md` / `debug-async.md` → `triage.md`;
`experiment-records.md` → `records.md`; `orchestration.md` → `method/orchestration.md`;
`method-reference.md ## Stop conditions` / `## Non-Negotiable Rules` → `close.md` / `index.md`;
`scripts/gfx950_isa.py`, `hw_sources.sh`, `probe.py` → `kernel_workflow/scripts/kernel_tools/`.
