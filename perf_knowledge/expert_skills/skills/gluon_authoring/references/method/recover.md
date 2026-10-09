# Recover: pay the transcription debt by hand, then clear the parity gate

**What this stage decides.** How far below the champion the faithful anchor sits, **who owes the gap**
— exactly one or more of `lost_pipeline` / `lost_layout` / `lost_RA` — and whether the debt has been
repaid to the run's declared parity criterion. Until it has, every round is a `recovery`, never a win,
and the climb does not start.

**When you are here.**
- **Mode A** (converted anchor): immediately after the equivalence gate in `transcribe.md` passes
  (`PASS` or `RECONCILED`). Always.
- **Mode C** (re-entry): only if the interrupted run carried its debt unpaid (`parity_unreached`, or a
  `parity_gate.py` exit 2 it never closed) — re-run the gate before the first new climb, not after.
- **Mode B** (incumbent, already explicit Gluon): **not a stage.** Nothing was transcribed, so there is
  no debt; `parity_gate.py` must not be run as a gate (it returns a vacuous CLEARED). The tool remains a
  diagnostic for a regression you introduce mid-run — say in `decision_log.md` which you are doing
  (`entry.md`, mode B).

**How the debt is repaid, in priority order.** By hand, per suspect: `lost_layout` by re-recovering the
layout; `lost_pipeline` by an **authored** pipeline (register prefetch → authored LDS ring →
`warp_pipeline_stage` + scheduling model); `lost_RA` by slicing / register budget. Re-injecting plain's
auto-pipeliner is **lowest** — a diagnostic that measures the `lost_pipeline` debt, or a last resort when
the hand-written pipeline cannot reach parity — and its numbers are labelled `injected`, never a win,
never on an incumbent (last section of this page).

Paths: `$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`; `$KT` =
`kernel_workflow/scripts/kernel_tools`. Examples use `--arch gfx950`; gfx942 differences are downgrade
notes.

---

## Stage-Recover: pay the transcription debt, then satisfy the parity gate

**A faithful anchor sits below the champion by construction.** `gluon_to_ttgir` does not run the
software pipeliner (`add_schedule_loops` + `add_pipeline`) or the automatic block-ping-pong that plain's
`make_ttgir` does — on **any** upstream version (3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 all checked) — so plain's
`num_stages` overlap does not survive the transcription and the anchor loses the compiler's scheduling
even when every layout is recovered perfectly. On the Gluon path in 3.8.0 no pass consumes
`num_stages` at all: on this tier it is a budget parameter and a champion record, not a knob to carry
over and tune. That gap is a **debt you knowingly took on**, and the run's first job is to pay it back
with the mechanism that actually owns it, not to bury it under new optimizations.

**But the debt is often zero, and paying a zero debt is how a round gets spent for nothing.** Two ways it
comes out zero, both common enough that assuming a debt is the wrong default: a champion that compiled
at `num_stages=1` has no overlap to lose at all, and a champion whose shipped `num_stages` is itself a
*pessimisation* would have the loss recovered along with the pipeline. That is why `lost_pipeline` is
sized with a measurement — `plain@ns=1` at the champion's own config (§2.1) — before anything is built.

**The pipeline is only one of the reasons**, and a pipeline debt is common but far from universal. Naming
the owner before acting is what stops a round being spent building a pipeline into a kernel whose gap was
somewhere else entirely.

**Not running by default is not the same as unavailable — and re-injectability is not priority.** The
pipeliner passes are in `libtriton` on every version checked and can be re-injected into the Gluon path
without a rebuild. But re-injection reaches *plain's* overlap and stops there: its ceiling is plain
parity, its measured net varies by kernel and version in both directions, and it cannot express
per-tensor depth, staggered chains or anything plain could not. So the repayment of a named
`lost_pipeline` debt is **authored**, and re-injection is ranked last (final section). A loop with no
`tt.dot` is where re-injection is *unavailable* anyway. And the pipeline layer's own first question comes
before either — whether this loop wants a pipeline at all
(`../tile-programming/pipeline.md ### First: does this loop want a pipeline at all?`; a large fraction of
surveyed production source files carry no pipelining construct at all, and that is the right answer
rather than an unfinished one). The route table is
`../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question per tier`.

**Attribute the gap before closing it, from the compiled artifacts rather than from a story** (§1–§2),
then repay each suspect by hand (§3), then clear the gate (§4).

**Both gates are mandatory by instruction, not by interception.** This pack runs the `tool_first`
regime: it ships no per-round write gate, no bound-class decision tree and no JSON round contract, so
nothing stops a round that skipped `champion_gate.py` or ignored a `2` from `parity_gate.py` — the stage
cards carry no `entry_gate` and the four middle stages require no artifact. What makes the two gates
binding is that the close is audited against them: run them, keep the output, and reference it. A close
whose numbers cannot be traced to a passing champion gate and a parity disposition is the failure
discovered at close, and that is the whole enforcement mechanism. The deep_engineer's round log must
carry the command output (GEAK gate G2); "the precondition holds" is not a gate.

Two failure modes this stage refuses, both of which read as progress:

- **Climbing on an unattributed defect.** Optimizing a kernel that is down on a mis-recovered layout
  tunes *around* the defect; the residual is then mis-attributed for the rest of the run, and the run can
  report a clean-looking improvement while still losing to plain. A run has gone to the climb from a
  **0.715×** anchor, reported 1.19× against that anchor, and closed at **0.85×** against the champion.
- **Booking the recovery as the win.** "3× faster than the anchor" is a statement about the anchor, not
  about Gluon. The comparator is `champion_ms`, always.

If parity is not reached within the budget you allotted the stage, record `parity_unreached` with the
attributed residual split and carry that caveat on every later number. A residual attributed to a
mechanism Gluon genuinely cannot express at parity is a **handoff-back signal** — say so with the
evidence rather than spending the climb budget out-running it (§5).

---

## Two baselines

- `plain_baseline` = original plain-Triton perf = the **target line** to beat.
- `gluon_anchor` = the faithful transcription = the **working baseline** for recovery and the layer loop.

**In this pack these two already have names, and you must not recompute either.** `plain_baseline`
**is** `champion_ms` from the inherited `plain_champion.json` — the tuned, pinned, sha-verified plain
number the front end published — so the definition below ("the best plain you can produce this run") was
satisfied upstream, not here. Re-measuring a plain number in the deep pack and calling it
`plain_baseline` silently replaces the comparator, which the non-negotiable rules forbid (`index.md`).
(You *do* re-measure the champion's number on your own box at entry, to assert it — `entry.md` — and you
measure `plain@ns=1` as a **control**, §2.1; neither replaces `champion_ms`.) `gluon_anchor` is the
transcription anchor, and a delta against it is a **recovery**, not a win, until the parity gate holds.

**Baseline identity (what `plain_baseline` must be).** The acceptance target is the **live production
baseline measured at the production boundary** — the current shipping path as actually served (eager or
CUDA-graph), not a frozen/strawman re-implementation, a kernel-only number for a graph-served op, or a
default-config build when production uses tuned configs (`benchmark-hygiene.md`, same-knob baseline and
tuned-config audit). Record the baseline's identity + boundary alongside its numbers; a win measured
against the wrong baseline identity or the wrong boundary is not a win.

**`plain_baseline` = the BEST plain you can produce, not the shipped default.** Before any speedup is
claimed, the plain comparator is swept to **its own best config** (tile / `num_stages` / `GROUP_SIZE_M` /
split-K / `matrix_instr_nonkdim`). A win over an **un-tuned / default-config** plain is **invalid** — a
shipped tuning table may be `tuned=False` and badly under-fill the machine, so "beats the default config"
measures the config gap, not the Gluon win — and so is a parity ratio measured against one. Record the
comparator's tuning knobs + sweep range next to its number. In this pack that sweep is the front end's
(in GEAK, GEAK's plain rounds), asserted by `champion_gate.py`.

**When a fair plain A/B is not constructible.** Sometimes a warm-start Gluon champion and plain use
**different layouts/semantics** (e.g. a packed / split-KV vs a flat KV layout) so a like-for-like plain
target line **cannot be built**. Then do NOT compare to a strawman: use the **production Gluon config** +
an **external ASM/reference ceiling** as the baseline instead, and **state this substitution in the
contract** (`comparator` field + `baseline_status`). Record it as a `tie_kept` / `unblock` per the
win-type rules (`close.md`), never as a fabricated speedup.

---

## 1. Re-profile and recalibrate the anchor (mandatory)

Immediately re-profile the anchor (`profile.md`; profiler entry is GEAK's
`kernel_workflow/scripts/profile_kernel.sh`, under `gpu_lock.sh`) and **recalibrate the theory and
budget** (`budget.md`, re-calibrate): now that layouts are explicit, the as-built budget can be truly
decomposed (`R_acc/R_operand/R_prefetch`, exact LDS bytes, stage structure), and effective peaks come
from the anchor's trace. The calibrated budget is the reference for recovery and the layer loop. Read
`$KT/probe.py measure --dir ir/anchor/` — both occupancy limiters — before any timing (G3).

---

## 2. Attribute: which of the three suspects owes the gap

```bash
python3 "$SKILL/scripts/parity_gate.py" --champion-ms <C> --anchor-ms <A> \
    --champion-asm ir/champion/champion.amdgcn --anchor-asm ir/anchor/anchor.amdgcn \
    [--champion-ttgir ir/champion/champion.ttgir --anchor-ttgir ir/anchor/anchor.ttgir] \
    [--champion-lds <N> --anchor-lds <N>] \
    [--champion-arch gfx950 --anchor-arch gfx950] \
    [--champion-workgroup-size <threads> --anchor-workgroup-size <threads>] \
    --threshold <declared, default 0.95> [--json parity.json]
```

It **exits 2** while `champion_ms / anchor_ms` is under the threshold, and names which of three suspects
owes the gap; exit 0 = CLEARED. `--anchor-ms` is the **current** Gluon number, not only round 1's.

Both artifact pairs are `$KT/dump_ir.sh --variant <name>` output, which lands at
`<out>/<variant>/<variant>.{ttgir,amdgcn}` — so dump the champion and the anchor under their own variant
names and the paths above are literal. Pass `--*-lds` from the Triton cache metadata's `shared`, which
`dump_ir.sh` writes beside them (`meta_*.json`): the `.amdgcn`'s own `LDSByteSize` is a **structural 0**
on Triton kernels, and without it the tool says the LDS half went untested instead of clearing it
silently. Occupancy is compared as **min(register term, LDS term)** per arm, and the gate **names the
term that bound each arm** — that name routes the owner. `; Occupancy: N` alone is the register term,
and an anchor that lost a wave to a materialized staging buffer moves no register count, so a
register-only comparison is silent on the most common `lost_layout` mechanism. Every input the LDS term
needs (arch, group-segment bytes, workgroup size) is **refused rather than defaulted** when the artifacts
do not state it — pass them. The register cliff is measured on `.amdhsa_next_free_vgpr` (ArchVGPR + AGPR
share one 512-register file per SIMD), not on `num_vgpr`. An anchor **faster** than the champion clears
the gate and is told to attribute that too rather than pocket it.

Two independent reads on each suspect: what the gate sees in the compiled artifacts, and what the
profile (the floor probe plus B1/A1, `profile.md`) sees at runtime. They should agree; when they do not,
that disagreement is the finding.

| suspect | the gate's signal (artifacts) | the profile's tell (B1/A1) | repaid by (§3) |
| --- | --- | --- | --- |
| **`lost_pipeline`** (usually dominant where it exists) | the champion's TTGIR carries `ttg.memdesc_index` / `ttg.local_store` / `num_stages > 1` / a peeled prologue and the anchor's does not. **`iter_args >= 2` is not evidence** — every accumulator loop, including any online-softmax kernel, has it, and `ttgir_bridge.py` reports INCONCLUSIVE for the same reason. Size it with `plain@ns=1` (§2.1): the anchor landing on `plain@ns=1` < `plain` is the debt | no prefetch/overlap; MFMA stalls on a full `vmcnt`/`lgkmcnt` drain | the **authored** pipeline (layer 4), below this gate only up to parity; a loop that wanted no pipeline owes nothing here |
| **`lost_layout`** | a load-width or LDS-op histogram narrowed (`dwordx4`→`ushort`, `ds_read_b128`→`ds_read_u16`), or `shared` bytes/WG crossed an LDS/CU divisor, or occupancy fell with the anchor's binding term the LDS one | the layout-diff was not clean, or the `ds_read` interval is off the conflict-free steady state | **re-recover** the layout from the IR (layer 3); never hand-derive a basis |
| **`lost_RA`** | same instruction multiset, and the allocator serialized it anyway — VGPR rose (especially across a wave threshold), spill appeared, occupancy fell with the anchor **register**-bound, or the number of **distinct address registers** feeding the LDS read burst collapsed. The tell is an address **rematerialized** into a register immediately above the `ds_read` that consumes it, which puts every read behind a WAR hazard on that one register — no layout-equivalence check can see it (*equivalent layouts, equal counters, unequal address-register pressure*). Read the `ds_read` **operands**, not just the count | hot-loop `scratch_*`, or an AGPR/VGPR split the transcription did not preserve | slicing / register budget (layer 5) — an LLVM-stage concern, never in the TTGIR |

**Signatures that are sub-cases, not extra suspects.** The gate emits exactly `lost_pipeline` /
`lost_layout` / `lost_RA` and nothing else, so use those names verbatim in the ledger — a fourth name
makes the attribution unmatchable against the gate's own output. Earlier GEAK tables carried more rows;
they fold in as follows:

| older row | how it shows | belongs to | repair |
| --- | --- | --- | --- |
| **lost vectorization** | anchor ≈ 0.5× or worse, and the asm load-width histogram shifted (`dwordx4` → `ushort`) — a `convert_layout` folded backwards into the load | `lost_layout` | **re-recover**: the faithful form gives that `local_alloc` its own `allocate_shared_memory` with the recovered `*_SMEM` layout, which is what pins the wide load (`transcribe.md` §3). Not a hand-derived fix |
| **LDS budget** | every layout verifies and the anchor is still slower, because `shared` crosses the LDS/CU divisor and costs a workgroup per CU. **The divisor is arch-specific — 160 KiB on gfx950; gfx942 downgrade: 64 KiB** — so pass `--arch` and never carry the verdict across generations | `lost_layout` (the shared-bytes half) | compare `recover`'s `LDS:` line (an upper bound) and `probe.py measure` against plain's; `verify` is blind to allocation size. The pass-through/staged `local_alloc` choice lives here (`transcribe.md`, "Not every `ttg.local_alloc`...") |
| **lost schedule** | the instruction *multiset* matches plain exactly, but the waits do not (one kernel: 21 relaxed `lgkm` waits vs plain's 10) | `lost_RA` family (same multiset, serialized differently) | reorder the body toward plain's program order; not a layout, pipeline, or selection problem |
| **"lost-interleave"** | MFMA clumping behind memory | `lost_pipeline` (a *symptom*: no overlap was built) | paid down by the pipeline layer |

**The usual order that follows**: rebuild the pipeline (layer 4) where `plain@ns=1` says the debt is
real, then re-recover the layout (layer 3), then slicing/RA (layer 5). The **largest attributed bucket
sets which layer opens first** — directed, not blind.

### 2.1 The `plain@ns=1` control — sizing `lost_pipeline` before paying it

This is the measurement that decides whether there is a pipeline debt at all. Re-run the *plain* champion
with its pipeline turned off, at its own config, and compare three numbers:

| reading | what it means | do |
| --- | --- | --- |
| `plain@ns=1` ≈ `plain` | the champion was never pipelined; there is no overlap to lose | **no `lost_pipeline` debt.** A faithful anchor should land **≈1.00** here, and anything well below that is a transcription defect, not a debt — go to `lost_layout` / `lost_RA` |
| `plain@ns=1` **faster** than `plain` | the shipped `num_stages` is a *pessimisation* | **do not recover it** — recovering it recovers a negative. Not a rare case: a library kernel whose wrapper passes no `num_stages` inherits the AMD default of 2, which nobody chose for that body. Report it to the kernel's owner |
| your anchor ≈ `plain@ns=1` **<** `plain` | the entire gap is the missing pipeline, and no layout work will move it | this is the real debt — repay it with an authored pipeline (§3.1) |

**On the middle row, read `spill=` before you write the report, because a pessimising depth is usually a
register wall and then the bug is the tile rather than the depth.** Measured on one card with only the
tile varying: the wide tile pessimised at `ns=3` (1.45× worse than its own `ns=1`) while the narrow tile
gained, and the discriminator was the spill column — the losing arm pinned at the 256-VGPR cap and
spilled hundreds of bytes per wave *inside* the loop, the winners spilled nothing. `WGs/CU by LDS` was 1
for both and could not separate them. This also means **the sign of a debt does not transfer across
generations for arch reasons alone**: a narrower MFMA shape needs more instructions and keeps twice the
dot-operand registers live per K-tile, so the same source config can sit on either side of the wall on
two gens.

**Find out WHICH `num_stages` knob your champion actually uses before you flip one — there are two, they
are not equivalent, and turning the wrong one manufactures a "no debt" verdict.** Read plain's
`num_stages` off the **loop**, not the launch:

| the loop | what sets the depth | flipping the launch arg |
| --- | --- | --- |
| carries `tl.range(..., num_stages=N)` | **the annotation, outright** | **inert** — every launch depth compiles byte-identically |
| bare `range`, **with a dot** | the launch argument | works; it is the only knob |
| bare `range`, **dot-free** | nothing — no anchor to pipeline | inert, and so is the launch arg |

So the trap is a kernel that hard-codes `tl.range(num_stages=1)`: the three arms come back
**byte-identical**, read as "no pipeline to lose", and the real depth is one token away in the kernel
source. Measured, that recovered a large win on a kernel this screen had written off. Conversely a
champion whose launch passes nothing may still be fully pipelined (tuned kernels routinely carry
`tl.range(..., num_stages=2)` on the reduction loop and pass nothing at the launch; `num_stages=None` on
the loop **inherits** the launch value — real tuned kernels write `num_stages = None if ENABLE_PIPELINING
else 1`), and one that passes `num_stages=2` may never reach the pipeliner at all. **Settle it on the
dump rather than the source**: re-dump plain at launch `num_stages` 1 / 2 / 3 and diff the `.ttgir` —
where the loop carries the annotation the three dumps are byte-identical and the load count tracks the
annotation instead. Flip the annotation the champion actually uses, and re-dump to confirm the depth
moved (load count and `memdesc_index` both scale). A depth frozen in source where the tuner cannot reach
it is itself a library bug worth reporting. `$SKILL/scripts/pipeline_survey.py <tree>` classifies a
source tree by which pipeline form each kernel can exercise (A = cross-iteration software pipeline,
B = block ping-pong, C = async copy / direct-to-LDS); treat it as a **screen for what to measure**, not a
verdict — only a dump settles whether a given dispatch compiled pipelined.

> **`recover`'s pipeline verdict reads `tt.num_stages`; do not let a carry count overrule it.** The
> attribute is decisive. The `iter_args` count is not — an online-softmax or reduction loop carries
> accumulators for algorithmic reasons, so it reads ≥ 2 while un-pipelined, and one dump cannot separate
> the pipeliner's carries from the algorithm's. The tool once asserted a positive off the carry count
> alone and mislabelled an attention champion that its own output showed at `tt.num_stages=1`; it now
> lets the attribute decide and says INCONCLUSIVE when the attribute is absent.

**Size the prize at the champion's own tile, and per shape.** On one tuned attention champion the
pipeline's own contribution was small and **changed sign across sequence lengths**; its tuning notes
advertised a much larger win that was the *combined* effect of a smaller `BLOCK_N` **and**
`num_stages=2` against the shipped tile. Only `plain@ns=1` **at the champion's tile** separates the two,
and a debt that flips sign with shape cannot be reported as one number. A loop whose dispatched config
runs **once** (no trip count) has nothing to pipeline. Details:
`../gluon/pipeline/reinjection.md ### What to expect, and what to measure yourself`.

**Then the anchor-ratio expectation follows.** If the champion compiled at `num_stages=1` there was no
auto pipeline to lose, so a faithful anchor should land at **≈1.00**, and a residual below the criterion
is *not* a pipeline problem — no amount of pipeline work will move it; go to `lost_layout` / `lost_RA`.
If `plain@ns=1` matches your anchor, the whole gap is the pipeline and no amount of layout work will move
it.

---

## 3. Repay by hand, per suspect

Every repayment round is recorded as a `recovery` against the suspect it closes (§4, ledger spelling).
Fix a **structural success signal before you read a clock** — buffer count, barrier count, the
`lgkmcnt` shape, `ds_read` width, spill — so an unchanged IR is diagnosed as such instead of being
mistaken for a lever that did not pay.

### 3.1 `lost_pipeline` → an authored pipeline (hand-written first)

Only once `plain@ns=1` says the debt is real and the loop wants a pipeline at all. Author the overlap in
this order — the single definition of the order is `../tile-programming/pipeline.md`
(`## Authoring the overlap yourself (the climb default on the Gluon path)`), and its API router is
`../gluon/pipeline-reference.md`:

1. **Register-level prefetch** — issue the next tile's global loads into registers before the current
   tile's MFMA (`../tile-programming/pipeline.md ### Prefetch has three orthogonal degrees of freedom, and an unpinned one does not land`).
2. **An authored LDS ring** — double/multi-buffer with `allocate_shared_memory`. **gfx950 (main line):**
   `async_copy` (`global_load_to_shared` / `buffer_load_to_shared`) + `commit_group` / `wait_group`, each
   lane making exactly one access of a native direct-to-LDS width (128 or 32 bits); the falsifiable
   signature is `ds_write == 0` with a matching count of direct-to-LDS loads. **gfx942 downgrade:** sync
   staging (`.store()` / `.load()`; barriers are inserted for you — do not add `gl.barrier()` in the loop
   "to be safe"); async lowers only at 32 bits per lane with a clean tiling and a destination
   `order=[1, 0]`, any other width fails to translate, so do not plan the overlap around it. Start from
   `../tile-programming/pipeline.md ### Vetted double-buffer skeleton (copy, then specialize)`;
   `scripts/pipeline_examples_cdna4.py` is the runnable gfx950 check (`pipeline_examples_cdna3.py` the
   downgrade). Size depth before deepening (`## Budget before deepening`): each extra buffer is one more
   copy of the staged tiles, and the LDS/CU divisor halves workgroups silently.
3. **`warp_pipeline_stage` + the scheduling-model choice** (layer 1.5) — only with two resident waves
   (`num_warps >= 8`, Gate 0) and a stage body free of waits
   (`../tile-programming/scheduling-model.md ## How to choose`, `../tile-programming/warp-pipeline.md`).

**Recover the structure plain had, then improve on it**: the authored ring reproduces plain's
double-buffer as the parity starting point and then goes past it — per-operand depth, operand prefetch,
manual interleave — which is what makes it the climb's own route above the gate
(`../tile-programming/pipeline.md ## Recovering the structure, then improving on it`). The faithful
anchor's explicit-smem body is where the hand-built double buffer starts — keep it.

Two things not to mix into this round: a hand register-prefetch and a re-injected pipeliner do not
compose (the pipeliner *is* the prefetcher and a manual one consumes its slot), and a hand-staged loop
and a re-injected one are two different loop bodies, not two halves of one.

### 3.2 `lost_layout` → re-recover

Back to `transcribe.md` §2–§5: re-recover the layout from the IR, re-apply it, re-verify. **Never
hand-derive a basis** — a hand-intuited one silently fails LLVM translation. For a narrowed load (lost
vectorization) the fix is the recovered staging; for a grown `shared` total it is the measured
pass-through/staged choice per `local_alloc` (an elective divergence — both variants measured, the
`s_barrier` / binding-limiter / `ds_read_b64_tr` tells read off the artifacts); to choose between
numerically identical shared layouts, `ttgir_bridge.py view` (their `ds_read`/`ds_write` mix) before the
clock — one run took an anchor from roughly half of plain to parity in a single round that way.

### 3.3 `lost_RA` → slicing / register budget

Layer 5 (`climb.md`, `../tile-programming/slicing.md`): register slicing, accumulator residency,
AGPR/VGPR split. Not in the TTGIR, never visible to `ttgir_bridge.py`. For the same-multiset "lost
schedule" signature, reorder the body toward plain's program order. Do not carry the champion's
`waves_per_eu` onto the anchor (it caps occupancy outright — `transcribe.md` §3).

**Be willing to be wrong about the cap.** A residual can be correctly attributed and still not move when
you fix it, because a second resource binds at the same point — occupancy is the usual place, since LDS
footprint and register pressure each pin it independently and freeing the one you diagnosed buys nothing
on its own. Read that as the ranking being wrong, not the diagnosis.

---

## 4. The parity gate

The starting gap is a **debt**, and this stage pays it back with the mechanism that owns it — not by
climbing on top of it. Until

```text
champion_ms / current_ms >= <the criterion this run declared>      (performance parity vs champion)
```

a round's outcome is a `recovery` against the suspect it closed, never a win; reaching parity is not a
win either, it is getting back to a number the front end already measured. Only an improvement above the
declared criterion credits the explicit tier.

**The number is the run's, not this page's.** `parity_gate.py` compares `champion_ms / anchor_ms` against
`--threshold`, a parameter whose default is `0.95` — so 95% is what you get if the run declares nothing,
and this skill's frontmatter (`expects.parity: required`, ≥95%) is its default declaration, not a
contract the method fixes. Declare the criterion for this run in `decision_log.md` before the first
recovery round and pass it explicitly, so the ledger and the gate's own `threshold` field agree on what
was being cleared. The comparison is inclusive (`>=`): a ratio exactly at the criterion CLEARS. The
comparator must be the plain kernel tuned to its own best config — a 95% measured against a default
strawman is as invalid as a win over one.

**Two checkpoints, in order — neither is where the track stops.**

| checkpoint | what it is | when |
| --- | --- | --- |
| Layout equivalence + bit-parity | `ttgir_bridge.py verify` PASS/RECONCILED and the other three equivalence checks, no numeric delta vs plain (transcription is layout-only, so any delta is a bug, not a Gluon property) | exit condition of `transcribe.md` — always, never deferred |
| **performance parity vs champion** — decided by `parity_gate.py` (exit 0), whose output goes in the round log | the port has actually landed | exit condition of this stage — immediately when `plain@ns=1` found no debt and no other suspect fires; otherwise once the repayment is confirmed in the IR. **Exit 2 forbids the climb**: the round's outcome is `recovery`, and the next round closes the suspect the tool named |

They are ordered, not scheduled: the first makes the anchor trustworthy, so nothing downstream means
anything until it passes. The second is only readable once attribution has settled which case you are
in — a throughput number taken while that is still open is uninterpretable either way. They fail
differently, which is why the second gets cycles rather than a single attempt: the first converges on a
diff you can read; a pipeline repayment can only be confirmed by recompiling and reading the IR back, so
on a kernel that genuinely owes a pipeline expect several adjust-and-recheck cycles — **spend them rather
than reading the first recompile as a failure.** Where `plain@ns=1` found no debt there is nothing to
reproduce and both checkpoints close together.

**Below the criterion, do not start optimizing layouts.** Check, cheapest first:
1. **the pipeline you built did not land** — no `ttg.memdesc_index` / no ring in the Gluon TTGIR,
   `local_alloc` / `local_store` / `local_load` counts that did not move toward plain's, and a full-drain
   `s_waitcnt lgkmcnt(0)` that never relaxed;
2. **a layout recovered wrong, or recovered and never wired onto an operand** — re-run `verify` rather
   than eyeballing it, and check its `missing` list against your own preamble before concluding the
   recovery itself was wrong;
3. **a non-pipeline suspect** — `lost_layout` (LDS budget, narrowed width) or `lost_RA` (spill, lost
   schedule).

**And one non-cause worth naming, because it looks like all three.** If `plain@ns=1` said the champion
compiled at `num_stages=1`, a residual below the criterion is *not* a pipeline problem and no amount of
pipeline work (authored or injected) will move it.

**How to spell this in the ledger, because `recovery` is not one of its enums.** `round_record.py`
validates `--work-kind` against `(sweep_batch, branch_arm, climb_round)` and `--mode` against
`(sweep, branch, climb, ...)`; neither contains `recovery`, so writing `climb_round` alone loses exactly
the distinction the non-negotiable rules ask you to keep. `--stage` is free-form, so the convention is:

```
--work-kind climb_round --mode climb --stage recover \
  --hypothesis "recovery/<lost_pipeline|lost_layout|lost_RA>: <what closing it should move>"
```

and above the gate, `--stage climb`. `--stage` is the one queryable field that separates the two roles of
layers 2-5, and `--hypothesis` (already required) carries which suspect the round is paying down. That
pair is what lets a reviewer tell a recovery from a win without re-reading the transcript.
`parity_gate.py` prints `round_outcome_allowed: "recovery"` to tell you which side of the gate you are
on; it is a signal for these fields, not a value any enum accepts. The same rule holds
for GEAK's round log: a sub-parity candidate can be tracked (the PORT `candidate_floor`) but never
banked.

**This is a gate on the claim, not on the anchor**: a slow faithful anchor is still the correct anchor,
and the equivalence gate remains the only rejection of the anchor itself.

**Passing is not arriving.** Both checkpoints can close on a transcription that reproduces the comparator
and stops there — that is a port which landed, and it is the *floor* of the track, not its result. Read
a cleared gate as permission to start the climb (`climb.md`), and note the asymmetry it hides: two ways
of writing the loop can both clear the criterion while being different schedules, so clearing the bar
does not tell you the body you hold is the better of the two. The climb settles that first.

---

## 5. `parity_unreached`, and when to hand back

**Bound the recovery by convergence, not by ambition.** Once the checks above are exhausted, the
hand-written repayment is confirmed in the IR, and you are *still* short of the criterion, stop treating
it as a transcription problem: record **`parity_unreached`** with the attributed residual split
(`parity_gate.py --json` carries it), carry it in `caveats[]` on every later number, and either continue
into the climb as an ordinary optimization target — saying plainly that the port closed below parity —
or hand back. Continuing to re-transcribe past that point is the one way to spend a whole budget and land
nothing.

**Hand back when the residual is inexpressible.** If the residual is attributed to a mechanism Gluon
cannot express at parity (an `UNRECOVERABLE` staging layout such as `amd_rotating_shared`, an instruction
Gluon has no spelling for such as fp8's unscaled `tt.dot_scaled`), that is a **handoff-back signal**:
record `structure_suspect` with the evidence (or, for a tile/config cause, `resweep_request`; both go
in the deep_engineer's result for tech_lead to turn into the next round's direction), rather than spending the climb budget out-running it. A deep result that closes under
`champion_ms` is a negative (`negative_revert_plain`), never the winning stage (`close.md ## Stop conditions`).

Mode B never records `parity_unreached` — the caveat does not apply there; say so with the entry mode.

---

## Last resort: re-injecting plain's pipeliner

> **Diagnostic / last resort only.** Use this section in exactly two situations, both **below the parity
> gate** and both on a **converted (mode A) anchor**:
> - **as a diagnostic**, to measure how much of the gap is `lost_pipeline` — the injected arm reproduces
>   plain's own overlap on your explicit loop, so (injected − anchor) bounds the debt;
> - **as a last resort**, when the hand-written repayment (§3.1) has been tried and cannot reach parity,
>   and parity is needed to proceed.
>
> Every number measured under injection is labelled **`injected`** in the ledger and the report — it is
> not an upstream number (no upstream version calls these passes from `gluon_to_ttgir`; a patched tree is
> a local change) and it is **never booked as a win**: its ceiling is plain parity. **Never on an
> incumbent** (mode B): there is no debt to repay, and an already-Gluon kernel's pipeline is authored.
> Run every injected/patched variant in **its own process** — the patch is process-global, and two
> differently-patched variants co-loaded in one process silently measure the same pipeline. A reverted
> tree has to be confirmed reverted (`patch_reinject.py status`) before anything else is measured in it.
> The measured outcome on this platform is at best parity, and on a measured kernel the net stayed
> negative on most toolchain versions; the per-version numbers live in one versioned table in
> `../gluon/pipeline/reinjection.md` — do not quote them from here.

**Why it is possible at all.** At the TTGIR level an explicit Gluon loop is a non-pipelined `scf.for`,
exactly the object the AMD pipeliner is written to consume, and `add_schedule_loops` + `add_pipeline` are
already in `libtriton` on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0. Only the Python pass list omits them, so reaching
them needs **no `libtriton.so` rebuild** and — via `scripts/gluon_swp.py`, which wraps `gluon_to_ttgir`
in-process — **no edit to any installed file**. Upstream's `add_stages_inspection_hook` is the other seam.
Where there is a debt, injection alone still changes nothing: the kernel has to be a candidate.

### Three conditions, all required — none of them alone does anything

| # | condition | why |
| --- | --- | --- |
| 1 | **the loop needs an anchor, and the anchor is a `tt.dot` — not the loop syntax** | `add_schedule_loops(pm, ns)` takes the depth as a **pass argument** and reads no attribute off the loop. So a loop **containing a dot pipelines on a bare `range`**; a **dot-free** loop has no anchor and is the *only* case that needs `tl.range(..., num_stages=N)` (usable from a `gluon_jit` body), where `None` inherits the launch value |
| 2 | **the loads must still be `tt.load` when the pipeliner runs** | plain orders the pipeliner at #15/#16 and `add_convert_to_buffer_ops` at **#28**, so plain's own pipeliner only ever sees `tt.load`. An anchor written as the transcription asks (explicit `gl.amd.cdna3.buffer_load`, because `gluon_to_ttgir` runs no buffer conversion) hands it ops it cannot recognise. The two pieces of guidance genuinely conflict, and the resolution is to restore plain's *order*: write `gl.load` and arm `buffer_ops=True` (pipeline first, buffer conversion after) |
| 3 | **the staging the pipeliner is asked to build must be its own** | a hand-written `allocate_shared_memory` + `gl.barrier()` body starves it, and to give the pass a loop it will take, that staging has to come out **entirely** — one `ttg.barrier` left in the loop makes the pass skip that loop wholesale (`isSafeToPipeline` bails on any loop containing a `ttg.barrier`). **Scope: per-operand, not per-loop** (below) |

> **Do not generalise the 2×2 in `../gluon/pipeline/reinjection.md`.** It reports `gl.load` + bare
> `range` as not pipelining, which is true **on the dot-free kernel it was measured on** and false with a
> dot in the loop. Two authors rewrote working bare-`range` dot loops into `tl.range` for no effect before
> that was caught. Condition 1 is the rule; the table is one row of it.

**Decoding the error text on condition 3.** `'ttg.local_alloc' op pipeliner doesn't know how to predicate
this op` is the **symptom of staging not removed**, not a language wall — the first investigation to hit
it concluded that Gluon's `allocate_shared_memory` was fundamentally incompatible with the pipeliner, and
that was wrong.

**Condition 3 is narrower than "hand-written staging blocks the pass" — a mixed loop qualifies.** What the
pass needs is *one* dot whose operands it can trace back to a `tt.load` (it walks SSA operands backwards
from the dot; a `local_store` is a side effect, not an SSA edge, so through a hand-staged operand `tt.load`
is never reachable). Measured on an attention body with three dots: the SSA walk back from two of them
dies at a `local_store`, but the third is a pure register path (`gl.load` → `convert_layout` → permute →
`convert_layout` → dot), so the pipeliner fires on that one and multi-buffers the rest of the loop along
with it — visibly, the body went 3 dots / 104 `v_mfma` to 6 / 208. That arm was the kernel's best result.
So confirm the pass did nothing from the **IR**, not from the shape of your source.

**`buffer_ops=True` is opt-in because it fails three ways:** on an anchor whose **loads** are already
`buffer_load` it aborts the pass manager loudly (`PassManager::run failed`); on one whose **stores** are
buffer ops it does not raise at all — `LLVM ERROR: Fatal pipeliner error` kills the interpreter; and a
single buffer op left **outside** the loop is enough, because the rejecting pass
(`TritonAMDGPUCanonicalizePointers`) runs over the whole function rather than the pipelined region. Arm it
only on a body written throughout — the whole function, not the loop — with `gl.load` / `gl.store`. It is
not free either: on one kernel it was a further penalty on top of the injection.

**Un-staging and injecting are one step, and the pair is only *conditionally* worth it.** Un-staging alone
removes a multi-buffer that was doing real work and puts nothing back, so it is a regression, not a neutral
intermediate — measure three arms (hand-staged no injection; **un-staged, injection OFF** — the trap;
un-staged, injection ON). The injection does not always cover that cost: measured across four versions on
one kernel it lost three times and won once. **Judge it on the same-window per-rep ratio `L+P ÷ L`** —
subtracting two percentages against a drifting plain arm cannot resolve a 3% verdict. Keep the
explicit-smem version either way; it is where the hand-built double buffer (§3.1) starts. Two other ways to
starve the pass: a hand register-prefetch (the pipeliner *is* the prefetcher, and a manual one consumes the
slot — not additive) and a loop-variant `scf.if` (split a causal mask into two loops; it blocks
`BlockPingpong` too).

**The cheap pre-check that predicts the verdict:** compare **plain's own `ns=1` against its `ns=2`**. On a
version where plain itself gets no pipeline benefit, recovering the pipeline for it is not worth it either
— that held on all four versions of the kernel above, with the one version that showed plain a real gain
being the one version where the recovery paid.

**The exception, and it is why re-injection is never the first move:** do not re-inject into a loop that
already hand-authors its own staging. There it provably cannot fire (the `ttg.barrier` bail-out and the
SSA walk above). It runs and changes the IR by zero bytes, which reads exactly like a tuning problem and is
not one. And on a body whose staging plain puts on a layout Gluon cannot express (`amd_rotating_shared`),
the pass can fire perfectly and still lose, because it falls back to an unvectorised
`swizzled_shared<vec=1, perPhase=1, maxPhase=1>` (`../gluon/pipeline/reinjection.md ### And on attention — two dots chained through a softmax`).

### Inject, without editing anything — `gluon_swp.py`

`scripts/gluon_swp.py` wraps `HIPBackend.gluon_to_ttgir` in-process and runs the two passes as a second
pass manager over the module the stock function returns:

```bash
python3 "$SKILL/scripts/gluon_swp.py"            # capabilities of THIS build, probed not inferred
python3 "$SKILL/scripts/gluon_swp.py" --selftest # offline; skips cleanly with no AMD backend
```
```python
import gluon_swp
with gluon_swp.pipelined(2, buffer_ops=True):    # compile INSIDE the block -- Triton caches
    out = my_anchor[grid](...)                   # one injected variant per process
```

It produces **byte-identical TTGIR to the on-disk splice on all four versions**, armed and unarmed, so
nothing is given up by not touching site-packages — while a read-only or shared install, a later
`pip install --force-reinstall`, and a crash mid-experiment all stop being hazards. `capabilities()`
inspects the *original* function, so a second `enable()` at a new depth is not refused as a fork. It
**refuses** to install on a fork that already splices the passes in (running them twice is a different
experiment), and refuses `num_stages < 2`, where the pipeliner is a no-op.

> **Known limitation: it stops at `add_pipeline` and omits plain's post-pipeline tail**, and the
> consequence is not "less gain". The pipeliner's `local_load` lands in a blocked layout with a separate
> `convert_layout` to the dot operand, so **each operand takes an extra LDS round trip** on top of the
> multi-buffered staging. On a tile whose staging already fills the budget that surfaces as
> `OutOfResources: Required <n>, Hardware limit <the arch divisor>` — the injection succeeded and looks
> broken. **gfx942 downgrade vs gfx950:** that is how it announces itself on CDNA3 (64 KiB); on gfx950
> (160 KiB) the larger ceiling absorbs it and it becomes a **silent slowdown**.
>
> **Where the ceiling is roomy enough to absorb it, the exception disappears and the cost does not** —
> injecting without the tail can be *worse than not injecting at all*, by a multiple rather than a margin,
> while compiling cleanly. Do not read the LDS growth as the cause: the tell is **register spill**,
> because pipelining without the buffer conversion leaves 64-bit pointer *tensors* live across the peeled
> stages. Occupancy can look untouched — only the `spill=` field moves — so **check the pass list, not the
> exception**, and check `spill=` rather than `shared`.
>
> **The pipeliner and the tail are a pair, and each half alone is a trap in a different direction.**
> Pipeliner without the tail is the case above. Tail without the pipeliner is worse than slow: it returns
> **NaN**, because `add_block_pingpong` assumes the pipeliner has run. Neither half is a valid arm.
>
> `add_remove_layout_conversions` is the pass that folds the double trip. If you hit it, splice plain's own
> order after `add_pipeline`: `add_convert_to_tensor_ops` → canonicalizer →
> **`add_remove_layout_conversions`** → `add_reduce_data_duplication` → `add_move_up_prologue_loads` →
> `add_block_pingpong(ns)`. Record a pass absent on this build as `-name` rather than skipping it
> silently, or a cross-version regression becomes invisible; 3.6.0 lacks the first and the fifth and still
> reaches the same op census with the other four.

**The on-disk form.** `scripts/patch_reinject.py apply|revert|status` is kept for when you want the pass
list visible in `compiler.py` while reading. It is env-armed (`TRITON_GLUON_SWP=N`, plus
`TRITON_GLUON_SWP_BUF=1` for the buffer half — this skill's own patch variables, read only by the patched
`compiler.py`) so armed and unarmed are the same binary, which is the only way an IR diff between them
means anything. Its splice point is version-dependent and measured: before `add_warp_pipeline` on 3.7+;
after the last `add_*` call on 3.6, which has no warp pipeline at all. It writes a `.orig_swp` backup;
`revert` restores it and clears `__pycache__`. `--selftest` pins both splice points.

> **`TRITON_GLUON_SWP_PIPELINE` is not the knob**, and neither are `TRITON_GLUON_COOP_LDS` or
> `TRITON_GLUON_PINGPONG`. **Fork-only**: all three belong to a vendor fork's `GetEnv.h`; **no upstream
> version reads any of them** (upstream `main`, the `v3.8.0` tag and every fork tag searched, plus
> `git log --all -S`). Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — as is a knob
> invented on the spot — which is the worst of the three available outcomes: nothing errors, nothing
> changes, and the null result reads as "this technique does not work here".

### Confirm it fired from the IR, before you time anything — and read the right tell for your shape

Dump the Gluon TTGIR armed and unarmed (separate processes) and compare. **The tell differs by whether the
loop has a dot**, and using the wrong one produces a confident false negative on a loop that did pipeline:

| loop | landing tell |
| --- | --- |
| **contains a dot** | `ttg.memdesc_index` appears — the multi-buffered-LDS signature, and the cheapest single signal |
| **dot-free** | **not** `memdesc_index`, which stays 0 because a dot-free loop prefetches into *registers* and never touches LDS. Read `iter_args` 0→1 and the load count scaling with depth (2→4→6), plus `tt.num_stages` on the `scf.for` and a visible peeled prologue |

Either way the full-drain `s_waitcnt lgkmcnt(0)` should relax to `lgkmcnt(N>0)`. IR **identical** armed
and unarmed means the pass ran and rewrote nothing — a different failure from the passes being absent, and
one no availability probe can see: `probe_levers.py --all` reports that the symbols are in this
`libtriton.so`, not that they will bite on your IR.

**Below parity under injection, the two injection-specific causes** (they share the symptom of §4's first
check, which is why the armed/unarmed dump is what separates them):
1. **the pass never ran, or ran on a loop it does not consider a candidate** — the loop is a bare `range`
   with no dot in it, `num_stages` resolved below 2, or (on the on-disk form) the splice went in at the
   wrong point for this version. IR differs armed vs unarmed only if it ran at all;
2. **it ran and rewrote nothing**, because the body left it no work: IR identical armed and unarmed.
   Hand-authored LDS staging is the usual reason on a GEMM — and un-writing it is only half the move;
   `gl.amd.cdna3.buffer_load` in the loop is the other usual reason, and a hand prefetch or a loop-variant
   mask covers the rest.

**A missing signal is not counter-evidence until you have seen it on the reference arm.** `s_setprio`
staying 0 after injection was once read as "the mechanism never started" — but plain's own `ns=2` build of
the same kernel also had `s_setprio == 0`, because `add_block_pingpong` does not accept that loop shape at
all. Two other signals said the injection had fired. Confirm a tell would have appeared on the comparator
before treating its absence as a verdict. (Ping-pong never fires on hand-authored staging: it collects
`local_load`s sourced from a loop-carried `BlockArgument`, and a hand-written one comes from
`memdesc_index`.)

**The cache trap has two halves, and each on its own gives a false negative.** It was first hit here, but
it applies to every A/B on this path, including a layout sweep whose variants differ by a `constexpr`.
*In process*, Triton's JIT cache is keyed on `(function, signature, constexprs)` and **knows nothing about
the injection**, so two arms differing only by the wrapper hit the same compiled artifact and the second
silently reuses the first's code; `TRITON_ALWAYS_COMPILE=1` does **not** fix it. *On disk*, a per-arm
`TRITON_CACHE_DIR` does not encode the **depth**, so an `ns=3` probe pointed at the `ns=2` directory is
served the `ns=2` binary and reads as "depth does nothing". The rule for injected/patched arms: **one arm
per process** (this removes the in-process half), each with its own `TRITON_CACHE_DIR` keyed on the arming
— `triton.knobs.cache.dir = f"/tmp/run_{arm}_{gluon_swp.cache_tag()}"` (`cache_tag()` encodes depth, post
recipe, buffer_ops, ping-pong and async-copy arming) — and confirm each arm's own `.ttgir` carries the tell
above. Earlier GEAK guidance interleaved differently-armed arms in one process behind distinct kernel
objects; that protocol is withdrawn.

**Depth is a knob, not a monotone, and the LDS cost is arithmetic you can do before you measure.** The
pipeliner builds a rotating stage of **`num_stages − 1`** buffers, readable straight off the TTGIR as the
leading dimension of the staged `memdesc` (`memdesc<Nx…>` at depth `N+1`, absent at depth 1). So each depth
past the first costs one more copy of the staged tiles, `ns=2` is single-buffered with a peeled prologue
rather than double-buffered (prologue peeled so the global load overlaps the MFMA, LDS unchanged — why 2
often wins), and any tile's whole depth series is predictable from one dump rather than swept blind. Two
ceilings then apply and they are **different limits with different symptoms**: the per-workgroup
**allocation** ceiling refuses the launch outright, while the **LDS/CU divisor** (gfx950 160 KiB; gfx942
downgrade 64 KiB) silently halves workgroups per CU. A depth can therefore compile and still be the wrong
depth, and the curve commonly turns at the divisor rather than at the refusal. `recover`'s `LDS:` line
predicts this before a clock is read, as an **upper bound** — it sums declared allocations without
modelling liveness reuse, so quote `$KT/probe.py measure` off the artifact once one exists. A depth plain
itself cannot compile is not available to you either (same pass, byte-identical LDS requirement).
`num_stages=3` was worse than 2 on every kernel measured, and on one refused to launch. Start at 2 and
sweep. Full mechanism, the per-shape behaviour, the ping-pong window and why async copy is not reachable
from plain on gfx942: `../gluon/pipeline-reference.md` (router) and `../gluon/pipeline/reinjection.md`.

**A separate, narrower patch — `patch_async_reinject.py`.** `make_ttgir` runs `add_coalesce_async_copy`
whenever async copy is on; `gluon_to_ttgir` never calls it. `scripts/patch_async_reinject.py
apply|revert|status` splices it, env-armed (`TRITON_GLUON_ASYNC=1`, so unarmed is byte-identical to stock),
with a `.orig_async` backup that `revert` restores. **Scope, corrected by measurement: this is NOT what
makes async copy work on gfx950.** Both `global_load_to_shared` and `buffer_load_to_shared` lower and are
numerically correct on stock `gluon_to_ttgir` when each lane makes exactly one access of a native
direct-to-LDS width — 4 B or 16 B on gfx950; gfx942 downgrade: 4 B only. What fails without coalescing is
the **non-native** pattern: 8 B and 32 B per lane, and any layout whose per-lane contribution is a native
size but split across repetitions (a `BlockedLayout` covering `[64, 16]` on a `[32, 32]` tile makes two
accesses per lane and does not lower — with `PassManager::run failed` / LLVM-translation wording that
reads like a missing op). The pass repairs that class by adding a bounce the read has to go through —
a cost easily mistaken for an intrinsic cost of the async path. So **prefer fixing the layout to match a
native width** (hold op, arch and width fixed and vary only the tiling before recording an arch ceiling);
reach for this patch only after `pipeline_examples_cdna4.py` has shown the width is the problem, price
the bounce, and label its numbers like any other patched variant. It belongs to the authored async ring
(§3.1), not to pipeliner re-injection.
