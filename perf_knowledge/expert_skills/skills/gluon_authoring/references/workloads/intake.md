# Variant intake — what is this kernel, and how deep may I go

Read this when a kernel arrives that no recipe covers: a GEMM or attention **variant** the pack
has not seen. It answers two questions that are routinely conflated, and it is the only page that
crosses them:

- **Axis 1 — what IS it?** Read the variant through a fixed set of variables instead of looking
  for its recipe. Different values, same framework.
- **Axis 2 — how deep may I recompile?** The levers available are a property of the **build in
  front of you**, not of preference or of how hard the problem looks.

It is an intake procedure, not an optimization method: it ends with a recorded plan and a spend
order, and hands off to the layer backbone (`../method/climb.md`). It decides nothing that a
measurement should decide. The archetype router it feeds is `index.md`; the per-workload fills of
Axis 1 are `attention.md` and `gemm.md`.

Three disciplines run through it. **Axis 1's values decide which rungs of Axis 2 are worth
climbing** — a variant with vector work between its matrix ops has already ruled out the
matrix-accumulator-chain compiler knobs, whatever the build admits. **Axis 2 is a gate, not a
menu** — a design can be *unwritable* on a given build rather than merely slower, and the honest
comparison is then against what that build can express. **The deeper the rung, the more reversible
it has to be** — default-off, A/B against the unmodified build on the same boundary, and a verdict
that is scoped and expires rather than a hardware wall.

**Arch and version baseline.** Everything here is stated for gfx950 (CDNA4, MI350X / MI355X) on
upstream Triton 3.8.0; gfx942 (CDNA3, MI300X / MI325X) and 3.6 / 3.7 differences are downgrade
notes. The gfx942 downgrades that change an intake answer are listed once in
`index.md` (no scaled MFMA / MXFP, no `ds_read_b64_tr`, async direct-to-LDS only at 32-bit with
`order=[1,0]`, 64 KiB / 32-bank LDS, FNUZ fp8), and the two places they bite below are marked.

## 0. First: is it the archetype its name says?

Cheapest possible step, and the one that most often changes everything after it. The workload
models are **archetypes**, and `perf_knowledge/hardware/data/workload_models.json` says in its own
scope field that the archetype is the assumption most likely to be wrong. It carries a catalogue of
the recurring ways a real kernel fails to match the row that shares its name; walk it before
accepting a class:

- a single dtype cannot carry per-tensor precision (an fp8-operand matmul with a wider output has
  most of its bytes in the output);
- the kernel takes pointer arguments it never dereferences on the taken path;
- a workspace or cache is sized for the general case and used for one sequence;
- an operand is re-read once per tile of the other, so issued traffic is a multiple of the
  footprint that depends on the **tile**, not the shape;
- a grid dimension is dead on the taken path;
- padding to a block boundary makes executed work exceed useful work;
- a decode shape is a **gather and a reduction** rather than the quadratic thing its name suggests;
- the timed region is several kernels, against which a per-kernel floor means nothing.

**When any of these holds, do not stretch a row to fit.** Declare what the kernel moves as a
manifest — `scripts/hw_budget.py --tensors` returns a bracket instead of a false point
(`index.md ## When no row fits` has the manifest syntax). A class recorded for a kernel that does
not match it will mis-set the regime branch, the compute floor, and the structural routing
downstream, and none of those failures announces itself.

**Walking the catalogue and finding nothing is not a pass.** The list above enumerates mismatches
that have been seen, so it can only fire on a shape someone already met; it is not a decision
procedure and a kernel can clear all of it and still be in the wrong class. Three reads off the
body decide the class positively rather than by exclusion — which matrix op the accumulator chain
issues, whether VALU work sits between the matrix ops, and whether the operand assignment is
data-dependent. They are stated once, with their recorded consequences, in the table at the top of
`index.md`; take them from there. Two points that matter at intake specifically:

- The first read is also the **counter convention** every later bound claim is made under — on
  gfx950 a per-`(version, M, N, K)` question rather than a per-dtype one
  (`../hardware/optimization-gotchas.md` rows 2 and 5; `../tile-programming/low-precision.md`).
  **gfx942 downgrade:** there is no scaled op to distinguish, so the read is "none or regular".
- The second read is a **legality** question, not a tuning one: on a VALU-between-matmul body the
  in-tree scheduling toggle emits invalid IR / a verifier assertion rather than a slowdown
  (`../hardware/optimization-gotchas.md` row 7). It removes rungs of Axis 2 before any measurement.

A name that matches the body is the common case. The reason to spend the two minutes is that these
three reads cost less than the round a wrong class spends.

Record the class from the authoritative vocabulary (`workload_models.json` models and aliases), not
from a free-text guess. `other` is not a class: it reads as unresolved and **withholds the
structural axes** rather than widening them. Decode / GEMV and paged bodies additionally run the
Quick Reject (`../pitfalls/negative-patterns.md ## Quick Reject Checklist (= gate skip-transcription conditions)`)
as an admission hint before any transcription is planned.

## Axis 1: the variant variables

Do not memorize per-variant recipes. Read a new variant through these variables, then apply the
principles. Each one names what it *decides*, which is why the list is short and why the order
matters — the first three constrain more than the rest.

1. **Which matmuls, which reductions, and on which axes.** Fixes the structure decision and where
   recompute or atomics land. Two reductions on different axes is a structural choice problem
   before it is a tile problem
   (`../tile-programming/mental-model.md ## Reduction & parallelization structure (decide before tiles)`).
2. **What sits between the matmuls** (softmax, gating, smoothing, dequant). Its presence breaks
   the pure matrix-accumulator-chain assumption, which is what the GEMM-class compiler knobs are
   built on — so it removes a whole rung of Axis 2 regardless of the build, and it may force
   accumulator round-trips through a second register file.
3. **Where that third instruction category has to ISSUE, and whether it fits.** Separate from (2):
   that one says the stage exists, this one says whether it can be hidden. Per compute region,
   capacity against demand; when demand exceeds capacity no ordering wins and the work must shrink
   or move. Two cheap gates come first — two resident waves per SIMD, and a bound class that is
   actually matrix-issue. Arithmetic and procedure:
   `../tile-programming/llir-codesign.md ## Attention: the co-execution budget`.
4. **Operand and intermediate footprint.** Feeds the capacity wall and the occupancy budget; a
   compressed or low-rank operand changes the per-lane register count, which is the resource that
   caps resident waves and therefore gates (3). The LDS side of that budget is 160 KiB per CU on
   gfx950 and 64 KiB on the gfx942 downgrade, so the same footprint can bind on a different
   resource per arch.
5. **Precision side-paths** (quantization, per-block descale). A side path with its own traffic and
   its own vector work, which competes with (3) for the same issue slots
   (`../tile-programming/low-precision.md`). **gfx942 downgrade:** with no scaled MFMA, every descale
   is explicit VALU, so this variable always adds demand to (3) there.
6. **Data access pattern** (dense vs sparse / indirect / paged). Sets the memory path and can
   change the parallel structure.
7. **Does the loop body's composition vary per iteration?** The only variable that can
   **invalidate** the others rather than re-value them: it decides whether a *schedule* is a
   well-posed question at all. A data-dependent body makes the budget a distribution rather than a
   number, removes the static-unroll premise, and adds an address chain that must be placed in the
   memory region or the compute regions stop being two-category
   (`attention.md ### The dynamic-body variable (sparse / paged / varlen)`).

Variables 1 through 6 are **static** — given shapes and dtypes you can evaluate them on paper
before compiling. That is the point: the intake is cheap.

**Per-workload fills.** The variables are the same; their values and the decisions they reach are
workload-specific. Attention: `attention.md ## Applying the framework to attention variants`. GEMM:
`gemm.md ## Applying the framework to GEMM variants`. MoE adds its own typing on top
(`moe.md ## Type the kernel before advising it — three questions`). A variant that lands between two
fills (low precision *and* a vector stage, sparse *and* quantized) is not covered by either — read
`attention-lowprec.md` for what that looks like and expect to have to measure rather than compose.

### Four things the source will not tell you, and the optimum is undefined without them

Axis 1 reads the kernel. This reads what the kernel **assumes**, and it is a separate pass because
none of it is recoverable from the body: these are decided by whoever calls the kernel, they are
constants from inside it, and a tuning result is only valid for the values that were in force when
it was measured. Production makes all four explicit — as manifest fields, as load-time
transformations, and as registration-time contracts that **refuse to start** on a mismatch rather
than running slower. Write down a value or write down `unknown`; an unrecorded assumption is
indistinguishable from an absent one, and it is what makes a result unportable in a way nobody can
see later.

| | the assumption | how to read it | why a wrong value invalidates the round rather than costing a few percent |
| --- | --- | --- | --- |
| **P1** | **operand layout** — is the weight in its canonical dense form, or a vendor-preshuffled one? | not from a dtype and not from a stride; it is an out-of-band property of the caller's pipeline. Ask, or read the caller. | The same GEMM under two layouts is **two different optimization problems** — production separates them with two named layout values and refuses to load a kernel built for one into a deployment using the other. The relayout itself happens once at load time, so it appears in no per-call benchmark while being a precondition of the kernel running at all. Distinct from the case where the kernel *contains* the unshuffle (`gemm.md ## Preshuffled / block-scaled signals`) — there you can see it. |
| **P2** | **is the activation already quantized upstream?** | look at what the *producer* kernel emits, not at this kernel's signature | A fused producer can emit both a wide and a narrow form while the module interface carries only the wide one, so the consumer re-quantizes something that was already quantized. The redundancy exists **between two kernels** and each one is individually correct, so no single-kernel profile shows it. On the gfx942 downgrade also check the producer's fp8 *encoding*: an OCP-e4m3 producer feeding an FNUZ consumer is a silent numeric mismatch, not a redundancy. |
| **P3** | **the fusion boundary** — which ops are inside this kernel, and which were deliberately left out | read the name and the body, then ask *why* the ones next to it are not in it | Axis 1 variable 2 reads this boundary as given. It is a decision, and sometimes a reversible one — but some of the reasons it was drawn there are not performance reasons at all (a downstream consumer needs a different numeric path; a stage must not be frozen into a replayed graph). Assume it is negotiable and you may propose a fusion that is already known-wrong. |
| **P4** | **the execution context** — eager, or captured into a replayable graph? and is the batch padded? | this is a property of the *deployment*, and it is the one most often silently assumed to be eager | A kernel with a host-side data dependence — a `.item()`, a shape read on the host, a trip count derived from CPU-visible metadata — is **correct in eager and wrong under capture**, because the value gets frozen into the replayed graph. Production makes this a registration contract with its own field. Note the asymmetry from the measurement side: `production boundary` already appears in the task contract as *where you time*, which is a different question from whether this kernel is legal there. |

**The general rule these four are instances of, which is also the test for whether you have found a
fifth:** when the **cause** of an inefficiency and the **point where it could be removed** sit on
opposite sides of the kernel boundary, it is not solvable at this tier. Record it — both sites, and
why they are not in the same kernel — set the layer `out_of_scope` with that reason, and carry on
with the rest. This is a recording rule, not an escape hatch: naming only one of the two sites does
not qualify, and "I could not make this faster" is not a boundary finding.

## Axis 2: the recompile-depth census

This axis asks how deep the compilation stack may be rewritten, and **whether this build admits
it**. It matters most for kernels whose performance comes from a *manually arranged* pipeline,
because the constructs that express such an arrangement are not uniformly available: cluster
markers, a per-wave predicate, and in-band scheduling-group ops each appear at a different point
in the version lineage, and one of them is absent from the DSL surface on every upstream version.
So the same design is authorable, half-authorable, or unwritable depending on the build — and that
is a fact to establish, not to assume. On upstream 3.8.0 the per-compile route is `llvm_fn_attrs`
(`../tile-programming/llvm-fn-attrs.md`), and the stock `coexec` scheduler strategy
(`TRITON_HIP_USE_COEXEC_SCHEDULER`) is the answer to check first for matrix-plus-VALU regions; its
default and arch gate are stated in
`../tile-programming/compiler-contract.md ## Which strategy your build admits`. Environment names
that exist only on a vendor fork are labelled there as fork-only — inert, not an error, on upstream
(`../tile-programming/non-upstream-reserve.md`).

Walk the rungs shallow-to-deep. At each one ask three questions, in this order:

1. **Does this build admit it?** Establish identity before capability, and probe rather than infer
   — `../tile-programming/compiler-contract.md ## Toolchain identity` pins the five facts and
   `## Which strategy your build admits` maps them to what is reachable. `scripts/probe_levers.py`
   answers presence on the box; whether a pass then *bites* is a separate question only the
   lowered code answers.
2. **What does it unlock for the arrangement I need?** Not "what does it do" — what construct does
   it put within reach that the rung below does not.
3. **If unavailable, what do I record?** See the verdict column of the ladder. This is where the
   census is most often written wrong.

The order of establishment, which is not the same as the rung order:

- **Reachability of the source and the launcher** — and note this is *not* edit permission. A
  harness resolving a kernel by name dispatches through whatever the source exports under that
  name, so a source-owned launcher can take grid, configuration and split decisions back from a
  call site nobody may edit. In packs that ship a structure census the distinction is written up
  there; where they do not, the cheap probe is to export a callable of the kernel's own name from
  the source and measure whether the harness dispatches through it.
- **DSL expressibility** — which arrangement constructs exist on this minor at all.
- **Build identity** — the five fields, recorded beside the numbers.
- **Host build state** — symbol visibility, whether a target machine survives plugin loading, and
  whether a plugin binary's ABI target matches the host's compiler revision.
- **Source tree and sanction** — whether the compiler's own source is in hand, and whether
  changing it is authorized.

**The failure mode to guard against is writing "unavailable" as "nonexistent".** Each rung has its
own shape of quiet failure: an environment knob from the wrong generation is *inert, not an error*,
so probe the mechanism and never the variable; the plugin rung has three build states of which two
fail silently; and an unsanctioned rung is **out of scope**, which is a different verdict from a
ceiling and must not be recorded as one.

## The recompile-depth ladder

Six rungs. The rung number is the **cost order** of rewriting the stack, not the order in which
overlap is pursued: the hand-written arrangement (rung 0) is always spent first, and rung 1's
re-injection is the lowest-priority *overlap* route even though it is cheap to apply. Spend the
shallowest rung that reaches the binding variable.

| rung | does this build admit it | what it unlocks for a manual arrangement | cost / risk | verdict if unavailable |
| --- | --- | --- | --- | --- |
| **0 — kernel source and layout** | always | the arrangement itself, in the order `../tile-programming/pipeline.md` defines: register-level prefetch, an authored LDS ring (gfx950 `async_copy` + `commit_group` / `wait_group`; gfx942 downgrade sync staging, async only 32-bit with `order=[1,0]`), `warp_pipeline_stage` markers where `num_warps >= 8`; plus slicing and epilogue placement | none beyond the round | n/a — this rung cannot be unavailable |
| **1 — in-process pass re-injection, no rebuild** | the passes are present in the installed library on every checked minor; needs the loop shape they anchor on | reproduces the auto-pipeliner's cross-iteration overlap on an explicit loop, at unchanged occupancy — **as a below-parity diagnostic (sizing `lost_pipeline`) or a last resort** when the hand-written ring cannot reach parity; numbers labelled *injected*, never a win, never on an incumbent Gluon kernel | mutually exclusive with hand-written staging in the same loop; a partial de-stage does not count; ceiling is plain parity | scoped toolchain ceiling for the injection; the hand-written arrangement of rung 0 stands regardless |
| **2 — per-compile attributes and environment knobs** | upstream 3.8.0: `llvm_fn_attrs` per compile and the stock `coexec` strategy; other env names are generation-dependent and the wrong one is **inert, not an error** (fork-only names are labelled in `../tile-programming/non-upstream-reserve.md`) | the AGPR-form / scheduler-strategy attributes, the stock matrix+VALU scheduler, or on a fork the in-tree interleave / allocation / peephole stack | knob-class assumptions (matrix-accumulator chain only, for the in-tree stack); a sweep is needed for the portable option | scoped, and only for the rung probed — a missing rung is not a ceiling for the whole card |
| **3 — out-of-tree compiler pass plugin** | **a default-visibility host build** plus an ABI-matched compiler revision and a host that keeps its target machine for plugins | emits the in-band scheduling-group ops the DSL surface does not expose at all, so a declared interleave becomes reachable without touching compiler source | **no LLVM rebuild, but a host rebuild**; three build states, two fail quietly; ABI mismatch crashes rather than degrades | scoped tool ceiling; fall back to rung 0's hand arrangement or the wave-level markers |
| **4 — compiler front-end source + rebuild** | source tree in hand **and** sanctioned | changes the pass list, a lowering pass's placement decisions, or adds a language construct — the only rung that can undo a lowering change that took an arrangement away | sanction-gated; must be branch-isolated, default-off, and A/B'd against the unmodified build | **out of scope**, not a ceiling, when unsanctioned |
| **5 — compiler backend source + rebuild** | source tree, pinned revision, and sanctioned | a new or altered backend pass, or an upstream fix not yet in the pin | heaviest; keep the edit to one target so the rebuild stays minute-scale | out of scope when unsanctioned; otherwise a scoped, dated result |

Two boundaries in that table are the ones people get wrong. **Rung 3 already costs a host
rebuild** — the accurate claim is "no backend rebuild, but a default-visibility host build", not
"no rebuild". And **rung 4 is not rung 5**: rewriting the front end's own lowering is a different
act, with a different entry gate, from writing a backend pass, and the two were previously lumped
together. A third is new with the hand-written-first order: **rung 1 being cheap does not make it
early** — its result is evidence about the debt, not a candidate to keep.

The plugin rung's four gates, in the order they must be checked and with the wrong conclusion each
one produces when skipped, are in `../tile-programming/llir-codesign.md ## The plugin tier`; not
repeated here. Rungs 4 and 5 are gated by the same sanction flag the classifier enforces, so this
table introduces **no second scope judgement** — an unsanctioned rung is filtered as out-of-scope
upstream of any lever choice.

**Where each existing ladder lives.** Six of them exist already, each with local detail worth
keeping in place; this table is the index, not a replacement:

- `../tile-programming/compiler-contract.md ## Compiler scope: kernel-first; co-design only when sanctioned`
  — the permission tiers (may / may-with-a-host-build / may-not), and rung 4's own entry gate and
  procedure
- `../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question per tier`
  — which tier builds the overlap, the hand-written-first order, and why re-injection and authored
  staging are mutually exclusive in one loop body
- `../tile-programming/compiler-contract.md ## VALU-between-matmul: manual interleave, or author a pass`
  — the same ordering stated for the vector-stage case
- `../tile-programming/llir-codesign.md ## The plugin tier` — rung 3's gates
- `../method/entry.md` — the plain-to-explicit seam (the escalation gate), which is a different
  question from depth
- the router pack's cross-DSL ladder — **changing language**, which is an orthogonal axis to
  recompiling deeper; consult it in parallel, not as rung 6

## Crossing the axes: the spend order

**Rule: the shallowest rung that reaches the binding variable.** Not the deepest available, and not
the one the last kernel needed. For the overlap specifically, rung 0's hand-written order is spent
before rung 1 regardless of cost (`## The recompile-depth ladder`).

Three vetoes, each of which removes a rung before any measurement:

- **Axis 1 already ruled it out.** A variant with vector work between its matrix ops does not get
  the matrix-accumulator-chain knobs, on any build. Reaching for them produces invalid output or a
  regression, not a slowdown to tune away.
- **Axis 2 does not admit it.** Record a scoped ceiling and fall back; do not re-attempt it at a
  deeper rung hoping the gate was wrong.
- **The budget says no schedule wins.** When demand exceeds capacity in a region, every scheduling
  rung is off the table until the work shrinks or moves — the problem is not ordering
  (`../tile-programming/llir-codesign.md ## When the schedule is not the problem`).

A fourth, softer rule: a change that only **creates headroom** is not a win by itself and must be
attributed as a pair with whatever spends it, or it reads as neutral and gets reverted
(`../method/close.md ## Attributing a change that only creates headroom`).

## Search breadth is a different axis, and it is owned per pack

Do not fold configuration sweeping or structural fan-out into Axis 2. Those are **search breadth**,
and whether they are yours to spend is fixed by the pack's own contract rather than by the build:

- a **broad-search front end** owns them — it is where the configuration surface is swept and where
  structural hypotheses fan out (`../method/front-end.md`);
- a **deep-dig back end** does not. Its configuration arrives already pinned with the champion, its
  round shape is one coupled layer at a time, and its contract says so explicitly. The published
  search policy in its own card set says so too.

So in a deep-dig pack these are not unavailable freedoms, they are **already-spent** ones. Treat a
gap that looks like "the configuration was not explored" as a message to send back to the front
end — a `resweep_request` — not as a rung to climb here. GEAK's `kernel_workflow` owns the round
loop, so the deep_engineer returns that request in its result and tech_lead turns it into the
next round's direction (`../method/orchestration.md`).

## The intake record

One block, recorded before the first round, and aligned with the identity block in
`../method/profile.md` rather than duplicating it:

```text
class          : <from the authoritative vocabulary>  + archetype-mismatch verdict (which of the
                 catalogue items apply, or none) + the manifest if the row did not fit
arch           : gfx950 | gfx942 (+ SKU from gpu_identity) -- which downgrade notes apply
preconditions  : P1 operand layout      : canonical | vendor_preshuffled | unknown
                 P2 activation prequant : yes | no | unknown   (+ fp8 encoding: OCP | FNUZ)
                 P3 fusion boundary     : what this kernel takes in, what was left out, and why
                 P4 execution context   : eager | graph_capture | both | unknown
                                          + padded batches? + max context, where they bind
                 # `unknown` is a legal value and MUST be repeated in the final report; an
                 # unrecorded assumption reads later as an absent one
axis 1         : the seven variables, with values; which one is binding
axis 2         : rung reached, per-rung verdict, and the identity five-tuple it rests on
spend order    : the rungs to be spent, shallowest first, with the vetoed ones and WHY
                 (rung 1, if listed at all: diagnostic | last_resort -- never a climb step)
handoff        : the layer this hands to, and what would make the plan invalid
```

The two halves that are most often left implicit are the **vetoed** rungs and the **binding**
variable. Both matter later: a rung recorded as vetoed does not get re-litigated next round, and a
binding variable that turns out wrong invalidates the spend order rather than just one lever.

## Cross-refs

- `perf_knowledge/hardware/data/workload_models.json` — the authoritative class vocabulary and the
  archetype-mismatch catalogue this page consumes; `index.md` is its prose router
- `../hardware/capability-matrix.md` — which mechanisms exist per target, with evidence status
- `../tile-programming/compiler-contract.md` — build identity, the permission tiers, rungs 4 and 5
- `../tile-programming/llir-codesign.md` — rung 3's gates, and the budget that vetoes scheduling rungs
- `../tile-programming/pipeline.md` — the single definition of the hand-written-first overlap order
- `../tile-programming/mental-model.md` — the structure-first decisions variable 1 reaches, and the
  cross-architecture porting filter (`## Porting a technique across architectures`)
- `attention.md`, `gemm.md`, `moe.md` — the per-workload fills of Axis 1
- `../pitfalls/platform-known-issues.md ## Toolchain-pin regressions` — when the number you measured
  is a compiler regression rather than a ceiling, and what rung 4 can do about it
