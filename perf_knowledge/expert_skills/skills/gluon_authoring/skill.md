---
id: gluon_authoring
title: "Author Gluon on CDNA: the language surface, the do-not-write list, and a deterministic TTGIR→Gluon first round"
kind: expert_skill
authors: [qiongz]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  operator: dense_gemm
  arch_class: ['*']
  gens: [gfx942, gfx950]
  dtypes: [bf16, fp16, fp8_e4m3_fnuz]
  regimes: [prefill, training]
  # triton = port a tuned plain kernel; gluon = the source is already Gluon, optimize it in place
  from_backend: [triton, gluon]
  to_backend: gluon
# ---- expected effect: the validation gate's pass criteria ----
expects:
  isolated_speedup_min: 1.10
  parity: required
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use

The workflow is working a **Gluon** kernel on gfx942/gfx950 and needs two things this file supplies: the
**Gluon API surface**, and the **do-not-write list** — the constructs and instruction choices that compile
and then silently cost you.

**Two entry states, and they do not share a round 1:**

- **Porting from plain Triton** (`from_backend: triton`). A tuned plain kernel exists, so the opening move
  is the fixed transcribe-then-re-inject pair in `## Procedure`, closing against that kernel at **≥95%**.
  Round 1 carries the transcription to layout equivalence; the re-injection may take round 2 when plain is
  auto-pipelined. Two rounds is the ceiling.
- **Already Gluon** (`from_backend: gluon`). Nothing to transcribe and no plain anchor to hold parity
  with, so skip step 1 and step 3's gates; keep step 2 only if the loop is not already pipelined, and go
  straight to GEAK's ordinary loop with the API pages and the do-not-write list as reference.

**Everything about process stays with GEAK.** This skill carries no round loop, no budget model, no
profiling pipeline, no lever catalogue and no escalation gate. Run your normal `kernel_workflow` /
`e2e_workflow` loop; the one difference is that **a port needs no direction fan-out while it is landing**,
because those moves are deterministic rather than hypotheses worth exploring several ways (see
`## Procedure`). Once the port is at parity it is your ordinary loop, and what to try next is your call,
not this file's.

Do not use it to decide *whether* Gluon is the right direction. Short version: Gluon pays only when the
residual is layout-shaped, because LDS swizzle/padding choice and LDS dedup are the two things plain
Triton has no syntax for.

**Lazy-loading contract.** Read this file plus [`reference.md`](reference.md). Everything under
[`references/`](references/) is lazy — 14 files, ~2 k lines, load one only when you reach for the
construct it documents.

## Mechanism

**What Gluon is, and what becomes yours to author, is not repeated here** — that onboarding already lives
in GEAK's language layer: [`languages/gluon/overview.md`](../../../languages/gluon/overview.md) (what
Gluon is versus Triton, when to reach for it) and
[`programming_model.md`](../../../languages/gluon/programming_model.md) (explicit layouts, pipeline
stages, register budgeting, MFMA intrinsics, the two CDNA wave patterns). Read those first if you have not
written Gluon before. This section covers only what that layer does not: why the opening move on a
*migration* is mechanical, and why it takes two steps rather than one.

**Why the opening move is deterministic.** The layouts plain Triton's compiler inferred are recorded in
the tuned kernel's `.ttgir`, so the first Gluon version is not a design exercise — it is a mechanical
re-expression of layouts that already exist, and `scripts/ttgir_to_gluon.py` performs it. There is nothing
to search over: any two agents transcribing the same pinned `.ttgir` should land on the same layouts. That
is why it is one fixed action instead of a fan-out. Deterministic does not mean single-shot: `--verify`
is a diff you converge on over several passes, it just converges to one answer.

**Why the port also has to re-inject the pipeline to land.** `gluon_to_ttgir` does not run plain's
`add_schedule_loops` + `add_pipeline`, so a transcription alone loses plain's cross-iteration overlap and
reads as a regression for a reason that has nothing to do with your layouts. Those two passes are
**re-injectable**: at the TTGIR level an explicit Gluon loop is a non-pipelined `scf.for`, the kind of
loop the AMD pipeliner is written to consume, and nothing about the Gluon path makes them inapplicable.
Splicing them back is a Python-only `compiler.py` edit — **no `libtriton.so` rebuild**, the passes are
already in the `.so`.

**Re-injectable is not the same as certain to bite, and the two steps pull against each other.** The
pipeliner's job is to *create* the LDS staging and the overlap around it, so what it needs is a loop that
still has that work left in it. A transcription that hand-authors `allocate_shared_memory` +
`local_store` + `local_load` — which is how you most directly express a recovered `#shared` / `#dot_op`
pair — has already done that work by hand, and the pass will run over it and change nothing. So step 1
can walk you straight into the shape step 2 cannot act on. Step 2 says how to keep room for it and how to
tell the difference between a pass that is absent and a pass that fired and did nothing. Near-parity is
what the two steps together are *aiming* at — it is what the ≥95% gate encodes, not a property the port
arrives with.

## Procedure

The port **from plain Triton**, from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`. Steps 1 and 2 fit in round 1 when the
transcription converges quickly, and may span two rounds when it does not — step 3 says which gate is due
when. Coming in with a kernel that is already Gluon, start at step 2 and treat step 3 as not applicable.

**Before step 1, settle the comparator, and then stop moving it.** The kernel you pin must be the plain
champion at *its own best config*, and the `.ttgir` you dump must come from that config: every number
below is relative to it, and the layouts you recover are the ones that config produced. Two consequences
worth knowing up front. A port measured against a shipped default is not a port that landed, however good
the ratio looks. And the anchor is **bound to the config it was dumped at** — the recovered layouts carry
literal `warps_per_cta` and tile extents, so if the champion's best config differs per shape bucket you
dump and transcribe per bucket rather than expecting one anchor to follow it.

**1. Transcribe, and drive it to layout equivalence.** Pin the tuned plain kernel, dump its IR, recover
the layouts, and iterate until `--verify` passes:

```bash
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest    # 10 s, offline: is the converter itself sane?
bash    "$SKILL/scripts/dump_ir.sh" <compile cmd> --variant plain --out ir/ --emit-gluon layouts
python3 "$SKILL/scripts/recover_gluon.py" ...            # calls ttgir_to_gluon.py underneath
python3 "$SKILL/scripts/recover_gluon.py" ... --verify   # layout equivalence — do not skip
```

**Budget several passes here, not one.** `--verify` is a diff, and the expected shape of round 1 is
verify → read the missing/extra layout attributes it names → fix that one layout → recompile the anchor →
verify again, until it reports PASS. Run the converter's `--selftest` first so that a failure is
attributable to your kernel rather than to the tool. Do not proceed to step 2 on a FAIL and do not
hand-wave a near-match: `--verify` is the only check that catches a transcription which **passes the
numeric oracle while having recovered the layout wrong** (wrong `order`, wrong `kWidth`), and a
numerically-correct wrong-layout anchor poisons every delta you measure afterwards.

**The recovered layouts arrive as a preamble; applying them is yours.** `recover_gluon.py` emits a
`gl.constexpr` block holding the layouts plain's compiler inferred, and — under `--with-skeleton` — a
kernel body beside it that still carries the translator's own default MMA layout and `AutoLayout`.
Nothing connects the two. Wiring each recovered layout onto the operand it belongs to is the manual step
that turns the file into an anchor, and it is not optional: a layout the body never mentions never
reaches the TTGIR, so `--verify` will report exactly those layouts missing. Read a `missing` list that
matches your preamble as "declared but not applied" rather than as "recovered wrong".

The converter covers `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`,
`ttg.dot_op` and `ttg.slice`. Two gaps that are **not** tool bugs, so do not spend verify cycles on them:
`ttg.convert_layout` placement is manual — take it from the recovery map in
`references/tile-programming/layout-recipes.md`; and `amd_rotating_shared` has no `gluon.language`
constructor at all.

**2. Re-inject the pipeline.** Do it in round 1 if step 1 converged quickly; **if the plain kernel is
auto-pipelined, you may take a second round for this step alone** rather than compressing both into one
(see the gate in step 3). Check the build first, then splice into `gluon_to_ttgir` after
`add_combine_tensor_select_and_if`, gated on `options.num_stages > 1` so it stays cache-key-safe and
default-OFF:

```bash
# read the reinject_ttgir_pipeliner entry: are the passes in THIS libtriton.so?
python3 "$SKILL/scripts/probe_levers.py" --all --arch gfx950
```
```python
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
; add_convert_to_tensor_ops; canonicalizer; remove_layout_conversions; reduce_data_duplication
; (in_thread_transpose if enabled); move_up_prologue_loads; canonicalizer; cse
```

**The splice point moves with the version** — read the installed
`third_party/amd/backend/compiler.py` before editing, and expect a vendor fork to possibly carry its own
variant of this patch already:

| installed version | splice after | adjust |
| --- | --- | --- |
| 3.6 | `add_combine_tensor_select_and_if`, which ends the function | gate on `knobs.amd.use_async_copy`; drop `add_convert_to_tensor_ops`; `add_reorder_instructions` in place of `move_up_prologue_loads` |
| 3.7, 3.8 | same, but **before** `add_warp_pipeline` | the list above as written |

Give the pass room or it silently does nothing. Three ways to starve it, the first of which step 1 walks
you straight into:

- **Hand-authored LDS staging.** `allocate_shared_memory` + `local_store` + `local_load` in the loop body
  *is* the pass's output, written by you — it wants to create that staging and finds it already there, so
  it rewrites nothing. On a plain GEMM this is the likeliest reason the counts do not move — and the
  other two below do not apply there, so satisfying both of them still leaves you here. The fix is to
  un-write part of the transcription: let the operands arrive as in-body loads and reach their
  dot-operand layout through `convert_layout`, and the pass has something to stage. Keep the
  explicit-smem version — it is where you start the hand-built double buffer if the pass still will not
  bite.
- **A hand register-prefetch.** The pipeliner *is* the prefetcher; a manual one consumes the slot.
- **A loop-variant `scf.if`.** Split a causal mask into two loops; it blocks both this pass and
  `BlockPingpong`.

Launch `num_stages=2`. The recipe, and the hand-built cross-iteration double buffer that is the remaining
route once the pass is established as inert on hand-authored smem, are in
`references/tile-programming/pipeline.md` (it calls that one "Route 2"). Its vetted skeleton indexes the
buffer as `s.index(i % 2)` — see the first entry in `## Knobs & pitfalls` before you copy that.

**Confirm it fired before you time anything**, and expect to go round this loop more than once too: dump
the Gluon TTGIR again and compare its `local_alloc` / `local_store` / `local_load` counts against plain's,
and check that the full-drain `s_waitcnt lgkmcnt(0)` has become a relaxed `lgkmcnt(N>0)`. Those counts
moving toward plain's is the signal the pass landed; a throughput number is not, because a pass that
silently did nothing and a pass that fired but did not help look identical on the clock.

**Dump once with the splice gated off as well.** Counts identical on and off means the pass ran and
transformed nothing — a different failure from the passes being absent, and the one `probe_levers.py`
cannot see: `available: true` reports that the symbols are in this `libtriton.so`, not that they will
bite on your IR. Separating those two is the whole point of reading the IR back.

**3. Close on two gates — and know which round each one belongs to.**

| gate | what it is | due |
| --- | --- | --- |
| Layout equivalence + bit-parity | `--verify` PASS, and no numeric delta vs the plain anchor (transcription is layout-only, so any delta is a bug, not a Gluon property) | **end of round 1, always** |
| **≥95% of the tuned plain kernel's throughput** | the port has actually landed | end of round 1 — **or end of round 2 if the plain kernel is auto-pipelined** |

The relaxation exists because the two steps fail differently. Step 1 converges on a diff you can read, so
it belongs in one round. Step 2 is an edit to `compiler.py` whose effect you can only confirm by
recompiling and reading the IR back, and on an auto-pipelined kernel that is normally a few
adjust-and-recheck cycles — **spend round 2 on it rather than declaring round 1 a failure**. If plain was
not auto-pipelined there is nothing to reproduce and both gates are due in round 1.

**Two rounds is the ceiling for the port itself.** Still short of 95% after re-injection has been
confirmed to fire? Then it is no longer a transcription problem — hand it back to GEAK's ordinary loop as
a normal optimization target rather than spending a third round here.

Below 95%, do not start optimizing layouts. The cause is almost always one of these — check them in this
order, cheapest first. Note that the first two share a symptom (`local_alloc` / `local_store` /
`local_load` counts in the Gluon TTGIR did not move toward plain's, and the full-drain
`s_waitcnt lgkmcnt(0)` never relaxed), which is why the on/off dump is what separates them:

1. **the pass never ran** — `num_stages` left at 1 so the gate stayed off, or the splice went in at the
   wrong point for this version (re-read the table above). Counts differ between splice-on and
   splice-off only if it ran at all,
2. **it ran and rewrote nothing**, because the body left it no work: counts identical on and off.
   Hand-authored LDS staging is the usual reason on a GEMM, a hand prefetch or a loop-variant mask on
   everything else,
3. **a layout recovered wrong, or recovered and never wired onto an operand** — re-run `--verify` rather
   than eyeballing it, and check its `missing` list against your own preamble before concluding the
   recovery itself was wrong.

The port ends at parity, not at a win. **Once both gates are met it is GEAK's ordinary loop** — the layout
levers plain cannot express are the reason you came here, and which one to spend a direction on is the
workflow's decision, informed by its own profile.

## Knobs & pitfalls

Do-not-write list, i.e. what compiles and then costs you:

- **Runtime buffer indices.** `smem.index(k % nBuffers)` is an anti-pattern: the scheduler cannot prove
  overwrite-safety and refuses to interleave. Buffer indices must be compile-time constant, and
  `wait_group(N)` must be recomputed whenever the prologue, region or unroll factor changes. **The vetted
  double-buffer skeleton in `references/tile-programming/pipeline.md` writes exactly this** (`cur = i % 2`)
  — it is upstream text and this list overrides it. Unroll the loop by 2 so each index is a literal.
- **A hand register-prefetch next to a re-injected pipeliner.** Not additive — it consumes the slot the
  pass wanted.
- **The LLIR scheduler toggle on anything with VALU between the matmuls** (softmax, scale, dequant). It
  assumes a pure MFMA→MFMA accumulator chain and emits **invalid IR**, not a slowdown. Default-skip on
  attention shapes. **Fork-only**: `TRITON_ENABLE_LLIR_SCHED`, `TRITON_ENABLE_AMDGCN_AS` and
  `TRITON_ENABLE_AMDGPU_RA_HINTS` (`dump_ir.sh --knobs LLIR_SCHED|AMDGCN_AS|RA_HINTS`, and the
  `gemm_compiler_stack` probe's `llir`/`ra` rungs) exist in no upstream Triton — 3.6, 3.7 and 3.8 alike.
  On a stock build they export an env var nobody reads, so the knob is a silent no-op rather than an
  error. Confirm with `probe_levers.py --all` before attributing any delta to them.
- **`disableSched`** — costs occupancy outright. Never on this path.
- **Deep pipelines by reflex.** Software prefetch regresses once waves are already VGPR- or LDS-capped,
  and aggressive unrolling can raise `s_nop` count rather than lower it. `num_stages=2` is the start;
  deepen only against a profile.
- **`convert_layout` scratch after the pipeliner pass.** Scratch allocated post-pass is outside hazard
  analysis and fails silently.
- **`amd_rotating_shared`** — has no `gluon.language` constructor; do not hand-roll a basis for it.
- **RDNA WMMA formulas.** Not vendored and not applicable; `match.gens` is CDNA.
- **Sweeping `num_warps` on a transcribed kernel.** `ttgir_to_gluon.py` emits the IR's *literal*
  `warps_per_cta`, so any other warp count disagrees with the layouts — a correctness bug, not a slow
  config. `triton.autotune` itself works on `gluon.jit` and upstream's own Gluon examples use it, but they
  sweep tile shapes and stage counts with `num_warps` pinned. Tile-shape autotuning is GEAK's to do after
  the port lands; only the warp count is off the table.

`num_stages` itself is not a Gluon dead knob — it is a buffer-count budget on the default lowering and the
pipelining trigger again after re-injection, so carry the plain winner's value over as a starting depth.
Full lists: `references/gluon-negative-patterns.md`, `references/platform-known-issues.md`.

## Do-no-harm notes

- **Advisory only.** This file supplies API surface and mechanics, never a verdict. The workflow's
  isolated A/B against the immutable oracle decides, and a result below the measured baseline is a
  negative whatever this says.
- **It does not narrow the workflow's search.** Only the port is fixed — at most two rounds, and only
  because those moves are mechanical; every round after it is GEAK's own loop and every matched skill
  enters the candidate set rather than pre-empting it.
- **A Gluon result slower than its comparator is a revert.** A Gluon wall does not prove the comparator
  was at its ceiling.
- **On a port, the comparator must be the plain kernel tuned to its own best config.** A win over a
  default strawman is invalid, and so is a 95% measured against one. Starting from an existing Gluon
  kernel there is no plain comparator at all — the workflow's own frozen baseline is the floor, and the
  ≥95% bar does not apply.
- **Correctness gates before timing**, and round 1 additionally gates on layout equivalence — a
  numerically-correct wrong-layout anchor poisons every later delta, which is why that gate is never the
  one allowed to slip into round 2.
- **Do not read a missing pipeline as a ceiling.** It is the auto-pipeliner that `gluon_to_ttgir` skips by
  default. An older note that re-injection "regresses" was a specific misfire — a hand-async loop with no
  in-body `tt.load` to anchor on, plus the `disableSched` cliff — not the passes.
- **Keep it off decode / skinny-M shapes.** Memory-latency-bound with no tile structure worth
  re-laying-out; authoring cost cannot be repaid there.

## Sources

- Vendored from `AMD-AGI/TileProgrammingAgentSkills@c14a583`
  (`.cursor/skills/tile-programming-triton-gluon/`), **pruned to the API surface, the do-not-write lists
  and the two mechanics above**: 14 of 72 reference files (~2 k of ~11 k lines) and 6 of 41 scripts.
  Upstream remains the SSOT; this is a one-way snapshot. `references/` is unmodified. `scripts/` carries
  **one correction**, which should go back upstream: `ttgir_to_gluon.py` dropped `tilesPerWarp` and
  `elementBitWidth` when emitting `gl.amd.AMDMFMALayout`, so a chained-dot (gfx950, 16×16 mfma) or
  scaled-MFMA kernel — both of which make `AccelerateAMDMatmul` choose non-default values — transcribed
  to a silently different layout. `--verify` caught it as a text mismatch but named no cause. Both are
  optional `AMDMFMALayout` fields in 3.6 through `main`; they are now emitted when, and only when, the
  TTGIR prints them, with a self-test on both directions.
- **Deliberately not vendored — the whole process layer**: round loop, hardware budget / roofline models,
  profiling (rocprof / ATT / PMC), the lever-card catalogue, bound-class signals, the escalation gate,
  orchestration, experiment records, benchmark hygiene, the transcription *protocol* page, and the
  workload strategy pages. GEAK's workflows own process; a second regime inside an advisory prior would
  compete with them. Retained files still cite the dropped paths in 64 places — **dead by design**, do
  not hunt for them and do not read a citation as an instruction to rebuild the regime. For the hardware
  facts among them (instruction availability per gen, LDS banking, VGPR/AGPR budget, occupancy), use
  GEAK's own [`perf_knowledge/hardware/`](../../../hardware/) (`cdna3_mi300/`, `cdna4_mi350/`, `shared/`)
  and [`perf_knowledge/languages/gluon/`](../../../languages/gluon/).
- Onboarding and concepts are **not duplicated here**; they live in
  [`languages/gluon/`](../../../languages/gluon/) (`overview.md`, `programming_model.md`,
  `gemm_cookbook.md`), which is also where the AMD-measured Gluon GEMM ceilings belong. This skill adds
  the API-page detail, the do-not-write list, and the two migration mechanics.
- Gluon language surface upstream: ROCm Gluon GEMM tutorial
  (https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html), gfx950 Gluon
  tutorials (https://github.com/ROCm/gfx950-gluon-tutorials).
- Pipeline re-injection — pass list, the `num_stages>1` gate, kernel-side conditions, proof-it-landed
  signals: `references/tile-programming/pipeline.md`
  `## Reproduce plain's software pipeline on the Gluon path`.
- **No measurement is baked into this file, by design.** It carries no timings, speedups, TFLOPS,
  occupancy percentages or bucket splits — those belong in the operator SOTA cards (`operators/<op>/`) and
  the language docs, which are versioned against a specific SKU, container and date and go stale on their
  own schedule. A method file that hard-codes yesterday's numbers ages badly and invites reasoning from
  them instead of from measurement.
- The **≥95%** parity bar is a **target**, not a measured result: transcription is layout-only and
  re-injection is what restores plain's own overlap, so near-parity is what the two steps are aiming at —
  reachable when the re-injected pass actually bites, which is why step 2 makes you confirm that from the
  IR instead of assuming it. The bar exists to trigger the diagnostic list above rather than to record an
  achievement. The same reasoning sets its deadline — one round when there is no pipeline to reproduce,
  two when there is, because the second step's cost is adjust-and-recheck cycles rather than search.
  Nothing here has been
  measured on-box in GEAK — hence `validation.status: draft` and no auto-application.
  `expects.isolated_speedup_min: 1.10` is the template floor for the skill's eventual win, not the parity
  bar.
- **Vendored files under `references/` are unmodified and do contain upstream's own measured claims.**
  Treat those as upstream's evidence, dated to upstream, and not as a GEAK measurement.
