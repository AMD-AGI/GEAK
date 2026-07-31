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
**re-injectable**: at the TTGIR level an explicit Gluon loop is just a non-pipelined `scf.for`, exactly
the object the AMD pipeliner consumes. Splicing them back is a Python-only `compiler.py` edit — **no
`libtriton.so` rebuild**, the passes are already in the `.so`. Transcription plus re-injection is what
makes near-parity with plain the expected outcome of the port rather than a hoped-for one.

## Procedure

The port **from plain Triton**, from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`. Steps 1 and 2 fit in round 1 when the
transcription converges quickly, and may span two rounds when it does not — step 3 says which gate is due
when. Coming in with a kernel that is already Gluon, start at step 2 and treat step 3 as not applicable.

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
python3 "$SKILL/scripts/probe_levers.py" reinject_ttgir_pipeliner   # are the passes in THIS libtriton.so?
```
```python
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
; add_convert_to_tensor_ops; canonicalizer; remove_layout_conversions; reduce_data_duplication
; (in_thread_transpose if enabled); move_up_prologue_loads; canonicalizer; cse
```

Give the pass room or it silently does nothing: stream the operands **in-body with no hand
register-prefetch** (the pipeliner *is* the prefetcher — a manual one consumes the slot), and **split a
causal mask into two loops**, since a loop-variant `scf.if` blocks both this pass and `BlockPingpong`.
Launch `num_stages=2`. Recipe and the hand-built cross-iteration double buffer for later rounds:
`references/tile-programming/pipeline.md`.

**Confirm it fired before you time anything**, and expect to go round this loop more than once too: dump
the Gluon TTGIR again and compare its `local_alloc` / `local_store` / `local_load` counts against plain's,
and check that the full-drain `s_waitcnt lgkmcnt(0)` has become a relaxed `lgkmcnt(N>0)`. Those counts
moving toward plain's is the signal the pass landed; a throughput number is not, because a pass that
silently did nothing and a pass that fired but did not help look identical on the clock.

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

Below 95%, do not start optimizing layouts — the cause is almost always one of these, in order of how
often it is the answer:

1. the pipeliner did not fire (`local_alloc` / `local_store` / `local_load` counts in the Gluon TTGIR did
   not move toward plain's; full-drain `s_waitcnt lgkmcnt(0)` never became relaxed `lgkmcnt(N>0)`),
2. a hand prefetch left in place, or the mask still in one loop, so there was no room for it,
3. a layout recovered wrong — re-run `--verify` rather than eyeballing it,
4. `num_stages` left at 1, so the re-injected passes stayed gated off.

The port ends at parity, not at a win. **Once both gates are met it is GEAK's ordinary loop** — the layout
levers plain cannot express are the reason you came here, and which one to spend a direction on is the
workflow's decision, informed by its own profile.

## Knobs & pitfalls

Do-not-write list, i.e. what compiles and then costs you:

- **Runtime buffer indices.** `smem.index(k % nBuffers)` is an anti-pattern: the scheduler cannot prove
  overwrite-safety and refuses to interleave. Buffer indices must be compile-time constant, and
  `wait_group(N)` must be recomputed whenever the prologue, region or unroll factor changes.
- **A hand register-prefetch next to a re-injected pipeliner.** Not additive — it consumes the slot the
  pass wanted.
- **The LLIR scheduler toggle on anything with VALU between the matmuls** (softmax, scale, dequant). It
  assumes a pure MFMA→MFMA accumulator chain and emits **invalid IR**, not a slowdown. Default-skip on
  attention shapes.
- **`disableSched`** — costs occupancy outright. Never on this path.
- **Deep pipelines by reflex.** Software prefetch regresses once waves are already VGPR- or LDS-capped,
  and aggressive unrolling can raise `s_nop` count rather than lower it. `num_stages=2` is the start;
  deepen only against a profile.
- **`convert_layout` scratch after the pipeliner pass.** Scratch allocated post-pass is outside hazard
  analysis and fails silently.
- **`amd_rotating_shared`** — has no `gluon.language` constructor; do not hand-roll a basis for it.
- **RDNA WMMA formulas.** Not vendored and not applicable; `match.gens` is CDNA.

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
  Upstream remains the SSOT; this is a one-way snapshot. The vendored `references/` and `scripts/` are
  unmodified.
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
  re-injection restores plain's own overlap, so near-parity is the structurally expected outcome of the
  port and the bar exists to trigger the diagnostic list above rather than to record an achievement. The
  same reasoning sets its deadline — one round when there is no pipeline to reproduce, two when there is,
  because the second step's cost is adjust-and-recheck cycles rather than search. Nothing here has been
  measured on-box in GEAK — hence `validation.status: draft` and no auto-application.
  `expects.isolated_speedup_min: 1.10` is the template floor for the skill's eventual win, not the parity
  bar.
- **Vendored files under `references/` are unmodified and do contain upstream's own measured claims.**
  Treat those as upstream's evidence, dated to upstream, and not as a GEAK measurement.
