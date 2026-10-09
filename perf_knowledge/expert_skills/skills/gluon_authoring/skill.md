---
id: gluon_authoring
title: "Author Gluon on AMD CDNA — gfx950 first, gfx942 as the downgrade: assert the champion, budget it, transcribe faithfully, recover the debt by hand, climb a hand-written pipeline — inside GEAK's own roles and round loop"
kind: expert_skill
authors: [qiongz]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  # Not operator-gated. What decides whether this skill pays is the STATE OF THE SOURCE
  # (see `requires` below), not what the kernel computes: the same kernel and the same
  # starting point can yield a clear win or a clear loss depending on how the transcription
  # is done, so an operator filter selects the wrong thing in both directions.
  operator: '*'
  arch_class: ['*']
  # gfx950 (CDNA4) is the main line; gfx942 (CDNA3) is supported as a downgrade.
  gens: [gfx950, gfx942]
  # fp8 is spelled per generation, not once: CDNA4 is OCP and CDNA3 is FNUZ, and
  # `index/capability_index.yaml` uses both names — so claiming only one spelling makes this
  # skill silently unselectable on the other generation's fp8 bottleneck.
  dtypes: [bf16, fp16, fp8_e4m3, fp8_e4m3_fnuz]
  regimes: ['*']   # decode included -- see the admission notes in `## When to use`
  # triton = port a tuned plain kernel; gluon = the source is already Gluon, optimize it in place
  from_backend: [triton, gluon]
  to_backend: gluon
# ---- entry precondition: the one gate that IS predictive ----
# A port measured against an unoptimized plain kernel measures the config sweep, not Gluon.
requires:
  from_backend_triton:
    # all three must hold before the first round; if any fails, tune plain FIRST and re-enter
    - "the plain source is at its own best config (a config sweep has run and its winner is pinned)"
    - "`plain@ns=1` has been measured alongside it, so a mistuned `num_stages` cannot be
       mistaken for a Gluon win"
    - "the comparator recorded for every later number is that tuned kernel, never the shipped default"
  from_backend_gluon:
    - "the incumbent is a measured, asserted number on this GPU and container — the workflow's own
       frozen baseline is the floor, and the plain-parity gate does not apply"
# ---- expected effect: the validation gate's pass criteria ----
# Measure this over the WHOLE track (entry -> close), not over the transcription: the port on its own
# is built to land near parity, not above it, so a validation run that stops at the parity gate has
# measured the transcription rather than the skill. See `## Sources`.
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

You are taking a GPU kernel to peak in **Gluon** (`triton.experimental.gluon`, explicit tile programming)
on AMD CDNA. **gfx950 (CDNA4 — MI355X / MI350X) is the main line** of everything below; **gfx942 (CDNA3 —
MI300X / MI325X) is the downgrade**, written as "what is unavailable or changes there" after each gfx950
mechanism. Gluon/Triton usage follows **upstream Triton 3.8.0**; 3.6 / 3.7 differences appear only as
downgrade notes next to the mechanism they affect.

This skill is the complete upstream `tile-programming-gluon` method, reorganized for GEAK: a hardware
budget, a champion assertion, a faithful transcription, an attributed recovery of what transcription lost,
and a climb through explicit layers whose pipeline is **written by hand**. It carries the Gluon API surface,
the do-not-write lists and the tools, mapped onto GEAK's existing roles (`## Roles in GEAK`).

**Three entry modes** — settle which one you are in before anything else
([`references/method/entry.md`](references/method/entry.md)):

| mode | you hold | the comparator | what changes |
| --- | --- | --- | --- |
| **A. Port** (`from_backend: triton`) | a tuned plain-Triton champion (`plain_champion.json`) | `champion_ms`, the plain kernel at its own best config | full spine: transcribe → recover to parity → climb |
| **B. Incumbent** (`from_backend: gluon`) | a kernel that is already explicit Gluon (`@gluon.jit` with explicit `gl.*Layout` — read the body, not the name) | the incumbent's asserted, re-measured time | transcribe / recover / parity are a recorded **no-op**; climb is the whole procedure; `parity_gate.py` must **not** be run as a gate (equal sides → a vacuous CLEARED) |
| **C. Re-entry** | a checkpointed run | the checkpoint's comparator, re-asserted | resume at the recorded stage; re-verify before trusting a carried number |

Neither a bundle nor a measured incumbent is a **return to the front end** — this skill has no plain tier.

**How it runs in GEAK — existing roles, no extra agents**
([`references/method/orchestration.md`](references/method/orchestration.md)). GEAK's `kernel_workflow`
injects this skill into its own roles (`use_expert_skills=true`); GEAK owns the round loop, the GPU lock,
timing (`harness_lib`) and acceptance (`verify_engineer`, Director, `MIN_IMPROVE`). The skill is an
**advisory prior**: follow its stages and evidence rules inside GEAK's rounds; it never overrides an on-box
measurement. Who does what is in `## Roles in GEAK` below.

**Whether Gluon pays.** Gluon pays when the residual is something plain Triton cannot express: an explicit
LDS swizzle/padding choice, LDS dedup, per-operand buffering, an authored pipeline or wave schedule, an
explicit memory path. A residual the plain front end could still sweep away is a `resweep_request`, not a
Gluon round. Use [`references/pitfalls/negative-patterns.md`](references/pitfalls/negative-patterns.md)
`Quick Reject Checklist` as an **admission hint** (paged / indirect decode, no tile structure, …) — it is advice
about cost, not a shape filter: the state of the plain source decides, so check `requires` first.

**What is invariant, whatever the route.** Three things make numbers unfalsifiable rather than merely
worse, so they never bend: the **equivalence gate** on the anchor (a numerically-correct wrong-layout anchor
poisons every later delta), **never mixing transcription with optimization in one edit**, and a **frozen
comparator** (re-tune plain before the track or after it, never during). Everything else is a route chosen
from measurement.

**Run the track as one `deep_explore` direction, not a fan-out.** Transcription is deterministic —
parallel arms measure the same layouts and share the same bug — and the climb is stateful: what to try next
comes from the IR and profile of the anchor the same engineer just built. So tech_lead dispatches **one**
`deep_explore` direction and the `deep_engineer` carries the whole track in its own loop; bounded
measurements (a timing, a knob probe, an anomaly check) are run by that engineer itself under `gpu_lock.sh`
(`references/method/records.md ## 9. Experiment / hypothesis-test contract`).

**Tell the loop the track's shape.** A port lands **below** the comparator and climbs back; at
the harness defaults that phase is invisible (no candidate, no saved patch, no verify, the loop stops two
rounds in). There is no port mode — there are four launch args, checked as gate **G0** before dispatch:

| arg | default | pass on a port (mode A) | why the default breaks a port |
| --- | --- | --- | --- |
| `candidate_floor` | `1.0` | from the debt `parity_gate.py` reports | a faithful anchor is below the comparator by construction and never enters the candidate list |
| `max_no_improve` | `2` | `6` | ends the run before recovery finishes |
| `budget` | `6` | `20` | counts directions; a `deep_explore` direction costs 2, so 20 ≈ 10 rounds |
| `progress_delta` | `+min_improve` | `-0.05` | a port that gives ground while exploring reads as a stall |

On mode B leave all four at their defaults — the mirror-image mistake keeps a stalled search alive. Run a
port at **`mode: optimize`** (author mode would overwrite the source being transcribed), steer the round to
a single **`deep_explore`** direction, and never let a ground-up rewrite replace the transcription. The
measured calibration behind these values: `references/method/entry.md`.

```js
Workflow({
  scriptPath: "<REPO>/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path: "<TASK_DIR>", workflow_dir: "<REPO>/kernel_workflow",
    mode: "optimize", target_language: "gluon", use_expert_skills: "true",
    budget: 20, max_no_improve: 6, candidate_floor: 0.5 /* reset from parity_gate */, progress_delta: -0.05,
    gpu_ids: "0",
  },
})
```

**Lazy-loading contract.** Read this file, then [`reference.md`](reference.md) — the router from each
section here to the one reference file that continues it. Everything under `references/` is lazy: load the
file the current stage names, and query one section by heading rather than reading a tree.

## Arch dispatch: gfx950 (CDNA4) baseline, gfx942 (CDNA3) downgrade

The mechanics are shared across both — the passes, the stage spine, recover→apply→verify. What is **not**
shared is this table, and every row can change a verdict rather than a magnitude. **Numbers live in
[`perf_knowledge/hardware/data/hw_constants.json`](../../../hardware/data/hw_constants.json)** (per-arch)
and [`sku.json`](../../../hardware/data/sku.json) (per-SKU peaks), never in prose: take the value from the
named key and pass `--arch` explicitly everywhere — the tools refuse rather than default it.

| what | gfx950 — the baseline | gfx942 — the downgrade | key | why it flips a conclusion |
| --- | --- | --- | --- | --- |
| **LDS per CU** (occupancy divisor) | 160 KiB | 64 KiB | `lds_per_cu_kib` | a depth or tile that is "unavailable" on gfx942 may simply fit on gfx950; never carry the divisor across |
| **LDS banking** | 64 banks, and a 2-way `ds_read_b128` stride | 32 banks | `lds_banks`, `ds_read_b128_full_conflict_stride_bytes`, `ds_read_b128_2way_stride_bytes` | a conflict-free swizzle on one is not known to be on the other; the sign can invert |
| **Async direct-to-LDS** | `cdna4.async_copy` + `commit_group` / `wait_group`, 32/64/128-bit per lane | **32-bit only**, clean tiling, destination `order=[1, 0]`; anything else fails LLVM translation | `direct_to_lds_bit_widths` | gfx950's authored LDS ring is async; gfx942's is sync staging (`allocate_shared_memory` + `.store()`/`.load()` + barrier) |
| **Read-with-transpose** | `ds_read_b64_tr_*` — the compiler emits it instead of `amdg.in_thread_transpose` | absent | `ds_read_tr` | on gfx950 the "Gluon cannot express in-thread transpose" gap has no subject; on gfx942 it does |
| **MFMA family** | `version=4`, 16×16 / 32×32 shapes, block-**scaled** MFMA (MXFP8/6/4) | `version=3`, no scaled MFMA | `matrix_layout_family`, `mfma_cadence_cyc`, `scaled_mfma` | layout digests **should** differ across gens — assert consistency within one arch, never cross-gen equality |
| **fp8 spelling** | OCP `fp8_e4m3` | FNUZ `fp8_e4m3_fnuz` | `fp8_dtype` (absent on gfx950 — do not fall back to gfx942's) | Triton silently upcasts the wrong spelling to fp16 with only a warning |
| **The omitted post-pipeline tail** (re-injection only) | silent slowdown — the larger ceiling absorbs it | `OutOfResources` | — | check the pass list, not the exception |
| **Ping-pong reachability** | probe the ISA for `s_setprio` | same | — | the predicate is read only inside `make_ttgir`, which the Gluon path skips; judge from the ISA |

**Keys that are traps rather than gaps on gfx950.** `fp8_dtype` (above); `lds_min_alloc_bytes` — gfx950's
`lds_align_bytes` is **not** the allocation granularity despite looking like it; `cvt_off_f32_i4` is a
binding-availability record, not a silicon fact. **Two gaps no arch closes**, because they are Gluon-side:
`amd_rotating_shared` has no `gluon.language` constructor, and a user `allocate_shared_memory` is totalled
**by scope** where plain's allocator peaks **by liveness** — read `shared` bytes off the artifact.

RDNA (gfx1201 / gfx1151) is out of `match.gens`; the WMMA downgrade surface is documented in
[`references/gluon/rdna-wmma-reference.md`](references/gluon/rdna-wmma-reference.md) for reference only.

## Mechanism

Onboarding — what Gluon is versus Triton, explicit layouts, MFMA intrinsics — is GEAK's language layer:
[`languages/gluon/overview.md`](../../../languages/gluon/overview.md) and
[`programming_model.md`](../../../languages/gluon/programming_model.md). This section covers what that
layer does not: why a migration opens mechanically, what it costs, and the order the explicit layers are
climbed in.

**The opening move is deterministic.** The layouts plain Triton's compiler inferred are recorded in the
champion's `.ttgir`, so the first Gluon version is a re-expression, not a design: `ttgir_bridge.py` hands the
TTGIR to the compiler's own parser and upstream's `layoutToGluon()`, so there is no mapping table to fall
behind Triton, and an unsupported kind surfaces as a named `UNRECOVERABLE` row instead of a plausible wrong
constructor. Any two agents transcribing the same pinned dump land on the same layouts.

**A faithful anchor sits below the champion by construction — a debt, not a ceiling.** `gluon_to_ttgir`
runs no software pipeliner and no block ping-pong (on any of 3.6.0 – 3.8.0), so the overlap plain's
`num_stages` bought does not survive transcription; nor does any allocator luck. The debt has exactly
three owners — **`lost_pipeline`**, **`lost_layout`**, **`lost_RA`** — and `parity_gate.py` splits the
anchor→champion gap across them from the compiled artifacts. The debt is repaid **by hand**, attributed,
before any delta may be called a win. It is often zero: a champion that compiled at `num_stages=1` had no
overlap to lose, and one whose shipped depth is a pessimisation would lose nothing worth recovering —
which is why recovery opens with the `plain@ns=1` control rather than with a fix.

**`num_stages` is dead on the Gluon path.** No 3.8.0 pass consumes it from a Gluon kernel; it survives
only as a budget parameter and as part of the champion's record. Do not carry plain's value over as a
tuning knob.

**The pipeline is written by hand, in this order** (the single definition is
[`references/tile-programming/pipeline.md`](references/tile-programming/pipeline.md)):

1. **register-level prefetch** — the default first move;
2. **an authored LDS ring** — on gfx950 `async_copy` + `commit_group` / `wait_group` with compile-time
   buffer indices; on gfx942 sync staging (async only at 32-bit, `order=[1, 0]`);
3. **`warp_pipeline_stage` and the scheduling model** (layer 1.5) — `authored_stage`,
   `compiler_interleave`, or `inter_wave` ping-pong, which needs `num_warps >= 8`;
4. **lowest — re-injecting plain's auto-pipeliner** (`gluon_swp.py`, `patch_reinject.py`). It exists in
   `libtriton` and can be spliced back without a rebuild, but it is **parity-capped**: it can only
   reproduce what plain already had. Use it as a **diagnostic** below the parity gate (how much of the gap
   is `lost_pipeline`?) or as a **last resort** when the hand-written ring cannot reach parity; its numbers
   are labelled `injected`, are never a win, and it is never used on an incumbent (mode B). Full recipe:
   [`references/method/recover.md`](references/method/recover.md) `Last resort: re-injecting plain's
   pipeliner`.

**Layer Backbone — the climb's spine** ([`references/method/climb.md`](references/method/climb.md)).
Advance one coupled layer per round. Layer 0 (parallel axis, reduction landing) is the front end's and
arrives decided in the champion; which of these layers has any surface is decided by the archetype
(`references/workloads/index.md`, the row `hw_budget.py --workload` prints).

| layer | decides | reference |
| --- | --- | --- |
| 1 | the transcription anchor — explicit layouts, correct-but-slow | `references/method/transcribe.md` |
| 1.5 | scheduling model: `authored_stage` / `compiler_interleave` / `inter_wave` (`num_warps >= 8`, else silently absent). Sets warps/CTA, so it gates 4–5 | `references/tile-programming/scheduling-model.md`, `warp-pipeline.md` |
| 2 | memory path: `buffer_load` → async direct-to-LDS (gfx950) | `references/tile-programming/memory-path.md` |
| 3 | LDS layout: swizzle / padding / dedup — gates layer 2's async form (one access of a native width per lane) | `references/tile-programming/layout-recipes.md` |
| 4 | pipeline, hand-written (order above). Settle first whether the loop wants one at all | `references/tile-programming/pipeline.md` |
| 4+ | instruction scheduling within the structure: tied identity, `s_nop` placement, `llvm_fn_attrs` (3.8.0) | `references/tile-programming/instruction-scheduling.md`, `references/gluon/inline-asm-reference.md` |
| 5 | slicing / registers / occupancy: joint VGPR+AGPR budget, zero hot-loop spill | `references/tile-programming/slicing.md` |
| 6 | beyond the hot loop: XCD-aware PID remap, L2 grouping | `references/workloads/gemm.md`, `attention.md` |
| 7 | matrix engine + low precision: which matrix op this dtype gets **at this shape** (gfx950 scaled MFMA) | `references/tile-programming/low-precision.md` |

The numbering is an index, not a work order: **1.5 before 4 and 5**, **3 before 2's async form**, **4
before 4+**. Layers 2–5 are visited twice in role — below the parity gate to repay a named debt, above it to
buy what plain never had — so every round records which claim it makes. A layer is landed on **three pieces
of evidence**: budget consistent, profile delta positive, IR/asm signal confirmed.

**Ten reversed-intuition traps** — where a strong default read of a profile is wrong (low MFMA util is not
an occupancy wall; L2-low plus coalescing-low is a symptom; a full-drain `lgkmcnt(0)` at equal occupancy is
an overlap gap, not bandwidth; a compiler lever exploits structure and never creates it; …): the table is
in [`references/method/climb.md`](references/method/climb.md), and it is read once, before the first
round.

## Procedure

From the GEAK repo root: `SKILL=perf_knowledge/expert_skills/skills/gluon_authoring` (the skill, its
Gluon-specific tools and runtime) and `KT=kernel_workflow/scripts/kernel_tools` (GEAK's shared kernel tools:
ISA/occupancy, profiling parsers, roofline). `$SKILL/scripts/<tool>` shims keep working for every tool that
moved to `$KT`. Every stage below is one row of the stage spine, and continues in its method file.

```text
ENTRY      settle the mode, assert the bundle (champion_gate.py)            method/entry.md
BUDGET     hw_budget.py, no GPU: bound-class prior, floor, multiple over it  method/budget.md
TRANSCRIBE champion TTGIR -> explicit layouts; 4-check equivalence gate     method/transcribe.md
RECOVER    attribute the debt (parity_gate.py), repay it by hand; parity    method/recover.md
EVIDENCE   anchor profile + the four dials, refreshed every round           method/profile.md
CLIMB      one coupled layer per round, hand-written pipeline, depth only   method/climb.md
CLOSE      outcome enum, closure self-review, Director arbitration         method/close.md
```

### The gates are executable — a gate is passed by tool output, not by assertion

Each exit condition has a command, and **the round log carries the command's output**. "The precondition
holds" is not a gate; it is the claim the gate tests.

| # | gate | command | mode A (port) | mode B (incumbent) |
| --- | --- | --- | --- | --- |
| G0 | loop shape matches the entry | inspect the four launch args | PORT shape set | defaults — setting them is the mirror-image mistake |
| G1 | **champion** — the comparator can support a claim | `$SKILL/scripts/champion_gate.py --champion <bundle>` | on `plain_champion.json` | on an incumbent bundle |
| G2 | **equivalence** — the anchor is the champion | `ttgir_bridge.py verify` + oracle + determinism + asm parity | before recovery | n/a (nothing transcribed) |
| G3 | **parity** — the debt is paid, and if not, who owes it | `$SKILL/scripts/parity_gate.py …` | before the first climb | **must not be run as a gate**; diagnostic only, say so |
| G4 | occupancy is not already lost | `$KT/probe.py measure --dir ir/<tag>/ --arch gfx950` | yes | yes |
| G5 | **attribution** — the number is a number | `ab_bench.py` screen in the engineer's loop; acceptance = `verify_engineer` (`harness_lib` legs) | every kept round | every kept round |

On failure: **G0** stop before dispatch. **G1** stop, edit nothing, report `blocked` — a gate failure is the
front end's to fix. **G2** do not recover on a wrong anchor. **G3** exit 2 = **do not climb**; the round's
outcome is `recovery` against the suspect it closed. **G4** read **both** limiters (registers and LDS each
cap workgroups per CU). **G5** a delta inside the measured spread or under `MIN_IMPROVE` is not a result.

**Two comparators, both carried in every result.** Correctness and equivalence are versus the **anchor**;
performance is versus the **champion**. `vs_anchor` alone hides the question — the anchor is a regression you
created — and the same climb reads very differently against the two, so the champion one decides.

### 1. Entry — assert the bundle, pin the comparator

Settle the mode (table in `## When to use`). Then `champion_gate.py` on that bundle; it prints every check
by name — read the list and the escape flags rather than four headline failures. The ones that make a run
unfalsifiable: `[SOURCE]` (the champion changed since it was measured), `[CONFIG]` (the TTGIR is not from
the pinned config — the anchor starts below plain-best by construction), `[COMPARATOR]` (`champion_ms`
slower than the shipped default — a strawman), `[SAMPLING]` (a capped sweep — provisional; the caveat
travels with every number). Also `[CLIMB]`: a front end that stopped climbing early hands you settled work
that was not settled — return a `resweep_request` with the evidence rather than re-sweeping here.

**The comparator is the champion at its own best config, and the anchor is bound to that config** — the
recovered layouts carry literal `warps_per_cta` and tile extents, so per-bucket best configs mean per-bucket
dumps and anchors. Config knobs are not rounds and not this skill's job: a tile change re-recovers the whole
layout set, so a suspect config is a `resweep_request` (returned to tech_lead, which hands it to GEAK's tuning), a suspect
structure is `structure_suspect.json`. Full contract, modes, the depth contract for coupled directions:
[`references/method/entry.md`](references/method/entry.md).

### 2. Budget — before touching the kernel

```bash
python3 "$KT/hw_budget.py" --sku MI355X --workload <gemm|attention|moe|norm|…> \
        --shapes M=…,N=…,K=… --dtype bf16        # no GPU
```

It assembles `sku.json` / `hw_constants.json` / `workload_models.json`, prints the bound-class prior, the
MFMA-only floor and your multiple over it, and the `workloads/index.md` row for the archetype —
**read that row**: getting the archetype wrong is a whole run on the wrong layer. `bound_direction` says which
verdict is robust (`upper` → trust memory; `lower` → trust compute; `ambiguous` → measure, do not resolve from
the model). Every "% of roofline" you later report carries **`numerator_basis`** (model | counters) and
**`denominator_basis`** (datasheet | empirical@tool-version | in-shape probe): datasheet denominators **rank**,
only a measured numerator over a probed or calibrated denominator may **gate or close**. gfx942 SKUs
(MI300X / MI325X) are the downgrade rows of the same table. Detail, calibration, rocprof-compute caveats:
[`references/method/budget.md`](references/method/budget.md).

### 3. Transcribe — a faithful anchor, driven to equivalence

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest                      # offline: is recovery sane here?
bash    "$KT/dump_ir.sh" <compile cmd> --variant plain --out ir/ --arch gfx950 \
        [--kernel-name <substr>]                                         # PIN the body on a multi-kernel op
python3 "$SKILL/scripts/ttgir_bridge.py" recover --ttgir ir/plain/plain.ttgir --arch gfx950 \
        --out anchor_layouts.py                                          # via the compiler's layoutToGluon
python3 "$SKILL/scripts/recover_gluon.py" …  --with-skeleton                # anchor assembly
python3 "$SKILL/scripts/ttgir_bridge.py" verify --plain ir/plain/plain.ttgir \
        --anchor ir/anchor/anchor.ttgir --arch gfx950                    # layout diff — never skipped
python3 "$KT/probe.py" measure --dir ir/anchor/ --arch gfx950             # occupancy, compile-only (G4)
```

Follow [`references/method/transcribe.md`](references/method/transcribe.md) — recover, **apply**, compile,
verify, record, attribute. What decides the stage:

- **The anchor is the champion, 1:1** — every layout, every conversion at the same program point. Hold the
  pinned config fixed. Do not mix transcription with optimization.
- **Declaring a layout is not applying it.** The recovered `gl.constexpr` preamble and the skeleton body are
  not connected; a body left on `AutoLayout` compiles, passes the oracle and is several times slower, and
  `verify` reports exactly the unapplied layouts as `missing`.
- **Equivalence is four checks, all recorded (G2):** layout diff (`ttgir_bridge verify` — `PASS`, or
  `RECONCILED` where every difference has a named structural cause), the numeric oracle at tolerance,
  determinism over ~40 launches, and asm parity (instruction mix **and** sequence). Budget several verify
  passes — it is a diff you converge on.
- **Every divergence is named** in the ledger: **faithful** (owes nothing), **forced** (Gluon has no
  constructor — `UNRECOVERABLE` — so you owe the row and what you wrote instead), **elective** (you judged the
  champion's choice sub-optimal in Gluon — you owe the row **and the faithful variant measured**, and it is
  banked as a win, not as the anchor). A >100% anchor is allowed; an unrecorded improvement is not.
- **`UNRECOVERABLE`**: probe your own build first (the Gluon surface moves), re-dump at `ns=1` and re-recover;
  if it is still absent, record a forced divergence or a `structure_suspect` — never hand-roll a basis
  (`amd_rotating_shared` has no constructor, and no Python binding reaches its normal form).
- **Classify each `ttg.local_alloc` against an `ns=1` dump** (staged → `allocate_shared_memory`; pass-through
  → `convert_layout`, buffer compiler-owned), and read `recover`'s `buffer_load` operand buckets, its `LDS:`
  line and the `constants-digest` — three things layout equivalence cannot see.
- **A real expressibility wall reads as `FAIL`, correctly**: fp8 `tt.dot` lowered by plain to an unscaled
  `tt.dot_scaled` has no Gluon spelling. Report the wall; do not reconcile it away.

### 4. Recover — attribute the debt, repay it by hand, then clear parity

Start as soon as equivalence passes. Mode B skips this stage (record the no-op).

```bash
python3 "$SKILL/scripts/parity_gate.py" --champion-ms <C> --anchor-ms <A> \
        --champion-asm ir/champion/champion.amdgcn --anchor-asm ir/anchor/anchor.amdgcn \
        [--champion-ttgir … --anchor-ttgir …] [--champion-lds <N> --anchor-lds <N>] --arch gfx950
```

It exits **2** while the ratio is under the run's declared criterion (default 0.95) and names who owes the
gap. Each suspect has an artifact read and a profile read; when they disagree, the disagreement is the
finding.

| suspect | artifact signal (`parity_gate`) | repaid by hand at |
| --- | --- | --- |
| **`lost_pipeline`** | champion TTGIR has `memdesc_index` / `local_store` / `num_stages > 1`, the anchor's does not (`iter_args >= 2` is **not** evidence) | layer 4 — the hand-written order in `## Mechanism`: prefetch → authored ring → stage marker |
| **`lost_layout`** (incl. lost vectorization) | load-width or LDS-op histogram narrowed (`dwordx4`→`ushort`, `ds_read_b128`→`ds_read_u16`), or `shared`/WG grew across the divisor | layer 3 — **re-recover** from the IR; never hand-derive a basis |
| **`lost_RA`** | same instruction multiset, serialized anyway — an address rematerialized right above the `ds_read` that consumes it | layer 5 — slicing / register budget |

**`plain@ns=1` is the attribution control for `lost_pipeline`** — re-run the plain champion with its pipeline
off, at its own config: `plain@ns=1 ≈ plain` → no overlap to lose (a residual is a transcription defect, not a
debt); `plain@ns=1` **faster** → the shipped depth is a pessimisation, do not recover it (read `spill=` first —
it is usually a register wall, and then the bug is the tile); anchor ≈ `plain@ns=1` < `plain` → the whole gap
is the pipeline. Find out **which `num_stages` knob the champion uses** before flipping one: a
`tl.range(..., num_stages=N)` annotation overrides the launch argument outright, and a dot-free loop has no
anchor at all.

**Until parity holds, a round is `recovery` with the suspect it closed — never a win.** Do not climb on an
unattributed defect, and do not book the recovery as the win ("3× the anchor" is a statement about the
anchor). Parity not reached within the stage budget: record **`parity_unreached`** with the residual split
and carry it on every later number; a residual owned by a mechanism Gluon cannot express is a **handoff**
back to the front end, not a climb target.

**Last resort, and diagnostic only — re-injecting plain's pipeliner.** When `lost_pipeline` is named and the
hand-written ring cannot reach parity, or when you need to size the pipeline debt, `gluon_swp.pipelined(N,
buffer_ops=True)` splices plain's pipeliner back in-process. It needs a `tt.dot` anchor, `tt.load`s at
pipeline time (`buffer_ops=True` restores plain's order), and staging the pass builds itself; it must be paired
with plain's post-pipeline tail; its landing tell depends on the loop's shape; and its in-process and on-disk
caches both lie unless keyed. Every number it produces is labelled **`injected`**, is never a win, needs one
process per variant, and it is never applied to an incumbent. The full recipe — conditions, error-text
decoding, the tail splice order, the IR tells, the cache trap, `patch_reinject` / `patch_async_reinject` —
is [`references/method/recover.md`](references/method/recover.md) `Last resort: re-injecting plain's
pipeliner`.

### 5. Evidence — the four dials, every round

Profile the anchor once it exists, re-profile every round, and **name the reading behind every edit**
("B1 says issue-bound and A2 says spare registers, so grow the tile" — not "grow the tile"). A missing number
is reported missing, never as a zero.

| dial | what you must know | tool |
| --- | --- | --- |
| **A** occupancy & registers | `next_free_vgpr`/`agpr`/`spill` → waves/SIMD → **wg/CU**; LDS bytes/WG from the Triton cache `shared` (KD and rocprof-compute report a structural 0) | `$KT/asm_loop_audit.py`, `$KT/probe.py`, `$KT/amd_occupancy.py` |
| **B** instruction stream | ranked inter-MFMA bubble ownership, LDS-vs-global feed split, per-opcode histogram diff, MFMA operand-layout facts | `$KT/mfma_efficiency.py` (ATT), `$KT/asm_loop_audit.py --opcodes`, `$KT/layout_facts.py` |
| **C** memory side | does a roofline apply at all; achieved fabric BW and L2 (every route, as a range); intensity vs ridge; the in-shape ceiling; the floor probe before calling anything memory-bound | `$KT/hw_budget.py`, `$KT/parse_pmc.py`, `$KT/mem_bw_probe.py` |
| **D** timing & correctness | correctness gating before timing; same-window interleaved A/B; a default-off knob must not move the shipped instruction stream | GEAK `harness_lib` (acceptance), `$KT/ab_bench.py` (screening) |

**Profiler entry is GEAK's** `kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>` under
`gpu_lock.sh` (optional PMC / ATT modes; parsers in `$KT`); never pin `HIP_VISIBLE_DEVICES` inline.
**Interpretation** follows `kernel_workflow/knowledge/profiling_guide.md`: ratios over Total; dependency
wait is a latency chain (C1), not memory-bound; issue wait is too few resident waves (C2); read the busy
counters `VALUBusy` / `MfmaUtil`, not duty-cycle `VALUUtilization`. A PMC-blind box (RDNA4) degrades — it is
not a failed run. Full dial table and caveats:
[`references/method/profile.md`](references/method/profile.md).

**Acceptance timing is GEAK's `harness_lib`** — CUDA events with a sync per sample, read-evict cache flush,
median, a fresh process per leg, the same-window baseline — and the commit gate is `MIN_IMPROVE`. `ab_bench.py`
is a **screening** instrument on the same timer (read-evict cold by default, median headline, min as an extra
field): its control arm, `--permute` and compiled-artifact `fingerprint()` catch cache collisions and position
effects, and a screen pass needs both a delta beyond the measured noise band and the 2% gate. [`references/method/benchmark-hygiene.md`](references/method/benchmark-hygiene.md).

### 6. Climb — one coupled layer per round, depth only

```text
per round:  profile -> analyze -> edit -> verify.   ONE attributable change, ONE layer, vs champion_ms.
```

**No sweep, no branch** — both are the front end's. Preserve the inherited config and structure; a suspect
one is a `resweep_request` / `structure_suspect` with evidence. The layer order is the Backbone in
`## Mechanism`; within it, work down what this language newly made expressible:

1. **The pipeline, hand-written** (layer 4), in the order of `## Mechanism`: register prefetch; an authored
   LDS ring — on gfx950 async direct-to-LDS with compile-time buffer indices and `wait_group` recomputed on
   every prologue/unroll change; on gfx942 sync staging; then `warp_pipeline_stage` with the scheduling model
   the profile supports (A1's AGPR/VGPR split and B1's cadence verdict pick it; the two non-base models are
   exclusive — climb one to its wall before trying the other).
2. **Per-operand buffering** — explicit allocation lets operands differ in depth, the lever plain cannot
   express whenever uniform depth does not fit the LDS budget.
3. **The `#shared` footprint** — swizzle vs padding, LDS dedup (layer 3) — re-derived per arch from
   `hw_constants.json`, never carried across.
4. **Declining to stage an operand** — global straight into the dot-operand layout; re-check the load width,
   since the LDS round trip was often what made it wide.
5. **The memory path** (layer 2) — `buffer_load`, then async direct-to-LDS on gfx950.
6. **Registers and occupancy** (layer 5) — slicing, live ranges, zero hot-loop spill; then **instruction
   scheduling** within the structure (4+) and **beyond the hot loop** (6) and **low precision** (7) where the
   archetype gives them surface.

AMD-specific levers are catalogued as cards: `python3 "$SKILL/scripts/lever_index.py" --bound <class> --arch
gfx950` lists the ones expressible in Gluon for the bound your profile named — use it before inventing a lever
and after; a ranked bucket with no card is a catalogue gap to report. **Inline asm** has its own gate
(`references/pitfalls/negative-patterns.md` `Inline asm: justify the reach before taking it`); a round touching
`inline_asm_elementwise` is accepted on the disassembly plus a determinism race test, not a counter delta.

Two disciplines keep the climb honest: **fix a structural success signal before reading a clock** (buffer
count, barrier count, the `lgkmcnt` / `vmcnt` shape), and **be willing to be wrong about the cap** — a
correctly diagnosed residual may not move because a second resource binds at the same point. Detail:
[`references/method/climb.md`](references/method/climb.md).

### 7. Close

`outcome` is one of `win | partial | negative_revert_plain | negative_keep_baseline | timeout` (with
`deferred[]` on `partial`). `structure_suspect` and `resweep_request` are artifacts beside an honest outcome,
never outcome values; `parity_unreached` goes in `caveats[]`. A Gluon result slower than the champion is
`negative_revert_plain`. Budget left is not a stop — closing with rounds unspent and a climb below the floor is
a finding against the close. Before an at-ceiling / keep-baseline / negative close, the deep_engineer writes the **closure
self-review** (`closure_review.md` in its OUTPUT_DIR: strongest counter-evidence, untried alternatives,
gotcha checks — advisory, on-disk evidence only) and returns it with its result; tech_lead reads it before
planning the next round and Director quotes its strongest counter-evidence when arbitrating. Acceptance is
GEAK's: `verify_engineer` re-benchmarks, Director validates.
[`references/method/close.md`](references/method/close.md).

## Knobs & pitfalls

What compiles and then costs you (full lists:
[`references/pitfalls/negative-patterns.md`](references/pitfalls/negative-patterns.md),
[`references/pitfalls/platform-known-issues.md`](references/pitfalls/platform-known-issues.md)):

- **Probe the build before sweeping a version-sensitive knob.** `python3 "$SKILL/scripts/probe_levers.py"
  --all` reports `live` / `dead-declaration` / `absent`; out-of-range knobs are accepted and change no IR, so a
  flat sweep on the wrong build reads as a kernel fact.
- **`num_stages` on a Gluon kernel** — dead in 3.8.0 (no pass consumes it). Not a tuning knob.
- **Runtime buffer indices in an async ring.** `smem.index(k % nBuffers)` stops the scheduler proving
  overwrite safety; use compile-time indices and recompute `wait_group(N)` whenever prologue, region or unroll
  changes — but over sync staging the two forms can emit identical ISA, so unroll when the ISA says it bought
  something.
- **Async copy layout contract** (gfx950 baseline; gfx942 only at 32-bit with `order=[1, 0]`): each lane must
  make **exactly one** access of a listed width — a layout that repeats over the tile fails with wording that
  reads like a missing op, so vary the tiling before recording an arch ceiling. `add_coalesce_async_copy`
  rescues off-width patterns with a bounce; it is not what enables async. Build a **swizzled** destination
  first — a padded destination fails translation or miscompiles; check numerics on any padded async arm. The
  signature that async replaced staging: `ds_write == 0` with a matching count of direct-to-LDS loads.
- **`sched_barrier` / `sched_group_barrier` / `set_prio`** — absent from `gl.amd.cdna3` and `.cdna4`;
  production code that stubs them on `ImportError` is dead code that reads like scheduling control.
- **`gl.warp_specialize`** — present in core `gl`, fails the pass manager on CDNA. `gl.amd.warp_pipeline_stage`
  works and emits `s_setprio`, but only at `num_warps >= 8`; it is a hint, so measure it.
- **3.8.0 scheduler surface.** The coexec scheduling strategy is default-on **only for gfx1250** (at
  `num_warps <= 4`); on gfx950 and gfx942 it is opt-in — process-wide via `TRITON_HIP_USE_COEXEC_SCHEDULER=1`
  (still only at `num_warps <= 4`) or per compile via `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]`,
  which is the only route for a `warp_pipeline_stage` kernel (`num_warps >= 8`). Full statement:
  `references/tile-programming/compiler-contract.md ## What upstream 3.8.0 actually gives you`. **Fork-only, inert on stock builds:**
  `TRITON_GLUON_SWP_PIPELINE`, `TRITON_GLUON_COOP_LDS`, `TRITON_GLUON_PINGPONG`, `TRITON_ENABLE_LLIR_SCHED`,
  `TRITON_ENABLE_AMDGCN_AS`, `TRITON_ENABLE_AMDGPU_RA_HINTS` — tolerated and ignored, so a null result reads
  as "the technique does not work". The in-tree LLIR scheduler toggle emits **invalid IR** with VALU between
  matmuls — default-skip on attention.
- **Ping-pong** fires only in a narrow window and never on hand-authored staging; judge it from the ISA
  (`s_setprio`), and do not try to earn it by satisfying `is_pingpong_schedule_enabled` — the Gluon path never
  consults it.
- **`disableSched`** costs occupancy outright. **Deep pipelines by reflex** regress once waves are VGPR- or
  LDS-capped. **`convert_layout` scratch after a pipeliner pass** is outside hazard analysis.
- **Copying the champion's `waves_per_eu`** — it caps occupancy (`amdgpu-waves-per-eu`), tuned for plain's
  pipelined body; leave it off until it earns its way back.
- **Sweeping `num_warps` on a transcribed kernel** — the layouts carry the literal `warps_per_cta`; another
  warp count is a correctness bug (and editing `warps_per_cta` alone crashes the pass manager with an
  `iota_range` assert and no attribution). Tile or warp changes are a `resweep_request`.
- **Cache collisions.** Variants differing only by a `constexpr` share a Triton cache entry, and per-arm
  `TRITON_CACHE_DIR`s that do not encode the variant serve the wrong binary: give each arm its own kernel
  object and cache dir, prove the binaries differ (`fingerprint()`), and `--permute` before recording a flat
  verdict.
- **A tolerance check cannot fail on NaN** (`NaN > tol` is False); scan for non-finite outputs.
- **`amd_rotating_shared`** — no constructor; do not hand-roll a basis.

## Do-no-harm notes

- **Advisory only.** This file supplies method, API and mechanics, never a verdict: GEAK's
  isolated A/B against the immutable oracle decides, and a result below the measured baseline is a negative.
  It does not narrow GEAK's search — a matched skill enters the candidate set rather than pre-empting it.
- **The non-negotiable rules** (both modes): roofline/budget before authoring; anchor before optimization;
  one coupled layer landed on three pieces of evidence; a comparator tuned to its own best config; a
  falsifying probe before any scoped-ceiling deferral; benchmark hygiene; a close that names its auditor.
  And for Gluon: the champion gate passes before the first round; every anchor divergence is named (elective
  ones with the faithful number); the comparator is `champion_ms`; recovery and improvement are separate
  stages split by the parity gate; a Gluon result slower than the champion is `negative_revert_plain`; a win
  is validated across the **served** shape range, not one anchor shape.
- **Disbelieve a fast number as hard as a slow one.** Re-measure in a clean workspace, interleaved with the
  comparator on the same device; treat anything inside the spread as no result; and **confirm the Gluon
  kernel actually ran** — a dispatcher that silently falls back to the plain kernel produces a numerically
  perfect "Gluon" number. Check the launched kernel name.
- **A number measured under injection is not an upstream number.** Every measurement with `gluon_swp` armed
  or `patch_reinject` applied says so, and a reverted tree is confirmed reverted before anything else is
  measured in it.
- **Correctness gates before timing**, and the anchor additionally gates on equivalence.
- **Cheaper LDS is not faster.** Pass-through `convert_layout` can cut `shared` several-fold and still lose:
  backend scratch reuse adds full-drain `s_barrier`s, registers may be the binding limiter anyway, and the
  hardware transpose (`ds_read_b64_tr`, gfx950) can disappear. Read `s_barrier`, `lgkmcnt(0)`, the binding
  limiter and the transpose count off the artifacts.
- **Writing variants without timing them is not a search.** Time each against the comparator as it lands and
  dispose of it.
- **Re-check the generation before trusting a figure or a digest.** gfx950 is the baseline; nothing — a
  divisor, a conflict-free swizzle, a layout digest — carries to gfx942 unchecked (`## Arch dispatch`).

## Tools

Full per-tool usage: [`scripts/USAGE.md`](scripts/USAGE.md). Gluon-specific tools and the pack runtime live
in `$SKILL/scripts`; tools GEAK shares across workflows live in `$KT` (with `$SKILL/scripts` shims); GEAK
infrastructure owns locking, profiling entry and timing.

| stage | tools |
| --- | --- |
| entry | `$SKILL/scripts/champion_gate.py`, `env_gate.sh`, `probe_levers.py` |
| budget | `$KT/hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`, `extract_sku.py`; data `perf_knowledge/hardware/data/` |
| transcribe | `$SKILL/scripts/ttgir_bridge.py`, `recover_gluon.py`, `ttgir_to_gluon.py`, `smoke_test_recover.sh`; `$KT/dump_ir.sh`, `probe.py`, `amd_occupancy.py` |
| recover | `$SKILL/scripts/parity_gate.py`, `pipeline_survey.py`; last resort: `gluon_swp.py`, `patch_reinject.py`, `patch_async_reinject.py` |
| evidence | GEAK `kernel_workflow/scripts/profile_kernel.sh` (+ `gpu_lock.sh`); `$KT/capture.sh`, `rocprofv3_safe.sh`, `rocprof_compute_probe.sh`, `parse_pmc.py`, `parse_rc.py`, `kernel_breakdown.py`, `hotspot_analyzer.py`, `att_*.py`, `asm_loop_audit.py`, `asm_schedule_viz.py`, `mfma_efficiency.py`, `deep_mfma_analysis.py`, `layout_facts.py`, `gfx950_isa.py`, `hw_sources.sh` |
| measure | GEAK `e2e_workflow/scripts/harness_lib.py` (acceptance); `$KT/ab_bench.py` (screening), `create_harness.py`, `parse_correctness.py` |
| climb | `$SKILL/scripts/lever_index.py`, `pipeline_examples_cdna4.py` (gfx950 reference), `pipeline_examples_cdna3.py` (gfx942 downgrade) |
| close / runtime | `$SKILL/scripts/toolctl.py`, `canonical_record.py`, `round_record.py`, `recordctl.py`, `close_audit.py`, `report_lint.py`, `served_envelope.py`, `run_state.py`, `stage_context.py`, `context_query.py`, `wait_for.sh` |

## Roles in GEAK

No new roles and no extra agents: the method maps onto `kernel_workflow`'s existing roles and phases
(`kernel_workflow/kernel_lane.js`), the same way GEAK's other expert skills are consumed.

| stage | GEAK phase / role | what this skill asks of it |
| --- | --- | --- |
| entry | Setup / Benchmark — `benchmark_engineer` (harness, `COMMANDMENT`, baseline); tech_lead | pin the comparator at the tuned plain config; G0 launch args; the champion assertion (G1) |
| budget, evidence | Profile — `profile_engineer` (`kernel_workflow/scripts/profile_kernel.sh`); the deep_engineer re-profiles in its own loop | `hw_budget.py` before authoring; the four dials every round |
| transcribe → recover → climb | Optimize — tech_lead dispatches **one** `deep_explore` direction; `deep_engineer` runs it | the whole track in one engineer's loop: faithful anchor + equivalence, attributed recovery to parity, one coupled layer per round, hand-written pipeline; no config sweep, no branch — a suspect structure/config comes back as `structure_suspect` / `resweep_request` in its result for tech_lead's next round |
| measure | in-loop screening by the deep_engineer (`ab_bench.py`, under `gpu_lock.sh`); Verify — `verify_engineer` | acceptance numbers only from the independent re-benchmark (`harness_lib`) |
| close | Report / Validate — tech_lead, Director | the closure self-review travels with the result; Director arbitrates and quotes it |

The deep_engineer's return is GEAK's worker result, carrying in its notes and artifacts: champion/gate
references, anchor and best measurements vs `champion_ms`, recovery state (`parity` / `parity_unreached`),
checkpoint, rounds spent, and any `structure_suspect` / blocker / deferred work. The pack's record tools
(`toolctl.py`, `canonical_record.py`, `round_record.py`, `close_audit.py`; stage cards in `runtime/`) are
**optional bookkeeping** the engineer may use for those records — not a separate run mode. GPU access is
always GEAK's `kernel_workflow/scripts/gpu_lock.sh` (lock dir `/tmp/team_gpu_locks`); the broker in
`scheduler/` is opt-in (`GEAK_GPU_BROKER=1`).

## Sources

- **What this skill is made of.** The complete upstream `tile-programming-gluon` pack —
  `AMD-AGI/TileProgrammingAgentSkills@029f3d6f` plus that checkout's uncommitted working tree (its
  `compose.py --check` passed), `.claude` flavor, all 209 files (the three upstream agents' contracts folded into GEAK's existing roles, the agent files themselves not kept) — **reorganized into
  GEAK's structure** rather than vendored side by side. The upstream entry (`SKILL.md`) and its full method
  (`references/method-reference.md`) were folded into this file's chapters and into
  `references/method/*.md` (one file per stage); the loose upstream reference files were merged into
  `references/method/` and `references/pitfalls/`; `gluon/`, `tile-programming/`, `hardware/` and
  `workloads/` keep upstream's layout, deduplicated and reordered gfx950-first. The first GEAK snapshot
  (`@541a180`, pruned to the API surface) and its GEAK-authored entry are the base this file grew from: its
  transcription detail, recovery diagnostics, launch-arg calibration and do-not-write findings were kept
  and moved to the method file that owns each topic. Upstream remains the SSOT for the method; this is a
  one-way, restructured snapshot, so a re-sync is a merge against the headings recorded in each method
  file's own `## Sources`, never an overwrite.
- **Priority rules applied while merging** (decided for this snapshot): Gluon/Triton usage and method follow
  **upstream and upstream Triton 3.8.0**, with 3.6/3.7 as downgrade notes; **gfx950 is the main line and
  gfx942 the downgrade**; the pipeline is **hand-written first** and re-injecting plain's auto-pipeliner is
  the lowest-priority route (diagnostic / last resort, numbers labelled `injected`). GEAK content that
  conflicted was demoted or dropped, each with its reason in the method file's `## Sources`. The most visible
  ones: re-injection is no longer the opening move of recovery or a lever on an already-Gluon kernel;
  `num_stages` is dead on the Gluon path (the old "carry plain's value over" advice is withdrawn); the
  "no full profile before the port" rule gave way to budget-first and re-profile-every-round; the residual
  owners are exactly `lost_pipeline` / `lost_layout` / `lost_RA`; in-process interleaving of differently
  patched arms is withdrawn (one process per variant). Checked against the v3.8.0 source while merging:
  the coexec scheduling strategy is default-on only for gfx1250, opt-in on gfx950 / gfx942.
- **Where GEAK infrastructure owns a concern, it wins.** GPU locking (`kernel_workflow/scripts/gpu_lock.sh`,
  one lock dir for every tree), the profiler entry (`kernel_workflow/scripts/profile_kernel.sh` +
  `profile_policy.sh`), acceptance timing and correctness (`e2e_workflow/scripts/harness_lib.py`), GPU
  identity (`scripts/gpu_identity.py`) and the commit gate (`MIN_IMPROVE`). The pack's equivalents were merged
  into those or reduced to shims; `ab_bench.py` remains as a screening instrument on `harness_lib`'s timer.
  Tools with no GEAK counterpart — ISA/asm audit, occupancy, IR dump, ATT and PMC parsers, rocprof wrappers,
  the roofline/budget stack — moved to `kernel_workflow/scripts/kernel_tools/` so the kernel workflow can use
  them too; `scripts/<tool>` shims keep every pack path working. The per-arch and per-SKU data moved to
  `perf_knowledge/hardware/data/` and is the **single source** for both this skill and GEAK's e2e roofline
  skill; `kernel_tools/_hwdata.py` resolves it.
- **Still GEAK-local — owed upstream, and must survive the next sync:**
  - `scripts/ttgir_to_gluon.py` emits `tilesPerWarp` / `elementBitWidth` on `AMDMFMALayout` when, and only
    when, the TTGIR prints them (dropping them transcribed chained-dot / scaled-MFMA kernels to a silently
    different layout; the witness is still synthetic — only a genuine `tt.dot_scaled` body reaches
    `deduceTilesPerWarpForScale`).
  - `scripts/recover_gluon.py` peels `@triton.autotune` / `@triton.heuristics` before translating, and
    `scripts/smoke_test_recover.sh` runs every offline `--selftest` of the toolchain.
  - `scripts/patch_reinject.py` exits 2 with a message where `triton` is absent and pins its splice point by
    selftest; `scripts/patch_async_reinject.py` (GEAK-only) covers a hand-written async body with no
    pipeliner; `scripts/pipeline_survey.py` keeps its dot-candidacy `--selftest`.
  - `scripts/ttgir_bridge.py`: an LDS/CU fallback when `amd_occupancy` cannot be imported, and the "a deeper
    depth may still be reachable via the annotation" advice on a `num_stages=1` verdict.
  - `kernel_tools/probe.py measure` prints both occupancy limiters per kernel and names the binding one;
    `kernel_tools/amd_occupancy.py` never walks the filesystem for its data and keeps the RDNA4
    `--compiler-sweep` tool GEAK's RDNA4 knowledge calls; `kernel_tools/asm_loop_audit.py` keeps a linear
    wait-counter regex; `kernel_tools/dump_ir.sh` pins with `--kernel-name` and refuses a bare name on
    `--kernel`, moves caches aside instead of deleting them, and refuses `--emit-gluon` without an arch.
  - `perf_knowledge/hardware/data/hw_constants.json` keeps GEAK's RDNA4 notes; GEAK's uncited gfx1151
    `lds_per_wgp_kib` was dropped in favour of upstream's deliberate absence. `sku.json` carries the
    corrected MI325X / MI350X / gfx1151 rows.
  - `references/pitfalls/platform-known-issues.md` keeps GEAK's RDNA4 profiling wording (asserted by
    `kernel_workflow/scripts/test_rdna4_policy.js`).
  - `scripts/skill_index.py` / `scripts/close_audit.py` recognise this directory as a pack without an
    upstream `SKILL.md`.
- **Not vendored, and what degrades without it:** the `tile-programming-triton` front end that emits
  `plain_champion.json` (use mode B, or GEAK's own tuned plain kernel as the comparator); the
  upstream `gluon-direction` / `gluon-bench` / `amd-closure-skeptic` agents and the `kernel-opt-run` /
  `kernel-opt-fleet` captains — their contracts are mapped onto GEAK's existing roles (`## Roles in GEAK`); the
  `third_party/` submodules — `amd_matrix_instruction_calculator` (non-gfx950 operand layouts fall back to the
  distilled references), `rocprof-trace-decoder` (ATT timing; set `ROCPROF_ATT_LIBRARY_PATH`), the pinned ISA
  XML (in-pack encoding databases cover gfx950 / cdna3 / rdna3 / rdna4).
- **Measurements are not baked into this file.** The calibration behind the launch args, the measured
  pass-through-vs-staged results, the ping-pong window and every other number live in the method or
  reference file that owns the topic, each with its own provenance; vendored numbers are upstream's
  evidence, dated to upstream, not a GEAK measurement.
- **The two bars measure different things.** The parity criterion (default 0.95 of `champion_ms`) scopes
  transcription + recovery, where near-parity *is* success; `expects.isolated_speedup_min` scopes the whole
  track including the climb, so an A/B that stops at the parity gate is not a datapoint against it.
  `validation.status` stays `draft`, with no auto-application, until a `--record` run stamps it.
- Gluon language surface: ROCm Gluon GEMM tutorial
  (https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html), gfx950 Gluon
  tutorials (https://github.com/ROCm/gfx950-gluon-tutorials); onboarding and the measured GEMM ceilings stay in
  [`languages/gluon/`](../../../languages/gluon/).
