# Compiler Contract: the LLVM co-design tier + IR verification

Read this whenever a body change should land via the compiler half of the
co-design. In Gluon the author writes independent AC/LR/DOT and the register
budget; the compiler interleaves, allocates, and peepholes. An optimization is
**not landed** until the IR/asm confirms it.

> **Dispatch entry vs detail.** This file is the **detail reference** for the
> `compiler_interleave` branch of the layer-1.5 scheduling-model choice
> (`scheduling-model.md`) and for the LLVM-tier lever cards
> (`../hardware/lever-cards.json`, route=compiler-stack). The **card** is what you dispatch —
> the measured bound emits it per round, probe-gated + default-off — carrying the *when/what*
> (gate / expected direction / risk). This file carries the *how*: the mechanism, the
> throughput model, and the Scenario B authoring loop. Keep the
> generalizable decision logic on the card and the mechanism here; do not bake
> experiment-run numbers into either (numbers are measured fresh each round; arch
> constants live in `perf_knowledge/hardware/data/hw_constants.json`, read through
> `kernel_workflow/scripts/kernel_tools/_hwdata.py`).
>
> **Canonical split of the LLVM co-design material.** This page owns scope and sanction, toolchain
> identity, the upstream 3.8.0 capability table (`## What upstream 3.8.0 actually gives you`), the
> IR dump / audit workflow and what "landed" means. `llvm-codesign-handbook.md` owns the stage
> entry contract and how to author an out-of-tree pass plugin. `llir-codesign.md` owns the two
> LLIR scheduling models (throughput, co-execution), the plugin-tier gates and the region budget.
> `llvm-fn-attrs.md` owns the `llvm_fn_attrs` mechanism. `non-upstream-reserve.md` quarantines
> fork-only mechanisms. Scheduling-model choice (layer 1.5) is `scheduling-model.md`; the
> wave-level stage markers are `warp-pipeline.md`; per-instruction pacing and the declarative
> hints (`sched_barrier` / `sched_group_barrier` / `iglp_opt`) are `instruction-scheduling.md`.
> Each fact is stated once, on its owner page; the others point.

## Compiler scope: kernel-first; co-design only when sanctioned

By **default** this skill optimizes the **kernel only** and treats the
Triton/LLVM build as **read-only**.

- **MAY (always)**: write the Gluon structure (layout chain, the hand-written pipeline with
  independent `AC`/`LR`/`DOT` — register-level prefetch, then an authored LDS ring, then
  `gl.amd.warp_pipeline_stage`, in the order `pipeline.md ## Where the overlap comes from, and it
  is not the same question per tier` defines — and the register budget); set the **upstream**
  per-compile and env controls in `## What upstream 3.8.0 actually gives you` (`llvm_fn_attrs`,
  `TRITON_HIP_USE_COEXEC_SCHEDULER`, `waves_per_eu`); dump + verify IR (`scripts/dump_ir.sh`, now
  `kernel_workflow/scripts/kernel_tools/dump_ir.sh` behind a shim).
- **MAY, as the lowest rung only**: re-inject plain's already-compiled TTGIR pipeliner passes over
  the module `gluon_to_ttgir` returns. It needs NO rebuild and NO edit to an installed file and is
  NOT Scenario B (upstream's `add_stages_inspection_hook` or the pack's in-process shim), but it is
  **not a climb lever**: use it only as a **diagnostic below the parity gate** (to size the
  `lost_pipeline` debt) or as a **last resort** when the hand-written pipeline cannot reach parity;
  label every number it produces **"injected"**, never report it as a win, and never apply it to an
  incumbent (already-Gluon) kernel. Scope and ceiling:
  `pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`;
  full recipe: `../method/recover.md`.
- **MAY, with a host build — the plugin tier (between handoff and Scenario B)**: load an
  **out-of-tree pass plugin** into the existing compiler at an end-of-pipeline extension point.
  This modifies no compiler source and rebuilds **no LLVM**, so it is not Scenario B; but it is
  not free either — it needs a **default-visibility host build**, two of its three build states
  fail quietly, and the plugin binary is ABI-locked to the host's LLVM revision. Build states,
  gates and probe order (stated once): `llir-codesign.md ## The plugin tier`; skeleton and build
  invocation: `llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton`.
- **MAY NOT (default) — and it is two different tiers, not one.** Both are forbidden until
  sanctioned, but they have different entry gates, different procedures and different reach, and
  lumping them cost readers the middle one:
  - **the compiler front end's own source** — the pass list, a lowering pass's placement
    decisions, a language construct. This is the only tier that can undo a lowering change which
    *took an arrangement away*. Procedure: `## Sanctioned tier: changing the front end's own
    source`.
  - **the backend's source** — a new or altered backend pass, or an upstream fix not yet in the
    pin. Procedure: `## Scenario B: sanctioned compiler co-design`.

  Note what separates these from the plugin tier above: **the plugin tier already costs a host
  rebuild** (default visibility) and simply does not *edit* anything. The line is source
  modification, not the presence of a build. The full cost order across all of it, with what each
  rung unlocks and what to record when it is unavailable, is indexed in
  `../workloads/intake.md ## The recompile-depth ladder`.
- If the passes are **missing** on this build, or present but **cannot realize the
  structure** (still clump / spill / leave a gap only a pass change would close),
  that is a **scoped toolchain ceiling**: record it (`../method/records.md`
  Toolchain Ceiling record), **fall back** (the manual multi-buffer pipeline still
  hides some latency; or switch layer; or revert to plain), and emit a
  **compiler-change handoff** — which pass, the IR/profile evidence
  (clump/spill/gap), and a one-line proposed change + expected effect.

The handoff is the **default** form of "Scenario B" (compiler co-development).
When the operator/user **sanctions a rebuild**, Scenario B becomes an *executable*
tier -- the same handoff carried out under the safety loop in `## Scenario B`
below -- rather than only a note. Until sanctioned, do not edit the compiler.

### 适用条件 (when working at this tier is worth it)

Only reach for the compiler tier when ALL hold; otherwise it is irrelevant or premature:

1. target gfx950 (gfx942 downgrade: the same tier applies, with the CDNA3 co-issue windows and
   no async-LDS ring beyond 32-bit, so less structure exists to schedule) AND the bound class is
   compute / LDS / pipeline (layout-bound). Launch/wrapper/memory-bound or any
   stay-plain verdict -> do not go here.
2. the mechanism you intend to use exists on the installed build, established from the
   **artifacts** rather than from a version string (`scripts/probe_levers.py --all`);
   absent -> immediate scoped ceiling.
3. constexpr `K` is a hard prerequisite for a scheduled interleave; dynamic K
   -> constexpr-K bucket + fallback, or non-scheduled mechanisms only.
4. the explicit structure is already written: independent `AC`/`LR`/`DOT`,
   prefetched multi-buffer, `R_total <= 512`. Every lever at this tier only exploits
   existing structure — no structure, nothing to interleave.
5. compute-bound, large K before forcing AGPR accumulators, so the AGPR
   epilogue cost (`v_accvgpr_read`) amortizes.
6. IR is verifiable (`dump_ir.sh` works); no verification -> cannot claim "landed".

**The accumulator-chain distinction runs through everything below, so fix it here.** A *pure
GEMM accumulator chain* (MFMA -> MFMA, copies only between, accumulator read once at the
epilogue) and a *VALU-between-matmul* chain (softmax / gate / scale / dequant, accumulator
read-modified every iteration) want opposite things from the compiler, and a lever tuned for
one is not merely weaker on the other — it is usually wrong. Route by this before anything
else; the per-lever consequences are in ## 风险 and in
`llir-codesign.md ## Region routing: the model is a property of the region`.

### 风险 / 误用 (risks + mitigations)

- **Mis-attribution: compiler ceiling vs kernel bug (most important).** Do not
  record "needs compiler change" when the real fault is kernel structure (AC/LR/DOT
  not actually independent, wrong `wait_group`/prefetch, hot-loop `convert_layout`,
  budget overflow). A ceiling is valid only after IR shows the structure IS correct
  (prefetched MFMA regions present, no unhoisted convert, budget fits) but the pass
  still clumps/spills.
- **RA-hint regression.** AGPR-pinned accumulators add epilogue `v_accvgpr_read` (the downcast
  needs VGPR inputs), so the hint pays only where the epilogue is a small share of runtime; on
  small-K / epilogue-heavy / non-compute-bound kernels this regresses. Gate by bound
  class; compare with and without the AGPR hint and keep the best.
- **GEMM-class assumption (workload gate).** Forcing AGPR accumulators assumes a *pure
  GEMM accumulator chain* (MFMA -> MFMA, only copies between). The governing fact is
  **accumulator read-cadence**: AGPR-pinning pays only for a **write-only-until-
  epilogue** accumulator (GEMM — read once at the end). An accumulator that is
  **read-modified every iteration** (online-softmax `acc *= alpha`, gating, rescale)
  must stay in **VGPR**; forcing it to AGPR adds a `v_accvgpr_read`/`write`
  round-trip per iteration. So ops with VALU between matmuls (attention, MLA,
  Sage-attention, DSA, ...) break the assumption. **Default-skip the AGPR hint for such ops.**
  Critically, **LLVM's default register allocator already makes the right split for
  them** (operand tiles -> AGPR, read-modify acc -> VGPR); forcing the GEMM hint
  (`amdgpu-agpr-alloc=256`) is what *breaks* that, and any partial budget also
  regresses — so the "attention RA hint" is simply *not applying the GEMM hint*. If
  tried, gate by bound class and keep the best. A pre-RA selective-AGPR pass can also be
  the wrong answer for a different reason: a kernel that is *register-demand-bound* (too
  many live values) rather than *split-bound* (wrong VGPR/AGPR division) cannot be helped by
  re-splitting registers at all. Separate the two before building anything.
- **A throughput-pairing scheduler can assert rather than regress on VALU-between-matmul.**
  Any scheduler that assumes a pure MFMA -> MFMA accumulator chain will, on softmax /
  scale-between-matmul kernels (attention, MLA, DSA, scaled-MFMA + shuffle), meet a
  `QK -> PV` dependency that breaks its dominance assumption, and can emit **invalid IR /
  a verifier assertion** ("Instruction does not dominate all uses") rather than a slowdown.
  This is the failure mode to expect from a pass written for the GEMM case, and it is why a
  pass authored for the mixed case has to be explicitly **dependency-preserving** and
  transactional (`llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton` — the
  skeleton there rolls back rather than emitting unverifiable IR, which is the point of it).
  For these kernels interleave manually (`pipeline.md`), use wave-level stage markers
  (`warp-pipeline.md`), or try the stock 3.8.0 `coexec` scheduler strategy
  (`## What upstream 3.8.0 actually gives you`) before considering a pass at all.
- **Portability / build drift.** A gfx950 result is not portable; the installed Triton can
  change under you (a production-framework reinstall can swap it). Record the build
  identity, re-verify locally, do not carry negative conclusions across builds.
- **False "landed".** Require the IR/asm signal (three-evidence), not a timing delta
  that did not change the IR.
- **Correctness.** Pipeline-depth / buffer-index / layout changes can silently break
  results — equivalence gate vs the plain oracle every round.
- **Weak handoff.** A "needs compiler change" report is useless without the pass,
  the IR/profile evidence, and a concrete proposed change.
- **Over-reach into B.** Default build is read-only — record + hand off; execute B
  **only when sanctioned**, under the `## Scenario B` safety loop (gated, verified,
  restorable). An unsanctioned compiler edit is still out of scope.

## Sanctioned tier: changing the front end's own source

Peer to `## Scenario B`, and **forbidden by default in exactly the same way** — the difference is
the object. Scenario B writes or alters a *backend* pass. This tier changes the DSL compiler's own
source: the pass list it runs, the placement decisions inside one of its lowering passes, or a
language construct it exposes. It is the tier people skip, because the permission table used to
name only "any compiler or LLVM source" and the procedure was written for backend passes.

**Why it exists as its own tier.** A manually arranged pipeline depends on what the lowering
passes do *around* the author's markers — where a barrier is placed, whether a priority hint is
set at a section head or a section end. Those are decisions inside a front-end pass, and when one
of them narrows between two pins, the arrangement the kernel was built on is gone. No kernel-side
lever wins it back, and nothing below this tier can reach it
(`../pitfalls/platform-known-issues.md ## Toolchain-pin regressions`). This is also the cheapest tier to
*test*, because the hypothesis is usually a single file.

**Entry test (all must hold).** Same shape as Scenario B's:

- kernel-side layers are `closed | blocked`, and the gap is attributed to a **specific pass, or to
  one placement decision inside it**, with lowered-code and profile evidence — not to "the
  compiler";
- a rebuild is sanctioned and the build identity is recorded (`## Toolchain identity`, five
  fields — the before/after comparison is meaningless without it);
- the change is expressible as a **localized, gated** transform, or as a substitution of one file
  by a known-good prior version;
- the arrangement it is meant to restore is one the kernel actually expresses — verify the markers
  or constructs are present in the lowered code first, or the rebuild proves nothing.

**Procedure** — same safety loop as Scenario B, with the object swapped:

1. **Matched source, new branch.** Work on the installed/given source at its recorded identity.
   Never mutate the pinned build in place; the ability to return to it is what makes the A/B mean
   anything.
2. **Gate it off by default.** An environment toggle or a build flag, defaulting to the unmodified
   behaviour, so the build is a no-op until enabled and remains A/B-able afterwards. A file
   substitution is its own gate: keep both files.
3. **Rebuild only what changed.** One file, one target, relink — a minute-scale loop, not a full
   rebuild. Design the hypothesis to fit that.
4. **Verify in the lowered code, not in the timing.** Confirm the intended difference appears
   (a barrier moved, a hint emitted at a different point, a construct now available). A rebuild
   that changed nothing observable is not evidence about the hypothesis.
5. **A/B on the same boundary, then record.** Against the unmodified build, same harness, same
   pin identity. Report the mechanism alongside the delta.
6. **Restore.** Return to the pinned build unless the change is being upstreamed.

**Verdicts.** Unsanctioned, this tier is **out of scope** — a different verdict from a ceiling, and
it must not be recorded as one. Sanctioned and ineffective: a scoped result naming the pass and the
pin, which expires when the pin moves. Sanctioned and effective: the win belongs to the
**toolchain**, not to the kernel — record it as a scoped, dated toolchain finding plus a
compiler-change handoff, so the kernel's own ceiling is not credited with it.

## Scenario B: sanctioned compiler co-design

> **Authoring a pass from scratch (the normal case on stock Triton, where no reference
> implementation of such a pass exists in the tree):** when the sanctioned change means WRITING
> a new LLVM/LLIR pass or RA policy, follow
> **`llvm-codesign-handbook.md`** (capability-probe -> grep-discover the version's hooks ->
> minimal skeleton + wiring points + isolation ->
> verify). And BEFORE you build anything, run the two decision gates in
> **`../hardware/llvm-backend-invariants.md`** (allocation-vs-live-set; schedule-vs-chain-length) —
> a cheap probe kills most register/schedule co-design ideas in minutes, before a multi-hour build.
> That shared file also carries the DSL-agnostic backend invariants (256 arch-VGPR clamp, MFMA
> `_vgprcd` mixed form, class-granular sched_group_barrier), the 3-law verification protocol
> (prove-fires -> literal-asm-diff -> positive-control; never TFLOPS), and the falsified-levers
> ledger.

Default is handoff (above). When a rebuild is **sanctioned**, a compiler-level
hypothesis can be *executed* as a first-class tier -- but only after the kernel
side is exhausted and the bottleneck is **provably compiler-addressable** (IR
shows the structure is correct yet the pass still clumps / spills / leaves the
gap -- not a kernel bug; see the mis-attribution risk above).

The sanction comes from the **user's prompt** (e.g. "LLVM co-tuning allowed");
without it, stay at handoff. Upstream ships no declarative interleave and no
scheduling pass you can toggle on (## What upstream 3.8.0 actually gives you) — the one stock
scheduler aimed at matrix-plus-VALU regions is the `coexec` strategy, which is an LLVM
scheduling strategy selected per function, not a pass, and does not place chosen ops in chosen
MFMA shadows. So once that strategy has been checked by assembly diff, the canonical sanctioned
move for a **non-GEMM structure** (attention / softmax-between-matmul) is to **author a new
scheduling pass** — the agent implements the pass itself — not to look for a knob.

Entry test (all must hold):

- kernel-side layers are `closed | blocked`, and the gap is attributed to a
  specific pass/decision with IR + profile evidence;
- a rebuild is sanctioned and the build identity is recorded;
- the change is expressible as a localized, **gated** transform.

Reusable loop (target-agnostic; the specific pass/transform is the variable):

1. **Matched source + new branch.** Work on the installed/given Triton source;
   record the build identity and fetch LLVM at the build's pinned hash
   (`cmake/llvm-hash.txt`), matching the build type / assertions so the rebuilt
   library is ABI-compatible with the prebuilt set. Create a **new feature branch**
   so the pinned build stays restorable — never mutate it in place.
2. **Gate it off by default.** Put the transform behind a `cl::opt` (and a Triton
   backend env toggle) defaulting **off**, so the build is a no-op until enabled.
   This is what makes it A/B-able and safe to leave in tree.
3. **Fast iterate.** Rebuild only the changed target library, swap that single
   `.a` into the prebuilt LLVM dir (keep a `.orig` backup), and relink
   `libtriton.so` -- a minute-scale loop, not a full LLVM build.
4. **Verify the IR, not just timing.** Run the machine verifier (`MF.verify()`),
   dump IR, confirm the intended signal; make the transform **bail out unchanged**
   on any unexpected shape so correctness is always preserved (worst case = no
   change).
5. **A/B + record.** Toggle on/off on the same benchmark boundary; record the
   delta + IR signal as a compiler co-design experiment (`../method/records.md`).
6. **Restore.** Revert the swapped lib from `.orig` when done (unless upstreaming).

**Authoring a new LLIR scheduling pass (the general non-GEMM co-design capability).**
Upstream ships no scheduling pass of this kind at all, so on any VALU-between-matmul kernel
this is authoring, not toggling. The design that works is a **dependency-preserving**
schedule that interleaves MFMA among the VALU / LDS / convert ops **without breaking SSA
dominance** — which is precisely the property a throughput-pairing GEMM scheduler violates
on this shape, and why one written for GEMM asserts here rather than regressing. Build it
from these pieces:

- **classify, then route.** Separate matrix, global-read, local-read, local-write and
  convert instructions, and decide the model per region from what the region *contains*
  (`llir-codesign.md ## Region routing: the model is a property of the region`).
- **preserve dependencies explicitly — the safe template is a dependency-preserving list
  scheduler.** Emit only ready ops, **never reorder memory or side-effecting ops relative to
  each other**, and move only provably-independent ops. Built this way the IR always satisfies
  SSA/dominance and *always verifies* — that is the template for any "interleave A with B" pass.
- **pin the result without disabling the machine scheduler globally.** An IR/LLIR ordering is
  re-sorted by the target's `misched` / post-RA scheduler, so your order survives only behind
  hard scheduling boundaries (sched-barrier / region markers) or with `misched` off for that
  region. Prefer the boundary: a full reorder barrier after each memory anchor preserves the
  interleave while leaving `misched` enabled for the prologue, the epilogue and every skipped
  region; disabling `misched` outright costs you the backend's latency-aware scheduling
  everywhere else and can itself regress — so measure the transform **both** with and without
  `misched`, and require the win to survive the backend pass
  (`llir-codesign.md ## GEMM: the throughput model`).
- **make it transactional.** Snapshot, transform, verify, roll back on failure, so the worst
  case is "no change" rather than bad code — the skeleton in
  `llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton` does exactly this.

Load it through the stock `LLVM_PASS_PLUGIN_PATH` door if you can (no compiler-source edit),
or wire it into `make_llir` behind a default-off `cl::opt` (step 2) if a source change is
sanctioned; then run the same gated build / verify / A/B / restore loop above. **This path is
derivation-grade and has not been validated here to a working authored pass** — do not
present it to a run as a proven route.

Diagnose before building: confirm the bottleneck is the one the pass would fix.
*Register-demand-bound* (too many live values) and *register-split-bound* (wrong
VGPR/AGPR split) are different problems -- re-splitting registers cannot help a
kernel that simply needs fewer live values. A negative result is still valid:
record it and fall back.

**Some gaps are not compiler-expressible even with full freedom.** A
correctness-preserving pass **must** fence cooperatively-loaded shared buffers whose
indices are runtime values (`k % nBuffers`); that fence re-syncs staggered
warp-groups every iteration and cannot be removed without racing (verified: dropping
it gives wrong results; only a redundant LDS *drain* — already ordered by an explicit
`wait_group` — is safely removable). Hand-asm omits the fence only because it
**cycle-counts** buffer disjointness, a timing fact the compiler cannot prove. So a
tight `s_setprio`-only cooperative-load ping-pong is a hand-asm capability — record it
as a scoped abstraction ceiling vs the hand-asm target
(`mental-model.md ## The hand-asm ceiling`), not as an open compiler task.

**But the removable drain has a portable knob — recognize the over-sync first.**
Signature: an explicit-async (manual multi-buffer) kernel emits **more LDS barriers
per loop than the compiler's auto-pipeline would** — a RAW/WAR barrier guarding a
shared read that an explicit `wait_group` already orders. That barrier is redundant
and elidable two ways (both **gated per-kernel** by the determinism race-test,
`../method/benchmark-hygiene.md ## Determinism race-test` — never assume it transfers):

- **kernel-level (no rebuild):** a relaxed shared-load that stamps the read with the
  async-wait-synced attribute (this build: `load_shared_relaxed`) so the membar
  filter skips that barrier — the sanctioned way to express "this read is already
  ordered by my `wait_group`."
- **compiler-level (Scenario B):** the AMD membar filter early-returns when **both**
  hazard ops are async, so a barrier between two async LDS loads is never filtered;
  a localized async↔async filter extension removes it.

Both drop the loop barrier count toward the auto-pipeliner's. **Safe only for
compile-time-literal buffer indices**; on a runtime-index (`k % nBuffers`) /
early-prefetch structure the elision silently corrupts output (the cross-warp fence
above is load-bearing there) — which is why the determinism gate is mandatory, not
optional.

**Practices that make the co-design loop fast and the pass *stick*:**

- **Make the pass's policy env-selectable.** Read the scheduling strategy/mode + a
  numeric knob from the environment (not a hard-coded constant) so multiple
  strategies are A/B-tested **without recompiling** — recompile only to change
  *mechanism*, not policy. Register any new env var in the toolchain's
  recognized-env set or the build asserts.
- **Scope a global knob per-kernel when it helps one shape and hurts another.** A
  global env knob (pass toggle / sched strategy / membar flag) that wins on one
  kernel can regress a sibling compiled in the **same process** (a batch sweep, a
  multi-shape deliverable). Apply it per-kernel: save the prior env around that
  kernel's first compile (set → launch → restore in `finally`) and assert it is
  restored (== prior/None) so it cannot leak to later compiles. Gotcha: a
  `static`/once-cached `getenv` inside the pass defeats per-call scoping — read the
  env **per invocation**. (Env-selectable makes the knob A/B-able; this makes it
  per-kernel-applicable.)
- **A single-`.a` relink is cheap.** Design the gated tier around touching **one**
  pass file so the edit -> build -> swap one `.a` -> relink loop stays minute-scale
  (step 3), not a full LLVM build.
- **Reorder needs slack.** A scheduler that only **reorders** within a basic block
  **cannot create independence** — if the kernel is dependency/latency-bound (busy
  counter under 100%, `../method/profile.md ## Rule: read the busy/throughput
  counter`), pinning a different instruction order is neutral by construction.
  Confirm there is reorderable *slack* (independent ready instructions in the
  window) before authoring a reorder pass.
- **Template and pinning:** the dependency-preserving list scheduler and the
  "pin with boundaries, measure with and without `misched`" rule are the two pass-design
  bullets above; they are practices of the loop as much as of the pass.

## Scope: reference for authoring, not a portable recipe

Nothing at this tier makes a kernel fast by itself, and none of it is a guaranteed path. Any
win here is a **co-design**: the author writes the explicit layout and the explicit
multi-stage pipeline (independent `AC` / `LR` / `DOT`), and a scheduler then *realizes* that
structure by interleaving the instructions. Treat everything below as **heuristics to reason
from while authoring**, not a final answer:

Scheduling / asm work **cannot beat a throughput wall** — but first *prove* it is a
throughput wall (read the **busy/throughput** counter, not the duty-cycle one;
`../method/profile.md ## Rule: read the busy/throughput counter`). When the kernel
is actually **latency/ILP-bound** (which a ~100% duty-cycle counter can hide), these
scheduling levers *do* help; when it is genuinely throughput-bound they cannot, and
the lever is algorithmic work reduction. Post-assembly rewriting in general inherits a
**single-BB self-loop** assumption and does not transfer to branchy VALU-between-matmul
loops — a hot loop with a last-iteration prefetch guard or a conditional store is multi-BB
and outside such a tool's loop detector.

- **Build-specific.** *How* a scheduling capability is reached is a property of the installed
  build, not of a version string. Establish it from the artifacts
  (`## Toolchain identity`); absent -> record a toolchain ceiling
  (`../hardware/capability-matrix.md`) and fall back to non-scheduled latency hiding.
- **Condition-stacked.** A scheduled interleave needs constexpr `K` **and** the explicit
  author-written pipeline with its schedule-math filled in; miss one and the interleave does
  not land.
- **Not portable.** A gfx950 scheduled result does not carry to gfx942 or to a
  different kernel structure. Re-derive and re-verify per kernel / target / build.

Plain Triton's automatic software pipeliner does not run on Gluon — on 3.8.0 `num_stages` is a
**dead knob on the Gluon path** (no pass consumes it; it survives only as a budget parameter and a
field of the champion record) and you own the pipeline structure explicitly, hand-written first
(`pipeline.md ## Where the overlap comes from, and it is not the same question per tier`). The
same `add_schedule_loops` / `add_pipeline` passes are present in `libtriton` and *can* be
re-injected into the Gluon pipeline, but that is the **lowest** rung — a below-parity diagnostic
or a last resort, numbers labelled "injected", never on an incumbent kernel
(`pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`).
What genuinely does not carry over is compiler-*automatic* ping-pong scheduling — the wave-level
phase offset has to be requested with `gl.amd.warp_pipeline_stage` (`warp-pipeline.md`, Gate 0:
`num_warps >= 8`). Without an interleaving
scheduler the default LLVM scheduler **clumps** like instructions (all MFMA in one
block, all `ds_read` in another), serializing MFMA behind memory; the author makes
`AC` / `LR` / `DOT` independent so the scheduler can spread MFMA at an O(1) rate
between loads.

## Toolchain identity

Everything below is gated on **which build you are on**, and "Triton 3.7.1" does not say
it. Two independent axes decide different things, and conflating them is how a plan gets
built on a mechanism that is not present:

- **the upstream minor** (3.6.0 / 3.7.0 / 3.7.1 / 3.8.0) decides the DSL *language* surface —
  which markers and async ops exist at all, and which compile options are live;
- **the host build and its LLVM pin** decide the *compiler co-design* surface — whether a pass
  plugin can bind at all, and what it sees when it does.

A build can be new on one axis and empty on the other. Pin all five facts before the first
measurement and record them beside the numbers; a result whose toolchain identity is unknown
cannot be compared with one taken elsewhere.

```text
upstream_triton_minor:  3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 | other
build_lineage:          clean upstream | vendor-modified (record how you established it)
llvm_commit:            the pin the host was built from  (e.g. cmake/llvm-hash.txt)
host_build:             exports LLVM symbols (default visibility, TRITON_EXT_ENABLED)?
                        which build flag was used?
plugin_abi_target:      the LLVM commit any loaded pass plugin was built against
```

The last two matter only if you intend to load a pass plugin, and they are the two that fail
quietly — `llir-codesign.md ## The plugin tier`. `plugin_abi_target` must equal `llvm_commit`
exactly: a mismatch **crashes**, it does not degrade.

Beside the five facts, record the **active** scheduling configuration of each compiled kernel, read
from the artifact rather than from what you set: the `amdgpu-sched-strategy` attribute actually on
the kernel function in the `.llir` (backend-set `coexec`, an `llvm_fn_attrs` value, or none), and
the `TRITON_HIP_USE_COEXEC_SCHEDULER` / `TRITON_HIP_USE_EXPERT_SCHEDULING` values in the process
environment (both are cache-invalidating upstream knobs, so they belong in the record).

**Probe, do not infer.** Availability is a property of the installed artifacts, not of a
version string a note carried. `scripts/probe_levers.py --all` answers whether symbols and
knobs are present on this box; whether a pass then *bites* is a separate question answered
only by the IR (`## Per-change protocol`).

## Which strategy your build admits

Read your identity off the block above, then take the row. The point of the table is that
each row has a **different ceiling**, so the same kernel closed at different numbers on
different builds is not evidence about the kernel.

| your build | what you have | the strategy, and its ceiling |
| --- | --- | --- |
| **upstream 3.8.0 (the baseline this pack is written for)** | authored ring + `gl.amd.warp_pipeline_stage` markers, register-only slicing; **`llvm_fn_attrs`** (arbitrary LLVM function attributes reach the kernel, per compile); the stock **`coexec`** scheduler strategy (`TRITON_HIP_USE_COEXEC_SCHEDULER`); pipeliner re-injection as the lowest rung | **No declarative interleave.** Get demand under capacity *structurally* — shrink the vector work, balance the regions — because nothing here assigns ops to windows for you. That structural work is most of the available prize. The per-compile attribute route adds the AGPR-form and scheduler-strategy attributes (`llvm-fn-attrs.md`); `coexec` is the stock answer for matrix+VALU regions (opt-in on gfx950/gfx942, table below). The last part of the prize is not reachable without a pass you write yourself. `schedule_hint` is a **dead declaration** here (`## Portable scheduler co-design`). |
| *downgrade:* upstream 3.7.0 / 3.7.1 | markers, slicing, re-injection, `schedule_hint` presets (which reach `amdgpu-sched-strategy`) | Same structural strategy and ceiling, minus `llvm_fn_attrs` (raises as an unknown option) and minus `TRITON_HIP_USE_COEXEC_SCHEDULER` (not read — inert if set). The AGPR rung has no per-compile route. |
| *downgrade:* upstream 3.6.0 | no marker path at all (`../gluon/pipeline-reference.md`) | Stage-marker overlap is unreachable. For an attention-shaped kernel: the hand-written ring without markers, then (last resort, labelled "injected") pipeliner re-injection, then record a **toolchain ceiling** rather than spending rounds. |
| any build, plus a **default-visibility host build** | the stock LLVM pass-plugin door is usable, so a pass you author can be loaded | The plugin tier. Verify its build states before budgeting a round (`llir-codesign.md ## The plugin tier`). |
| any build, on a target your pass's cost model does not cover | the shape is unpriced, so **every region is skipped, silently** | A scoped **tool** ceiling — not a hardware wall. The matrix/VALU overlap mechanism exists across the CDNA family; it is the tool that is scoped. Fall back to hand interleave or the wave-level stage markers (`llir-codesign.md ## Applicability gate: shapes the tool does not model`). |

A language extension carried only by a modified build belongs in this table too: if a kernel
design depends on one (a per-wave predicate to skip work, say), that design is simply **not
writable** on an upstream build, and the honest comparison is against what upstream *can*
express (`../gluon/pipeline-reference.md`).

**Pricing the last row is a measurement you do not have.** How much the missing declarative
interleave costs is a per-kernel, per-shape quantity; the measurement that would settle it is an
A/B of the same sources at the same benchmark boundary, one with the declarative interleave and
one without, with in-loop matrix-issue efficiency recorded alongside wall time. Do not carry a
figure from a note. What *is* safe to carry is the shape of the answer: the structural work in
the first row is reachable everywhere, and a design that frees budget pays only if something
downstream spends it — so a kernel with little headroom to begin with has little to lose here.

## What upstream 3.8.0 actually gives you

> **The scope statement that governs this whole page.** A family of environment variables for
> toggling an LLIR scheduler, an AGPR/RA-hint policy, a post-assembly peephole and an attention
> scheduler circulates in kernel notes. **None of them exist in upstream Triton 3.8.0** — none
> appear in `python/triton/knobs.py` or `include/triton/Tools/Sys/GetEnv.h`, and none of the
> passes they name is in the tree. Setting one is **tolerated and inert**: nothing errors and
> nothing changes, so the null result reads as "this technique does not work on my kernel"
> rather than "that variable does not exist in this build". Do not build a plan on them, and do
> not record their absence as a ceiling. Each such name is **(fork-only; absent in upstream
> 3.8.0, inert if set)**; the full index is `non-upstream-reserve.md ## 6. Environment-variable
> index`, and the mechanism descriptions are preserved there for the case where you have
> established from the *artifacts* that you are on a build which ships them.

**The two first-class 3.8.0 controls on this tier.** Both are upstream, both are verified at the
`v3.8.0` tag (`third_party/amd/backend/compiler.py`, `python/triton/knobs.py`), and both should be
checked before anything that needs a build:

- **`llvm_fn_attrs`** — a per-compile launch/compile option that attaches arbitrary LLVM function
  attributes (`name=value` pairs) to the kernel, applied in `make_llir` *after* the backend's own
  attributes (so an explicit value overrides a backend-set one). It is the per-compile route to the
  AGPR form, the scheduler strategy, a VGPR cap, IEEE mode and target features, for `@triton.jit`
  and `@gluon.jit` alike. **3.8.0 only** (raises as an unknown compile option on 3.6.0 / 3.7.x).
  Mechanism, gates, acceptance by assembly diff: `llvm-fn-attrs.md`.
- **The stock `coexec` scheduler strategy** — `make_llir` adds the function attribute
  `amdgpu-sched-strategy=coexec` when the co-exec scheduler is enabled for the arch **and**
  `num_warps <= 4`. Enablement is `TRITON_HIP_USE_COEXEC_SCHEDULER` if set (either direction),
  otherwise the arch default, which at `v3.8.0` is **on for gfx1250 only**. So:
  **default-on on gfx1250; opt-in on gfx950 / gfx942** — process-wide with
  `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (honoured only at `num_warps <= 4`), or per compile with
  `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` (no `num_warps` gate, and it overrides the
  backend-set value). It is the stock answer for the **matrix-plus-VALU** region class and costs
  nothing to try; whether the pinned LLVM's `coexec` strategy pays on gfx950 is **probe-per-build**
  — accept it only on an assembly diff plus an A/B. Two interactions to carry: the env route cannot
  coexist with a `warp_pipeline_stage` kernel (Gate 0 wants `num_warps >= 8`, the env gate wants
  `<= 4`), so for such a kernel the per-compile attribute is the only route; and the knob is in
  `CACHE_INVALIDATING_ENV_VARS`, so toggling it recompiles correctly.
  It does **not** give you *placement* (a chosen vector op in a chosen MFMA's shadow) — that is
  `llir-codesign.md ## Declaring the interleave`. Its neighbour `TRITON_HIP_USE_EXPERT_SCHEDULING`
  (adds `amdgpu-expert-scheduling-mode`; arch default also gfx1250 only) is upstream too.

The capability table (one row per capability; the other pages point here):

| capability | upstream 3.8.0 route | reach |
| --- | --- | --- |
| **throughput interleave** — spread matrix ops between `buffer_load` / `ds_read` instead of letting the default scheduler clump them | author the structure (independent `AC`/`LR`/`DOT`, prefetched multi-buffer — `pipeline.md`) and pace it with `gl.amd.warp_pipeline_stage` (`warp-pipeline.md`) or the per-compile scheduler strategy (`llvm_fn_attrs`); or author a pass and load it through the stock `LLVM_PASS_PLUGIN_PATH` door | structural route is full; a *declarative* interleave needs a pass you write |
| **matrix/VALU co-execution scheduling** (attention, softmax / scale / dequant between matmuls) | the stock `coexec` strategy (above); kernel-side region balancing (`llir-codesign.md ## Attention: the co-execution budget`); a declaring pass you author for placement | strategy: stock, opt-in on gfx950/gfx942; placement: needs a pass |
| **scheduler strategy per kernel** (`amdgpu-sched-strategy=<s>`) | `llvm_fn_attrs` (3.8.0); `schedule_hint` presets on 3.6.0/3.7.x only (dead declaration on 3.8.0) | full, per compile; value enum is probe-per-build (`llvm-fn-attrs.md`) |
| **accumulator in AGPR** | `llvm_fn_attrs="amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0"` — these are LLVM **function attributes**, and 3.8.0 passes arbitrary function attributes through to the kernel (`llvm-fn-attrs.md`) | **full**, and per-compile rather than global, which is strictly better. GEMM-class only (## 风险) |
| **machine-IR hand edit** | `TRITON_DUMP_MIR` / `TRITON_SWAP_MIR` / `TRITON_SWAP_MIR_ENABLE_MISCHED` — a stock dump-edit-reassemble path over MIR, with none of the plugin host-build gates | full, per kernel, manual |
| **keep a balanced region's placement** | `DISABLE_LLVM_OPT=disable-machine-sink` (`llir-codesign.md ## Two preconditions the kernel must protect`) | full |
| **post-assembly peephole** — LICM of loop-invariant LDS address math, scalar ops packed into matrix gaps | none — upstream has no post-assembly rewriting stage | **none**. Attack it at source level (emit less loop-invariant address arithmetic, less in-loop scalar work) and record what remains as a scoped toolchain ceiling |

Verify each of them the same way — in the generated code, never from the fact that you set
something:

| capability | verify it landed by |
| --- | --- |
| throughput interleave | MFMA no longer clustered; loads for `k+1` issue before `DOT(k)` |
| co-execution / scheduler strategy | the attribute is on the kernel function in the `.llir`, **and** the `.s` differs from the same compile without it; vector ops sit between MFMAs rather than piled before the first / after the last |
| accumulator in AGPR | fewer in-loop `v_accvgpr_mov`; accumulator in AGPR |
| source-side scalar/address reduction | fewer loop-invariant address ops inside the loop body |

Record the **active** configuration (not the intended one) in the profile record, together with
the build identity that makes it meaningful (`## Toolchain identity`).

## Portable scheduler co-design (the env knobs, and where the real lever lives)

Between the structural work above and authoring a pass (`## Scenario B`) there is a
**per-launch, portable** scheduler layer that needs no custom pass and no rebuild, and it applies
to any layout-bound kernel **including the VALU-between-matmul case**.
Its operational form -- how to write it, what the strategy values actually are, and the two other
instruction-scheduling mechanisms production reaches for far more often -- is
`instruction-scheduling.md`. This section carries only what belongs to *this* page: which knob
names are real on which build.

> **Before the scheduler sweep, check whether the gap is a MISSING pipeline rather than a badly
> scheduled one.** If the kernel is schedule/overlap-bound (equal VGPR+AGPR and occupancy vs plain,
> lower `MfmaUtil`, more full-drain `s_waitcnt lgkmcnt(0)` --
> `../method/profile.md ## Rule: equal registers+occupancy but lower MfmaUtil`) **and** it is a
> Gluon transcription that lost the pipeline plain had (`lost_pipeline`), repay that debt rather
> than schedule around it — **by hand first**: register-level prefetch, then the authored LDS ring
> (gfx950: `async_copy` + `commit_group` / `wait_group`; gfx942 downgrade: sync staging, or 32-bit
> async with `order=[1,0]`), then `warp_pipeline_stage`
> (`pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`).
> Re-injecting plain's pipeliner (lever `reinject_ttgir_pipeliner`,
> `pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`)
> is the **lowest** rung: a diagnostic below the parity gate that sizes the debt, or a last resort
> when the hand-written pipeline cannot reach parity. Its ceiling is plain parity, its numbers are
> labelled "injected" and never count as a win, and it is never applied to an incumbent
> (already-Gluon) kernel. It needs **no `libtriton.so` rebuild** and is **allowed with LLVM
> co-tuning OFF** (it runs already-compiled passes over the module the stock lowering returned; it
> does not edit or rebuild compiler source), so it sits outside the MAY-NOT list above and is not
> Scenario B.

> **There is no environment variable that arms the re-injection, and `TRITON_GLUON_*` names are
> not it.** Upstream reads no `TRITON_GLUON_*` variable of any kind; the re-injection is armed
> **from Python**, and the on-disk patch form is armed by pack-local script variables
> (`TRITON_GLUON_SWP*`; their cache-key caveat is in `non-upstream-reserve.md ## 4. Names that
> were checked and have no referent — do **not** look for these`). Two related upstream facts
> worth carrying: `BlockPingpong` is reachable only through the pass list and will **not** fire on
> hand-authored staging — it collects `local_load`s sourced from a loop-carried `BlockArgument`,
> and a hand-written one comes from `memdesc_index`; and `TRITON_ALWAYS_COMPILE` is real and
> upstream, so use it to force recompiles while iterating.
>
> The general rule behind this: **an env knob that no installed code reads fails silently by
> design.** Probe the mechanism (does the IR change?), never the variable.

**The knob names in this area are version-gated, and `schedule_hint` / `llvm_fn_attrs` are
version-disjoint; each fails silently (or raises) outside its range** — the same rule as the box
above, arriving from the opposite direction. 3.8.0 first; the 3.6/3.7 column is the downgrade:

| Lever | Real on | Outside that range |
| --- | --- | --- |
| `llvm_fn_attrs` (uncurated `name=value` pass-through, reaches `amdgpu-sched-strategy`; mechanism chapter: `llvm-fn-attrs.md`) | **3.8.0 only** | 3.6.0 / 3.7.0 / 3.7.1: not declared, so it raises as an unrecognized compile option — the attribute has **no downgrade**, so downgrade the thing you wanted it for instead |
| `TRITON_HIP_USE_COEXEC_SCHEDULER` (env; sets `amdgpu-sched-strategy=coexec` at `num_warps <= 4`) | **3.8.0 only** (default-on for gfx1250, opt-in on gfx950 / gfx942) | 3.6.0 / 3.7.0 / 3.7.1: not read by anything — **inert if set**, no error |
| `schedule_hint` (HIP backend option, curated presets) | 3.6.0 / 3.7.0 / 3.7.1 (downgrade only) | **3.8.0: a dead declaration.** The field is still on the options dataclass with no readers left, so it is accepted, hashes into the compile key, and changes no IR |

They are not replacements in kind: `schedule_hint` selected curated presets, `llvm_fn_attrs` is an
unfiltered attribute pass-through, so it is both more general and entirely uncurated.

Consequence for a sweep: **read the version before attributing a null result to the scheduler.** A
`schedule_hint` sweep on 3.8.0 and an `llvm_fn_attrs` sweep on 3.7.1 both come back flat, and both
flat results mean "this build has no such knob", not "this kernel is not ILP-starved". Probe it
rather than reading a table: `scripts/probe_levers.py --all` reports `version_disjoint_knobs` with
`live` / `dead-declaration` / `absent` distinguished.

*Provenance:* the 3.8.0 half of that table is confirmed against a complete 3.8.0 source tree
(`third_party/amd/backend/compiler.py`: `schedule_hint` declared with no reader, `llvm_fn_attrs`
applied in `make_llir`, `is_coexec_scheduler_enabled`). The 3.6.0 / 3.7.0 / 3.7.1 half came from
container probes and is consistent with those tags' `third_party/amd/backend/compiler.py` and
`python/triton/knobs.py` (no `llvm_fn_attrs`, no `TRITON_HIP_USE_COEXEC_SCHEDULER`, `schedule_hint`
read).

## VALU-between-matmul: manual interleave, or author a pass

For softmax / scale-between-matmul kernels the GEMM-class answers do not apply: a
throughput-pairing scheduler can emit invalid IR here rather than merely regressing, and
forcing AGPR accumulators is actively wrong when the accumulator is read-modified every
iteration (## 风险). The kernel-agnostic levers, in order:

1. **Kernel side (default, no rebuild):** interleave independent work in the manual
   Gluon pipeline (`pipeline.md`) — write independent `AC` / `LR` / `DOT` and
   stagger them yourself so the default scheduler cannot serialize MFMA behind the
   VALU. This needs no compiler change and works on any build.
2. **Stock scheduler strategy (no pass, no rebuild):** **read the version table
   first** — the lever table in ## Portable scheduler co-design. On **3.8.0** check the
   stock **`coexec`** strategy first — it is aimed at exactly this region class
   (opt-in on gfx950 / gfx942: `TRITON_HIP_USE_COEXEC_SCHEDULER=1` at `num_warps <= 4`, or
   `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` at any `num_warps`; details in
   ## What upstream 3.8.0 actually gives you) — then the other `amdgpu-sched-strategy`
   values through `llvm_fn_attrs`. *Downgrade (3.6.0 / 3.7.x):* `schedule_hint` presets are
   the rung there. **On 3.8.0 `schedule_hint` is a dead declaration** — accepted, hashed
   into the compile key, and read by nothing, so a sweep of it comes back flat for a reason
   that has nothing to do with your kernel. "Probe its presence on the build" does **not**
   catch this: the field is still on the options dataclass in 3.8.0, so a presence probe
   answers yes. Probe the mechanism (does the IR change?), not the name. Apart from
   `coexec` on a matrix+VALU region, a strategy helps only ILP-starved kernels, so sweep +
   A/B by bound class.
3. **Plugin tier (a host build, no LLVM rebuild):** load a **region-classifying**
   scheduler as an out-of-tree pass plugin. Regions mixing matrix with memory keep the
   throughput interleave; regions mixing matrix with VALU are instead **declared** as
   scheduling groups and built by the backend's group-pipeline solver. Because the routing
   is per region, the GEMM kernels are unaffected. Gate, the three build states, and the
   declaration disciplines: `llir-codesign.md`.
4. **Compiler side (only under sanctioned co-design):** author the **policy** yourself —
   a **dependency-preserving** schedule for your structure that does not break SSA
   dominance. The load path, extension point and transaction skeleton are in
   `llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton`; see ## Scenario B for
   the build/verify/A-B/restore loop and the sanction it needs.

**Before concluding a scheduler "does nothing" on this kernel, check two things that both
read like a hardware ceiling.** First, whether the loop's matrix **shape is in the tool's cost
model** — a shape it cannot price is skipped silently, and adding a shape row does not finish a
port because the window model needs recalibrating from that target's own co-issue counts. Second,
**who built the overlap**: an authored ring with stage markers reaches both models, while a loop
whose overlap came from re-injecting the auto-pipeliner (the lowest rung, numbers "injected")
carries no stage markers, so it can only reach the throughput model, and it arrives in plain's loop
shape rather than the shape that model's inferred regions want. Both in
`llir-codesign.md ## Applicability gate: shapes the tool does not model` and `## Route the loop
first`.

**There is no upstream attention-scheduler toggle, and its absence is not a ceiling.** A
variable name for one circulates in kernel notes (`TRITON_ENABLE_ATTN_SCHED` — no referent in
any tree, `non-upstream-reserve.md ## 4. Names that were checked and have no referent — do **not**
look for these`); upstream 3.8.0 ships neither the variable nor the pass. Do not instruct a run to
set one, and do not record a scoped ceiling because it is missing — the stock `coexec` strategy
(lever 2) is the upstream answer for the region class, lever 3 reaches placement with no
compiler-source change, and lever 1 reaches most of it with no build at all.

## Per-change protocol

1. Make the body change (memory path / LDS / pipeline / slice).
2. `base`: dump IR, confirm the structural change is present.
3. **+ scheduling** (GEMM-class only): dump IR, confirm the interleave; reprofile MFMA
   efficiency. For attention / VALU-between-matmul confirm the *manual* interleave
   (or, if tried, the `coexec` strategy's attribute in `.llir` plus an `.s` diff)
   instead — a GEMM-class scheduler can assert here rather than regress.
4. **+ AGPR accumulators** where the card calls for it
   (`llvm_fn_attrs="amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0"`, 3.8.0): dump IR,
   confirm the accumulator is in AGPR; reprofile. GEMM-class only — skip for
   VALU-between-matmul, where it is actively wrong (## 风险).
5. Keep the configuration that maximizes MFMA efficiency without spills.

## IR dump workflow

Use `scripts/dump_ir.sh` (a shim; the tool lives in `kernel_workflow/scripts/kernel_tools/dump_ir.sh`),
e.g. `bash scripts/dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts --arch gfx950`
— `--emit-gluon` consumes the arch and refuses when none is given or detectable. Adding `--emit-gluon {layouts|anchor|pipeline}` also auto-recovers
the inferred layouts (and, opt-in, the async pipeline) into Gluon via
`scripts/recover_gluon.py` — gluon pack — and closes the transcribe loop with a layout-equivalence
`--verify`. The dump itself works in both packs; `--emit-gluon` needs the gluon pack, because
transcription is the deep-dig tier's job (the triton pack exits 3 and says so). Recoverable from TTGIR: layouts (deterministic) and the pipeline structure
(opt-in). **Not** recoverable from TTGIR: register allocation / spills -- those happen in
LLVM after `make_ttgir`, so they are never recovered here and stay slicing + RA hints.
Artifacts per variant (one `TRITON_CACHE_DIR` each):

| File | Role |
| --- | --- |
| `.ttgir` | Triton GPU IR — shows layouts (`#blocked`/`#mma`/`#shared`) + `num_stages` (consumed only on the plain path; on the Gluon path in 3.8.0 no pass reads it — record it, do not tune it) |
| `.llir` | LLVM IR |
| `.amdgcn` | emitted assembly |
| `.s` | `.amdgcn` stripped of `.loc` and `.Ltmp*:` for stable line anchors |

Strip command:

```bash
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' in.amdgcn > out.s
```

Pin **one** Triton build for the whole comparison and record which tag it was, so dump line
numbers are comparable across variants; the tag lineage moves and a dump taken on another
one is not a control.

### Auditing the hot-loop schedule from the `.s`

To judge a loop's *scheduling quality* (not just whether a structural change
landed), run the mechanical pass with `scripts/asm_loop_audit.py <variant>.s --arch gfx950`
(shim; the tool lives in `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`); it
emits the loop's **op-class symbol stream + histogram**, classifies each
`s_waitcnt` as relaxed (`lgkmcnt`/`vmcnt` `> 0` = the compiler is pipelining the
consumer behind in-flight loads = good) vs full-drain (`(0)` before every consumer
= conservative, serialized handoff), and counts producer<->consumer sync barriers
per iteration.

**First, check how many kernels the dump holds.** A module that emits more than one — a split
reduction, a prologue, a fused epilogue compiled alongside the main body — has no safe default
pick, so the audit **refuses and lists them** rather than choosing; pass `--kernel <name>` (a
unique substring is enough). Do not work around it by auditing the module as a whole: a
whole-file read maxes the register budget across kernels and scores every kernel's loops
together, which yields a complete, internally consistent report about a kernel you did not ask
for — with no error and nothing in it that looks wrong. **The general form of that hazard is
worth carrying past this tool: when a reading is assembled from several sources, check that the
thing being measured and the name being printed were chosen by the same step.**

Then read the verdict:

- for each **exposed single-class run** (a long run of one class with no
  interleaved MFMA / overlap target), ask: *was there any independent ready op that
  could have filled it?* If **no**, it is **structural (no slack)** — a
  scheduler/asm change cannot help (see the reorder-needs-slack rule in
  `## Scenario B`); if **yes**, it is a real scheduling miss worth a co-design pass;
- repeated full-drain waitcnts = **conservative-waitcnt**; barriers-per-iter scaling
  with pipeline depth = **excess-sync**;
- **`s_nop N` in the body = an exposed FIXED-latency hazard** (a hardware result
  write->read latency, classically MFMA-accumulator-write -> VALU-read of that
  result; the audit counts `s_nop` + the requested stall cycles). Unlike a
  conservative waitcnt, this is **not** reorder-fixable and **not** a scheduler miss
  — a fixed-latency hazard is hidden only by *filling* it with independent work
  (more unroll / more resident waves), so high `s_nop`/iter means raise occupancy or
  unroll, not reorder. The same hazard reads `s_nop ~= 0` once the unroll + waves/CU
  cover it (`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`).

The audit output is a *verdict* (`structural` / `conservative-waitcnt` /
`excess-sync` / `unhidden-fixed-latency`), not a fix. Only conservative-waitcnt and
excess-sync are reorder-actionable (and only when reorder slack exists); a high
`s_nop` is occupancy/unroll-actionable, not a scheduling problem. A schedule can have
perfect op ordering yet be slow purely from full-drain waitcnts or unhidden `s_nop`.
The script reports signals only; the structural-vs-schedulable judgment stays with you.

## Preconditions

- `K_is_constexpr: yes` is a hard prerequisite for a scheduled interleave.
  If `K` is dynamic, either route to a constexpr-K bucket + dynamic-K
  fallback, or record a scoped ceiling and use non-scheduled mechanisms only.
- A multi-region or >=3-stage pipeline must have filled schedule-math values
  (wait groups, loop skeleton, TCP/FIFO budget, epilogue/store spacing) before
  you claim the scheduler will interleave it. Otherwise keep the pipeline layer
  open.

## What "landed" means (the IR half of three-evidence closure)

A layer closes only when, in addition to budget consistency and a positive
profile delta, the IR/asm shows the intended signal from the tables in
`memory-path.md` / `pipeline.md` / `slicing.md` / `low-precision.md`. If the
profile improved but the IR signal is absent, the win may be noise or an
unrelated effect — do not close the layer.

## Body micro-edits the compiler may already handle

Expect **neutral** results unless IR/ISA/profiler evidence says otherwise; do not
spend multiple rounds on these without a named compiler miss:

- simple instruction reordering for load/compute overlap;
- dead allocation or unused-helper elimination;
- scalar broadcast expansion;
- scheduling-hint identity/no-op elimination;
- moving independent value conversions across unrelated compute.

Benchmark if the hypothesis is cheap, but the burden of proof is an IR/asm or
profiler signal, not intuition.

## `tl.assume` audit

`tl.assume()` can prove facts to the compiler, but excessive stride/shape
assumptions may constrain optimization on newer Triton builds. When body work is
otherwise neutral, test a same-source variant with unnecessary assumptions
removed. Keep assumptions needed for correctness, division/modulo safety, or
required lowering behavior.

## `tl.trans` operand reuse

On CDNA matrix paths, transposing a large operand between two `tl.dot` calls may
require a real register shuffle, not a free reinterpretation. If `tl.trans` reuse
is slower than a separate load, the kernel is limited by register movement or
memory latency, not HBM bytes alone — prefer loading data directly in the layout
each dot needs.

## CDNA4 scaled-helper argument types

CDNA4 scaled-matrix helpers such as `gl.amd.cdna4.scaled_upcast` accept a
Triton dtype **object** for the source/result type, not a `"bf16"` string.
Passing the string can produce an error message that itself mentions `bf16`,
reading like a target ceiling. Pass `tl.bfloat16` / the appropriate `tl.float*`
object first; only if it still rejects the type is it a real target/API ceiling.

## Async / buffer path failures are layout-first

`buffer_load_to_shared` can fail in lowering for layout-contract reasons before
backend-support reasons:

1. shrink to an async-copy smoke against the same shared buffer shape, dtype, and
   consumer load;
2. verify offset layout family, rank, dtype, unit, and total size for the copy
   width;
3. if `BlockedLayout` lowering fails, try a source-proven
   `DistributedLinearLayout` matched to the consumer before recording a ceiling;
4. the offset layout must be **pre-coalesced** for `canLoadDirectToLDS` (contiguous
   threads on the LDS-fast dim, 128-bit run, `vec == contig`) — the Gluon lowering
   runs **no** `CoalesceAsyncCopy` pass (plain does), so an un-coalesced offset
   silently falls back to register staging + `ds_write` or fails lowering
   (`memory-path.md ## Async copy: smoke-test before wiring`);
5. if the async path is correct but reads wrong values, that is a staging-plan
   correctness gate, not a buffer-op rejection;
6. only after well-formed layouts persistently fail is it a build capability
   ceiling.

*gfx942 downgrade:* direct-to-LDS is 32-bit only there and the destination must be `order=[1,0]`;
a padded destination fails translation, so build the swizzled destination first, and fall back to
sync staging (register load + `ds_write`) rather than recording a ceiling
(`memory-path.md ## Async copy: smoke-test before wiring (mandatory)`).

## Source-sensitive traps

- A source-proven `layout: gl.constexpr = Layout(...)` inside a JIT body is not
  the same as generated runtime layout construction.
- `tl.*` scalar/control-flow helpers may be acceptable when source-proven; tensor
  dataflow (`tl.arange`, `tl.load`, `tl.store`, `tl.dot`) needs an explicit Gluon
  plan.
- `DistributedLinearLayout` basis vectors can freeze tile dimensions — check basis
  offsets before changing `BLOCK_SIZE_*`.
- AOT compile success does not prove the package/runtime path is valid.
- A production-framework editable install can replace the installed Triton
  (preserve it via the framework opt-out, e.g. `AITER_USE_SYSTEM_TRITON=1`); record
  install policy before interpreting version-sensitive failures.
- Triton updates can change scalar return dtype (`int32` vs `int64`) even when
  tensor math is unchanged; fix at the wrapper/oracle boundary, separate from
  numerical correctness.

## Blacklist-style warnings

- Do not fix verifier failures with arbitrary powers of two.
- Do not move layout casts earlier just because they look cleaner.
- Do not treat index tensors as ordinary value tensors for layout conversion.
- Do not add target-specific matrix/memory ops before a simpler path is correct.
- Do not invent helper names or import paths (`../method/triage.md`, missing-doc protocol).
- Do not add a `constexpr` or pipeline stage without checking unrelated branch
  specialization and resource footprint.
- Do not carry negative body-level conclusions across Triton/ROCm updates without
  re-testing compiler-sensitive knobs.

## CDNA scheduling barriers (late-stage, and narrower than the mechanism suggests)

Scheduling hints are late-stage overlap/ordering controls, not a substitute for understanding the
hot path: reach for them only after the algorithm, layout, memory path and benchmark boundary are
credible. Consider them when the kernel already has an executed path, the VMEM/LDS/MFMA work is
independent enough to overlap, and timing suggests instruction scheduling rather than traffic
volume is the bottleneck. Do **not** start here when correctness is unstable, layouts are still
changing, the kernel is bandwidth-bound, or the only evidence is one quick run.

**What is actually reachable is narrow.** `sched_barrier` / `sched_group_barrier` / `iglp_opt` have
no user-facing surface on 3.8.0 from either tier, and adoption across surveyed gfx950 production
kernels is zero. The masks, the `count` discipline, what the Gluon path does reach indirectly, and the
three mechanisms production uses instead are in
`instruction-scheduling.md ## The declarative hints that are not there`. If a helper is not exposed
by the installed API, record a version/API blocker (`../hardware/capability-matrix.md`) rather than
concluding the mechanism cannot help.
