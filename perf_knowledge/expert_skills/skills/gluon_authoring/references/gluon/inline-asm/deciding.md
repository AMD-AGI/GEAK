# Do I need inline asm here? — four evidence lanes

Part of `../inline-asm-reference.md`; first rule out `## Before you reach for it` there. **Lane 0
(structural facts) names the reasons this page knows**; Lanes 1–3 sharpen a Lane-0 hypothesis. A dump/profile
signal with no Lane-0 rule behind it is an untested hypothesis: an asm site built on it is an I-D experiment to A/B.

## What this page can localize, and where it stops

Instruments reach kernel → loop/stage, rarely the line: the commonest site (empty asm) emits zero
instructions, so profilers see the consequence (the MFMA where pressure is *spent*) never the lever (the
producer→consumer edge). Exception: spills emit `scratch_*`, which carry source lines — aggregating them
by line localizes the block. **No hotspot at a site is not evidence it does nothing.**

## Lane 0 — structural facts, readable from the kernel alone

The first rule that fires names the cell. Each: signal → cell → write → wrong if.

- **L0.0 headroom (applies to I-D only):** launch-parameter / memory-round-trip / pipeline slack is the
  cheaper thing to check when the reason for an I-D site is register or schedule pressure (in the upstream
  rebuilds most gaps closed this way); one knob swept is not "no headroom"; record a skipped L0.0 as skipped.
- **L0.1 order without data:** two ops whose order is correctness with no value flowing between them ⇒
  **I-B**. The bindings (`gl.barrier()`, `commit_group`/`wait_group` — `../pipeline/async-ordering.md ## Getting wait_group right`)
  keep the compiler's dependency checking; the gap asm fills is intra-wave, sub-group order — canonically the
  LDS **reader** drain before a refill. Wrong if a value does flow (then the data dependency already orders them).
- **L0.2 cross-CTA/device handshake** needing a poll loop or a cache scope ⇒ **M3/I-A** protocol block —
  when `gl.atomic_*(sem=, scope="sys")` and the uncached-memory route with `.cv` polls do not cover the
  protocol (`../../workloads/collective.md ## Publish and consume`). What usually survives: the publish fence
  (bare drain + barrier) and a poll loop that needs a wall-clock bound or many sources per iteration.
- **L0.3 the answer depends on the `HW_REG_MODE` narrow-float bit** ⇒ **I-A/M1**, normally a set/restore bracket (the bit is sticky per wave;
  an unrestored write is legal only under a launch partition — `classes.md ## Class 3 — machine state`).
  Wrong if a clamp / `gl.where` / conversion rounding produces the difference.
- **L0.4 algorithm defined over lanes** (ballot, elect, rank, lane exchange, reg↔lane bit swap) ⇒ **I-C**
  cross-lane, construction A/B/C. Wrong if a layout puts the axis on lanes and `gl.max`/`gl.sum` does it
  (a common shape: tensor reduction, then a lane ladder for the remainder).
- **L0.5 a downstream instruction form needs an SGPR or a register pair** ⇒ **I-C/M0** (`"=s,0"`, 64-bit
  tie). Wrong if you cannot name the uniformity proof (`program_id` / constexpr / `static_range`).
- **L0.6 `hasattr(rocdl, '<name>')` is False and the mnemonic assembles** ⇒ **I-C/M1**, `is_pure=True`,
  plus the plain-Gluon `else:` **in the same commit**; reproduce the caller-side contract (`classes.md`).
- **L0.7 live range (L0.0 slack is the cheaper check):** read the static `.amdgcn` in order — (a) spill
  slots / private segment / `scratch_*` census, (b) AGPR count 0→positive with unchanged MFMA count, (c)
  VGPR demand only when spills are zero (a spilling kernel reports its cap). Occupancy steps: `../../hardware/planning-constants.md ## VGPR / occupancy thresholds`.
  ⇒ **I-D/M0-tie** on the loop-carried accumulator (after its last update, before its next use); check first
  whether the MFMA builtin takes a register-class argument. **Wrong if the normalized-dump hash is unchanged
  (inert) or spills move the wrong way — removing a tie is an equally valid response.** Several ties on
  one budget: `classifying.md ## BUDGET-GROUP`.
- **L0.8 nothing fired ⇒ this page has no structural reason for asm here.** A site written anyway is an
  I-D experiment: A/B it against the plain arm (Lane 2). Record target, compiler version, launch geometry and shape bin with any decision.

## Lane 1 — the compiled artifact

Dump with `dump_ir.sh` + `TRITON_ALWAYS_COMPILE=1`; hot-loop reading procedure owned by `../../tile-programming/compiler-contract.md`.

| Signal in the `.s` | Cell / write | Wrong if |
| --- | --- | --- |
| full drain (`cnt == 0`) where only some traffic matters, or a wait before the loads it covers | prefer **I-D/M0-tie** of the read result + `~{memory}` (no count to go stale) over **I-B** `vmcnt(N)` | same wait, same place after the change — revert; that signal does not support a hand count |
| a copy/store/LDS read crossed a barrier or wait vs source order | untied marker `("", "=v,~{memory}", [])` | the motion followed a value dependence you broke; note it may protect a hand-counted `vmcnt` |
| provably uniform value used as a vector base | `"=s,0"` next to the uniformity proof, or `v_readfirstlane_b32` | still a vector base, or a compiler `v_readfirstlane` ⇒ not uniform |
| hot-loop `scratch_*` around a big accumulator | widen a fragment tie — or remove one | spills do not move / move the wrong way (a direction, never a conclusion) |
| your mnemonic is missing | — | verification failure: `costs.md ## Verifying that it worked — write, compile, dump, check` |

## Lane 2 — counters and timing

**Admissible only for I-D, between two already-correct arms** (I-A/I-B/M3 are verified, not timed). **Prior,
not a ranking for your kernel: in an upstream A/B over one asm-heavy production pack (not reproduced here)
most sites moved nothing, register-class pins and instruction selection moved most often, and mode writes,
drains and `s_setprio` brackets least** (no priority site there moved past the noise floor). GEAK runs
that tried exact waits and `s_setprio` kept no gain in the cases tried; the positive half has no local
evidence. A/B on your kernel before closing either direction. Procedure: hash first (step 1.5) → preflight PMC (`../../method/profile.md`)
→ asm vs `else:` arm behind one constexpr → interleave arms in one window (`../../method/benchmark-hygiene.md`)
→ prefer clock-insensitive counters (`../../hardware/bound-class-signals.md ## rocprof-compute block-ID cheat-sheet`)
→ close the loop in Lane 1 (`../../tile-programming/instruction-scheduling.md ## Acceptance`) → noise
floor from a byte-identical arm → budget groups need all-off + grouped arms → re-measure across dispatched
shapes (the deciding axis may not be in the dispatch key).

## Lane 3 — a source-level audit of *your* kernel

AST pass over call sites (match `inline_asm`, keep arguments as expressions). Hard stop: output slots ≠
arity law. Flags: tie digits not `0..n-1`; `is_pure=True` + `~{memory}`; `is_pure=True` result not
rebound; `vcc`/`scc` written without clobber; physical registers; shape-constant guard (⇒ bins); `if`
with `else` (delete candidate only); I-B with no `PAIRED-WITH`; no `static_assert` / `assert_trivial` near
the site.
