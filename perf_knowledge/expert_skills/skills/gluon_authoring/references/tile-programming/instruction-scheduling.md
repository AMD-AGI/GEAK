# Instruction scheduling — pacing what the overlap already created

Read this **after** an overlap structure exists, not instead of building one. The pipeline layer
decides which instructions can overlap at all (`pipeline.md`); this layer decides where in the
instruction stream they land. A lever from this page applied to a loop with nothing to interleave
measures as noise, which is the most common way it gets written off. In the overlap order
(`pipeline.md ### The order to reach for these in`) this page is the intra-wave half of rung 3 —
the realization of `compiler_interleave` and of hand-authored pacing — after register prefetch and
the authored ring (rungs 1–2), and above re-injecting plain's pipeliner (rung 4). It is the single
statement of the per-instruction scheduling levers; `scheduling-model.md` (which model) and
`warp-pipeline.md` (the `inter_wave` realization) are its two siblings, and other pages point here
rather than restating these mechanisms.

**Both tiers reach this layer**, and that is unusual enough to state: `llvm_fn_attrs` and the LLVM
pass plugin are applied in `make_llir` / `optimize_module`, which the plain and Gluon lowerings
**share**. So nothing on this page needs a re-injection shim, a monkeypatch, or a rebuild.

Two references this page leans on rather than restates. The API surface for the two inline-asm
mechanisms below — the constraint strings, `is_pure`, and what the only asm door in Gluon can and
cannot enclose — is `../gluon/inline-asm-reference.md`. Whether a reorder can pay at all is an
instruction-rate question, and those rates are in
`../hardware/isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign (CDNA3/4)`:
a pairing the hardware cannot co-issue does not become schedulable by moving it.

## What production actually reaches for

On gfx950 production Gluon kernels this layer is reached for **second only to the authored ring
itself**. It is not one technique but three that barely overlap, and the split matters because
they have different costs and different failure modes:

| Mechanism | Production use | What it controls | Reaches |
| --- | --- | --- | --- |
| empty-asm scheduling fence | **the most common construct here** | a data dependency the scheduler must respect, emitting no instruction | Gluon (needs inline asm) |
| `s_nop` placement | common, concentrated in dependency ladders | explicit delay between dependent instructions | Gluon (needs inline asm) |
| `llvm_fn_attrs` | common, and decided per kernel rather than per workload | the LLVM machine scheduler's strategy, per compile | **both tiers**, and neither DSL is a precondition — `llvm-fn-attrs.md` |
| stock `coexec` strategy (3.8.0) | set by the backend only on gfx1250 at `num_warps <= 4`; on gfx950 / gfx942 an opt-in A/B | the matrix + VALU co-execution machine-scheduler strategy | **both tiers**: `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (process-wide, `num_warps <= 4`) or `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` (per compile) — `compiler-contract.md ## What upstream 3.8.0 actually gives you` |
| `sched_group_barrier` / `iglp_opt` / `schedule_hint` | **never** | — | see `## The declarative hints that are not there` |

That last row is the useful negative: the declarative scheduling-hint family that this layer is
usually *described* in terms of has no user-facing surface on 3.8.0 and no production adoption.
Do not budget a round on it before reading that section.

## llvm_fn_attrs — the portable, per-compile scheduler strategy

**The mechanism has its own chapter: `llvm-fn-attrs.md`.** Read it before the first use or the
first sweep — it owns what you may write on the option, the two failure modes, the assembly-diff
acceptance procedure, the cost, and the version story. What stays here is the part that is a
*scheduling-layer* decision: whether this layer should reach for it at all.

**3.8.0 only.** On 3.6.0 / 3.7.0 / 3.7.1 the option does not exist on the AMD backend, so passing
it raises an unrecognized-compile-option error rather than measuring flat. That distinction is the
whole reason to probe before sweeping: `python3 scripts/probe_levers.py --all` reports
`version_disjoint_knobs`, which separates `live` from `absent` and from `dead-declaration`.

```python
kernel[grid](
    ...,
    num_warps=4,
    # a sequence of (name, value) pairs; `"amdgpu-sched-strategy=iterative-ilp"` also parses
    llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]],
)
```

What the option accepts, how it overrides the backend's own attributes (including a
backend-set `coexec` on gfx1250 and the attributes derived from `waves_per_eu` / `num_warps`),
which strategy names are evidenced in the Triton tree, and why an unrecognized attribute measures
as a silent null result are stated once in `llvm-fn-attrs.md ## What you can write on it` and
`llvm-fn-attrs.md ### Scheduler strategy — be honest about which names are established`. The consequence for this
layer: **the assembly diff is the acceptance signal, not the timing**
(`llvm-fn-attrs.md ## Verifying it took effect — the assembly diff is the only signal`). On gfx950,
`amdgpu-sched-strategy=coexec` through this door is the per-compile way to try the stock
co-execution strategy, which the backend does not set there on its own.

**It is decided per kernel, not per workload.** Structurally similar kernels — same archetype, same
tile shape, adjacent launches in one dispatcher — routinely want different settings, or want it on
one and not the other. There is no archetype-level or dtype-level rule to inherit, so a setting
lifted from a neighbouring kernel is a guess that still owes you an assembly diff
(`llvm-fn-attrs.md ## And the adoption rule that surprises people: this is a per-kernel patch`).

**Rule — only ILP-starved kernels benefit; sweep by bound class.** A non-default strategy helps
when the kernel is ILP-starved: low occupancy (e.g. 1 wave) with independent work the default
scheduler leaves unscheduled, so an ILP-iterative strategy packs more matrix ops per iteration.
When the kernel is already occupancy-rich (2+ resident waves hide latency) or throughput-bound (a
unit saturated), the strategy is neutral or regresses, and the default is usually best for
LDS-bound / memory-clause-friendly loops. So sweep by bound class and A/B it; do not default it
on. **Apply the ILP-starvation rule only once the knob is known live on the build in front of
you** — otherwise you are reading a version fact as a kernel fact.

## The empty-asm scheduling fence — the most-used and least-documented lever here

An inline-asm block with an **empty asm string** and a **tied operand** emits no instruction at
all. What it creates is a data dependency the **compiler's** scheduler must respect, which lets you
bound instruction motion between two computations without paying an instruction for it.
**"Fence" is this layer's local jargon and it is a motion boundary, not a hardware fence.** It
issues nothing, waits for nothing and orders nothing at the hardware level, so it never substitutes
for a `gl.barrier()`, a `wait_group` or an `s_waitcnt` — and adding `~{memory}` to the constraint
string does not change that, it only widens what the *compiler* may not move across. Reading it the
other way and then deleting the real barriers next to it is the single most likely error on this
construct (`../gluon/inline-asm-reference.md ## Class 2 — scheduling control`, which
quotes the production comments that say so). In production this is the single most common
instruction-scheduling construct on the Gluon path, and it is
usually written across several lines, which is why a single-line grep under-counts it badly.

```python
# Pin an accumulator so the scheduler cannot sink or hoist work across this point.
# "=v,0" ties output 0 to input 0: same register, so the dependency is real to the
# scheduler while no instruction is emitted. is_pure=False is REQUIRED -- a pure
# block whose result nobody reads is deleted, taking the fence with it.
acc = gl.inline_asm_elementwise(
    "",                      # empty: this is the whole point
    "=v,0",
    [acc],
    dtype=acc.dtype,
    is_pure=False,
    pack=1,
)
```

Two details that decide whether it works:

- **`is_pure=False` is load-bearing**, not defensive. The op registers resource-free read+write
  memory effects **to the compiler's effect system** only when impure — that is what keeps the
  passes from moving or deleting it, and it is not a machine-level effect; pure blocks are
  dead-code-eliminated when their result is unused, and "unused" is easy to arrange by accident.
- **The constraint string has to cover the whole value.** A production accumulator is many
  registers, so the constraints are generated from the tile size rather than written by hand —
  `"=v,0"` repeated per 256-element group is the shape you will see.

**Verify by placement, not by timing alone**: the fence emits nothing, so the evidence is that the
instruction mix *around* it moved in the disassembly. Phase effects are non-monotone — sweep the
placement rather than bisecting it, and read the `s_nop` / `s_waitcnt` census before and after so
you are not looking at a hazard the compiler inserted, which is a different problem with the
opposite fix.

**Risk, and it is the highest effort-to-durability ratio on this page:** a scheduling win is
pinned to one compiler version and evaporates on upgrade with nothing warning you — the asm still
assembles. Record the toolchain identity next to the number.

## `s_nop` placement — a pacing lever, not a hazard fix

Explicitly placed `s_nop` is the deliberate-delay counterpart of the fence: use it when dependent
instructions need to be spread rather than merely ordered. Production uses it in dependency ladders
(DPP reduction chains) and after lane-broadcast reads.

The reading trap is that `s_nop` in the disassembly usually means the **opposite** thing — a
hazard the compiler inserted, which you fix by removing the dependency, not by adding delay. Tell
them apart before acting: a placed `s_nop` is one you can find in your own source; an inserted one
appears in the asm census and not in your asm strings.

## Wave priority — the hand-rolled form, and its four roles

`s_setprio` raises or lowers the issuing wave's arbitration priority. There is a checked route to
it — the warp-pipeline stage markers emit it from a stage's `priority=` — but that route needs two
wave groups and a stage body free of waits, so the form that appears inside ordinary bodies is a
bare `s_setprio` through inline asm. It belongs on this page for the same reason the empty-asm
fence does: single-wave, hand-placed, and selected per shape.

**Read the marker path's polarity rule as scoped to the marker path.** "Memory outranks compute"
is a statement about two wave groups contending for the same issue slots, where raising the
compute group starves the memory group's address updates and the overlap disappears. The
hand-rolled form usually runs at `num_warps` of 1 or 4 — **one group, nothing to starve** — and
there both polarities are shipped forms. Four roles are in use across surveyed production source:

| role | shape |
| --- | --- |
| **memory-high** | priority raised across the wait / LDS-read / barrier / copy window, dropped before the matrix op |
| **compute-high** | exactly the inverse: memory window pinned at 0, priority raised before the matrix op |
| **blanket** | the whole main loop at one raised priority, arbitrating waves that sit in different kernel phases |
| **whole-CTA arbitration** | the CTA raised once and **never lowered**, giving one role in a role-split kernel priority over the other |

**Polarity, window width, and transitions per iteration are all shape-gated parameters, not a
recipe.** The strongest evidence is intra-file rather than cross-file: in one surveyed prefill GEMM
the *same function* runs two narrow windows per K-group below an M threshold and a single wide
window above it — one that swallows an LDS read, a barrier and a copy issue. Same author, same
edit, two M regimes. That rules out style as the explanation and puts this knob in the same class
as `llvm_fn_attrs` above.

**Placement is a compiler-observable condition, not a matter of taste.** The MFMA lowering checks
for a preceding `SetPrioOp` with `getPrevNode()` — so the raise has to be the **immediately
preceding op**, and one statement in between silences the recognition
(`v3.8.0:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`, in both `convertDot` and
`convertScaledDot`). This constrains only the hand-rolled form; on the marker path the pass places
the mnemonic itself and adjacency is automatic.

Two cautions. These counts are **static sites**, so they say a form is written, not that a bin
executes it. And the two mechanisms are not a combined design: across surveyed production source
**no compilation unit contains both** a stage marker and a hand-rolled `s_setprio`, so there is no
instance of priority arbitrating *between* pipeline stages to copy.

## How long an LDS burst may be before placement starts to matter — a countable precheck

The facts this rests on are in
`../hardware/isa-mechanisms.md ## LDS instruction behavior (gfx950)`, which owns them and states
its own calibration status; what belongs here is the placement judgement they route to.

The request queue in front of LDS is **8 entries deep per SIMD pair**, so a cluster's LDS ops are
not all the same price: the ones that fit the queue are absorbed, and the ones past it wait for it
to drain. That gives this layer a threshold it can actually count instead of a preference:

- **Count `ds_read` + `ds_write` per cluster, not per loop.** `scripts/asm_loop_audit.py --opcodes`
  reports the hot-loop mix; the unit that matters is the run of LDS ops between two non-LDS
  instructions, because that is what occupies the queue at once.
- **At or under 8, placement is not the question** — a scheduling change to such a cluster is
  expected to read as noise, and writing it off is the correct read rather than a missed win.
- **Past 8, the lever is interleaving, not prefetch.** Non-LDS work placed *inside* the cluster
  lengthens the absorbed window; more LDS loads issued to hide latency lengthen the queue instead,
  which is the mechanism by which a latency-shaped fix makes a throughput-shaped problem worse.
- **`ds_write` counts against the same queue.** A cluster that looks read-only in the source can be
  over the threshold in the asm once the stores are counted, so count the emitted mix and not the
  source.

**Status: registered as a hypothesis, not measured.** The queue depth is a machine-readable
constant; "8 is where a cluster starts to pay" is a *model* of it, and neither the threshold nor the
size of the penalty has been measured in this repo. Use it as a precheck that says which clusters
are worth an A/B — over-8 clusters first — and record the A/B result as the evidence. Do not quote
it as an effect size, and do not restructure a cluster on the strength of the count alone.

## Where a scalar operand is loaded — a zero-resource placement axis

The levers above pace or fence instructions that already exist. This one only moves **when** one
instruction issues, and unlike a tile or depth change it costs **no register and no LDS**, which makes
it the cheapest thing on this page to try.

The operand is a scalar the loop needs but does not stream: a `descale` / `softmax_scale` factor, a
per-tensor quantization constant. The candidate positions are the sides of the synchronization the loop
already has — before vs. after the **first async wait**, before vs. after **operand prefetch**, or
after the whole **QK / first matmul** instead of before it.

Production treats this as an enumerated axis rather than a default: one kernel family exposes it as a
four-valued `constexpr` and its shape variants do not all pick the same value, which is the tell that
there is no single right answer. Sweep it — four compiles, no resource change, and the reading is just
the interleaved A/B.

**Status: registered, not priced.** No measured win is claimed. The reason to try it is that it costs
nothing, not a known effect size, and a flat result is a real answer for that kernel rather than a fact
about the axis.

## The declarative hints that are not there

`sched_barrier`, `sched_group_barrier` and `iglp_opt` are how AMD scheduling control is usually
*described*, and on 3.8.0 they are **not reachable from either tier's language surface**. A
whole-tree search finds them only in Proton's profiling instrumentation, which is not a user lever.
Production agrees: they appear in no production kernel at all.

What this leaves:

- **`sched.barrier` with mask 0 is reachable indirectly on the Gluon path**, as a side effect: the
  wave-level stage markers emit one at each boundary. You get it at the boundary and you do not
  choose the mask. Because there is no other spelling, **a stage marker written purely for this fence
  is a legitimate shape** — one resident wave, possibly a `static_range` loop, possibly no matrix op
  at all. The wave-pipeline page's candidate gates grade the *phase offset* and do not apply to it;
  read `warp-pipeline.md ### Before reading a "no" as a defect` before calling such a kernel broken.
- **Everything finer is inline asm**, which puts it in the same category as the fence above.
- A shipped production kernel imports these three names inside a `try/except` and defines **no-op
  stubs** on `ImportError`. That kernel is running the stubs and its hints are dead code. Do not
  copy the pattern expecting scheduling control.

For completeness, the mask values that circulate for `sched_group_barrier` (`0x008` MFMA, `0x020`
buffer load, `0x100` DS read, `0x200` DS write, `0x000` reset) describe the **LLVM intrinsic**, not
a Triton-level API. They are only actionable through inline asm, and the `count` argument has to
match the number of relevant instructions in the controlled region — count source instructions, not
loop iterations, and update the counts whenever tiling, unrolling or stage structure changes. A
count that helps one shape bucket routinely hurts another.

## Where this sits relative to the other layers

| Layer | Owns | Page |
| --- | --- | --- |
| overlap structure | whether two instruction classes can overlap at all | `pipeline.md` |
| **instruction scheduling** | **where in the stream they land** | this page |
| wave-level phase offset | whether two waves run a stage apart | `scheduling-model.md`, then `warp-pipeline.md` |
| authored LLVM pass | a scheduling policy this layer cannot express | `llir-codesign.md`, `llvm-codesign-handbook.md` |

Two ordering rules follow from that table. **Do not reach for an authored pass before this page**:
a per-compile attribute needs no rebuild, no plugin and no sanction, and it covers the
VALU-between-matmul case the GEMM-only env knobs reject. And **read a scheduling regression as a
layout question first** — on gfx950 a campaign that added seven LLVM co-execution regions at zero
AGPR still lost to a plain synchronous retiling, so the residual was never in the schedule
(`llir-codesign.md ## Route the loop first: who built the overlap`).

## Acceptance

A scheduling change is landed on the same three-evidence bar as any other layer, with one
addition specific to here: **the static read is mandatory rather than optional**, because the whole
claim is about instruction placement. `scripts/asm_loop_audit.py` for the hot-loop mix,
`scripts/mfma_efficiency.py` for cadence, and a same-window A/B for the timing. A scheduling win
that shows in timing but not in the instruction mix has not been attributed to the scheduler yet.
