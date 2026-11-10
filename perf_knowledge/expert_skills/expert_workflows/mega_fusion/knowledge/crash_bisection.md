# Crash Bisection — Locating a Fault Inside One Kernel Without a Debugger

A **method** card for the situation a fused kernel puts you in: the kernel dies with an illegal
memory access or hangs, the failure is 100% reproducible, and there is no line number. A fused
kernel is a single launch, so nothing about *where* it died is visible from the outside — the trace
has one record, and that record says "it failed".

The reflex is to read the code harder. That does not converge: a fused megakernel is thousands of
lines of index arithmetic and every line looks plausible. What converges is bisection, and the unit
being bisected is **how far the kernel is allowed to get**.

## The compile-time truncation ladder

Instrument the kernel with numbered cut points and select one at build time:

```
    cut 0   ──  return immediately after the prologue
    cut 1   ──  return after the descriptor/index computation
    cut 2   ──  return after the first tile's loads issue
    ...                       (15 cut points is a workable number)
    cut N   ──  no truncation; the whole kernel
```

Each cut is a `return` guarded on a compile-time constant read from an environment variable, so
every rung is a **different binary with no runtime branch on the hot path**. Then walk the ladder:
the first cut that fails is the segment that contains the fault. One measured instance ran at
**~33 s per rung** — fifteen rungs is under ten minutes of wall clock, and each rung's answer is
binary and uncontestable.

Why compile-time and not a runtime flag:

- A runtime branch changes register allocation and scheduling, so the truncated build and the full
  build are not the same kernel. A heisenbug that moves when the flag moves has told you nothing.
- Truncation deletes the code below the cut, so **the code that would have run cannot be blamed**.
  A runtime early-exit still compiles everything below it, and on a resource-limited kernel that
  code is still costing registers and LDS — which may be the actual fault.
- Dead-code elimination below the cut is a feature: if a rung stops failing because the compiler
  removed something, that something is implicated. Check the emitted ISA per rung and record the
  register/LDS/spill numbers alongside the pass/fail. The ladder produces an occupancy sweep for
  free.

## What the rungs must be

Cut at **semantic boundaries**, not every N lines. The useful boundaries in a fused kernel are the
ones where a class of fault becomes possible for the first time:

| rung is placed after | first becomes possible |
|---|---|
| prologue / role assignment | a block index outside the resident range |
| descriptor and index computation | an out-of-range table lookup |
| the first load issue | a bad address, a missing bounds clamp |
| the first barrier | a non-uniform barrier — a subset of waves reaching it |
| the first counter wait | a deadlock (target never reachable) |
| the first counter publish | a premature publish, visible only to a consumer |
| the epilogue store | a bad output address, an aliasing write |

Two rungs deserve special handling because they fail *differently*:

- **A rung that hangs instead of crashing has still answered.** Hang and illegal-access are
  different faults and the ladder separates them for free; run every rung under a timeout so a hang
  is a recorded result rather than a lost lease.
- **A rung that passes when the one below it failed is a finding, not a mistake.** It means the
  fault is masked by code that runs later — most often a clamp or a barrier that happens to make an
  out-of-range index harmless.

## Size the hang watchdog to the healthy run, not to the lease

"Run every rung under a timeout" (above) is not enough on its own: the timeout must be sized to how
long a **passing** rung takes, not to the whole-lease cap. A healthy fused correctness rung is
seconds-to-tens-of-seconds (the ~33 s measured above). Setting the per-rung timeout to the lease cap
(`GEAK_GPU_RUN_TIMEOUT`, e.g. 15–30 min) means every hung rung sits in deadlock for the full cap
before the coarse `timeout` fires — a three-rung bisection of a hang then costs ~40 min of an 8-card
lease doing nothing. That is the "it stayed stuck for a long time before anything reacted" failure,
and it is self-inflicted by a loose watchdog.

Two rules:

- **Per-rung deadline = a small multiple of the healthy rung time**, not the lease cap. If a clean
  rung finishes in ~T, set the correctness watchdog to ~3–5×T (a floor of 90–120 s is plenty for a
  fused MoE correctness screen). Use `timeout --kill-after=10s <deadline>` so a rung that ignores
  SIGTERM is SIGKILLed promptly rather than hanging the driver too.
- **Prefer a stdout-silence watchdog when you do not know T.** A deadlock produces **no new output**;
  a slow-but-progressing run keeps printing. Wrap the run so it is killed after W seconds of stdout
  silence (e.g. pipe through a reader that resets a timer on each line, or poll the log's mtime). This
  distinguishes *hung* from *merely slow* without having to predict the healthy runtime, and it
  reacts in ~W seconds instead of at the full cap. Emit `HANG cut=<c> reason=silence` as the rung's
  recorded verdict and continue the ladder.

The point of both is that a hang must be **detected and killed on the timescale of a healthy run**,
so a deadlocked lease self-terminates in seconds-to-minutes and the next rung (or the next lease)
starts immediately — never wait out a lease-length timeout to learn a rung hung.

## A non-monotonic marker across rungs — deterministic target-unreachable vs a race

> **CORRECTION — read before applying the
> `wait_until`-target-unreachable reading below to the MegaMoE substrate.** On the MegaMoE V2 EP8
> fused megakernel, a deterministic fault/hang at the arrival-ticket handshake in
> `mega_moe_stage1.py` was **root-caused NOT to a counter target-unreachable** (an expected-count /
> tile-index mismatch), but to a **latent use-of-uninitialized / miscomputed base-or-predicate
> ADDRESS in the substrate spin-wait loops**, deterministically exposed by the register-allocation
> shift that adding *any* `ready1` publish forces. The tell that falsifies the counter theory: a
> benign constant `0xBEEF` store — no atomic, no computed pointer, no counter at all — faults
> **identically**, with identical instruction/mem-op mix, identical LDS, zero spill. So for a fault
> in `mega_moe_stage1.py`'s owner/producer gate handshake, do **not** stop at "print the counter and
> its expected value" (the probe below). Also diff the emitted ISA of the faulting cut against the
> clean canary to find which base/predicate register the publish perturbs, and reduce substrate SGPR
> pressure so the publish does not perturb the arrival-ticket allocation. The suspects are the
> LDS-aliased ticket scratch (`ticket_scratch = recast_iter(Int64, a_buf.ptr)`, aliasing the GEMM
> staging pool) and the un-pinned gate base/predicate (`gate_addr = a_epoch_gate + grid_epoch_slot*4`,
> `gate_epoch = generation+1`) — where the runtime culprits are `a_epoch_gate` (a VGPR pointer) and
> `generation` (a VGPR), NOT `grid_epoch_slot` (a Python constexpr → constant offset); the fix is to
> `readfirstlane` the gate base and predicate to SGPR in the common path before the owner/non-owner
> split, so both branches consume identical scalars — the base via a **direct 64-bit** readfirstlane
> (`fx.rocdl.readfirstlane(T.i64, a_epoch_gate.ir_value())`, the combine substrate's own idiom at
> `mega_moe_stage2.py:126`), the predicate via the i32 readfirstlane stage1 already uses at 239–240;
> do **not** hand-roll a bitwise 32-bit split on `fx.Int64` (unsupported/unnecessary here). **UPDATE:
> base-pinning is necessary but was FALSIFIED as the cut4 fix** — an engineer
> applied it correctly plus base-pinned every source-visible g2 base, cut3 stayed clean at 128/8192 but
> cut4→cut8 still faulted identically (`0x7e7e…` poison, before combine). The corrupted register is not a
> source-visible base. The **primary** fix is a **resource-extent clamp** of the GEMM2-role tables (bind
> each `create_buffer_resource_from_addr` to its real byte extent, the `num_records_bytes=` kwarg stage2:127
> already uses) so a wild index clamps to 0 instead of faulting — correctness-safe at cut4/cut5 since g2
> output is discarded there. See `knowledge/gfx950_lowering.md` (regalloc can mask *or* expose a
> latent uninit address). If an operator-specific candidate recipe was explicitly supplied to your
> lane, its diagnosed substrate fix supersedes the generic reading below; do not search the skill
> index for an unassigned operator recipe.

A **marker that is non-monotonic across cuts** — a deeper rung prints a progress/path marker that a
shallower rung did not — has two very different causes, and the fix depends on which:

- **Deterministic (the common case).** Different cuts are different binaries, so whether the marker
  line is *reached* depends on where it sits relative to each cut's truncation, not on timing. A rung
  that hangs at a `wait_until(counter)` whose target the producer never reaches will hang the SAME way
  every run. This is a **`wait_until` target that is never reachable** (an expected-count / tile-index
  mismatch between the publisher and the waiter), not a race. Confirm it by re-running ONE hanging cut
  a few times: a stable hang at the same point is deterministic.
- **A genuine race** is the other cause, and it is nondeterministic *within one binary* — the same cut
  hangs on some runs and not others. Establish the failure *rate* on a fixed cut to tell them apart.

For the deterministic target-unreachable case — which is what an intra-rank readiness edge
(GEMM1→GEMM2) almost always is — **do not change the fence scope.** That edge is correctly agent-scope
(`fence_agent_release` + `atomic_add_agent`); system scope there is a cross-L2 per-token regression on
this target, not a fix. The move is the probe in the next section: print the counter, its expected
value, and the work count in the implicated segment, and read the mismatch directly. Only a *proven*
cross-rank visibility gap warrants a scope change, and even then a per-token system-scope atomic is a
known regression — shard the counter instead. Re-run the ladder only to confirm the fix held.

## Verify the instrument before you trust a rung verdict (FlyDSL: two silent no-ops)

The ladder's whole value is that each rung's pass/fail is uncontestable. That guarantee is void if the
rung you *think* you built is not the binary that ran. On FlyDSL this happened for **two consecutive
rounds** on the MegaMoE cut4 fault — both rounds' on-device "results" were measuring nothing, and the
diagnosis thrashed (register-corruption → write-to-RO-page → back to a read-path wild base) purely
because the instrument was broken. Before believing any rung verdict, confirm both:

- **A compile-time guard on a Python constant is NOT a compile-time branch unless it is wrapped in
  `const_expr(...)`.** A bare `if PYCONST:` *reads* like dead-code elimination but actually lowers to a
  device `scf.if`, which **strands the branch's local writes at their use site** and silently changes the
  kernel — so a "toggle" written as a plain `if` on a constant tests two bodies that differ in the wrong
  way, or does nothing at all. Every cut point and every A/B toggle keyed on a constant must be
  `const_expr(int(cut) >= N)` / `const_expr(FLAG)`, never a bare `if`. Re-audit any toggle result you
  inherited: if the guard was a plain `if`, it measured nothing.
- **Confirm the body actually traced and reached the GPU.** A carried/fused body can fail to trace at a
  given cut (e.g. a FlyDSL `NameError` at trace time — a point-of-use pin defined *above* a `while`/`if`
  is not visible inside the generated `__while_after`/`__then` branch closures, because FlyDSL lowers
  those into separate closures). When trace fails, the on-device run either never launched that body or
  ran a stale cached module — and every "it still faults" / "the pin didn't help" conclusion from that
  run is void. Make the harness assert a per-cut trace/compile success marker (and a fresh JIT key)
  before it records a rung verdict; a rung with no trace marker is `INSTRUMENT_VOID`, not a fail.

The tell that you were in this trap: redirecting or removing the suspected faulting op **does not move
the fault**, or a toggle you expected to change behaviour changes nothing. That is not evidence the
suspect is innocent — it is evidence the knob was disconnected. Fix the instrument, then re-bisect.

## Confirm the location before fixing it

The ladder gives a segment, not a cause. Before writing a fix, add a probe inside the implicated
segment that prints the quantities the segment computes, and check them against what they should be.

A worked example of why this step is not optional: a hang was attributed to a stale read past a
barrier, on the theory that the barrier does not invalidate the vector L1. A probe placed in the
implicated segment printed `num_valid=52864` and `total_work=4956`, and `4956 = ceil(52864/128)*12`
— the work count was **7.5× below the clamp**, so the clamp was never the reachable limit and the
theory was false. That probe cost one run. The fix derived from the theory would have cost a wave
and would not have worked.

## When the failure only appears on the second replay

A fused kernel that publishes results between its own stages can pass single-shot and fail under
graph replay. The mechanism is worth recognizing on sight, because the ladder above will show every
rung passing:

the payload buffer is zeroed once at allocation, and graph replay feeds identical inputs, so on
**replay 1** a premature or unordered read returns zeros and adds nothing — correctness passes and
the error metric does not move. On **replay 2** the same read returns the *previous generation's*
partial result. So the acceptance rule is not "it ran"; it is **it ran N times**, and any candidate
that publishes across a stage boundary must be replayed before its correctness number is believed.
See `gfx950_lowering.md` for the ordering bug that produces this on this target.

## Cost, and when not to do this

Fifteen rungs at half a minute each is cheap in wall-clock and free in reasoning risk, and it is
device-time that produces an answer rather than a hypothesis. Do it as soon as a fused candidate
fails, not after two rounds of reading the source.

It is the wrong tool when the failure is **not** reproducible. The ladder assumes rung N's verdict
is stable; against a race that fires one run in ten, a rung that passes means nothing. Establish
reproducibility first (run the untruncated kernel enough times to bound the failure rate), and if
the fault is intermittent, that fact is itself the finding — an intermittent illegal access in a
kernel with counter-based handover is an ordering bug, and the ladder will not localize it.
