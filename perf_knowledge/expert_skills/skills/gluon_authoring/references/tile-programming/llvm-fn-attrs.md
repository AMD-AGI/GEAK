# llvm_fn_attrs — attaching an LLVM function attribute to one kernel

The layer page (`instruction-scheduling.md`) owns *when in a round* you reach for a scheduler
change. This page owns the **mechanism**: what the option actually is, what you may write on it,
how you confirm it took, and what it costs. Read it before your first use and before any sweep.

**It is not a Gluon mechanism, and mis-filing it costs rounds.** `llvm_fn_attrs` is a compile
option of the Triton JIT, handled in `make_llir`, which the plain and Gluon lowerings **share**. It
applies to a `@triton.jit` kernel and a `@gluon.jit` kernel identically, and a meaningful share of
the kernels that carry it are plain ones. Nothing about it requires an explicit-tile DSL, a layout
decision, or anything else the Gluon surface adds.

## What it is — a note on the kernel function, read by the backend

The option carries `name` / `value` pairs. The backend parses them and applies each one to the
generated LLVM function with `addFnAttr`, unchanged. It implements nothing itself; it is a door.
What any given attribute *does* is decided entirely on the LLVM side, by whichever part of the
backend reads that name.

Four properties decide how you use it, and none is obvious from the option name:

- **No allow-list.** Any attribute LLVM accepts on a function is reachable, and one it does not
  recognize **does not raise** — it is dropped in silence. Upstream's own documented example pairs
  a scheduler strategy with `noinline`, so scheduling is one use of this door and not its boundary.
- **It is applied BEFORE the LLVM optimization pipeline runs**, so the whole pipeline sees it. It
  is not decoration on the final module.
- **Your pair overrides what the backend already set**, because the apply loop *removes* the
  attribute name and then adds it. This is the trap with the longest fuse: the loop runs **after**
  the backend has written its own attributes, so a name that collides with one of them wins over
  the launch option that normally owns it. The colliding names to know are
  `amdgpu-waves-per-eu` (normally written from `waves_per_eu=`),
  `amdgpu-flat-work-group-size` (normally derived from `num_warps`),
  `denormal-fp-math-f32` (normally derived from the denormal option), and
  `amdgpu-sched-strategy` itself, which the backend sets to `coexec` for itself on gfx1250 by
  default — and on gfx950 / gfx942 when `TRITON_HIP_USE_COEXEC_SCHEDULER=1` — at `num_warps <= 4`
  (`compiler-contract.md ## What upstream 3.8.0 actually gives you`).
  Setting one of those here is a silent, launch-option-shaped regression that no diff of your
  launch arguments will show.
- **Replacement, not merge, and this bites hardest on `target-features`.** Because the name is
  removed first, a `target-features` pair you pass replaces the function's whole feature string
  rather than appending to it — including a feature the backend added for a sanitizer build.

Two spellings are accepted, and they mean the same thing:

```python
# comma-separated string; a bare name becomes a valueless attribute
llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp,noinline"

# any sequence of (name, value) pairs -- prefer this one
llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]
```

**Prefer the pair form for two reasons.** It survives serialization intact when launch options are
carried across a process boundary for out-of-process compilation, where the string form has to be
re-parsed; and `[]` is then the natural, *total* "leave the backend's choice alone" value, so a
conditional expression has a real else-arm instead of a branch that omits the keyword.

## Is it even a candidate? The ILP-starvation gate

This is a gate, not a preference. Check it before writing the attribute, not after a flat A/B.

**Gate — the kernel must be ILP-starved.** A non-default strategy has something to do only when
the body is low-occupancy (down to one resident wave) *and* carries independent work the default
scheduler leaves unscheduled. That is the case the ILP strategies are named for.

**Anti-gate — a serial dependence chain.** If the hot loop is one chain where each instruction
consumes the previous one's result, there is no reordering freedom to hand the scheduler, and every
strategy is neutral-to-negative. Same verdict when the kernel is already occupancy-rich (2+
resident waves hiding latency) or throughput-bound with a unit saturated, and the default is
usually the right choice for LDS-bound / memory-clause-friendly loops. **Never default it on.**

**Order gate — an overlap structure has to exist first.** This layer paces a stream; it does not
create one. Applied to a loop with nothing to interleave, the result reads as noise, and that flat
reading is correct rather than a missed win (`pipeline.md`).

**Version gate — apply the ILP rule only once the option is known live on your build**, or you are
reading a version fact as a kernel fact (`## Which versions have it`).

Two pages settle the gate rather than this one. The bound class the first two gates are stated in
comes from `../hardware/bound-class-signals.md` — read the bound before you read this page, since
"ILP-starved" is a reading of that, not a guess from the source. The support and version cell is
`../hardware/capability-matrix.md`, which is the authority for what this build has.

### And the adoption rule that surprises people: this is a per-kernel patch

There is **no scenario attribution** for this attribute, and inventing one is the most expensive
mistake available here. Kernels with the same archetype, the same tile shape and the same author
routinely disagree: one variant carries it, its structural sibling does not; one launch in a
dispatcher gets one strategy and the next launch in the same function gets another; two variants of
the same family select on predicates that contradict each other at a shared shape. That is a real
property, not sloppiness — the attribute is per compile, which is exactly the granularity a
process-wide environment variable cannot express.

So: **do not generalize it by workload, archetype or dtype.** "GEMMs want this" and "reductions
want that" are not available claims. Verify per kernel, per launch, and re-verify after any change
to the tile shape or the loop body. If you are tempted to write a rule of the form "kernels of kind
X should set this", you have left what the mechanism supports.

The corollary is practical: **in a multi-kernel pipeline, decide per stage.** A producer and its
epilogue are separate compiles and take separate settings, including "one gets an attribute and the
other gets `[]`".

## What you can write on it

The attribute classes below are the ones with a stated purpose on this target. They are not a
closed list — the door has no allow-list — but anything outside them is yours to justify and yours
to verify.

| class | attribute | what it changes | reach for it when |
| --- | --- | --- | --- |
| ILP strategy selector | `amdgpu-sched-strategy=<s>` | which machine-scheduling strategy the backend runs, hence instruction order | the ILP gate above passes |
| co-execution strategy | `amdgpu-sched-strategy=coexec` | the stock matrix/VALU co-execution strategy, per compile and without the env route's `num_warps <= 4` gate | a matrix-plus-VALU region (attention-shaped); gate and acceptance in `compiler-contract.md ## What upstream 3.8.0 actually gives you`, not the ILP gate |
| IEEE-mode control | `amdgpu-ieee=false` | min/max NaN semantics, which is what blocks a native DPP reduction form | the kernel's job is a max/min **selection** and you can bound the numerics |
| target features | `target-features=-<feature>` | turns a subtarget feature off for this function (e.g. suppressing packed-FP32 forming) | you need a specific instruction form *not* to be formed |
| register budget | `amdgpu-num-vgpr=<n>` | caps the allocator's per-thread VGPR budget, hence occupancy | you are pinning occupancy by hand and have a number you can defend |
| accumulator placement | `amdgpu-agpr-alloc=0,0` | keeps accumulators out of AGPR | the wave-ping-pong model, which wants them in VGPR (`scheduling-model.md`) |
| generic LLVM attributes | e.g. `noinline` | whatever LLVM does with that attribute | rarely; listed so you know the door is this wide |

### Scheduler strategy — be honest about which names are established

LLVM is not vendored in the Triton tree (it is a prebuilt pinned by hash), so the accepted strategy
enum **cannot be read from the source you have**. Evidenced inside Triton itself: `iterative-ilp`
(upstream's documented example and its test) and `coexec` (which the backend sets for itself on
gfx1250). The other names that circulate — `max-ilp`, `iterative-minreg`, `max-memory-clause`,
`iterative-occupancy`, `max-occupancy`, `iterative-maxocc` — have **zero occurrences** in the tree.

They are not all equal, though, and the distinction changes how you read a flat sweep. The first
four of those are names that shipped kernels do set, so a sweep over them is sweeping values
somebody used rather than names somebody invented; `max-occupancy` and `iterative-maxocc` are
names that only circulate. Neither group is upgraded to source-proven by that, because **an ignored
value and an honored one are indistinguishable from Python** — which is why the acceptance signal
is the assembly diff and not the timing.

**A valid strategy is not guaranteed to compile on every lowering, and the failure is at compile
time.** The reported shape of this is a strategy that compiles on contiguous, aligned operand
signatures and fails on mixed-stride or unaligned ones — so the arm of a dispatcher that serves
ragged or generic inputs can stop building because of an attribute added for the fast path. The
rule that follows: **put the attribute inside the branch that proved its operand signature, never
above it**, and leave the generic arm on the default scheduler. Status: reported behaviour, not
reproduced in this pack, and not pinned to a toolchain version — so treat a compile failure right
after adding a strategy as this, and narrow the predicate rather than debugging the kernel.

### IEEE mode — a numerics change, filed as one

`amdgpu-ieee=false` relaxes the IEEE exception mode so the backend may lower `max` / `min` to a
native DPP reduction form. The stated intent where it is used is exactly that and nothing more:
enable the native reduction, **without** disturbing which index a stable-ID selection returns, and
with **denormal handling explicitly unchanged**.

Three consequences:

- **It has a precondition, and it is not "this kernel is slow".** It belongs on a kernel whose
  actual job is a max/min selection (top-k, argmax, a routing selector). On anything else it is a
  numerics change with no mechanism behind it.
- **It is a floating-point semantics change, so manage it like one** — in the same register as
  `enable_fp_fusion` and rounding-mode choices, not as a performance knob. Write the numerics scope
  down next to the launch: what is preserved (finite / infinity / quiet-NaN behaviour, denormals,
  the selected index) and what is not.
- **Scope it to the selecting kernel.** It is a per-compile attribute; do not let it ride along on
  the surrounding pipeline's kernels just because they are in the same file.

## A worked example — one dispatcher, two kernels, two answers

```python
# Both launches are in the same host function. The attribute is a property of THIS
# compile, not of the file, the archetype, or the shape family.

# The strategy goes INSIDE the branch that proved the operand signature. `[]` is the
# total else-arm: "leave the backend's own choice alone".
stage_sched = (
    [["amdgpu-sched-strategy", "iterative-ilp"]]
    if x.stride(-1) == 1 and w.stride(-1) == 1
    else []
)

stage_operands[(cdiv(M, BLOCK_M),)](
    x, w, acc,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=4,              # pin occupancy FIRST -- this attribute is not a substitute
    llvm_fn_attrs=stage_sched,
)

# Different kernel, different compile, different answer. This one is a selection, which
# is the precondition for the IEEE-mode pair; the strategy choice here is its own A/B.
reduce_rows[(rows,)](
    acc, out_idx,
    BLOCK_M=BLOCK_M,
    num_warps=1,
    llvm_fn_attrs=[
        # selection only: enables the native DPP max form. Selected index unchanged,
        # denormal handling unchanged, non-finite behaviour unchanged.
        ["amdgpu-ieee", "false"],
        ["amdgpu-sched-strategy", "max-ilp"],
    ],
)
```

**The failures this shape prevents, in order of how much time they cost:**

- Hoisting `stage_sched` above the stride test and applying it unconditionally: the ragged arm
  **stops compiling**, and the error does not mention the attribute you added.
- Writing `llvm_fn_attrs=[["amdgpu-waves-per-eu", "4"]]` next to `waves_per_eu=2`: the attribute
  loop runs last and your pair wins, so the launch option you thought you set is gone.
- Copying the `reduce_rows` pair onto `stage_operands` because "the file uses it": a floating-point
  semantics relaxation now applies to a kernel that does no selection, and nothing reports it.
- Typing `amdgpu-sched-stratgy`: everything compiles, the assembly is byte-identical, the A/B is
  flat, and the flat result gets recorded as "this kernel is not ILP-starved".

## Verifying it took effect — the assembly diff is the only signal

There are **two different failure modes** here and they look nothing alike:

| what is wrong | what happens |
| --- | --- |
| the option name `llvm_fn_attrs` itself (absent on 3.6.0 / 3.7.0 / 3.7.1) | **raises** — an unrecognized compile option |
| the attribute name or value you passed, when LLVM does not recognize it | **silently ignored** — indistinguishable from "applied and made no difference" |

The second row is the entire reason the acceptance procedure is what it is. A Python-side check
that the option was accepted proves nothing; neither does a timing A/B, because "ignored" and
"honored but neutral" produce the same number.

**The procedure:** compile the kernel twice, identical in every respect except the one attribute,
and diff the generated assembly.

- **No diff ⇒ the attribute did not take.** Suspect the spelling first, the value second, the
  version third. Do not proceed to a timing A/B, and do not record a bound-class conclusion.
- **A diff ⇒ the attribute took**, and only now is a timing A/B measuring the thing you think it
  is. Read the hot-loop instruction mix as well as the timing; a scheduling win visible in timing
  but not in the instruction mix has not been attributed to the scheduler yet
  (`instruction-scheduling.md ## Acceptance`).
- **Diff the whole function, not just the loop**, for the register and target-feature classes —
  a VGPR cap or a suppressed instruction form shows up in the prologue and in the register counts
  the assembler reports, not necessarily in the loop body.

## What it costs

- **Nothing at build time.** No plugin, no rebuild, no environment variable, no monkeypatch, no
  sanction. That is why it precedes an authored LLVM pass in every ordering on this layer.
- **One compile per value swept**, and each distinct value is a distinct compile-cache entry.
- **A durability cost, and it is the real one.** A scheduling result is pinned to one toolchain:
  the same source on a different compiler build can schedule differently, with nothing warning you
  — the attribute still applies, it just means something else. Record the toolchain identity next
  to any result you keep, and re-run the assembly diff after a toolchain change rather than
  assuming the decision carried.
- **A maintenance cost proportional to the guard.** The tighter the predicate you need in order to
  apply the attribute safely (an exact stride or dtype test, a shape equality), the more shape
  drift silently routes around it. A guard as long as the launch itself is a defensible outcome for
  a hard register pin; it is a poor trade for a speculative strategy sweep.
- **A numerics cost for exactly one class**: `amdgpu-ieee`. Everything else on this page changes
  instruction selection, order or allocation and leaves arithmetic alone.

## Which versions have it

**3.8.0 only.** On 3.6.0, 3.7.0 and 3.7.1 the option is **not declared** on the AMD backend, so
passing it **raises** an unrecognized-compile-option error. It does not degrade to a no-op, and
there is no downgrade of the attribute itself — the mechanism is simply absent.

**What that means for a reader on an older build.** Do not look for a replacement spelling for the
attribute; look for a replacement for whatever you wanted it *for*.

- Wanted a scheduler change: `schedule_hint` presets are the downgrade rung on 3.6.0 / 3.7.0 /
  3.7.1 (a dead declaration on 3.8.0); the version table and why the two are not replacements in
  kind are in `compiler-contract.md ## Portable scheduler co-design (the env knobs, and where the
  real lever lives)`. The `coexec` strategy has no downgrade either — its env knob is 3.8.0-only.
- Wanted more ILP without a scheduler knob: author the overlap and pace it at the source level —
  independent work staggered by hand in the pipeline (`pipeline.md`), and the inline-asm mechanisms
  on `instruction-scheduling.md`, which are available on all four builds.

**Probe, do not infer.** `scripts/probe_levers.py --all` reports `version_disjoint_knobs` and
separates `live` from `absent` and from `dead-declaration`. A flat sweep on 3.7.1 means "this build
has no such knob", never "this kernel is not ILP-starved".
