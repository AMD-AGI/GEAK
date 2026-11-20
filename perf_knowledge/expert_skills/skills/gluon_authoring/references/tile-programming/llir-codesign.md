# LLIR schedule co-design (layer 4, below the DSL)

Read this when the residual is **instruction interleave inside the hot loop** — the structure is
right, the overlap exists, and the matrix pipe still idles between MFMAs. It covers the two
scheduling models that live at the LLVM-IR level, which one a given loop region gets, and what the
kernel has to guarantee for either to work.

It does **not** replace the pipeline layer. Overlap has to exist before anything can be
interleaved: `pipeline.md` first, this second. And it is not a knob list — the models are
arithmetic, so most of the work here is deciding *before* measuring whether any schedule can win.

Constants live one layer down (`../hardware/isa-mechanisms.md ## MFMA throughput model (planning cycles)`,
`## Matrix/VALU co-execution`). This page carries the formulas and the decisions; it deliberately
names no cycle counts, because every one of them is per shape and per target.

## Route the loop first: who built the overlap

The first question is not which model to use — it is **who authored the overlap**, because that
decides which models are even reachable.

- **The kernel author built it** — the climb default on the Gluon path (register prefetch, an
  authored LDS ring, then stage/cluster markers:
  `pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`,
  `scheduling-model.md ## inter_wave — realize with warp_pipeline_stage`). Those markers survive
  lowering as scheduler-barrier boundaries, so the LLVM layer receives the author's own stage
  structure and both models are reachable, chosen per region.
- **The compiler built it** (plain's auto-pipeliner, or — the lowest rung on the Gluon path, a
  below-parity diagnostic or last resort whose numbers are labelled "injected" and which is never
  applied to an incumbent Gluon kernel — plain's pipeliner re-injected into an explicit loop:
  `pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`).
  The loop carries no stage markers, so nothing has told the LLVM layer where the stage boundaries
  are. Such a loop can only get the **throughput** model, whose regions are *inferred* from
  instruction order.

These two are **mutually exclusive claims of ownership**, not options to stack. Re-injection's
premise is "remove the hand-written staging so the pass can build it"; the co-execution model's
premise is "the author owns the cluster structure and the compiler only fills windows". A loop
cannot be both.

**Read that as scoped to re-injection, not to instruction scheduling in general.** What cannot
coexist is *re-injected* staging and *authored* regions, because they disagree about who writes
the loop body. An **authored** ring and instruction scheduling on top of it agree about that, and
production layers them freely: on gfx950 the two commonly appear together, and the hand-rolled
wave-priority form **usually** appears on top of a ring — usually, not without exception; some
surveyed files drain a single slot with `wait_group(0)` and carry no ring at all. The exclusivity
is per loop and per owner, not per layer.

**Re-injection has a second, easier-to-miss consequence.** It reproduces *plain's* loop shape,
which is not the shape the throughput model wants: the inferred-region rule keys on memory ops
**preceding** the MFMAs that open a new region, and a pipeliner-built body puts its
LDS read at the top and its global load + LDS write at the bottom. So before pairing re-injection
with the throughput model, **read the IR and confirm the regions actually formed** — a
region count of one, or a body where every MFMA lands in the same region, means the model has
nothing to interleave against. TFLOPS cannot tell you this; the dumped IR can
(`scripts/dump_ir.sh`).

> **Probe, do not assume:** the re-injected pipeliner and the stage-marker pass sit in a fixed
> order in the pass list, but whether marker regions survive a loop the pipeline expander has
> rewritten is **not measured**. If you need both, verify the markers still produce boundaries in
> the lowered code before budgeting anything on them.

## Region routing: the model is a property of the region

A scheduling region is routed by **what it contains**, not by which kernel it came from. The same
pass therefore serves a GEMM and an attention kernel without either knowing about the other.

```text
region contains          ->  model
-----------------------      -----
mfma + memory, no valu   ->  THROUGHPUT   (pair each memory op with enough MFMA to cover it)
mfma + valu, no memory   ->  CO-EXECUTION (assign each vector op to a specific MFMA's window)
mfma + valu + memory     ->  neither: skip
no mfma                  ->  nothing to schedule around (phase pacing only)
```

The three-category region is skipped rather than guessed at, because the two models disagree about
what an MFMA is *for* — covering memory latency versus covering VALU issue. On a target whose
issue rules make VALU and memory in the same wave wasteful, a well-built kernel has no reason to
create such a region in the first place (`## Attention: the co-execution budget`).

**Only one of those rows is an argument for being on this page at all, and knowing which saves the
whole round.** The two models differ in what the *hand-authored* path can already reach:

- **Throughput regions (mfma + memory) are fully expressible by hand.** The two instruction
  classes come from two different waves, so the overlap is a structural property that staging and
  a wave-level phase offset already deliver. A hand-built two-stage GEMM on this target reaches
  near-full per-SIMD matrix efficiency with no pass, no plugin and no env var — so for this row
  the honest prior is that there is nothing here to add, and a flat result is the expected one
  rather than a tuning failure.
- **Co-execution regions (mfma + valu) are not.** The vector op has to issue from the *same* wave
  as the matrix op, inside the window of a *particular* matrix instruction, in a *particular*
  form — and the source language can say which stages exist and what each issues, but not which
  MFMA's shadow a given vector op lands in, nor forbid the packed form that cannot co-issue at
  all. That placement is what `## Declaring the interleave` exists to express, and it is the one
  capability that genuinely has no hand-authored spelling.

So route by region before budgeting: reach for this page for a co-execution region, and read a
throughput region's residual as a layout or staging question first
(`## Route the loop first: who built the overlap`).

Both models share one judgement, and it is worth stating once:

```text
per region:  capacity = (number of MFMA in the region) x window_for_that_shape
             demand   = sum over non-matrix ops of their class issue cost
             exposed  = max(0, demand - capacity)
```

`window_for_that_shape` and the class costs are the L3 facts
(`../hardware/isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget`). **Derive the window from the MFMA
the region actually contains** — carrying one shape's window to another over-fills every window by
the ratio of their intervals, and one target's window is not another's minus a constant.

## GEMM: the throughput model

Here the non-matrix work is **memory**, and the MFMAs exist to cover its issue occupancy. The
model is a throughput pairing, not a dependency analysis.

**Gate.** A pure matrix-accumulator chain (MFMA to MFMA, only copies between); independent
memory/local-read/dot work inside one iteration; a prefetched multi-buffer body; register budget
inside the joint file; the MFMA shape covered by the tool's cost model. Miss any of these and the
region is skipped, not merely unhelped.

**Authoring.** The pairing counts are ratios of issue occupancies, so they come from L3 rather
than from tuning:

```text
# global load: cover its issue occupancy with MFMA
mfma_per_global_load = ceil(global_load_issue_cyc / mfma_cycles)      # both from ../hardware/

# LDS: reads and writes share ONE issue port, so carry one running balance
#      across every LDS access in the region, and emit whole MFMAs from it
balance    += access_bits / bits_per_cycle
emit_here   = floor(balance / mfma_cycles);  balance -= emit_here * mfma_cycles
```

Two structural facts make this cheap and safe, and both are worth knowing because they tell you
when it will *not* apply:

- **The region rule carries its own dependency invariant.** A region opens at each MFMA that
  follows a memory op, so every load feeding an MFMA necessarily sits in an *earlier* region. Any
  reordering *within* a region is therefore dependency-safe without consulting a def-use graph —
  which is also why the loop shape matters so much (`## Route the loop first`).
- **The schedule is pinned, not enforced by disabling the scheduler.** A full reorder barrier after
  each memory anchor keeps the backend from re-clustering the interleave, while leaving the machine
  scheduler enabled for the prologue, the epilogue, and any region the pass skipped. The barrier is
  `llvm.amdgcn.sched.barrier(i32 0)` — mask `0` means *nothing* may cross during scheduling, which
  is what makes it a wall rather than a preference. If you are reading a tool that disables
  `misched` / `post-misched` globally instead, that is the older design and it gives up scheduling
  everywhere else.

**Acceptance.** MFMA interleaved with the memory ops rather than clustered, the anchor barriers
present, and the iteration-end accumulator-copy block no worse
(`## Acceptance signals`).

## Attention: the co-execution budget

Here the non-matrix work is **vector math** between two matrix phases, and it is the *MFMA's
shadow* that is the scarce resource. This is a materially harder problem than the GEMM case: every
vector op has to be assigned to a specific MFMA's window, in the right form, or it falls outside
and adds directly to the loop's cycle count.

**Gate, in order — and the first two are cheap enough that skipping them is never justified:**

1. **Is the overlap structural?** The vector work must share a wave with the MFMA while the memory
   work runs in the *other* wave, which requires two resident waves per SIMD. Below that there is
   one instruction stream and the co-execution premise is gone — fall back to the
   compiler-interleave model (`scheduling-model.md`).
2. **Is the bound class actually matrix issue?** A latency- or bandwidth-bound dispatch does not
   have a co-execution problem, and this budget is not its binding axis.
3. **Does the work fit?** Compute `capacity` and `demand` per region on paper. `demand > capacity`
   means **no ordering wins** — the work has to shrink or move. This is a budget statement, not a
   scheduling one, and it costs ten minutes.

**Authoring — balance the regions that share a budget.** Only the per-region totals matter, so any
work made of the same elementwise pieces can be moved between regions to level them. Two rules
make that affordable and safe:

```python
# The move must be FREE. A register-only view of a distributed tensor keeps the source
# layout, so it is a partition of each lane's own registers and emits no instructions --
# and the compiler is made to PROVE that at build time rather than being trusted.
# The register-only slice is `ttgl.amd.slice` (`slicing.md ## Slice recipe (ttgl.amd.slice)`);
# a slice that does NOT preserve the layout costs a shuffle, which eats the imbalance you
# were recovering, so the layout-preservation is the part to verify, not assume.
lo = ttgl.amd.slice(tile, [M, N // 2], [0, 0])          # offsets into the SAME layout
hi = ttgl.amd.slice(tile, [M, N // 2], [0, N // 2])

# Move a producer TOGETHER WITH its consumer. Splitting them leaves the producer with its
# only consumer in another region, and a scheduler that can see a consumer downstream drags
# the producer toward it -- a pure elementwise op carries no chain edge, so no barrier holds
# it in place. Carry the RAW value across and compute both where it is consumed.
```

Slice granularity is what makes the ratio reachable: cutting a tile into halves gives one coarse
choice, into eighths gives seven. **The balance ratio is a per-(head-dim, block-N, dtype) quantity
and does not transfer across shapes** — capacity and demand both move with the tile, at different
rates (`../workloads/attention.md ## Applying the framework to attention variants`).

**Which work is movable at all is decided by the dependency chain, not by size.** Before looking
for the ratio, find the cut points, and there are usually very few:

- **A reduction cannot move, and neither can anything upstream of it.** A row max is not complete
  until the whole tile has been reduced, so it pins itself and the subtract that consumes it.
- **A cut is only available where the chain can be severed with one value crossing.** Look at each
  operation's consumers: if they all sit on the far side of a candidate cut, that cut works. If
  they straddle it, the value has to be materialized on both sides and the "free" move is not
  free.

Run that over a streaming-softmax chain and exactly one natural cut survives — after the
exponential, whose two consumers (the row sum and the downcast) are both downstream. That is why
the split lands where it does rather than somewhere more convenient. The tunable part is *not*
where the cut is; it is how many of the tile's independent column slices are cut there versus
carried across raw, which is what makes a discrete chain yield a continuous-looking ratio.

**Treat the budget as an estimate with error bars, not as exact arithmetic.** The per-class costs
are a model of issue behaviour and the instruction inventory is the one you counted at the DSL
level — the backend then adds address arithmetic the table never priced. So a configuration that
clears the window by a couple of instructions has not actually cleared it: that margin is inside
the model's own error. When sweeping the ratio, prefer the setting that leaves **visible** room in
both regions over the one that technically fits in both, and expect the tightest-fitting
configuration to measure worse than the arithmetic predicts.

**Authoring — reduce the demand, then make sure something spends the headroom.** A change that
removes vector work from a region (skipping a conditional rescale, folding a scale onto an operand
loaded once upstream) does not make the loop faster by itself. It **frees budget**, and only
something downstream that spends that budget converts it into cycles. Attribute the two separately
or the enabling change reads as a null result and gets reverted
(`../method/close.md ## Attributing a change that only creates headroom`).

> **Before treating this region class as plugin-only, check the stock 3.8.0 `coexec` scheduler
> strategy** — default-on on gfx1250 only; **opt-in on gfx950 / gfx942** via
> `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (honoured at `num_warps <= 4`, so not for a
> `warp_pipeline_stage` kernel) or per compile via `llvm_fn_attrs`. The gate, the overrides and
> the acceptance check are stated once in
> `compiler-contract.md ## What upstream 3.8.0 actually gives you`. What it does not give you is
> *placement* — it will not put a chosen vector op in a chosen MFMA's shadow, which is what
> `## Declaring the interleave` is for — but it is the stock answer for the region class and it
> costs nothing to check.

**Acceptance.** Per-region window fill, not throughput. See `## Acceptance signals`.

## Declaring the interleave

Once the assignment is decided, there are two ways to get it into the generated code, and they are
not equally strong.

- **Reorder and pin.** Physically move the instructions and emit reorder barriers. A full-mask
  barrier is a hard wall, but a bare "do not cross" marker is **advisory**: codegen has been
  measured still consolidating the last sub-regions of a stage — an MFMA migrating toward the
  region front, leaving a tail of vector ops past the final MFMA with no window over them — even
  when the pass's own IR had every group inside its window.
- **Declare.** Emit class-and-count scheduling-group requests — `sched_group_barrier` — and let
  the backend's group-pipeline builder construct the schedule. On AMDGPU that builder is
  **IGroupLP**, which runs inside the machine scheduler and forms the requested groups. This is
  the stronger form: it builds the requested pipeline rather than merely forbidding motion. The
  Gluon surface does not expose `sched_group_barrier` (masks, `count` discipline and what is
  reachable instead: `instruction-scheduling.md ## The declarative hints that are not there`), so
  a declaration is emitted one layer down, from a pass.

Three disciplines make a declaration work, and each has a silent failure mode:

1. **Declare in instructions, not elements.** Group sizes are instruction counts. A window holding
   three packed ops must be declared as three, not as the six elements they become. Declared in
   instructions the request is satisfiable whether or not the ops are still packed, and the
   late backend peephole splits whatever ended up in a shadow.
2. **Emit the declaration after every real instruction of the region.** IGroupLP forms its groups
   scanning *upward*, so a declaration at the top of a region yields empty groups and **silently
   does nothing**.
3. **Never declare an instruction that will not be emitted.** If a group cannot be filled, the
   solver abandons the pipeline for the **whole region** and leaves the instruction-selection
   order — which reads exactly like "the scheduler did not try". The usual culprits are ops that
   vanish before issue: values folded into a consumer as a source modifier, and reduction steps the
   backend fuses pairwise. Count what will be *issued*, not what the IR contains
   (`../method/entry.md`, diagnose).

## Two preconditions the kernel must protect

Both of these are outside the scheduler, and both invalidate its work silently.

- **Placement.** The late machine-level pass that does this is **`MachineSink`**, which runs on MIR
  long after any IR pass and moves an op toward its consumer. When one region's producer feeds the
  next region's consumer — the normal case in a software-pipelined loop — "toward its consumer"
  means *out of the region it was assigned to*. The symptom is a carefully balanced region arriving
  at the scheduler with its work somewhere else and most of its shadow empty. The switch is
  `DISABLE_LLVM_OPT=disable-machine-sink`. Preserving the placement is not a scheduling decision
  being overridden; it is the input the scheduling needs.
- **Form.** Start from packed math and let the late peephole split only what landed in a shadow.
  That peephole is **`SIPreEmitPeephole`**, which finds a packed op sitting in an MFMA's shadow,
  recognises that it cannot co-execute there, and breaks it into scalars — the correct local
  decision, made at the one point where the answer is known, after scheduling has settled what
  sits where. A blanket "scalarize every packed op in a block containing an MFMA" also splits the
  ops that were deliberately left packed *because* nothing was going to hide them, and one packed
  op retires two elements where two scalars need two issues
  (`../hardware/isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign (CDNA3/4)`).

## Phase pacing: moving a burst instead of removing work

Issuing in the same cycle is necessary for two instructions to overlap, not sufficient — they can
still collide in the register file. Where a region's 3-source vector work coincides with the other
wave's LDS-return burst, the pairing the budget assumed does not materialize, and no reassignment
inside the region fixes it because the conflict is *between* waves.

Two independent levers, from opposite ends:

- **Move the burst.** A small fixed delay at the head of the memory stage shifts that wave's reads
  later, so they arrive past the region's 3-source block and land on cheaper work instead. The
  kernel is unchanged; only the phase relationship between the two waves moves.
- **Remove the third source.** Folding a scale into an operand loaded once upstream turns a
  3-source fused op into a 2-source one, so those stop competing for read ports at all. Verify by
  counting 3-source vector ops in the loop body before and after.

Pacing is a **phase** effect, not a quantity: the sweep is not smooth and bisection will mislead
you. Sweep the small integer range exhaustively and keep the winner per kernel.

### Before treating an `s_nop` as either one, read the instruction after it

An `s_nop` in the loop has three origins, and only two of them are levers. Besides the compiler's
hazard padding and the pacing delay above, there is the **ISA-mandated wait state**: DPP, `permlane`
and the other cross-lane movers require a fixed gap after the VALU write they consume, and a kernel
that hand-writes them in inline asm writes that gap itself. It looks identical in the listing — a
bare `s_nop 1` inside the hot loop — so the discriminator is the *next* instruction, not the count
or the placement: a cross-lane or DPP-modified op immediately after means the nop is correctness,
and removing it produces wrong results rather than a faster loop. Sweep the pacing delays; leave
these alone. The required gap per hazard class is in `../hardware/isa-mechanisms.md`.

## Layout and pipeline are one decision

A pipeline that fires perfectly can still lose on the layout it was forced to fall back to. The
recorded case (an "injected" arm — re-injection is the lowest rung, see `pipeline.md`): a
re-injected pipeliner built its staging on a low-vectorisation, unswizzled shared
layout where the plain comparator got a rotating shared layout the DSL cannot express — the pass
was demonstrably firing and the arm was a large regression
(`../gluon/pipeline-reference.md ## Re-injecting plain's pipeliner — the measured recipe`). Read a pipeline regression as
a layout question before reading it as a scheduling one.

**Axes that must move together.** Changing one without recomputing the others is the common way a
correct pipeline becomes a slow or wrong one:

- **Structural:** buffer count ↔ the retire count each iteration waits on ↔ shared-memory capacity
  ↔ the shared layout (padding costs capacity, swizzle does not). The dot-operand K width is not a
  free knob either — it is tied to the matrix shape by the K-dim identity
  (`layout-recipes.md ## Layout families`).
- **Co-execution-specific:** the warp arrangement simultaneously decides whether a row-wise
  reduction stays inside a wave, how many MFMAs each compute region contains (= capacity), and the
  accumulator's per-lane register count. So a tile-shape change moves **capacity and demand at
  different rates**, and the region balance has to be recomputed rather than carried.

**Verification ladder**, cheapest first — the first four already exist, the last is what this page
adds:

1. Offline, before compiling: render the layout and its bank pattern
   (`layout-recipes.md ## Layout self-check tool (layout_plot)`).
2. At build time: make the compiler prove a register-only slice is free (`assert_trivial`).
3. Layout equivalence against the comparator's own inferred layouts
   (`../method/transcribe.md`, `ttgir_bridge verify`).
4. Conflict-free local-read interval from the static audit (`scripts/asm_loop_audit.py --arch gfx950`).
5. **Per-region window fill** — did the budget actually land (`## Acceptance signals`).

One trap worth repeating in this context: **matching assembly is not correctness.** A recovered
store layout is specific to the matrix warp arrangement; reusing it under a different warp split
gives numerically wrong output while the assembly still shows the ideal shuffle
(`layout-recipes.md ## Epilogue store convert: permlane vs LDS (fidelity, not speed)`).

## The plugin tier

Between "hand the compiler team a report" and "rebuild the toolchain" there is a third tier: an
out-of-tree pass plugin loaded into the existing compiler at an end-of-pipeline extension point.
It changes no compiler source and rebuilds no LLVM. It is **not** free of a build, and the
difference matters because two of the three states fail quietly.

Concretely, on the Triton-on-AMD stack this is a new-PassManager LLVM plugin: it registers through
`llvmGetPassPluginInfo` and inserts itself at the **`OptimizerLast`** extension point, and it is
loaded with `LLVM_PASS_PLUGIN_PATH`. The plugin does not link LLVM — it resolves LLVM symbols
from the host at load time, which is why visibility and symbol scope decide whether it works.

- **Stock wheel:** the plugin cannot load at all. A hidden-visibility build exports no symbols for
  it to bind against, and loading fails with an undefined symbol. This is the **loud** state.
  Default visibility is the `TRITON_EXT_ENABLED` CMake option, and it **defaults OFF**.
- **Self-built with default visibility, on upstream's plugin wiring:** the plugin loads, and the
  optimizer then runs its full pipeline **with no target machine** — not as a defect but *by
  design* (see the note below). Codegen changes accordingly, so a plugin-on run is not a clean
  A/B against plugin-off.
- **A host patched to keep the target machine for plugins** (a fork/host patch — its request
  variable `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE` is **fork-only; absent in upstream 3.8.0, inert
  if set**, `non-upstream-reserve.md ## 1. The LLIR-scheduler plugin`): works, and is the only state in
  which plugin-on and plugin-off are the same optimization environment. Whether a host does this
  is a **source-level property of the host build**, not something an environment variable can
  supply — a build that lacks it will accept a variable that appears to request it and still run
  untargeted. Probe it in the generated code; never infer it from having set a flag.

So the accurate claim is **"no LLVM rebuild, but a default-visibility host build"**, not "no
rebuild".

> **`LLVM_PASS_PLUGIN_PATH` itself is upstream, and it reaches the Gluon path.** It is read in
> `optimize_module`, which `make_llir` calls — and `make_llir` is a stage both the plain and Gluon
> lowerings share. So the plugin door is not a vendor addition; what a vendor build adds is a host
> built to let anything bind to it.
>
> **And there is a trap in how it is wired that changes what an A/B means.** Stock builds the
> target machine only when no plugin is set — `if (!arch.empty() && pluginFile.empty())` — so
> setting a plugin path deliberately leaves the optimizer **without a target machine**, by design,
> to avoid mismatching plugin-inserted precompiled code. The consequence for measurement is the
> part to carry: **plugin-on and plugin-off are not the same optimization environment**, so a
> plugin-on regression is not attributable to your pass until you have separated the two effects.
> The third state above (a host that keeps the target machine for plugins) is what makes the
> comparison clean; without it you are measuring your pass and an untargeted pipeline together.

> **The MLIR-pass plugin tier is a different door with a harder gate.** `TRITON_PLUGIN_PATHS` loads
> TTIR/TTGIR passes and exposes each one as a Python symbol automatically, which is a much better
> fit for a scheduling policy expressed at tile level. But it is compiled out unless Triton was
> built with `TRITON_EXT_ENABLED=1`, and **that option defaults OFF**. The good news is that this
> gate is the one gate in this area that does *not* fail quietly: an install without it prints a
> multi-line warning naming the extension it is skipping. That makes it a reliable probe — set the
> variable to a dummy path and read stderr.

Two runtime constraints on top:

- **ABI lock.** The plugin binary is locked to the exact compiler revision the host was built
  against. A mismatch is a crash, not a graceful degradation, so record the revision beside the
  binary and re-check it after any toolchain re-pin. The pin is `cmake/llvm-hash.txt`; builds
  that share it can share one plugin binary, and the first build that moves it **segfaults**
  against the previous binary rather than warning.
- **Symbol scope, and why it hides.** Python loads extension modules into a local symbol scope by
  default, so the plugin fails to resolve host symbols — and this only surfaces on a **cold**
  compile. A warm cache makes a broken configuration look like a working one. Prove the plugin ran
  before attributing anything to it.

**The four gates, in the order they have to be checked.** Each one produces a *different* wrong
conclusion if skipped, and only the second is loud:

| # | gate | how it fails |
| --- | --- | --- |
| 1 | the host build keeps its target machine for plugins | quiet — plugin appears live, **codegen changes for a reason that is not your pass**, and the arm reads as "the schedule made it slower". Upstream does **not** keep it (above), so on a stock host this gate is failed by default and the A/B has to account for it |
| 2 | the host exports compiler symbols (default visibility, `TRITON_EXT_ENABLED`) | loud — load fails with an undefined symbol |
| 3 | the plugin binary's ABI target == the host's compiler revision | crash / segfault |
| 4 | the loop's matrix shape is in your pass's cost model | quiet — the region is dropped without a word, and it reads **exactly** like a hardware ceiling (`## Applicability gate: shapes the tool does not model`) |

Gate 1 is the one most often mis-diagnosed, because a host's behaviour and a variable that
appears to request it are two different facts. Gate 4 is the one most often mistaken for a result. Record
which of the four you actually checked; "the plugin did nothing" is not a finding until all four
are green (`compiler-contract.md ## Toolchain identity`).

This section is the **single statement** of the plugin build states, the ABI lock, the
symbol-scope trap and the upstream wiring of `LLVM_PASS_PLUGIN_PATH` / `TRITON_PLUGIN_PATHS`; the
handbook, `compiler-contract.md` and `non-upstream-reserve.md` point here. The pass skeleton, the
build invocation, and the probe commands for the three states are in
`llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton`. Authoring the *policy* inside such
a pass remains gated by `compiler-contract.md ## Scenario B: sanctioned compiler co-design`.

## Applicability gate: shapes the tool does not model

A schedule tool carries a per-shape cost model, and **a shape missing from it is not a slow path —
the region is skipped entirely.** This is the single most common reason an interleave tool
"does nothing" on a target it was not developed against, and it reads identically to a hardware
ceiling.

Check this before concluding anything:

```bash
# 1. Which matrix shapes does this loop actually emit?
#    (dump the lowered code and read the mfma mnemonics)
# 2. Are those shapes in the tool's cost model?
#    A shape it cannot price yields a zero cycle cost -> region skipped, silently.
# 3. Is the WINDOW model calibrated for this target, not just the shape table?
```

Step 3 is the one that gets skipped. A window computed as "interval minus a constant" is calibrated
for one target's shape; another target's authoritative co-issue counts are **not** that formula's
output (`../hardware/isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget` carries both, and the CDNA3
pair is the counter-example). Adding a shape row to a cost table therefore does **not** finish the
port — the window model needs recalibrating from that target's own published counts.

Until both are done, record a **scoped tool ceiling** for that target and fall back: author the
interleave by hand in the pipeline layer, or use the wave-level stage markers, which need no
plugin at all (`scheduling-model.md ## inter_wave — realize with warp_pipeline_stage`). Do not record it as a hardware wall — the
mechanism (matrix/VALU overlap) exists on the whole CDNA family; it is the *tool* that is scoped.

## When the schedule is not the problem

Two exits from this layer, and taking neither is how a scheduling investigation runs forever.

- **Demand exceeds capacity.** No ordering wins; the work has to shrink or move. This is decided
  on paper, before any sweep (`## Attention: the co-execution budget`).
- **Demand fits, every in-band knob comes back neutral, and the window fill still sits low.**
  Then the bubble is **structural** — an operand or convert dependency the scheduler cannot
  reorder around, because the instruction it would hoist is not ready. Stop spending rounds on
  scheduler hints and attack operand production instead: the chain feeding the region, not the
  order inside it.

The second exit is the one that gets missed, because a neutral result reads as "the knob is
weak" rather than "the question is wrong". Two neutral sweeps against a budget that fits is the
signal to change layer, not to widen the sweep. Rule out the two silent skips first
(`## Applicability gate: shapes the tool does not model`, and an unsatisfiable declaration in
`## Declaring the interleave`) — both also present as "the scheduler did nothing".

## Acceptance signals

Landed, in the generated code:

- **Throughput model:** MFMA interleaved with the memory ops instead of clustered; a reorder
  barrier after each memory anchor; region count matching the loop's memory-op structure.
- **Co-execution model:** each compute region's vector ops sitting between its MFMAs rather than
  piled before the first or after the last; the declared groups all placed (a region that fell
  back to instruction-selection order is the signature of an unsatisfiable declaration).

Won, at the boundary:

- Per-region **window fill** and the in-loop matrix-issue efficiency, which are frequency
  independent and therefore the signal that the overlap is working.
- Interleaved timing at the pinned boundary as the arbiter — `scripts/ab_bench.py` for search
  screening; acceptance numbers come from the harness_lib-timed legs
  (`../method/benchmark-hygiene.md`; GEAK's `verify_engineer` owns them) — plus
  correctness and the determinism check after any barrier or buffering change.
- **Judge a scheduling change by cycles and the kernel by both cycles and wall time.** They
  disagree for a real reason (`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`), and a
  change that improves cycles while flat on wall time is usually still the right change.

## The same levers under other names

Published kernels and vendor notes name these mechanisms after the knob that switches them rather
than after what they do, so a technique you already have can arrive looking new. The mapping:

| Name you may meet | What it is here |
| --- | --- |
| `MEMNOP`, "mem-stage nops" | phase pacing — `## Phase pacing: moving a burst instead of removing work` |
| `SCALE_ON_Q`, "pre-scale Q" | remove the third source — same section; also the cheapest-carrier fold in `../workloads/attention.md` |
| "lazy rescale", lagging max | conditional rescale skip — `../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)` |
| `VEC1` / `VEC2` | the two halves of a split softmax, one per compute region — the balance in `## Attention: the co-execution budget` |
| "declare with `sched_group_barrier`" | the declarative form in `## Declaring the interleave` |
| "warp pipelining", "8-wave ping-pong" | the wave-level schedule these regions sit inside. Explicit stage markers are a Gluon-path mechanism, documented in `warp-pipeline.md` (Gate 0: `num_warps >= 8`); on plain Triton the analogous schedule comes from the automatic ping-pong pass |
| "coexec scheduler", `TRITON_HIP_USE_COEXEC_SCHEDULER` | the stock 3.8.0 `amdgpu-sched-strategy=coexec` strategy for co-execution regions — `compiler-contract.md ## What upstream 3.8.0 actually gives you` |

The mapping matters in one direction in particular: a kernel note reporting that `MEMNOP=2` and
`SCALE_ON_Q` together bought a couple of points of matrix efficiency is reporting **two different
mechanisms** — one moves a burst in time, the other deletes register-file pressure — and they are
tuned separately.

## Cross-refs

- `warp-pipeline.md` — the wave-level schedule that creates the regions this page schedules
  within; explicit stage markers are Gluon-only; plain Triton gets the same structure,
  unplaceable, from the automatic ping-pong pass
- `pipeline.md` — the overlap has to exist first, hand-written first; authored staging vs
  re-injection (lowest rung) and their exclusivity
- `scheduling-model.md` — layer 1.5: which model owns the overlap, per region
- `instruction-scheduling.md` — per-instruction pacing and the declarative hints
- `compiler-contract.md` — the upstream 3.8.0 capability table (`coexec`, `llvm_fn_attrs`), the
  per-change protocol, and the sanctioned tier
- `llvm-codesign-handbook.md` — the plugin skeleton and the build/probe procedure
- `../hardware/isa-mechanisms.md` — intervals, windows, per-class issue costs (all per shape)
- `../workloads/attention.md` — the variable table a new attention variant is read through
