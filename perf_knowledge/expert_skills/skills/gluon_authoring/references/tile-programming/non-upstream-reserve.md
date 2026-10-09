# Reserve: fork-only compiler mechanisms (LOWEST PRIORITY — not part of the method)

> ## STOP. Read this box before anything else on this page.
>
> **Nothing on this page works on upstream Triton 3.8.0.** Every environment variable, pass,
> plugin and ladder rung below was added by a **private vendor fork** and is **absent from a
> complete upstream 3.8.0 source tree** — absent from `python/triton/knobs.py`, from
> `include/triton/Tools/Sys/GetEnv.h`, from `third_party/amd/`, from the CMake files and from
> the test tree.
>
> **Do not set these variables and do not reach for these passes on a stock install.** An
> environment variable that no installed code reads is **tolerated and inert**: nothing errors,
> nothing changes, and the null result reads as *"this technique does not work on my kernel"*
> rather than *"that variable does not exist in this build"*. That mis-read is the single
> most expensive failure this page exists to prevent — which is why the material is quarantined
> here instead of sitting on a working page.
>
> **Two further warnings before the entries.** First, several of the mechanisms here are
> **superseded even on the fork** — they existed at early fork tags and were removed later, so
> they do not work on a current fork build either; each entry says so. Second, one capability
> this page describes as fork-only is in fact **shipped by upstream under a different name**
> (## 0. Read this first: one capability here is upstream after all).
>
> **This page is not a route.** It is not on any ladder, in any stage graph, or in any decision
> order. Reach it only when you have **already established, from the installed artifacts**, that
> you are running on a fork that ships the mechanism — not from a version string, not from a
> note, and not because a kernel report mentioned a knob name. The live pages carry the upstream
> answer for every capability discussed here; where an upstream route exists it is named in the
> **"upstream instead"** row of each entry below, and that row is where the work belongs.

## 0. Read this first: one capability here is upstream after all

**Matrix/VALU co-execution scheduling — the capability the LLIR-scheduler entry below exists to
provide — is shipped by upstream Triton 3.8.0 as the stock `coexec` scheduler strategy.** Verified
at the `v3.8.0` tag: `TRITON_HIP_USE_COEXEC_SCHEDULER` is a cache-invalidating upstream knob, and
`make_llir` sets `amdgpu-sched-strategy=coexec` when the co-exec scheduler is enabled **and**
`num_warps <= 4`. It is **default-on on gfx1250 only; opt-in on gfx950 / gfx942** (the env knob,
or `llvm_fn_attrs` per compile). The gate, the overrides, the interaction with
`warp_pipeline_stage` and the acceptance check are stated once in
`compiler-contract.md ## What upstream 3.8.0 actually gives you` — together with the two upstream
neighbours (`TRITON_HIP_USE_EXPERT_SCHEDULING`; the `TRITON_DUMP_MIR` / `TRITON_SWAP_MIR` /
`TRITON_SWAP_MIR_ENABLE_MISCHED` machine-IR dump-edit-reassemble path, which carries none of the
host-build gates the plugin material below exists to navigate).

So an agent whose region mixes matrix with VALU should reach for that strategy **before** reading
any further on this page, and should check whether it is already active (the attribute in the
`.llir`) rather than assuming the capability is missing.

This page's LLIR-scheduler entry is therefore about a *particular fork implementation*, not about
a capability upstream lacks.

The hardware facts any of this is budgeted against — matrix instruction intervals, the
matrix/VALU co-execution window, per-class issue costs — are per shape and per target and live
one layer down in `../hardware/isa-mechanisms.md`. Nothing on this page changes them; a fork
mechanism schedules against those numbers, it does not alter them.

## How to read an entry

Each mechanism is recorded in three parts, because the three fail differently:

- **What it did** — the mechanism, preserved as it was written, so the knowledge survives.
- **What it required** — the build property, not the variable. Setting a variable is not having
  the mechanism, and on this material the two are routinely confused.
- **Upstream instead** — what a stock 3.8.0 install actually gives you for the same capability.
  Some of these are full substitutes, some are partial, one is nothing at all. The entry says
  which.

## 1. The LLIR-scheduler plugin

### What it did

An LLVM `FunctionPass` scheduling the hot loop at the LLVM-IR level. It classified instructions
into matrix (`MFMA`), global-read, local-read, local-write and convert classes, and interleaved
*prefetched* matrix regions with the memory ops at the hardware throughput rate — an O(n)
pairing, not a dependency analysis.

Its model is the **throughput model** the live page carries — pairing arithmetic, region rule and
its dependency invariant are stated once in `llir-codesign.md ## GEMM: the throughput model`. The
fork-specific consequence: the loop had to be written DOT-first — a loop shape that puts its LDS
read at the top and its global load plus LDS write at the bottom forms one region and gives the
model nothing to interleave against.

It exploited an author-written contract; it did not discover one. Without independent
accumulator / local-read / dot work inside the iteration and a prefetched multi-buffer body, the
region was skipped rather than merely unhelped.

**Two generations, and they behaved differently.** The earlier one **disabled** LLVM's `misched`
/ `post-misched` globally so they could not re-cluster the result — which gave up latency-aware
scheduling in the prologue, the epilogue and every skipped region. The later one **pinned** the
interleave with a full reorder barrier after each memory anchor and left the machine scheduler
enabled. Notes describing a misched-disabling scheduler describe the older component, and its
failure modes are not the later one's.

**Workload boundary.** It assumed a pure matrix-to-matrix accumulator chain, so on
VALU-between-matmul kernels it produced **invalid IR / a verifier assertion**, not a slowdown —
the general hazard of any throughput-pairing scheduler, stated in `compiler-contract.md ## 风险 /
误用 (risks + mitigations)`.

**Applicability gate that reads like a hardware ceiling.** It carried a per-shape cost model and
skipped unpriced shapes silently, and its window model was calibrated for one target — the
general gate is `llir-codesign.md ## Applicability gate: shapes the tool does not model`.

### What it required

| generation | how it was reached | status on the fork itself |
| --- | --- | --- |
| in-tree | `TRITON_ENABLE_LLIR_SCHED=1` (fork-only; absent in upstream 3.8.0, inert if set) | **superseded** — present only at early fork tags, removed at the tag that moved the pass out of tree. Setting it on a current fork build does nothing |
| out-of-tree | `LLVM_PASS_PLUGIN_PATH=<abs>/libLlirSched.so` (upstream door) **+** `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1` (fork-only; absent in upstream 3.8.0, inert if set) | current |

**Do not reach for the in-tree spelling.** It is inert on upstream *and* on a current fork build,
which makes it the worst of the names here: two different wrong conclusions are available from the
same null result.

The out-of-tree form additionally required a **default-visibility host build**, and a host whose
target-machine retention for plugins was a **source** property — a build lacking it accepted the
fork variable and still ran the optimizer untargeted. The three build states, the ABI lock and
how each fails are stated once in `llir-codesign.md ## The plugin tier`.

### Upstream instead

- `LLVM_PASS_PLUGIN_PATH` **is upstream** and reaches the Gluon path; `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE`
  is not, and upstream deliberately builds **no** target machine when a plugin is set. Both
  facts, the source lines and the measurement consequence (plugin-on and plugin-off are not the
  same optimization environment upstream) are in `llir-codesign.md ## The plugin tier`. The
  **door** is stock; what was fork-side is a host built so anything can bind to it, and the
  scheduling *policy* that was loaded through it.
- Writing your own pass and loading it through that stock door is the supported route:
  `llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton` is upstream-valid as written.
- For matrix/VALU regions, the stock `coexec` strategy (## 0) comes first.
- The kernel-side capability — an authored ring with wave-level stage markers — is upstream via
  `gl.amd.warp_pipeline_stage`, which needs no plugin at all (`warp-pipeline.md`).
- `TRITON_PLUGIN_PATHS` (the MLIR TTIR/TTGIR pass-plugin door) is also upstream, behind the
  `TRITON_EXT_ENABLED` CMake option (default OFF; fails loudly) — `llir-codesign.md ## The plugin tier`.

## 2. The AMDGCN-assembler plugin (`amdgcnas`)

### What it did

A post-assembly tool running at the `make_amdgcn` stage: loop-invariant code motion of LDS
address arithmetic out of the hot loop, plus a matrix/scalar-ALU peephole packing scalar ops
into matrix-instruction gaps — scalar work that an IR-level scheduler cannot reach because those
ops appear only at MIR codegen. Levels `1` and `2` were increasing peephole aggressiveness.

Contribution scaled with how much scalar work the loop actually had: a low-precision body with a
scale pipeline gave it more to pack than a plain f16 body.

**Applicability boundary, independent of any scheduler generation.** Its loop analysis assumed a
**single-basic-block self-loop** (one block whose predecessor and successor are itself). A hot
loop with internal branches — a last-iteration prefetch guard, a conditional store, any `if` in
the body — is multi-BB and outside the loop detector, so the tool did not apply and might fail
to find the loop at all. This is why it fitted clean GEMM K-loops and not the branchy, guarded
loops typical of attention prefetch pipelines. Post-assembly rewriting in general inherits this
assumption.

Verification carried a correctness hazard worth keeping: kernel-descriptor / metadata VGPR
counts had to stay consistent with `.text`, or the result was a silent wrong value or a hang.

### What it required

| generation | how it was reached | status on the fork itself |
| --- | --- | --- |
| in-tree | `TRITON_ENABLE_AMDGCN_AS=1` (levels `1` / `2`; fork-only; absent in upstream 3.8.0, inert if set) | **superseded** — early fork tags only |
| out-of-tree | `TRITON_AMDGCNAS_PLUGIN=1` (fork-only; absent in upstream 3.8.0, inert if set) | current |

Either way it needed the fork's `amdgcnas` module available to the Triton install. Note the
out-of-tree spelling is **not** a compiler env var at all: it is read with `os.environ` by the
plugin's own Python, which is why this one mechanism is decoupled from the LLVM plugin machinery
and its host-build gates.

### Upstream instead

**Nothing equivalent.** Upstream 3.8.0 has no post-assembly rewriting stage and no module by
this name anywhere in the tree. This is the one entry on this page with no upstream substitute,
partial or otherwise.

The addressable part of what it did is reachable from the source side rather than the assembler
side: reduce the loop-invariant address arithmetic the loop emits in the first place, and reduce
in-loop scalar work, so there is less for a peephole to have recovered. Whether that closes the
same gap is **not measured here**; the measurement that would settle it is an A/B of the same
kernel at the same benchmark boundary with the address arithmetic hoisted at source level.

## 3. The RA-hints / AGPR-form knobs

### What it did

Forced matrix accumulators into AGPRs by setting the LLVM function attributes
`amdgpu-agpr-alloc=256` and `amdgpu-mfma-vgpr-form=0`, freeing VGPRs and removing in-loop
`v_accvgpr_*` copies.

**Not free, and the governing fact is accumulator read-cadence** (pays only for a
write-only-until-epilogue GEMM accumulator on compute-bound large-K kernels; wrong for a
read-modified accumulator; LLVM's default allocator already splits VALU-between-matmul kernels
correctly). Stated once, with the epilogue-cost reasoning, in `compiler-contract.md ## 风险 / 误用
(risks + mitigations)`; it applies unchanged to the upstream spelling below.

### What it required

| generation | how it was reached | status on the fork itself |
| --- | --- | --- |
| in-tree | `TRITON_ENABLE_AMDGPU_RA_HINTS=1` (fork-only; absent in upstream 3.8.0, inert if set) | **superseded** — early fork tags only; replaced by the row below |
| out-of-tree | `TRITON_FORCE_MFMA_AGPR=1` (fork-only; absent in upstream 3.8.0, inert if set) | current |

### Upstream instead

**This one is fully reachable on upstream 3.8.0, and the live pages carry it.** The fork
variables were only a wrapper: `amdgpu-agpr-alloc` and `amdgpu-mfma-vgpr-form` are **LLVM
function attributes**, reachable per compile through `llvm_fn_attrs` (3.8.0 only; raises on 3.6.0 /
3.7.x, no downgrade) — strictly better than a global environment variable, because a global knob
that wins on one kernel can regress a sibling compiled in the same process. Mechanism:
`llvm-fn-attrs.md`; capability row: `compiler-contract.md ## What upstream 3.8.0 actually gives you`.

## 4. Names that were checked and have no referent — do **not** look for these

Three-way provenance (upstream `main`, the `v3.8.0` tag, and every tag of the fork lineage,
including `git log --all -S` over both repositories' full history) found **no trace** of the
following. They are not fork mechanisms whose build we lack; they describe nothing, and they are
recorded here only so that meeting one in an old note ends the search instead of starting one:

`TRITON_GLUON_SWP_PIPELINE`, `TRITON_GLUON_COOP_LDS`, `TRITON_GLUON_PINGPONG`,
`TRITON_ENABLE_ATTN_SCHED`, a source file named `AttnSchedule.cpp`, and a scheme called
"cooperative-LDS staging".

No mechanism description is given for them because there is nothing to describe. If you need the
*capabilities* those names gesture at: multi-buffered LDS staging is fully expressible by hand;
wave-level ping-pong is `gl.amd.warp_pipeline_stage` (or `TRITON_HIP_USE_BLOCK_PINGPONG` on the
plain path); and attention-shaped instruction scheduling is the stock `coexec` strategy (`TRITON_HIP_USE_COEXEC_SCHEDULER`, opt-in on gfx950 / gfx942)
(## 0).

**Not to be confused with the pack's own script variables.** `TRITON_GLUON_SWP`,
`TRITON_GLUON_SWP_BUF` and `TRITON_GLUON_SWP_NOPP` are legitimate: they are defined *and* consumed
by this pack's `scripts/patch_reinject.py` (gluon pack), so they work anywhere, need no fork, and
are not on any list on this page. They arm re-injection of plain's pipeliner, which is the
**lowest** pipeline rung (below-parity diagnostic or last resort, numbers labelled "injected",
never on an incumbent Gluon kernel — `pipeline.md`). One caveat on them: because they are not Triton knobs they sit
outside `CACHE_INVALIDATING_ENV_VARS`, so they change the pass list **without** changing the
compile cache key — set `TRITON_ALWAYS_COMPILE=1` when toggling them or you will A/B a cached
binary against itself.

## 6. Environment-variable index

Everything in the left column is **absent from upstream Triton 3.8.0 and present in the fork
lineage**. The evidence for every row is the same and is reproducible: the name does not appear in
`python/triton/knobs.py`, in `include/triton/Tools/Sys/GetEnv.h`, or anywhere else in a 3.8.0 tree
— checked at the `v3.8.0` tag, not only on `main`. Names that appear in **neither** tree are in
§4 and are deliberately not listed here. Every row is **fork-only; absent in upstream 3.8.0,
inert if set** — this is the label the live pages use wherever one of these names appears.

| fork-only variable | mechanism | upstream route for the same capability |
| --- | --- | --- |
| `TRITON_ENABLE_LLIR_SCHED` (*superseded on the fork too*) | LLIR scheduler, in-tree generation | the stock `coexec` strategy for the matrix/VALU case (`TRITON_HIP_USE_COEXEC_SCHEDULER` or `llvm_fn_attrs`; opt-in on gfx950 / gfx942, ## 0); else `gl.amd.warp_pipeline_stage`, or author a pass and load it via stock `LLVM_PASS_PLUGIN_PATH` |
| `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE` | host-side target-machine retention for LLVM plugins | none — upstream drops the target machine when a plugin is set, by design |
| `TRITON_ENABLE_AMDGPU_RA_HINTS` (*superseded on the fork too*) | AGPR accumulator pinning, in-tree | `llvm_fn_attrs="amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0"` (3.8.0) |
| `TRITON_FORCE_MFMA_AGPR` | AGPR accumulator pinning, out-of-tree | same as above |
| `TRITON_ENABLE_AMDGCN_AS` (*superseded on the fork too*) | post-assembly peephole, in-tree | none |
| `TRITON_AMDGCNAS_PLUGIN` | post-assembly peephole, out-of-tree | none |

**Genuinely upstream — do not remove these from a plan.** `TRITON_CACHE_DIR`,
`TRITON_ALWAYS_COMPILE`, `TRITON_F32_DEFAULT`, `TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`,
`DISABLE_LLVM_OPT` (which takes a comma-separated LLVM flag list, so
`DISABLE_LLVM_OPT=disable-machine-sink` is a real upstream control), `TRITON_PLUGIN_PATHS`,
`LLVM_PASS_PLUGIN_PATH`, `TRITON_EXT_ENABLED` (a CMake option, default OFF),
`TRITON_HIP_USE_ASYNC_COPY`, `TRITON_HIP_USE_BLOCK_PINGPONG`, `TRITON_HIP_USE_COEXEC_SCHEDULER`,
`TRITON_HIP_USE_EXPERT_SCHEDULING`, `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`, `TRITON_DUMP_MIR`,
`TRITON_SWAP_MIR`, `TRITON_SWAP_MIR_ENABLE_MISCHED`. `ROCM_PATH` is a ROCm variable and
has nothing to do with Triton's knob set.

## 7. The source artifacts, and what a fork-build note names

None of these files exist in an upstream tree; a note or a report naming one is describing a
fork build, and that is the most reliable single tell:

| artifact | what it was |
| --- | --- |
| `LLIRSchedule.cpp` (AMD backend), wired via `add_llir_schedule_pass` in `make_llir` | the in-tree LLIR scheduler |
| `libLlirSched.so` | the out-of-tree generation of the same |
| the `amdgcnas` module under `triton/tools/`, run in `make_amdgcn` | the post-assembly tool |

A fork's tag lineage also moved these between generations — an early series shipped the
scheduler and the assembler tool **in tree**, and a later tag **removed both from Triton and
re-shipped them as out-of-tree plugins**. That boundary is why the same component is reached by
two different variable names, and why a name from the other generation is inert rather than an
error.

## 8. The ladder, recorded whole

The fork's GEMM-class configuration ladder was `base -> llir -> llir+ra -> llir+amdgcnas`, with
the active configuration recorded as *the component plus the mechanism used to reach it*
("llir" alone does not say which generation ran). It was GEMM-only at every rung, and
default-skipped for any kernel with vector math between matmuls.

**There is no upstream equivalent of the ladder as a ladder.** Two of its three rungs have
upstream routes (the scheduler via an authored plugin pass or stage markers; the RA rung via
`llvm_fn_attrs`) and one has none (the assembler peephole). Do not carry the rung names, the
configuration labels, or a conclusion recorded against them onto a stock build — and do not
record the ladder's absence as a hardware ceiling. It is a **toolchain** scope, and the
mechanisms the rungs exploited (matrix/memory overlap, matrix/VALU co-execution) exist across
the CDNA family regardless of which build you are on.
