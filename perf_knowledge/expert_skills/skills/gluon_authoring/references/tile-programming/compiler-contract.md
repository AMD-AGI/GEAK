# Compiler Contract: what stock Triton 3.8.0 gives you + IR verification

An optimization is **not landed** until the IR/asm shows it. This page owns: the scope (kernel only,
compiler read-only), the upstream 3.8.0 compiler controls, the IR dump / hot-loop audit workflow, and
a few compiler-side traps. `llvm_fn_attrs` mechanics: `llvm-fn-attrs.md`. Fork-only names:
`non-upstream-reserve.md`. Per-instruction hints: `instruction-scheduling.md`.

## Compiler scope: kernel-first; the compiler is read-only

- **You may:** write the Gluon structure (layout chain, hand-written pipeline with independent
  `AC`/`LR`/`DOT`, `gl.amd.warp_pipeline_stage`, register budget); set the upstream per-compile / env
  controls in `## What upstream 3.8.0 actually gives you`; dump and verify IR.
- **Re-injection:** re-inject plain's already-compiled pipeliner passes (no rebuild, no file edit);
  mainly a below-parity diagnostic, numbers labelled "injected"; on an already-Gluon kernel it would replace
  the authored staging (`pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`).
- **Not reachable on the stock 3.8.0 wheel without a rebuild:** front-end or backend source changes, LLVM
  rebuilds. LLVM pass plugins: `libtriton.so` reads `LLVM_PASS_PLUGIN_PATH` / `TRITON_PLUGIN_PATHS`, but loading
  a plugin on the stock wheel was not demonstrated here (reported to need a default-visibility host build;
  `TRITON_EXT_ENABLED` is a CMake option, OFF by default).
- **Ceiling vs kernel bug.** Record "needs a compiler change" only after the IR shows the structure is
  correct (prefetched MFMA regions present, no hot-loop `convert_layout`, budget fits, AC/LR/DOT really
  independent) and the compiler still clumps / spills. Then: record a scoped toolchain ceiling, fall
  back (manual multi-buffer, other layer, or plain), and write a handoff: pass, IR/profile evidence,
  one-line proposed change.

### Sanctioned tier: changing the front end's own source

Not reachable on the stock 3.8.0 wheel (source edit + rebuild). If a pin change took away an
arrangement your kernel relied on (barrier placement, priority-hint position), record it as a scoped,
dated toolchain finding (`../pitfalls/platform-known-issues.md`), not as a kernel ceiling.

### Scenario B: sanctioned compiler co-design

Not reachable on the stock 3.8.0 wheel (backend pass / LLVM rebuild). Two facts from this tier that
still bound kernel work:

- **A runtime-indexed shared-buffer fence is load-bearing.** A correctness-preserving compiler must
  fence cooperatively loaded shared buffers indexed by runtime values (`k % nBuffers`); dropping that
  fence gives wrong results (verified). Hand-asm omits it only by cycle-counting buffer disjointness, so
  a tight `s_setprio`-only cooperative-load ping-pong is a hand-asm capability — a scoped abstraction
  ceiling (`mental-model.md ## The hand-asm ceiling`) rather than an open compiler task.
- **Removable over-sync, kernel side.** A relaxed shared-load (`load_shared_relaxed`) makes the membar
  filter skip the RAW barrier before that read only when no `wait_group` sits between copy and read
  (readiness by a hand-issued `s_waitcnt`); after a `wait_group` the compiler emits an `s_barrier` with
  the wait regardless and the relaxed read removes nothing
  (`pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`).
  Safe only for compile-time-literal buffer indices; gate every elision with the determinism race-test
  (`../method/benchmark-hygiene.md ## Determinism race-test`). (The compiler-side variant, extending
  the AMD membar filter to async<->async pairs, which v3.8.0 never filters, needs a rebuild.)

## AGPR hint workload gate: accumulator read-cadence

**Raising the AGPR budget for accumulators (`llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]`) has a plausible
payoff mainly for a write-only-until-epilogue accumulator (pure GEMM chain, compute-bound, large K)** — not
measured over that scope here; on 3.8.0 the value is a minimum, and a compile check left a 128x128
accumulator in VGPRs with it (`llvm-fn-attrs.md`).

- **Epilogue cost.** AGPR accumulators add `v_accvgpr_read` in the epilogue (the downcast needs VGPR
  inputs); on small-K / epilogue-heavy / non-compute-bound kernels the hint can regress. A/B with and
  without it.
- **VALU between matmuls (attention, MLA, Sage-attention, DSA, gating, dequant): the accumulator is
  read-modified every iteration** (`acc *= alpha`, rescale), which in AGPR pays a `v_accvgpr_read`/`write`
  round-trip per iteration, so the GEMM-style AGPR hint works against these ops. At 2 waves/SIMD an
  attention body near 256 registers spilled under LLVM's default VGPR/AGPR split and needed
  `amdgpu-agpr-alloc=0` (all arch VGPRs); the GEMM hint (`amdgpu-agpr-alloc=256`) did not launch at 2
  waves/SIMD and cost occupancy at 1 (GPU-measured, gfx950 / Triton 3.8.0). Lighter bodies may fit the
  default split — read `v_accvgpr_*` / spills in the loop. A partial budget is a separate lever with its own signal (the default half split
  spills or shuttles `v_accvgpr` in the loop, or `"0"` spills): it regresses outside that signal and can
  win inside it (`llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`) — so the "attention RA hint" is simply *not
  applying the GEMM hint*. If tried, gate by bound class and keep the best.
- **Demand-bound vs split-bound.** A kernel with too many live values (demand-bound) is not helped by any
  VGPR/AGPR re-split (the total does not change); a wrong split (split-bound) is. `slicing.md` separates the
  two.
- `amdgpu-agpr-alloc="256"` is valid only at 1 wave/SIMD; budget semantics live in `llvm-fn-attrs.md`.
- A custom GEMM-style throughput-pairing pass applied to a VALU-between-matmul loop was reported
  upstream to emit invalid IR (`Instruction does not dominate all uses`) rather than a slowdown
  (unverified here; pass authors only). The stock 3.8.0 block-pingpong pass
  (`TRITON_HIP_USE_BLOCK_PINGPONG=1`) on an attention loop bails silently instead (compiles, no
  `s_setprio`, no `sched_barrier`; compile-checked gfx950), so "no change" there is a non-application, not a
  schedule.

## Scope: reference for authoring, not a portable recipe

A scheduled result is per kernel / target / build (gfx950 results are not assumed to carry to gfx942).
Scheduling cannot beat a throughput wall; the busy counter is what shows the wall
(`../method/profile.md ### Rule: read the busy/throughput counter, not the lane-activity one`). On the
3.8.0 Gluon path `num_stages` has no effect (no pass reads it) and compiler-automatic ping-pong does not run
(BlockPingpong is only in the plain pass list); the phase offset comes from `gl.amd.warp_pipeline_stage`
(`warp-pipeline.md`, `num_warps >= 8`).

## Toolchain identity

Record beside every number: Triton version (3.8.0 baseline; read `triton.__version__`, and note that a
framework editable install can swap it, see `## Source-sensitive traps`), build lineage (stock wheel vs
vendor build), LLVM pin. Also record the **active** scheduling configuration read from the artifact:
the `amdgpu-sched-strategy` attribute actually on the kernel in the `.llir` and the process value of
`TRITON_HIP_USE_EXPERT_SCHEDULING` (cache-invalidating). A probe is more reliable than a version string:
`scripts/probe_levers.py --all` (reports `version_disjoint_knobs` as `live` / `dead-declaration` /
`absent`). Downgrade 3.7.x: no `llvm_fn_attrs` (raises as an unknown compile option), `schedule_hint`
live; 3.6.0 has no `warp_pipeline_stage` marker path (`../gluon/pipeline-reference.md`).

## What upstream 3.8.0 actually gives you

**Fork env vars (LLIR scheduler, RA hints, post-assembly peephole, attention scheduler) do not exist in
upstream 3.8.0 and are inert if set** (no hit for any of the names in the installed 3.8.0 tree, `libtriton.so`
included) — no error, no change; a flat result means "not in this build", not "technique fails", and their
absence is not a ceiling. Names and upstream substitutes:
`non-upstream-reserve.md ## 6. Environment-variable index`. **Overrides KB** for upstream Triton 3.8.0:
`perf_knowledge/operators/dense_gemm/backends/gluon.md`, `perf_knowledge/operators/scaled_quant_gemm/backends/gluon.md`
and `perf_knowledge/operators/quant_fp4_mxfp/backends/gluon.md` give `TRITON_ENABLE_LLIR_SCHED=1
TRITON_ENABLE_AMDGCN_AS=1` as the way to enable IR interleave and the post-assembly peephole; verified by a
string search of the installed 3.8.0 tree (Python sources and `libtriton.so`): neither name occurs, so both
are inert there (the tutorial fork reads them). Same override for `perf_knowledge/languages/gluon/overview.md`
("enabled at runtime via `TRITON_ENABLE_LLIR_SCHED=1` / `TRITON_ENABLE_AMDGCN_AS=1`"), `perf_knowledge/languages/gluon/programming_model.md`
("The interleaving passes are toggled at runtime") and `perf_knowledge/languages/gluon/gemm_cookbook.md` (v9, "`TRITON_ENABLE_LLIR_SCHED=1
TRITON_ENABLE_AMDGCN_AS=1`").

**`llvm_fn_attrs` (3.8.0 only).** Per-compile option attaching arbitrary LLVM function attributes to
the kernel in `make_llir`, after the backend's own (explicit value wins). Route to AGPR budget,
`amdgpu-sched-strategy`, VGPR cap, IEEE mode, target features, for `@triton.jit` and `@gluon.jit`.
Mechanism and acceptance: `llvm-fn-attrs.md`.

**`coexec` strategy: gfx1250 only.** `make_llir` adds `amdgpu-sched-strategy=coexec` when
`is_coexec_scheduler_supported(arch)` (`arch == "gfx1250"`) and `num_warps <= 4`. There is **no**
`TRITON_HIP_USE_COEXEC_SCHEDULER` env knob on 3.8.0. Forcing
`llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` on gfx950/gfx942 compiles, LLVM warns
`'amdgpu-sched-strategy'='coexec' is only supported for gfx1250` and does not run the co-execution strategy,
yet the schedule differs from the default (and from every honoured strategy; an unknown name gives exactly the
default), so an asm diff is **not** evidence coexec ran (compile-checked gfx950 / gfx942). On CDNA3/4 use `iterative-ilp`, `max-ilp`,
`iterative-minreg`, `max-memory-clause`, `iterative-maxocc` (unknown values silently = default).
`TRITON_HIP_USE_EXPERT_SCHEDULING` (adds `amdgpu-expert-scheduling-mode`; arch default gfx1250 only) is
upstream too. Verified on the installed 3.8.0 wheel (`triton/backends/amd/compiler.py`, `triton/knobs.py`).

| capability | upstream 3.8.0 route | reach |
| --- | --- | --- |
| throughput interleave (MFMA spread between `buffer_load` / `ds_read`) | author the structure (`pipeline.md`), pace with `warp_pipeline_stage` (`warp-pipeline.md`) or a strategy via `llvm_fn_attrs` | full structurally; no declarative interleave |
| matrix/VALU co-execution (softmax / scale / dequant between matmuls) | region balancing by hand (`llir-codesign.md ## Attention: the co-execution budget`); `coexec` on gfx1250 | no stock 3.8.0 scheduler route on gfx950/gfx942; placement needs a pass |
| scheduler strategy per kernel | `llvm_fn_attrs=[["amdgpu-sched-strategy", "<s>"]]` | full, per compile |
| accumulator in AGPR | `llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]` (function attribute). **`amdgpu-mfma-vgpr-form` is an `llc` option, inert as an `llvm_fn_attrs` entry** (identical asm for 0/1 and unset, compile-checked 3.8.0 gfx950) | GEMM-class only (`## AGPR hint workload gate: accumulator read-cadence`) |
| machine-IR hand edit | `TRITON_DUMP_MIR` / `TRITON_SWAP_MIR` / `TRITON_SWAP_MIR_ENABLE_MISCHED` (dump, edit, reassemble MIR) | full, per kernel, manual. Run on 3.8.0 / gfx950: the dump has the scheduling-DAG dump appended (cut it off before swapping, else the swap fails to parse); the dump is pre-scheduling MIR, so without `ENABLE_MISCHED` the swapped code is emitted unscheduled |
| keep a balanced region's placement | `DISABLE_LLVM_OPT=disable-machine-sink` (`llir-codesign.md ## Two preconditions the kernel must protect`) | full |
| post-assembly peephole (LICM of LDS address math, scalar ops into MFMA gaps) | **none** — upstream 3.8.0 has no post-assembly stage | reduce loop-invariant address math / in-loop scalar work at source |

Setting a knob is not evidence it landed; the generated code is:

| capability | landed when |
| --- | --- |
| throughput interleave | MFMA no longer clustered; loads for `k+1` issue before `DOT(k)` |
| scheduler strategy | attribute on the kernel in `.llir` **and** `.s` differs from the same compile without it **and** the compile log has no LLVM "only supported for ..." warning (`coexec` on gfx950 / gfx942 passes the first two checks while unsupported, 3.8.0) |
| accumulator in AGPR | `v_mfma` destination is `a[..]`; fewer in-loop `v_accvgpr_mov` |
| source-side address reduction | fewer loop-invariant address ops in the loop body |

## Portable scheduler co-design (the env knobs, and where the real lever lives)

- **`schedule_hint` is a dead declaration on 3.8.0** (installed `compiler.py`: declared, read nowhere): still on
  the options dataclass, accepted, hashed into the compile key — a sweep comes back flat and a presence probe
  says "yes". The 3.8.0 route is `llvm_fn_attrs` (on 3.6/3.7 `schedule_hint` presets were the live knob).
  **Overrides KB** for Triton 3.8.0 (gfx950 and gfx942): `perf_knowledge/languages/triton_amd/knobs.md` and the
  operator cards that recommend `schedule_hint="attention"` (e.g. `perf_knowledge/operators/attention_prefill_fmha/tuning.md`,
  `.../softmax/tuning.md`); verified by a compile-only check: an attention loop compiled with `""`,
  `attention`, `memory-bound-attention` and `local-prefetch` gives identical asm on both arches.
- **`BlockPingpong` does not fire on hand-authored staging (3.8.0).** It is reachable only through the plain
  pass list and collects `local_load`s sourced from a loop-carried `BlockArgument`; a hand-written one
  comes from `memdesc_index`.
- **Upstream 3.8.0 reads no `TRITON_GLUON_*` variable** (no hit in the installed tree); re-injection is armed from Python (the pack's
  `TRITON_GLUON_SWP*` script variables: `non-upstream-reserve.md ## 4. Names that were checked and have
  no referent`). `TRITON_ALWAYS_COMPILE=1` is upstream and forces recompiles while iterating.
- **Rule: an env knob no installed code reads fails silently.** Probing the mechanism (does the IR
  change?) is what tells; the variable's presence does not.
- A loop that lost the pipeline plain had shows equal VGPR+AGPR and occupancy vs plain, lower `MfmaUtil`
  and more `s_waitcnt lgkmcnt(0)` (`../method/profile.md ### Rule: equal registers+occupancy but lower
  MfmaUtil => schedule/overlap gap`); for that signal a hand-built pipeline (`pipeline.md ## Authoring the
  overlap yourself (the climb default on the Gluon path)`) addresses the cause, while a scheduler sweep works
  around it.

## VALU-between-matmul: manual interleave, or author a pass

For softmax / scale-between-matmul loops the GEMM-side hints are not a default win: the AGPR hint can add
round-trips (gate it by bound class, above), and a custom throughput-pairing pass was reported upstream to
emit invalid IR there (unverified here; pass authors only; the stock block-pingpong pass bails silently
instead). The options:

1. **Kernel side:** independent `AC`/`LR`/`DOT` staggered by hand in the Gluon pipeline
   (`pipeline.md`), wave-level stage markers (`warp-pipeline.md`), region balancing
   (`llir-codesign.md ## Attention: the co-execution budget`).
2. **Strategy via `llvm_fn_attrs`:** on gfx950/gfx942 the non-`coexec` strategies listed above; which one pays
   varies by kernel and bound class (the recorded wins came from different strategies on different kernels),
   so A/B them. `coexec` is gfx1250-only (3.8.0).
3. A placement-declaring pass needs a plugin or a rebuild (`llir-codesign.md ## The plugin tier`).

There is no upstream 3.8.0 attention-scheduler toggle (`TRITON_ENABLE_ATTN_SCHED`: no hit in the installed tree);
its absence is not a ceiling. Before concluding a scheduler "does nothing", rule out the two silent
skips: `llir-codesign.md ## Applicability gate: shapes the tool does not model` and
`llir-codesign.md ## Route the loop first: who built the overlap`.

## Per-change protocol

Per change: dump IR, confirm the intended signal (structure, interleave, `a[..]` destinations), reprofile;
MFMA efficiency and the spill count are the two numbers to compare. A profile gain with no IR signal is not "landed".

## IR dump workflow

```bash
bash kernel_workflow/scripts/kernel_tools/dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts --arch gfx950
```

`kernel_workflow/scripts/kernel_tools/dump_ir.sh` takes one
`TRITON_CACHE_DIR` per variant. `--emit-gluon {layouts|anchor|pipeline}` needs an arch (refuses
otherwise) and recovers the TTGIR layouts (opt-in: the async pipeline) into Gluon via
`scripts/recover_gluon.py` with a layout-equivalence `--verify`. Register allocation / spills are not
recoverable from TTGIR (they happen in LLVM).

| File | Role |
| --- | --- |
| `.ttgir` | layouts (`#blocked`/`#mma`/`#shared`), `num_stages` (plain path only) |
| `.llir` | LLVM IR (function attributes, e.g. `amdgpu-sched-strategy`) |
| `.amdgcn` | emitted assembly |
| `.s` | `.amdgcn` without `.loc` / `.Ltmp*:` for stable line anchors |

```bash
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' in.amdgcn > out.s
```

Pin one Triton build for the whole comparison.

### Auditing the hot-loop schedule from the `.s`

`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py <variant>.s --arch gfx950` emits the loop's op-class stream + histogram,
classifies each `s_waitcnt` as relaxed (`lgkmcnt`/`vmcnt` `> 0`: pipelined) vs full-drain (`(0)` before
every consumer), and counts producer/consumer barriers per iteration.

- **Multi-kernel module:** the audit refuses and lists the kernels; pass `--kernel <name>` (unique
  substring). A whole-file audit gives wrong data: it maxes the register budget and merges loops across
  kernels with no error.
- **Verdicts:** `structural` (an exposed single-class run with no independent ready op to fill it —
  reordering cannot help), `conservative-waitcnt` (repeated full drains), `excess-sync` (barriers per
  iteration scale with pipeline depth), `unhidden-fixed-latency`. Only the middle two are
  reorder-actionable, and only when slack exists.
- **`s_nop N` in the body = an exposed fixed-latency hazard** (e.g. MFMA accumulator write -> VALU read).
  Hazard nops appear where no independent instruction is available, so the remedy is independent work between
  producer and consumer (source reordering of independent work, unroll) to absorb it, or more resident waves to
  hide it; reordering a stream with no independent work does not help. It reads `s_nop ~= 0` once
  unroll + waves/CU cover it (`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`).
  Before deleting an `s_nop`, read the next instruction: a DPP / cross-lane op after it means an ISA wait
  state (`llir-codesign.md ## Phase pacing: moving a burst instead of removing work`).

## Body micro-edits the compiler may already handle

Reordering for overlap, dead-allocation removal, scalar broadcast expansion and moving conversions were
usually neutral in the cases recorded (the compiler already does much of it); an IR/asm or profiler signal is
what distinguishes a real change before more rounds. `tl.trans` of a large
operand between two dots can be a real register shuffle: load each dot's operand in its own layout.

## CDNA4 scaled-helper argument types

**`gl.amd.cdna4.scaled_upcast` wants a Triton dtype object (`tl.bfloat16`), not the string `"bf16"`** (3.8.0
gfx950 compile check: the string form fails).
The string form raises an error that mentions `bf16` and reads like a target ceiling. Pass the dtype
object first; only a rejection after that is a real API/target limit.

## Async / buffer path failures are layout-first

`buffer_load_to_shared` lowering failures have mostly been layout-contract violations rather than missing
backend support; a suggested triage sequence (width, warp-row mapping and destination mismatches all surface as
the same `unrealized_conversion_cast` on 3.8.0, so the message alone does not say which):

1. reduce to an async-copy smoke on the same shared shape, dtype and consumer load
   (`memory-path.md ### Async copy: smoke-test before wiring`);
2. check per-lane width (gfx950 exactly 128 or 32 bit, gfx942 32 bit;
   `../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`), offset
   layout family, rank, element unit;
3. **the offset layout must be pre-coalesced** (`canLoadDirectToLDS`: contiguous threads on the LDS-fast
   dim, legal-width run, `vec == contig`): **Gluon runs no `CoalesceAsyncCopy` pass on 3.8.0** (plain does), so
   an un-coalesced offset fails lowering (compile-checked 3.8.0 gfx950) rather than being rewritten;
4. if `BlockedLayout` fails, try a `DistributedLinearLayout` matched to the consumer (padded
   destination: producer derived from its `offset_bases`, `layout-recipes.md ## Compute the layout
   instead of tabulating it (constexpr_function)`);
5. correct lowering but wrong values = a staging-plan bug, not a buffer-op rejection;
6. only well-formed layouts that persistently fail are a build ceiling.

gfx942: 32-bit only (the ISA shows only `buffer_load_dword ... lds`), destination `order` following the producer's
contiguous dim (`[1, 0]` for a row-major producer; compile-checked 3.8.0); a padded destination needs an interval
>= 256 B and, for non-identity offsets, an `offset_bases`-derived producer; a swizzled destination is the simplest form to bring up, and
sync staging is the alternative.

## Source-sensitive traps

- **A framework editable install can replace the installed Triton** (AITER and similar); keep the
  system one with the framework's opt-out (`AITER_USE_SYSTEM_TRITON=1`) and record the version before
  reading version-sensitive failures.
- Triton updates can change scalar return dtype (`int32` vs `int64`); fix at the wrapper/oracle
  boundary.
- `DistributedLinearLayout` basis vectors freeze tile dimensions: re-derive them before changing
  `BLOCK_SIZE_*`.
- AOT compile success does not prove the runtime path works.

## CDNA scheduling barriers (late-stage, and narrower than the mechanism suggests)

`sched_barrier` / `sched_group_barrier` / `iglp_opt` have no user-facing surface on 3.8.0 Gluon. What
is reachable instead: `instruction-scheduling.md ## The declarative hints that are not there`.
