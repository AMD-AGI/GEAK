# llvm_fn_attrs — attaching an LLVM function attribute to one kernel

**`llvm_fn_attrs` is a Triton 3.8.0 compile option applied in `make_llir`, shared by `@triton.jit`
and `@gluon.jit` kernels** — not a Gluon mechanism. When in a round to reach for a scheduler change:
`instruction-scheduling.md`. This page: what the option is, what you may write, how to confirm it took.

## What it is — a note on the kernel function, read by the backend

Each `name` / `value` pair is added to the generated LLVM function with `addFnAttr`, unchanged; what
it does is decided by whichever LLVM pass reads that name.

- **No allow-list; an unknown name or value is silently ignored.** It is carried into the `.llir`
  and ignored (verified on 3.8.0 / gfx950: `amdgpu-sched-stratgy=...` and
  `amdgpu-sched-strategy=bogus` both appear in `attributes #0` and produce the default's ISA byte for
  byte). Scheduling is one use of the door, not its boundary (upstream's example pairs a strategy
  with `noinline`).
- **Applied before the LLVM optimization pipeline**, so the whole pipeline sees it.
- **Your pair overrides what the backend already set** (the loop removes the name, then adds it,
  after the backend wrote its own attributes). Colliding names: `amdgpu-waves-per-eu` (normally from
  `waves_per_eu=`), `amdgpu-flat-work-group-size` (from `num_warps`), `denormal-fp-math-f32` (from the
  denormal option), and `amdgpu-sched-strategy`, which the backend sets to `coexec` only on gfx1250
  at `num_warps <= 4` (`is_coexec_scheduler_supported` in `triton/backends/amd/compiler.py`; no env
  knob on 3.8.0 and `schedule_hint` is read by nothing, so on gfx950 / gfx942 no strategy is written and this
  option is the only per-kernel route on 3.8.0). A
  collision is a silent regression no diff of your launch arguments shows.
- **Replacement, not merge:** a `target-features` pair replaces the function's whole feature string,
  including features the backend added (e.g. for a sanitizer build).

Two spellings (`HIPOptions.llvm_fn_attrs: str | Tuple[Tuple[str, str], ...]`, normalized by
`_parse_llvm_fn_attrs` in `triton/backends/amd/compiler.py`):

```python
# string: split on "," then on the first "="; a bare name becomes a valueless attribute
llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp,amdgpu-agpr-alloc=0"
# (name, value) pairs -- lists or tuples
llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"], ["amdgpu-agpr-alloc", "0"]]
```

**Use the pair form for any `min,max` value.** The string form splits on every comma:
`"amdgpu-agpr-alloc=0,0"` becomes `"amdgpu-agpr-alloc"="0"` plus a stray valueless attribute `"0"`
(verified in the 3.8.0 `.llir`), and `"amdgpu-waves-per-eu=2,2"` silently loses its max. Single
values: the forms are equivalent. `[]` / `""` = leave the backend's choice alone.

## The recipe that paid (gfx950, A/B-measured)

- **`amdgpu-agpr-alloc=0` on an 8-warp `warp_pipeline_stage` GEMM tile** (two 4-warp groups per CU
  halve each wave's register budget): under the default VGPR/AGPR split the pipelined body spilled;
  with `"0"` it gave `.agpr_count: 0`, spills 12 -> 4, no `v_accvgpr` moves, and adding
  `iterative-ilp` reached spill 0 (compile-verified kernel: `warp-pipeline.md ### A GPU-verified gfx950 snippet`).
  Signal: `.vgpr_spill_count > 0` or `scratch_*` in the loop together with non-zero `.agpr_count` /
  `v_accvgpr_*`. **`"0"` vs `"0,0"` depends on waves/SIMD:** at 2+ waves `"0"` is already a hard cap
  (same ISA as `"0,0"`); at 1 wave/SIMD `"0"` leaves AGPRs to the allocator — a 4-warp 256x256x64
  `tl.dot` needing more than 256 registers kept `.agpr_count: 256` under `"0"`, and pair-form
  `"0,0"` forced `.agpr_count: 0` with VGPR spills 32 -> 762. Read `.agpr_count` after every variant.
- **`amdgpu-sched-strategy=iterative-ilp`** on GEMM mainloops paid on some prefill tiles and spilled
  and lost on a sibling path of the same kernel family: per kernel, per launch, accepted only on the
  assembly diff plus a timing A/B. It is one of five honoured strategies; the one that paid differed by
  kernel (`max-ilp` on a 1-wave attention body, `iterative-minreg` as an occupancy enabler), and a config
  retune or a waves / AGPR change can move the answer, so a sweep across the five is the measurement —
  `### Scheduler strategy — which names the 3.8.0 LLVM honours on gfx950`).

### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)

Verified on Triton 3.8 / gfx950 (MI350X) on attention-style MFMA kernels with a VALU epilogue and on
an 8-warp ping-pong GEMM.

**N is a split of the per-wave budget, not a switch.** The per-wave vector-register budget `B` comes
from the occupancy target (`waves_per_eu`): 512 at 1 wave/SIMD, 256 at 2, 168 at 3, 128 at 4. With
the attribute `"N"` the allocator gets an arch-VGPR cap of `min(256, B - N)` and the rest of `B` as
AGPRs. Without it, the MFMA kernels measured here got `B/2 : B/2` — at 2 waves `"128"` compiled
byte-identical to no attribute. So at 2 waves: `"0"` = 256 arch / 0 AGPR (same ISA as `"0,0"`), `"64"` =
192 / 64, default = 128 / 128 on those kernels. The default is not universal: a compile check of a plain
8-warp Gluon MFMA GEMM loop (no stage markers) at `waves_per_eu=2` got 256 / 0 by default and 128 / 128
under `"128"` (3.8.0 gfx950), so read `.agpr_count` rather than assuming the default split. At 1 wave/SIMD every `N <= 256` leaves the 256 arch VGPRs uncapped, so the
attribute only changes anything at 2+ waves.

**What lives in AGPRs on gfx950** (the placement rule other pages point to) is whatever the allocator puts there: MFMA A/B source operands
(e.g. a loop-invariant Q fragment, read by `v_mfma ... a[..]` directly) and spill copies
(`v_accvgpr_write` / `v_accvgpr_read`). The MFMA *accumulator* is not pinned to AGPRs: in every
kernel and every N measured here the `v_mfma` destination was `v[..]`; a compile check of a 256x256
accumulator (demand past 256 VGPRs) did put part of the MFMA destinations in AGPRs. The destination operand
is what tells; "accumulators live in AGPRs" is not a safe assumption. (**Overrides KB** for
Triton / Gluon kernels on Triton 3.8.0, gfx950 and gfx942: every unlabelled statement in `perf_knowledge/` that the
MFMA accumulator lives or stays in AGPRs, among them `perf_knowledge/operators/attention_prefill_fmha/tuning.md`
("the MFMA accumulator lives in AGPR"), `perf_knowledge/languages/triton_amd/deep_codegen.md` ("-> AGPR accumulator"),
`perf_knowledge/languages/triton_amd/isa_verify.md` ("acc stays in AGPR"), `perf_knowledge/optimization/mfma_scheduling.md`
("the running `C` tile lives in AGPRs"), `perf_knowledge/index/glossary.md` ("MFMA accumulators live in AGPR"),
`perf_knowledge/hardware/cdna3_mi300/isa_notes.md` ("keep accumulators in AGPRs through the K-loop"),
`perf_knowledge/hardware/cdna4_mi350/isa_notes.md` and `perf_knowledge/hardware/cdna3_mi300/matrix_core.md` ("AGPR
accumulators"), `perf_knowledge/hardware/shared/matrix_core_mfma_smfmac.md` ("### Accumulators in AGPRs") and
`perf_knowledge/optimization/numerical_stability.md` ("fp32 accumulators in AGPRs"). Hand-written HIP / asm that
places the accumulator itself is not covered. Basis and scope in
`../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`, **Accumulator placement**.)

**Signal.** Registers pin occupancy (`probe.py measure`: registers bind, `next_free_vgpr` near `B`),
and either (a) the default split leaves the arch-VGPR half full while the VALU stage (epilogue, fold,
softmax) is what holds the live values — `.amdhsa_accum_offset 128` at 2 waves, `v_accvgpr_read` /
`v_accvgpr_write` inside the hot loop, or `.vgpr_spill_count > 0` — or (b) `"0"` itself spills.

**What the measured cases showed, by signal:**

0. **No signal.** Where the default split was already spill-free with no `v_accvgpr_*` in the hot loop,
   `"0"` and every N tied it (8-warp and 4-warp MLA prefill); the attribute bought nothing there.
1. **`"0"` is the cheapest probe.** Where it was spill-free and the hot loop had no `v_accvgpr_*`, every
   intermediate N measured tied it or lost (ping-pong GEMM, multi-wave logits kernels), and the default
   lost by the largest margin of any setting there (in-loop spills plus shuttling).
2. **If `"0"` spills, an N sweep is the next measurement.** A starting point is the AGPR-eligible demand: the per-lane registers of MFMA
   operands that stay live across the loop (a `[rows, K]` fp8 operand tile = `rows*K/256` registers
   per lane on a 1-wave tile), then step up by 8-16 to about twice that. The **smallest N with
   `.vgpr_spill_count: 0` and no `v_accvgpr_*` in the hot loop** was the keeper in the sweeps recorded, with a
   flat plateau above it and a loss once N pushed loop values into AGPRs (shuttling returns). This is the
   case where an intermediate N beat both `"0"` (spills) and the default (shuttling): a 1-warp
   attention-style kernel with a VALU epilogue and a loop-invariant fp8 Q operand.
3. **Per launch variant.** Siblings of one kernel (an extra MFMA per tile, a different epilogue) moved the
   spill-free threshold by one step, so each compile has its own N.

**Verify** each variant from the `.s`: `.amdhsa_accum_offset` (arch-VGPR region), `.agpr_count`,
`.vgpr_spill_count`, the `v_mfma` destination (`v[..]` vs `a[..]`), and `v_accvgpr_*` / `scratch_`
counts inside the hot loop (`asm_loop_audit.py` prints `accum_vgpr` and `next_free_vgpr`; its
opcode histogram is top-12 only, so grep the loop for `v_accvgpr`). Then time it.

**Hard limits and traps:**

- **`N >= B` failed where it was tried (gfx950, Triton 3.8.0).** Only `N = B` was run (`"256"` at 2 waves); larger N is expected to fail the same way
  but was not tried. `"256"` at 2 waves leaves zero arch VGPRs: LLVM prints *"no registers from
  class available to allocate"*, Triton still returns a binary (both re-checked by compile, 3.8.0 gfx950), and that binary **fails at launch**:
  a re-run on gfx950 / 3.8.0 (`num_warps=8`) raised a GPU memory fault (illegal memory access); an earlier run
  recorded `hipErrorLaunchOutOfResources`. Either failure after adding `amdgpu-agpr-alloc` points at `N >= B`.
  `"192"` at 2 waves compiles but spills heavily with hundreds of `v_accvgpr_*`.
- **Re-run full correctness after every N** — it reschedules the kernel even with no AGPR used, and can
  break inline-asm LDS fences with unchanged register counts.
- **8-warp `warp_pipeline_stage` ping-pong: `"0"` was the measured keeper.** Small N tied it only while the
  loop stayed `v_accvgpr`-free; larger N and the default lost on the kernels measured. Occupancy is not a
  register source there: an 8-warp CTA needs 2 waves per SIMD, so its budget stays 256 — an AGPR-forcing
  `N = 256` failed at launch (above; larger N not run) and a smaller N only re-splits the 256.
- `"256"` as an "AGPR accumulator hint" is the 1-wave/SIMD case only, where the default already
  grants a 256/256 split — ISA-diff it against no attribute before keeping it.

## Is it even a candidate? The ILP-starvation gate

The bound class read before the attribute explains a flat A/B better than one read after it. **The ILP
strategies paid where the loop was ILP-starved:** low occupancy (down to one resident wave) with independent
work the default scheduler left unscheduled — a reading of the bound class
(`../hardware/bound-class-signals.md`), not a guess from the source; `iterative-minreg` paid by a different
route (occupancy, next section). They were neutral-to-negative in the recorded cases on a serial dependence
chain, on occupancy-rich (2+ waves) or unit-saturated kernels, and the default usually suited LDS-bound /
memory-clause loops — A/B before closing the direction on a new kernel. A strategy reorders within the loop
body; cross-iteration overlap comes from the staging (`pipeline.md`). Applied as a default-on lever it tied or
lost in the runs recorded. Support per build: `../hardware/capability-matrix.md`.

### And the adoption rule that surprises people: this is a per-kernel patch

Kernels of the same archetype, tile and author disagreed in the cases recorded; one launch in a dispatcher
got a strategy and the next did not. **Results did not generalize by workload, archetype or dtype:** the
answer is per compile (a producer and its epilogue separately, `[]` being a valid answer) and can change
after any tile-shape or loop-body change.

## What you can write on it

Not a closed list (no allow-list), but anything outside it is yours to justify and verify.

| class | attribute | what it changes | signal that points at it |
| --- | --- | --- | --- |
| ILP strategy selector | `amdgpu-sched-strategy=<s>` | which machine-scheduling strategy runs (honoured names: next section) | the ILP gate above passes |
| co-execution strategy | `amdgpu-sched-strategy=coexec` | **gfx1250 only.** On gfx950/gfx942 LLVM 3.8.0 warns "'amdgpu-sched-strategy'='coexec' is only supported for gfx1250" and does not run the co-execution strategy (the ISA still differs from the default, so an ISA diff is not evidence it worked) | none on gfx950 / gfx942 (no co-execution effect there on 3.8.0); `compiler-contract.md ## What upstream 3.8.0 actually gives you` |
| IEEE-mode control | `amdgpu-ieee=false` | min/max NaN semantics, which is what blocks a native DPP reduction form and a single-instruction ReLU | the kernel's job is a max/min **selection**, or a ReLU / clamp in a VALU-bound epilogue, and you can bound the NaN behaviour |
| target features | `target-features=-<feature>` | turns a subtarget feature off for this function (e.g. suppressing packed-FP32 forming); replaces the whole string | a specific instruction form must *not* be formed |
| register budget | `amdgpu-num-vgpr=<n>` | a **cap** on the vector budget: it binds only when below the kernel's own demand and never raises occupancy above what the kernel already gets. On gfx950 / 3.8.0 (GPU-measured occupancy) the resulting budget was about `2n` registers, not `n` - the allocator added AGPRs (as spill space or accumulators) on top of the `n` arch VGPRs - so read `.amdhsa_next_free_vgpr` and the occupancy after compiling | you must hold a kernel under a register count; to pin a wave count use `waves_per_eu` (which also caps a lighter kernel at that count) |
| VGPR/AGPR split | `amdgpu-agpr-alloc` = `min[,max]` (pair form) | splits the per-wave budget (`### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`); `amdgpu-mfma-vgpr-form` is an `llc` option, inert here | an MFMA body that spills or shuttles `v_accvgpr` under the default split |
| generic LLVM attributes | e.g. `noinline` | whatever LLVM does with it | rarely |

### Scheduler strategy — which names the 3.8.0 LLVM honours on gfx950

LLVM is a prebuilt linked into `libtriton`, so the strategy enum is not in the Python source; it
was settled by CPU-only AOT compiles for gfx950 (an MFMA GEMM, 4 and 8 warps, with and without
`warp_pipeline_stage`), comparing the ISA against the default:

| value | ISA vs default on 3.8.0 / gfx950 |
| --- | --- |
| `iterative-ilp`, `max-ilp`, `iterative-minreg`, `max-memory-clause`, `iterative-maxocc` | **changes** (different schedule and register counts) — honoured |
| `coexec` | ISA changes, but LLVM warns it is **only supported for gfx1250** — not the co-execution scheduler on gfx950 |
| anything else (`max-occupancy`, `iterative-occupancy`, a typo) | not a strategy name in this LLVM; identical to the default (the max-occupancy scheduler) — silently ignored |

Honoured is not the same as faster: on the same GEMM `iterative-minreg` cut registers the most and
`iterative-ilp` was the only one that removed spills when combined with `amdgpu-agpr-alloc=0`.
`iterative-minreg` is also the **occupancy route** for a VALU-epilogue MFMA body that sits just
above a wave budget: with `amdgpu-agpr-alloc=0` and `waves_per_eu=4` it fit a DSA-indexer body that
needs 176 VGPRs at the default scheduler into 124-127 arch VGPRs, 0 scratch, and the extra waves
paid — but only once the fold was scalar (`../workloads/attention.md #### DSA indexer: occupancy
needs the scalar fold and iterative-minreg`). A bare `waves_per_eu=3/4` on the same body spills.
Re-run the comparison after any toolchain change; the acceptance signal stays the assembly diff.

**A valid strategy can still fail to compile on some operand signatures** (reported, not reproduced
here, toolchain unpinned: compiles on contiguous aligned operands, fails on mixed-stride or
unaligned ones). The worked example below keeps the attribute inside the branch that proved the operand
signature and the generic arm on `[]`; a compile failure right after adding a strategy points at the
predicate.

### IEEE mode — a numerics change, filed as one

`amdgpu-ieee=false` relaxes IEEE mode so `max` / `min` may lower to a native DPP reduction (compile + run checked
on gfx950 / 3.8.0: a row-max reduction dropped the per-operand canonicalize and fused the DPP lane shuffle into the
max, i.e. fewer VALU ops). It fits a
kernel whose job is a max/min **selection** (top-k, argmax, router) or the ReLU / clamp below; scoped to
that one compile, with what is preserved written down (finite / infinity / quiet-NaN behaviour, denormals,
the selected index). It is the only class in the table above that changes arithmetic, so it is a numerics
change like `enable_fp_fusion` as well as a performance knob.

**ReLU in an MFMA epilogue (the other legitimate use).** `gl.maximum(x, 0)` with NaN propagation
lowers to VOP3 `v_maximum3_f32`; a plain `maxnum` under IEEE mode gets a canonicalizing `v_max` per
element (two ops). With `amdgpu-ieee=false` the plain `maxnum` is one VOP2 `v_max_f32` per element
and a NaN score becomes 0 instead of NaN — write that down. Small and body-dependent (it paid on one
DSA-indexer body, a tie on another); A/B it, and check the opcode swap in the `.s`. Not an
inline-asm `v_max_f32`: the hazard recognizer does not pad an asm consumer of the MFMA result, so it reads
stale values silently (an asm consumer needs its own `s_nop`; if aiter is available, see the
workaround comment in `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla_fp8.py`).

## A worked example — one dispatcher, two kernels, two answers

```python
stage_sched = ([["amdgpu-sched-strategy", "iterative-ilp"]]          # INSIDE the proven branch
               if x.stride(-1) == 1 and w.stride(-1) == 1 else [])   # [] = backend's choice
stage_operands[(cdiv(M, BLOCK_M),)](x, w, acc, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                    num_warps=4, llvm_fn_attrs=stage_sched)
reduce_rows[(rows,)](acc, out_idx, BLOCK_M=BLOCK_M, num_warps=1,     # a selection kernel
                     llvm_fn_attrs=[["amdgpu-ieee", "false"],
                                    ["amdgpu-sched-strategy", "max-ilp"]])
```

Failures this shape prevents: the strategy hoisted above the stride test (the ragged arm stops
compiling, and the error does not name the attribute); `[["amdgpu-waves-per-eu", "4"]]` next to
`waves_per_eu=2` (your pair wins, the launch option is gone); the IEEE pair copied onto a kernel that
does no selection or ReLU, unrecorded (silent numerics change); the typo `amdgpu-sched-stratgy` (byte-identical asm,
flat A/B, recorded as "not ILP-starved").

## Verifying it took effect — the assembly diff is the only signal

| what is wrong | what happens |
| --- | --- |
| the option name `llvm_fn_attrs` itself (absent on 3.6.0 / 3.7.0 / 3.7.1) | **raises** — an unrecognized compile option |
| an attribute name or value LLVM does not recognize | **silently ignored** — indistinguishable from "applied, no difference" |

Compile twice, identical except the one attribute, and diff the assembly — a timing A/B cannot tell
"ignored" from "honoured but neutral".

- **No diff -> it did not take, or the value equals what the backend already chose** (e.g. a split equal
  to the default). Spelling, value and version are the usual suspects; a timing A/B or a bound-class
  conclusion from it says nothing.
- **A diff -> it took**; now time it and read the hot-loop instruction mix
  (`instruction-scheduling.md ## Acceptance`).
- **Diff the whole function** for register and target-feature classes: caps and suppressed forms
  show in the prologue and the reported register counts.
- **Results are toolchain-pinned:** record the toolchain identity with any kept result and re-diff
  after a toolchain change. Each swept value is one compile and one cache entry.

## Which versions have it

**3.8.0 only** (absent from the 3.4.0–3.7.1 AMD `compiler.py`, read from source); on 3.6.0 / 3.7.0 / 3.7.1
passing it raises (no no-op fallback). Older-build
substitutes by goal: `schedule_hint` presets (a dead declaration on 3.8.0;
`compiler-contract.md ## Portable scheduler co-design (the env knobs, and where the real lever lives)`),
or hand-staged overlap (`pipeline.md`) and the inline-asm mechanisms (`instruction-scheduling.md`).
`scripts/probe_levers.py --all` reports `version_disjoint_knobs` (`live` / `absent` /
`dead-declaration`): a flat sweep on 3.7.1 means "no such knob", not "not ILP-starved".
