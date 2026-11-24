---
id: gluon_authoring
kind: expert_skill_playbook
# must match skill.md `revision` (validate_skill.py)
revision: "2026-11-21"
entry_for: kernel_workflow roles when use_expert_skills is on (see roles/_fragments/expert_skills.md)
---

# gluon_authoring — playbook (mechanism menu)

What paid on gfx950, the mechanism menu and the checked pitfalls. Target **gfx950 (CDNA4, MI350X/MI355X),
Triton 3.8.0**; gfx942 differences inline. Advisory: your plan, the frozen COMMANDMENT and GEAK's A/B decide.
**Any symptom, error string or question not answered here: [`reference.md`](reference.md)** (entry index);
flow and tools: [`skill.md`](skill.md).

## Skill levers add to the broad levers

The levers below add to, not replace, the broad ones. In the recorded runs, tile shape / grid fill
(including non-pow2 and small-`BLOCK_M` exact-fill tiles and a re-tiled last wave), config retunes and
algorithm changes produced wins that the skill levers did not, so they stay in play alongside these. A
skill lever that measures flat closes that recipe on that kernel, not its direction. Fine-scheduling
levers (`s_setprio`, waits, fences) measured small in the cases tried, except where the loop audit
showed that exact stall.

## Skill levers on gfx950 (signal-gated)

Numbered for reference, not a try-first order: a lever is a candidate when its *signal* is in the evidence.

1. **8-warp warp-pipeline ping-pong + `amdgpu-agpr-alloc=0`** — *signal:* a large compute-bound
   tile (prefill GEMM) whose loop serializes LDS/VALU work and MFMA within a wave, or a 4-warp tile
   that cannot grow without spilling. *Precondition seen in the wins:* the tile count filled at least
   about one full wave at 1 WG/CU; below that, an exact-fill tile is the other candidate to A/B.
   *Recipe:* `num_warps=8` (two 4-warp groups; every layout's `warps_per_cta` must be rewritten for 8 —
   a layout sized for 4 warps fails to compile at 8 on Triton 3.8.0 / gfx950), an
   authored `async_copy` ring, and in the loop `with gl.amd.warp_pipeline_stage("mem", priority=1)` (LDS reads + next copies + address math)
   then `("mfma", priority=0)`, `wait_group` on the `mem` -> `mfma` border, plus one wait before the loop
   for the first tile — not at the top of the loop before `mem`: there the leading group reads LDS the
   other group's copies have not finished writing (wrong results that compile cleanly;
   [`warp-pipeline.md ## LDS hazards close a stage early`](references/tile-programming/warp-pipeline.md#lds-hazards-close-a-stage-early)).
   The marker path is GPU-verified race-free on Triton 3.8.0 / 3.9.0 (gfx950); on 3.7.1 it is not. This polarity is the shipped
   Gluon convention (AMD gfx950 Gluon tutorials, aiter); on a gfx950 / Triton 3.8.0 two-stage GEMM the
   inverted and equal polarities measured the same as it, so polarity is a low-priority knob there — the
   markers and `amdgpu-agpr-alloc=0` were what mattered (without the attribute an odd-K variant spilled and
   lost). The KB asm card (`perf_knowledge/languages/asm_mfma/raw_asm.md`, `s_setprio`) raises priority for
   the MFMA burst in hand-scheduled HIP/asm loops, and the four-stage attention loop (#6) measured
   matrix-high ahead only with P's `exp2` pinned in the PV stage; without the pin the polarities tied
   (gfx950 / 3.8.0). The 8-warp body halves each
   wave's register budget and tends to spill under the default VGPR/AGPR split;
   `llvm_fn_attrs=[["amdgpu-agpr-alloc", "0"]]` (at 2 waves/SIMD this gives all 256 registers to arch
   VGPRs, same ISA as `"0,0"`). *Check:* `.agpr_count: 0`, no `v_accvgpr_*` in the loop, spills below
   the default's (reaching `.vgpr_spill_count: 0` may also need `amdgpu-sched-strategy=iterative-ilp`, as
   in the snippet's table), and a race test before any timing — random inputs, NaN-filled output, many
   runs, bitwise against the same body without markers. Time it under the deployment's cache state: the
   marker gain measured on warm (L2/MALL-resident) operands disappeared when every launch started from
   flushed caches on one shape. *A/B it on a finished tile:* coalesced epilogue (#4) in, `GROUP_SIZE_M`
   re-swept for its grid (a store cache modifier only with a measured reason — they lost or tied in the
   gfx950 GEMM epilogues measured, `memory-path.md ## Epilogue stores`), and a fractional last round of its
   1-WG/CU tiles closed by re-tiling the last wave, then stream-K
   (`gemm-small-m.md ## Stream-K and persistent grids at small M`)
   — measured with MFMA-layout stores or a 1.5-round grid it reads as a tie against a 4-warp
   2-WG/CU tile. `"0"` is what the ping-pong wins used; when `"0"` itself spills on a non-ping-pong
   MFMA kernel, an intermediate N is an option to A/B: [`llvm-fn-attrs.md ### AGPR budget`](references/tile-programming/llvm-fn-attrs.md#agpr-budget-amdgpu-agpr-allocn-intermediate-values). GPU-verified snippet (Triton 3.8.0 / 3.9.0; on 3.7.1 the marker path is not race-free):
   [`warp-pipeline.md ### A GPU-verified gfx950 snippet`](references/tile-programming/warp-pipeline.md#a-gpu-verified-gfx950-snippet-triton-380-and-390-num_warps8);
   attribute detail: [`llvm-fn-attrs.md`](references/tile-programming/llvm-fn-attrs.md).
2. **Scheduler strategy (`llvm_fn_attrs` `amdgpu-sched-strategy=...`), after the config is retuned** —
   *signal:* spills or `v_accvgpr_*` in the loop, a `waves_per_eu` / `amdgpu-agpr-alloc` change just
   made, or independent loads/MFMAs left unscheduled (ILP-starved). The strategy that paid differed per
   kernel, so one name tried does not close the knob: `iterative-minreg` enabled occupancy after a
   waves/AGPR change, `max-ilp` paid on 1-wave attention bodies, `iterative-ilp` on some GEMM tiles (it
   *spilled* on a sibling path). The choice is per kernel, per launch, judged on an ISA diff plus timing.
   Works for plain `@triton.jit` too; on a plain kernel a config retune is the cheaper comparison.
   Strategy names in this build (Triton 3.8.0): `iterative-ilp`, `max-ilp`, `iterative-minreg`,
   `max-memory-clause`, `iterative-maxocc`; an unknown name compiles silently to the default. `coexec`
   compiles on gfx950/gfx942 (Triton 3.8.0) with the LLVM warning "only supported for gfx1250".
3. **Per-shape kernels** — *signal:* one config serving decode and prefill; skinny-M grids
   launching far fewer workgroups than 256 CUs; a mid-M ridge. Separate decode (GEMV / split-K),
   mid and prefill kernels or configs can be dispatched by shape; a dispatch only helps if each bucket
   actually launches the kernel you think. Decode time is modelled as launch floor + busiest CU (pass
   `hw_budget.py --floor-us`). On the gfx950 decode GEMMs measured, split-K paid below ~half a wave of
   tiles with long K and lost from ~¾ of the CUs up; **between ½ and ¾ an exact-fill re-tile beat split-K and a
   minimal stream-K on the bf16 GEMMs measured (gfx950 / 3.8.0; fp8 / MXFP4 not measured in that band) — A/B
   both**. In those cases `M == 1` ran reduce-free, and under graph capture a separate reduce
   beat a fused last-arriver one:
   [`gemm-small-m.md ## Split-K decision rule`](references/workloads/gemm-small-m.md#split-k-decision-rule-small-m-gemm).
   [`gemm.md ## Small-M`](references/workloads/gemm.md),
   [`method/dispatch-table.md`](references/method/dispatch-table.md).
4. **Coalesced epilogue** — *signal:* output writes visible in a large-tile kernel; MFMA-layout
   stores are 8 B/lane. The mechanism: stage through the dead operand ring and store 16 B/lane
   row-major — it paid where the epilogue is a large share (short K) and tied at long K (gfx950 / 3.8.0); the
   store cache modifier is a separate A/B that lost or tied in the GEMM epilogues measured.
   [`memory-path.md ## Epilogue stores`](references/tile-programming/memory-path.md).
5. **Targeted inline asm** — *signal:* the ISA shows loads hoisted above the MFMAs and spilling → an
   empty tied-identity asm as a motion fence; single instructions (`v_rcp_f32`, `v_pk_fma_f32`). Gains
   were small in the cases tried, and a fence without its signal measured flat.
   [`inline-asm-reference.md`](references/gluon/inline-asm-reference.md).
6. **Attention: classify the scenario, then read the per-scenario signal → lever table** —
   *softmax attention (MHA/GQA prefill)*, levers measured by leave-one-out on one MLA/MHA prefill family
   (unordered; pick by signal): register budget (*signal:* a wide head spilling at 2 waves → 1
   wave/SIMD with AGPRs usable); every head slice (PE/rope) on the pipelined loop (*signal:* an extra MFMA
   chain gated off the loop); LPT grid, then split-KV with an in-kernel last-arriver merge (*signal:* an
   under-filled grid); at `num_warps=4` stage markers replaced by `wait_group` + one barrier (*signal:*
   extra `s_barrier` per border); tile per (causal, length) bucket. Each of these measured larger than
   the 8-warp four-stage ping-pong on that family. The ping-pong paid where an 8-warp tile already
   stayed, with matrix stages `priority=1`, memory `0` **and** an empty asm pin on P's `exp2` in the PV
   stage (no pin: flat). At 4 warps the stages have no second group to offset (no overlap by
   construction). Re-tiling a 4-warp kernel to 8 warps to reach it lost on a VALU-latency-bound
   scaled-fp8 body and on a GQA paged D64 prefill; A/B it before closing the direction elsewhere.
   *DSA indexer (MQA logits, no softmax):* not flash attention — the levers that paid there were grid
   fill, K reuse across rows, register diet, and a scalar (non-`v_pk`) head fold plus 3-4
   waves/SIMD via `iterative-minreg` + `agpr-alloc=0` (with the packed fold or the default scheduler
   the extra waves lost); the 8-warp ping-pong lost in the indexer runs recorded (H32/H64 prefill).
   *Sparse MLA / DSA indexer decode:* own tables (prefill on-chip bound; decode = split count for one
   residency round). Measured on Triton 3.8 / gfx950 on the families named.
   [`attention.md ## Prefill levers by signal`](references/workloads/attention.md#prefill-levers-by-signal-gfx950-triton-380-bf16-varlen-prefill),
   [`### DSA indexer`](references/workloads/attention.md#dsa-indexer-mqa-logits-not-flash-attention),
   [`### Sparse MLA`](references/workloads/attention.md#sparse-mla-dsa-measured-levers),
   [`### DSA indexer decode`](references/workloads/attention.md#dsa-indexer-decode-paged-mqa-logits-measured-levers).

## Mechanism menu

| mechanism | signal | constraints on gfx950 / Triton 3.8.0 → detail |
| --- | --- | --- |
| **Register prefetch** (tile k+1 while computing k) | MFMA idle behind load latency | needs no LDS; costs live registers → [`pipeline.md`](references/tile-programming/pipeline.md) |
| **Authored LDS ring** — `gl.amd.cdna4.async_copy` + `commit_group` / `wait_group(N)` | register-staged loads with full-drain waits | on Triton 3.8.0 / gfx950, per lane per instruction a 128- or 32-bit chunk, and the layout must cover the tile's contiguous dim exactly (repeating along rows lowers); otherwise lowering fails (compile-checked). Failures read as a missing op (`unrealized_conversion_cast`). Recompute `wait_group(N)` when prologue/unroll changes; a runtime `k % N` index is legal — unroll to literal indices when the ISA shows it pays. gfx942: 32-bit only → [`pipeline-reference.md`](references/gluon/pipeline-reference.md), [`authored-overlap.md`](references/gluon/pipeline/authored-overlap.md), [`async-ordering.md`](references/gluon/pipeline/async-ordering.md), [`memory-reference.md`](references/gluon/memory-reference.md) |
| **LDS layout** — swizzle, `PaddedSharedLayout`, `cdna4.compute_efficient_padded_shared_layout` | `SQ_LDS_BANK_CONFLICT`, `ds_read` stalls (64 banks, `ds_read_b64_tr`) | as an `async_copy` / `buffer_load_to_shared` destination (GPU-checked, 3.8.0 gfx950): a `SwizzledSharedLayout` with `vec` < the producer's `size_per_thread` and `max_phase > 1` compiles and returns **wrong data silently**; a `PaddedSharedLayout` lowers only with the producer derived from its `offset_bases` and a padding interval ≥ one wave's copy (64 lanes × per-lane bytes: 1 KiB at 128 bit, 256 B at 32 bit) — otherwise `unrealized_conversion_cast` (compile-verified 3.8.0, gfx950 and gfx942; exact by copy-back on gfx950) — a swizzled destination with `vec` ≥ `size_per_thread` is the checked-safe choice, and any other choice needs a copy-back check; fewer conflicts bought with added `s_barrier`s can still lose — time it → [`layout-recipes.md`](references/tile-programming/layout-recipes.md), [`smem-lds-reference.md`](references/gluon/smem-lds-reference.md) |
| **Registers / occupancy** | `scratch_*` in the loop, `v_accvgpr` shuttling, occupancy 1 | VGPR + AGPR share one budget; the default split varies by kernel (128/128 or 256/0 seen at 2 waves) — read `.agpr_count`; K/M-slice the accumulator, or set the split with `amdgpu-agpr-alloc=N` ([`### AGPR budget`](references/tile-programming/llvm-fn-attrs.md#agpr-budget-amdgpu-agpr-allocn-intermediate-values)) → [`slicing.md`](references/tile-programming/slicing.md) |
| **Matrix / low precision** — `mfma_scaled`, 16×16 vs 32×32, `scaled_upcast` | upcast-then-MFMA where a scaled MFMA exists | `mfma_scaled` formats `e2m1`/`e4m3`/`e5m2`; `cdna4.scaled_upcast` wants raw E8M0 (`int8`/`uint8`) already expanded to the output shape and layout; fp8 is OCP `float8e4nv` (gfx942: `float8e4b8`); `mfma_scaled` takes no bf16 operand; with driven scales `k_width=16` is the one value measured correct across the shapes and formats tried (others gave silently wrong results in some) (MXFP4 MoE: [`moe-lowprec.md`](references/workloads/moe-lowprec.md)) → [`low-precision.md`](references/tile-programming/low-precision.md), [`matrix-reference.md`](references/gluon/matrix-reference.md) |
| **Beyond the loop** — XCD-aware PID remap, L2 grouping, persistent grid | low L2 hit with reuse available; wave tail | 8 XCDs × 32 CUs; a remapped PID that is not used does nothing; a wave tail: re-tiling the last wave and stream-K / persistent are both candidates — re-tiling won the recorded prefill GEMM case, so A/B both ([`gemm-small-m.md ## Stream-K and persistent grids at small M`](references/workloads/gemm-small-m.md#stream-k-and-persistent-grids-at-small-m)) → [`gemm.md ## Beyond hot loop`](references/workloads/gemm.md) |

Entry index: [`reference.md`](reference.md); skeletons: [`gfx950-minimal-examples.md`](references/gluon/gfx950-minimal-examples.md).
Evidence before and after each change (registers, spills, waits, counters): the tools in
[`kernel_workflow/scripts/kernel_tools/CARD.md`](../../../../kernel_workflow/scripts/kernel_tools/CARD.md) —
`dump_ir.sh --all` → `probe.py measure` → `asm_loop_audit.py --diff`; before sweeping a knob, or on a flat one → `scripts/probe_levers.py --all --arch gfx950 --brief` (is it read by this build);
a new async/pipeline construct → compile-checked templates in `scripts/pipeline_examples_cdna4.py`.

## ISA and SKU facts — look them up, do not recall them

Offline, page-cited answers from the CDNA4 ISA and the SKU table
(`KT=kernel_workflow/scripts/kernel_tools`; no GPU; `--arch gfx950` answers every query offline; gfx942 has only
the instruction chapters, and its MFMA `facts`/`layout` need AMD's Matrix Instruction Calculator, so the tool
reports a `SCOPED TOOL-GAP` instead of guessing). `hazard`, `waits`, `modifiers`, `lds`, `mem`, `limits` print the
whole chapter (300–500 lines): pipe through `grep -i -A8 '<word>'` or use `search`. A fact that
decides a layout, a wait or a tile size comes from here, not from memory of an older part (CDNA4 LDS has
64 banks, not GCN's 32).

| Gluon decision | query |
| --- | --- |
| anything ISA-shaped, not sure where | `python3 $KT/amd_isa.py --arch gfx950 search "<words>"` |
| pick the MFMA (`instr_shape`, 16×16 vs 32×32, K per instr, cycles, operand VGPRs) | `… list`, `… facts <V_MFMA_…>` |
| hand-build or debug a `DotOperandLayout` / linear layout; wrong elements after a layout change | `… layout <V_MFMA_…> -A` (`-B`, `-D`), `… locate <V_MFMA_…> A <row> <k>` |
| `mfma_scaled` / scale operands, CBSZ/ABID/BLGP | `… modifiers`, `… facts <V_MFMA_SCALE_…>` |
| LDS swizzle / padding, bank conflicts, `ds_read_b64_tr` operand loads | `… lds`, `… search ds_read_b64_tr` |
| inline asm next to MFMAs: NOPs / independent instructions needed | `… hazard`, `… waits` |
| reading `s_waitcnt vmcnt/lgkmcnt` in the ISA, `wait_group(N)` vs what drains | `… waitcnt` |
| buffer addressing, OOB / `mask`-free loads, alignment | `… mem` |
| VGPR/AGPR/SGPR/LDS capacities and allocation granularity | `… limits` |
| grid sizing vs CUs, XCD remap (8 XCDs × 32 CUs), L2 per XCD, MALL, per-dtype MFMA peak | `python3 $KT/extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` (the part this run is on) — each fact with the decision it drives; `hw_budget.py --brief` then turns peaks + shapes into the floor and your multiple |

## Porting a plain Triton kernel

A port pays when a Gluon-only mechanism addresses the identified bottleneck; the plain kernel at its
tuned config is the comparison bar for every shape, and Gluon for some shape buckets is a legitimate
result. A faithful port loses plain's software pipeline (`num_stages` is ignored by Gluon — identical ISA,
compile-checked on Triton 3.8.0 / gfx950): on the GEMM port measured it landed on `plain@num_stages=1`, and
an authored `async_copy` ring in plain's compiled loop form brought it back to parity. The numbered steps,
each with its check: [`transcribe.md ## Port checklist`](references/method/transcribe.md#port-checklist-verified-on-triton-38--gfx950).
The traps that cost the most on this route (Triton 3.8 / gfx950):

- **Declared is not applied**: a translated body whose loads stay on the translator's scalar default
  layout was bit-exact and far slower in the port measured; signature `global_load_ubyte` in the hot loop. Giving every `gl.arange` a recovered
  layout removed it.
- **The dump's `COMPILED FORM` can differ from the source**: `tl.load(..., other=0.0)` compiled to a bare
  or mask-only buffer load; `other=` on a Gluon `buffer_load_to_shared` measured slower in the port tried.
- **Buffer offsets are in elements**, not bytes (fp8 operands hide a byte-offset bug; the f32 scales do not).
- **fp8 `tl.dot` → `mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`** on the recovered `[16, 16, 128]`
  MMA; the translator skeleton's default `[32, 32, 32]` fails with `no matching matrix core intrinsic`.
- **16-byte async copies need 16-byte aligned rows**: otherwise `LLVM Translation failed for operation:
  builtin.unrealized_conversion_cast`; a register-staged kernel or plain covers those shapes.

Errors and symptoms, verbatim: [`transcribe.md ## Port error strings and symptoms`](references/method/transcribe.md#port-error-strings-and-symptoms).

## Knobs & pitfalls

(Each checked against Triton 3.8.0 for gfx950.)

- **`num_stages` does nothing on a Gluon kernel** (Triton 3.8.0) — `gluon_to_ttgir` does not read it (identical ISA).
- **`wait_group(N)` counts commit groups and the barriers come with it** — it lowers to
  `llvm.amdgcn.wait.asyncmark(N)` and LLVM, not Triton, derives the `s_waitcnt vmcnt` (on a simple ring
  `N x copies per group`; GPU race tests on gfx950 found it exact whenever every copy was committed and the depth
  right — a missing commit or a wrong depth makes it silently too loose; check the ISA, see the link); the
  compiler emits an `s_barrier` right after every `wait_group` and before each refill, so a ring needs
  no `gl.barrier()`; `load_shared_relaxed` removes no barrier after a `wait_group` (it only adds alias
  info). [`pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`](references/tile-programming/pipeline.md#what-the-compiler-inserts-after-wait_group-barriers-and-load_shared_relaxed).
- **`warp_pipeline_stage` at `num_warps=4` gives no offset but is not free** — no thread is in the
  second group, yet each stage border still emits a cluster barrier (a four-stage attention body:
  ~4.5 `s_barrier` per tile). Replacing them with `wait_group` + one `gl.barrier()` per landed
  buffer won on one 4-warp attention loop (Triton 3.8 / gfx950); the `s_barrier` count per loop
  iteration is the check, then timing. `s_setprio` appears only when a stage
  sets `priority=`. A single stage is a compile error; a `wait_group` after another op inside a stage fails
  (*"barrier or wait op cannot appear inside a warp_pipeline_stage region"*) — as a stage's first op it
  compiles to the same ISA as between stages (same ISA, not same correctness: put the publish-wait on the `mem` -> `mfma` border, lever 1); `gl.barrier()` anywhere in the pipelined loop fails
  (*"unexpected op inside pipelined_for body"*).
- **Scheduling ops are not user-facing**: no `sched_barrier` / `sched_group_barrier` / `set_prio` in
  `gl.amd.cdna3/.cdna4` on Triton 3.8.0 — stubs on `ImportError` are dead code there. `gl.warp_specialize`
  errors on gfx950 and gfx942 (*"Warp specialization is only supported on gfx1250"*).
- **`llvm_fn_attrs` string form splits on commas** (`triton/backends/amd/compiler.py`
  `_parse_llvm_fn_attrs`) — `min,max` values need the pair list. `amdgpu-agpr-alloc="N"` caps arch VGPRs at
  `min(256, B - N)` of the per-wave budget `B` (256 at 2 waves/SIMD); `N >= B` compiles to a binary that
  fails at launch (`hipErrorLaunchOutOfResources` or a memory fault) — [`### AGPR budget`](references/tile-programming/llvm-fn-attrs.md#agpr-budget-amdgpu-agpr-allocn-intermediate-values). Unknown names/values compile and do
  nothing; your attribute replaces the backend's own (e.g. `amdgpu-waves-per-eu`). On gfx950 the
  backend sets no scheduler strategy (`coexec` is for gfx1250; on gfx950/gfx942 LLVM ignores it with a warning).
- **Inert env knobs**: `TRITON_GLUON_SWP_PIPELINE`, `TRITON_GLUON_COOP_LDS`, `TRITON_GLUON_PINGPONG`,
  `TRITON_ENABLE_LLIR_SCHED`, `TRITON_ENABLE_AMDGCN_AS`, `TRITON_ENABLE_AMDGPU_RA_HINTS`,
  `TRITON_HIP_USE_COEXEC_SCHEDULER` appear nowhere in the Triton 3.8.0 Python sources or `libtriton.so`;
  a null result with them is no verdict. **Overrides KB** for upstream 3.8.0 (string search of the installed
  tree): the "Scheduling passes" lines in `perf_knowledge/operators/{dense_gemm,scaled_quant_gemm,quant_fp4_mxfp}/backends/gluon.md`
  and the env-var lines of `perf_knowledge/languages/gluon/{overview,programming_model,gemm_cookbook}.md`
  (`references/tile-programming/compiler-contract.md ## What upstream 3.8.0 actually gives you`).
- **LDS is 160 KiB per CU** (gfx942: 64 KiB); on Triton 3.8.0 an over-cap kernel compiles and fails at
  first launch. Deeper rings regressed in the cases where waves were already VGPR/LDS-capped.
- **Copying plain's `waves_per_eu`** can cap occupancy for a body it was not tuned for. Two variants
  measured equal may share a binary (stale cache, silent fallback); a binary diff settles it. A
  tolerance check written as `max(err) > tol` passes on NaN (NaN compares false) — scan for
  non-finite outputs. More: [`negative-patterns.md`](references/pitfalls/negative-patterns.md),
  [`platform-known-issues.md`](references/pitfalls/platform-known-issues.md).

## Do-no-harm

- **Correctness first** on every serve and edge case; confirm the Gluon kernel actually launched
  (a silent fallback to the plain kernel gives a perfect "Gluon" number).
- **Per case, not just the aggregate**: no serve case may regress beyond noise, even when the
  weighted metric improves — route that shape back to the incumbent kernel.
- **Disbelieve a fast number as hard as a slow one**: re-measure interleaved with the baseline on
  the same device; inside the spread is no result. Nothing carries to gfx942 unchecked.
