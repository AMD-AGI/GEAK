# Workload: Attention / Fused (prefill / decode / extend)

Two matrix phases with a softmax between them (`S = Q K^T`, `O = softmax(S) V`) — not a repeated
GEMM. Before importing FA-4 or another published non-MMA trick, read the porting filter and AMD
validated strategies in `flash-attention-structural-insights.md` (full FA-4 catalogue:
`tile-programming-cutedsl`, B200 only).

## Logical tiles

**Layout target: the QK accumulator (`AMDMFMALayout`) feeds the PV operand (`DotOperandLayout`)
without a hot-loop `convert_layout`** (a convert there adds a register shuffle or an LDS round trip
per tile; read the `.s` to see which). When M is small (decode / MLA / indexer with heads on M, M ≤ one MFMA tile per
warp) keep M inside one warp (`warps_per_cta=[1, num_warps]`, split N/D); splitting M half-fills the MFMA
tile (`../gluon/matrix-reference.md ## Matrix-Family Details`). For prefill tiles (`BLOCK_M >= 32·num_warps`) put the
warps on M (`[num_warps, 1]`, transposed MFMA) so each wave owns whole rows and row max/sum stay intra-warp;
splitting N there makes every wave hold all rows and adds cross-wave reductions and barriers per tile
(GPU-measured both ways on gfx950 / Triton 3.8.0: the N/D split for decode-sized M, the M split for prefill).
With the transposed MFMA layout and the matching `k_width` (4 for bf16 P on 32x32) the QK accumulator
converts to the PV A operand in registers — `convert_layout(p, dotP, assert_trivial=True)` compiled and ran
on gfx950 / 3.8.0, and the loop's `v_permlane32_swap`s belonged to the row reductions; check with
`assert_trivial=True` rather than assuming a shuffle. The opposite face (large D split, too-small
`BLOCK_M`) and the under-amortized case: `../method/profile.md ## Rule: low MfmaUtil has two distinct tile-size causes`.

## Prefill levers by signal (gfx950, Triton 3.8.0, bf16 varlen prefill)

**Scope: softmax attention** — MHA/GQA prefill and MLA run as decompressed MHA, flash-style online
softmax, causal and varlen. The DSA/MQA **indexer** (scores → ReLU → weighted head sum, no softmax,
no PV) is a different kernel shape with its own list: `### DSA indexer (MQA logits): not flash attention`.
Split-KV decode: `## Decode (split-KV): a structure, not a config knob`.

Levers measured on one bf16 varlen prefill kernel family (D 128 + 64 PE, and D 256) on Triton 3.8 / gfx950
by leave-one-out ablation. On that kernel they touched disjoint code (config key, launcher, loop body) and
stacked. The table is unordered: pick rows by the signal in the profile. The evidence column is that
kernel family's result, not an order for another kernel.

| Signal | Lever | Evidence (that kernel family, gfx950 / Triton 3.8.0) |
| --- | --- | --- |
| spills on a wide head at 2 waves/SIMD | **register budget:** 1 wave/SIMD with AGPRs usable (no `"0,0"` AGPR cap) (AGPR card in `../tile-programming/llvm-fn-attrs.md`) | removing it cost the most of any lever there |
| an extra MFMA chain (rope/PE slice, padded head) gated to the one-tile loop at 4 warps | **every head slice on the pipelined loop:** the slice rides the authored K/V ring when the tile fits without scratch | routing it back off lost on the PE cases measured. Scope: with the barrier-only 4-warp loop (row below). aiter's gfx950 Gluon MHA (`aiter/ops/triton/_gluon_kernels/gfx950/attention/mha.py`, `PE_IN_PIPELINE = num_warps >= 8`) keeps PE off the pipeline at 4 warps because its stage-marker loop measured slower there; that holds for the marker version |
| varlen load imbalance; M-blocks under-fill the device (*check:* grid size vs CU count) | **grid order, then fill-gated split-KV:** a longest-first (LPT) compacted grid computed in-kernel from `cu_seqlens` (no host sync; gate: `## Scheduling (load-imbalanced grids)`); then a key-axis split merged in-kernel by the last arriver (`### In-kernel split merge on gfx950`) | on a full grid the split variants tried lost |
| `warp_pipeline_stage` at `num_warps=4`: one extra `s_barrier` per border (compile-checked at 4 and 8 warps) and several `lgkmcnt(0)` per tile (*check:* barriers per tile in `asm_loop_audit.py`) | **barrier-only 4-warp loop:** markers have no second group to offset at 4 warps; use `wait_group` + one `gl.barrier()` per landed buffer (its `lgkmcnt(0)` covers RAW on the landed tile and WAR on the slot the next copy overwrites); a 3-slot ring needs one barrier per tile | paid on the 4-warp PE tiles |
| one global tile across causal / length buckets | **tile by sequence regime:** route the tile per (causal, length) bucket | on long varlen causal a 128-row 4-warp lean tile beat the 256-row 8-warp four-stage tile (twice the workgroups, less diagonal waste; it beat the prioritized ping-pong too). The shipped aiter gfx950 `mha/DEFAULT.json` (`causal_wide_min_seqlen_k`) sends long causal to 256/8, which holds for its stage-marker kernel; with the barrier-only 4-warp loop and an LPT grid, 128/4 won on long causal as well — re-A/B the switch point after those levers |
| `SQ_LDS_BANK_CONFLICT` / `SQ_LDS_IDX_ACTIVE` nonzero on K reads with the backend's 1 KB padding interval | **LDS bank spread for K:** one 16 B pad per 4 K rows of 256 B separates only `BLOCK_N/4` rows (8 at `BLOCK_N=32`) while a gfx950 `ds_read_b128` phase reads 16 K rows, so rows that share an interval collide (2- or 4-way depending on which rows a phase reads; read the counter ratio); move the remaining in-interval row bits into the 16 B-slot bits (`PaddedSharedLayout` bases + a matching async-copy layout). Per-row padding (272 B pitch), or an xor swizzle, reached the same conflict floor in a gfx950 / 3.8.0 microbenchmark | conflicts dropped to near zero; the time effect was small once the other rows were in, larger before |
| an 8-warp tile is kept | **8-warp four-stage ping-pong** with a pin (priorities alone were near-neutral): `### 8-warp four-stage ping-pong: when it pays` below | a small gain where the 8-warp tile stayed; re-tiling a 4-warp kernel to 8 warps to reach it lost in the cases tried (A/B before closing it on a new kernel class) |

**Measured small on that kernel:** 3-slot ring vs 2-slot (mostly a tie, one shape gained); exp2
placement between clusters (tie-level); deferred row sum (tie — see the row-reduction routes).

**Often already present in a tuned gfx950 kernel (grep before proposing):** the V operand of PV read
with `ds_read_b64_tr_b16` (grep the `.s`), `qk_scale` folded into Q, lazy rescale behind a branch.
**Hygiene, not a lever:** under CUDA-graph capture a lazily allocated `torch.zeros` scratch becomes
a memset node replayed every launch — reusing the warm-up buffer removes it; it measured a tie on the
kernel tried, so it is unlikely to repay a round unless the memset node shows in the trace.

## Online softmax (the recurrence to preserve)

The recurrence (running `m`, `l`, `acc`), masks and split boundaries are the semantics every later
layout / MFMA / scheduling change has to preserve; a correct-but-slow transcription of them is the
usual anchor to A/B against.

- **Write the update as `acc = mfma(p, v, acc * alpha)`, not `acc * alpha + mfma(p, v, acc)`** —
  the latter double-counts `acc` and fails at >= 2 K-tiles
  (`../gluon/matrix-reference.md ## Accumulator + per-step rescale (online normalization)`).
- **`qk_scale` folded into Q at load** (`q = (q * qk_scale).to(dtype)` once, before the loop): removes
  the per-block `S * scale` and result-tile registers. The same idea in base 2:
  `exp2(qk*(scale*log2e) - lse*log2e)`, the `log2e` moved onto the per-row term (`../method/profile.md ## Reducing compute-class VALU`).
  *Check:* the per-element scale multiply is gone from the loop. A folded `lse` changes base — see
  `### The merge is a weighted sum, not an atomic add`.
- **Partial FMA polynomial `2^x`** (`../method/profile.md ### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)`):
  targets an exp-unit bound (*signal:* transcendental busy counter saturated, MFMA/VALU idle); without
  that signal it lost in every kernel tried (gfx950 / Triton 3.8.0: `v_exp_f32` issues on the same VALU
  port, and the polynomial's own instructions cost more than the exp they replace). That closes *that
  recipe* on such a profile, not softmax arithmetic. Separate options, each its own A/B with numerics on
  every case: a bit-construction (Schraudolph-style) `2^x` — scaled input added into the exponent field,
  no polynomial — on unmasked tiles when P is then quantized to fp8 (few mantissa bits survive), and
  only when exp is a large share of the VALU issue (it paid on one fp8-P kernel and tied on one that was
  not VALU-issue bound; on bf16 P it gave no gain and a much larger error — not there). Write its
  float-to-int step so it does not raise register pressure (on 3.8.0 a `.to(uint32)` / bitcast form
  spilled at the 256-VGPR budget where an inline-asm `v_cvt_u32_f32` did not). Also: P written straight
  as fp8 codes with the scale folded; a max-free softmax where the score range is bounded.

### Conditional rescaling (skip the per-block acc *= alpha)

FA-4 lazy rescale: skip `acc *= alpha` (and the max update) while `m_new - m_old <= 8` (base-2 exp);
the epilogue `O / l` renormalizes (worked example, if aiter is available: FlyDSL
`fmha_gfx950/op_softmax.py`). Where it paid (GPU-measured, gfx950 / Triton 3.8.0: a low-occupancy decode and a 2-wave
prefill), `acc *= alpha` sat on the PV MFMA dependency chain and the kernel was not memory-bound. It was neutral where the loop already schedules the rescale off that chain (a
four-stage pipeline with the rescale in a memory stage) and on memory-bound decode. Two mechanisms
(apply / do-not-apply checks: `flash-attention-structural-insights.md ## Conditional rescale skip`):

1. **Low-occupancy decode (latency):** the exposed rescale disappears and PV accumulates in place
   (`acc = mfma(p, v, acc)`; the rescale branch keeps `mfma(p, v, acc * alpha)`). On an
   occupancy-hidden multi-wave build this mechanism has little to remove; it also paid on a 2-wave
   prefill where `acc *= alpha` sat on the PV dependency chain — confirm waves/CU and A/B.
2. **Two-wave dense prefill over its co-execution budget:** the rescale is the largest per-tile item
   in one compute region; skipping it frees window capacity
   (`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`). Freed budget buys
   nothing until something spends it — attribute the two changes separately.

**Write the skip as a branch, not a select.** The branch form is one `gl.map_elementwise` per lane that
covers all of that lane's accumulator elements (`pack = acc.numel // (num_warps * 64)`; multiply only if
any alpha != 1). It lowers to an exec-masked block guarded by `s_cbranch_execz`: when no lane of the wave
needs the rescale the multiplies are skipped, so it is effectively wave-uniform at runtime — confirm
`s_cbranch_execz` around the `v_pk_mul` block in the `.s`. A per-element `if` (`pack=1`) is if-converted
by LLVM to `v_cndmask`, the same as `gl.where` (compile-checked gfx950 / 3.8.0): the multiplies issue on
every lane plus a select per element. An added cross-lane / CTA vote costs a reduce plus VGPRs. Where
measured (gfx950 / 3.8.0) the select form lost on a 2-wave prefill microkernel and the vote form was negative
on a 1-wave MLA forward; recipe and ISA check in
`../pitfalls/negative-patterns.md ## Conditional rescale skip — measured negative at the occupancy its own path A asks for`. For mechanism 2 the branch was placed
in a **memory** region: control flow in a compute region can be scheduled ahead of the first matrix
op (check the cluster order in the `.s`).

**Accumulator placement** (**Overrides KB**: `perf_knowledge/languages/triton_amd/deep_codegen.md`,
`perf_knowledge/optimization/mfma_scheduling.md` and `perf_knowledge/operators/attention_prefill_fmha/tuning.md`
place the MFMA accumulator in AGPRs; verified by a compile-only ISA check on gfx950 and gfx942). On gfx950 / Triton 3.8 the allocator kept the accumulator in arch VGPRs
in every non-spilling attention build surveyed; in spilling variants and very wide tiles MFMA destinations
moved to AGPRs (placement rule: `../tile-programming/llvm-fn-attrs.md ### AGPR budget`). Read the `v_mfma` destination; only an
`a[..]` accumulator pays a `v_accvgpr_read` round trip. `v_accvgpr_*` in the loop or a spilling VALU
epilogue: `../tile-programming/llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`.

## Usual bound class

- **prefill / extend:** compute-bound on QK+PV (MFMA continuity); LDS on the S buffer.
- **decode (split-KV):** latency/memory-bound; the structure around the split count is
  `## Decode (split-KV): a structure, not a config knob`. **At short served context the KV
  footprint can sit in the memory-side cache** — then HBM peak is not the denominator and the scratch
  account has no DRAM traffic to remove (`../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`).
  (plausible, not measured on a decode kernel here). Software prefetch regressed where it raised VGPRs at
  an occupancy cap, so waves/CU is the first number to read before adding it (`../tile-programming/slicing.md ## Occupancy budget (P8)`).

## Layer roadmap

1. **anchor:** explicit layouts for Q/K/V loads, S and both dots (transcribe).
2. **memory:** K/V `buffer_load` -> `buffer_load_to_shared` (gfx950). gfx942 downgrade: direct-to-LDS
   async lowers only at 32 bits per lane (compile-checked on Triton 3.8.0: 16/64/128-bit fail) and
   the downgrade notes use `order=[1,0]`, so K/V usually stage through registers
   (`buffer_load` + `ds_write`).
3. **LDS:** V of PV read with `ds_read_b64_tr_b16` on gfx950; gfx942 (`index.md` downgrade list):
   re-derive padding and the number of K/V stages that fit.
4. **softmax state:** one `DistributedLinearLayout` consistent across the running state.
5. **matrix:** QK/PV operand and accumulator layouts matched (`## Logical tiles`).
6. **authored K/V-loop overlap** (`../tile-programming/pipeline.md`; options, not a fixed order — the
   profile picks): register prefetch; an authored K/V ring (gfx950 `async_copy` + `commit_group` /
   `wait_group`; gfx942 sync staging); `warp_pipeline_stage` (`num_warps >= 8`; at 4 warps bare
   `wait_group` + barrier, `## Prefill levers by signal` barrier-only 4-warp loop row). Free loop-shape edit: **split the causal
   mask into two loops** (branch-free hot region). **Decode replaces step 6** with
   `## Decode (split-KV): a structure, not a config knob`.

**fp8 / fp4 attention** (e.g. INT8 Q/K + FP8 V + per-block descale): the descale is a side path
(`../tile-programming/low-precision.md`); OCP fp8 on gfx950, FNUZ on gfx942 (no scaled MFMA there).
`attention-lowprec.md` prices descale against the softmax budget.

### Staged QK <-> softmax overlap (when it pays off)

Two different forms, with different answers (GPU-measured, gfx950 / Triton 3.8.0):

- **Intra-wave FA3-style staging** (softmax of tile *i* under QK of *i+1* in one wave) holds *k* live
  tiles and lowers waves/CU; resident waves often supply that ILP already. The leaner non-staged kernel
  won on throughput-bound 2-wave prefill shapes, and staging paid on a critical-path-bound shape.
  *Signal for staging:* latency-bound at low occupancy (busy counter far from saturated) with the stage
  below the occupancy cliff (`../tile-programming/slicing.md ## Occupancy budget (P8)`).
- **Cross-wave four-stage pipeline** (`warp_pipeline_stage`, 8 warps, softmax of tile *j* beside the QK
  of *j+1* in the other wave) won over the leaner one-tile loop when both ran at the same waves/SIMD;
  buying a third wave for the lean loop spilled and lost. Compare at equal occupancy before choosing. The
online-softmax recurrence blocks only *same-phase* symmetric ping-pong; the warp-pipeline shape (two
waves a full stage apart, memory vs compute) is compatible with it and authored in Gluon
(`../tile-programming/warp-pipeline.md`); plain Triton 3.8.0 has no stage-marker API (checked by
introspection) and reaches that shape through the automatic ping-pong pass.

### 8-warp four-stage ping-pong: when it pays

Measured on softmax attention (on the DSA indexer it measured negative, below). Measured on a 256×32,
8-warp, bf16 PE tile already written as the four-stage `dot1 / mem1 / dot2 / mem2` loop (Triton 3.8 /
gfx950):

- **On that loop priorities alone were near-neutral (timing only); the pin carried the gain.** Without
  it LLVM sank dot2's P `exp2` burst past `mem2`'s lazy-rescale branch, leaving the PV stage MFMA-only.
  Pin P at the end of the `dot2` stage with an empty side-effecting asm
  (`gl.inline_asm_elementwise("", "=v,0", [p], …, is_pure=False)`). Trace-confirmed (ATT, gfx950 /
  Triton 3.8.0 / rocprofiler-sdk 1.3.5): without the pin nearly all of dot2's `v_exp` moved into `mem2`,
  `mem2` grew several-fold in cycles, the partner group waited hundreds of cycles at each memory-stage
  barrier, and far fewer exps issued in the wave's own MFMA shadow. The pin lost on VALU-bound fp8 and
  unified-attention kernels (timing) — A/B it per kernel.
- Matrix stages above memory (`dot*` 1, `mem*` 0, via a `warp_pipeline_stage` subclass with a constexpr
  priority) — the inverse of GEMM (`warp-pipeline.md ## Stage priority`); measure both polarities. The
  polarity result is tied to the pin: on the same four-stage loop without the pin, neither polarity
  changed time (gfx950 / 3.8.0).
- **Against the best non-ping-pong variant:** on long varlen causal the 128-row 4-warp lean tile
  still won (small margin), so there the ping-pong did not beat it. On non-causal long-context chunks,
  where the 256-row 8-warp tile stays, the prioritized+pinned ping-pong gave a small win and stacked
  with the structural levers of `## Prefill levers by signal`.
- **Measured negative builds** (each closes that build, not the ping-pong): at `num_warps=4` (no
  second group; co-resident 4-warp workgroups are not phase-locked — 4-warp `s_setprio` lost); a
  128-row tile split into 8 warps of 16 rows (every warp still reads the whole K/V tile: LDS reads per
  FLOP double; lost); a 1-wave/SIMD wide-head kernel (no partner wave; the 8-warp body spilled the D256
  accumulator). Converting a 4-warp kernel to 8 warps to meet the precondition lost on a scaled-fp8
  body bound by VALU latency and on a GQA paged D64 prefill; re-test if the 8-warp tile is
  spill-free and keeps two workgroups' worth of waves per CU.
- *Check:* `s_setprio` in the loop, `v_exp_f32` inside the dot2 MFMA cluster (audit symbol stream),
  spills unchanged.

**Gluon anchor schedule-bound against a plain anchor** (equal VGPR+AGPR and occupancy, lower
`MfmaUtil`, more full-drain `lgkmcnt(0)`): the gap is in the overlap (`## Layer roadmap` step 6).
Re-injecting plain's TTGIR pipeliner (`add_schedule_loops` + `add_pipeline`) is a diagnostic route —
its target is plain's own schedule, it is not designed to compose with hand staging in the same loop,
and on an incumbent Gluon kernel it replaces rather than extends the authored overlap: `../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`;
loop shape it needs: `flash-attention-structural-insights.md ### Software-pipeline reproduction — the kernel shape re-injection needs`.

### Where the softmax goes, and why it rides with the matrix op

**The softmax VALU in the compute stage with the MFMAs, the other wave supplying the memory
traffic,** is the placement that carried the gain on the measured four-stage loop. The mechanism: a
wave issues at most one instruction per cycle, so VALU and memory issue pair only across waves (CDNA
issue model). Softmax in the memory stage takes turns with the same wave's reads; a stage of its own
needs three phase-locked waves per SIMD, which the two-group warp pipeline does not provide. So attention is inter-wave between memory
and compute and intra-wave inside each compute cluster (`../tile-programming/scheduling-model.md`),
which is why the clusters have a co-execution budget
(`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).

**Known-exposed work before the cluster's first matrix op:** a packed f32 op (`v_pk_*_f32`) right after a matrix op
waits for the MFMA to finish (packed f16/bf16 co-issues like scalar VALU) and can
pay a hazard stall on top of being uncovered (`../hardware/isa-mechanisms.md`). Which work is exposed
is per kernel.

### What the authored path can express, and where it stops

Authorable on stock 3.8.0: which stages exist, their contents, which wave is in which stage (the
ping-pong). Not authorable: that a given VALU op lands in a given MFMA's shadow — the scheduler may
ignore or move it; say which half an estimate depends on. Author-side heuristics: control flow kept
out of compute clusters; clusters sharing a budget balanced; rebalancing with free moves (a register
view or slice — a copy adds the very work being moved); known-exposed work first; **a VALU op moved
together with its producer**, since an orphaned producer can be sunk anywhere. Many attention kernels use no stage markers and carry
the overlap with intra-wave levers (independent partial accumulators, operand-prefetch placement,
wave-local row reduction) — decide per kernel, not as a ladder with ping-pong on top.

### The row reduction has three routes, and they are mutually exclusive

The deciding quantity is how many waves the reduce axis spans (inside one wave: nothing to restructure). Routes:
(a) a layout keeping it wave-local (+1 `convert_layout`); (b) two-level — in-wave, then per-wave
scalars through LDS; (c) defer the cross-lane step to the epilogue (registers). All three occur in
production; the choice is shape-dependent.

Measured (gfx950 32×32 MFMA, `BLOCK_N=32`, one column per lane over 32 lanes, so the in-loop row
sum is all cross-lane): deferring it costs `BLOCK_M·BLOCK_N / (64·num_warps)` VGPRs (16 on a 128×32
4-warp tile); where the loop was already near the VGPR limit it spilled and tied. On a 1-wave/SIMD D256
kernel with serial VALU issue, taking the row sum through the matrix pipe paid.

## Decode (split-KV): a structure, not a config knob

One query row per head and a long KV: beyond batch x heads (and output-channel sharding, below) the
parallelism left is cutting the KV axis, and the cut
reaches slicing, memory path, epilogue and layout at once. Read before proposing a decode edit.

### The dimensions that exist

A map, not a ranking — the profile says which binds.

| Dimension | Decides |
| --- | --- |
| split count | KV-axis parallelism; the grid is built from it |
| split assignment | contiguous span vs cyclic interleave (more even under skewed lengths) |
| split pitch | scratch stride per split, padded to the merge's alignment |
| grid dimension order | split vs channel shard fastest: L2/XCD locality |
| record geometry | partial-record grouping (channel x head group): merge granularity, producer store coalescing |
| output-channel sharding | more programs, QK recomputed per shard — budget it |
| resident tail | last chunk stays in registers (`../tile-programming/slicing.md ## Budget to compute first`) |
| local checkpoints | first N partials in LDS instead of global scratch |
| guarded scalar loads | skip scale loads on a split that owns no work |

### The scratch round trip is the byte account decode adds

```text
scratch_roundtrip = 2 * split_count * record_bytes      # written once, read once by the merge
kv_bytes          = history_len * head_dim * dtype_bytes
```

**At short served lengths the scratch term can exceed the KV term.** More splits shorten the KV
walk and lengthen the scratch: price both at the served length distribution, not the max context.
Amplifiers: the producer writes full (zero) records for splits with no work; the merge reads every
record when masked by split count instead of live length. **The read-side fix** (mask the merge by
live length) carries no correctness risk; skipping inactive writes is safe only if the merge provably
never reads them, including a previous graph replay's residue — re-check per scratch layout.

### The merge is a weighted sum, not an atomic add

Deterministic rescaled merge (`exp(m_split - m_global)`), max + denominator packed per record. An
atomic add cannot apply the `m_global` rescale before every split has reported, and its result
depends on arrival order (`../tile-programming/memory-path.md ## Output / reduction path (atomics)`). With `log2e` folded the
record's `lse` is base 2: state the base next to the record layout and fold on both sides or neither.

### In-kernel split merge on gfx950

Prefill uses the same split structure when a launch under-fills the device (a context chunk with
few query blocks, a short causal batch). Parts of the design that paid (unordered; softmax attention,
verified on Triton 3.8 / gfx950 for the split as a whole; the parts are from the run that built it, and
the signal for the whole design is the fill check at the end of this list):

- **Last arriver merges, no second kernel.** Every chunk but one publishes a normalized 16-bit
  partial + fp32 base-2 LSE and bumps a per-tile counter; the merging chunk waits, folds the
  partials in registers, resets the counter (so the buffer is zero again for the next replay —
  graph-safe, no memset) and stores O. Producers were ordered before the merger in `pid` order, so
  the merger waits only on programs dispatched ahead of it. That relies on in-order workgroup
  dispatch, which an ordinary launch does not guarantee (no forward-progress guarantee without a
  cooperative launch: `collective.md`); a last-arriver ticket with no spin avoids the dependency.
  One dispatch instead of two.
- **Make the merging chunk short and dispatch it last**, with near-equal long chunks before it,
  chosen by a host-side simulation of dispatch order from shapes only (no host sync). It also
  halves the partials the merger reads.
- **gfx950 cross-workgroup visibility without flushing L2:** `.wt` stores, an inline
  `s_waitcnt vmcnt(0)`, `gl.barrier()`, a *relaxed* atomic, then `.cv` loads on the consumer.
  Compile-checked on Triton 3.8.0 / gfx950: a `gpu`-scope `acq_rel` atomic emits `buffer_wbl2 sc1`
  before and `buffer_inv sc1` after (they evict co-resident tiles' K/V; `cta` scope and `relaxed` emit
  neither); `gl.barrier()` after a plain store emits `s_barrier` with no `vmcnt` wait, so it does not
  wait for stores (`gl.store(.wt)` already adds its own trailing `vmcnt(0)` — costly in a store-heavy loop
  at low occupancy, GPU-timed — `buffer_store` `.wt` does not); `.cv` loads carry `sc0 sc1`. A relaxed atomic does not order the later partial loads (memory
  model) — add an explicit dependency.
- **Store partials from a row-major layout, whole 16 B row segments per lane** (full lines);
  write-through on scattered 8 B MFMA-layout stores measured slower.
- **Publish with a release, not an `acq_rel` RMW per tile**; a heavier variant of the merge lost, so
  A/B the in-kernel merge against a separate merge kernel on the target.
- **Gate on fill.** On a full grid every split variant tried lost on the measured prefill kernel (each
  piece adds a fixed prologue, a partial store and a merge); the deciding signal is the fill check,
  not the split itself.
- **Decode with a short per-split walk measured the other way.** On sparse-MLA decode (one or a few
  key tiles per split, a merge reading many splits) the last-arriver merge lost on every path tried
  against a separate, wide merge kernel: the merging program's serial tail (several dependent memory
  round trips at low occupancy) is longer than the second graph node it removes. The prefill win came
  from long producer chunks that hide the tail; whether the target has them decides which result
  transfers (`### Sparse MLA (DSA): measured levers`).

### LDS: stage once, then re-view

Decode bodies are often LDS-capped. Re-view one allocation for a second consumer (a transposed
*view* costs no traffic; a transposed *copy* is a full round trip), and allocate reusable planes (a
P staging plane) once outside the loop (spelling: `../gluon/memory-reference.md`).

### Two places where the usual advice inverts

- **At one workgroup per CU, extra LDS costs no wave** while it still fits the CU's LDS
  (`../tile-programming/slicing.md ## Occupancy budget (P8)`) — an LDS-hungry option carries no
  occupancy penalty there.
- **Streaming split-KV partials past L2 measured negative:** the merge reads them immediately on the
  same device. On sparse-MLA decode with a separate merge kernel (Triton 3.8 / gfx950), `.wt` and `.cs`
  partial stores both lost against the default store, and so did `.cg` on the merge's partial loads.
  Re-test if the merge runs much later or the partials exceed L2.

### Grid fill when the body is LDS-capped

At one workgroup per CU the grid is the occupancy (`../hardware/roofline-models.md ## Saturation / wave quantization`):
size `rows x head_blocks x splits` to about one resident round; past it lost in the cases measured.
Where a split walks one tile, a few more programs per CU can still pay — measure both sides of the round.

## Backward pass (general)

`dk`, `dv` reduce over queries, `dq` over keys — transposed parallel axes, no single clean grid
(`../tile-programming/mental-model.md ## Reduction & parallelization structure (decide before tiles)`).
**Split** (separate `dq` and `dk/dv` kernels, deterministic, `S` recomputed in both) vs **fused**
(`S` once, `dq` atomic-added: write amplification = #key tiles; packed bf16 atomics narrow it,
`../tile-programming/memory-path.md ## Output / reduction path (atomics)`). **Neither is the fast path in
general on gfx950**: on one Triton/Gluon MHA-bwd case (MI350) the deterministic split beat fused
atomic-`dq` clearly, while AMD's asm `fmha_bwd` v3 defaults to fp32 atomic `dq` — A/B both against the
CK/asm reference. The FlyDSL backward skill
(`perf_knowledge/expert_skills/skills/flydsl_fused_attention_backward/skill.md`) fuses all backward
GEMMs with atomic `dq`; that holds for its FlyDSL kernel, not as a Gluon default. In the recorded case,
adding key-splits to a deterministic `dq` increased atomics without a gain. The CK reference gives the
ceiling to compare against.

## Scheduling (load-imbalanced grids)

Causal / varlen grids are load-imbalanced; a longest-first (LPT) `pid -> worktile` remap needs no
body change, and its measured gains were largest on causal (makespan rule: `gemm.md ## Worktile scheduling for load imbalance`):

- **Causal fwd/bwd:** reverse `mblocks` within each `(head-section, batch)`; batch axis outermost;
  heads swizzled into L2-capacity-sized sections.
- **Deterministic `dQ` with lock-ordered atomics:** SPT order (KV blocks descending, query blocks
  ascending from the diagonal) shortens waits but is **not** a forward-progress guarantee between
  blocks of an ordinary launch.
- **Varlen:** a cached preprocessing sort writes a virtual->actual batch map — worth it past the gate below.
- **Conflicts with an L2 `pid` swizzle:** `flash-attention-structural-insights.md ## LPT causal`.

**LPT uniformity gate (varlen).** LPT assumes worktile cost monotone along the reordered axis (causal yes;
varlen only with uniform packing). On mixed prefill/decode, few-long-among-many-short, or pre-balanced
packing it can be neutral or negative — A/B three-way (LPT / swizzle / both) on the served packing;
accept on the makespan spread, not the mean.

### `waves_per_eu` on attention dispatches

Per-seqlen-bucket values (sweep incl. `0`); confirm occupancy as `min(register term, LDS term)` (`kernel_workflow/scripts/kernel_tools/probe.py measure --dir <ir>` or `amd_occupancy.py`; `../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`), not from the hint or the `.s` `; Occupancy:` line — that line has no LDS term and overstated occupancy on dynamic-LDS kernels (gfx950 / Triton 3.8.0).

## Applying the framework to attention variants

Read a new variant (MLA, Sage, DSA, paged) through the same variables, not a per-variant recipe:

- **Matmuls + reductions and their axes** -> structure, where atomics / recompute land.
- **VALU between matmuls** (softmax / gating / smoothing / dequant) moves the problem outside
  GEMM-only compiler answers: **the GEMM throughput-pairing interleave is reported to assert on
  VALU-between-matmul, and the AGPR accumulator hint regressed** (compiler-contract page; re-check on
  the toolchain in use). `amdgpu-sched-strategy` (`../tile-programming/llvm-fn-attrs.md`) still
  applies to ILP-starved kernels (A/B); the other route is a hand interleave
  (`../tile-programming/compiler-contract.md ## VALU-between-matmul: manual interleave, or author a pass`).
- **Co-execution budget** (whether that VALU fits the MFMA shadow): per compute region
  `capacity = (MFMA in region) x window` vs `demand = sum of class issue costs`; `demand > capacity`
  means shrink or move work. Preconditions for the budget to apply: (a) two resident waves per SIMD; (b) matrix-issue bound
  (not a latency/bandwidth-bound decode). Arithmetic:
  `../tile-programming/llir-codesign.md ## Attention: the co-execution budget`; per-shape window:
  `../hardware/isa-mechanisms.md`.
- **Operand / latent footprint** (MLA, compressed KV) -> LDS / register budget.
- **Precision side paths** -> `attention-lowprec.md`.
- **KV access** (sparse / paged / indirect) -> gather address chain, possibly the parallel structure.
- **Static vs data-dependent loop body** -> next section; it can invalidate the others.

Measured DSA levers: `### Sparse MLA (DSA): measured levers`, `### DSA indexer decode (paged MQA logits): measured levers`.

### The dynamic-body variable (sparse / paged / varlen)

- **The address chain is VALU and sits naturally in the memory region** with the loads it addresses;
  matrix + VALU + memory in one region is the combination the region model describes as skipped
  rather than scheduled
  (`../tile-programming/llir-codesign.md ## Region routing: the model is a property of the region`).
- **A data-dependent trip count removes the static unroll** that restores swapped loop-carried
  buffers; the guarded remainder is a different region structure the main-body budget does not describe.
- **The budget becomes a distribution:** balance for the common case, check the tail on the worktile
  duration spread (`## Scheduling (load-imbalanced grids)`).

A budget computed on a minimal dense body does not describe a dynamic body; the real kernel's lowered
code is the input (`../gluon/pipeline-reference.md`).

### Substituting into the co-execution variable (head dim, MLA)

- **Head dim `D`:** row-wise reduction tails are D-independent, per-element terms scale with the
  score tile's N, the rescale with `D`. The balance ratio is per `(D, block-N, dtype)` and does not
  carry across them; the window comes from the MFMA the region actually contains.
- **MLA:** a large latent dim raises accumulator registers and can drop occupancy to 1 wave/SIMD,
  failing precondition (a); the read-modify accumulator also puts it outside the GEMM
  `compiler_interleave` ladder (reported to assert on VALU-between-matmul) — `../tile-programming/scheduling-model.md ## When steps 1 and 2 both say no — the neither-model cell`.
  The budget depends on occupancy and bound class (sparse-MLA decode measured: compute a small slice,
  the rest gather chain, merge node and launch).
- **Beyond attention:** any fused loop with matrix + memory + a third category (MoE gating, dequant
  unpack, fused norm) is the same allocation; dequant VALU can also be *hidden* in the MFMA shadow if
  it is not a packed op and the budget admits it (`../tile-programming/low-precision.md`).

### DSA indexer (MQA logits): not flash attention

`logits[i, j] = Σ_h w[i, h] · relu(q[i, h] · k[j] · s[j])` over a key window: **one** matmul (heads
on M, fp8), a VALU fold over heads, and a large f32 logits write. There is no softmax recurrence,
no PV and no V operand (a key split owns disjoint logits columns): a skinny GEMM + cross-head
reduction + streaming store. Levers that paid on gfx950 / Triton 3.8 (prefill, the H32/H64 indexer bodies
measured); unordered, pick by signal:

| Signal | Lever | Evidence / check |
| --- | --- | --- |
| grid under-filled or imbalanced per shape | **per-shape grid fill:** wave-quantized key split, longest-first | paid on the bodies measured |
| K re-read per query row | **K reuse across query rows:** two rows per wave, then one LDS K tile shared by row-partitioned warps | paid on the bodies measured |
| small body with exposed load latency | **a software-pipelined small body** (1 warp, short key tile) with an explicit LDS reader drain — on Triton 3.8.0 / gfx950 a 1-warp `gl.barrier()` emits no `lgkmcnt` wait before a direct-to-LDS async refill of the same buffer (compile-checked, and GPU-confirmed on gfx950: stale reads at `num_warps=1`, none at 4 warps; a `ds_write` refill is ordered anyway), so the WAR fence is a correctness requirement there | paid on the bodies measured |
| epilogue spill; Q operand pack wide in registers | **register diet:** remove the epilogue spill; pin the Q operand pack | halved registers on the 64-head path |
| 1-2 waves/SIMD on the heavy head counts after the above | **scalar fold, then 3-4 waves/SIMD through the scheduler** — `#### DSA indexer: occupancy needs the scalar fold and iterative-minreg` below | mid-size on the heavy 32-head shapes; the same occupancy with the packed fold lost |

*Measured small (no kept gain in the runs that tried them, after the levers above were in):* exact waits —
prologue issue order (weights, K, scales, then Q) and a scale prefetch so `vmcnt` waits are exact —
with `.wt` / paired logits stores; a single-group `s_setprio` window around the fold (not the
warp-pipeline ping-pong). Both act on the body: their result changes with any structural change, and
they can turn negative on top of it.

**Measured negative on these indexer prefill bodies, in several independent runs** (re-test if the
fold stage stops being a serial VALU chain or the body changes shape): the 8-warp
`warp_pipeline_stage` ping-pong (lost in each run that tried it: the fold stage is a serial VALU chain
with one wave per SIMD in it, so the lockstep barrier waits; 8-warp workgroups also cost one
workgroup per CU), in-wave accumulator ping-pong (spills / VALU is the longer pipe), 3 waves/SIMD
**at the default scheduler** (spills or `v_accvgpr` shuttles; not the minreg route of `#### DSA indexer: occupancy needs the scalar fold and iterative-minreg`), folding
head weights through an MFMA (numerics or MFMA cost).

#### DSA indexer: occupancy needs the scalar fold and iterative-minreg

The heavy body is VALU-bound (ReLU + weighted head fold per score element) and sits above the
3-wave budget at the default VGPR/AGPR split. Measured: the two changes together moved it; the
scalar fold alone tied, and the occupancy step with the packed fold kept lost:

- **Fold in scalar f32 FMA chains rather than `v_pk_fma_f32`.** Packed f32 is recorded as not
  co-issuing with a matrix op (trace-measured table in
  `../hardware/isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget`; not in
  the ISA document), and on Triton 3.8.0 / gfx950 the SLP vectorizer packs adjacent FMA pairs into
  `v_pk_fma_f32` by default (compile-checked; if aiter is available, its gfx950
  `attention/sparse_mla_fp8.py` header says the same). The scalar form is one serial chain per output
  column (`v_fmac_f32`), or an inline-asm `v_fma_f32` / `v_fmac_f32` whose inputs are VALU results
  (the ReLU). An asm read of a raw MFMA result is unsafe: on Triton 3.8.0 / gfx950 the hazard
  recognizer pads a compiler VALU read of an MFMA destination (`s_nop`) but not an asm read
  (compile-checked; `../gluon/inline-asm-reference.md`).
- **Then raise occupancy:** `waves_per_eu=4` (or 3) with
  `llvm_fn_attrs` `amdgpu-sched-strategy=iterative-minreg` + `amdgpu-agpr-alloc=0`
  (`../tile-programming/llvm-fn-attrs.md ### Scheduler strategy — which names the 3.8.0 LLVM honours on gfx950`).
  On that body the default scheduler spilled at 4 waves (128-register budget) and minreg fit the 32-head
  body in 124-127 arch VGPRs, 0 AGPR, 0 scratch; at 3 waves the default scheduler can fit with
  `agpr-alloc=0`. The occupancy gain showed up only with the scalar fold. Re-derive the key-split count for the new slot count
  (`waves_per_eu x 1024 / num_warps` workgroups per round).

*Order of the A/B:* scalar fold at 2 waves (tie to small), then the occupancy step on top; the
occupancy step with the packed fold kept is clearly slower than not taking it, so a "3 waves lost"
result measured on a packed fold says nothing about the scalar body. *Check:* `asm_loop_audit.py`
counts (`v_pk_fma_f32` 0 in the loop), `probe.py measure` (registers within the budget, scratch 0,
no `v_accvgpr_*`). Verified on Triton 3.8 / gfx950 on a 32-head prefill body (2-row x 4-warp
row partition, 64-key tiles), in two independent kernels.

### DSA indexer decode (paged MQA logits): measured levers

The decode form of `### DSA indexer (MQA logits): not flash attention`: one query row per request
(or `next_n` rows with MTP), a paged fp8 key cache (per page: keys + per-token scales), heads on M,
and a logits row written per query. **A key split owns disjoint logits columns**, so the split-KV
machinery of `## Decode (split-KV): a structure, not a config knob` mostly does not apply: no
partial records, no scratch round trip, no merge. What does apply is grid fill. Verified on
Triton 3.8 / gfx950:

- **Fuse the `next_n` rows that read the same keys into one program** (each key page loaded once).
   After it, the measured MTP body was bound by the VALU fold (ReLU, weight, cross-head sum), not by K
   latency: an LDS async K ring lost there. What paid was cutting the fold — packed FMA for the
   weighted head sum, one cross-lane reduction shared by all rows. (Packed is this decode body's result; on the prefill
   body the scalar chain won once occupancy was raised — `#### DSA indexer: occupancy needs the
   scalar fold and iterative-minreg`; A/B the form per body.)
- **Split count = one residency round.** Size the key split so `B x splits` is one round of
   resident programs; both half and double that lost on the single-row cases measured. The round
   depends on the kernel variant (rows per program, waves per EU), so a table built for another
   variant does not carry over.
- **Cyclic split assignment** (split `s` owns chunks `s, s+S, ...`): its first page-index loads do
   not depend on the context length, so they issue with the Q/weight loads — one dependent HBM trip
   shorter at program start. Small; it pays on the start-up chain, not on balance.
- **Pin the back-edge.** A loop-carried K tile gets a `v_mov` copy at the back edge that forces a
   wait on the load just issued (prefetch depth 0); unroll to a two-chunk ping-pong with an empty
   side-effecting asm fence (`../gluon/inline-asm-reference.md`).

**Floors before a round:** single-row cases sit within a small factor of the graph-launch floor
plus the paged-gather time at the *probed* in-shape bandwidth (`mem_bw_probe.py --runlen <page
bytes> --nprog-sweep`), not the datasheet; the datasheet multiple from `hw_budget.py` overstates the
room (pass `--floor-us <empty-kernel us>` and the probed bandwidth), and it counts only MFMA FLOPs, so it labels the VALU-bound MTP body as memory-bound.

### Sparse MLA (DSA): measured levers

Every head reads the **same** gathered latent row (`kv_lora` 512 + rope 64; packed fp8 with
per-group scales, or bf16), selected per query by a top-k index list: QK over 576, PV over 512,
heads on M (one program holds all heads of a query), and a two-hop address chain
`indptr -> indices -> row`. Prefill and decode bind on different resources and their lever lists
barely overlap — the bound class picks the list. Measured on Triton 3.8 / gfx950 by lever-by-lever
ablation of two independently optimized kernels (they found the same levers; little stacks across them).

**Prefill (thousands of query rows, full grid) — on-chip bound, not HBM.** *Signal:* forcing the
gather to hit L2 barely moves it; `vmcnt(0)` inside the QK chain. Levers (unordered; pick by signal):

| Signal | Lever | Evidence / check |
| --- | --- | --- |
| a register-staged gather shares `vmcnt` with an in-loop scratch spill or a loop-carried copy, so it waits mid-QK | **make the prefetch real** | *check:* `.vgpr_spill_count`, and the `s_waitcnt vmcnt` before the first QK MFMA (`asm_loop_audit.py`) |
| converted `ds_write` in the loop; LDS too tight for a bf16 tile beside the DMA buffers | **raw fp8 row in LDS, dequantize on the operand read** (half a bf16 tile's LDS) | on that body a bf16 dequant-once tile dropped to one workgroup per CU and lost |
| gather staged through registers | **gather straight into LDS with the async copy, two buffers** (`wait_group` already emits the barriers: `../tile-programming/pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`) | this and the raw-fp8 row together were the largest prefill change there; each paid on its own |
| `.vgpr_spill_count > 0` and `v_accvgpr_*` copies in the loop at 2 waves/SIMD (the 576-wide walk sits near the 256 arch-VGPR budget) | **read the register split:** `amdgpu-agpr-alloc=0`, or an N large enough to be spill-free, fit in arch VGPRs with no copies | the default split spilled and lost heavily; a sibling kernel at fewer VGPRs needed nothing. *Check:* spills and `v_accvgpr_*` per build, then the AGPR budget rule in `../tile-programming/llvm-fn-attrs.md` |

*Tie / small:* LDS pad or swizzle of the raw tile, scaled-fp8 QK on raw K.
*Measured negative on this sparse-MLA prefill body* (re-test if its VALU / register balance
changes): fp8 PV through a hi/lo split of P (the split's VALU equals the converts it removes), Q
out of VGPRs to reach 3 waves/SIMD, a lazy-max branch.

**Decode (one row per request, `B x head_blocks` below the CU count) — fill, then fixed costs.** Levers
(unordered; pick by signal):

| Signal | Lever | Evidence / check |
| --- | --- | --- |
| grid well below one resident round of the CUs | **split count fills the device, one round:** splits up to `B x head_blocks x splits` ≈ one resident round (power-of-two splits, cap around 32, at least one tile per split) | the largest decode change measured there; past one round the extra programs queued and lost. *Check:* the grid against the CU count, `## Grid fill when the body is LDS-capped` |
| many splits to merge | **a separate, wide merge kernel:** column-block the 512-wide row so the merge launches many small workgroups, read partials without `.cg`, mask by live splits, bf16 partials | the in-kernel last-arriver merge (`### In-kernel split merge on gfx950`) **lost on every decode path tried** here: the merging program's serial tail is longer than one graph node |
| two KV segments (sliding window + compressed top-k) walked back to back | **one segment per program:** each segment gets its own split programs (no two dependent gather chains in one program) | paid where measured |
| a split walks several tiles | **body levers:** the fp8-in-LDS async walk (and a two-ahead DMA at one program per CU); 8 warps on a grid of at most a quarter of the CUs spreads one tile's gather over all SIMDs | paid at large batch; at one tile per split there is little left to overlap |

*Floors to price before a round:* the graph node floor per dispatch (two nodes: main + merge),
random-row gather throughput below the linear HBM ceiling (a 656 B row touches 6 lines), and
the scratch round trip at short top-k (`### The scratch round trip is the byte account decode adds`).

**Per-group scales** fold into the scores and P, not Q (`attention-lowprec.md` section 4).

## Family signals (attention / decode)

A patch that changes RoPE, online softmax, QK/PV, split logic, K/V indirection, LDS and MFMA together
leaves a failure or a regression with no attributable cause.

- **Paged / indirect KV** (`kv_loc = page_table[offs//PAGE]*PAGE + offs%PAGE`): the index tensor may
  carry a `BlockedLayout` without `SliceLayout` lineage; if the 2D address needs
  `expand_dims(kv_loc, 1)`, migrating that stage to Gluon directly is the hard case — the options are
  a plain indirect load, or a source-proven gather.
- **Quantized attention:** `quant/preprocess`, `attention`, `dequant` and `reduce` timed separately
  show where the time is; a material preprocess stage is its own target (compute-bound attention
  configs do not transfer to memory-bound quant kernels).
- **Atomic / nondeterministic reductions:** compare against a deterministic high-precision
  reference with task tolerance, not two atomic outputs.

QK/PV layout chain: `../gluon/matrix-reference.md`; decode split / dispatch: `../method/front-end.md`.
