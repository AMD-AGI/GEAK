---
id: megamoe_ep_mega_fusion
title: MegaMoE EP8 two-launch persistent fusion
kind: expert_skill
mode: mega
authors:
- GEAK Team
scope: kernel
match:
  operator: moe_dispatch_combine
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp8_e4m3
  - mxfp4
  regimes:
  - prefill
  - decode
  from_backend: ''
  to_backend: flydsl
  profile_signature:
    op_name_regex: mega_moe|dispatch_combine|p2p_scatter
    min_pct_gpu: 20.0
  config:
    framework: MegaMoE_v2
    parallel: EP8
    arch: gfx950
    precision: a8w4
expects:
  isolated_speedup_min: 1.03
  e2e_delta_min_pct: 1.0
  parity: required
role: experimental_authoring_prior
normative: true
normative_scope: explicitly_pinned_authoring
planner_extension_file: planner_extension.yaml
contract_file: contract.yaml
runtime_validation_file: runtime_validation.py
validation_file: validation.yaml
---

## When to use

The validated evidence covers MegaMoE V2 on one EP8 gfx950 node and the
complete two-launch persistent target. In that target quantization remains a
separate launch; dispatch, GEMM1, GEMM2, P2P publication and Combine execute
in one persistent grid.

Token range of the evidence (operator capacity `max_tok_per_rank` = tokens per
rank, uniform route, rank-max against the frozen baseline):

| mtpr | earlier build (v10) | current levers (8192 tuning) | evidence |
|---|---|---|---|
| 64 | 0.90x | | not this Skill's range |
| 128 | 0.877x | | not this Skill's range |
| 256 | 1.035x | | |
| 512 / 1024 | 1.060x / 1.050x | | official test, 5 interleaved runs |
| 2048 | 1.096x | 1.075x | |
| 4096 | 1.207x | 1.179x | |
| 8192 | 1.163x | 1.216x | validated (1.155x certified pinned-Skill run) |
| 16384 | 1.081x | 1.194x | |

Unless marked otherwise: paired CUDA-graph A/B against the frozen tree
(`tools/mega_graph_ab.py`, 3 rotated passes, all passes faster), output
bit-exact with the frozen tree, `path=MEGA` on all eight ranks. The
current-lever build ran its 8192 tuning unchanged at the other sizes; replay
with route mutation and the acceptance chain were run only at 8192.

So this structure wins from 256 up to at least 16384 and loses at 128 and below,
where `megamoe_small_mtpr_fusion` applies. Below 2048 the earlier build is
ahead, so the current levers (tuned at 8192) need re-tuning per size; at
16384 they matter most.

This Skill declares measured Pattern evidence; it does not own applicability
control flow. Analyze reports the current semantic fingerprint and bindings.
Planner owns whether to apply, adapt, partially use, request re-analysis, or
decline the guidance. Source names and layout are not part of the Pattern.

## Validated profile and semantic fingerprint

The measured evidence context is one EP8 gfx950 node, wave size 64, A8W4,
BF16 external input/output, per-1x32 MXFP8 activation quantization with E8M0
scales, MXFP4 expert weights with the supplied shuffle/scale contracts,
384 experts (`48/rank`), top-k 6, hidden/intermediate dimensions 7168/3072,
and bounded SwiGLU with limit 10.

Analyze should expose this checklist to Planner before using profile-scoped
constants:

- stage graph: separate input quantization, routed first expert GEMM,
  activation plus output-scale production, dependent second expert GEMM,
  route-weight multiplication, cross-rank publication, and top-k output
  reduction;
- topology: one node, eight ranks, experts divisible by ranks, rank-local
  expert weights, global expert IDs, and exactly top-k routed contributions per
  output token;
- numerical fingerprint: BF16 external rows, MXFP8 first/second activations,
  MXFP4 weights, 1x32 E8M0 scale groups, SwiGLU semantics, route weights
  applied before publication, and FP32 final accumulation;
- transport fingerprint: large-MTPR direct slots indexed by
  `(destination_token,route_slot)`, block-FP8 payload plus one E8M0 byte per
  32 values, and BF16 final output;
- lifecycle fingerprint: symmetric peer-visible payload/readiness ownership,
  one in-flight invocation per operator instance, persistent backing storage,
  stream ordering, and graph replay across changing routes and token counts;
- architecture fingerprint: gfx950 cache-scope semantics, 64-lane waves,
  163840-byte LDS limit, and enough CUs for the profile grid.

A mismatch is evidence, not a Skill-side applicability decision. Analyze
reports it, re-discovers the current role bindings, re-derives every affected
formula or profile constant, and exposes the result to Planner for
workflow-owned selection. Profile-scoped measured constants are not
extrapolated silently.

## Non-normative discovery and mapping

Analyze discovers current bindings from dataflow, calls, types, shapes,
address spaces and launch sites:

1. Follow BF16 rows into the 1x32 quantizer and identify its MXFP8 payload and
   E8M0 scale outputs.
2. Follow route IDs/weights into the rank-local first expert contraction and
   identify the activation-plus-scale rows consumed by the dependent
   contraction.
3. Follow the dependent contraction through route weighting into peer-visible
   direct slots and their readiness publication.
4. Follow every top-k slot into the FP32 reducer and BF16 output.
5. Find the host path that owns buffers, selects bucket parameters, constructs
   the JIT identity and issues launches.
6. Record discovered files and symbols in the ordinary Analyze modifiable-file
   set, task graph, PlanIR regions, ABI arguments and compiler constraints.

When the baseline evolves, repeat this mapping. Current class, function and
module names are neither requirements nor evidence of the Pattern.

## Mechanism

This structure is validated at large token counts (the 8192 profile). It does
not win at small token counts: a quantization launch plus this persistent grid
measured 0.999x at 64 tokens per rank, and the fused kernel 0.885x at 128. The
weight-streaming regime (`max_tok_per_rank` about 32-128) is covered by the
separate Skill `megamoe_small_mtpr_fusion`; never carry this structure into it
for code reuse.

The canonical large-token BF16 path has exactly two launches per rank:

1. the existing BF16-to-MXFP8 `per_1x32` quantizer;
2. one persistent grid containing dispatch, GEMM1, GEMM2, weighted P2P
   publication, readiness, and final Combine.

The full path is opt-in and must be selected only when the fusion switch is the
exact string `1` and the selected Stage1 wave count is divisible by four.
The corresponding explicit activation is `AITER_MEGAMOE_FUSE_ALL=1`.
Combine is present in the persistent grid. Quantization is not. The host
returns the persistent kernel's output view and launches no Stage2 or Combine
tail.

The second launch is one CTA-local state machine:

- One ticket identifies the launch generation and the CTA's startup role.
  Ticket zero is the plan owner; tickets `1..dispatch_blocks` are payload
  producers. All roles join the compute queues after startup.
- G1 work is the flat stripe domain
  `unit = m_tile * g1_n_stripes + n_stripe`.
- G1 activation and scale stores use cache bits `sc0|sc1` (`17`). Every wave
  retires its stores, all waves reach one block barrier, and lane zero performs
  one system-scope completion add for the owning m-tile. There is no
  post-publication L2 writeback.
- G1 and G2 are chosen inside the same loop. G1 is the default. A CTA may peek
  and preempt with ready G2 only when both `ticket % 6 == 0` and
  `max_expert_tiles * experts_per_rank * stage1_tile_m * 4 >
  valid_rows * 5` hold.
- After G1 exhaustion, every CTA may claim G2 and wait for its dependency. An
  unready G2 claim is pending work, not an empty queue. A CTA reaches Combine
  only after it owns no continuation and its G2 shard head is out of range;
  consequently every valid G2 chunk has already been claimed by some CTA.
- G2 claims are contiguous in claim-position space. For shard `s`, local claim
  `q`, shard count `S`, and chunk `C`: `claim_id = s + q*S`, `base = claim_id*C`,
  and the claim owns positions `[base, min(base+C,total_g2_units))`. A position
  maps to a `(m_block,n_block)` unit through the G2 claim order (Scheduling
  blueprint); the unit id stays `m_block*g2_n_blocks + n_block`, and close
  counters and dependencies use that id. With the column-grouped order use
  `C=1` at every bucket. `C=16` at `run_tokens>=4096` belongs only to the
  m-major reference order. Readiness of the last in-range unit dominates the
  contiguous claim. The continuation carries only two block-uniform scalars,
  `next_unit` and `units_remaining`; it performs no further head atomic,
  readiness poll, or LDS handoff.
- The G2 single-unit body owns all route, weight, peer, and readiness
  descriptors. No descriptor, GEMM accumulator, or nested G2 loop crosses the
  outer scheduler backedge.
- Stage2 writes payload and block scales with `sc0|sc1|nt` (`19`). After a
  workgroup release and uniform barrier, lane zero adds one to the rank-local
  m-block close counter. The previous value is broadcast through LDS. The
  closing CTA resets the counter by an agent-scope `-g2_n_blocks` add, then the
  first `BM` threads publish exactly one system-scope arrival for their
  accepted routed row.
- Token-arrival counters are never cleared after initialization. A
  Combine-specific generation increases only on launches that contain both
  arrival publication and Combine. The wait target is `topk * generation`;
  tokens outside `[0,run_tokens)` are advanced to that target before startup
  publication.
- Combine is an irreversible third queue with no grid barrier. One agent-scope
  block claim advances by one; claim `b` gives wave `w` item
  `b*cta_waves+w`. Each item waits independently in lane zero, reads all top-k
  direct slots with cache bits `19`, reduces in FP32, and stores BF16 with
  rank-local `nt` cache bits `2`.
- Producer and consumer share ONE token index. The scattered payload slot and
  the published arrival row both decode from the sorted-meta token field:
  payload at `dest_lid*topk + k`, arrival at `dest_lid`. Combine reads
  `tok_id*topk + k` and waits on arrival counter `tok_id` for
  `tok_id in [0,run_tokens)`. Therefore the sorted-meta `dest_lid` MUST equal the
  owner's dense token id — this is the `MTPR*topk` (`[owner_tok][topk]`) layout,
  NOT the `world_size*MTPR` (`src_rank*MTPR + slot`) layout a standalone combine
  uses; do not import that remap here. A permuted `dest_lid` corrupts payload and
  arrival together: every off-token reads wrong data (relL2 -> nan) and its
  counter never reaches `topk*generation`, so Combine waits forever. This is not
  catchable structurally — it is a numerical co-indexing invariant, verify it
  on-card at bs=128.
- Producer scatter and Combine decode MUST agree on the P2P payload format, and
  BOTH must be derived from ONE source — the selected bucket's `p2p_quant`. The
  producer writes either `none` -> BF16 rows (`hidden*2` bytes/token) or
  `fp8_blockwise_1x32` -> MXFP8 payload + one E8M0 byte per 32-value block
  (`hidden + hidden/32` bytes/token). The Combine consumer's blockwise-decode
  flag AND its per-token `nbytes` stride MUST resolve to that SAME `p2p_quant`;
  neither end may hardcode the format. If the fused arm enables fp8 on the
  producer it must thread the identical flag to Combine, and if it leaves the
  producer at `none` the Combine decode must be BF16 too. A hardcoded blockwise
  decode over a `none` (BF16) producer scatter reads E8M0 scale bytes out of BF16
  payload -> inf/nan on essentially every token, and the `hidden*2` vs
  `hidden+hidden/32` stride disagreement mis-strides the whole payload slab. This
  fault is unconditional (all bs), first-order ABOVE the co-indexing invariant
  above, and not structurally catchable — verify the producer emit and the
  Combine emit resolve the SAME `p2p_quant` at bs=128.

## Normative implementation recipe

This is one dependency-ordered construction, not six independent optimization
ideas. Reuse the supplied baseline's numerical bodies; change ownership,
placement, queueing and publication around them. Complete each stage before
using evidence from the next stage.

### Stage 0 — freeze baseline semantics

Discover and record the current bindings for quantization, GEMM1, GEMM2,
weighted scatter and Combine. Preserve:

- input/output dtypes and layouts;
- expert/routing/top-k indexing;
- weight and E8M0 scale layouts;
- SwiGLU and route-weight math;
- FP32 Combine accumulation.

Do not author new GEMM math. The fused implementation calls, extracts, inlines
or equivalently rewrites these existing numerical bodies.

### Stage 1 — Host selection, ownership and ABI

Host owns all persistent allocations and two immutable compile specs:

- Stage2 spec: `NW`, `BM/BN/BK`, `SBM`, `p2p_quant`, extents and publication
  mode;
- Combine spec: the same `p2p_quant`, output geometry and reduction policy.

Host also owns distinct runtime roots for:

- Stage2/GEMM2 inputs, route metadata and peer payload table;
- Combine input/output and token-readiness peer table;
- G1 completion counters;
- sharded G2 claim heads;
- Stage2 m-block close counters;
- Combine claim head and generation.

Compile specs belong in the cached JIT identity. Pointer roots and dynamic
extents are real kernel runtime parameters, not captured Python objects or
compile-time tuple splats. The fused launch signature and launch call must have
the same positional order and count.

The selected path performs:

```text
quant_launch()
persistent_launch(stage2_roots, combine_roots, dynamic_extents)
return persistent_combine_output
```

It launches no Stage2 or Combine tail. This stage is complete only when the
active host call reaches the cached compile call and real launch ABI.

### Stage 2 — G1 production and publication

Keep the baseline dispatch and GEMM1 numerical body. Change the consumer
epilogue so each flat G1 stripe:

1. waits for its dispatch payload;
2. executes GEMM1 once;
3. stores activation and scales with cache bits `17`;
4. drains every wave's stores;
5. reaches one uniform workgroup barrier;
6. lets thread zero perform exactly one system-scope completion increment for
   the owning Stage1 m-tile.

The G1 completion region is separate from G2 claim heads and Stage2 close
counters. A G2 dependency wait reads only the G1 completion region.

### Stage 3 — shared G2 body and weighted P2P publication

Extract one shared Stage2 single-unit emitter used by both standalone Stage2
and the persistent scheduler. Its unit is exactly one
`(m_block,n_block)`. Inside that frame construct all route, expert, weight,
peer and readiness resources; do not carry descriptors or accumulators around
the scheduler backedge.

The Host-provided `NW` is load-bearing:

- A-load rows use `BM/NW`;
- GEMM2 columns and scale words use `BN/NW`;
- scatter rows use `BM/NW`;
- the 8192 profile resolves to BM64/BN512/BK256/NW8.

After GEMM2, apply route weight and write the direct peer slot with cache bits
`19`. Every wave releases its stores, all waves meet a uniform barrier, and
thread zero increments the dedicated m-block close counter. The last N-stripe
owner self-clears that close slot and the first `BM` threads publish one remote
arrival per valid row.

The emitter receives a caller-provided LDS allocation handle. Obtain the
address from the materialized `SharedAllocator().allocate(...).peek()` handle
and its `.buf.ptr`; never pass a raw storage type where the emitter expects the
peek handle. Every derived address includes the caller's slab base.

### Stage 4 — unified persistent G1/G2 scheduler

Use one block-uniform outer loop. Its decision order is:

```text
if a claimed G2 chunk has continuation:
    run next G2 unit                         # no claim, poll or LDS handoff
else if this CTA is in the skew cohort and next G2 chunk is ready:
    claim G2 chunk; clamp final unit; run G2
else if a G1 stripe remains:
    claim G1; wait payload; run G1; publish G1 completion
else if a G2 chunk remains:
    claim G2; wait for its last in-range dependency; run G2
else:
    exit scheduler
```

G1 is the forward-progress default. Preemption requires both ticket-mod-6 and
the 5/4 expert-skew predicate. A chunk claim seeds only
`next_unit`/`units_remaining`; continuation performs no further atomic or
readiness wait. Clamp the last chunk before mapping it to a G1 dependency.

Only after the scheduler owns no continuation and its G2 shard is exhausted
may the CTA enter Combine. Do not append a second fallback G1 drain after the
unified loop.

### Stage 5 — Combine as the irreversible third queue

Create one decorated Combine item emitter. Runtime block claims advance by one;
claim `b`, wave `w` executes item `b*cta_waves+w`. The item:

1. derives `tok_id`, partition and bounded hidden range;
2. lane zero waits monotonically for `topk*generation`;
3. reads all direct top-k slots using cache bits `19`;
4. decodes either BF16 or block-FP8+E8M0 according to the same `p2p_quant`
   used by Stage2;
5. reduces in FP32 through one pressure-selected U1/U2/U4 main path;
6. stores BF16 with cache bits `2`.

Readiness is a correctness precondition. A production item never times out and
continues into incomplete payload reduction. Bounded waits and cut-points are
diagnostic variants with distinct JIT identities and cannot be committed as
the final path.

### Stage 6 — generation, replay and acceptance

Before startup publication, the owner:

- clears only per-launch claim heads and G2 heads;
- increments Combine generation;
- advances omitted token counters to `topk*generation`;
- leaves arrival counters monotone across launches.

Changing routes and smaller-to-larger token replays must reuse persistent
storage without stale reads. Validate in order:

1. active-path trace/JIT at bs=128;
2. `path=MEGA` on eight ranks and exactly two launches;
3. relL2 below 0.10 at 128, 512 and 8192;
4. route-changing graph replay and liveness, including a route with one hot
   source rank that moves on every replay; the first replay after each change
   must match a repeated replay of the same route;
5. paired rank-max speedup on the target guard (8192-uniform and at least
   1.03x unless the workflow names another target; see Acceptance).

### Construction checkpoints

After each stage, run only the earliest meaningful check:

- Host/ABI: Python compile, import identity and exact compile/launch binding;
- G1/G2 construction: active-path FlyDSL trace with the real fused switches;
- publication/scheduler: bs=128 bounded diagnostic smoke, then restore strict
  production waits;
- Combine/lifecycle: independent exact-HEAD runtime Verify;
- performance: only after all correctness and replay gates pass.

An inactive fallback, a same-named dead helper, or a run without `path=MEGA`
does not complete a stage.

Timed builds fix native Stage2 geometry, G2 cadence, cache policies, readiness,
and Combine geometry to the rules below. Every non-target launch topology,
publication cut, fused-quant path, wait timer, profiler, and diagnostic is
disabled and has a distinct JIT identity.

## Semantic role boundaries

Keep these responsibilities separate regardless of source organization:

- The host-selection role validates the semantic fingerprint, selects one
  bucket, keeps quantization separate, builds semantic argument groups, launches
  the persistent grid once, and returns its output without a tail launch.
- The first-compute item role emits exactly one flat output stripe. Queue
  progression and completion publication belong to the persistent scheduler.
- One derivation authority validates every dependent-GEMM tile, scale,
  transport, LDS and bounds constant. One single-unit role emits exactly one
  `(m_block,n_block)` unit. Standalone and persistent work distributors may
  share, inline, or equivalently lower the body and P2P epilogue.
- The dependent-GEMM role constructs route, weight, expert, peer and readiness
  resources inside the single-unit frame. Every LDS address is relative to a
  caller-supplied slab base.
- One output-reduction item role derives runtime `P`, partition width and `U`,
  owns direct-slot resources, the per-wave wait, all top-k loads, FP32
  accumulation and bounded BF16 stores. Its U-step and resource collections
  remain inside that decorated item frame. The scheduler owns only block
  claims.

Treat the supplied baseline's numerical bodies as concrete implementation
references after verifying these contracts. Reuse, extract, inline, or rewrite
them to fit the fused schedule while preserving quantization, scale layout,
expert indexing, activation, weighting and accumulation semantics.

Do not copy any external implementation tree; source authoring remains
independent and the reference is calibration-only evidence.

## Baseline math, ABI and state layout

Analyze the supplied baseline's numerical bodies and use them as implementation
references after verifying their dataflow and type contracts:

- keep the standalone BF16-to-MXFP8 `per_1x32` quantization launch;
- preserve GEMM1 math, activation/scale layout and weight shuffle;
- preserve GEMM2 math and weighted P2P scatter semantics;
- preserve Combine BF16 and block-FP8 decode/FP32-accumulate behavior.

The implementation form is open; the observable operator contract is not.
Do not silently change quant formats, scale grouping, expert indexing, top-k
weights or accumulation dtype to obtain the launch reduction.

### Canonical configuration

Token buckets are fixed to
`[1,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]`.
Choose the nearest bucket, resolving a tie upward. `max_tokens_per_rank` is a
power of two. Values at or below 1024 retain their own MTPR class; larger
values use class 32768. Fixed-slot dispatch is used only for MTPR at or below
255. The expert-count class is `ceildiv(experts_per_rank,64)*64`; dispatch CTA
counts below are multiplied by that class divided by 64 and capped at 224.

Stage1 defaults are fixed: `tile_k=256`, weight pipelining on, A swizzle on,
`waves_per_eu=2`, and eight work shards unless a rule below overrides them.
Grid multipliers are drawn from `[1,2,3,4,6,8,12,16,24,32]`; the launch
generation state has one slot for each value.

For fixed-slot MTPR:

- `SBM=32`, `waves=4`, `tile_n=256` through bucket 8 and 128 thereafter.
- `grid_mult=max(1,bucket//4)` through bucket 16 and 3 thereafter.
- Dispatch CTA bases are 64 at bucket 1, 128 through bucket 8, 96 through
  bucket 16, 128 through bucket 32, then
  `min(224,16*(bucket.bit_length()+7))`.
- MFMA A-major and asynchronous A copy are off. Tile resources are on through
  bucket 16. B cache mode is 0 at bucket 1 and 3 otherwise.
  `waves_per_eu=1` only at bucket 16.

For non-fixed MTPR at or below 1024:

- Buckets through 4 use `(SBM,tile_n,waves,grid_mult)=(32,256,4,1)` with
  A-major and asynchronous A copy off.
- Buckets 8 through 128 use `(32,512,8,1)` for `inter_dim>=2048`
  (`tile_n=256` otherwise), with A-major and asynchronous A copy on.
- Buckets 256 through 1024 use `SBM=64`, the same tile-N rule and eight waves;
  `grid_mult=1` at 256 and 2 otherwise.
- Dispatch CTA bases are the compact sequence
  `224,128,192,64,128,192,128` for bucket ceilings
  `1,4,8,16,32,64,>=128`, then 160 at 256 and 128 above it.
  When MTPR exceeds the bucket, override bases at 32/64/128 to 64/160/192;
  bucket 512 additionally fixes `grid_mult=1`, dispatch CTAs 64, tile
  resources on and B cache mode 0.
- Tile resources are otherwise on only at bucket 256. B cache mode is 0 at
  bucket 1 and at or above 1024, and 3 otherwise. Work shards remain eight.

For fixed-slot and non-fixed MTPR at or below 1024, Stage2 uses `BK=256`,
B hoisting and A-scale prefetch on, spatial code 402, and BF16 LDS off. If the
path is non-fixed and MTPR exceeds the selected bucket, fix `BM=32`,
`BN=128` only at bucket 256 and 256 otherwise, persistence on with 240 CUs,
non-temporal B loads through bucket 128, and persistent striding from bucket
512 through 2048. Otherwise use `BM=32`; choose `BN=256` for buckets
1/4/64, at or above 1024, or non-fixed buckets below 128, and 128 for the
remaining buckets; force BN=128 when `model_dim<4096`. Persistence begins at
bucket 128, with 128 CUs at bucket 256 and 240 CUs otherwise; non-temporal B
loads are used through bucket 128 and persistent striding from bucket 512
through 2048.

For MTPR above 1024, including the measured 8192 class:

```text
bucket       1   4   8  16  32  64 128 256 512 1024 2048 4096 8192 16384 32768
SBM         32  32  32  32  32  32  32  64  64   64   64  128  128   128   128
tile_n     256 256 512 512 512 512 512 512 512  512  512  512  512   512   512
waves        4   4   8   8   8   8   8   8   8    8    8    8    8     8     8
dispatch   224 128 192  64  64 160 192 160  64   64   64   64   96    32    32
G1 shards    1   1   1   1   1   4   4   4   4    4    8    4    4     4     4
payload rows384 384 384 384 384 384 384 384 256  384  384  384  384   384   384
G2 BM       32  32  32  32  32  32  32  32  32   32   32   64   64    64    64
fused G2 BN256 256 512 512 512 512 512 256 512  512  512  512  512   512   512
G2 persist 240 240 240 240 240 240 240 240 240  224  256  240  240   192   240
G2 skew CUs  0   0   0   0   0   0   0   0  96   96   96   96   96    96    96
G2 chunk     1   1   1   1   1   1   1   1   1    1    1   16   16    16    16
```

In this class `grid_mult=1`, tile resources and payload-tile readiness are on,
and transport is block-FP8 `1x32`. A-major and asynchronous A copy are off
through bucket 4 and on thereafter. B cache mode is 3 only for buckets
`4..256`, otherwise 0. External grouping is on at bucket 4 and at or above
256; external counting is on at or above 256. Stage2 B loads are non-temporal
through bucket 128 only; persistent striding is on for buckets 512 through
2048. Stage2 `BK=256`, B hoisting and A-scale prefetch are on, BF16 LDS is off,
and the configured spatial code is 402.

The persistent body, not standalone Stage2, owns work distribution. It
therefore forces Stage2 persistence off and spatial remapping to zero; the G2
claim order (Scheduling blueprint) must restore the W2 reuse that the
standalone mapping provided. It also uses
one slab, sets `NW=Stage1 waves`, and widens the configured Stage2 BN by
`Stage1 waves/4`. GEMM2 uses FP8 A, FP4 B, scaled 16x16x128 MFMA, `BK=256`,
two 128-K halves, three A LDS stages, `wave_n=BN/NW`, and
`num_acc_n=wave_n/16`. B payload loads are four i32 per lane. The
BM64/BN256 case alone uses the one-stage B path; all other canonical fused
shapes use the two-stage B pipeline with B hoisting and A-scale prefetch. At
the measured 8192 shape this fixes:

```text
EP=8, experts/rank=48, topk=6, hidden=7168, inter=3072, MTPR=8192
Stage1: SBM=128, tile_n=512, tile_k=256, waves=8, grid=256 CTAs
         dispatch=96, shards=4, payload_rows=384, B cache=0
Stage2: BM=64, BN=512, BK=256, NW=8, one slab, 14 N blocks
         B non-temporal=off, B-hoist=on, A-scale-prefetch=on
Scheduler: preemption modulus=6, skew ratio=5/4,
           G2 order=column-grouped GM=32 (skew interleave when gated),
           G2 chunk=1   (reference m-major order: chunk=16)
```

### Host ABI and persistent state

The host ABI carries semantic groups: G1 tensors and dispatch state; raw G2
activation, scale, weight and route addresses; selected BM/BN/BK and maximum
m-block count; Combine payload/output and peer-ready addresses; two local
counter arenas; and the Combine control words. Device buffer resources are
constructed only inside the role that consumes them.

For compact dispatch:
`num_valid_max = world_size*MTPR*topk + experts_per_rank*128`.
For fixed-slot dispatch:
`num_valid_max = experts_per_rank*roundup(world_size*MTPR,32) + 256`.
Metadata reserves `ceildiv(num_valid_max,32)` entries.

The canonical host owns these allocations for the object's whole graph
lifetime:

```text
dispatch work heads       = 8 * 16 i32; one 64-byte line per shard
fixed-slot group_done     = world_size i32
G1/G2 counter arena       = max(1,num_valid_max) i32
G2-close counter arena    = max(1,num_valid_max) i32
Combine control           = 4 i32: block head at 0, generation at 1
token arrivals            = MTPR + 8 symmetric i32
peer-pointer tables       = world_size i64
Combine backing rows      >= max(world_size*MTPR, MTPR*topk)
Combine output rows       = MTPR BF16 rows
```

Within the G1/G2 arena, G1 completion entries cover
`ceildiv(max_g2_m_blocks*G2_BM,SBM)`. G2 claim head `s` is at
`max_g2_m_blocks + 16*s`, so completion and head regions are disjoint and
every head has its own 64-byte line. The second arena is indexed by G2
m-block and self-clears on close. The Stage2 payload resource is bounded to
`MTPR*topk*row_bytes`, where `row_bytes=hidden+hidden/32` for block-FP8 and
`2*hidden` for BF16, even when the backing allocation is larger.

All active addresses are nonzero and match emitted size/alignment. Route,
weight, expert, tile-row and peer resources use their real byte capacities.
Source-token encoding requires `world_size*MTPR <= 2^24`; route-slot encoding
requires `topk <= 2^8`; the Stage2 payload resource must remain below `2^31`
bytes. Disabled diagnostic roles may use zero placeholders only when
compile-time eliminated.

## Scheduling blueprint

Launch tickets are monotone per grid-multiplier slot. The owner toggles the
dispatch parity bank, advances its expected cross-rank value, performs the
launch handshake, resets G1 heads and active completion entries, resets the
Combine block head, increments the Combine generation, pads omitted tokens,
then publishes the epoch gate. No producer or consumer passes that gate first.

G1 heads are sharded exactly like this:
`shard = ticket & (S-1)`, `local = atomic_add(head[shard],1)`,
`unit = shard + local*S`. Each head begins on its own 64-byte line.
`g1_n_stripes=(2*inter_dim)/tile_n`; the owning completion index is
`unit/g1_n_stripes`.

The fused loop follows this order on every iteration:

1. A carried c16 continuation executes immediately. `next_unit` and
   `units_remaining` are block-uniform SSA values outside the lane-zero
   selection branch; no six-word LDS scheduler frame is permitted.
2. An eligible skew CTA non-destructively peeks its G2 shard head with an
   agent-scope add of zero. It tests the last in-range unit's owning G1
   completion. If ready, it atomically claims one chunk; because another CTA
   may win between peek and claim, it still performs the monotone blocking wait
   on the actual claim.
3. If no preemption was selected, claim one G1 stripe. Compact dispatch waits
   for that stripe's payload before executing it.
4. If G1 is out of range, atomically claim one G2 chunk regardless of current
   readiness and wait on the claim's last in-range dependency.
5. Broadcast only `kind` and `unit` through the shared scratch. Execute exactly
   one G1 stripe or one G2 unit. Loop while `kind!=none`.

The loop-top block barrier is mandatory: Stage2's epilogue has no trailing
barrier, so it separates the previous unit's LDS reads from the next scratch
write. Bounds dominate every G1/G2 readiness address. A CTA never waits for G2
while holding an unexecuted G1 claim.

When a CTA observes both heads out of range and has no continuation, it enters
Combine permanently. Other CTAs may still finish already-owned G2 chunks.
This is safe because no unclaimed G2 unit remains. Combine uses:

1. one unconditional block barrier;
2. lane-zero `atomic_add(block_head,1)` and one LDS claim store;
3. one unconditional block barrier;
4. `_base=claim*cta_waves`, `_item=_base+wave`;
5. one direct per-wave item call when `_item<total_items`.

The next loop-top barrier is the post-item barrier. The item contains no
workgroup barrier and no CTA-wide readiness mask or retry frame.

### G2 claim order (W2 reuse)

The claim position decides which W2 slice the concurrently running CTAs read.
The m-major order (`unit = position`) streams every n-block's weights once per
m-block and leaves the fused GEMM2 no faster than standalone Stage2. Walk a
group of `GM` m-blocks down one n-block column before moving to the next
column, so concurrent claims share one W2 slice in L2:

```text
group    = position // (GM * g2_n_blocks)
r        = position - group * GM * g2_n_blocks
gm_eff   = clamp(total_m_blocks - group*GM, 1, GM)   # last group is short
n_block  = r // gm_eff
m_block  = group*GM + (r - n_block*gm_eff)
unit     = m_block * g2_n_blocks + n_block
```

Measured at 8192 uniform with `C=1` (rank-max e2e): GM=1 (m-major) 4.681 ms,
GM=8 4.419, GM=16 4.089, GM=32 4.047, GM=64 4.023. G2 span fell from 2.05 to
1.36 ms while G1/G2 overlap stayed at about 3%, so the gain is G2 efficiency,
not overlap. An n-major order (one n-block column per claim range, the claim
ranges kept local to a chiplet) measured the same direction. With this order,
larger chunks lose: at GM=16, C=1 4.089, C=2 4.202, C=4 4.424, C=16 4.880 ms
(tail quantization over about 11.6k units on 256 CTAs). For a drain smaller
than about 64 units per shard, keep the m-major order.

The Host passes the chunk (`fused_g2_chunk`) and the group size
(`fused_g2_group_m`) into the cached compile spec; both are part of the JIT
identity, so a sweep never reuses another variant's kernel. The contract's
advisory `host_specializes_g2_chunk` hint still describes the m-major
reference mapping (16 at `>=4096` tokens); a `C=1` candidate reports it as an
advisory miss, which is expected.

Measured losses with this order at 8192: GM 48-128 flat against 32; a W2 load
cache modifier that bypasses L2 (`sc1`) +5.5% slower, so W2 must stay in L2;
dispatch geometry (dispatch CUs 64-160, payload rows 256, work shards 8) flat,
but 128 payload rows +40% slower.

The mapping must be a bijection over `[0, total_g2_units)`: every CTA must be
able to claim every unit, or a CTA waits on a unit no one will run. A subset
drain that excluded some CTAs deadlocked.

On a rank-mixed-skew route one expert holds most rows. Its rows are laid out
by source rank, so consecutive m-blocks all return to one peer and the
column-grouped order serializes the P2P stores on one link: skew fell to
0.58x. Gate on the plan's `max_expert_tiles * 4 > num_m_tiles` and, when it
holds, permute the m-block index across `npes` equal segments before the
column grouping (`m' = (m % npes) * (total_m // npes) + m // npes` for
`m < npes*(total_m // npes)`, else `m`). That restored skew to 1.21x and leaves
a balanced rank's order unchanged.

### G1 payload-spin fill and deferred G2 completion

Two hot-path levers measured on the column-grouped build at 8192:

- Payload-spin fill. A CTA that holds a G1 stripe whose payload is not ready
  does not spin idle: it peeks the G2 head, checks that the peeked position's
  owning G1 completion is ready, and claims exactly that position with a
  compare-and-swap, then returns to its stripe. It never takes a blocking G2
  wait while holding an unexecuted G1 claim. Measured -0.7% uniform, -0.9% on
  skew. An atomic-add claim instead of the compare-and-swap tripled the fills
  for only -0.2%; a peek only at the loop top, or more polls between peeks,
  gave nothing.
- Deferred G2 completion. With a coarse pre-Combine barrier, G2 completion
  needs no per-unit drain, barrier and add: thread zero counts the
  workgroup's units in LDS, and on leaving the G1/G2 loop the workgroup
  drains once, meets one barrier and adds its count; the add that reaches
  the G2 total publishes the epoch. Measured -0.8%.

## Memory-order blueprint

- G1 activation and scale stores use `sc0|sc1=17`; order is stores,
  `s_waitcnt(0)`, uniform block barrier, then one system-scope add. A cached
  store, cache value 2, a release after the add, or an RMW before the barrier
  is not equivalent.
- G2 waits for `g1_ready[owner_tile] >= g1_n_stripes`.
- P2P payload and block-scale stores use `sc0|sc1|nt=19`. Each active lane
  writes eight contiguous payload bytes. For block-FP8, one lane per four-lane
  group writes the E8M0 byte at
  `hidden + n_block*(BN/32) + lane/4`.
- Readiness publication is compiled only with in-kernel Combine. All waves
  execute the workgroup release and barrier before lane zero closes the local
  m-block. The close counter is monotone only within that m-block operation and
  is self-cleared by the unique closer; it is not a per-route or cross-rank
  close table.
- The closer broadcasts through a dedicated LDS word. Threads `0..BM-1`, one
  row per thread, decode the route metadata already in LDS and issue at most
  one remote system arrival each. Lane zero must not serially loop over BM
  rows.
- Combine waits with `ready >= topk*combine_generation`; lane zero uses the
  greater-than primitive with target minus one. There is no acquire fence after
  the wait. Payload and scale loads themselves use cache value 19, avoiding a
  per-item L2 invalidate. BF16 output stores use rank-local cache value 2.
- Arrival counters are initialized once and never cleared. The Combine
  generation is independent of dispatch generations that did not publish
  arrivals. Omitted tokens are written to the current target before the epoch
  gate, so graph replay may safely grow a later batch. The cross-rank launch
  gate prevents the next launch's single-buffered P2P writes until every rank
  has finished the prior launch's Combine and entered the next generation.

## Combine work geometry

The shared Combine item emitter derives geometry from runtime
`run_tokens` and `nominal_waves=launch_grid_ctas*cta_waves`; `run_tokens` is not
a compile-time specialization added solely for Combine.

```text
safe_tokens = 1 if run_tokens == 0 else run_tokens
P = ceildiv(nominal_waves, safe_tokens)
if block_fp8:
    scale_blocks = hidden_i32 / 8
    P = min(P, scale_blocks)
    hidden_i32_per_wave = ceildiv(scale_blocks, P) * 8
else:
    hidden_i32_per_wave = ceildiv(hidden_i32, P)
total_items = run_tokens * P
```

For block-FP8, `hidden_i32=hidden/4`; for BF16 it is `hidden/2`.
Item `i` maps to `token=i/P`, `partition=i%P`, starting at
`partition*hidden_i32_per_wave`. Tail length is
`min(hidden_i32-start,hidden_i32_per_wave)`.

Reduction width is fixed by the item emitter:

```text
U=1                                  when hidden_i32_per_wave <= 895
U=1                                  when hidden_i32 % 256 != 0
U=4                                  when width > 895 and width % 256 == 0
U=2                                  otherwise when width > 895
```

The U-wide step first issues all top-k payload and scale loads for every `u`,
then broadcasts scale bytes, performs top-k FP32 sums, and emits independent
stores. Per `(expert,u)`, each lane loads one i32 payload; one lane per
eight-lane group loads one i8 scale and broadcasts it within that group.
`u>0` payload offsets advance 256 bytes and block-FP8 BF16 output offsets
advance 512 bytes. A U1 tail handles the remainder.

For the measured MTPR-8192 class, all three acceptance shapes launch 256 CTAs:

```text
tokens  nominal waves  P   hidden_i32/wave  U  total items
128          2048      16        112         1      2048
512          2048       4        448         1      2048
8192         2048       1       1792         4      8192
```

At 8192 each eight-wave block claim therefore covers eight distinct output
tokens, not eight partitions of one token.

## LDS and register budget

Shared memory is reused by lifetime, not added by role:

```text
role_pool = max(gemm1_pool, stage2_slab * stage2_halves)
group_segment = role_pool + gemm1_scale_storage
```

For the large NW8/BN512 path:

```text
gemm1_pool          = 131072 B
stage2_slab         = 131728 B
stage2_halves       = 1
gemm1_scale_storage =  28672 B
group_segment       = 160400 B
gfx950 limit        = 163840 B
headroom            =   3440 B
```

The allocation must therefore remain `max(role lifetimes) + scale storage`.
Allocating G1 and G2 slabs additively exceeds the hardware limit. Every helper
LDS address is relative to the caller-provided slab base; no helper may assume
offset zero. The 131728-byte large slab is exactly:
131072 compute bytes, `BM*4` packed-route bytes, `BM*4` weight bytes,
`world_size*8` payload-peer bytes, `world_size*8` readiness-peer bytes, and a
16-byte readiness broadcast tail.

Measured bs=128 final-path compiler anchors are:

```text
workgroup threads   = 256 (4 waves)
group segment       = 32144 B
kernarg segment     = 440 B
VGPR                = 168
SGPR                = 106
VGPR spills         = 0
SGPR spills         = 106
private/scratch     = 0 B
```

Measured 8192 final-path compiler anchors are:

```text
workgroup threads   = 512 (8 waves)
group segment       = 160400 B
kernarg segment     = 440 B
VGPR                = 256
SGPR                = 106
VGPR spills         = 80
SGPR spills         = 116
private segment     = 316 B
resident workgroups = 1 per CU
```

These are emitted-resource envelopes, not exact source-contract values. The
large measured path has controlled spill/private storage, so a blanket
zero-spill rule is wrong. Controlled is not free, though: anchoring G1's last
K-step A offset to the per-tile row base (author guide, codegen) cut the 8192
VGPR spills from 53 to 20 and private storage from 216 to 84 B, and was worth
about 1%. For every bucket, collect group-segment, VGPR, SGPR,
spill, private-segment and occupancy metadata. Require LDS within the gfx950
limit, at least one resident workgroup per CU, and no unexplained growth beyond
the corresponding measured envelope. Constructing Stage2 descriptors in the
outer Stage1 frame unnecessarily extends SGPR live ranges into combine and must
be avoided.

## Geometry and compile-time specialization

- G2 chunk, Stage1/G2 tile geometry, dispatch CTAs, grid multiplier, work
  shards, payload rows, transport, preemption modulus and skew ratio are the
  fixed bucket values above.
- Native Stage2 uses all Stage1 waves on one tile and one slab. Widen BN by the
  wave ratio; never run two lockstep four-wave halves in the timed target.
- The complete Python compile key contains the full hashable Stage2 and Combine
  specifications plus every Stage1 argument. The emitted identity includes
  dispatch mode, Stage1 tiles/waves/grid/dispatch/cache/shards/payload mode,
  Stage2 halves/BM/BN/preemption/chunk/skew, Combine presence, wait-timer
  presence, fused-quant presence, and every LDS-affecting option.
- Runtime Combine `P` and `U` do not create a host-specialized kernel identity.
  Diagnostic no-publication, timer, and fused-quant variants must have distinct
  identities and must not populate the timed cache entry.

## Compiler pitfalls

- A source-level two-launch declaration is not emitted-artifact evidence.
- Do not move Stage2 descriptors out of the single-unit body or Combine
  resource lists/U-step out of the decorated item frame.
- FlyDSL rewrites dynamic branches and loops; values created only inside a
  rewritten branch must not be consumed outside it without explicit carried
  state.
- A workgroup barrier must be reached uniformly. Per-wave combine items contain
  no workgroup barrier; barriers belong to the surrounding block-claim loop.
- Names, comments, metadata attributes, host-only work specifications, and a
  separately reimplemented fused body do not satisfy a device semantic.
- Do not infer safety from VGPR/SGPR counts alone. Validate the exact emitted
  artifact and execute the all-six-slot bs=128 path before broader testing.
- FlyDSL rewrites runtime `if`/loops only in the kernel body and in `@flyc.jit`
  helpers; a plain nested closure runs as Python at trace time. Thread-0, lane-0
  and valid-row side effects belong in a `@flyc.jit` helper or the kernel body.
  Never emulate them with all-lane masked atomics or whole-wave spins plus
  fences; that multiplies atomics and polls by 64-512 per event.
- Never split a 16-byte vector copy into scalar stores to get past a backend
  legalization error; restore the baseline operand form (wave-uniform resource,
  dtype, offset type). Scalarized shared copies also slow the unfused path.
- Never coarsen the G1 work unit to a whole m-tile, or move the completion
  publish out of the per-stripe iteration, to escape a frame fault. Shrink live
  ranges instead (per-unit helper, descriptors inside the unit frame).

## Performance budget

Measured on the validated profile (8192 uniform, rank-max):

- Frozen four-launch baseline ~4.70 ms; Stage1+Stage2 persistent with a
  separate Combine launch ~4.62 ms; complete fusion with the m-major reference
  order ~4.50 ms (1.0448x). The fusion gain alone is ~0.2 ms, so one expensive
  construct erases it.
- The column-grouped G2 order with `C=1` alone reached ~4.05-4.12 ms
  (~1.14-1.155x): G1 stays at frozen-Stage1 parity, G2 plus transport plus
  Combine fell from ~2.31 to ~1.42 ms, and G1/G2 overlap is ~3%. That ~10% is
  G2 efficiency, not removed waiting.
- Best known, ~3.84 ms (~1.22x): on top of that order, one build stacked, each
  measured same-tree in fresh processes:
  1. the three G1 codegen items in the author guide together: 4.115-4.127 ->
     3.925-3.933 ms (-4.6%), VGPR spills 75 -> 31, private 272 -> 92 B;
  2. the non-blocking G1 payload-spin fill (Scheduling blueprint): -0.7%;
  3. per-workgroup deferred G2 completion (Scheduling blueprint): -0.8%;
  4. the Combine memory order (no acquire after the barrier, cache-19 payload
     and scale loads, cache-2 output): -0.5%, Combine 89-96 -> 83-88 us per
     workgroup.
  This build keeps one G2 head and a coarse pre-Combine barrier. In it G2 is
  ~44 units of ~29 us per workgroup, 82% of that in the K-loop; by the stage
  meter (which itself adds ~2.7%) the per-workgroup busy sum is about the e2e
  time, so the kernel is near work and bandwidth bound. Remote P2P traffic costs only ~0.11 ms. The G2 K-loop
  waits `vmcnt(2)` before every BK256 back-edge, so a two-deep W2 prefetch is
  the open lever.
- G2's B and A-scale loads issued before the barrier and the next A DMA was
  worth ~3% on a build that lacked it; current Stage2 already hoists them.
  The G1 items also speed the unfused path, because GEMM1 code is shared.
- Combine readiness is not decisive once G2 reuses W2: an n-major build reached
  ~1.154x with per-token readiness, and a column-grouped build reached ~1.155x
  with a coarse pre-Combine barrier, where per-row readiness measured ~10%
  slower.
- Single costs that erase the fusion gain: two lockstep Stage2 halves
  +0.53 ms; per-block L2 writeback instead of cache-19 write-through P2P
  stores +0.41 ms; skew preemption on a uniform route +0.30 ms (it saves
  0.51 ms on skew);
  quantization work-stealing atomics +0.30 ms; a nested G2 loop around the
  GEMM2 body spills (~2.6x). Stage2 plus Combine measured separately ~2.08 ms.
- Hot-path budget per unit of work: G2 continuation 0 atomics and 0 polls; new
  G2 chunk 1 claim and 1 wait; G1 stripe 1 claim and 1 system increment; Stage2
  close at most 1 atomic per n-block plus at most BM remote arrivals per m-block;
  Combine 1 claim per NW items, 1 lane-0 poll per item, no fence.
- Fusion removes waiting (launch gaps, tails, cross-rank barriers), never work.
  The work's cost still moves: the claim order, chunking and codegen inside the
  fused kernel decide how fast each unit runs, in either direction. Attribute
  such changes with the same tree's unfused control; they are not fusion gains.
  A fused candidate that misses the target is attributed per component as work
  versus wait (ablation of one component in a diagnostic build, or per-CTA
  phase timestamps) before any tuning, and a component whose work exceeds its
  frozen counterpart is fixed in place rather than abandoning the architecture.

## Procedure

Run one normal `mode=mega` lane:

1. Analyze the supplied baseline's semantic fingerprint and discover current
   bindings for every stage and state edge. Report mismatches and derived
   replacements to Planner; the workflow owns applicability selection.
2. Lower the topology of the target's bucket and its exact configuration and
   state-machine rules: the two-launch recipe above. A bucket in the
   weight-streaming regime is outside this Skill (see Mechanism). Do not create
   alternative schedules or tuning branches within a bucket.
3. Intermediate WIP commits are permitted and required in the same lane. Each
   authoring turn closes a coherent subset of required failures, commits that
   checkpoint, and continues the same candidate/head lineage. For that
   committed-turn artifact set `candidate_status=authoring`,
   `claim_complete=true`, and every runtime/correctness/performance flag false
   or pending. Never benchmark or promote it until all required implementation
   checks pass; a WIP checkpoint is not a separate candidate or topology.
4. Run the GPU-free contract after the workflow selects this guidance. Its
   required implementation checks reject any claimed-complete checkpoint
   missing the complete host path, flat stripe publication, unified G1/G2
   loop, shared Stage2 body, P2P visibility, progressive readiness, Combine
   item, block-claimed Combine, or bucket rule. The contract validates claims;
   it does not decide applicability. A structural pass makes no runtime claim.
5. In the later hardware phase, compile the exact artifacts, inspect resources,
   validate bs=128, then 512 and 8192, and use the same operator instance for a
   smaller-to-larger graph-generation check with route mutation.
6. Complete the same chain with eight path markers, two launches on the shared
   large-token operator (at most two for an `_mtpr` target), direct
   numerical parity, liveness, the four established route
   guards, and paired rank-max performance on the target guard.
7. Return any concrete failure to this one candidate. Do not replace the
   canonical target with diagnostics, partial fusion, or open-ended search.

## Do-no-harm notes

- Skill-off must preserve the ordinary Mega lifecycle and receive none of this
  knowledge.
- This guidance does not create a reproduction mode, reserved candidate source,
  or alternate lifecycle; it is consumed by the ordinary workflow roles.
- Engineer must not receive a reference path, reference source, local run IDs
  or private candidate identities.
- A structural pass means only that the complete source is ready for hardware.
- A silent scattered fallback, three-launch partial, timeout, device fault or
  missing path marker is not a valid result.
- Timed runs disable fused quantization, no-publication cuts, wait timing,
  profiling, readiness reports and stage-isolation controls.
- Measure against the frozen scattered baseline on the same machine. No
  previously reported latency is a denominator.

## Acceptance

The Skill succeeds only when its independently authored candidate has:

- `path=MEGA` from all eight ranks;
- exactly two launches on the large-token BF16 path, and at most two for a target
  operator built at its own capacity;
- no JIT or device fault;
- `relL2 < 0.10` at 128, 512 and 8192 tokens;
- same-instance graph replay with route mutation and a smaller-to-larger token
  transition, proving generation padding;
- on every route, including `moving-hot-rank`, the first replay after a route
  change within relL2 0.01 of a repeated replay (no stale route metadata).
  Bit-exact repeats need the frozen Stage1 LDS race fixed on every path the
  candidate runs, fused or fallback: the persistent loop broadcasts `work`
  through an LDS word and then overwrites the same word with `has_work` without
  a barrier, so a slow wave reads the wrong tile (repeats differ by relL2
  0.04-0.08 at capacities 32-128). Give the work word its own LDS slot and let
  every wave compute `has_work = work < total_work`;
- emitted 128 and 8192 resource envelopes within the values above and at least
  one resident workgroup per CU;
- no regression on fixed, compact or skew guards;
- frozen-relative 8192-uniform rank-max speedup (frozen `BASELINE_PER_CASE`
  latency divided by the candidate latency, paired against a frozen-tree base
  arm) of at least 1.03x after the established noise allowance. The m-major
  reference order measured 1.0448x, the column-grouped order ~1.155x, and the
  column-grouped order with the stacked levers of the Performance budget
  ~1.22x. A same-tree switch-off pair is a diagnostic only.
  When the workflow names another target guard and minimum, that guard and
  minimum replace 8192-uniform and 1.03x here; everything else above still
  applies, and the speedup must stay above 1.0x.

The runtime validator (`runtime_validation.py`, executed through the
workflow's runtime tool) takes `--target-guard` (default `8192_uniform`),
`--frozen-baseline-ms` (the frozen rank-max latency of that guard) and
`--resource-evidence <json>`, where the JSON is
`{"cases": {"fixed_128": {...}, "large_8192": {...}}}` and each case reports
`threads`, `group_segment`, `kernarg`, `vgpr`, `sgpr`, `vgpr_spills`,
`sgpr_spills`, `private` and `resident_workgroups` read from the emitted
code-object metadata of the persistent kernel. It gates `--min-speedup` only on
`absolute_speedup` (frozen latency / candidate median), reports the same-tree
pair as `incremental_switch_speedup`, requires the path marker on every rank,
and keeps `claim_complete=false` with `claim_blockers` (exit 1) when the
marker, the frozen latency or the resource evidence is missing. The gate never
passes at or below 1.0x, whatever `--min-speedup` is. A `<T>_uniform` target is
timed on the shared operator; a `<T>_uniform_mtpr<M>` target (M = next_pow2(T))
is timed on its own operator built with max_tok_per_rank=M, which must print
the path marker on every rank and emit at most two launches (the same count on
every rank) even when M is at or below `--mtpr-fallback-max`; the shared
operator still emits exactly two. The switch-off control uses the target's
operator capacity. When the run optimizes only the `_mtpr<M>` target and the
shared capacity is a declared FALLBACK bucket, add `--shared-fallback`: the
shared operator must then print `path=FALLBACK` on every rank (any launch count,
the same on every rank) and still pass accuracy and replay on the caller's
routes, the claim's
activation is the target's `path=MEGA`, and the resource JSON carries one
`target_mtpr<M>` case for the target kernel instead of the two envelopes
(metadata reported, at least one resident workgroup; no validated envelope).
The frozen path itself fails the `moving-hot-rank` first-call check at 8192
(measured relL2 0.145 on the frozen tree), so on the FALLBACK operator that
route is recorded under `shared_fallback_diagnostics` and not gated; the target
operator is still gated on every route.

## Sources

- Final measured persistent implementation, used only to distill and calibrate
  this Pattern.
- Public baseline supplied by the invoking task.
- Independent candidate validation produced by the one acceptance chain.

## End of Skill pattern
