---
id: megamoe_small_mtpr_fusion
title: MegaMoE EP8 small-batch single persistent kernel (dispatch + GEMM1 + GEMM2 + combine)
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
  isolated_speedup_min: 1.10
  e2e_delta_min_pct: 1.0
  parity: required
role: experimental_authoring_prior
normative: false
normative_scope: explicitly_pinned_seed_recipe
planner_extension_file: planner_extension.yaml
contract_file: contract.yaml
runtime_validation_file: runtime_validation.py
validation_file: validation.yaml
---

## When to use

One measured solution for the **weight-streaming regime** of MegaMoE V2: an EP8
gfx950 node, `max_tok_per_rank` (mtpr) of the operator about 32–128, all 48
local experts active, so the floor is reading every expert weight once
(≈1.68 GB per rank, ≈244 µs at the measured 6.9 TB/s HBM read ceiling).
Measured at tokens = mtpr = 64: frozen baseline 0.3458 ms (4 launches:
quantize, stage1 dispatch+GEMM1, stage2 GEMM2, combine) → **0.2946 ms
(1.174x, official bench median)**, graph time ≈291 µs, one launch
(quantization fused; keeping it as a separate launch costs ≈2–3 µs).
Only 64 is measured (paired graph A/B: 1.19x). 32 and 128 lie in the same
regime by the first-order model, but the v3.5 build does not run there: its
single-kernel path asserts `stage2.block_m == stage1.sort_block_m`, a
non-persistent and nontemporal Stage2, and the frozen 128 and 256 bucket
configurations fail that (measured: AssertionError at both). Extending the
recipe to 128 needs a bucket configuration that meets it, then measurement. At
128 the large-token two-launch structure measured 0.877x, so 128 belongs here
rather than there; at 256 it measured 1.035x.

Inside that range this Skill is a **seed recipe**: reproduce it first (it is the
fastest known point), then search beyond it. It is not normative: the Planner
may keep, re-tune or replace any step once the reproduced candidate is
measured. Outside the range (T ≤ 8 latency regime, T ≥ ~2048 compute +
communication regime) use only its mechanisms and the workflow's
`MEGA_REGIMES_GUIDE`; what transfers is decided by measurement.

Analyze reports the current semantic fingerprint and bindings; the workflow
owns applicability. Source names and file layout below describe the validated
baseline and are discovery hints, not requirements.

## Validated profile and semantic fingerprint

- 8 ranks, 384 experts (48 per rank), topk 6, hidden 7168, inter 3072,
  SwiGLU limit 10.
- BF16 external input and output; activations MXFP8 per 1x32 with E8M0
  scales; weights MXFP4 per 1x32 with E8M0 scales (pre-shuffled). Weight
  precision and the observable math stay unchanged.
- Baseline stage1 is a persistent kernel: a ticket gives each CTA a role
  (owner / payload producer / compute), producers P2P-write MXFP8 rows and
  scales into expert slots on the destination rank, the owner publishes a plan,
  and all CTAs then claim GEMM1 tiles from a work counter. Stage2 (GEMM2 with
  route weighting and P2P publication of the weighted rows) and combine (wait
  for peers, top-k FP32 reduce, BF16 store) are separate launches.
- Bucket configuration at mtpr 64 (stage1 `sort_block_m 32, tile_n 128,
  tile_k 256`, 8 waves; stage2 `BN 256, BK 256`, no persistence, NT loads,
  `p2p_quant = none`). The fused path applies only to buckets with
  `p2p_quant = none` and `stage2.block_m == stage1.sort_block_m`.

## Mechanism

At most two launches per call. The stage1 persistent kernel absorbs GEMM2 and
combine (tail); input quantization may stay a separate kernel before it or be
absorbed into the front end (optional lever, ≈2–3 µs, bit-exact; the seed does
it, giving one launch). Every CTA of a fully resident grid (2 CTAs per CU,
512 CTAs) runs one state machine:

1. **Front end (≈20 µs).** Producers (optionally) quantize their BF16 rows
   in-kernel (bit-identical to the 1x32 quantizer), prefetch route data before the epoch
   gate, issue local row reads before remote atomics, and send each token's
   payload row once per destination rank (not once per top-k slot). The owner
   sends per-expert route counts early (epoch-tagged, parity double-buffered),
   so the destination writes its metadata before payload completion flags
   arrive. The plan is one i64 word `(epoch << 32) | num_valid`; one CTA per XCD
   acquires it and forwards it to a per-XCD line.
2. **GEMM1 (bulk of the time, HBM-bound).** Each XCD owns a contiguous
   m-major slice of the work space (all n-tiles of an expert on one XCD, so A
   stays in that XCD's L2). Tiles load only their valid rows. Each tile issues
   its B (weights) before the A gather; the K loop keeps 2 K-steps of B in
   flight; at K-step 24 of 28 the CTA claims its next unit and the last two
   K-steps issue that unit's first B. Every byte of weight is read once with
   per-workgroup contiguous chunks and nontemporal loads.
3. **GEMM2 with no grid barrier.** Each GEMM1 tile adds to a per-m-tile
   counter (one 64 B line each); the tile that reaches the threshold sets the
   m-tile's ready word to the launch epoch. GEMM2 units sit after the GEMM1
   units on the same XCD counter, so a CTA that runs out of GEMM1 work moves
   straight on. A GEMM2 unit issues B and its route metadata first, then waits
   for ready inside the tile, invalidates only L1 (A was written by the same
   XCD), and loads A with a row mask. The K loop keeps 3 B half-tiles in
   flight. Each XCD's last m-tile runs as BN/2 units to shorten the drain.
4. **Combine tail.** Each CTA takes a ticket at exit; the last one signals
   every peer. The last `FCR = 256` tickets wait for all peers, invalidate L1,
   and reduce the top-k components of each token in a fixed order (FP32) into
   BF16 output — bit-identical to the separate combine.
5. **Launch protocol.** No launch handshake: every launch ends with the
   cross-rank combine exchange, so a rank entering launch N+1 proves all peers
   finished N−1. Every cross-rank buffer written before the plan is
   parity-double-buffered. The owner clears work counters with several lanes and
   one release before opening the gate, so the first work claim overlaps the
   plan wait. All grid sizes share one epoch slot (each CTA adds `2 // grid_mult`).
6. **Memory ordering (the cheap scopes are a design requirement).** On gfx950
   each XCD has its own L2, so an agent- or system-scope *release* fence writes
   back that L2 and a matching *acquire* invalidates it. Issued per tile or per
   CTA they cost tens of µs. The recipe is built so that almost none are needed:
   - GEMM1 → GEMM2 handoff stays on one XCD (XCD-contiguous slices, GEMM2 units
     after GEMM1 units on the same XCD counter). The producer CTA waits for its
     own stores (`s_waitcnt` 0 + barrier) and then bumps the m-tile counter with
     an agent-scope atomic, with **no release fence**; the counter that reaches
     the threshold raises the ready word. The consumer polls the ready word and
     invalidates **L1 only** (the data is already in the shared L2 of its XCD).
   - P2P rows and peer flags live on the uncached shmem heap, so store
     completion makes them visible. CTAs take the exit ticket after
     `s_waitcnt` 0 + barrier, with **no per-CTA release**; only the single last
     ticket holder does one system release before signalling the peers, and
     reducers invalidate L1 only after seeing all peer flags.
   - A single kernel that still issues a release per GEMM1 tile or per exit
     ticket lands ≈ 75–100 µs above the ledger's single-kernel step; check the
     fence scopes first when step (a) measures far above ≈ 326 µs.

## Procedure

Run one ordinary `mode=mega` lane on the target guard (`<T>_uniform_mtpr<M>`):

1. Analyze the baseline bindings: stage1 kernel and its persistent loop,
   dispatch (producer/owner), GEMM1/GEMM2 tile bodies, combine kernel, host
   bucket selection and workspace allocation.
2. **Host switch first.** `AITER_MEGAMOE_FUSE_ALL=1` (exact string) selects
   the fused path for buckets that satisfy the conditions above; unset or any
   other value must run the frozen path. On the first forward of each operator
   instance print `[megamoe] path=MEGA rank=<r>` when the fused path runs, or
   `[megamoe] path=FALLBACK rank=<r>` when the switch is on but the bucket is
   outside the fused range. The fused path returns the combine output view and
   launches nothing else besides an optional separate quantize kernel.
3. Build in this dependency order, committing each step and checking
   accuracy (relL2 < 0.10 at T = 64 and mixed 1..64 calls) before the next:
   a. GEMM2 + combine fused into stage1 on XCD-contiguous GEMM1 slices, with
      the same-XCD ready protocol and fence scopes of Mechanism 6
      (per-m-tile counter → ready word, no per-tile or per-CTA release),
      whole-grid residency asserted (`grid_mult ≤ 2`, `2 × LDS ≤ 160 KiB`,
      ≤ 256 VGPR). Expected ≈ 326–330 µs; if it is far above, fix fence scopes
      and XCD locality before anything else.
   b. Row clamps (`rows` for GEMM1, row-masked A for GEMM2).
   c. Launch protocol simplification (section 5 of Mechanism), plan word with
      num_valid, per-XCD acquire, L1-only invalidation after ready and in the
      reducer.
   d. Front-end: optionally quantization fused into producers (drops the
      quantize launch), route prefetch, local-read-before-remote, parallel flag
      publication, payload dedup per (token, destination rank), early route
      counts.
   e. HBM pipelining: B before A, lookahead 2, cross-tile B (`xs1`), GEMM2
      B half-tile ring of 3, BN/2 tail split, combine reducers 256.
4. Measure every step with the workflow's paired graph A/B (`MEGA_AB_TOOL`),
   ≥ 3 interleaved passes; keep a step only with a consistent sign. Expected
   ladder (graph µs at T = 64): see the gain ledger below.
5. After the reproduced candidate is verified, search beyond it. The remaining
   items are listed under "Remaining time"; consult the rejected ledger before
   retrying an idea.

## Gain ledger (measured at T = 64, graph time, interleaved A/B)

| step | graph time |
|---|---|
| baseline, 4 launches | ≈344 µs |
| two kernels (stage1 with fused quant + GEMM2 with fused combine), row clamps | ≈329 µs |
| (a separate quantize kernel instead of fused quant costs ≈ +2–3 µs at any step) | |
| single kernel (GEMM2 + combine in stage1), handshake still present | ≈326.5 µs |
| no launch handshake | −2.5 |
| counters cleared before gate, first claim overlaps plan wait | −7 |
| plan word with num_valid + per-XCD acquire | −1.5 |
| L1-only invalidate after GEMM2 ready | −6 |
| reducer L1-only invalidate | −1 |
| gate without acquire, epoch read with sc0\|sc1 | −2.5 |
| no srcmap pad rows, GEMM2 row mask | −1.5 |
| owner multi-lane fast path | −1 |
| GEMM2 setup (peer table, pointers) hoisted before the plan wait | −1.7 |
| combine reducers 64 → 256 | −1.5 |
| payload dedup per (token, destination rank) | −1.0 |
| GEMM2 tail m-tile as BN/2 units (BN = 256 buckets only) | −0.8 |
| early route counts → metadata before completion flags | −1.4 |
| ready counters one per 64 B line | −0.8 |
| GEMM1 B before A gather | −1.3 |
| GEMM2 ready wait inside tile after B issue | −0.7 |
| A-scale issued first | −0.4 |
| GEMM1 lookahead 1 → 2 | −0.7 |
| cross-tile B (`xs1`, claim at K-step 24 of 28) | −1.2 |
| GEMM2 B half-tile ring of 3 | −0.4 |
| **final** | **≈291 µs** |

Also required (correctness, not speed): the stage1 persistent loop broadcast
`work` through LDS word 0, then overwrote the same word with `has_work` without
a barrier; slow waves read the wrong tile (non-deterministic output, relL2
0.058–0.075 in the baseline). Give the work word its own LDS slot and let
every wave compute `has_work = work < total_work`. Apply this to every path
(fused and fallback): output becomes deterministic, relL2 0.0532.

## Rejected ledger (do not retry without a new reason)

| idea | result |
|---|---|
| any early HBM read during the front end (B prefetch before plan, W1 warm-up, G1 preload, MALL prefetch) | +1 to +10 µs: competes with the payload writes |
| work stealing across XCDs (GEMM1 or GEMM2, all units / tail half units) | +1.3 to +8 µs: re-reads A from HBM, needs L2 writeback/acquire |
| deeper GEMM2 whole-tile B buffering | +2–3 µs and VGPR spills |
| GEMM1 lookahead 3 / 4 | +1 to +4 µs |
| claim-ahead at unit start (removes the claim gap, adds no bytes in flight) | +1 to +4 µs |
| claim at K-step 20 (`xs3`) | slower than `xs1` |
| GEMM2 BN 128 / BK 128 in the fused kernel | +5 / +3.7 µs |
| BN/2 split of a BN = 128 bucket | wrong results (BN/2 = 64 tile) |
| split the last 2 m-tiles | −0.5 µs, worse than 1 |
| previous launch pre-opens the next gate | +4 to +7 µs (hot epoch word) |
| combine reducers > per-XCD residency in a two-kernel path | deadlock |
| system release per CTA before the exit ticket | +54 µs (L2 writeback per CTA) |
| agent release per GEMM1 tile before the ready count; GEMM1 tiles interleaved across XCDs | single kernel ≈ 75–100 µs slower than the ledger |
| letting CTAs past the work bound take exit tickets / hit one atomic | hundreds of µs (one hot atomic) |
| write-through GEMM1 stores | +10 µs |
| removing the explicit release before the plan word | +0.9 µs |
| pairing slot atomics, payload before atomics, i64 gate | noise |

## Constraints

- Whole-grid residency is a liveness requirement: reducers and ready waits
  spin. ≤ 256 VGPR and `2 × LDS ≤ 160 KiB` for 2 CTAs/CU (validated: 250
  VGPR, 0 spills, 40304 B LDS). Assert it in the host.
- Cross-rank readiness is monotone (epoch / parity) and graph-replay safe;
  never reset a peer-visible word with a handshake.
- Payload dedup needs `num_valid_max ≥ world_size × mtpr` receive rows and
  route prefetch; otherwise disable it automatically.
- Early route counts need an LDS pool ≥ 8 × total experts.
- Route prefetch needs `run_tokens × topk ≤ producers × waves`; check it on
  the host per call (small buckets have fewer dispatch CUs).
- Cross-tile B and the GEMM2 half-tile ring rely on exact-size weight buffer
  descriptors: out-of-range loads past the last tile are dropped by hardware.
- Mixed calls (1..mtpr tokens on one operator) select different buckets and
  grid sizes; epochs must stay consistent across them.

## Remaining time (v3.5 timeline at T = 64, ≈292 µs)

- Front end → GEMM1 start ≈ 20–21 µs (P2P write floor ≈ 11–12 µs).
- GEMM1 per-XCD tile time varies 36–40 µs and the slow XCD changes per run:
  GEMM1 end differs by up to 23 µs within a rank. Largest remaining item.
- GEMM2 end per rank 273–282 µs; combine ≈ 4 µs after the slowest rank.

## Do-no-harm notes

- Switch off (`AITER_MEGAMOE_FUSE_ALL` unset or not `1`) must time like the
  frozen baseline; only the LDS-race fix may change its bits.
- Outside the fused range the switch-on path must be the original path and
  print `path=FALLBACK`; a silent mixed path is invalid.
- No debug code in the candidate: no timeouts/watchdogs, in-kernel timestamps,
  stage-skip switches or experiment environment variables.
- Measure against the frozen baseline on the same machine in interleaved runs;
  machine drift between sessions is 1–2 µs.
- Engineer prompts carry no reference path, run IDs or private identities.

## Acceptance

The candidate is accepted when the runtime validator (`runtime_validation.py`,
run through the workflow's runtime tool) completes with `claim_complete=true`:

- the target operator (`--target-guard <T>_uniform_mtpr<M>`, M =
  next_pow2(T)) prints `path=MEGA` on all eight ranks and emits the same 1..2
  kernels per call on every rank (`--max-launches`, default 2; this recipe
  emits 1);
- every other operator in the run (the shared one sized for the largest
  accuracy case, and the other `--mtpr-cases`) prints `path=MEGA` or, where
  fallback is allowed, `path=FALLBACK` on all ranks;
- relL2 < `--rtol` at 128, 512 and 8192 tokens and at every mtpr case;
  graph replay with route mutation (including `moving-hot-rank`) for
  `--replays`, first-after-change vs repeat within 0.01. A FALLBACK operator
  skips `moving-hot-rank`: the original dispatch reads the previous launch's
  route metadata there (the frozen tree fails it, relL2 ≈ 0.15 at 8192), so
  its row reads `known_baseline_defect`, which is not a liveness failure.
  Every `path=MEGA` operator must pass that route;
- `absolute_speedup` = `--frozen-baseline-ms` / candidate median ≥
  `--min-speedup` and > 1.0; the same-tree switch-off pair is diagnostic;
- `--resource-evidence` JSON describing every kernel the target operator
  launches, read from emitted code-object metadata:
  `{"cases": {"<kernel label>": {"threads": 512, "group_segment": 40304,
  "vgpr": 250, "vgpr_spills": 0, "sgpr_spills": 0, "private": 0,
  "grid": 512, "cus": 256, "resident_workgroups": 2}}}`.
  Limits: threads ≤ 1024, group_segment ≤ 163840, vgpr ≤ 512,
  resident_workgroups ≥ 1, and `grid ≤ resident_workgroups × cus` when both
  are given. Spills and private segment are reported, not gated.

## Sources

- Measured single-kernel implementation (official bench 1.174x at T = 64),
  used only to distill this recipe and its ledgers.
- Public baseline supplied by the invoking task.
- Independent candidate validation produced by the workflow's acceptance chain.

## End of Skill pattern
