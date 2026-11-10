# MegaMoE size regimes — what limits each tokens/rank, and which levers fit it

You get this file as `MEGA_REGIMES_GUIDE` in a `mode=mega` run that pins a MegaMoE Expert Skill
(or sets `mega_regimes=true`). It names Skill designs, so a generic run does not get it. It says **which design family fits which size**, so a run at a new
tokens/rank (`T`, the call's tokens per rank; `mtpr` is the operator capacity `max_tok_per_rank`)
can choose a direction instead of re-discovering it. It never replaces measurement: every number
below is a planning estimate or a measurement on one machine, and the run must re-measure what it
relies on (`MEGA_MEASUREMENT_GUIDE` §1/§2 where present; `MEGA_AB_TOOL` for fast paired A/B).

An Expert Skill, when pinned, is one validated solution for one regime. Inside its applicability
range it is a **seed recipe**: reproduce it first, then search beyond it. Outside its range it is
only a reference for mechanisms; the regime table below decides what transfers.

---

## §1 Profile and first-order model (EP8, gfx950, a8w4 MegaMoE V2)

Profile: 8 ranks, 384 experts (48 per rank), topk 6, hidden 7168, inter 3072, MXFP8 activations
(per 1x32, e8m0), MXFP4 weights (per 1x32, e8m0), SwiGLU limit 10.

| quantity | value |
|---|---|
| routed rows received per rank | `6T` (uniform routing) |
| active local experts per rank | `48·(1−(1−1/48)^(6T))`: T=1→5.7, 2→11, 8→30.7, 32→47.8, ≥64→48 |
| weight bytes per expert | W1 22 MB + 0.69 MB scales; W2 11 MB + 0.34 MB scales |
| all 48 experts | ≈1.68 GB → **≈244 µs at 6.9 TB/s** (measured HBM read ceiling) |
| FLOPs per rank | ≈792.7 M·T → ≈0.159 µs·T at ~5 PFLOP/s MXFP4 |
| compute/weight ridge | T ≈ 1500 (weights fully read once) |
| dispatch payload per rank | ≈7 KiB·6T (FP8 row + scales); 8192 → ≈363 MB, xGMI-bound |

Measured anchors (rank-max, CUDA graph, this machine; frozen baseline, 4 launches):

| T = mtpr | frozen baseline | best known fused | where it came from |
|---|---|---|---|
| 64 | 0.3458 ms | 0.2946 ms (1.174x) | single persistent kernel, Expert Skill `megamoe_small_mtpr_fusion` |
| 128 | 0.3612 ms | none faster yet (two-launch 0.877x; single-kernel recipe does not run) | — |
| 256 | 0.4573 ms | 1.035x (graph A/B) | two-launch pipeline, earlier build |
| 2048 | 1.5721 ms | 1.096x (graph A/B) | two-launch pipeline, earlier build |
| 4096 | 2.5075 ms | 1.207x (graph A/B) | two-launch pipeline, earlier build |
| 8192 | 4.6751 ms | ≈3.85 ms (1.216x graph A/B; 1.155x certified run) | two-launch pipeline, Expert Skill `megamoe_ep_mega_fusion` |
| 16384 | 9.0004 ms | 1.194x (graph A/B) | two-launch pipeline, 8192 tuning |

Any other size: **measure the frozen baseline at that size first** (official bench,
`--tokens T --mtpr T`, at least 3 passes, rank-max). Do not interpolate the table.

---

## §2 Regimes

### R1 latency / partial-weight (T ≈ 1–8)
- Only 6–31 of the 48 local experts are touched. The weight read shrinks to ~12–64 % of 1.68 GB,
  so fixed costs dominate: launch gaps, cross-rank handshakes, flag round trips (~1–3 µs each
  over xGMI), plan/count exchange, and the combine exchange.
- Levers: one launch; skip untouched experts entirely (no G1/G2 work units for empty experts);
  fewest cross-rank round trips (fold count exchange into the payload signal, double-buffer by
  parity instead of a reset handshake); keep the whole grid resident; avoid any grid-wide barrier.
- G1/G2 tiling: work units must spread the *active* experts' weights over all CUs; per-expert
  N-stripes become the parallel axis because M is tiny (≤ 6T rows per expert).
- Expect the 64-recipe's dispatch front end (~20 µs) to be a large fraction; measure it.

### R2 weight streaming (T ≈ 32–1024)
- All 48 experts are active; the floor is reading 1.68 GB once (~244 µs), plus dispatch and
  combine latency at the ends. At T=64 the measured best is ~291 µs graph time: G1+G2 ≈ 1.12x of
  the HBM ideal, front end ≈ 20 µs, combine tail ≈ 4 µs.
- Levers (measured at T=64, see the Skill's gain ledger): fuse everything into one persistent
  kernel; read every weight byte exactly once with per-CU contiguous chunks and nontemporal loads;
  start G2 on an m-tile as soon as its G1 stripes are done (per-tile counters, no grid barrier);
  overlap the dispatch front end with the first weight loads only through bytes already in flight
  (do not issue extra early HBM reads, they always cost); fuse the combine into the tail.
- As T grows toward 1024, compute per byte grows (6T rows per expert ⇒ 384 rows/expert at 64,
  6144 at 1024). Larger M tiles become worthwhile and A re-reads start to matter; the dispatch
  payload (7 KiB × 6T per rank) grows to ~44 MB at 1024, so dispatch/combine overlap matters again.
- Middle sizes (256–1024): the large-token two-launch structure measured 1.043x / 1.060x /
  1.050x at 256 / 512 / 1024 (an earlier build); the single-kernel R2 recipe is untested there.
  Measure both families at the size before choosing.

### R3 compute + communication (T ≥ ~2048)
- Compute ≈ 0.16 µs·T (1.3 ms at 8192) and the dispatch/combine payloads (~0.7 ms each way at
  8192, estimate) both exceed the weight read. The baseline is already a pipeline of large
  kernels; gains come from overlapping communication with compute at item level.
- Levers (from `megamoe_ep_mega_fusion`): persistent two-launch pipeline (quant separate),
  stripe-level G1 completion feeding G2, G2 preemption under skew, in-kernel combine after local
  drain, block-FP8 transport.
- Single-kernel R2 tricks that rely on the whole problem fitting one resident wave (one work unit
  per CTA, per-tile readiness by polling) do not transfer unchanged; their costs scale with T.

---

## §3 Rules that hold in every regime (measured, keep)
1. HBM reads reach ~6.9 TB/s only with per-workgroup contiguous chunks and nontemporal loads.
   Flush caches with reads, never with `fill_`. MALL prefetch did not help.
2. Every extra early HBM read (warm-up, preload of the next stage during the front end) cost
   time in every variant tried (+1–10 µs). Overlap through bytes that must be read anyway.
3. Whole-grid residency is a liveness requirement for any kernel that waits on other CTAs
   (counters, flags, reducer tickets). Size the grid to the measured residency (CTAs/CU from
   VGPR and LDS) and assert it; an extra reducer count beyond per-XCD residency deadlocked.
4. Cross-rank readiness must be monotone (epoch/parity) and graph-replay safe: the second call
   must not read the first call's flags. Test with ≥100 repeated calls and with alternating inputs.
5. A shared LDS word read by all waves after a barrier is a race unless every wave re-reads it
   after the barrier that follows the write; give each wave its own copy of control words.
6. Correctness first: accuracy vs the reference at several T (1, 32, 64, 128, 512, 8192 as
   applicable), deterministic output across calls, and ragged/mixed per-rank tokens.
7. Work stealing across XCDs re-reads A from HBM and has lost every time it was tried at T=64.

---

## §4 Measuring a candidate fast (`MEGA_AB_TOOL`)

`MEGA_AB_TOOL` (`tools/mega_graph_ab.py`) runs a paired, rotated CUDA-graph A/B: one torchrun job
per arm per pass, rank-max graph time, per-pass paired deltas vs the first arm, output relL2 /
bit-exactness vs the first arm, `[megamoe] path=` marker counts, retries and dropped-leg records.

```
python $MEGA_AB_TOOL --tokens 64 --mtpr 64 \
    --arm base=<frozen tree> \
    --arm cand=<candidate tree>#AITER_MEGAMOE_FUSE_ALL=1 \
    --expect-path cand=MEGA --passes 3 --frozen-baseline-ms <ms> \
    --work-dir <EVAL_DIR>/ab_<tag> --jit-root <outside every tree> --json-output <file>
```

- Arms: `NAME=TREE[#KEY=VAL...]`; values may contain commas. Use the switches the candidate
  actually reads. Wrap the command in the GPU lock as for every GPU command.
- Graph timing noise is ~±1 µs; an official-bench pass is ±2–3 µs. A change below ~1 µs needs ≥5
  passes with a consistent sign; never accept one pass. The machine drifts by ~1–2 µs between
  sessions: compare only within one interleaved run.
- This tool is for iteration. Acceptance numbers still come from the workflow's Verify
  (`GRAPH_CONTRACT_TOOL`) and official bench.

---

## §5 Procedure for a new size
1. Measure the frozen baseline at the size (≥3 passes) and its per-stage split.
2. Place the size in a regime with the §1 model, then check the model against the measurement:
   the weight-read floor (active experts × 35 MB / 6.9 TB/s) and the baseline's stage split.
3. If a pinned Skill covers the size, reproduce it first and confirm the speedup with
   `MEGA_AB_TOOL`; that is the seed. If not, start from the nearest regime's design family and
   expect to re-tune tiles, grid, and queue order.
4. Search one lever at a time, each with a paired A/B and an accuracy screen; keep the ledger of
   rejected ideas from the Skill and do not retry them without a new reason.
5. Guard the other sizes: a change for one T must not regress the guards (fallback to the frozen
   path outside the candidate's range is allowed when the workflow says so).
