# GEMM small-M / decode: grid fill, split-K, reduce, stream-K

The decision page for decode-shaped GEMMs (M from 1 to a few hundred, weight stream dominant). The
regime itself is in `gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound`; this page
says which fill lever to take and how to check it in two timings. Verified on Triton 3.8 / gfx950
(fp8 block-scale and MXFP4 GEMMs, CUDA-graph replay, cold weights) unless marked otherwise.

## Decode GEMM time = launch floor + busiest CU

- **The launch floor:** a trivial output-writing kernel timed like the op (`floor1`), two in one
  graph (`floor2`); the room for a kernel change is measured against it. Room = measured − `floor1`; pass `hw_budget.py --floor-us <floor1>` (its HBM floor omits it).
- **Waves counted as `CTAs / CUs`, not occupancy slots:** on the gfx950 decode tiles measured, a second
  co-resident CTA added little (shared load path), and a grid just over a multiple of the CU count
  measured as a cliff; a grid at or just under it avoided that. Re-check where a decode tile is not
  load-path bound.
- A streaming probe's CTA count is a poor guide to a GEMM grid size: in the decode measurements the
  pure stream needed several CTAs per CU, while an async-ring tile saturated near one CTA per CU.

## Split-K decision rule (small-M GEMM)

A natural starting candidate is the smallest legal tile that keeps one M-tile (`BLOCK_M >= M` where the
MFMA allows) — one candidate among others (rule 3). Let
`t = cdiv(M,BM) * cdiv(N,BN)`, `k_iters = cdiv(K, BK)`, `CUs` from `extract_sku.py --facts`.

**Between ~1/2 and ~3/4 CUs (bf16 GEMMs measured): an exact-fill re-tile beat split-K; A/B both.** This band
was measured on bf16 only, not on the fp8 block-scale / MXFP4 GEMMs the rest of this page covers. Measured on
gfx950 / Triton 3.8.0 (bf16 decode at M of a few hundred under graph replay, and bf16 prefill): split-K at
the same tile, split-K at a larger tile sized to one wave, and a minimal (untuned) stream-K all lost to an
exact-fill re-tile (a non-pow2 `BLOCK_N` where the pow2 tiles miss), and overfilling past one wave lost too.
At `t ≈ 3/4 CUs` and above, split-K lost on the fp8 block-scale and MXFP4 decode GEMMs measured on gfx950
(reduce dispatch + partials).
Re-tile candidates: smaller `BLOCK_M`, non-pow2 `BLOCK_N`, a K split over the warps of one CTA, a
larger `BLOCK_K`. In the one case tried, re-tiling an under-half grid *into* this band lost to split-K
at the big tile (A/B per shape);
stream-K at decode: `## Stream-K and persistent grids at small M`.

1. **`t >= ~3/4 CUs`: split-K measured slower in every case tried.** The GEMM part did not get faster;
   split-K added the reduce dispatch and fp32 partial traffic, and tile / stage tuning was where the
   remaining time went. Scope: measured under
   graph replay with a separate reduce and short K loops (about 7-16 `BLOCK_K` iterations). The learned
   card `kernel_workflow/knowledge/learned/split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md`
   reports `KS=2` paying at about one CTA per CU with a fused in-kernel reduce and a K loop several
   times longer; with that shape (long K, no second dispatch) A/B `KS=2` before closing the direction.
2. **`t <= ~1/2 CUs` and long K: split-K paid in the decode cases measured.** `KS = floor(CUs / t)`,
   keeping about six or more `BLOCK_K` iterations per split; with a large `BLOCK_K` and few K-iterations
   in total (common for fp4) each split gets little work and the reduce cost is expected to dominate
   (reasoning, not measured — A/B it).
3. **Filling with K, with N, or with `BLOCK_M < M` are all candidates; each has a cost to price.**
   Shrinking `BLOCK_M` below M re-reads the weight once per M-tile; at decode M (one to a few tens of
   rows) that dominant stream is expected to lose to split-K (reasoning; not measured at those M). At M
   of a few hundred, with the weight L2/MALL-resident, an
   exact-fill tile with `BLOCK_M < M` (CTA count at a multiple of the CUs; a non-pow2 `BLOCK_N` built
   from split accumulators, e.g. 64 + 16) runs with no split and no reduce — A/B it against split-K
   before splitting. Shrinking `BLOCK_N` re-reads the activation once per N-tile: cheap only while
   `M*K` is tiny next to `N*K`.
4. **Price the partials:** `2*KS*M*N*4` fp32 bytes vs weight bytes `N*K*w_bytes` (the larger-M, short-K
   cases measured lost to their partial traffic).
5. **`M == 1`: a reduce-free kernel beat split-K on the M=1 decode GEMMs measured** — no
   cross-workgroup K split at all: one CTA per narrow N slice, K split across its **warps**, one
   intra-CTA sum, no workspace, no second dispatch.

## Split-K reduce: separate flat dispatch vs fused last-arriver

- **A separate 1-D reduce beat the fused form on the graph-replayed decode GEMMs measured** (gfx950):
  contiguous `(KS, M*N)` fp32 partials, one flat tile per program. Its details — every plane's load
  issued before the sum, no power-of-two padding of KS, fixed k order (deterministic), L2-allocating
  (not streaming) partial stores — are design choices that were not separately A/B'd.
- **The captured second dispatch is cheap but not free:** measure it as `floor2 - floor1`. It is the
  same order as the time left above the floor, which is why rule 1 and rule 5 exist.
- **Fused last-arriver reduce lost on graph-captured decode GEMM** (small M, gfx950). Scope: that
  case only — on attention split-KV (`attention.md ### In-kernel split merge on gfx950`) an in-kernel
  last-arriver merge has won, and on a prefill a8w8 block-scale split-K GEMM (gfx950 / Triton 3.8.0, graph
  replay, M of 512-2048) the fused reduce tied or won narrowly against the separate one — though split-K
  itself was not the best point at every M there, and the fused attempt returned wrong results at a larger
  tile than the one timed (check numerics per tile). A/B it per shape. The
  partials have to be visible to the last arriver; the relaxed form that passed the value checks (a value
gate is not ordering evidence — `../pitfalls/negative-patterns.md`) used write-through
  partial stores, a per-wave `s_waitcnt vmcnt(0)` before the barrier (on Triton 3.8.0 / gfx950
  `gl.barrier()` emits no `vmcnt` wait for outstanding stores, compile-checked), a relaxed gpu-scope
  ticket atomic, and cache-bypassing loads of the partials. A `release` / `acq_rel` gpu-scope atomic
  also orders them but emits `buffer_wbl2` (whole-L2 writeback), plus `buffer_inv` for `acq_rel`, per
  wave (compile-checked, 3.8.0 / gfx950) — measured far slower.
  Even that relaxed form was slower than the graph-captured separate reduce: the tail
  (ticket → partial loads → store) serializes on one CTA per tile. Re-measure for an eager
  (not graph-captured) caller. The learned split-K card above prefers a
  fused reduce over a torch.sum-style *multi-kernel* reduce; that is a different baseline from the
  single flat reduce kernel here.

## XCD placement for split-K grids

With the default round-robin dispatch on an 8-XCD part (MI300X / MI350X), workgroup `i` runs on XCD
`i % 8`. When `cdiv(M,BM) > 1`, the M-tiles that share one
`(N-tile, K-split)` weight block should land on one XCD; if their pid stride mod 8 is non-zero the
weight block is fetched once per XCD. Remap so each XCD owns contiguous `(n, k-split)` groups with
all their M-tiles (`remap_xcd`-style). With a single M-tile there is nothing shared and the remap is a
tie (measured on one decode GEMM). Gate as in `gemm.md ## Beyond hot loop (L2 / XCD locality)`.

## Stream-K and persistent grids at small M

- **Decode: split-K or a re-tile measured ahead of stream-K** — split-K already fixes fill; stream-K
  adds fixups and cross-CTA sync with no tail to remove, and single-dispatch split-K / stream-K measured
  slower on the bf16, fp8 block-scale and MXFP4 decode GEMMs tried near the launch floor (gfx950). A
  decode shape well above the floor with a real tail is untested. The KB card `perf_knowledge/operators/splitk_streamk_gemm/tuning.md`
  says "far below CU count: stream-K (or split-K for large K)"; a decode GEMM is its large-K case,
  and its stream-K-for-wave-quantization heuristic is the prefill case below.
- **Prefill — re-tiling the fractional last wave** needs no partials and no cross-CTA sync: run the
  whole waves on the big tile and one wave of a half-width tile (e.g. 256x256, then 256x128 for the tail), or pick an exact single-wave
  tile (non-pow2 `BLOCK_N`). On the bf16 prefill GEMM where both were tried, the wave split beat
  stream-K. A tail that survives the re-tile is the stream-K case below; a stream-K loss does not close
  the tail.
- **Prefill stream-K (A/B it per shape):** the case where it paid had a 1-WG/CU tile, a large
  fractional last round, partials limited to the boundary tiles and few releasing CTAs
  (release/acquire cost grows with the number of releasing waves). It lost when the grid was under one
  round of big tiles (every tile split, fp32 partial traffic larger than the quantization it removes)
  and when a second co-resident WG already covered the tail. Other combinations are untested.
  *Verified case (Triton 3.8 / gfx950, blockscale fp8):* the 8-warp 128x256 ping-pong tile on a grid
  of 1.5 rounds (tiles in `(G, 1.5G]`, `G` = CUs): grid = `G` CTAs, each owns one data-parallel
  tile, the `T - G` leftover tiles are K-halved over CTA pairs (same XCD when `(T - G) % 8 == 0`),
  write-through fp32 partials, one relaxed device-scope arrival counter, the second arriver adds and
  stores. Removing it put that tile behind the 4-warp 2-WG/CU tile; with it the big tile won. So
  the tile choice and the stream-K decision are one A/B, not two: a 2-WG/CU tile has no tail to
  remove, a 1-WG/CU tile may need a tail re-tile or stream-K to be competitive at all.

## A check sequence for a decode GEMM

A sequence that worked on the decode GEMMs measured (gfx950, graph replay); each step is a check, not
a mandate — skip or reorder when the signal says so.

1. `floor1` / `floor2` in the task's own timer; note time above the floor per case.
2. Per-shape config buckets (`../method/dispatch-table.md`) with the smallest single-M-tile tile;
   compute `t`, `t/CUs`, `k_iters`, partial bytes.
3. Apply the split-K rule; run the **2-point sweep**: `KS=1` vs `KS=floor(CUs/t)` (add `KS-1` if
   `t*KS` sits right at the CU count), plus, at M of a few hundred, an exact-fill `BLOCK_M < M` tile at
   `KS=1` (rule 3). Keep split only if it beats both beyond noise.
4. `M == 1`: reduce-free kernel. Split grid with several M-tiles: XCD-colocate. Then pipeline depth.
5. A suggested stopping signal: when time above `floor1` (measured in the harness under graph replay, not
   taken from a datasheet) is small next to the dispatch cost, the remaining room is likely in the caller
   (fusing neighbours) rather than in this kernel.

## Verification commands

```bash
python3 $KT/extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json   # CUs, XCDs
python3 $KT/hw_budget.py --hardware-context $EVAL_DIR/hardware_context.json --brief --workload gemm \
  --shapes M=..,N=..,K=.. --dtype fp8 --measured-ms <ms> --floor-us <floor1 us> --grid <t*KS> --dispatches <1|2>
# floors: time `fill[(cdiv(M*N,1024),)](y, M*N)` once and twice in one CUDA graph with the task's timer
# effective KS: out = op(..., skip_reduce=True); out.shape[0]  (wrappers round NUM_KSPLIT to whole BLOCK_K / scale blocks)
# occupancy of the chosen tile: $KT/dump_ir.sh ... then python3 $KT/probe.py measure --dir <ir>/<v>_c<i>
# hw_budget --grid refuses a grid under 3/4 of a wave ("fill the machine first"); split-K paid below about
# half the CU count in the cases measured (split-K rule 2) -- between the two, A/B a re-tile against split-K
```
