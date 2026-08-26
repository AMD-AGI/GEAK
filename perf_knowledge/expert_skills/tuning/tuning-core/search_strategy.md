# Search strategy

How to turn a combinatorial config space into a search you can actually finish, without
losing the winner.

## Prune before you race, not after

A modest Triton GEMM space — `BLOCK_M`, `BLOCK_N` ∈ {16..256}, `BLOCK_K` ∈ {32..256},
`num_warps` ∈ {1,2,4,8}, `num_stages` ∈ {1..4} — is **1600 configs**. Racing all of them
at ~1 s each is nearly half an hour per shape, and most are unbuildable or hopeless.

Two hardware constraints, applied as a filter before anything is compiled, cut it to 912
(57%):

```python
LDS_LIMIT = lds_bytes()        # read from the device -- 65536 gfx942, 163840 gfx950

def viable(BM, BN, BK, warps, stages):
    # operand tiles, bf16, double-buffered when pipelined
    lds = (BM*BK + BK*BN) * 2 * min(stages, 2)
    if lds > LDS_LIMIT:          return False   # won't fit / won't compile
    threads = warps * 64                        # wave64 on CDNA
    if BM*BN < threads:          return False   # fewer outputs than threads
    if (BM*BN)//threads > 256:   return False   # >256 accumulator VGPRs -> spill
    return True
```

The LDS figure must come from the device, and it is the single line in this filter most
likely to be quietly wrong on a new part. gfx950 has 163 840 bytes per workgroup against
gfx942's 65 536; the *formula* above needed no change (re-derived on gfx950, the compiler's
`Required` / `Hardware limit` ratio came back at 1.00 on both parts), so a stale constant
inside a correct formula rejects **28% of the tiles that would have fit** and reports the
result as "no candidate beat the noise floor."

Reading it is not a one-liner across images: `shared_memory_per_block` exists on torch 2.10
and not on torch 2.9.1, so fall back to the GROUP segment in `rocminfo`.
`../tuning-core/arch_migration.md` has the helper.

Then narrow by **regime**, which is the bigger cut: of those 912, only 240 have
`BM,BN ≥ 64` (the compute-bound candidates) and 456 have `BM ≤ 32` (the decode/skinny
candidates). Those sets barely overlap. Searching the right 240 beats searching all 1600.

## Regime determines the answer — measure it, don't assume it

Racing a pruned space across three shapes on one MI300X, bf16:

| regime | shape (M,N,K) | winning BM,BN,BK,warps | throughput |
| --- | --- | --- | --- |
| square / compute-bound | 4096, 4096, 4096 | **256, 256, 32, 8** | 478.9 TFLOPS |
| tall-skinny | 8192, 1024, 1024 | 128, 256, 32, 8 | 310.1 TFLOPS |
| decode GEMV | **1**, 8192, 8192 | **16, 32, 64, 2** | 1.3 TFLOPS |

The compute-bound and decode winners sit at **opposite corners of the space** — largest
tile with 8 warps versus smallest tile with 2. There is no single default that serves both.

This is the central practical fact of GEMM tuning: **you are not looking for the best
config, you are looking for the best config per regime.** Bucket your shapes by regime
first, then search within each.

Also note the decode row's 1.3 TFLOPS is not a failure — at M=1 there is no arithmetic
intensity to exploit; the op is bandwidth-bound and FLOPS is the wrong yardstick. Judge
memory-bound shapes against achievable bandwidth, not peak FLOPS.

## Bucket M rather than tuning every M

In serving, M is the token/batch dimension and varies continuously. Tuning every observed M
is wasteful: neighbouring M values almost always select the same config.

Bucket to a small ladder — e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 —
tune each bucket, and let lookup round to it. N and K, by contrast, are fixed by model
architecture: enumerate the actual (N,K) pairs from the model rather than sweeping them.

Get the boundaries right at the small end. M=1 (pure decode), small M (batched decode), and
large M (prefill) are genuinely different problems; the transitions between them are where
bucketing errors cost the most.

## Order the search by expected payoff

When a full race is too expensive:

1. **Start from the tool's heuristic pick**, not from scratch. hipBLASLt's `heuristic` mode
   and a library's default config are informed starting points; measure that first as the
   number to beat.
2. **Coordinate descent** — vary one parameter at a time over its range, fix the best, move
   to the next. Cheap, and usually lands close to the exhaustive winner. Tile dimensions
   first (largest effect), then warps/stages, then the rest.
3. **Escalate to exhaustive** only for shapes that matter — the ones dominating runtime.

Change **one variable at a time** when hand-tuning. Three simultaneous changes that yield a
win teach you nothing you can transfer.

## Parallelize, and beware cross-GPU variance

Most tuners take a multi-process flag (`--mp N`) to race candidates across GPUs. Two cautions:

- Only use GPUs you have verified are idle; a busy one produces slow candidates that get
  wrongly rejected.
- Timing on different GPUs is not perfectly comparable. Where a tool offers it, group all
  candidates for a given shape onto one GPU (aiter's tuner calls this `--shape_grouped`) so
  that comparisons within a shape are apples-to-apples, and parallelize *across* shapes
  instead.

Watch for fork storms: some tuners spawn a process per candidate, and a large space × many
GPUs can overwhelm the host.

## Know your noise floor before believing a ranking

The repeat-measurement spread on this box is ~9% for an unchanged kernel (see
`measurement.md`). When the top several candidates land within that band, they are tied —
picking "the winner" among them is picking noise. Either take more samples to separate
them, or accept any of them and move on.

Corollary: a tuner's own reported ordering is a single sample per candidate. Re-measure the
top few before committing.

## When to stop

- **Gate met** → stop. Define the target before starting.
- **Three consecutive iterations under ~2%** → plateaued. More config search is not the
  lever; the kernel needs a structural change.
- **Near a hardware limit** — MFMA utilization saturated, or at achievable bandwidth for a
  memory-bound op → stop.
- **Register pressure at an occupancy boundary** → restructure, don't re-search.

## Cache and reuse results

Racing is expensive; do it once. Every tool here can persist its winners — Triton's
`cache_results`, FlyDSL's `FLYDSL_AUTOTUNE_CACHE_DIR`, the CSV/JSON databases used by the
library tuners. Persist them, and record alongside each entry: GPU architecture, library
version, dtype, and layout. An entry without that provenance cannot be safely reused, and
tuned artifacts never transfer across architectures.
