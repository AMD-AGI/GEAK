---
key: Triton paged grouped-query decode attention stage-1 over a paged KV cache on gfx950 — latency-bound, small grid, tuned by launch metaparameters
type: lever
confidence: ★★
effect: 1.2114x isolated weighted geomean from launch-meta alone (num_warps=8, num_stages=2, matrix_instr_nonkdim=16); per-case c32 ~1.31x, c64 ~1.35x, c2 ~1.01x (launch-floored); a scalar page-index load added bit-identical +2% on c64; cumulative banked 1.2261x, best validated pass 1.24x over 61 passes.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-11
name: launch-metaparameters-carry-a-latency-bound-paged-decode-att-attention-decode-gfx950-decode
description: Launch metaparameters (num_warps/num_stages/nonkdim) carry ~1.21x of a 1.23x total on paged grouped decode attention, gfx950/Triton; body rewrites add little.
keywords: ['launch-meta', 'num-stages', 'num-warps', 'software-pipelining', 'attention-decode', 'paged-kv', 'occupancy', 'triton']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
roofline: memory-bound at ~0.18 of empirical roof -> compute-bound at ~0.39; c64 sits at ~55-65% of achievable HBM bandwidth, latency-bound at ~1 workgroup per CU
levers: ['host.launch-meta', 'mem.scalar-page-load']
---
# Launch metaparameters carry a latency-bound paged decode attention
- lever: Before rewriting the body of a paged decode-attention stage-1, sweep launch metaparameters (num_warps, num_stages, matrix_instr_nonkdim) and a scalar (non-vector) load of the page-index; that pair took ~1.21x and ~+2% respectively.
- apply: Pure launcher/decorator change plus one load-form change, no algorithmic edit: num_warps=8, num_stages=2, matrix_instr_nonkdim=16; page index read as a scalar and reused across the KV block.
- stack: total ~1.226x weighted geomean = 1. launch-meta ~1.2114x standalone (verified) — the bulk; 2. scalar page-index load +2% on the long-context case on top of (1), bit-identical; 3. occupancy-hint drop ~+0.7% on top (1,2). Attribution is incremental in landing order.
- verify: Interleave candidate and frozen-baseline timings in one process and compare medians per case; a sub-1% launcher-only delta can be reported as not-improved by a single-shot harness verdict.
- pitfall: Lowering num_stages or splitting the KV loop cost 4-5% on the long-context case -> the pipeliner had been overlapping the page-index to K address chain -> keep two stages when adding any loop restructure.
- caution: Also verify numerics per lever: an exp2 substitution for the logsumexp exceeded the 1e-2 max-relative gate here, and a KV block size unequal to the page size broke the scalar-page numeric path, while the launch-meta and scalar-page changes were exactly bit-identical.
- source: 16h per-kernel time-budget campaign (chuschen16h wave, 61 passes), 2026-08-11
