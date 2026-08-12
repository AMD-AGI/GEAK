---
key: Triton paged grouped-query decode attention stage-1 over a paged KV cache on gfx950 — latency-bound, small grid, tuned by launch metaparameters
type: lever
confidence: ★★
effect: Reproduced on a second independent run of the same class: launch-meta alone 1.4008x isolated (num_warps=8, num_stages=2, matrix_instr_nonkdim=16, per-grid stage table 2/3/3) inside a 1.53x director-verified geomean (per case ~1.45x smallest-grid / ~1.67x mid / ~1.51x largest-batch); the earlier run measured 1.2114x of a 1.226x total on a differently-frozen baseline. num_stages 1->2 is most of it whenever the baseline body has no pipelining across its serial KV iterations.
confirms_cited: 3
confirms_blind: 0
losses: 3
attempts: 12
toolchain: unknown
last_seen: 2026-08-12
name: launch-metaparameters-carry-a-latency-bound-paged-decode-att-attention-decode-gfx950-decode
description: Launch metaparameters (num_warps/num_stages/nonkdim) carry 1.21-1.40x of a 1.23-1.53x total on paged grouped decode attention, gfx950/Triton; body adds little
keywords: ['launch-meta', 'num-stages', 'num-warps', 'software-pipelining', 'attention-decode', 'paged-kv', 'occupancy', 'triton', 'bit-exact', 'dispatch-floor', 'roofline', 'lds-bank-conflict']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
roofline: unpipelined/latency-bound -> mixed; largest-batch case ends at ~93% of an ON-BOX-MEASURED read roof (itself ~71% of nameplate) with DRAM reads ~1.005x the KV working set, i.e. closed; smallest-grid case ends at ~10% of the compute roof against a ~3.5% memory floor, i.e. pure serial chain
levers: ['host.launch-meta', 'mem.scalar-page-load', 'mem.address-hoist', 'host.launch-overhead']
---
# Launch metaparameters carry a latency-bound paged decode attention
- lever: Before rewriting the body of a paged decode-attention stage-1, sweep launch metaparameters (num_warps, num_stages, matrix_instr_nonkdim, per-grid-band stage table), then take the bit-exact address arithmetic: uniform scalar page-index load, loop-invariant offset matrices hoisted out of the KV loop, and constexpr-proved elision of mask terms.
- apply: Launcher/decorator change plus address-form changes only, no reassociation: num_warps=8, num_stages=2 (3 for the larger grids), matrix_instr_nonkdim=16, 2-way KV gather unroll; page index read as a scalar and reused across the KV block; drop the block-offset term of the score mask where it is provably in range.
- stack: total 1.53x geomean (director-verified) = 1. launch-meta 1.4008x standalone (round 1, verified) — the bulk; 2. bit-exact address hoist + mask elision +7.65% standalone (round 1, verified) — the largest body win; 3. 2-way gather unroll, landed inside a 1.2053x direction, never isolated on its own; 4. host dispatch collapse to a cached direct-backend launcher ~+1.2% on top (round 3, verified, bit-exact); 5. score-tile hoist +0.16% (round 4, inside spread). Integration of 1-3 gave +7.7% over the best individual. Attribution is incremental in landing order, not independent.
- verify: Interleave candidate and frozen-baseline timings in one process and compare medians per case; a sub-1% launcher-only delta can be reported as not-improved by a single-shot harness verdict. Also diff the ISA/counter census (instruction mix, MFMA and VMEM per wave, wait cycles) between rounds — that is what separates a device-side win from a host-side one.
- pitfall: Three rounds labelled compute/algorithm/lds bought ~+2% -> the per-wave instruction census and wait cycles were unchanged from round 1, so the gain was entirely host-side dispatch -> census the ISA before believing a device-side mechanism story.
- pitfall: A shared-memory-usage vs limit ratio was read as an occupancy pot for a whole round -> the grid offers only ~1 workgroup per CU at every band, so shared-memory size buys nothing -> compare grid size to CU count before pricing any occupancy lever.
- pitfall: Roofline headroom looked large for three rounds -> it was computed against the vendor nameplate -> an in-session bandwidth calibration put the real read roof at ~71% of nameplate and showed the largest case already at ~93% of it, with zero removable bytes.
- pitfall: A verified, body-disjoint host patch was left unmerged for a round and had to be re-derived -> nothing merges a round's non-winning survivor -> merge body-disjoint verified patches the round they land.
- pitfall: The gather-unroll knob was filed as "-19%", then "silently incorrect", then "correct but -38%" across three rounds -> each verdict was measured against a different loop body -> re-price a shared knob whenever the body under it changes.
- caution: Also verify numerics per lever: the max-relative gate behaves as bit-exactness here, so tile-width growth and any KV reassociation fail on worst-element error while cosine stays ~1.0; the launch-meta, address-hoist and dispatch-collapse changes were all exactly bit-identical.
- caution: Also verify where the remaining time is before funding another body round: region controls priced the whole global KV gather at ~0% on the smallest-grid case, with the row-max/running-max serial chain at ~27%, and LDS bank conflicts at ~25% of LDS-active cycles identically across all cases — a per-dot warp-layout artifact rather than a traffic problem.
- source: 16h per-kernel time-budget campaign (chuschen16h wave, 61 passes), 2026-08-11; reproduced on a 2h KB-seeded 4-round run, 2026-08-12
