---
key: reading a compute roofline on a partitioned MI355X-class device where the visible CU count is far below the marketing part
type: instrument
confidence: ★★
effect: Same two large-M cases read as 33% of full-chip nameplate MFMA peak but ~72% of the peak realizable on the visible partition (118 of 256 CUs), leaving ~4% to a pure-fp8 GEMM floor rather than the 67% implied. Campaign evidence: the ceiling landed at pass 14 of 35, and the remaining 21 passes chasing the phantom headroom moved the geomean by under 1% (about 60% of the wall-clock budget for 20.04x -> 20.13x), including six independent re-confirmations of the same ceiling.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: triton-on-rocm
last_seen: 2026-08-11
name: score-against-realizable-partition-peak-not-full-chip-namepl-method-gfx950-n-a
description: Roofline read against full-chip nameplate showed 33% of peak; against the visible CU partition it was ~72%, i.e. the headroom being chased did not exist
keywords: ['roofline', 'nameplate-peak', 'partition', 'measurement-methodology', 'budget-saturation', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
---
# Score against realizable partition peak, not full-chip nameplate
- lever: Before opening another round against 'we are only at X% of peak', recompute the denominator from the CU count the runtime actually reports and scale the part's peak by it; a partitioned device can turn an apparent two-thirds of headroom into a few percent, and that changes the stop decision more than any knob.
- apply: Take the CU count from the device query on the box under test, scale nameplate peak by visible-CU / full-part-CU, and express the result as a fraction of achievable peak in the report; pair it with a stop rule, e.g. treat N consecutive passes within the run-to-run spread as a ceiling rather than as bad luck.
- verify: Cross-check the recomputed fraction against a same-dtype dense-GEMM reference on the same partition: if the op sits within a few percent of that floor, the remaining gap is the reference's gap too, not the candidate's.
- pitfall: an escape route flagged for five waves as 'the only remaining headroom' was finally attempted and banked nothing -> it had been justified by the nameplate gap rather than by a mechanism -> record an attempted-and-empty escape in the ledger so later waves do not re-flag it as fresh.
- caution: The partition ratio is a property of how the box was carved, so also verify the reported CU count per box rather than carrying the fraction across machines, and also verify that the reference floor you compare against is the same dtype and shape family.
- source: 16h per-kernel time-budget campaign, block-scaled fp8 dense-GEMM lane, insights + 35 pass records, 2026-08-11
