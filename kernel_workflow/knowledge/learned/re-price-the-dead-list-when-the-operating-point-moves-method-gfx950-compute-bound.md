---
key: search discipline for a long Triton GEMM campaign on gfx950: re-pricing dead-listed knobs once a structural lever moves the operating point, and sweeping the config as a tuple rather than one knob at a time
type: method
confidence: ★★
effect: the round's +6.2% was ~2/3 resurrected knobs: the structural change alone measured 0.99x at M=2048 and 1.02x at M=65536, with the rest coming from split-K 2->1 and num_stages 2->1, both previously dead; a later joint re-tile added +2.1% on all three M cases where 21 one-knob variants had returned ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound
description: After a decode/layout lever lands, re-price the knobs already dead-listed and sweep the config as a tuple: two thirds of one round win was resurrected knobs.
keywords: ['config-sweep', 'dead-list', 'tile-shape', 'num-stages', 'split-k', 'quantized-gemm', 'gfx950', 'triton', 'search-strategy']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: method
regime: compute-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-08
---
# Re-price the dead list when the operating point moves
- lever: Treat the dead list as valid only at the operating point that produced it: whenever a structural change alters register pressure, LDS bytes or per-MFMA work, re-run the previously rejected knobs, and move the tile shape jointly with the decode/layout plan instead of one knob per direction.
- apply: Keep a forced-config hook in the launcher so any (tile, num_warps, non-k-dim, stages, split-k, pre-pass) combination can be replayed without a source edit, then sweep the tuple; a structural change that measures ~1.00x on its own can still be worth taking because it is the carrier for the knobs it re-enables.
- verify: Score the tuple, not the knob: re-measure the isolated A/B for the combination and separately for the structural change alone, so a carrier that looks neutral is not discarded before its dependents are priced.
- pitfall: a one-knob-at-a-time re-pricing across 21 variants concluded the config table was exhausted -> the axis it tested was tile size and warp count independently, while the payoff needed tile and decode layout moved together -> the tuple sweep found +2.1% the single-knob sweep could not see
- caution: also verify the reverse direction: a knob resurrected this way can go dead again after the next structural landing, so re-price rather than assume the new value is now permanent.
- source: run _w8a8_triton_block_scaled_mm-own16h, 2026-08-08 campaign, director validation accepted 2026-08-12
