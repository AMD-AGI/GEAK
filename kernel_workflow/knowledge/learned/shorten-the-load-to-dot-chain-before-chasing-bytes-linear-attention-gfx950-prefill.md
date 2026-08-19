---
key: chunked K-Kt / Gram-matrix preprocessing for gated linear attention, Triton bf16 on gfx950, after the launch grid is already de-duplicated
type: lever
confidence: ★★
effect: ≈1.65x on top of the de-duplicated head (frozen-baseline geomean 17.4x → 28.8x); the two large-batch cases (B=32, B=64) carry all of it, the small case stays pinned at ~3.4x by the launch floor
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 13
toolchain: unknown
last_seen: 2026-08-17
name: shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill
description: Chunked linear-attention gfx950: hoist scales out of the contraction, share the Gram matrix across the GQA group, write-through the stores — ~1.65x stacked
keywords: ['linear-attention', 'dependency-chain', 'gqa-head-sharing', 'cache-modifier', 'loop-hoisting', 'mfma-tiling', 'gfx950']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
origin_kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
---
# Shorten the load-to-dot chain before chasing bytes
- lever: the limiter of this class was the load→dot dependency chain, not bytes moved: any per-row scale that factors out of the contraction can be hoisted past the k-loop, and heads sharing a K tile can share one set of MFMAs with the epilogue replayed per member head.
- apply: fold the scale into log2/exp2 form and lift it above the k-loop; pack 2-4 heads of a GQA group per workgroup (single-wave, num_warps=1 once a workgroup owns several heads); mark the write-only output stores write-through; retune matrix_instr_nonkdim/num_stages/BK after the occupancy regime changes.
- stack: total ≈1.65x over the de-duplicated head, five directions compounded
  - 1. scale hoisted out of the contraction — the largest body step; it also made head packing viable
  - 2. Gram-matrix sharing across the GQA group (2 heads/WG) — +4.8% integrated, enabled by (1)
  - 3. write-through modifier on the write-only stores — +6.3%, gated by workgroup count
  - 4. both heads' scale+gate hoisted above the k-loop — +2.0%, zero extra bytes moved
  - 5. deep rewrite: 4 heads/WG at num_warps=1 in a single-wave workgroup — +6.0%
  - note: attribution is incremental in landing order; two patches on the same chain did not add
- verify: ablate each direction separately against the same frozen baseline and check per case — lanes disjoint by dependency chain added and sometimes super-added, while two patches shortening the same chain or touching the same prologue came out anti-synergistic (−1.2% became −3.5%).
- pitfall: deleting the structurally-zero output stores paid +3.5% but narrows the output contract (caller has to hand in the tile pre-zeroed above the diagonal) → true for the harness and the framework call site, still a deviation from the golden → get it signed off before shipping.
- caution: the write-through modifier trades a fixed intercept for byte rate and inverts past the last-level-cache knee, so also verify the workgroup-count gate is refit, not extrapolated, whenever the batch range widens.
- source: run chunk_scaled_dot_kkt_fwd_kernel-own16h, 2026-08-12, director-validated (rounds 1-10, isolated A/B vs frozen baseline)
