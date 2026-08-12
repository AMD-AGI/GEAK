---
key: chunked K-Kt preprocessing for gated linear attention under varlen packing, Triton on gfx950 — headline dominated by a dead batch dimension in the launch grid
type: method
confidence: ★★
effect: 28.9x geomean vs the frozen baseline (per case 3.5x / 56x / 122.6x at B=2/32/64) collapses to 2.45x geomean (3.17x / 2.21x / 2.11x) once both sides run the de-duplicated grid
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill
description: Varlen chunked linear-attention: audit the baseline grid first — most of a 28.9x headline was a B-fold redundant grid; 2.45x when deduped both sides
keywords: ['linear-attention', 'varlen', 'grid-dedup', 'launch-overhead', 'frozen-baseline', 'harness-artifact', 'gfx950']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
---
# Audit the baseline launch grid before believing a large headline
- lever: when varlen packing puts everything in batch=1 tensors but the launcher still issues grid_y = B*H, the batch index is dead and B-fold bitwise-identical programs execute; clamping grid_y to the distinct-work dimension is the single largest lever available.
- apply: derive the grid from the distinct work (grid_y = heads under varlen), pin the launch config on the host, and memoize the compiled-kernel handle so the fast path skips the Python launcher wrapper.
- verify: re-time the golden at both grids and report two ratios — vs the frozen baseline as specified, and vs the de-duplicated golden; the gap between them is the benchmark-construction factor, and it grew ~1.0x / ~26x / ~57x with B=2/32/64.
- pitfall: speedup rising steeply with batch size → the redundancy scales with B while kernel efficiency does not → publish the de-duplicated ratio next to the headline so the win is not read as kernel work.
- caution: also verify the production call site's grid convention (frameworks often pass batch=1 for the varlen path) before assuming the redundancy exists outside the harness.
- source: run chunk_scaled_dot_kkt_fwd_kernel-own16h, 2026-08-12, director-validated (correctness pass, same-window pristine control)
