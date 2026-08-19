---
key: bf16 fused post-norm + pre-norm/GEMM chain (residual, row sqrsum, iterative row/col normalisation) authored in TileLang and JIT-compiled for gfx950/MI355, serving one decode and one prefill token count per hidden size under vLLM
type: lever
confidence: ★★
effect: 3.4364x geomean isolated vs frozen baseline, director-verified, no case regressing at any integration step; per case 4.49x and 3.70x on the two 64-token decode cases (hidden 7168 / 4096) and 3.00x and 2.80x on the two 7211-token prefill cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both
description: Split a fused norm+GEMM op into decode/prefill dispatch arms, collapse 3 kernels to 1-2 inside each, then stack the arms: ~3.44x geomean
keywords: ['dispatch-collapse', 'kernel-fusion', 'launch-overhead', 'tilelang', 'decode', 'prefill', 'graph-replay', 'dispatch-floor', 'gfx950', 'stacking']
kernels: ['mhc_fused_decode_tilelang', 'mhc_pre_big_fuse', 'mhc_post_split']
platforms: ['gfx950']
kernel_class: fused_norm_gemm
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-14
roofline: prefill phase 1 ends at ~94% of the measured HBM read roof; decode ends at the graph-dispatch floor
levers: ['host.dispatch-collapse', 'compute.kernel-fusion']
origin_kernels: ['mi355x_vllm_tilelang_mhc_fused_post_pre']
---
# Collapse the dispatch chain inside each shape-regime arm
- lever: When one fused op dispatches a chain of three kernels per call, have each kernel also emit the partials the next one needs (residual plus the dot/sqrsum partials) so the chain collapses 3 to 2 to 1, and give each shape regime its own arm of the host if/elif so the two collapses are orthogonal by construction.
- apply: Keep the arm selection a token-count/hidden-size branch in the host wrapper; fold the small pre-pass body into the large one rather than launching it; the residual re-read round trip disappears with the round trip, which is where most of the prefill gain sits.
- stack: - stack: total 3.4364x geomean isolated (director-verified) = four directions compounded
  - 1. prefill dispatch collapse 3 to 2 to 1 - the 2 to 1 step alone -7.1% and -9.9% on the two prefill cases (rounds 2 and 6, verified)
  - 2. prefill VALU/instruction cut on the surviving arm - cumulative 1.87x to 2.18x, the largest single step (round 4, verified)
  - 3. decode collapse 3 to 2 plus halving the streamed weight re-read (+10%) and a zero-extra-byte load-width change (+15% on top) (rounds 3 and 6, verified)
  - 4. CTA linearisation on a fully resident decode grid - -20.0% and -16.9% on the two decode cases with zero instruction changes (round 7, verified)
  - note: attribution is incremental in landing order; the three arms stacked with plain apply, near-additive at +5.4% over the best individual
- verify: Count kernels per call to confirm the collapse engaged, then stack the arms in speedup order and re-time after each apply, requiring that no case regresses; an attribution run with the fusion gate switched off in the same tree separates two arms that look coupled.
- pitfall: Wrapper-level graph capture and entry-point env hatches bought nothing -> the harness already replays a graph, so only kernels inside it are countable, and the env hatch fell through to the unfused path -> cut dispatches inside the graph instead.
Five rounds paid a re-harvest tax on wins that sat in per-round workspaces -> each workspace is re-inited with a squashed start commit, so a recorded sha is not a durable pointer -> commit the round winner as canonical and cite content-level evidence.
- caution: Also verify that the last collapse still pays once fences appear: pushing decode from two dispatches to one measured 8% / 7.5% worse here even with the acquire fence deleted, and an agent-scope fence carried its own tax.
- source: run mi355x_vllm_tilelang_mhc_fused_post_pre-bmk7-12h, 15 rounds / 33 direction-units, director validation accepted 2026-08-14
