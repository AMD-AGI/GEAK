---
key: an op already sitting at its output-store bandwidth roof on gfx950, with the workgroup count already well matched to the CU count
type: anti-pattern
confidence: ★★
effect: all disconfirming, per-case on the store-bound large case: persistent grid-stride +16% to +21% slower (a logically no-op stride factor still cost +16%, so it is codegen), doubling rows-per-program +21% slower, finer 4x4 and 16x16 skip-store did not beat the coarse 2x2 form, VGPR 152->109 with occupancy 3->4 waves gave 0%; on the tiny case forced graph replay gave 0% scoreable gain
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 11
toolchain: unknown
last_seen: 2026-08-12
name: what-stops-paying-once-the-store-pipe-is-the-roof-linear-attention-gfx950-prefill
description: At an HBM-store-bandwidth roof, occupancy lift, persistent/fewer workgroups, finer store-skip granularity and graph replay all measured null or negative
keywords: ['anti-pattern', 'store-bandwidth', 'occupancy', 'persistent-kernel', 'grid-stride', 'graph-replay', 'launch-overhead', 'measurement-methodology']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
---
# what stops paying once the store pipe is the roof
- lever: once the profile says store-bandwidth-bound, put rounds into store POLICY and store BYTES; give the occupancy / workgroup-count / launch-overhead family a cheap probe first, because here every member of it came back null or negative
- apply: cheapest probes: one autotune point with lower VGPR pressure, one stride-factor persistent variant, one finer skip granularity — each is a single config, and each closed its axis in one round here
- verify: attribute per case, not on the geomean, and re-run the losing variant interleaved with the incumbent; a variant that is a logical no-op yet still regresses is telling you the cost is codegen, not the idea
- pitfall: a batched-throughput micro-bench (many launches, one sync) reported graph replay as not beneficial and the benefit gate silently left it disabled, while the scoring harness brackets every call separately -> the two disagree about whether a host launch gap exists -> probe the launcher mode after the real warmup instead of trusting the gate
- caution: an isolated in-process probe showed replay ahead of eager and a fresh short-lived process did not reproduce it, so also verify any launch-overhead win in the same interleaved alternating run as the incumbent; and re-check your own bound class and grid size before assuming these closures carry
- source: 16h single-kernel time-budget campaign (48 passes), 2026-08-11; eight dead_end / partial ledger entries across four waves
