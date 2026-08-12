---
key: trying to out-generate the shipped vendor assembly for bf16 dense GEMM on gfx950, once the op already routes to the tuned library
type: anti-pattern
confidence: ★★
effect: closed axis: five generators raced per-case against the shipped solution on the smallest case all lose - hand HIP 0.878x, vendor ASM library ~0.93x, Composable Kernel 0.834x, TileLang 0.63x, re-driven assembly backend 0.960x; four consecutive rounds returned ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound
description: For mid-size bf16 dense GEMM on gfx950, five independent code generators all land below the shipped vendor solution; that axis is closed, not underexplored.
keywords: ['dense-gemm', 'bf16', 'gfx950', 'vendor-library', 'codegen', 'anti-pattern', 'split-k', 'occupancy', 'tile-geometry', 'roofline']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
---
# Once it routes to tuned vendor assembly, out-generating it is a closed axis
- lever: Spend the round elsewhere: after a tuned library backend is in place and the op sits at high MFMA duty, the remaining named lanes here (rewriting the kernel in another generator, decomposition, fill/occupancy, tile padding) each returned ~1.00x or worse.
- apply: Before opening such a lane, race one cheap instance of the candidate generator against the shipped solution on the worst case only; if it lands under 1.0x there, the full port will too - all five here ranged 0.63x-0.96x of shipped.
- verify: Race per-case in one process against the shipped path, with parity checked on every arm (cos 1.000000) and proof the alternative binary really loaded; a generator that fails to load times as the incumbent.
- pitfall: Two plausible causes of the small case's gap - tile padding waste and sub-full occupancy - were eliminated by construction and the result got worse: a zero-padding exactly-full-occupancy tile measured 4.7% slower than the shipped 3.1%-padded one, and a resource-verified custom kernel with both fixed hit 0.878x.
Rebuilding the vendor library from its source tag is not a superset of what is installed: the shipped winner exists in no tagged logic file and the re-derived same-geometry control was 8% slower.
- caution: This is measured for one op/dtype/arch at high MFMA duty; also verify your own case is really inside vendor assembly (check duty per case - here 82%/82%/50%) before assuming the same ceiling, since a low-duty case can still have headroom of a different kind.
- source: run _gemm_a16_w16_kernel-own16h, 2026-08-12, kernel_workflow 16h campaign, rounds 8-10 dead-end ledger
