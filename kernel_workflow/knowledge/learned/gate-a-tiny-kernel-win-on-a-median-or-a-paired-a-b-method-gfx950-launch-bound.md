---
key: accepting or rejecting a tiny dispatch-bound kernel's win on gfx950 when the box throttles bimodally between sampling windows
type: method
confidence: ★★
effect: identical code scored 2.55x / 2.59x / 2.59x / 2.62x on single-shot verifies but 2.62x median-of-12 and 2.99x on a fast window; the paired A/B moved the largest batch case minimum by +12.7%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound
description: Bimodal box throttling rejected a bit-exact tiny-kernel win five times; gate on a median of >=10 samples or an interleaved paired A/B instead
keywords: ['measurement-noise', 'ab-methodology', 'tiny-kernel', 'dispatch-bound', 'launch-overhead', 'frozen-baseline']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: method
regime: launch-bound
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-07-30
---
# Gate a tiny-kernel win on a median or a paired A/B
- lever: For an op whose window is near the timer floor, judge a candidate on a median over at least ten back-to-back samples, or an interleaved eight-pair same-session A/B, rather than one single-shot verify.
- apply: Alongside the ratio, record a mechanism-specific signature noise cannot fake - here the gap between the two larger batch cases collapsing to near zero once the serial chain was removed, while the smallest case stayed flat.
- verify: Re-run the same binary in two different windows; if the two single-shot numbers straddle the gate, the gate is sampling the box rather than the patch, and the paired form settles it.
- pitfall: A bit-exact win was refused five times by the single-shot improvement gate -> whole-process bimodal throttling moves the mid and large cases together while the smallest stays flat -> median, per-case minima and the paired A/B all cleared the same gate.
- caution: Also verify parity with an exact comparison when the output is integer indices; a tolerance-based check can pass an off-by-one that the noise argument would then be blamed for.
- source: run write_req_to_token_pool_triton-ch16h, 2026-07-30, five integration attempts of one patch
