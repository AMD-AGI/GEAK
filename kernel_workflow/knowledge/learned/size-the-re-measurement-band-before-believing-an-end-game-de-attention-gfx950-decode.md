---
name: size-the-re-measurement-band-before-believing-an-end-game-de-attention-gfx950-decode
description: Size the re-measurement band before believing an end-game delta and quote the median pass, not the max: the band here was 3.5% around a 1.225x median
keywords: [measurement-method, interleaved-ab, env-switch, control-experiment, decode, attention]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: judging a sub-noise end-game delta on a decode attention kernel on gfx950, where the pass-to-pass spread of the banked state is wider than the candidate effect
lifecycle: active
type: instrument
confidence: ★★
effect: 59 accepted re-measurements of essentially the same banked state spread 1.1975x to 1.2398x — a 3.5% band around a 1.225x median — while the headline for this kernel is the max, 1.24x, i.e. ~1.2% above the median of its own passes. Two further passes came back 0.9546x and 0.936x and were flagged. Against that band, the campaign's last real lever was worth ~0.7% geomean (carried entirely by the mid case at ~1.9%; the other two cases tied), and it only became visible under an env-swept interleave (9 runs per arm, no rebuild, plus 4 baked confirmations of the winner) — the harness itself reported IMPROVED=false for it.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 2026-08-11
last_seen: 2026-08-11
---
# Size the re-measurement band before believing an end-game delta, and quote the median rather than the best pass
- lever: On a kernel whose remaining headroom is a few percent, establish the pass-to-pass spread of the current banked state first — re-measure it several times and write the band down. Any candidate delta smaller than that band needs interleaved A/B medians rather than a single before/after, and a single-shot harness verdict on such a delta is as likely to be a false negative as a true one.
- apply: Alternate the two variants within one session, ideally selected at runtime by an env knob so neither side pays a rebuild difference, then take medians per arm.
- verify: Re-measure the banked state itself several times and compare the candidate delta against that band before crediting it; the same band tells you what a reported number means, so quote the median when comparing kernels or campaigns.
- pitfall: the harness reported IMPROVED=false on a real lever -> the effect was several times smaller than the single-shot pass-to-pass spread -> re-judge it on env-swept interleaved medians with several runs per arm.
- caution: Also verify how the headline was formed before comparing two campaigns — the max over many passes is an order statistic of the noise and sits above the median of its own passes.
- source: chuschen 16h time-budget campaign run, 2026-08-11
