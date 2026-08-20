---
key: bf16 paged/unified attention at query length 1 (GQA 64/8 heads, head_dim 64) on gfx950/MI355, launched from a Python wrapper over a Triton body
type: lever
confidence: ★★
effect: 1.58x geomean isolated vs frozen baseline, director-verified x3 repeats, distributions non-overlapping per case: 2.04x at batch B=2, 1.60x at B=32, 1.20x at B=64 — the gain falls off as the case approaches its memory roof
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-17
name: dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode
description: Collapse host dispatch first, then per-grid-density launch tuning, mask hoisting and per-regime constexpr clones: ~1.58x geomean on paged decode attention
keywords: ['launch-overhead', 'host-dispatch', 'decode', 'constexpr-promotion', 'paged-attention', 'triton', 'launch-tuning']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead', 'compute.launch-tuning', 'compute.constexpr-promotion']
origin_kernels: ['kernel_unified_attention_2d']
---
# Dispatch collapse first, then per-regime specialisation, on latency-bound decode attention
- lever: On small-grid decode attention the host dispatch path is worth attacking before the tile loop; afterwards specialise the body per grid-density regime rather than globally.
- apply: Cache a pre-bound launcher so the per-call kwarg walk and JIT dispatch disappear, and let it own num_warps/num_stages; add a launch table keyed on program count; hoist the loop-invariant row mask into an additive bias; generate per-regime constexpr-promoted body clones, promoting only kernargs that feed an address (the win is the shortened scalar-load dependency chain, not the marshal).
- stack: total ~1.58x geomean isolated (director-verified) = four directions compounded
  - 1. host dispatch collapse via pre-bound launcher — 1.20x standalone (round 1, verified) — largest single lever; host dispatch fell to about a third
  - 2. launch table keyed on program count — +5.7% on top of (1) (round 2, verified); the optimum is non-monotone in occupancy
  - 3. loop-invariant row mask folded into an additive bias — +3.5% (round 3, verified)
  - 4. per-regime constexpr-promoted clones — +3.3% (round 4, verified), bit-exact vs golden
  - note: attribution is incremental in landing order, not independent
- verify: Re-time each candidate against the frozen baseline and benchmark every leave-one-out subset of the stack — in one round the full stack lost to a 2-of-3 subset.
- pitfall: One global constexpr-promotion subset was anti-synergistic (+2.4% alone, negative inside the stack) → a single promotion set does not fit every grid density → promote per regime from a table re-swept for each (geometry, body) pair.
The launch table went stale after body edits → a bucket kept a row tuned against the old body → re-sweeping after body changes recovered +0.9% at integration.
- caution: Also verify that a pre-bound or monomorphic launcher does not pin the stream at bootstrap: a capturing caller then records an empty graph that replays into a zeroed output with no error, and refreshing the stream slot per call costs a fraction of a percent of host time.
- source: run kernel_20_geak_0808_16h, TechLead report + director validation, 2026-08-12
