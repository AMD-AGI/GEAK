---
key: search and integration discipline for a kernel graded by a weighted mix of decode and prefill shapes, where one case carries most of the weight
type: method
confidence: ★★
effect: a lane recorded closed on the large-prefill launch config re-opened at -22% on its own dispatch when re-swept at the decode shape ten rounds later; the one integrate whose sign did not flip across three batches was the one whose two directions were gated to disjoint cases, while two clean non-overlapping patches that monetised the same dispatch integrated below either alone
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a
description: On a weighted multi-shape mix, a closure inherits the shape it was measured at and two winners compose only when gated to disjoint cases
keywords: ['method', 'measurement', 'closed-axis', 'size-gating', 'stacking', 'launch-config', 'frozen-baseline', 'negative-control', 'gfx950', 'triton']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-08-17
origin_kernels: ['mi355x_vllm_triton_fused_moe_gemma4']
---
# Scope a closure, a gate and a stack by case regime
- lever: Record every closure together with the shape and launch config it was measured under, re-price it per case regime, and predict a stack from case-disjoint gating rather than from how cleanly the patches merge.
- apply: Gate each landed patch on the regime it won on (a host-side size or element-count gate), and when a win needs a new constexpr on a jit helper shared with the dominant case, fork a separate entry so that case's specialization stays byte-identical.
- verify: After an integrate, check the sign per case, not only the aggregate, and carry a byte-identical null arm in the same pool; pool both orders, because a null seated only at the ends of a sextet says nothing about the middle slots the treated arm occupies.
- pitfall: The round gate returned no-improvement six rounds running while three separately reproduced wins were real -> its stored reference was measured in an earlier session and five sessions bracketed it by -2.7% to +1.1% -> re-measure a reference arm every session instead of carrying the number.
A direction reporting no result still held the run's last real win -> the status field was written before the pooled files finished -> check the directory and file sizes, not the field.
- caution: Also verify a gate before widening it: a fused path that paid on the small decode case was roughly a 2x loss when its gate was widened to the mid prefill case, because its cost is linear in element count.
- source: run mi355x_vllm_triton_fused_moe_gemma4-bmk7-12h, 2026-08-17, rounds 3/5/10/12/13/15 plus the report-phase pooled re-measurement
