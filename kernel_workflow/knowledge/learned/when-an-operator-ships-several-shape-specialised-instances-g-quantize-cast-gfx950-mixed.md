---
key: integrating round winners across the per-shape kernel instances of a per-token-group fp8 quant/cast operator (Triton, gfx950)
type: method
confidence: ★★
effect: Director-verified end state 4.16x geomean (3.40x / 4.83x / 4.39x from the smallest to the largest shape). Across three consecutive integration rounds the entire integrator gain was this cross-instance port and essentially none of it was the stack: porting one epilogue rewrite out of its assigned file was +3.95% and +1.60% on the two shapes it had not been written for, versus +0.31% for stacking three orthogonal patches; a later port was +0.30% and was the whole round's delta; the same port into a third instance measured +0.10% +/- 0.88 and was reverted.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-10
name: when-an-operator-ships-several-shape-specialised-instances-g-quantize-cast-gfx950-mixed
description: Port a winning mechanism into the sibling shape-specialised instances: cross-instance ports paid +1.60-3.95% while stacking three orthogonal patches paid +0.31%
keywords: ['cross-instance-port', 'measurement-method', 'interleaved-ab', 'quant', 'noise-band', 'control-experiment']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: mixed
layer: learned
lifecycle: active
---
# When an operator ships several shape-specialised instances, grep the other instances for the mechanism a direction just found
- lever: Parallel directions partitioned by file scope their mechanism to their own file, so once an operator has been split into per-shape kernel instances the highest-value integration act is not merging patches - it is grepping the sibling instances for the same code pattern and porting the mechanism into each.
- apply: After collecting the round's patches, diff each winning mechanism down to its source pattern and search the other instances for that pattern verbatim; apply it as a separate, individually revertible edit per instance, and re-tune any constant that was tuned against the slower pre-integration body.
- verify: A/B each port on its own instance with a position-balanced (ABBA, or each arm in each position) paired design, not against the round's absolute numbers; keep only the ports whose paired sign is consistent, and revert the ones inside the noise band.
- pitfall: one port measured +0.10% with a wider error bar than the effect -> the third instance did not carry the same slow form of the epilogue, so there was nothing for the mechanism to remove -> keep the port revertible per instance and revert on a sign inside the noise band.
- caution: Also verify that a lane previously declared closed by an ablation bound is really closed before skipping the port: a zero-arithmetic ablation bounds how much work exists, never how well that work is spelled, so it can be blind to one instance carrying a far worse form of the same epilogue.
- source: run kb_on_0810 2026-08-10
