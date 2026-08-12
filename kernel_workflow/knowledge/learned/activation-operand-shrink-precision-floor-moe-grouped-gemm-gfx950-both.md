---
key: shrinking the activation operand of an already-fp4-weight grouped GEMM on gfx950 under a cosine parity gate, Triton 3.6.0
type: anti-pattern
confidence: ★★
effect: Three disconfirmed sub-directions off a 42.2x incumbent, none landed: direct fp4 activations reach ~+13% (~45x cumulative, all cases) but sit at a cosine of 0.9883 against a 0.99 gate; fp6 activations do not compile or lower at all; outlier-preserving mixed precision tops out ~+5% at an unsafe error margin.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-11
name: activation-operand-shrink-precision-floor-moe-grouped-gemm-gfx950-both
description: Anti-pattern: once weights are fp4, shrinking the activation operand is closed three ways — parity floor, no fp6 lowering, no outliers to protect.
keywords: ['fp4', 'fp6', 'mixed-precision', 'parity-gate', 'outliers', 'activation-quant', 'dot-scaled', 'moe', 'grouped-gemm']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
verified_on: 2026-08-11
---
# activation-operand-shrink-precision-floor
- lever: Cache the shrunk-activation variant behind a flag and cost it against the parity gate rather than retiring it: it is a ~+13% option that re-opens the moment the gate loosens slightly or the real data grows per-channel outliers.
- apply: Measure the parity metric of the shrunk activation operand BEFORE writing the kernel variant: a pure numeric-floor check costs minutes, and it decided all three sub-directions here.
- verify: Check the frontend and the target actually lower the narrower format end to end; a format the language accepts in one place can still have no instruction lowering, which reads as a mysterious build failure rather than an unsupported feature.
- pitfall: An outlier-preserving mixed-precision split looked like free headroom -> the harness activations are i.i.d. Gaussian, so the max-abs channel is random and the split protects nothing -> check the operand's actual channel distribution before budgeting a round for it.
- caution: Also verify whether the narrow operand is consumed natively by the matrix instruction before proposing a cheaper unpack: consumed natively, there is no software unpack left to make cheaper, and the apparent lever does not exist.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, ledger dead_end entries, 2026-08-11
