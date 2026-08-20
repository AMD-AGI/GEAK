---
key: isolated A/B and upper-bound-oracle protocol for Triton kernels on a power-capped many-XCD GPU, where wall-clock tracks energy per unit work rather than cycles
type: method
confidence: ★★
effect: one event pair per launch cut the in-batch control spread from 4.39% to 0.08-0.49% on all three cases; the landed body then re-measured within 0.38% across six independent sessions, and two oracle confounds each forged an effect larger than the round's real win (+17.2% and 2.36x)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a
description: On a power-capped GPU, delete-the-work oracles and batched timers lie: per-launch event pairs and entropy-preserving probes cut in-batch control spread ~15x
keywords: ['ab-protocol', 'measurement', 'power-cap', 'oracle', 'triton', 'gfx950', 'anti-pattern', 'bit-exact']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['method.ab-protocol', 'method.upper-bound-oracle']
origin_kernels: ['fused_moe_kernel_gptq_awq']
---
# A/B protocol and oracle confounds on a power-capped GPU
- lever: Time one event pair per launch, alternating candidate and control inside a single locked process with a duplicate control in every batch; in any 'delete the work' oracle hold the operand VALUES and the address spread fixed and change only the thing under test.
- apply: Build variants by rewriting frozen defaults into the source rather than reading environment knobs at runtime, and assert the post-condition when landing a winner (the canonical tree actually changed), not just that a diff exists and re-measures well.
- verify: The duplicate control in each batch should land within a fraction of a percent of its twin; a control spread of several percent means the timer, not the candidate, is what is being measured.
- pitfall: An oracle that refilled the weight operand with a constant pattern read +17.2% faster on a byte-identical binary -> switching energy is data-dependent under a package power cap -> preserve operand entropy and vary addresses only.
- caution: Also verify a claimed register or energy premise against a counter/ISA dump before funding a round on it, and check that a variant knob is actually reachable at runtime: frozen defaults silently made every env-var knob a no-op for several rounds, and one collapsed-window traffic probe measured cache hotspotting (2.36x slower) instead of traffic.
- source: run fused_moe_kernel_gptq_awq-own16h, 19-round campaign 2026-08-08/09, director-validated 2026-08-12
