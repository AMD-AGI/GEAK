---
key: Triton @autotune config list on an AMD MFMA chunk-scaled-dot linear-attention kernel, gfx950, small launch-bound cases alongside larger batch ones
type: lever
confidence: ★★
effect: adding the MFMA instruction-size axis (with a wider K-tile) to a config list that omitted it beat the converged autotuner by 1.18x on the mid-size batch case and 1.14x on the largest, production-grid medians of >=3 sweeps each; VGPR 44 -> 28 lifted a 62.5% occupancy ceiling; ~1.0x on the smallest case, which was launch-bound
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
name: audit-the-autotune-config-list-for-a-missing-axis-before-tun-attention-gfx950-launch-bound
description: Audit the autotune config list for a missing axis (MFMA instruction size) before tuning inside it: up to 1.18x over the converged autotuner
keywords: ['autotune', 'config-sweep', 'mfma', 'occupancy', 'vgpr', 'launch-bound', 'interleaved-ab']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: launch-bound
lifecycle: active
---
# Audit the autotune config list for a missing axis before tuning inside it
- lever: Treat the @triton.autotune config list itself as an object to audit, not just its winner: on AMD MFMA kernels `matrix_instr_nonkdim` was absent from this class's vendored config list entirely, and a tuner converges confidently to a non-optimum inside whatever space it was given.
- apply: Enumerate which axes actually appear in the decorator's Config(...) entries, then sweep the missing ones (nonkdim 16 vs 32) jointly with the ones already present (K-tile width) - here they were additive and both beat the converged autotune winner.
- verify: Compare medians of >=3 independent sweeps against the unmodified decorator, because an autotune sweep is itself noisy and re-converges to different winners on identical source; also check the register count moved in the direction the occupancy story predicts.
- pitfall: a minimum-waves-per-EU request cost up to 2.2x -> the kernel was not register-capped, so the knob targeted a regime this kernel was not in -> check a knob's current default and whether the kernel is in that regime before spending a round on it.
- caution: The lesson stopped at the config list - a later enumeration of the whole AMD compile-knob surface found no second knob with headroom - so read this as a one-shot audit of what the decorator omits, and also verify a knob's current default before opening a general compile-knob lane on it.
- source: run kernel_20_geak_0808_4h 2026-08-08
