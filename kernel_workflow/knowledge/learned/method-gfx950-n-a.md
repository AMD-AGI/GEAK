---
key: measuring true per-CU occupancy of a JIT-compiled AMD GPU kernel from its ELF kernel descriptor rather than from a profiler register counter
type: instrument
confidence: ★★
effect: the profiler register count read about 0.5x the ISA-true value, enough to classify the kernel as occupancy-limited on every case; the descriptor read showed it already at the register-bound cap of 3 workgroups per CU on all three cases, turning an expected 1.3x direction into a measured 1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: method-gfx950-n-a
description: Read VGPR/LDS from the .kd descriptor in the loaded JIT object to get true occupancy; the profiler counter under-reported registers by about half
keywords: ['instrument', 'occupancy', 'vgpr', 'register-bound', 'profiler', 'jit', 'elf']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-08-11
---
# Read the achieved occupancy out of the code object before planning an occupancy round
- lever: Before planning any occupancy direction, extract the achieved occupancy from the kernel descriptor note of the AMDGPU ELF inside the loaded JIT shared object, and divide the architectural register/LDS/wave caps by the descriptor values.
- apply: Dump notes from the shared object the runtime actually loaded, select the AMDGPU machine object, and take register count, accumulator count, spill count, LDS bytes and block size for the exact variant dispatched.
- verify: Cross-check the descriptor numbers against the tile shape they imply; if a framework-level minimum-occupancy setting disagrees with the descriptor, the setting is a compiler floor the hardware can exceed.
- pitfall: The occupancy premise came from a profiler register counter that disagreed with the descriptor by roughly a factor of two -> the direction was budgeted as a large win and returned parity -> reading the descriptor first would have retired it before authoring.
- caution: Also verify which variant of the kernel was dispatched before trusting a descriptor read: a tail or odd-size specialization can carry a different register footprint than the main one.
- source: GEAK per-kernel time-budget campaign, chuschen16h lane, 2026-08-11
