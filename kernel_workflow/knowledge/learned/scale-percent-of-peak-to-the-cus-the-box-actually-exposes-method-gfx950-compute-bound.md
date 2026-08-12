---
key: reading percent-of-MFMA-peak on a GPU that exposes only a slice of the physical part, where nameplate peak is the wrong denominator
type: method
confidence: ★★
effect: same per-case measurements moved from ~33% of full-chip nameplate MFMA peak to ~72% of achievable partition peak once the denominator used the 118 of 256 CUs actually exposed, leaving ~4% to the pure-fp8-GEMM floor on the two larger cases instead of an apparent 67% of headroom
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound
description: On a partitioned GPU, percent-of-peak against full-chip nameplate under-reads by the CU ratio; rescale to the exposed CU count before calling the gap headroom
keywords: ['roofline', 'percent-of-peak', 'partition', 'measurement', 'mfma', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: compute-bound
layer: learned
lifecycle: active
---
# Scale percent-of-peak to the CUs the box actually exposes
- lever: Before treating a low percent-of-peak as headroom, take the CU count the runtime reports and divide the marketing peak by the fraction of the physical part you were given.
- apply: Read the exposed CU count from the device query, form the ratio against the full part's CU count, and use that fraction of nameplate as the roofline ceiling for every fraction-of-peak claim in the run's report and in any card it produces.
- verify: Cross-check the rescaled fraction against an independent instrument (the campaign's own empirical roofline for the same case) and against the arithmetic floor of the pure operation; a plausible number on two ceilings is the confirmation.
- pitfall: A compute-bound kernel looked to be sitting at a third of peak with plenty of room -> the ceiling had been computed from the whole physical part while the run held under half of it -> rescaling put the kernel near its realizable ceiling and correctly closed five subsequent rounds of chasing.
- caution: The exposed-CU ratio is not the only shrink between nameplate and realizable peak (clock state and the operation's own arithmetic floor also cut it), so also verify the rescaled ceiling against a microbenchmark of the bare operation before declaring a kernel finished.
- source: run gemm_a8w8_blockscale-ch16h, 2026-08-12, 16h per-kernel time-budget campaign on gfx950
