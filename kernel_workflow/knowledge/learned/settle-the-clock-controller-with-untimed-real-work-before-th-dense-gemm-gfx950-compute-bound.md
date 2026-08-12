---
key: measurement conditioning for a compute-bound a16w16 dense GEMM on a power/DVFS-limited gfx950 part, where the timed window is short relative to the clock transient
type: method
confidence: ★★
effect: +5.1% geomean with BYTE-IDENTICAL device code (3.7604 -> 3.9526 self-measured; director-verified end state 3.9603 geomean, per-case 3.56x at M~2K, 4.36x at M~32K, 4.00x at M~64K). The gain concentrates where the timed window is short relative to the DVFS transient: on this box the transient was about as long as the entire measured window on the small-M case.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: settle-the-clock-controller-with-untimed-real-work-before-th-dense-gemm-gfx950-compute-bound
description: Settle the clock with untimed real work before the timed window: +5.1% geomean on a dense GEMM with byte-identical device code
keywords: ['measurement-method', 'control-experiment', 'dense-gemm', 'compute-bound', 'interleaved-ab', 'harness-artifact', 'counters']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
lifecycle: active
---
# Settle the clock controller with untimed real work before the timed window
- lever: On a power/DVFS-limited part, the opening of a timed window can be spent at a clock the controller has not yet settled. Issue a fixed slab of UNTIMED real work (the same op, same shape key) on the first call for that key, so the timed window samples steady state instead of a ramp.
- apply: Memoise per shape key inside the wrapper and run the real op back-to-back for a slab at least as long as the timed window itself, with NO idle gap before returning; the device code is untouched. Slab duration was flat within 0.27% across a ~10x range of slab lengths from there up, while every inserted idle gap tried — from one comparable to the timed window upward — was strictly worse than gap=0 on every case — the mechanism is 'no idle before the window', not boost banking.
- verify: Re-measure with the conditioning removed under the same >=8-run interleaved per-case-median protocol, and publish worst-of-all beside each median; confirm the device code is byte-identical (empty ISA diff) so the delta cannot be a codegen change.
- caution: Also verify the baseline is measured under the same conditioning before scoring the ratio, or state the asymmetry in the report — this changes the measurement regime, not the kernel, and an unconditioned denominator flatters it. Also verify no SMU clock polling is running during authoritative runs: polling perturbed the measurement monotonically in poll rate (+2.6% mean, ~4.6x the stall rate at a 1 kHz poll).
- source: run kb_on_0810 2026-08-10
