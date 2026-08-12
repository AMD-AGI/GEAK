---
key: int32 request-to-slot scatter/fill in sglang, tiny grid on gfx950 - reported latency is flat across 32x the work, so the Python launch path and not the device code sets the price
type: lever
confidence: ★★
effect: 2.53x geomean director-verified vs the frozen baseline, non-overlapping (paired 2.51x agrees within 1.1%); per-case 2.48x / 2.56x / 2.55x at B=2 / 32 / 64 - nearly flat, the signature of a dispatch-bound op
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both
description: Dispatch-bound Triton scatter/fill: cached direct-launch object + re-grid + write-through store gives ~2.53x, flat across 32x the work
keywords: ['launch-overhead', 'host-runtime', 'dispatch-bound', 'triton', 'memory-movement', 'grid-fill', 'cache-modifier', 'gfx950']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead', 'device.grid-fill']
---
# Collapse the host launch path first on a dispatch-bound scatter, then re-grid
- lever: cache a direct-launch dispatch object that skips the JIT wrapper and the Python launcher down to the compiled C launch entry (raw int pointers, no launch metadata), then re-grid the device side so the machine actually fills.
- apply: in the Python wrapper beside the kernel: build the handle once, arm an identity-checked hot path with a pre-built launch tuple and a C-level __getitem__; on the device side replace an O(pid) serial prefix-sum with one masked vector load plus a reduction, and expand the grid to batch*CHUNKS grid-strided column tiles, CHUNKS an additive tl.constexpr whose default reproduces golden semantics.
- stack: total 2.53x director-verified geomean = four directions compounded
  - 1. cached direct-launch dispatch object (round 1, verified) - 2.30x standalone, the dominant lever; host enqueue down ~3.2x, to ~1.01x of a predicted raw-C floor
  - 2. serial prefix-sum removed + re-grid (round 1, merged) - 2.33x cumulative; unmeasurable alone (a throwaway host pedestal hid all device time), confirmed by counters instead
  - 3. identity-armed hot path + pre-built launch tuple + C-level __getitem__ (round 2, verified) - 2.41x cumulative, +5.0-5.8% with an A-vs-A null of 0.2%
  - 4. write-through store modifier on the write-once output + invariant prefix test hoisted out of the column loop (round 3, verified) - 2.53x cumulative, device time down ~1.03-1.10x
  - note: attribution is incremental in landing order, not independent.
- verify: paired interleaved A/B against the frozen baseline plus a dispatch-count gate (candidate dispatch count equal to baseline) so no launch is elided or memoized away, and exact-equality parity against golden.
- pitfall: a correct device rewrite measured ~1.00x standalone -> the un-bypassed host path hid all device time -> re-issue device candidates after the launch bypass lands and cross-check with counters (VMEM per wave, workgroup count).
- pitfall: a fast-path guard that drops the ROCm large-storage descriptor check buys speed by risking silent corruption -> keep that check inside the guard; the same hot path also turns an unhashable (list) grid into a TypeError.
- caution: also verify the guard-miss path under fresh per-call metadata objects, which is what the framework does in production: here it ran ~3.5x slower than baseline while the benchmark, reusing objects, could not see it.
- caution: also verify the sign of a cache modifier on this arch - the streaming variant cost +8-13% where write-through gained.
- source: run kernel_20_geak_0808_16h, memory-movement scatter lane, 2026-08-12, director-verified (accepted, correctness pass, dispatch gate pass)
