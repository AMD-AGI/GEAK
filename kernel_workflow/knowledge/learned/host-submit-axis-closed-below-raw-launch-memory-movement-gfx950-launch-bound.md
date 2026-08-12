---
key: exhausted host-submit and launch-knob directions for a dispatch-dominated pool-write op on gfx950, measured against a raw driver-launch seed
type: anti-pattern
confidence: ★★
effect: All disconfirming on the same op, per-case identical across batch 2/32/64: graph capture/replay 1.322x and native ctypes graph-launch 1.2487x both sit far below the 2.613x raw-launch seed; trimming the Python replay bracket 1.2814x (<1% over graph); native pybind submit shim 2.61x on top of 2.613x (neutral); persistent/doorbell scored 0 (round-trip exceeded a raw launch and the harness device-wide sync hung the resident kernel). Launch knobs likewise flat: num_warps/num_stages 1.000x, BLOCK_SIZE 1024 0.998x.
confirms_cited: 2
confirms_blind: 0
losses: 2
attempts: 14
toolchain: unknown
last_seen: 2026-08-12
name: host-submit-axis-closed-below-raw-launch-memory-movement-gfx950-launch-bound
description: Once a raw driver launch is in place, graph capture, doorbell/persistent kernels and native submit shims all measure neutral-or-worse on tiny ops.
keywords: ['launch-overhead', 'host-dispatch', 'hip-graph', 'persistent-kernel', 'latency-bound', 'memory-movement', 'triton', 'closed-axis']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# host-submit-axis-closed-below-raw-launch
- lever: Treat the submit-side axis as spent once a raw driver launch is measured: further host-side rewrites on an op this small landed inside the timer floor, so a round is better aimed at the remaining device-side dependency chain.
- apply: Before opening another submit-side direction, measure the empty bracket and the current candidate against it; if the delta you are chasing is smaller than that floor, the measurement cannot resolve it whatever the patch does.
- verify: Re-run each of these as a paired A/B against the raw-launch seed rather than against the original wrapper - measured against the slow original they all look like wins, which is how a 1.3x graph result reads as progress after a 2.6x seed already landed.
- pitfall: A resident/doorbell kernel appeared to score 0 -> the harness issues a device-wide sync that deadlocks against a kernel that never exits -> check the harness sync model before authoring anything that stays resident.
- caution: Also verify this on your own shapes before reusing the conclusion: the axis closed here because the kernel body is tiny relative to submit; on an op with real device work a graph or a persistent form may still have room.
- source: 16h per-kernel time-budget campaign, 62 resumed passes, gfx950, 2026-08-11
