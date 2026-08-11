---
name: re-export-the-launched-symbol-as-a-subscriptable-launcher-ob-quantize-cast-gfx950-memory-bound
description: Re-export the launched symbol as a subscriptable launcher and relaunch with your own tile and warp count: 2.32x, against ~1.15x from all in-body work
keywords: [launch-config, launch-overhead, kernel-cache, quant, memory-bound, tile-shape, roofline, env-switch, config-sweep]
kernels: [_per_token_group_quant_fp8]
platforms: [gfx950]
kernel_class: quantize_cast
regime: memory-bound
key: a memory-bound per-row quantize/cast Triton kernel on gfx950 whose runner resolves an exported symbol and calls it as sym[grid](...) with num_warps=1
lifecycle: active
type: lever
confidence: ★★
effect: 43 consecutive passes of in-body work stayed in 1.058x-1.207x (median ~1.146x); the first pass that reinterpreted the launch jumped to 2.2992x and every later pass stayed 2.2687x-2.3238x, campaign best 2.32x. Per-case at the banked 2.29x state: 1.1944x on the tiny case, 3.0994x and 3.2328x on the two large streaming cases. HBM utilisation 22% -> 62% of nameplate, roofline-emp 0.180 -> 0.580, output bit-exact. Winning inner config was one program per 32x128 row tile with num_warps=4, num_stages=1 and a .cs cache modifier on the quantized store.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
last_seen: 2026-08-11
---
# Re-export the launched symbol as a subscriptable launcher object when the caller's launch config is frozen
- lever: If the runner resolves an exported symbol and calls it as `sym[grid](...)`, the grid and num_warps baked into that call site are a calling convention, not a hardware constraint - and a kernel launched with num_warps=1 and one program per token row leaves most of the bandwidth on the floor. Check for this shape before spending rounds on the kernel body: on a memory-bound elementwise-per-row op the launch reinterpretation was worth ~2x while everything inside the body summed to ~1.15x.
- apply: Export an object whose __getitem__(grid) returns a callable with the caller's exact signature, ignore the incoming grid and warp count, and relaunch an inner jit kernel with a tile size and warp count you choose. Keep the new config env-overridable so the tile/warp/stage/cache-modifier axes stay sweepable without a rebuild.
- verify: Confirm bit-exactness rather than a tolerance gate - re-tiling changes nothing numerically, so any drift means the retile is wrong - and read the achieved fraction of nameplate bandwidth before and after to confirm the relaunch, not the body, moved it.
- caution: Also verify the tiny, launch-floored case separately: it gained least here, so a geomean over the streaming cases will overstate what this buys a small-grid shape.
- source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
