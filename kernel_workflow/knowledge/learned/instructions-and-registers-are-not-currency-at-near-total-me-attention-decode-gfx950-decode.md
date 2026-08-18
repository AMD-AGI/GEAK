---
key: a bandwidth-saturated Triton decode attention tile loop on gfx950 where instruction count or register pressure is being used as the proxy for a win
type: anti-pattern
confidence: ★★
effect: closed axis on the 83%-weight sliding-window decode case: a ground-up mask-free rewrite cut ~26% of instructions and 8 VGPRs with zero spill and measured 2.6% slower; 6 of 8 slower arms had fewer registers; persistent / multi-item-per-workgroup arms went -0.43%, -0.90%, -6.27% as in-flight items fell with occupancy unchanged; direct global-to-LDS async staging engaged and still lost 3.3%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode
description: When the decode tile loop is ~98% memory stall, instructions and registers are free: a -26% instruction, -8 VGPR rewrite with no spill measured 2.6% slower
keywords: ['anti-pattern', 'closed-axis', 'attention-decode', 'decode', 'memory-bound', 'register-pressure', 'vgpr', 'persistent-kernel', 'lds-staging', 'roofline', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-14
roofline: memory-bound throughout: ~98% of the measured streaming roof on the dominant geometry, and issue slots per tile are ~2% of elapsed cycles per tile
levers: ['compute.instruction-count', 'mem.lds-staging']
---
# Instructions and registers are not currency at near-total memory stall
- lever: Price the stall fraction before funding an instruction-count or register-pressure round: here issue slots per tile were about 2% of elapsed cycles per tile, so the loop is ~98% memory stall and issue slots are not what the case is spending.
- apply: Get the ratio from a static issue-slot count of the tile-loop body against measured cycles per tile; the arms that did pay on the same op deleted bytes instead - a provably dead re-mask whose removal deleted an unpack/select/repack round trip, and a page-sized tile that halved the per-workgroup footprint.
- verify: Re-time an arm whose only change is instruction or register count with byte traffic held constant; if the sign is null or negative with occupancy unchanged, retire the proxy rather than tuning it.
- pitfall: A rewrite with fewer instructions, fewer registers and no spill read as strictly better and lost -> the loop was stalled on loads the rewrite never removed -> attribute the earlier wins to deleted bytes, not to deleted instructions.
An LDS async-staging arm was funded on an occupancy premise -> the compiled object showed register use barely moved and the copy cannot swizzle on the way in -> read the object before believing an occupancy story.
A lever family was pruned for four rounds by a stale in-tree comment rather than a measurement -> re-arming the whole prose backlog came back six for six reconfirmed -> it is worth exactly one round of insurance, once.
- caution: Also verify the stall fraction again per geometry: on the same op's low-parallelism cases a per-workgroup footprint cut (page-sized KV tile, LDS halved, waves/SIMD doubled) still paid +2.3% and +4.9%, so this closure is about the bandwidth-saturated geometry and not about the op.
- source: run mi355x_vllm_triton_unified_attention_gemma4-bmk7-12h, deep_explore round 6 plus rounds 7-11, 2026-08-17
