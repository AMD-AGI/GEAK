---
name: narrowing-operand-storage-re-opens-the-non-temporal-and-occu-attention-gfx950-decode
description: Treat a storage-dtype narrowing as invalidating the surrounding tuning: re-swept non-temporal and occupancy knobs added 1.057x and 1.019x on top of 1.053x
keywords: [dtype-dialect, non-temporal, occupancy, vgpr, decode, l2-locality, operand-reuse, memory-bound]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: narrowing KV storage to fp8 with in-register upcast on a Triton grouped decode-attention kernel (gfx950) whose KV set is streamed once per call
lifecycle: active
type: lever
confidence: ★★
effect: Three chained rounds on the same body: KV storage narrowed to fp8 with in-register upcast back to bf16 for the math paid 1.053x (cumulative 3.137x); a conditional occ4 launch_bounds enabled only on the narrowed path paid 1.057x (3.161x) after the narrowing itself had already dropped registers 176->144 (auto occ3) and occ4 then dropped AGPRs 12->0 with no spill; and a one-line non-temporal/streaming KV load flag paid 1.019x (3.181x) on the largest case even though the SAME flag had measured -7% on that case under bf16 storage — the working set had halved and, even halved, was never L2-resident. End state cumulative 3.181x, per-case 3.626x / 3.127x / 2.839x on the three decode cases c2 / c32 / c64, best validated pass 3.3232x. Numerics held at SNR 31.37dB. The narrowing paid only as HBM-read byte-saving: feeding the narrow dtype to the matrix core at the SAME MFMA K-width (x32) measured -2% and added upcast VALU on the operand that was not stored narrow.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
last_seen: 2026-08-11
---
# Narrowing operand storage re-opens the non-temporal and occupancy knobs that were closed under the wide dtype
- lever: When a decode kernel's KV/weight set is streamed once per call, treat a storage-dtype narrowing as an event that INVALIDATES the surrounding tuning rather than as a standalone win: after it lands, re-measure the non-temporal / streaming load flag and the occupancy bound, because both optima can move and a knob that regressed under the wide dtype can flip sign once the set no longer fits L2.
- apply: Keep the arithmetic at the wide dtype and upcast in registers — the payment is on the HBM read, and pushing the narrow dtype into the matrix core at an unchanged K-width buys nothing; gate the streaming-load and launch-bounds changes on the storage dtype so the wide-dtype path stays byte-identical, then re-sweep.
- stack: three chained directions on the same body, cumulative 3.137x -> 3.161x -> 3.181x, individually 1.053x (fp8 KV storage), 1.057x (conditional occ4 launch_bounds, pays only because the narrowing freed registers) and 1.019x (non-temporal KV load, largest case); attribution is incremental in landing order, not independent.
- verify: Confirm the narrowed working set is genuinely larger than L2 before expecting the non-temporal flip, and count AGPRs alongside VGPRs when reading the occupancy change.
- pitfall: an accuracy gate tripped after the narrowing -> the cliff came from casting the softmax probabilities, not from the K/V storage -> check which operand carries the numerical floor before blaming the storage dtype.
- caution: Also verify the accuracy gate against the actual numerical floor of the narrow format rather than the gate's nominal threshold.
- source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
