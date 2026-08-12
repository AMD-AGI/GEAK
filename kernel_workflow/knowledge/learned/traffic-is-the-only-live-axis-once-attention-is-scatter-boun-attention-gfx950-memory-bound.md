---
key: attention pinned at the read-once paged-block scatter HBM roofline, gfx950 — four non-traffic axes measured closed in one campaign
type: anti-pattern
confidence: ★★
effect: ~1.00x across four directions: native fp8 MFMA on the PV dot 1.006x all-case with the two time-dominant heavy shapes flat at 1.000x/1.004x and only the compute-lightest case +1%; occupancy/launch-bounds re-tune 0.97x; wrapper graph replay ~0.86x on all 16 cases; fp4/mxfp4 KV storage fails the correctness gate outright
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound
description: On attention already at the paged-scatter HBM roofline, compute-precision, occupancy and launch-fusion each returned ~1.00x or worse; only traffic moved
keywords: ['paged-attention', 'hbm-bound', 'anti-pattern', 'fp8-mfma', 'occupancy', 'launch-overhead', 'mxfp4', 'memory-bound', 'gfx950']
kernels: ['paged_attention_large']
platforms: ['gfx950']
kernel_class: attention
regime: memory-bound
layer: learned
lifecycle: active
verified_on: 2026-07-30
roofline: 52-62% of the achievable scatter-limited HBM roofline before and after; the residual gap is inherent read-once random-block scatter with ~zero L2 reuse
---
# Traffic is the only live axis once attention is scatter-bound on HBM
- lever: When the profile puts the op at a scatter-limited HBM roofline, rank traffic-reducing directions first and cap compute-precision / occupancy / launch-fusion at a single cheap probe round each.
- apply: Read the register floor from the code object's ELF .vgpr_count (no spill produced => occupancy hints and wave-per-eu pragmas are no-ops) and the dispatch count from the profile, before assigning a compute or occupancy direction.
- verify: A/B the time-dominant heavy shapes on their own as well as the full mix; a direction that only moves the compute-lightest shape is relieving a tail that is not binding, and its all-case geomean will read as noise.
- pitfall: fp8 on the QK dot missed worst-element tolerance by ~3.6x while the P and V tensors tolerated fp8 fine -> the error localises entirely to the QK dot -> gate the two dots independently so PV-fp8 stays available.
split-KV partition size up cost 25-36% -> doubled live K/V register arrays spill ~21 VGPRs -> the partition axis is bounded above by registers and below by frozen scratch sizing, so it is not a free knob.
- caution: This closure is conditioned on the KV traffic already being halved and on a no-spill register floor; also verify your own roofline position and .vgpr_count before assuming the same four axes are exhausted on your shapes.
- source: run paged_attention_large-ch16h, 2026-07-30 — frozen-baseline isolated A/B, 6 passes over a 16h budget
