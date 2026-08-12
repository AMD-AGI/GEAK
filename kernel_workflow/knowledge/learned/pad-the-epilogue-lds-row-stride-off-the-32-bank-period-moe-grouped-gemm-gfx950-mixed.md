---
key: fp8 e4m3 blockscale grouped-MoE GEMM with preshuffled B and a CShuffle LDS epilogue, gfx950/CDNA4, CK C++ templates, small-to-large token batches
type: lever
confidence: ★★
effect: +3.34% cumulative isolated (1.3404x -> 1.3858x vs frozen baseline, director-verified); per-case device time -2.91% on the wide tile serving the large-batch cases and -7.85% on the narrow tile serving the small-batch case; conflict/index counter 0.6465->0.2621 wide, 0.5393->0.0983 narrow, instruction counts bit-identical
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed
description: Epilogue LDS row stride at a multiple of the 32-bank period: one pad element collapses the conflict counter, ~+3.3% on fp8 blockscale grouped MoE GEMM
keywords: ['lds-bank-conflict', 'lds-tiling', 'epilogue', 'cshuffle', 'counter-guided', 'moe-grouped-gemm', 'fp8-blockscale', 'gfx950']
kernels: ['moe_stage2']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
cost: L1
verified_on: 2026-08-11
roofline: issue-wait share of wall time -27.8%; still 31-44% of the achievable HBM roof, so no bandwidth wall is near
levers: ['mem.lds-bank-phase']
---
# Pad the epilogue LDS row stride off the 32-bank period
- lever: When an LDS staging tile's row stride is congruent to 0 mod the 32-bank period, every row starts on bank 0; add one pad element to the row so consecutive rows land on different banks.
- apply: Bump the row-length constant in the LDS descriptor of the shuffle/epilogue tile (a 32x128 f32 tile becomes 32x132) and widen the max() that sizes the group segment so the enlarged region actually fits; any nonzero pad mod 32 works, which shows the wide ds_write is decomposed into dword phases.
- verify: Read the bank-conflict / index-active counter ratio before and after, confirm the index-active count itself falls (a ratio alone can be an artifact), and check instruction mix, registers and occupancy are bit-identical so the phase change is the only variable; run a negative control with a pad congruent to the original phase but a larger allocation - if that control also wins you measured a size effect, not bank phase.
- pitfall: The enlarged tile silently did not fit and correctness failed on three earlier pad arms -> the group-segment size was clamped by a max() that was never updated -> widen the sizing expression together with the stride.
The axis looked closed for 22 rounds because it had only ever been probed by latency -> the bank counter had never been read once -> read the counter before declaring an LDS axis closed.
- caution: Counter-to-time is not linear: a follow-up took the same counter a further -89% for only 0.29% of wide-tile time, so also verify the residual conflict is on the critical path before spending another round on it.
- source: run moe_stage2-own16h, round 23, 2026-08-12; director-validated, correctness err_ratio 0.0000
