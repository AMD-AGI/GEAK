---
name: size-the-workgroup-per-stage-the-low-k-iteration-stage-wants-moe-grouped-gemm-gfx950-mixed
description: Halve the workgroup size on the low-K-iteration stage of a fused grouped GEMM: +5.4 to +11.4% across batch sizes, 1.26x geomean whole-stack
keywords: [workgroup-size, config-sweep, env-switch, occupancy, moe, interleaved-ab, mfma]
kernels: [moe_gemm_fp8_blockscale]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: per-stage workgroup sizing in a multi-stage fp8 block-scaled fused MoE grouped GEMM (hip-ck codegen) on gfx950, grid-starved through oversubscribed batches
lifecycle: archived
type: lever
confidence: ★★
effect: Director-verified whole stack 1.22x at the smallest scored batch, 1.27x mid, 1.29x at the largest (geomean 1.26); this lever alone, paired one-binary A/B at 5 reps, was +5.4% / +10.8% / +11.4% on those same three batch sizes - the same sign and size at a grid-starved batch (fewer tiles than CUs) as at a heavily oversubscribed one, which is the signature of a per-workgroup fixed cost rather than a traffic trade.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 12
toolchain: rocm7.2.3 / torch2.11.0 / hip-ck-codegen
source: run kernel_20_geak_0808_4h 2026-08-08
last_seen: 2026-08-08
---
# Size the workgroup per stage: the low-K-iteration stage wants fatter waves
- lever: In a multi-stage fused grouped GEMM the stages typically share only the M-blocking (they index the same sorted-token map), so workgroup size, K/N tile and pipeline version can be routed per stage. On a stage whose mainloop runs few K-iterations (small K against KPerBlock), try halving the workgroup size for that stage only: each wave then owns twice the MFMA tiles and the per-workgroup prologue / LDS fill / epilogue is amortised over more math.
- apply: Register a second instance at half the workgroup size for one stage, keep M-blocking identical on both stages so token sorting and M-padding stay byte-identical to the base, and select each stage's instance at runtime from an env var.
- verify: Interleaved A/B/A/B in one binary, medians; check the delta holds at both the grid-starved and the oversubscribed batch; re-profile and confirm MFMA-tiles-per-wave and per-stage kernel time moved - mean occupancy per CU can FALL while that stage gets 10% faster, so occupancy is the wrong confirmation.
- pitfall: the same halving on the sibling stage cost 13.4% -> that stage ran 24 K-iterations, so it had no per-workgroup prologue worth amortising -> read each stage's K-iteration count before carrying the change across stages.
- caution: It is an interior optimum, not a monotone trend - halving once won, halving again fell back to ~1.05x overall; also gate on SNR before timing, since tile / MFMA-shape variants in block-scaled families can compile, run, and still return numeric garbage.
- source: run kernel_20_geak_0808_4h 2026-08-08
