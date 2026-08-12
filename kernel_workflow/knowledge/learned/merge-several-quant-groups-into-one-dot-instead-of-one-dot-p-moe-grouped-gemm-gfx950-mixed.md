---
key: dequant granularity vs dot granularity for group-quantized weights in a fused-MoE grouped GEMM (Triton) on gfx950, small- and large-batch cases
type: lever
confidence: ★★
effect: This axis alone re-verified 2.24x against a 1.94x incumbent (+15%) and composed near-multiplicatively with the dequant axis to 2.69x the same round; the run ended director-verified 2.92x geomean, per-case 2.49x on the small-batch case (2 tokens/expert class) and 3.14x / 3.18x on the two large-batch cases. Barriers per k-chunk 34 -> 2, loop instructions -46%, max relative error 0.0136 -> 0.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: merge-several-quant-groups-into-one-dot-instead-of-one-dot-p-moe-grouped-gemm-gfx950-mixed
description: Merge G quant groups into ONE dot instead of one dot per group on a packed-weight MoE GEMM: 2.24x vs a 1.94x incumbent (+15%), 2.92x geomean end state
keywords: ['dequant', 'quantization-group', 'mfma', 'moe', 'isa-check', 'interleaved-ab', 'tile-shape']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
lifecycle: active
---
# Merge several quant groups into ONE dot instead of one dot per group
- lever: When a group-quantized weight forces a small k-tile (tile_k == the quant group size), keep that tile as the DEQUANT granularity but accumulate G groups' dequantized operands and issue a single dot over the merged K per chunk, so the layout conversions and their barriers are paid once per chunk rather than once per group.
- apply: Dequantize G groups into one [tile_m, G*group_k] operand with the per-group scale folded into the dequant arithmetic (this keeps it bit-exact), call the dot once, and then re-sweep tile_n / num_warps / stages.
- verify: Count s_barrier and MFMA in the disassembly before and after (the barrier count should collapse while MFMA count is unchanged), and take the verdict from a paired in-process A/B that re-times the incumbent as an in-batch control, since a case can be bimodal by 2-3% within one session.
- pitfall: the merged body first measured worse than expected at the incumbent's tuned knobs -> the pre-merge knob optimum inverted once the dot spanned the merged K, and the previous best tile_n became ~1.9x worse -> re-sweep tile_n / num_warps / stages as part of the same direction rather than reusing the incoming config.
- caution: Also price the reverse move before funding any manual dequant/MFMA pipelining inside the merged body: splitting the merged dot back out cost 23-27% here at constant instruction mix (an n-split changed only dataflow; a k-split even freed 16 VGPR), and the compiler already interleaved dequant with MFMA inside the single region.
- source: run kb_on_0810 2026-08-10
