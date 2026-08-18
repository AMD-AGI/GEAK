---
key: bf16 vLLM Triton fused MoE (gate/up + down grouped GEMMs plus align/activation/reduce periphery) on gfx950, cache-policy hints on the weight loads and the write-once buffers
type: lever
confidence: ★★
effect: weighted 1.22x -> 1.29x from a vL1D-bypass hint on the first grouped GEMM's weight load (58% of that gain landed in the untouched second GEMM), +5.2% more on the mid prefill case when the gate widened to BLOCK_M=128; write-through on write-once outputs cut those dispatches 4-15% isolated per case and -1.26% +/-0.10 in situ on the mid prefill case, all bit-identical
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both
description: bf16 Triton fused MoE: bypass hint on the first GEMM's streamed weight load plus write-through on write-once outputs; sign flips per GEMM and per M bucket
keywords: ['cache-modifier', 'non-temporal-store', 'moe-grouped-gemm', 'bf16', 'triton', 'gfx950', 'l2-residency', 'm-bucket', 'gated-lever', 'bit-exact']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-17
roofline: decode ends at ~0.9 of the MEASURED achievable roof (the nameplate roof over-states it by ~13%)
levers: ['mem.load-cache-policy', 'mem.store-cache-policy']
---
# Cache policy is a per-buffer, per-bucket decision on a bf16 fused MoE
- lever: Try a vL1D-bypass cache_modifier on the streamed weight operand of the gate/up grouped GEMM, and a write-through store hint on every buffer that is written once and never re-read (activation output, top-k reduce output); both are one argument, output stays bit-identical.
- apply: Gate each hint on the M bucket AND on which grouped GEMM it is applied to, in the host config or a constexpr, so decode and prefill can carry different policies on the same source.
- verify: A/B per case against the frozen baseline and also profile the dispatches you did NOT edit: most of the gain here appeared in the untouched second GEMM via L2/MALL spillover, so an edited-dispatch-only measurement under-reads the lever.
- pitfall: The same bypass hint on the second GEMM's weight load lost 5-16% depending on M bucket despite a correct tile-reuse premise -> its reuse distance differs from the first GEMM's -> gate per GEMM, not per file.
Store hints on the GEMM epilogue accumulator were double-digit losses (streaming -23%, write-through -11% at the largest prefill) -> that buffer is not write-once -> apply store hints only where write-once is proven.
- caution: Also verify the hint actually engaged (ISA or a null arm with byte-identical metadata), and also verify per case: the sign of the same hint flipped with the M bucket here.
- source: run mi355x_vllm_triton_fused_moe_gemma4-bmk7-12h, 2026-08-17, 15-round campaign rounds 2/5/14/15, director-validated (weighted 1.33x, correctness pass)
