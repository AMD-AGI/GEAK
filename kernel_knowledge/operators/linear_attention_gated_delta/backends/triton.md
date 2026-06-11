---
title: linear_attention_gated_delta on Triton — SOTA card
kind: sota_card
operator: linear_attention_gated_delta
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/fla-org/flash-linear-attention
  - https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
  - https://vllm.ai/blog/2025-09-11-qwen3-next
---

# linear_attention_gated_delta × Triton

## TL;DR
Triton is the **SOTA and production** Gated-DeltaNet path on AMD. The reference is
`fla-org/flash-linear-attention` (FLA, used by HF transformers for Qwen3.5); the **on-box production port**
is aiter's `aiter/ops/triton/gated_delta_net/` + `_triton_kernels/gated_delta_rule/{prefill,decode,utils}/`.
SGLang auto-detects hybrid layers and runs these GDN kernels with `--attention-backend triton`. There is
**no hand-tuned CK/asm Gated-DeltaNet kernel** on AMD as of 2026-06 — the chunked-scan + triangular-solve
structure is exactly what Triton expresses well, and it's the path AMD ships day-0 for Qwen3.5.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter GDN (prefill chunk + decode recurrent) | `ROCm/aiter@a6bb49937:aiter/ops/triton/gated_delta_net/gated_delta_rule.py`, `_triton_kernels/gated_delta_rule/{prefill,decode,utils}/` | gfx942/950; bf16/fp16 | no public per-shape number; the day-0 Qwen3.5 path on MI300X/325X/355X | Qwen3-Next/3.5, Kimi-Linear serving (inference, fwd-only) |
| FLA reference | `fla-org/flash-linear-attention` | CUDA-written; runs on AMD via Triton ROCm backend | reference; HF uses it for Qwen3.5 | training (has bwd) + research; parity oracle |
| sglang GDN integration | `sgl-project/sglang:.../attention/linear/kernels/gdn_flashinfer.py` + hybrid KV cache | gfx942/950 | day-0 Qwen3.5 (SGLang/vLLM) | sglang serving |

## Config space / knobs
- **Chunk size C** (prefill, ~64), state tile (d_k×d_v) sized to LDS, `matrix_instr_nonkdim=16`,
  `num_warps=4` (avoid 8), `num_stages=1`, `waves_per_eu=2–3`, `knobs.amd.use_buffer_ops=ON`.
- Decode is launch/bandwidth-bound: maximize grid, `num_warps=2–4`. See [tuning.md](../tuning.md).
- `use_qk_l2norm_in_kernel=True`, `cu_seqlens` for varlen, `initial_state`/`output_final_state` for the
  prefill→decode handoff.

## Numerics / parity
fp32 state accumulate; chunk-boundary state must match the recurrent path; aiter kernel is **forward-only
(no grad)** — train with FLA. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- sglang: `--attention-backend triton`; the GDN layer is auto-detected for hybrid models.
- HF/vLLM: FLA kernels via the model's linear-attention layer; install the Triton ROCm backend.
- The `@triton.jit` GDN kernels are clean Python rebind seams; overlay a tuned config and confirm via
  `TRITON_PRINT_AUTOTUNING=1`.

## Pitfalls & anti-patterns
- **Decomposing the scan** (S → HBM each step) → 10–50× slower. Use the fused kernels.
- FLA kernels carry CUDA tile defaults — re-tune for CDNA3 (wave64, 64 KB LDS, mfma_16x16).
- aiter GDN is inference fwd-only; don't wire it into a training graph.
- bf16 MFMA type-confusion corrupts the recurrent accumulate — verify the intrinsic matches the dtype.

## How to verify
Prefill time ~O(T/C), decode ~O(T); state parity vs FLA fp32 reference; greedy temp=0 parity ≥10 prompts
(some long); `AITER_LOG_MORE=1` to confirm the GDN kernels engaged.

## Alternatives / cross-links
[overview.md](../overview.md) · [hip.md](hip.md) · [tilelang.md](tilelang.md) · languages: [[triton_amd]] ·
ops: [[causal_conv1d]] · [[cumsum_scan]] · backend: [[aiter]] · [[sglang_kernels]].

## Sources
- aiter GDN kernels: `ROCm/aiter@a6bb49937:aiter/ops/triton/gated_delta_net/`, `_triton_kernels/gated_delta_rule/` (on-box).
- FLA: https://github.com/fla-org/flash-linear-attention
- AMD Qwen3.5 day-0 (SGLang auto-detect GDN, `--attention-backend triton`): https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html
- vLLM Qwen3-Next (FLA Triton kernels, hybrid KV cache): https://vllm.ai/blog/2025-09-11-qwen3-next
