---
title: vLLM × AITER integration on ROCm — custom-op registration & torch.compile
kind: backend
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://github.com/vllm-project/vllm/pull/16752
  - https://docs.vllm.ai/en/stable/design/custom_op/
---

# vLLM × AITER integration

## TL;DR
AITER (`ROCm/aiter`) is wired into vLLM through **`vllm/_aiter_ops.py`**, which wraps AITER kernels as
**PyTorch custom ops** registered with `direct_register_custom_op` (with fake/meta impls). That registration
is what keeps the hand-tuned AITER kernels **opaque through `torch.compile`** — Inductor fuses *around* them
instead of decomposing them into generated Triton. Everything is gated by the `VLLM_ROCM_USE_AITER*`
hierarchy ([overview.md](overview.md)). This card is the "how AITER plugs in and survives compilation" view.

## Concepts
- **AITER** = AMD's unified operator library (Triton + CK + hand-tuned ASM), framework-agnostic. vLLM is
  one consumer; SGLang another.
- **`_aiter_ops` / `rocm_aiter_ops`**: the module that gates (`is_mla_enabled()`, `is_mha_enabled()`,
  `is_fused_moe_enabled()`, `is_linear_fp8_enabled()`) and registers AITER wrappers as `torch.ops`.
- **`direct_register_custom_op`**: vLLM helper that registers a Python function as a `torch.ops` op **with a
  fake (meta) implementation** so it's traceable by Inductor but **not decomposed** — the kernel stays a
  black box through `torch.compile`.
- **CustomOp dispatch**: on ROCm vLLM dispatches to `forward_hip()`, falling back to `forward_cuda()` if
  unimplemented (`docs/design/custom_op`).

## Registered AITER ops (examples, PR #16752)
`rocm_aiter_ck_moe`, `rocm_aiter_fmoe_fp8_blockscale_g1u1`, `rocm_aiter_asm_moe`, `rocm_aiter_topk_softmax`,
`rocm_aiter_shuffle_weight` (Fused-MoE V1). Plus linear/GEMM (`_rocm_aiter_w8a8_gemm`), routing
(`_rocm_aiter_topk_softmax/sigmoid/biased_grouped_topk`), RMSNorm (fused add+rmsnorm+quant), MLA decode
(`_rocm_aiter_mla_decode_fwd`), FP8/FP4 BMM.

## The torch.compile interaction (why registration matters)
- When `compilation_config.backend == "inductor"` and mode ≠ NONE, vLLM **appends `"none"` to
  `custom_ops`** → CustomOp dispatch is disabled and Inductor generates fused Triton for those ops.
- AITER ops registered via `direct_register_custom_op` are **exempt**: they remain opaque `torch.ops`
  (with fake impls), so Inductor preserves the AITER kernel instead of replacing it with generated Triton.
- ROCm fusion passes (`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`) fuse AITER op chains
  (rms+quant, etc.) inside the compiled graph.

This is the crux: **register an AITER op as a custom op** → it survives `torch.compile`; **don't** → Inductor
decomposes/regenerates it (losing the hand-tuned kernel).

## Engage / knobs
- Master: `VLLM_ROCM_USE_AITER=1` (default 0). Sub-flags: `_LINEAR`, `_MOE`, `_RMSNORM`, `_MLA`, `_MHA`
  (all default 1 once master is on); `_FP4BMM=0` on MI300X (crash); `_FP8BMM`, `_TRITON_GEMM`,
  `_TRITON_ROPE`. Full table: [overview.md](overview.md).
- AITER GEMM tuned configs: drop a CSV; `_load_gemm_tuned_configs` / `_check_kernel_tuned(N,K,dtype,csv)`
  pin per-(N,K) kernels (build-specific — don't ship as portable).
- `AITER_ONLINE_TUNE=1` to retry on `RuntimeError: wrong! device_gemm` (missing shape).
- FP8 scaled matmul: `model_executor/kernels/linear/scaled_mm/rocm.py`.

## Pitfalls
- **Image mismatch**: `VLLM_ROCM_USE_AITER=1` but no aiter in the image → import/runtime failure. Use a
  matched `vllm/vllm-openai-rocm` image; don't ad-hoc pip-install AITER (ABI must match).
- **FP4BMM default-on crash on gfx942** (#34641) → `VLLM_ROCM_USE_AITER_FP4BMM=0`.
- **AITER MLA accuracy** regressions (Kimi-K2 DP2TP4, aiter #1455) — accuracy-gate.
- **Coverage gaps**: AITER tunes CDNA4 first; a missing gfx942 shape falls back to generic Triton (several×
  slower) — watch for it in traces; `AITER_ONLINE_TUNE=1`.
- Env-var sprawl: 13 `VLLM_ROCM_USE_AITER_*` vars; a config-based IR-op-priority system (`linear=aiter_ck`,
  `attention=aiter_asm`, …) is proposed (issue #33163) — expect the surface to change.

## Verify
- rocprofv3: confirm `*ck_*` / AITER asm kernels ran (not a Triton fallback). Inspect the compiled graph
  (vLLM `-O` / `--compilation-config`) to confirm AITER ops stayed opaque.
- Greedy/temp=0 parity + a small eval (gsm8k) when enabling AITER MLA/FP8 (reduction order + fnuz).

## Alternatives / cross-links
[overview.md](overview.md) · [rocm_kernels.md](rocm_kernels.md) (vLLM's own HIP ops) ·
[../pytorch_inductor/overview.md](../pytorch_inductor/overview.md) (custom-op preservation) ·
[../sglang_kernels/overview.md](../sglang_kernels/overview.md) (AITER in SGLang).

## Sources
- vLLM `_aiter_ops` (AITER gating/registration): https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
- vLLM envs (`VLLM_ROCM_USE_AITER*`): https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- AITER Fused-MoE V1 PR (registered ops): https://github.com/vllm-project/vllm/pull/16752
- CustomOp design (forward_hip, inductor "none" + custom_ops): https://docs.vllm.ai/en/stable/design/custom_op/
- AITER env-var → config refactor proposal (#33163): https://github.com/vllm-project/vllm/issues/33163
- AITER (ROCm/aiter): https://github.com/ROCm/aiter
