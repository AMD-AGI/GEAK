---
name: pytorch2flydsl-translation
description: Use when translating PyTorch GPU kernels to FlyDSL. Provides API reference, translation guides, and strategy for mapping PyTorch ops to FlyDSL equivalents.
---

# PyTorch to FlyDSL Translation Skill

This skill provides knowledge and strategy for translating PyTorch GPU kernels to FlyDSL, a domain-specific language for AMD GPU kernel programming.

## Translation Strategy (in order of preference)

- **GEMM / Linear**: Use `compile_preshuffle_gemm_a8()` from `kernels.preshuffle_gemm`.
  CRITICAL: B-matrix must be preshuffled with `shuffle_weight(B.contiguous(), layout=(16, 16))` from `tests.utils`.
  All tensor args must be `.view(-1)`. Scales: `torch.empty(0, device=dev, dtype=torch.float32)` for fp16.
- **Attention / SDPA**: ALWAYS use `build_flash_attn_func_module()` from `kernels.flash_attn_func`
  when head_dim>=64, head_dim%32==0, seq_len%128==0. NEVER decompose attention into separate
  GEMM+softmax+GEMM calls when flash attention fits — decomposed is 5-10x slower.
  NEVER use Python for-loops over batch*heads to call GEMM one at a time.
  Builder: `build_flash_attn_func_module(num_heads=H, head_dim=D, causal=True, dtype_str="f16")`.
  Launcher: `fn(Q.view(-1), K.view(-1), V.view(-1), O.view(-1), batch_size, seq_len, stream=stream)`.
  Note: num_heads is baked in at build time, NOT passed at launch time.
- **Softmax**: `build_softmax_module(M, N, dtype_str)` — call as `fn(input, output, M, stream=stream)`
- **LayerNorm**: `build_layernorm_module(M, N, dtype_str)` — call as `fn(input, gamma, beta, output, M, stream=stream)`
- **RMSNorm**: `build_rmsnorm_module(M, N, dtype_str)` — call as `fn(input, gamma, output, M, stream=stream)`
- **Element-wise ops** (relu, sigmoid, tanh, clamp, etc.): Write custom @flyc.kernel with layout algebra
- **Reductions** (sum, mean): Manual block reduction with wave shuffle
- **Conv/Pool/BatchNorm**: Use `torch.nn.functional` (ONLY ops with no FlyDSL equivalent)
- **Complex models**: Use FlyDSL for ALL ops except conv/pool/batchnorm

CRITICAL: Do NOT use torch.matmul, F.linear, nn.Linear, or F.scaled_dot_product_attention.
These ALL have FlyDSL pre-built replacements. PyTorch fallback is ONLY for Conv2d, MaxPool2d, BatchNorm2d.

## Reference Documentation

The `docs/` subdirectory contains detailed API references and translation guides:

- `flydsl_translation_api_reference.md` — FlyDSL compiler API, expression types, kernel patterns
- `flydsl_translation_guide.md` — PyTorch op mapping, structural patterns, common pitfalls
- `flydsl_translation_gemm.md` — GEMM/Linear translation with preshuffle_gemm
- `flydsl_translation_attention.md` — Attention/SDPA translation with flash_attn
- `flydsl_translation_reductions.md` — Reduction ops (sum, mean, softmax, layernorm)
