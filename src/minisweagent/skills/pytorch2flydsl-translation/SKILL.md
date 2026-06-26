---
name: pytorch2flydsl-translation
description: Use when translating PyTorch GPU kernels to FlyDSL. Provides API reference, translation guides, and strategy for mapping PyTorch ops to FlyDSL equivalents.
---

# PyTorch to FlyDSL Translation Skill

This skill provides knowledge and strategy for translating PyTorch GPU kernels to FlyDSL, a domain-specific language for AMD GPU kernel programming.

## Translation Strategy (in order of preference)

- **GEMM / Linear (fixed weight)**: Use `compile_preshuffle_gemm_a8()` from `kernels.preshuffle_gemm`.
  CRITICAL: B-matrix must be preshuffled with `shuffle_weight(B.contiguous(), layout=(16, 16))` from `tests.utils`.
  All tensor args must be `.view(-1)`. Scales: `torch.empty(0, device=dev, dtype=torch.float32)` for fp16.
- **GEMM (dynamic activations, small M)**: Use `hgemm_splitk_()` from `kernels.hgemm_splitk`.
  For activation×activation matmuls (e.g. decomposed Q@K^T, attn@V, paged KV) where B is NOT a fixed weight.
  `C = A @ B^T` with `A:(M,K)`, `B:(N,K)`. No preshuffle. See `flydsl_translation_gemm.md` section Split-K GEMM (hgemm_splitk).
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
- **Decode-mode attention** (`seqlen_q=1` with a paged KV cache: `kv_cache` or
  `k_cache`+`v_cache`, plus `block_table`/`page_table` and `cache_seqlens`) — covers
  MLA, PagedAttention, and any paged-decode kernel. See
  `flydsl_translation_attention.md` § Decode Attention.
  If a prebuilt fused kernel matches the source shape, wrap its launcher instead of
  generating a kernel. Otherwise decompose with `hgemm_splitk_` + `build_softmax_module`
  and apply the decode optimizations (these are what actually move FlyDSL kernel perf):
    - Stack all rows into `M_tot = B*H*Sq` and run a single stacked softmax — never
      loop per-(batch,head) calling GEMM/softmax one at a time.
    - Batch the KV gather / paged-cache reconstruction and any GQA expansion into one
      indexed gather, not a Python loop.
    - Pre-scale Q in fp16 before the QK GEMM; keep softmax dtype matched to the GEMM output.
    - Reuse persistent score/output buffers across calls; read `cache_seqlens` once via
      `.tolist()` — never `.item()` in hot loops.
    - Use `get_default_kwargs` for GEMM tiling; for long context use a page-tiled online softmax.
  MLA specifics: asymmetric `headdim_qk` (e.g. 576) vs `headdim_v` (e.g. 512); slice V
  from the shared latent cache. When a decode kernel is detected, GEAK loads both the
  `attention` KB (§ Decode Attention) and the `gemm` KB
  (`flydsl_translation_gemm.md` § Split-K GEMM).
  Do NOT wrap the kernel in a CUDA graph to obtain the speedup: graph capture only
  removes host-side launch overhead and is not a FlyDSL kernel improvement, so it makes
  the comparison against the (non-captured) PyTorch baseline misleading. Measure the
  FlyDSL kernel WITHOUT CUDA graphs so the reported uplift reflects the translation itself.

CRITICAL: Do NOT use torch.matmul, F.linear, nn.Linear, or F.scaled_dot_product_attention.
These ALL have FlyDSL pre-built replacements. PyTorch fallback is ONLY for Conv2d, MaxPool2d, BatchNorm2d.

## Reference Documentation

The `docs/` subdirectory contains detailed API references and translation guides:

- `flydsl_translation_api_reference.md` — FlyDSL compiler API, expression types, kernel patterns
- `flydsl_translation_guide.md` — PyTorch op mapping, structural patterns, common pitfalls
- `flydsl_translation_gemm.md` — GEMM/Linear translation: preshuffle_gemm (fixed weight) + split-K hgemm (dynamic activations / small-M decode)
- `flydsl_translation_attention.md` — Attention/SDPA translation with flash_attn, plus decode-mode paged attention (MLA, PagedAttention; wrap a matching fused kernel, else decomposed path)
- `flydsl_translation_reductions.md` — Reduction ops (sum, mean, softmax, layernorm)
