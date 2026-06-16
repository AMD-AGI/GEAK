# Decision trees — "what should I even try?"

Fast routing from (operator, gen, dtype, shape regime) to a backend/approach. These are *priors*; the
operator's SOTA cards + a same-session A/B are the judge.

## Dense GEMM (prefill, large M)
```
Is the GEMM on a serving stack that routes through aiter (sglang/vllm)?
├─ yes → tune aiter's per-shape DB (AITER_TUNE_GEMM=1 capture → gradlib → AITER_CONFIG_GEMM_BF16);
│        verify 'is tuned on cu_num' >0. Also author a Triton/FlyDSL GEMM and e2e-gate the best.
│        ⚠ TunableOp / HIPBLASLT_TUNING_FILE do NOT engage the aiter live path.
└─ no (raw torch/F.linear) → hipBLASLt offline tune (HIPBLASLT_TUNING_OVERRIDE_FILE) or TunableOp.
Need to beat the library matmul itself? hand-asm or FlyDSL; pure Triton usually loses to tuned hipBLASLt.
```

## Decode / skinny GEMM (M = batch ≤ 256)
```
aiter skinny/wvSplitK kernels first → else split-K Triton/FlyDSL (fill 304 CUs) → fuse into epilogue.
```

## Prefill attention (FMHA)
```
CK-Tile FMHA (default, fastest general) → Triton FA (editable, exposes a kernel surface) →
TileLang (≈1.5× Triton, CDNA3) → asm (peak, hard). VLLM_USE_TRITON_FLASH_ATTN=0 selects CK.
FlashInfer is NOT available on AMD.
```

## Editable custom kernels (gated-delta/mamba, norms, rope, act)
```
Triton first (fastest to iterate) → optimize via the recursive kernel layer → ensure the win
mechanism survives varlen serving (avoid per-call graph caches keyed on buffer pointers).
Small Amdahl mass (<0.5% each) → STACK the cluster and gate the combined stack.
```

## Quantized GEMM
```
CDNA3 (gfx942): fp8 FNUZ via aiter/CK scaled-GEMM.
CDNA4 (gfx950): MXFP8/MXFP6/MXFP4 block-scaled MFMA (FP6 runs at FP4 rate) — see quantization/.
Always accuracy-gate (task probe), never byte-parity, for quant.
```

## Sources
- aiter live-path / TunableOp dead-end, +2.23% GEMM tune: perf_knowledge e2e validation (2026-06-08); ROCm/aiter.
- CK default FA / triton flag: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- TileLang FA ≈1.5× Triton: https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- CDNA4 MXFP / FP6=FP4 rate: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
