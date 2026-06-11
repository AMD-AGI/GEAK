---
title: aiter integration — vLLM/SGLang switches, custom-op registration, env table
kind: backend
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/jit/utils/torch_guard.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://docs.vllm.ai/en/stable/design/custom_op/
---

# aiter integration

## TL;DR
aiter is enabled in vLLM by the single master switch **`VLLM_ROCM_USE_AITER=1`** (required even when you
force `--attention-backend`), and is **on by default in SGLang on ROCm**. Internally aiter registers its
dispatchers as **PyTorch custom ops with fake/meta impls** so they survive `torch.compile`/Inductor
tracing. The deploy seam for tuning wins is env vars only (`AITER_CONFIG_*`) — no site-packages edits.

## How frameworks turn aiter on

### vLLM (V1)
```bash
export VLLM_ROCM_USE_AITER=1          # master switch: GEMM, RMSNorm, MoE, attention
vllm serve <model> --tensor-parallel-size 8 --trust-remote-code
```
- `VLLM_ROCM_USE_AITER` is the **parent** switch; sub-flags (`VLLM_ROCM_USE_AITER_LINEAR`,
  `VLLM_USE_AITER_MOE`, `VLLM_USE_AITER_MLA`, `VLLM_USE_AITER_BLOCK_GEMM`, …) gate individual op families
  and are on by default when the parent is on.
- `--attention-backend` overrides only the attention kernel; the parent switch is still required for
  GEMM/RMSNorm/MoE.
- `VLLM_ROCM_USE_AITER=0` disables aiter entirely (falls back to Triton) — useful for debugging.
- `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1` lets the GEMM wrapper tune hipBLASLt algos at runtime on first
  sight of a shape.

### SGLang (default on ROCm Docker)
```bash
SGLANG_ROCM_AITER_BLOCK_MOE=1 CK_BLOCK_GEMM=1 \
python3 -m sglang.launch_server --model <model> --tp 8 --trust-remote-code
```
SGLang's FP8 linear routes through aiter (→ `torch._scaled_mm` / hipBLASLt-backed); gfx942 checkpoints are
re-quantized to E4M3**FNUZ**.

## Custom-op registration (why aiter survives torch.compile)
aiter dispatchers (e.g. `gemm_a16w16`, `fused_moe_`) are wrapped by
`@torch_compile_guard(gen_fake=...)` (`aiter/jit/utils/torch_guard.py`). On compile it:
1. infers a schema (`torch.library.infer_schema`, with a fallback to `torch._custom_op.impl.infer_schema`
   on older torch);
2. registers the op into a `torch.library.Library`;
3. registers a **fake/meta impl** via `aiter_lib._register_fake` from the supplied `gen_fake` (e.g.
   `gen_gemm_a16w16_fake_tensor` allocates `[*A.shape[:-1], B.shape[0]]`).

This is the AMD-side equivalent of vLLM's `direct_register_custom_op` (which wraps
`torch.library.custom_op` + `register_fake`): the kernel becomes an opaque, traceable op so Inductor
doesn't try to decompose or re-trace it. vLLM's own `CustomOp` dispatch additionally routes to
`forward_hip()` on ROCm (falling back to `forward_cuda()`), and disables custom ops in favor of Inductor
Triton when `backend == "inductor"` unless overridden.

## Env table (the ones that matter)

| Var | Effect |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | vLLM master switch (GEMM/RMSNorm/MoE/attn) |
| `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1` | runtime hipBLASLt online tuning in the GEMM wrapper |
| `SGLANG_ROCM_AITER_BLOCK_MOE=1`, `CK_BLOCK_GEMM=1` | SGLang block-MoE / CK block-GEMM paths |
| `AITER_TUNE_GEMM=1` | capture live GEMM shapes → `bf16_untuned_gemm.csv` |
| `AITER_CONFIG_GEMM_BF16=<csv[:csv]>` | deploy tuned bf16 GEMM DB (mergeable) |
| `AITER_CONFIG_FMOE=<csv>` | deploy tuned fused-MoE DB |
| `AITER_LOG_TUNED_CONFIG=1` | log every DB hit (`is tuned on cu_num`) |
| `AITER_LOG_MORE=1` | log JIT build + backend dispatch decisions |
| `AITER_USE_SYSTEM_TRITON=1` | use system Triton instead of the pinned one |
| `AITER_REBUILD=1` / `ENABLE_CK=1` | force JIT rebuild / enable CK backend (default on) |

## Pitfalls
- ⚠ Forcing `--attention-backend triton` does NOT disable aiter GEMM/MoE — `VLLM_ROCM_USE_AITER` still
  governs them (that's why the +2.23% GEMM tune stacks on a Triton attention backend).
- ⚠ Tuning hipBLASLt's override file or PyTorch TunableOp is inert under aiter — deploy via
  `AITER_CONFIG_*` instead.
- First call to each op pays a JIT compile; warm the server before benchmarking.

## How to verify aiter is live
`AITER_LOG_MORE=1` shows dispatch; `AITER_LOG_TUNED_CONFIG=1` + `grep -c 'is tuned on cu_num' server.log`
confirms DB hits on the live path.

## Cross-links
[overview.md](overview.md) · [tuned_gemm.md](tuned_gemm.md) · [configs_db.md](configs_db.md) ·
[fmoe.md](fmoe.md) · [attn_mla.md](attn_mla.md).

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/jit/utils/torch_guard.py`
  (`torch_compile_guard`, schema infer, `_register_fake`), `aiter/jit/core.py` (env vars).
- vLLM AITER switches: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM CustomOp dispatch (forward_hip / inductor): https://docs.vllm.ai/en/stable/design/custom_op/
- aiter default-backend / SGLang integration: https://github.com/ROCm/aiter (README) ·
  https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
