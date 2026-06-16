---
title: rocBLAS / TunableOp — when it wins, when it's a dead end
kind: backend
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
---

# rocBLAS / TunableOp — when it wins

## TL;DR
**Use TunableOp when your GEMM goes through PyTorch dispatch** (`torch.matmul`/`nn.Linear` on a plain
HF/PyTorch model, TGI) — it races rocBLAS vs hipBLASLt and ships the per-shape winner for free. **Do not
rely on it on sglang/vllm + aiter** — the live GEMM bypasses PyTorch dispatch, so TunableOp (and
`HIPBLASLT_TUNING_FILE`) gets **0 engagement**; the only lever that engages there is **aiter's per-shape
DB**. rocBLAS itself tends to win **small/odd-M (decode) and some strided-batched** GEMMs; hipBLASLt wins
**large/square/fp8/epilogue-fused** — which is exactly why TunableOp races both and the result CSV is a
*mix*.

## When rocBLAS beats hipBLASLt (per-shape, not universal)
- rocBLAS sometimes wins on **small / odd M** (decode-time tall-skinny GEMMs) and certain
  **strided-batched** cases.
- hipBLASLt usually wins on **large square / fp8 / epilogue-fused** GEMMs.
- A real TunableOp CSV routinely mixes `Gemm_Rocblas_21` (some shapes) and `Gemm_Hipblaslt_NN_52565`
  (others) — proof both libraries are needed; there is **no universal winner**, which is the whole point of
  racing.
- For skinny LLM-decode shapes, **TensileLite custom tuning** (rocBLAS or hipBLASLt) can beat generic
  pooled kernels by **1.6–2.6×** (generic kernels waste threads on small workloads).

## When TunableOp wins (the path matters more than the shape)
| stack / call path | TunableOp engages? | use it? |
|---|---|---|
| plain PyTorch / HF model (`torch.matmul`, `nn.Linear`) | ✅ yes (PyTorch dispatch) | **yes** — free per-shape win |
| TGI (AMD image) | ✅ yes (bundles TunableOp warmup) | yes (default; ~6–8% latency, ROCm 6.1/PT 2.3) |
| vLLM/SGLang **without** aiter on the GEMM path | ✅ partial (plain matmul/Linear) | yes for those GEMMs; fp8 paths prefer aiter/hipBLASLt own tuning |
| **sglang / vllm + aiter** live dense GEMM | ❌ **no — bypassed** | **no — dead end** |

## ⚠ The critical dead end: aiter bypasses PyTorch dispatch
On sglang/vllm the live dense GEMM is dispatched by **aiter** (`aiter/tuned_gemm.py::gemm_a16w16` → the
fastest of hipBLASLt `Cijk_*` / asm / skinny / triton / flydsl, chosen from aiter's own CSV DB). This call
site is reached **before** PyTorch's BLAS dispatch, so:
- `PYTORCH_TUNABLEOP_ENABLED=1` produces **no new CSV rows** and **0 latency change** on the serving path.
- `HIPBLASLT_TUNING_FILE` / override file is likewise **not consulted** on the aiter live path.
- **Smoking gun:** after a TunableOp run on sglang, the results CSV gains no GEMM rows for the served
  shapes → 0 engagement.
- **Correct lever:** tune **aiter's per-shape DB** (capture live shapes with `AITER_TUNE_GEMM=1` →
  `gradlib gemm_tuner.py` → deploy `AITER_CONFIG_GEMM_BF16=<tuned.csv>`). Measured **+2.23% e2e** with
  246 `is tuned on cu_num` hits @ MI300X gfx942, sglang 0.5.11/aiter, 2026-06-08. See
  `operators/dense_gemm/backends/aiter.md`.

## Decision flow
1. Does the GEMM go through `torch.matmul`/`nn.Linear` (plain PyTorch / TGI)? → **TunableOp** (online or
   offline + ship mode).
2. Is it sglang/vllm with aiter on the GEMM path? → **aiter DB**, not TunableOp. Confirm with
   `grep -c 'is tuned on cu_num' server.log > 0`.
3. fp8 / fused-epilogue GEMM? → hipBLASLt's own tuning / aiter scaled variants (not rocBLAS).
4. Hot skinny shape with no good pooled solution? → **TensileLite** custom kernel (1.6–2.6× on skinny).

## Pitfalls
- Treating TunableOp as a universal GEMM lever — it is **PyTorch-dispatch-only**.
- Tuning the wrong layer on sglang (TunableOp / hipBLASLt override file) and seeing no change — known dead
  end; verify engagement, don't assume.
- Shipping a CSV across a ROCm/lib/arch upgrade — validators reject it (parity guard); re-tune.

## How to verify engagement (don't trust, measure)
- TunableOp path: CSV gains `Gemm_*` rows for your shapes + A/B vs `ENABLED=0` shows a non-noise delta.
- aiter path: `grep -c 'is tuned on cu_num' server.log` > 0, then same-session 2-launch A/B (gate
  delta>0.5% AND non-overlapping AND parity).

## Sources
- rocBLAS-vs-hipBLASLt mix & TensileLite skinny 1.6–2.6×: MI300X workload guide
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ;
  TensileLite tuning blog https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- TunableOp dispatch-layer behavior: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable
- aiter bypasses PyTorch dispatch (live GEMM path) + measured +2.23%: `ROCm/aiter@HEAD:aiter/tuned_gemm.py`,
  `operators/dense_gemm/backends/aiter.md` (perf_knowledge e2e run 2026-06-08).
- API: [api.md](api.md) · tuning mechanics: [tunableop.md](tunableop.md)
