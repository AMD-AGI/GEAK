---
title: gemm_epilogue_fused on aiter — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
  - https://github.com/ROCm/aiter
---

# gemm_epilogue_fused × aiter

## TL;DR
> aiter is the **live path** for the epilogues it supports (bias, output scale, and fused act variants
> like `act_and_mul` / scaled fp8 GEMM) — `tuned_gemm` carries the epilogue flags in its lookup key and
> dispatches the fastest fused kernel per shape. To improve a fused GEMM on serving, tune aiter's DB on
> the **fused** shape. For epilogues aiter doesn't expose (residual-add, custom act), author in CK and
> register it.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `tuned_gemm` w/ bias/scale epilogue + fused act/quant GEMM ops | `ROCm/aiter@HEAD:aiter/tuned_gemm.py` (+ aiter fused GEMM ops) | gfx942/950; bf16, fp8 scaled | shares the dense tuning mechanism (**+2.23% e2e** demonstrated on the dense path, Qwen3.5-27B/sglang, MI300X, 2026-06-08) | live fused GEMM with supported epilogue |

## Config space / knobs
- Lookup key (9-tuple): `(cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)` — **the
  fused flags (bias, scaleAB, otype) must match the live fused call**.
- Capture fused shapes: `AITER_TUNE_GEMM=1` on a warm server (exercise the fused path so flags are real).
- Tune: `gradlib/gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` (gate `err_ratio<0.05`).
- Deploy: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` `AITER_LOG_TUNED_CONFIG=1`.

## Numerics / parity
Bias/scale fusion same-math → parity-safe; fp8 output quant task-gated. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Live call site `aiter.tuned_gemm:gemm_a16w16` / fused GEMM ops; env-overlay CSV, no package edit.

## Pitfalls & anti-patterns
- Tuning the bare GEMM (bias=false, no scale) while the live call is fused (bias=true / scaled) → key
  miss, 0 engagement (the canonical bias-mismatch failure, verified 2026-06-07).
- Expecting residual-add / arbitrary act on this path — not exposed; use [ck.md](ck.md).
- TunableOp / `HIPBLASLT_TUNING_FILE` → bypassed.

## How to verify
`grep -c 'is tuned on cu_num' <server.log>` > 0 with the fused path exercised, then same-session A/B,
accept iff delta>0.5% AND non-overlap AND parity/eval holds.

## Alternatives / cross-links
[ck.md](ck.md) (richer epilogues) · [hipblaslt.md](hipblaslt.md) · [triton.md](triton.md) ·
[hip.md](hip.md) · [../overview.md](../overview.md) · dense: [[operators/dense_gemm/backends/aiter.md]] ·
quant: [[operators/scaled_quant_gemm/overview.md]].

## Sources
- On-box source: `/sgl-workspace/aiter/aiter/tuned_gemm.py`, `gradlib/gradlib/gemm_tuner.py` (= ROCm/aiter).
- +2.23% dense-path validation (shared engine): perf_knowledge e2e run 2026-06-08.
- aiter engine: https://github.com/ROCm/aiter.
