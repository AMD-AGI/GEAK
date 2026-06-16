---
title: batched_gemm on aiter — SOTA card
kind: sota_card
operator: batched_gemm
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

# batched_gemm × aiter

## TL;DR
> aiter is the **live dispatch path** for uniform batched GEMM on sglang/vllm — same `tuned_gemm` engine
> as dense, racing hipBLASLt strided-batched / asm / triton per shape from the CSV DB. To improve it,
> tune aiter's per-shape DB. This is the only batched-GEMM lever that engages the serving path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `tuned_gemm` DB (per-shape) | `ROCm/aiter@HEAD:aiter/tuned_gemm.py` + `gradlib/.../gemm_tuner.py` | gfx942/950; bf16, fp8 scaled | inherits the dense-GEMM tuning mechanism (**+2.23% e2e** demonstrated on the dense path, Qwen3.5-27B/sglang, MI300X, 2026-06-08); batched matmuls in attention are usually subsumed by FMHA | uniform batched GEMM on the live path |

Most batched matmuls in LLMs are inside attention and never reach this path (FMHA fuses them) — see
[../overview.md](../overview.md).

## Config space / knobs
- Capture: `AITER_TUNE_GEMM=1` on a warm server (bias must match live) → appends shapes to the untuned CSV.
- Tune: `gradlib/gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` (gate `err_ratio<0.05`).
- Deploy: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` `AITER_LOG_TUNED_CONFIG=1`.
- Lookup key (9-tuple): `(cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)`.

## Numerics / parity
Same-dtype solution swap → parity-safe per batch; gradlib gates `err_ratio<0.05`. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
Live call site `aiter.tuned_gemm:gemm_a16w16` / `tgemm.mm`. No package edit needed — env-overlay CSV.

## Pitfalls & anti-patterns
- bias mismatch (tuned bias=true vs live bias=false) → 100% lookup miss, 0 engagement.
- Expecting it to speed up attention's internal matmuls — those are fused in FMHA, not here.
- TunableOp / `HIPBLASLT_TUNING_FILE` hook a path aiter bypasses → 0 engagement.

## How to verify
`grep -c 'is tuned on cu_num' <server.log>` > 0, then same-session A/B, accept iff delta>0.5% AND
non-overlap AND parity holds.

## Alternatives / cross-links
[hipblaslt.md](hipblaslt.md) (executed strided-batched) · [ck.md](ck.md) · [triton.md](triton.md) ·
[asm.md](asm.md) · [hip.md](hip.md) · [../overview.md](../overview.md).
Dense equivalent: [[operators/dense_gemm/backends/aiter.md]].

## Sources
- On-box source: `/sgl-workspace/aiter/aiter/tuned_gemm.py`, `gradlib/gradlib/gemm_tuner.py` (= ROCm/aiter).
- +2.23% dense-path validation (shared engine): perf_knowledge e2e run 2026-06-08.
- aiter engine: https://github.com/ROCm/aiter.
