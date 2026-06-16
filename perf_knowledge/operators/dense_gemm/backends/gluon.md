---
title: dense_gemm on Gluon — SOTA card
kind: sota_card
operator: dense_gemm
backend: gluon
gens: [gfx950]
dtypes: [fp16, bf16]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---

# dense_gemm × Gluon

## TL;DR (one-line decision)
> On **CDNA4 (gfx950)** Gluon is the SOTA *authoring path* for a near-peak FP16/BF16 dense GEMM —
> **1489 TFLOPS @ 98.75% MFMA eff** (4096×4096×8192), ~3× over a 520-TFLOPS naive baseline — by
> hand-placing layout + pipeline + MFMA schedule. Use it when Triton's autoscheduler leaves >20% on the
> table and you can afford manual tuning; for a no-tune default still call hipBLASLt/aiter.

## SOTA implementation(s)
Best known FP16 dense GEMM in a Python DSL on gfx950 is AMD's Gluon `a16w16` v9 kernel — the end of the
v0→v9 tutorial that introduces explicit LDS layouts, manual software pipelining, register budgeting, and
MFMA interleave scheduling (cite the Gluon GEMM tutorial).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| Gluon `a16w16` v9 | `ROCm/gfx950-gluon-tutorials:kernels/gemm/a16w16`; Gluon GEMM tutorial | gfx950; FP16; 256×256×64 tile; 4096×4096×8192 | **1489 TFLOPS @ 98.75% MFMA eff** @ MI355X gfx950, ROCm 7.0, AMD-measured, 2026 | near-peak FP16 GEMM when willing to hand-schedule |
| Gluon `a16w16` v0 (naive) | same repo | gfx950; FP16 | 520 TFLOPS @ 25% MFMA eff (baseline) | n/a — reference baseline (~3× headroom) |

Cross-gen note: gfx942 is supported (same Triton/Gluon backend, ping-pong/interleave apply), but the
headline 1489 number is gfx950. For reference, HipKittens BF16 256×256 hits 1610 TFLOPS @ 8192 on the same
HW (see [[languages/hipkittens]]).

## Config space / knobs
- **Tile**: `256×256×64` (M×N×K) for FP16 a16w16.
- **Wave schedule**: **ping-pong** (8-wave) or **interleave** (4-wave) — authored by hand, not a flag; this
  is the lever that takes 25%→98.75% MFMA eff. See [[optimization/mfma_scheduling]].
- **num_stages / pipeline**: explicit **2-/3-stage** software pipeline authored in the kernel (not Triton's
  `num_stages` autoscheduler). GR→LW→LR→MFMA.
- **LDS layout**: swizzled/padded shared layouts + `ds_read_tr` (transpose read) to kill bank conflicts.
- **Register budget**: keep live values within the **512-VGPR/EU** limit to avoid spill.
- **Scheduling passes**: `TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1` to enable IR interleave +
  post-assembly peephole (the v9 "beyond hot loop" step).

## Numerics / parity
FP16/BF16 in, **FP32 accumulate** → parity with hipBLASLt/aiter up to tiling rounding.

## Integration (rebind seam)
Gluon is an **authoring path**, not a production dispatch backend. A Gluon GEMM is engaged the same way an
authored Triton GEMM is: register it through the aiter `triton` libtype seam or a call-site rebind in the
model's `LinearMethod`, then e2e-gate. aiter/triton **relate** as: Triton = portable autoscheduled
authoring; Gluon = the low-level Triton dialect for the last 20–75%; aiter = the production dispatcher you
wire the authored kernel into. See [[operators/dense_gemm/backends/triton]] and
[[operators/dense_gemm/backends/aiter]].

## Pitfalls & anti-patterns
- **gfx950 headline.** The 1489 number is CDNA4; gfx942 runs but is not the measured headline.
- **Manual scheduling cost.** You hand-author layout, pipeline, register budget, and wave schedule — far
  more effort than Triton for the last ~20%. Don't reach for Gluon until Triton/hipBLASLt is the bottleneck.
- **Large-K, compute-bound shape.** 1489 is at 4096×4096×8192; skinny/decode GEMM is a different regime.
- **Don't treat it as a no-tune default** — hipBLASLt/aiter remain the zero-effort bar.

## How to verify (bench + oracle)
```bash
cd kernels/gemm/a16w16
python bench.py --version 0 --K 8192 --dtype fp16          # baseline 520 TFLOPS @ 25%
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 8192 --dtype fp16       # 1489 TFLOPS @ 98.75%
# parity: compare vs hipBLASLt/torch.matmul reference at fp32 accumulate; e2e-gate via aiter seam
```

## Alternatives / cross-links
[[operators/dense_gemm/backends/triton]] (portable authoring) ·
[[operators/dense_gemm/backends/hipblaslt]] (no-tune default) ·
[[operators/dense_gemm/backends/aiter]] (dispatch + e2e gate) ·
[[operators/scaled_quant_gemm/backends/gluon]] (BF8) ·
[[operators/quant_fp4_mxfp/backends/gluon]] (MXFP4) ·
[[languages/gluon]] · [[languages/triton_amd]] · [[optimization/mfma_scheduling]]

## Sources
- From Naive to Near-Peak: GEMM Kernels with Gluon (FP16 1489 @ 98.75%, naive 520 @ 25%, MI350/MI355, ROCm 7.0): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Gluon GEMM tutorial code (a16w16 v0→v9, 256×256×64): https://github.com/ROCm/gfx950-gluon-tutorials
- FP8 GEMM on CDNA4 — ping-pong scheduling origin: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- 4-wave interleave FP8 GEMM: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
