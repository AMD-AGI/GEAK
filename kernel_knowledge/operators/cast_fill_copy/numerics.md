---
title: cast_fill_copy — numerics & parity
kind: operator_overview
operator: cast_fill_copy
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/sgl-project/sglang/pull/2601
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# cast_fill_copy — numerics & parity

Copy and fill are **bit-exact** (no math) — their only "numerics" risk is dtype/stride bugs. **Cast** is
where the real hazards live: fp8/int8 saturation and the **FNUZ-vs-OCP dialect**.

## 1. fp8 dialect — FNUZ (gfx942) vs OCP (gfx950)
The single biggest cast bug on AMD. **CDNA3 / gfx942 uses FNUZ** fp8 (`fp8_e4m3_fnuz` bias 8,
`fp8_e5m2_fnuz`); **CDNA4 / gfx950 uses OCP** (`fp8_e4m3`, `fp8_e5m2`). Casting with the wrong dialect (or
loading an OCP-quantized checkpoint and treating it as FNUZ) is off by roughly **2× in scale** — a silent
correctness bug, not a rounding error. SGLang/vLLM normalize OCP checkpoints with
`normalize_e4m3fn_to_e4m3fnuz` before feeding gfx942 (PR #2601). Always confirm the dialect for the target
arch.

## 2. fp8 saturation (no Inf in e4m3)
e4m3 has **no Inf**: on overflow the cast **saturates to max-normal** (±448 for OCP e4m3, ±240 for FNUZ
e4m3fnuz). A value above range becomes the max, not Inf/NaN. This is usually paired with a **scale** (the
quant step) — the cast itself must saturate, and the scale must keep values in range. e5m2 has more range,
less precision. int8 saturates to `[-128, 127]`.

## 3. Rounding mode
- float→narrower-float (fp32→bf16, →fp8): **round-to-nearest-even** is the torch default; match it
  (`__float2bfloat16` does RNE; some fast paths truncate). bf16 truncation (drop low mantissa) is faster but
  biased — re-check parity if used.
- float→int8: define round (RNE vs trunc vs floor) and clamp; match the reference quantizer.
- `-ffast-math`/denormal-flush changes the result near zero — re-check.

## 4. fp32 → bf16/fp16 (the common safe cast)
Exact for representable values, RNE rounding otherwise. Round-trip bf16→fp32→bf16 is exact. fp16 has a
smaller exponent range than bf16 → fp32→fp16 can overflow to Inf where bf16 wouldn't (matters for large
logits/accumulators).

## 5. Copy/fill bit-exactness
Contiguous copy and `hipMemset`/byte-fill are bitwise identical — parity is trivial. The only bugs are
**stride/offset** (wrong layout) and **partial fill** (tail not covered) — test the exact shape and the
`.contiguous()` result against `torch.clone`/`.contiguous()` byte-for-byte.

## Parity gate
- copy/fill: **bitwise** vs `torch.clone`/`torch.full`/`.contiguous()`.
- float→float cast: atol vs torch (RNE).
- →fp8/int8 cast: confirm the **dialect** for the arch, the saturation, and gate on a **task eval** (byte
  parity won't hold; the quant accuracy gate is what matters — see
  [`../quant_dequant_fp8/numerics.md`](../quant_dequant_fp8/numerics.md)).

## Sources
- FNUZ (gfx942) vs OCP (gfx950) fp8, saturation/max-normal: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- `normalize_e4m3fn_to_e4m3fnuz` checkpoint normalization: https://github.com/sgl-project/sglang/pull/2601
- `__float2bfloat16` RNE, denormal flags: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
