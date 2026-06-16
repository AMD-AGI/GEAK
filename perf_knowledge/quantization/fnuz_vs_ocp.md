---
title: FNUZ vs OCP FP8 — the CDNA3/CDNA4 dialect split and the 2× trap
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
  - https://github.com/ROCm/AMDMIGraphX/issues/2717
---

# FNUZ vs OCP FP8 — the dialect split that bites

> **TL;DR.** AMD has **two FP8 dialects** with the *same byte layout but a different exponent bias*.
> **CDNA3 / gfx942 (MI300/MI325) = FNUZ** (bias 8 for E4M3); **CDNA4 / gfx950 (MI350/MI355) = OCP**
> (bias 7, the H100/Blackwell-standard E4M3FN). Bit-copy a checkpoint across the split and every value
> is **off by exactly a factor of 2 — silently**. Match the dialect to the arch; re-cast, never bit-copy.
> Element-level numerics: [[operators/quant_dequant_fp8]]. HW rates: [[hardware_support_matrix.md]].

## The two dialects
| | CDNA3 (gfx942, MI300/MI325) | CDNA4 (gfx950, MI350/MI355) |
|---|---|---|
| FP8 dialect | **FNUZ** ("finite, NaN, unsigned zero") | **OCP** (E4M3FN, E5M2) — also accepts FNUZ |
| E4M3 exponent bias | **8** | **7** |
| E4M3 max | **±240** | **±448** |
| E5M2 bias / max | 16 / ±57344 | 15 / ±57344 |
| ±inf | no | yes (E5M2 only; E4M3 OCP is NaN-only) |
| −0 | no (single +0) | yes |
| NaN encoding | sign=1, exp=mant=0 (`0x80`) | standard OCP NaN |
| FP4/FP6 | no | yes |

FNUZ trades the inf/−0 encodings for an extra exponent value, widening range — but the dialect never
became a standard; later AMD parts (MI325 onward in practice, fully on MI350) return to OCP FP8.

## Why it is a "2× trap" (the mechanism)
E4M3FNUZ and E4M3FN (OCP) share an identical 8-bit layout `S EEEE MMM`. The **only** difference for
finite values is the exponent bias (8 vs 7). Decoding the same bits with the wrong bias shifts the
exponent by one ⇒ the value is scaled by exactly **2× (or ½×)**. Nothing throws; the result is just
quietly wrong. Most of vLLM's FP8 code is aware of E4M3-vs-E5M2 but historically *not* of FNUZ-vs-OCP —
which is exactly how the trap fires when an OCP-quantized checkpoint lands on an MI300X
(per the DeepSeek-V4 MI300X bring-up writeup).

## Conversion across the split (OCP ↔ FNUZ)
Per the MIGraphX conversion issue, the bit-level rule for **OCP → FNUZ**:
1. **−0 → +0** (FNUZ has no −0).
2. **any NaN/inf → the single FNUZ NaN** (FNUZ has no inf).
3. **all other finite values: keep the byte** — because the bias differs by 1, keeping the encoding
   *automatically halves the value*, so to preserve the represented number you instead adjust the
   **scale**, not the bits.

When quantization scales are attached (QDQ graphs), the documented adjustment is: **multiply the scale
by 2 on QuantizeLinear and divide by 2 on DeQuantizeLinear** to absorb the bias shift. Range
consequence: OCP's ±448 max maps to **±224** under FNUZ (it never reaches the FNUZ ±240) — generally
acceptable per AMD's framework experiments, and it dovetails with the **224.0 ROCm dynamic cap**
([[scaling_strategies.md]], [[operators/quant_dequant_fp8]]).

## Doing it right in code (arch-gated dialect selection)
- HIP selects the dialect by arch string: code checks for `gfx94*` and picks `__HIP_E4M3_FNUZ`,
  otherwise defaults to OCP `__HIP_E4M3`. The same source compiled for MI300 yields FNUZ; for any other
  GPU, OCP.
- Use the arch-matching helper families, do **not** reinterpret bytes:
  - CDNA3: `__hip_fp8_*` (`hip_fp8.h`) — FNUZ.
  - gfx950: `__amd_fp8_*` / OCP path — and CDNA4's HIP FP8 extensions map ~1:1 to hardware instructions.
- vLLM on ROCm pins `FP8_TYPE = c10::Float8_e4m3fnuz` for the CDNA3 path
  (`csrc/quantization/fp8/common.cu`) — see [[operators/quant_dequant_fp8]].

## What this means for checkpoints & portability
- An **OCP FP8 checkpoint** (the de-facto standard, shared with NVIDIA H100) reads **incorrectly on
  MI300X** without dialect-aware conversion. AMD's Hugging Face **Quark OCP-FP8 collection** is the safe
  source; loaders must apply the correct dialect for the target arch
  ([[calibration_and_quark.md]], [[deployment_recipes.md]]).
- The trap is **specific to CDNA3**. On MI350/MI355 (OCP) the off-by-2× problem does not exist — OCP
  checkpoints load natively.
- gfx942 vs gfx950 is the line: treat any cross-gen FP8 move as a **re-cast**, gated on `err_ratio<0.05`
  and a task-accuracy check ([[accuracy_evaluation.md]]).

## Pitfalls
- **Bit-copying FP8 weights gfx942 ↔ gfx950** → silent 2× error.
- **Assuming "FP8" means OCP everywhere** — on MI300 it is FNUZ in hardware.
- **Forgetting the scale ×2 / ÷2 adjustment** when rewriting QDQ graphs across dialects.
- **Expecting ±448 dynamic range on MI300** — FNUZ tops out at ±240 (±224 from OCP via conversion).

## Verify
- Round-trip a known tensor through the *target-arch* cast and compare to fp32 reference; a clean
  factor-of-2 error in the histogram is the dialect-mismatch signature.
- Confirm the saturation point: ±240 (FNUZ) vs ±448 (OCP). MMA-Sim is bit-accurate
  ([[operators/quant_dequant_fp8]]).

## Sources
- HIP FP8 numbers, FNUZ definition, arch-gated `__HIP_E4M3_FNUZ` vs OCP: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html
- CDNA3 vs CDNA4 dialect, block-scaled MFMA type codes: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- The 2× trap on MI300X (vLLM unaware of FNUZ vs OCP): https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- OCP→FNUZ conversion rules, scale ×2/÷2, 448→224: https://github.com/ROCm/AMDMIGraphX/issues/2717
- Element numerics & 224 cap: [[operators/quant_dequant_fp8]] (`vllm@HEAD:csrc/quantization/fp8/common.cu`).
