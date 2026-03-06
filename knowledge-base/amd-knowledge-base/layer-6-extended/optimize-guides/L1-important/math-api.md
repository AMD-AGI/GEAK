Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "math", "reference", "api"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# HIP Math API Documentation

## Overview

HIP-Clang offers device-callable mathematical operations that support most functions available in NVIDIA CUDA. This documentation covers maximum error bounds for supported functions and lists unsupported operations.

Error measurements use Units in the Last Place (ULPs), representing the absolute difference between a HIP math function result and its corresponding C++ standard library equivalent.

## Standard Mathematical Functions

These functions prioritize numerical accuracy and correctness, suitable for applications requiring high precision. Unless specified otherwise, all functions below are available on the device side.

### Arithmetic Operations

Both single and double precision versions support:
- Absolute value (`abs`, `fabs`)
- Difference operations (`fdim`)
- Fused multiply-add (`fma`)
- Min/max functions (`fmin`, `fmax`)
- Modulo and remainder operations (`fmod`, `remainder`, `remquo`)
- Division (`fdivide`)

Most arithmetic functions maintain 0 ULP error difference from standard C++ implementations.

### Classification Functions

Available for both precisions:
- "Determine whether x is finite" (`isfinite`)
- "Determine whether x is infinite" (`isinf`)
- "Determine whether x is a NAN" (`isnan`)
- Sign bit checking (`signbit`)
- NaN generation (`nan`, `nanf`)

All classification operations achieve 0 ULP difference.

### Error and Gamma Functions

Error functions include `erf`, `erfc`, `erfcx` with ULP differences ranging from 2-5. Gamma functions (`lgamma`, `tgamma`) show 4-6 ULP differences.

### Exponential and Logarithmic Functions

Functions like `exp`, `exp2`, `exp10`, and logarithmic variants (`log`, `log2`, `log10`) maintain 1-2 ULP error bounds across test ranges.

### Floating Point Manipulation

Operations including `copysign`, `frexp`, `ldexp`, `nextafter`, and scaling functions achieve 0 ULP difference.

### Hypotenuse and Norm Functions

"Returns the square root of the sum of squares" functions (`hypot`, `norm3d`, `norm4d`) maintain 1-2 ULP differences.

### Power and Root Functions

- `cbrt`: 1-2 ULP difference
- `pow`: 1 ULP difference
- `sqrt`, `rsqrt`: 1 ULP difference
- `rcbrt`: 1 ULP difference

### Rounding Functions

Ceiling, floor, and rounding operations (`ceil`, `floor`, `round`, `trunc`) all maintain 0 ULP difference.

### Trigonometric and Hyperbolic Functions

Standard trig functions (`sin`, `cos`, `tan`) show 1-2 ULP differences. Hyperbolic variants (`sinh`, `cosh`, `tanh`) maintain similar bounds. Pi-scaled versions (`sinpi`, `cospi`) available for both precisions.

### Functions Without C++ STD Equivalents

These lack direct standard library comparisons:
- Bessel functions (`j0`, `j1`, `jn`, `y0`, `y1`, `yn`)
- Inverse error functions (`erfinv`, `erfcinv`)
- Normal distribution functions (`normcdf`, `normcdfinv`)

### Unsupported Functions

Modified cylindrical Bessel functions (`cyl_bessel_i0`, `cyl_bessel_i1`) are not supported.

## Intrinsic Mathematical Functions

Intrinsics prioritize performance over precision, ideal for efficiency-critical applications. Note: intrinsics operate on device only.

### Floating-Point Intrinsics

Single precision fast approximations include `__sinf`, `__cosf`, `__logf`, `__expf` with varying ULP bounds (1-18). Double precision variants (`__dadd_rn`, `__dmul_rn`, `__dsqrt_rn`) maintain exact or near-exact results.

**Note:** "Only the nearest-even rounding mode is supported by default on AMD GPUs."

### Integer Intrinsics

Operations include:
- Bit manipulation (`__brev`, `__clz`, `__popc`)
- Shift operations (`__funnelshift_l`, `__funnelshift_r`)
- Arithmetic (`__hadd`, `__mul24`, `__mulhi`)
- Special functions (`__ffs`, `__sad`)

These perform exact integer operations without ULP measurements.
