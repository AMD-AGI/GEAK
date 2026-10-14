// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// MSVC 14.51 (Visual Studio 17.13 / 2026 18.x) <cmath> added overloads of
// `isless` / `islessequal` / `islessgreater` / `isgreater` / `isgreaterequal` /
// `isunordered` / `isfinite` / `isinf` / `isnan` / `isnormal` that call
// `__builtin_*` directly. The MS STL guards them on `#ifdef __clang__` so they
// only fire under clang -- and in HIP compile mode clang treats their plain
// `inline` / `constexpr` declarations as implicitly `__host__ __device__`,
// colliding with the plain `__device__` overloads in clang-hip's bundled
// `__clang_hip_cmath.h` and `__clang_cuda_math_forward_declares.h`:
//
//   error: __device__ function 'isless' cannot overload
//          __host__ __device__ function 'isless'
//
// Status (MSVC 14.51.36231, May 2026): the primary fix is now the clang flag
// `-fno-cuda-host-device-constexpr` added to hipcc invocations in
// `hip_utils.cmake`, which the diagnostic itself recommends.  That flag tells
// clang not to implicitly mark unannotated `constexpr` functions as
// `__host__ __device__`, making MSVC's `<cmath>` overloads host-only and
// removing the collision entirely.
//
// This header remains in the build as a belt-and-suspenders backup.  It used
// to be the primary fix back when MSVC's pre-14.51.36231 `<cmath>` honoured
// pre-defined empty `_CLANG_BUILTIN1` / `_CLANG_BUILTIN2` macros; in
// 14.51.36231 the STL re-defines those macros unconditionally inside cmath
// (`#define _CLANG_BUILTIN2(NAME) ...` with no `#ifndef` guard), so empty
// pre-definitions no longer survive long enough to suppress the overload
// emission.  Force-included from `_hip_compile_sources` in
// `lib/Runtime/Kernels/cmake/hip_utils.cmake` (Windows only).

#ifndef HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
#define HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H

#if defined(_MSC_VER) && defined(__clang__) && defined(__HIP__)

// The MS STL's `_CLANG_BUILTIN1` / `_CLANG_BUILTIN2` / `_CLANG_BUILTIN2_TEMPLATED`
// expansion in `<cmath>` is what produces the conflicting `inline` overloads.
// Pre-define them as empty so the expansion produces no declarations.
#define _CLANG_BUILTIN1(NAME)
#define _CLANG_BUILTIN2(NAME)
#define _CLANG_BUILTIN2_TEMPLATED(NAME)

#endif // _MSC_VER && __clang__ && __HIP__

#endif // HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
