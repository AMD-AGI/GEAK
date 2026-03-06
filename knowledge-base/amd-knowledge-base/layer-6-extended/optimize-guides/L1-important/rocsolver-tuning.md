Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "rocsolver", "tuning", "linear-algebra"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/tuning.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Tuning rocSOLVER Performance

## Overview

rocSOLVER provides compile-time tunable parameters to optimize library performance for specific use cases, matrix dimensions, and hardware configurations. These constants are located in `library/src/include/ideal_sizes.hpp` and require rebuilding the library from source to take effect.

> "The effect of changing a tunable constant on the performance of the library is difficult to predict and such analysis is beyond the scope of this document."

Advanced users should exercise caution when modifying these values, as changes may improve or degrade performance unpredictably.

## QR/QL Factorization Functions

### GEQxF_BLOCKSIZE
Controls the column block size during QR/QL factorization (GEQRF/GEQLF). Applies to batched and strided-batched variants.

### GEQxF_GEQx2_SWITCHSIZE
Determines when to transition from blocked (GEQRF/GEQLF) to unblocked (GEQR2/GEQL2) algorithms. When remaining dimensions fall below this threshold, the unblocked method handles the final block.

## RQ/LQ Factorization Functions

### GExQF_BLOCKSIZE
Specifies the row block size for RQ/LQ factorization (GERQF/GELQF).

### GExQF_GExQ2_SWITCHSIZE
Establishes the threshold for switching from blocked to unblocked algorithms in RQ/LQ operations.

## Orthogonal Matrix Generation

### xxGQx_BLOCKSIZE & xxGQx_xxGQx2_SWITCHSIZE
Control block reflector size and algorithm switching for ORGQR/UNGQR and ORGQL/UNGQL functions.

### xxGxQ_BLOCKSIZE & xxGxQ_xxGxQ2_SWITCHSIZE
Manage block reflector dimensions and thresholds for ORGRQ/UNGRQ and ORGLQ/UNGLQ operations.

## Matrix-by-Q Multiplication

### xxMQx_BLOCKSIZE
Determines block reflector size for ORMQR/UNMQR and ORMQL/UNMQL. Acts as a switch parameter—when reflector count ≤ this value, unblocked routines execute directly.

### xxMxQ_BLOCKSIZE
Specifies block reflector size for ORMRQ/UNMRQ and ORMLQ/UNMLQ with similar switching behavior.

## Bidiagonal Reduction

### GEBRD_BLOCKSIZE & GEBRD_GEBD2_SWITCHSIZE
Control block size and algorithm selection for bidiagonal matrix reduction (GEBRD).

## Singular Value Decomposition

### BDSQR_SWITCH_SIZE
Determines thread group strategy for bidiagonal SVD. Single thread group handles singular vector updates when dimensions ≤ threshold; otherwise, multiple thread groups launch.

### BDSQR_ITERS_PER_SYNC
Specifies iterations executed between device synchronizations in multi-kernel SVD algorithm.

### THIN_SVD_SWITCH
Triggers thin SVD computation when one matrix dimension exceeds another by this factor. Leverages QR/LQ preprocessing for elongated matrices.

## Tridiagonal Reduction

### xxTRD_BLOCKSIZE & xxTD2_SWITCHSIZE
Control block size and algorithm switching for symmetric/Hermitian tridiagonal reduction (SYTRD/HETRD).

## Generalized Eigenproblems

### xxGST_BLOCKSIZE
Manages block size during reduction to standard form (SYGST/HEGST). Also serves as switch parameter—matrices ≤ this size use unblocked routines exclusively.

## Eigenvalue Computations

### STEDC_MIN_DC_SIZE & STEDC_NUM_SPLIT_BLKS
Govern divide-and-conquer eigenvector computation thresholds and parallel block analysis in SYEVD/HEEVD.

### SYEVJ_BLOCKED_SWITCH
Selects between single-kernel (small matrices) and multi-kernel (large matrices) approaches in Jacobi eigensolvers. Must be ≤ 64.

### SYEVDJ_MIN_DC_SIZE
Establishes minimum size for Jacobi divide-and-conquer method in SYEVDJ/HEEVDJ.

## Cholesky Factorization

### POTRF_BLOCKSIZE & POTRF_POTF2_SWITCHSIZE
Define block dimensions and algorithmic thresholds for Cholesky decomposition (POTRF).

## Bunch-Kaufman Factorization

### SYTRF_BLOCKSIZE & SYTRF_SYTF2_SWITCHSIZE
Manage maximum partial factorization size and algorithm switching for symmetric indefinite factorization (SYTRF).

## LU Factorization

### GETF2/GETRF Parameters
Multiple tunable constants govern block sizes and intervals across standard, batched, and non-pivoted variants of LU factorization.

## Matrix Inversion & Triangular Inversion

### GETRI & TRTRI Parameters
Separate tunable constants control block sizes and intervals for matrix inversion (GETRI) and triangular matrix inversion (TRTRI) across regular and batched implementations.
