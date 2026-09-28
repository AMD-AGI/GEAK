# GEAK task: matmul_nbits (fast oracle, gfx1151)

Optimizes the HIP EP `matmul_nbits` kernel. GEAK edits ONLY `kernel_src/matmul_nbits_kernel.hip`.

## Optimization targets (editable)
  - `matmul_nbits_kernel_u2`
  - `matmul_nbits_gemv_kernel_u2`
  - `GemmFp16U2Impl`
  - `dequant_u2_to_fp16`
  - `matmul_nbits_kernel_u3`
  - `matmul_nbits_gemv_kernel_u3`
  - `GemmFp16U3Impl`
  - `matmul_nbits_kernel_u4`
  - `matmul_nbits_gemv_kernel_u4`
  - `GemmFp16U4Impl`

## FROZEN contract (must NOT change — the generated EP code calls these)
  - `hip_matmul_nbits`
  - `ZeroPointsU2`
  - `zp_elem_size`

## Run
```
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```
