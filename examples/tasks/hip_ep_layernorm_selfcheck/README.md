# GEAK task: layernorm (selfcheck oracle, gfx1151)

A self-contained GEAK optimization task for a row-wise LayerNorm HIP kernel.
GEAK edits ONLY `kernel_src/layernorm_kernel.hip`; correctness and speed are
judged by the frozen harness, which carries its own CPU reference and does the
GPU/CPU timing — no ORT, NumPy, or external framework.

## Optimization target (editable)
- `kernel_src/layernorm_kernel.hip` — the `layernorm_kernel` __global__ body and
  the `layernorm_launch` wrapper (block size, reduction, tiling, vectorization
  are all fair game).

## FROZEN contract (must NOT change)
- `layernorm_launch` — the `extern "C"` ABI in `include/layernorm_op.h`. The
  harness calls it by name/signature. Name + signature must stay byte-identical.
- `harness/test_layernorm.cpp` — the CPU reference + timing driver.

## Oracle
`selfcheck`: `scripts/task_runner.py` builds the kernel + harness with hipcc
(two TUs), runs each shape in `task_meta.json`, checks GPU output vs the
in-harness CPU reference (PASS/FAIL), and reports the GPU ms per shape.

## Run
```
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```
