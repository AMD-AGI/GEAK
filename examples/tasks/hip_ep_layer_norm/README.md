# GEAK task: layer_norm (numeric oracle, gfx1151)

Optimizes the HIP EP `layer_norm` kernel. GEAK edits ONLY `kernel_src/layer_norm_kernel.hip`.

## Optimization targets (editable)
  - `hip_layer_norm_kernel`

## FROZEN contract (must NOT change — the generated EP code calls these)
  - `hip_layer_norm`

## Run
```
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```
