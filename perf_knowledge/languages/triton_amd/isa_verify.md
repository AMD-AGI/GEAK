---
title: Inspect Triton AMD lowering
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-10-16
---

# Inspect Triton AMD lowering

Inspect compiler evidence when a decision depends on layout, instructions,
scheduling or resource use. Timing establishes performance; ISA helps test an
explanation. Use the compiled object associated with the measured source/config,
not an unrelated cache entry.

For builds supporting these dump switches:

```bash
AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 \
TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 \
python my_kernel.py 2> dump.txt
rg 'vgpr_count|sgpr_count|group_segment_fixed_size|private_segment_fixed_size|num-warps|triton_gpu\.shared|ttg\.shared' dump.txt
```

`TRITON_ALWAYS_COMPILE` bypasses compilation reuse; instrumentation/compilation time
is not the accepted kernel score. Verify switch support in the installed backend.
Read the relevant region rather than treating every emitted pass as required evidence.
Newer Triton dumps use the `ttg.*` namespace (for example `ttg.shared` and
`ttg.num-warps`); older dumps can use `triton_gpu.*`. Check the actual keys.

| Question | Inspect |
|---|---|
| Was a load vectorized? | Actual load/store widths, alignment and masks; narrow accesses may be necessary for the layout |
| Did matmul use the intended mode? | `v_mfma*` variant, operand/accumulator type and dynamic work assumptions |
| Did a layout change add movement? | `convert_layout`, LDS/shuffle/reload instructions and barriers |
| Did live state increase cost? | Target-specific register allocation, `v_accvgpr` movement and actual scratch accesses |
| Was overlap realized? | Loop dependencies, wait instructions and available trace evidence; a static wait alone does not quantify stall time |

A nonzero private segment is not automatically dynamic spilling to HBM. Fixed
LDS fields exclude dynamic allocation, and compiler register estimates do not
measure achieved occupancy. Do not use a universal register rounding formula or
infer a bottleneck from one count. Use the [hardware-evidence guide](../../skills/kernel-hardware-evidence/references/ir-isa-dataflow.md).

Check FP8/FNUZ versus OCP mode against the actual target and input encoding. Match
MFMA layout definitions using the [matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).
After an explanatory change, retain correctness and separately measured unprofiled
full-call latency.

## Sources

- [Triton AMD backend](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)
- [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)
- [AMD Triton optimization](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)
