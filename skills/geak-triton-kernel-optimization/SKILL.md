---
name: geak-triton-kernel-optimization
description: This skill should be used when optimizing Triton kernels generated or tuned by GEAK on AMD GPUs, focusing on ROCm-specific occupancy, MFMA instruction selection, FP8 constraints, grid/XCD mapping, fast-math tradeoffs, and backend parameter tuning.
---

# GEAK Triton Kernel Optimization

## Purpose
Provide practical optimization patterns and decision rules for improving performance of GEAK-generated Triton kernels on AMD ROCm platforms, covering occupancy guarantees, FP8 MFMA enablement, padding strategies, fast-math tradeoffs, XCD remapping, and ROCm backend–specific tuning knobs.

This skill is intended to guide agent-driven Triton kernel generation and refinement, not just manual optimization.

## When to Use
- Optimizing or reviewing Triton kernels
- Performance tuning on AMD GPUs (e.g., MI300 / MI355X)
- Ensuring correct usage of FP8 MFMA instructions
- Diagnosing unexpectedly low occupancy or instruction selection issues
- Providing hardware-aware constraints to an LLM-based kernel generator


## Skill Execution Protocol
When this skill is activated, follow this procedure:
1. Identify hardware context
- GPU model (e.g., MI355X)
- Number of CUs and XCDs
- ROCm + Triton backend in use
2. Inspect kernel launch configuration
- Grid dimensionality
- Block sizes (BLOCK_M/N/K, num_warps, num_stages)
3. Validate instruction selection
- Confirm MFMA / FP8 MFMA generation at ASM level
4. Apply optimizations in priority order
- Occupancy and grid alignment
- MFMA shape constraints and padding
- Backend tuning knobs
- Cache / XCD locality
- Fast-math substitutions
5. Re-validate via assembly inspection and profiling

## Optimization Priority

**Phase 1: Structural & Low-Risk Fixes**
1. Ensure grid size is a multiple of CU count
2. Provide explicit hardware information (CU/XCD) to GEAK
3. Enable fast-math selectively where numerically safe
4. Verify generated instructions via ASM dump

**Phase 2: Targeted Performance Gains**
5. Adjust block shapes to enable FP8 MFMA
6. Add padding to satisfy MFMA shape constraints
7. Tune Triton ROCm backend parameters
8. Remap blocks to improve XCD/L2 locality

**Phase 3: Aggressive / Architecture-Aware Changes**
9. Restructure grids for XCD-aware scheduling
10. Trade precision for speed using exp2 substitutions
11. Large parameter sweeps (e.g., num_kv_split)

## Core Optimization Patterns

### 1. Grid Size and Occupancy
- **Guideline**: Always ensure: `grid_size % num_CU == 0`; Explicitly get the CU count (e.g., MI308 has 304 CUs).
- **Observed Impact**: Occupancy stabilization; Avoids partial CU utilization
- **Example**: In MLA optimization, increasing num_kv_split while keeping grid aligned to CU count can yield 10–20× speedups.

### 2. FP8 MFMA Instruction Enablement
- **Supported FP8 MFMA Instructions**: `V_MFMA_F32_16x16x128_F8F6F4`; `V_MFMA_F32_32x32x64_F8F6F4`
- **Hard Constraints**: 
  FP8 MFMA instructions are only generated if: `M`, `N`, and `K` are exact multiples of the MFMA shape
  If any dimension violates the constraint: Triton silently falls back to non-MFMA paths
- **Optimization Strategy**: 
  Adjust `BLOCK_M`, `BLOCK_N`, `BLOCK_K` to match MFMA shapes
  If exact matching is impossible: Add padding (usually net positive)
- **Example**:
  MLA kernel required adjusting `BLOCK_N` to trigger FP8 MFMA
  Padding introduced when shape alignment was impossible

### 3. Padding as a First-Class Optimization
- **Rule**: Padding to enable MFMA is usually worth it.
  Use padding when:
    It enables FP8 MFMA
    It preserves vectorized memory access
    It does not excessively inflate memory footprint

### 4. Assembly Inspection & Verification
- **Enable Triton Compilation Dumps**: Set environment variables:
  ```bash
  TRITON_ALWAYS_COMPILE=1
  MLIR_ENABLE_DUMP=1
  TRITON_DISABLE_LINE_INFO=1
  AMDGCN_ENABLE_DUMP=1
  ```

- **Inspect Generated Assembly**:
  ```bash
  llvm-objdump -d kernel.co
  ```

- **Validation Goals**:
  Confirm MFMA / FP8 MFMA instructions are present
  Detect fallback to scalar or vector FMA
  Check instruction mix after fast-math changes

- **Adaptive grid sizing**: Don't use fixed grid for variable problem sizes; adapt to small dimensions

### 5. Fast-Math Tradeoffs
- **Guideline**: Consider `exp2` instead of `exp` where numerically acceptable

- **Example**:
  MLA kernel:
    3 total exp usages
    2 can be replaced with exp2
    1 causes unacceptable precision loss

- **Rule**:
  Apply fast-math selectively, not globally
  Validate numerics after each substitution

### 6. XCD-Aware Grid Remapping
- Goal: Improve L2 cache locality by mapping adjacent blocks to the same XCD.
- Pattern: Generate a 1D grid; Remap block IDs based on `NUM_XCD`
- Reference Implementation: AITer provides a ready-to-use implementation
  ```python
  def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
      ## pid remapping on xcds
      # Number of pids per XCD in the new arrangement
      pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
      # When GRID_MN cannot divide NUM_XCDS, some xcds will have
      # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
      # We calculate the number of xcds that have pids_per_xcd pids as
      # tall_xcds
      tall_xcds = GRID_MN % NUM_XCDS
      tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
      # Compute current XCD and local pid within the XCD
      xcd = pid % NUM_XCDS
      local_pid = pid // NUM_XCDS
      # Calculate new pid based on the new grouping
      # Note that we need to consider the following two cases:
      # 1. the current pid is on a tall xcd
      # 2. the current pid is on a short xcd
      if xcd < tall_xcds:
          pid = xcd * pids_per_xcd + local_pid
      else:
          pid = (
              tall_xcds * pids_per_xcd
              + (xcd - tall_xcds) * (pids_per_xcd - 1)
              + local_pid
          )

      return pid
  ```


### 7. ROCm-Specific Triton Backend Tuning
- Triton ROCm backend exposes AMD-specific parameters, including: `waves_per_eu`, `matrix_instr_nonkdim`
- These parameters can materially impact: Occupancy, MFMA scheduling, Register pressure
- Guideline: Explicitly allow GEAK to tune these knobs; Treat them as architecture-dependent hyperparameters
- Reference
  ```python
  class HIPOptions:
      num_warps: int = 4
      waves_per_eu: int = 0
      num_stages: int = 2
      num_ctas: int = 1
      extern_libs: dict = None
      debug: bool = False
      sanitize_overflow: bool = True
      arch: str = None
      supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")
      deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
      default_dot_input_precision: str = "ieee"
      allowed_dot_input_precisions: Tuple[str] = ("ieee", 'bf16x3', 'bf16x6')
      enable_fp_fusion: bool = True
      launch_cooperative_grid: bool = False
      matrix_instr_nonkdim: int = 0
      kpack: int = 1
      allow_flush_denorm: bool = False
      max_num_imprecise_acc_default: int = 0
      backend_name: str = 'hip'
      instrumentation_mode: str = ""
  ```

## Validation Checklist
- [ ] Grid size is a multiple of CU count
- [ ] Hardware info (CU/XCD) provided
- [ ] Padding applied where MFMA constraints require it
- [ ] Fast-math substitutions validated for accuracy
- [ ] XCD remapping improves cache locality
- [ ] ROCm backend parameters tuned intentionally
- [ ] FP8 MFMA instructions confirmed in ASM

## Expected Output
When using this skill, produce:
- A prioritized list of Triton kernel changes
- Explicit constraints or hints to pass into GEAK
- Verification steps (ASM + profiling)
- Risk notes (precision loss, padding overhead)

## Performance Impact (Observed)

| Optimization | Typical Impact |
|-------------|----------------|
| Grid aligned to CU | Stable high occupancy |
| num_kv_split tuning | +10–20× (MLA cases) |
| FP8 MFMA enablement | Major throughput gain |
| Padding for MFMA | Usually positive |
| XCD remap | Higher L2 hit rate |
| Fast-math (exp2) | Noticeable speedup if safe |
| ROCm backend tuning | Architecture-dependent wins |
