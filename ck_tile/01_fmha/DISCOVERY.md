# CK-Tile FMHA Kernel Discovery Report

## Kernel Overview
- **Type**: Composable Kernel (CK) Tile-based FMHA (Fused Multi-Head Attention)
- **Language**: C++ with template meta-programming
- **Location**: `/workspace/GEAK/ck_tile/01_fmha/`

## Key Files
1. **fmha_fwd.hpp** - Main forward kernel template definitions
2. **fmha_fwd_runner.hpp** - Runner/launcher code
3. **example_fmha_fwd.cpp** - Example/test executable
4. **generate.py** - Python codegen script that generates kernel instances
5. **codegen/ops/fmha_fwd.py** - Forward pass codegen logic

## Kernel Architecture
- **Pipeline-based design**: Uses BlockFmhaPipeline with configurable tile sizes
- **Template parameters**:
  - Block tiles: BM0, BN0, BK0, BN1, BK1, BK0MAX
  - Warp/thread tiles: RM0, RN0, RK0, WM0, WN0, WK0 (for QK), RM1, RN1, RK1, WM1, WN1, WK1 (for PV)
  - Layout: V matrix layout (row-major or column-major)
  - Pipeline type: Different pipeline implementations
- **Supported features**:
  - Multiple data types: FP32, FP16, BF16, FP8, BF8
  - Batch and group modes
  - MQA/GQA support
  - Causal masking, sliding window attention
  - Bias, dropout, LSE (log-sum-exp)
  - Variable sequence lengths

## Optimization Strategy
Since this is C++ template code (not Python/Triton), the optimization approach is:
1. **Tile size tuning**: Adjust block/warp/thread tile dimensions
2. **Pipeline selection**: Choose optimal pipeline implementation
3. **Layout optimization**: Test row-major vs column-major V layout
4. **Occupancy tuning**: Adjust blocks per CU
5. **Memory access patterns**: Optimize padding and striding

## Build System
- Uses CMake with ninja
- Build command: `make tile_example_fmha_fwd`
- Binary location: `/workspace/composable_kernel/build/bin/tile_example_fmha_fwd`
- Build time: Several minutes (too long for interactive optimization)

## Testing
- Test script created: `test_fmha_combined.py`
- Build script: `build_and_test.sh`
- Configurations:
  - Correctness: `-mode=0 -b=1 -h=2 -s=128 -d=64 -v=1 -prec=fp16`
  - Benchmark: `-mode=0 -b=2 -h=8 -s=512 -d=128 -v=0 -prec=fp16 -repeat=20`

## Current Limitation
The standard OpenEvolve workflow is designed for Python/Triton kernels with `@triton.jit` decorators.
CK-Tile uses C++ templates with Python codegen, requiring a different optimization approach.

## Next Steps
Need to determine if:
1. Manual optimization of template parameters in codegen files
2. Automated search over configuration space
3. Or focus on analyzing and documenting optimization opportunities
