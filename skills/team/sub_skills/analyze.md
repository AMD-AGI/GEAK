# Phase A: Kernel Analysis

## Objective
Understand the kernel being optimized: its type, structure, dependencies, and hardware environment.

## Steps

### A1: Detect Kernel Type

Read all source files in the kernel directory. Classify using this pattern table:

| Pattern | Kernel Type |
|---------|-------------|
| `.py` file with `@triton.jit` or `tl.` imports | **triton** |
| `.hip` file with `__global__` | **hip** |
| `.cu` file with `__global__` | **cuda** (treat as hip on AMD) |
| `.cpp` file with `__global__` and HIP includes | **hip** |
| `.hpp` with CK templates (`ck::tensor_operation`) | **composable_kernel** |

Record: `kernel_type`, `kernel_file` (the primary source file containing the kernel), `kernel_language`.

### A2: Analyze Kernel Structure

Read the kernel source thoroughly. Document:
1. **Entry point**: The `__global__` function (HIP) or `@triton.jit` function (Triton)
2. **Algorithm**: What does the kernel compute? (e.g., KNN search, matrix multiply, softmax)
3. **Complexity**: What is the algorithmic complexity? (e.g., O(N*M) per query point)
4. **Data structures**: What local data structures are used? (arrays, heaps, accumulators)
5. **Memory access pattern**: How is global memory accessed? (sequential, strided, random)
6. **Launch configuration**: Block size, grid dimensions, shared memory usage
7. **Potential bottlenecks**: Initial assessment based on code structure

### A3: Map Dependencies

Scan all files in the kernel directory to build a dependency tree:
```bash
# Find all source files
find $KERNEL_PATH -type f \( -name "*.py" -o -name "*.hip" -o -name "*.cu" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \)
```

For each file, identify imports/includes and classify:
- **In-repo (modifiable)**: Files within the kernel directory that can be edited. **IMPORTANT**: Always include the Python wrapper file AND the C++ binding file (`.cpp` with `PYBIND11_MODULE`) as modifiable, not just the kernel source (`.hip`/`.cu`). Wrapper optimization is a critical optimization category.
- **External (read-only)**: System libraries, framework imports (torch, triton, etc.)

### A4: Query Hardware

```bash
# GPU info
rocminfo 2>/dev/null | grep -A 5 "Name:" | head -20
rocm-smi --showid --showproductname 2>/dev/null | head -10

# Available GPUs
rocm-smi --showid 2>/dev/null || echo "rocm-smi not available"
```

### A5: Detect Existing Test Infrastructure

Search for existing test/benchmark files:
```bash
find $KERNEL_PATH -type f -name "*.py" | xargs grep -l "benchmark\|correctness\|test\|perf" 2>/dev/null
find $KERNEL_PATH -type f -name "task_runner.py" -o -name "test_*.py" -o -name "*_test.py" -o -name "bench*.py"
```

Check for config files:
```bash
find $KERNEL_PATH -name "config.yaml" -o -name "config.json" -o -name "*.cfg"
```

### A6: Save Baseline

```bash
# Copy original kernel source to eval directory
mkdir -p $EVAL_DIR/baseline
cp -r $KERNEL_PATH/* $EVAL_DIR/baseline/
```

### A7: Output

Write `$EVAL_DIR/analysis.json`:
```json
{
  "kernel_name": "<name>",
  "kernel_type": "<triton|hip|cuda>",
  "kernel_file": "<primary source file path>",
  "kernel_language": "<python|cpp|hip>",
  "algorithm": "<description>",
  "complexity": "<big-O>",
  "entry_point": "<function name>",
  "launch_config": {
    "block_size": "<value>",
    "grid": "<expression>"
  },
  "dependencies": {
    "modifiable": ["<file1>", "<file2>"],
    "external": ["<lib1>", "<lib2>"]
  },
  "hardware": {
    "gpu_name": "<name>",
    "gpu_arch": "<arch>",
    "num_gpus": "<count>"
  },
  "existing_tests": ["<file1>", "<file2>"],
  "initial_bottleneck_guess": "<memory|compute|latency|unknown>"
}
```

Write `$EVAL_DIR/codebase_context.md` — human-readable summary of the above analysis. Include the full kernel source code for easy reference by engineers.
