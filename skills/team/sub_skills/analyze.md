# Phase: Kernel Analysis

Analyze the target kernel to understand its algorithm, dependencies, and hardware context.

## Steps

### 1. Read and Understand the Kernel Source

Read the kernel source file at `$TASK_PATH`. Identify:
- **Kernel type**: Check file extension and content patterns:
  - `.hip` / `.cu` / `.cpp` with `__global__` → HIP kernel
  - `.py` with `@triton.jit` or `tl.` → Triton kernel
  - `.hpp` with Composable Kernel templates → CK kernel
- **Algorithm**: What does the kernel compute? What is the mathematical operation?
- **Data flow**: Input tensors (shapes, dtypes) → computation → output tensors
- **Memory access patterns**: Sequential, strided, random? Read-heavy or write-heavy?
- **Hot loops**: Innermost loops with the most iterations — these are the optimization targets
- **Data structures**: Arrays, heaps, trees, etc. in the kernel
- **Fixed-size allocations**: Any hardcoded array sizes (e.g., `float data[100]`) — optimization opportunity

### 2. Build Dependency Tree

Starting from the kernel source, trace all includes/imports:

```bash
# For HIP kernels
grep -rn '#include' $TASK_PATH
# Find all source files in the repo
find $REPO_ROOT/src -name '*.hip' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp'
```

Classify each dependency as:
- **In-repo (modifiable)**: Source files in the project that can be optimized
- **External (not modifiable)**: System headers, library includes (torch, hip, etc.)

### 3. Check for Existing Infrastructure

Look for existing build/test infrastructure:

```bash
# Config file
cat $REPO_ROOT/config.yaml 2>/dev/null

# Task runner
ls $REPO_ROOT/scripts/task_runner.py 2>/dev/null

# Makefile
ls $REPO_ROOT/Makefile 2>/dev/null

# Kernel loader (PyTorch JIT)
cat $REPO_ROOT/kernel_loader.py 2>/dev/null

# Python wrapper
find $REPO_ROOT -name '*wrapper*' -o -name '*_ext*' | head -5
```

If `config.yaml` exists, read it for: `source_file_path`, `compile_command`, `correctness_command`, `performance_command`.

### 4. Query GPU Hardware

```bash
rocminfo 2>/dev/null | grep -A5 "Name:\s*gfx"
rocm-smi --showproductname 2>/dev/null | head -5
```

Record the GPU architecture (e.g., gfx942 for MI300X).

### 5. Snapshot Baseline

Copy the original kernel source to the baseline directory:

```bash
cp $TASK_PATH $EVAL_DIR/baseline/
```

### 6. Output

Write `$EVAL_DIR/logs/analysis.json`:

```json
{
  "kernel_type": "hip|triton|ck",
  "task_path": "/absolute/path/to/kernel.hip",
  "source_files": ["src/kernel.hip", "src/kernel.cpp"],
  "algorithm_summary": "Brief description of what the kernel computes",
  "hot_loops": "Description of the innermost computation loops",
  "memory_patterns": "Description of memory access patterns",
  "optimization_opportunities": ["opportunity 1", "opportunity 2"],
  "gpu_arch": "gfx942",
  "has_task_runner": true,
  "has_config": true,
  "compile_command": "python3 scripts/task_runner.py compile",
  "correctness_command": "python3 scripts/task_runner.py correctness",
  "performance_command": "python3 scripts/task_runner.py performance"
}
```

Write `$EVAL_DIR/logs/codebase_context.md` — human-readable summary of the kernel, its dependencies, and the repo structure. Include file paths and key code locations that engineers will need.
