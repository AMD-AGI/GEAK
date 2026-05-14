# Phase 1: Kernel Analysis

## Objective
Understand the kernel codebase, detect kernel type, build dependency context, and identify optimization targets.

## Steps

### 1.1 Read Kernel Source
```bash
cat "$KERNEL_PATH"
```
- Understand the full implementation
- Identify key functions, data structures, algorithms
- Note the programming model (Triton `@triton.jit`, HIP `__global__`, CK template, etc.)

### 1.2 Detect Kernel Type

Classify the kernel based on file extension and content:

| Pattern | Type |
|---------|------|
| `.py` with `@triton.jit` or `tl.` | **Triton** |
| `.hip`, `.cu`, `.cpp` with `__global__` | **HIP** |
| `.hpp` with CK templates | **Composable Kernel** |
| `.s` or `.asm` | **Assembly** |

Save the detected type as `$KERNEL_TYPE` for later phases.

### 1.3 Build Codebase Context

Analyze the repository structure around the kernel:

```bash
# Find the repo structure
find "$REPO_ROOT" -name "*.py" -o -name "*.hip" -o -name "*.cu" -o -name "*.cpp" -o -name "*.hpp" | head -50

# Find imports/dependencies of the kernel file
grep -n "import\|#include\|from .* import" "$KERNEL_PATH"
```

Build a dependency tree:
1. List all files the kernel imports/includes
2. For each dependency, check if it's in-repo (potential optimization target) or external
3. Note which functions are imported from each dependency

### 1.4 Hardware Context

Query the GPU hardware:
```bash
rocminfo | grep -A5 "Name:\|Marketing Name:\|Compute Unit:\|Wavefront"
rocm-smi --showproductname
```

Extract: GPU architecture (gfx942), CU count, wavefront size, memory bandwidth.

### 1.5 Snapshot Baseline Kernel

Copy the original kernel source to the baseline directory for reference:
```bash
KERNEL_FILENAME=$(basename "$KERNEL_PATH")
cp "$KERNEL_PATH" "$EVAL_DIR/baseline/$KERNEL_FILENAME"
```

If additional source files are part of the kernel (e.g., a `.cpp` binding alongside a `.hip` kernel),
copy those too:
```bash
# Copy all source files in the kernel's directory
for f in $(find "$(dirname "$KERNEL_PATH")" -maxdepth 1 -name "*.hip" -o -name "*.cu" -o -name "*.cpp" -o -name "*.py" | head -10); do
    cp "$f" "$EVAL_DIR/baseline/"
done
```

### 1.6 Output

Create `$EVAL_DIR/logs/analysis.json` with:
```json
{
  "kernel_path": "/absolute/path/to/kernel",
  "kernel_type": "triton|hip|ck|asm",
  "kernel_name": "name_of_kernel",
  "function_names": ["kernel_func1", "kernel_func2"],
  "language": "python|cpp|asm",
  "repo_root": "/absolute/path/to/repo",
  "dependencies": [
    {"file": "path/to/dep.py", "in_repo": true, "imports": ["func1", "func2"]}
  ],
  "gpu_arch": "gfx942",
  "gpu_name": "MI300X",
  "compute_units": 304
}
```

Also create `$EVAL_DIR/logs/codebase_context.md` with a human-readable summary:
```markdown
# Codebase Context

## Kernel: {kernel_name}
- Path: {kernel_path}
- Type: {kernel_type}
- Functions: {function_names}

## Dependency Tree
- {dep1}: imports {func1, func2} (in-repo, optimization target)
- {dep2}: imports {func3} (external, not modifiable)

## Repository Structure
{tree output}
```
