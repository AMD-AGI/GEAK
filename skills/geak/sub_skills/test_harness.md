# Phase 2: Test Harness Setup

## Objective
Discover existing tests or create a test harness that validates correctness and measures performance.
Generate the COMMANDMENT.md evaluation contract.

## Steps

### 2.1 Discover Existing Tests

Search the repository for existing test infrastructure:
```bash
# Find test files
find "$REPO_ROOT" -name "test_*.py" -o -name "*_test.py" -o -name "tests.py" | head -20

# Find benchmark files
find "$REPO_ROOT" -name "bench*.py" -o -name "*benchmark*.py" | head -20

# Find Makefiles, CMakeLists, setup files
find "$REPO_ROOT" -name "Makefile" -o -name "CMakeLists.txt" -o -name "setup.py" | head -10
```

If existing tests are found, check if they support the required modes:
- `--correctness`: Validate output correctness
- `--profile`: Run kernel once for profiling (minimal GPU allocations)
- `--benchmark`: Quick benchmark (few iterations)
- `--full-benchmark`: Authoritative benchmark (many iterations)

### 2.2 Create Test Harness (if needed)

If no suitable test exists, create one at `$EVAL_DIR/logs/test_harness.py`.

The harness MUST:
1. Use `argparse` with four mode flags: `--correctness`, `--profile`, `--benchmark`, `--full-benchmark`
2. Import and use the kernel correctly
3. Generate representative test inputs with correct shapes and dtypes
4. For correctness: compare kernel output against a reference implementation
5. For profile: run kernel once with minimal GPU allocations (no `torch.randn(..., device='cuda')` inside profile mode -- allocate on CPU then `.to('cuda')`)
6. For benchmark: run kernel N times and report `GEAK_RESULT_LATENCY_MS=<value>`
7. For full-benchmark: run kernel more times for authoritative measurement

**Critical output format**: The benchmark MUST print:
```
GEAK_RESULT_LATENCY_MS=<float_value>
```

Use the helper script:
```bash
python3 ${SKILL_DIR}/scripts/create_harness.py \
  --kernel-path "$KERNEL_PATH" \
  --kernel-type "$KERNEL_TYPE" \
  --output "$EVAL_DIR/logs/test_harness.py"
```

### 2.3 Validate Harness

**Static validation:**
- Verify argparse/click/typer is used
- Verify all 4 flag strings appear in source
- Verify no GPU tensor allocation in profile function

**Runtime validation:**
```bash
# Test all modes
python3 "$HARNESS_PATH" --correctness
python3 "$HARNESS_PATH" --benchmark
python3 "$HARNESS_PATH" --profile
python3 "$HARNESS_PATH" --full-benchmark
```

If any mode fails, fix the harness and retry (up to 3 attempts).

### 2.4 Generate COMMANDMENT.md

Create `$EVAL_DIR/logs/COMMANDMENT.md` -- the evaluation contract that all optimization workers must follow:

```markdown
# COMMANDMENT

## SETUP
```
cd $REPO_ROOT
# Any setup commands (install deps, source env, etc.)
```

## CORRECTNESS
```
python3 $HARNESS_PATH --correctness
```

## PROFILE
```
python3 $HARNESS_PATH --profile
```

## BENCHMARK
```
python3 $HARNESS_PATH --benchmark
```

## FULL_BENCHMARK
```
python3 $HARNESS_PATH --full-benchmark
```
```

**Rules:**
- Section headers MUST be exactly: `## SETUP`, `## CORRECTNESS`, `## PROFILE`, `## BENCHMARK`, `## FULL_BENCHMARK`
- Commands must NOT start with `cd`, `source`, `export`, or any shell built-in
- Use absolute paths in all commands
- The harness file and COMMANDMENT are IMMUTABLE during optimization -- workers must NOT modify them

### 2.5 Record Test Command

Save the canonical test command:
```bash
echo "python3 $HARNESS_PATH" > "$EVAL_DIR/logs/test_command.txt"
```
