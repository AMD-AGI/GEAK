# Phase B: Benchmark Setup

## Objective
Establish reliable, immutable measurement infrastructure. All engineers must use the same benchmark methodology.

## Steps

### B1: Check for Existing Infrastructure

From the analysis (Phase A), check if test infrastructure already exists:
- `task_runner.py` or `scripts/task_runner.py` — structured test runner
- `test_*.py` or `*_test.py` — test files
- `bench*.py` — benchmark files
- `config.yaml` — task configuration

If a `task_runner.py` exists with `compile`, `correctness`, and `performance` modes, validate it (skip to B3).

### B2: Create Test Harness (if needed)

If no suitable test infrastructure exists, create a test harness at `$KERNEL_PATH/test_harness.py`.

The harness MUST support these 4 modes via argparse:
1. `--correctness` — Validate kernel output against reference implementation
2. `--profile` — Single kernel run for profiler attachment (minimal GPU allocations)
3. `--benchmark` — Quick measurement (30 iterations, 10 warmup)
4. `--full-benchmark` — Authoritative measurement (100 iterations, 10 warmup)

**Harness requirements:**

**Timing**: Use CUDA events for GPU-accurate timing:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
# kernel call
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

**Output format**: Every benchmark run MUST print this line for each test case:
```
GEAK_RESULT_LATENCY_MS=<float_value>
```

**Test cases**: Include multiple shapes/configurations that cover:
- Small inputs (edge case, launch overhead dominated)
- Medium inputs (typical use case)
- Large inputs (memory/compute dominated)
- Different parameter variations

**Correctness**: Compare against a known-correct reference (PyTorch CPU, naive implementation, etc.). Use `torch.allclose()` or equivalent with appropriate tolerance.

**Profile mode**: For `--profile`, minimize GPU memory allocation:
```python
# GOOD: Allocate on CPU, then move to GPU
data = torch.randn(shape, device='cpu').to('cuda')
# BAD: Direct GPU allocation (interferes with profiler)
data = torch.randn(shape, device='cuda')
```

### B3: Validate Harness

Run all 4 modes and verify they work:

```bash
# Determine the test command based on what exists
# If task_runner.py exists:
cd $KERNEL_PATH && python scripts/task_runner.py compile
cd $KERNEL_PATH && python scripts/task_runner.py correctness
cd $KERNEL_PATH && python scripts/task_runner.py performance

# If test_harness.py was created:
cd $KERNEL_PATH && python test_harness.py --correctness
cd $KERNEL_PATH && python test_harness.py --profile
cd $KERNEL_PATH && python test_harness.py --benchmark
cd $KERNEL_PATH && python test_harness.py --full-benchmark
```

**Validation checks:**
- [ ] Correctness mode passes (all test cases)
- [ ] Benchmark mode produces `GEAK_RESULT_LATENCY_MS` output (or structured JSON performance report)
- [ ] Profile mode runs without crash
- [ ] No GPU memory errors

### B4: Create COMMANDMENT

Write `$EVAL_DIR/COMMANDMENT.md` — the immutable evaluation contract:

```markdown
# COMMANDMENT — Immutable Evaluation Contract

This file defines the EXACT commands for evaluation. 
Engineers MUST use these commands and MUST NOT modify them.

## SETUP
```
cd $KERNEL_PATH
rm -rf build/ __pycache__/ *.so
```

## CORRECTNESS
```
<exact correctness command>
```

## BENCHMARK
```
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <exact benchmark command>
```

## FULL_BENCHMARK
```
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <exact full benchmark command>
```

## PROFILE
```
bash $SKILL_DIR/scripts/profile_kernel.sh $GPU_ID "<benchmark_cmd>" $EVAL_DIR/profile_output
```

## MODIFIABLE FILES
Only files under `$KERNEL_PATH` may be modified (kernel source, wrapper, C++ bindings).
NEVER modify files outside `$KERNEL_PATH`.

## RULES
1. NEVER modify this file
2. NEVER modify the test harness or task_runner
3. NEVER modify files outside $KERNEL_PATH
4. ALWAYS clear build cache before benchmarking (SETUP)
5. ALWAYS run CORRECTNESS before BENCHMARK
6. ALWAYS use gpu_lock.sh for BENCHMARK and FULL_BENCHMARK
7. The BENCHMARK output is the source of truth for speedup claims
```

Adapt the commands based on what test infrastructure exists (task_runner.py vs test_harness.py).

### B5: Record Baseline Timing

Run the benchmark (with gpu_lock) and record baseline results:

```bash
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <benchmark_command>
```

Parse the output and save to `$EVAL_DIR/baseline_timing.json`:
```json
{
  "test_cases": [
    {"name": "case_0", "latency_ms": 0.123, "params": "..."},
    {"name": "case_1", "latency_ms": 0.456, "params": "..."}
  ],
  "total_latency_ms": 0.579,
  "num_test_cases": 2
}
```

### B6: Verify Baseline Reliability

Run the benchmark 3 times and check that results are within 5% of each other. If variance > 5%, investigate (GPU throttling, other processes, etc.) and re-run.

## Output
- Test harness (if created): `$KERNEL_PATH/test_harness.py`
- COMMANDMENT: `$EVAL_DIR/COMMANDMENT.md`
- Baseline timing: `$EVAL_DIR/baseline_timing.json`
