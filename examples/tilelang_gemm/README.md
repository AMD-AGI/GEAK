# TileLang GEMM Example

This example is a controlled TileLang GEMM optimization task derived from the official
`tile-ai/tilelang-benchmark` CDNA GEMM benchmark.

The starting kernel is intentionally slower than the measured reference kernel in two ways:

1. It changes several official schedule-space constants.
2. It changes the kernel body by staging the accumulator through shared memory before
   the final global store.

```diff
-BLOCK_M = 64
-BLOCK_N = 64
+BLOCK_M = 128
+BLOCK_N = 256
 BLOCK_K = 64
-NUM_STAGES = 0
-THREADS = 128
-ENABLE_RASTERIZATION = False
+NUM_STAGES = 4
+THREADS = 256
+ENABLE_RASTERIZATION = True
```

GEAK should recover performance by tuning several TileLang schedule constants back
toward a better official-search-space schedule, and by removing the unnecessary
output shared-memory staging when the benchmark confirms that direct accumulator
stores are faster.

The kernel-level degradation is:

```diff
-T.copy(C_local, C[by * block_M, bx * block_N])
+C_shared = T.alloc_shared((block_M, block_N), dtype)
+...
+T.copy(C_local, C_shared)
+T.copy(C_shared, C[by * block_M, bx * block_N])
```

## Run The Harness

Run from this directory with a TileLang-capable ROCm environment:

```bash
python test_tilelang_gemm_harness.py --correctness
python test_tilelang_gemm_harness.py --profile --iterations 3
python test_tilelang_gemm_harness.py --benchmark --iterations 5 --warmup 3
python test_tilelang_gemm_harness.py --full-benchmark --iterations 5 --warmup 3
```

The harness benchmarks:

- `(2048, 2048, 2048)`
- `(4096, 1024, 4096)`

and prints `GEAK_RESULT_LATENCY_MS` for GEAK result parsing.

## Run With GEAK

From the GEAK repository root:

```bash
python -m minisweagent.run.mini \
  --model-class codex_cli \
  --model codex-cli \
  --mode quick \
  --repo examples/tilelang_gemm \
  --kernel-path kernel.py \
  --test-command "python test_tilelang_gemm_harness.py --correctness" \
  --task examples/tilelang_gemm/task.md \
  --output examples/tilelang_gemm/geak_run \
  --num-parallel 1 \
  --gpu-ids 0 \
  -y -l 0
```

On MI300X with ROCm 7.2, the starting kernel changes multiple official-search-space
constants and adds an unnecessary output staging path. Exact numbers vary by runtime
state, but this example is designed to expose visible TileLang kernel-body and schedule
optimization opportunities.

## Source

This is a small, self-contained derivative of:

`tilelang-benchmark/cdna_benchmark/gemm_benchmark/1.tilelang_benchmark/benchmark_tilelang_matmul.py`

Reference commit used during extraction:

`4272166e995442bb1fe273b6764845bdb7c42416`
