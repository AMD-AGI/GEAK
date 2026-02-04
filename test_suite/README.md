# GEAK Agent Test Suite

Regression test suite with 10 diverse AITER kernels for measuring agent capabilities.

## Kernels (10 total)

| ID | Category | Description |
|----|----------|-------------|
| gemm_a16w16 | gemm_basic | Basic FP16 GEMM |
| gemm_a8w8 | gemm_quantized | INT8 quantized GEMM |
| fused_gemm_a8w8_blockscale | gemm_fused | Fused INT8 GEMM with block scaling |
| mha | attention | Multi-head attention |
| pa_decode | attention_paged | Paged attention decode |
| flash_attn_prefill | flash_attention | Flash attention forward prefill |
| moe_op | moe | Mixture of Experts |
| rmsnorm | normalization | RMS normalization |
| activation | activation | Activation functions |
| softmax | softmax | Softmax kernel |

## Usage

```bash
# List available kernels
python3 test_suite/run_suite.py --list

# Run full suite
python3 test_suite/run_suite.py

# Run specific kernels
python3 test_suite/run_suite.py --kernels gemm_a16w16 mha softmax

# Custom optimization iterations
python3 test_suite/run_suite.py --max-iterations 5
```

## Metrics

- `discovery_success` - Kernel analysis completed
- `test_generation_success` - Test cases created
- `benchmark_baseline_created` - Baseline metrics saved
- `optimization_completed` - OpenEvolve optimization finished
- `speedup_ratio` - Performance improvement (if measured)

## Results

Results are saved to `test_suite/results/` (gitignored).
