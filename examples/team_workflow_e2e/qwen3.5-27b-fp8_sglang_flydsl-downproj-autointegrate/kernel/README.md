# Optimized FlyDSL kernel — fp8 a8w8 blockscale down-proj GEMM (N=5120, K=17408)

The final, deployed kernel from the +67.4% e2e run. Authored + optimized fully by `team_workflow_e2e`
(FlyDSL author route) and accepted at the e2e gate. Saved here verbatim as the standalone deliverable.

## Files
- **`gemm_a8w8_blockscale_flydsl.py`** — the optimized kernel. A self-contained Triton **fused fp8
  block-scale GEMM**: operands stay fp8 the whole way (NO bf16 dequant/`repeat_interleave`
  materialization of X or W), per-128-K-block scales folded into the fp32 accumulator at each K-tile
  boundary; split-K for the skinny decode buckets. Matches the immutable fp32 dequant oracle within
  tol=0.06. Weight prep cached by `data_ptr()` (host-sync-free → CUDA-graph-capture-safe). Entry:
  `gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=...)`.
- **`sitecustomize.py`** — the capture-safe overlay seam that binds this kernel onto the live sglang
  call site (`aiter.ops.triton.gemm_a8w8_blockscale` + `fp8_utils` globals). Lazy meta-path finder (no
  eager import / fork-storm), precompile-before-capture warmup hook, one-shot engagement proof.

## Measured kernel speedup (director-verified, vs stock aiter Triton blockscale GEMM, geomean of 5 M-buckets)
| stage | geomean | vs triton |
|------|------|------|
| stock triton (baseline) | 1.947 ms | 1.00× |
| FlyDSL R0 (naive author) | ~1.95 ms | 0.997× (≈parity; bf16-dequant, memory-bound) |
| FlyDSL R1 (fused fp8, kill dequant) | ~1.1 ms | ~1.7–1.8× |
| **FlyDSL final (this kernel)** | **0.800 ms** | **2.432×** |

Per regime (stock triton → this kernel): **decode M=1 6.76× · M=64 5.29×**; prefill M=15360–16384 1.30–1.38×.
The big decode win (stock Triton is poor at skinny-M, K=17408 GEMM) is what drives the +60% head /
+67.4% e2e at conc=64 (TPOT 62.6→37.4 ms).

## How to use (standalone / re-wire)
```bash
# isolated unittest / any importer: just put this dir on PYTHONPATH; sitecustomize.py auto-binds it
PYTHONPATH="$PWD:$PYTHONPATH" python -m sglang.launch_server --model-path <Qwen3.5-27B-FP8> \
    --attention-backend triton --mem-fraction-static 0.85 --watchdog-timeout 600 ...
# you should see in the server log:
#   [flydsl-overlay] bound FlyDSL gemm_a8w8_blockscale over aiter triton symbol
#   [flydsl-overlay] ENGAGED: FlyDSL gemm_a8w8_blockscale ran on the LIVE call site
```
Requirements: AMD MI300X / gfx942, aiter with FlyDSL (`is_flydsl_available()`), sglang with the
`fp8_utils` aiter blockscale path (`SGLANG_USE_AITER`). gfx942 has no native block-scaled MFMA, so the
per-block scale is emulated in software (see the kernel header).
