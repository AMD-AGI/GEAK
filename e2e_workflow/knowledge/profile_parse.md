# Profile Parsing — the Standardized Top-N Contract

The Profile phase MUST produce ONE canonical artifact so every downstream agent reads the
bottleneck identically. The tool is `scripts/parse_profile.py`; this file is its contract.

## How to produce it
```bash
# torch/sglang profiler trace (gives op names + shapes):
python3 $WF_DIR/scripts/parse_profile.py --torch-trace <trace.json.gz> --top 25 --out $EVAL_DIR/profile_topN
# rocprofv3 kernel-trace (authoritative HW durations), or BOTH merged:
python3 $WF_DIR/scripts/parse_profile.py --rocprof-dir <dir> --torch-trace <trace.json.gz> \
        --top 25 --out $EVAL_DIR/profile_topN
```
Writes `profile_topN.json` (canonical schema) + `profile_topN.md` (human table). When both sources
are given, HW durations come from rocprofv3 and shapes are enriched from the torch trace.

## Canonical schema (profile_topN.json)
```
{ source, total_gpu_time_ms, num_kernel_launches, num_distinct_kernels,
  top_kernels: [ { rank, name, short_name, calls, total_ms, avg_us, pct_gpu_time,
                   shapes[], dtypes[], classification, backend_guess, editable, opt_hint } ] }
```

## The classification field (this is the triage signal the Architect routes on)
- `library_gemm` — hipBLASLt/Tensile/rocBLAS GEMM. **Not source-editable.** Route to Config Tuner
  (backend/env/heuristics swap: aiter vs hipBLASLt vs CK GEMM, tuning DB) — NOT to the kernel squad.
- `library_attn` — CK/AITER/FlashAttn paged attention. Route to Config Tuner (`--attention-backend`
  swap, per-shape backend). Source-edit only if it resolves to a Triton attention.
- `triton` / `fused_custom` — **editable.** Route to Kernel Extractor → kernel squad. This is where
  the recursive single-kernel kernel_workflow runs.
- `elementwise_overhead` — fill/cast/activation/copy. Route to host_runtime fusion (Lever 1) or
  config (e.g. enable fused activation). Often cheap per-call but high call count.
- `reduction_norm` — rmsnorm/rope/softmax. Editable (often Triton); candidate for fusion.
- `memory` — memcpy/memset. Reduce via native layouts.
- `other` — inspect source to route (the Profiler should try to resolve these before finishing).

## Reading the result (how the Architect should think)
1. **Amdahl first.** A kernel at 52% gpu time with a plausible 1.3x is worth far more than a 5x on a
   2% kernel. Rank candidates by `pct_gpu_time × achievable_speedup × editable`.
2. **GEMM/attn usually dominate** prefill (big M). They are library calls → the highest-ROI early
   move is the Config Tuner sweep (backend/quant/tuning), NOT a source rewrite.
3. **Editable Triton/custom kernels** (mamba/gated-delta, norms, activations) are where the kernel
   squad earns its keep. Carry their `shapes` into the Extractor so the unittest replays real shapes.
4. **Same name, many shapes** = one kernel serving both prefill (large M, e.g. 15362×…) and decode
   (small M, e.g. 1024×… or batch×…). These are different regimes → the Extractor may build separate
   unittests and the squad may produce regime-specific variants.
5. **High call-count tiny kernels** (e.g. elementwise at 1000s of calls) signal dispatch overhead →
   host_runtime fusion / cuda-graph.

## Reliability notes
- Profile with the SAME ISL/OSL/concurrency as the throughput benchmark, after warmup.
- Use a short, bounded profiling window (`--profile-num-steps`) so traces stay parseable.
- `total_gpu_time_ms` is summed kernel duration in the captured window, not wall-clock — use it for
  RELATIVE ranking (%gpu), not as the throughput number.
