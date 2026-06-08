# E2E Inference Optimization — Levers Above the Kernel

The headline metric for e2e is **serving throughput** (output tok/s at fixed ISL/OSL/concurrency),
secondarily latency (TTFT, TPOT). Unlike single-kernel geomean, e2e is dominated by **Amdahl**: only
a speedup on a kernel that is a large share of GPU time, multiplied by how often that path runs,
moves throughput. A 5x on a 2%-of-time kernel is invisible. Always reason in (pct_gpu_time ×
achievable_speedup).

## The two regimes (do not conflate them)
- **Prefill** — large M (= total prompt tokens in batch, e.g. 15362). Compute-bound: big GEMMs,
  prefill attention. Throughput-for-prefill ≈ raw FLOPs efficiency.
- **Decode** — small M (= running batch size, e.g. 64). Latency/memory-bound: skinny GEMMs, paged
  KV attention, per-step launch overhead, cuda-graph replay.
A kernel optimized for one regime may not help the other. The Profiler carries shapes so the
Architect can tell which regime a kernel serves; optimization may produce regime-specific variants.

## Lever tiers (highest ROI first for a fresh model)

### Tier 0 — Config / backend (Config Tuner, runs FIRST, no source edits)
Cheapest, biggest, and it reshapes the kernel landscape (so profile AFTER). Knobs:
- **Attention backend**: `--attention-backend {triton, aiter, ck, fa3, ...}`. Huge for attn-heavy.
- **GEMM backend / tuning**: aiter vs hipBLASLt; populate the hipBLASLt/Tensile tuning DB for the
  exact shapes (untuned GEMM falls back to a default solution — see the `aiter ... not found tuned
  config ... using default config` warnings; tuning these is often a free 1.1–1.4x on the GEMM).
- **Quantization**: fp8/int8 weights/kv (`--quantization`, `--kv-cache-dtype fp8`) when accuracy
  budget allows — the single biggest throughput lever for compute-bound prefill.
- **CUDA/HIP graph**: `--enable-cuda-graph` / graph batch sizes — kills decode launch overhead.
- **torch.compile**: `--enable-torch-compile` (fuses elementwise/norm chains).
- **chunked prefill / max-prefill-tokens / schedule**: balances prefill vs decode interleave.
- **TP/EP/DP and mem-fraction**: parallelism + KV cache budget (bigger KV → higher concurrency).
- **Speculative / MTP**: this model has MTP layers — enabling speculative decode can lift decode.
Sweep one axis at a time, measure throughput delta with a variance band, keep wins, re-profile.

### Tier 1 — Editable hot kernels (Kernel Extractor → recursive kernel squad)
For `triton`/`fused_custom`/`reduction_norm` kernels with meaningful pct_gpu_time: extract with real
shapes + recorded I/O oracle, optimize via the unchanged single-kernel workflow, compare backends
(triton/CK/HIP/asm) per the playbook, then overlay back and re-validate throughput.

### Tier 2 — Dispatch / host overhead (host_runtime specialist, or graph)
Many tiny elementwise/cast/copy kernels at high call counts → fuse (Lever 1 of geomean_levers) or
cover with a cuda-graph. Native layouts to drop transpose/contiguous passes.

## Amdahl stop rule
After each milestone, estimate remaining headroom = Σ over untouched editable kernels of
(pct_gpu_time × plausible_speedup_fraction). If the best remaining candidate can't plausibly move
end-to-end throughput by more than the measurement noise band (typically ~2–3%), STOP — further
kernel work won't show up at the e2e level even if the isolated speedup is real.

## Measurement discipline (e2e is noisy)
- Keep the server WARM across validations; never fold server-startup into the timed window.
- Run enough requests (≥ 5× concurrency) and repeat the bench ≥ 2–3×; report median + spread.
- Gate a kernel into e2e only when its isolated speedup is real AND Amdahl says it can move the
  needle. Accept an e2e change only if the throughput delta exceeds the measured noise band.
- Always check **output parity** (greedy/temp=0, fixed seed) vs baseline — a faster wrong server is
  a regression.
