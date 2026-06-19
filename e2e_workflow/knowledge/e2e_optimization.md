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
Sweep one axis at a time, measure throughput delta with a variance band, keep wins, re-profile.
**Config wins STACK and compound** — accept them incrementally, and then carry the **accepted config
stack as the REF leg** when gating a kernel, so the kernel's delta is isolated on top of the real serving
config (not on top of the bare baseline). A lossy config (e.g. fp8 KV cache) must clear the parity gate
before it joins the stack.

### Tier 1 — Editable hot kernels (Kernel Extractor → recursive kernel squad)
For `triton`/`fused_custom`/`reduction_norm` kernels with meaningful pct_gpu_time: extract with real
shapes + recorded I/O oracle, optimize via the unchanged single-kernel workflow, compare backends
(triton/CK/HIP/asm) per the playbook, then overlay back and re-validate throughput.

### Tier 2 — Dispatch / host overhead (host_runtime specialist, or graph)
Many tiny elementwise/cast/copy kernels at high call counts → fuse (Lever 1 of geomean_levers) or
cover with a cuda-graph. Native layouts to drop transpose/contiguous passes.

### Integrating an authored / JIT kernel into the live (graph-captured) path (general)
Getting an isolated-win kernel to actually move e2e — backend-agnostic rules:
- **Rebind via a passive/lazy seam.** Don't eager-import a heavy kernel lib at interpreter startup —
  the serving stack spawns many short-lived helpers (compile workers, mp children, trackers) that would
  each pay the init cost and can pile up into a process/enumerator storm. Install the dispatcher only
  when the target module is first imported by the real model worker; skip helper subprocesses.
- **JIT/eager-compiling code cannot live inside a CUDA-graph capture region.** Make the kernel
  compile-once / shape-agnostic and **pre-warm it during the warmup forward**, so the captured region
  only *launches* the cached kernel (no eager compile under capture → no crash, and the kernel ends up
  inside the graph). Raise the serving watchdog if first-call compile exceeds the startup budget.
- **Swap only where it wins for the throughput-dominant regime.** Steady-state serving throughput is
  **decode-dominated**, so a prefill-only kernel swap won't move output tok/s — confirm the op is on the
  decode hot path before integrating, and route per-shape (engage only the shapes/regimes that win).
- **Per-call preprocessing caches must cover the full working set** (≥ the number of distinct reuse keys,
  e.g. transformer layers). An undersized cache thrashes — re-doing the preprocessing every step — and
  can REGRESS e2e badly even when the kernel itself is faster.
- Prove **engagement** on the live path (log/probe that the new kernel actually ran), then gate on the
  same-session isolated A/B above. Profile again after it lands — the bottleneck shifts, exposing the
  next lever (e.g. a prologue/epilogue that can now be fused).

## Amdahl stop rule
After each milestone, estimate remaining headroom = Σ over untouched editable kernels of
(pct_gpu_time × plausible_speedup_fraction). If the best remaining candidate can't plausibly move
end-to-end throughput by more than the measurement noise band (typically ~2–3%), STOP — further
kernel work won't show up at the e2e level even if the isolated speedup is real. In practice the small
editable kernels (each a few % of GPU) usually land in-band or negative; spend the budget on the head,
not the tail.

## Finishing a long or interrupted run
Accepted work is durable on disk independently of the orchestrator: the config wins (`config/`), each
gate-accepted kernel overlay + its `integrate_result.json`, and the baseline all persist. If a long run
is interrupted (crash/timeout) after the head wins have landed, **do not resume to grind the remaining
low-pct_gpu_time milestone kernels** — finish with a direct **same-session Validate of the accepted
stack vs the true baseline** (the run-wide serving config TP=SERVING_TP GPU=SERVING_GPU, a couple of reps, + greedy parity). That recovers the official
number quickly without re-doing hours of low-value work.

## Measurement discipline (e2e is noisy)
- Keep the server WARM across validations; never fold server-startup into the timed window.
- Run enough requests (≥ 5× concurrency) and repeat the bench ≥ 2–3×; report median + spread.
- Gate a kernel into e2e only when its isolated speedup is real AND Amdahl says it can move the
  needle. Accept an e2e change only if the throughput delta exceeds the measured noise band.
- Always check **output parity** (greedy/temp=0, fixed seed) vs baseline — a faster wrong server is
  a regression.
- **Isolation (do this or the delta is fiction).** Measure baseline vs candidate **sequentially on the
  same serving config (TP=SERVING_TP GPU=SERVING_GPU), same session** (tear one server fully down before
  launching the next). Do NOT run two
  servers concurrently on one node for the headline ratio: shared-resource contention drags the baseline
  leg down and **inflates the ratio into a false win**. Trust a delta only when run spreads are tight and
  the two legs' run ranges are **non-overlapping**.
- **Harness pitfalls that silently corrupt a leg:** (a) never name an eval-dir output folder after an
  importable package (`triton/`, `flydsl/`, `aiter/`, …) — such a dir on the bench process's CWD
  **shadows** the real package (`import X`→empty namespace) and the bench crashes; looks like a kernel
  failure but is a naming bug. (b) Give each leg a **fresh JIT/compile cache dir** (e.g. `TRITON_CACHE_DIR`)
  to avoid cross-leg cache races. (c) Raise the serving **watchdog timeout** when a candidate JIT-compiles
  on first call (a cold kernel compile can exceed the default startup budget).
