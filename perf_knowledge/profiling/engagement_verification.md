---
title: profiling — engagement verification (proving your kernel/config is actually used)
kind: technique
gens: [gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
---

# Engagement verification: is your kernel/config even running?

## TL;DR
The most common reason a "tuned" kernel shows no speedup is that **it never ran** — the live dispatch
took a different path. Before you trust *any* A/B, prove engagement: (1) **`AITER_LOG_TUNED_CONFIG=1`**
then `grep -c 'is tuned on cu_num' server.log` > 0 for aiter; (2) confirm the env-flag / config file is
actually read; (3) know that **PyTorch TunableOp / `HIPBLASLT_TUNING_FILE` get 0 engagement on
sglang/vllm/aiter** because aiter bypasses PyTorch dispatch; (4) for a custom kernel, use the
ncu-equivalent (rocprofv3 `--kernel-trace`) to see *your* kernel name in the dispatch list. No engagement
proof → the benchmark is meaningless.

## The aiter engagement proof (validated)
aiter's tuned GEMM is keyed by a per-shape DB. To prove a DB hit on the live server:
```bash
export AITER_LOG_TUNED_CONFIG=1          # log every DB hit
export AITER_LOG_MORE=1                   # show dispatch decisions
# run the warm server, then:
grep -c 'is tuned on cu_num' server.log   # must be > 0
```
Each hit logs `... is tuned on cu_num = N in <file>, libtype is <ck|flydsl|...>`. The perf_knowledge e2e run
recorded **246 `is tuned on cu_num` hits** alongside the **+2.23% e2e** win @ MI300X gfx942,
sglang 0.5.11/aiter, 2026-06-08 — engagement and speedup measured *together*
([`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md),
[`../backends/aiter/overview.md`](../backends/aiter/overview.md)).

**Zero engagement traps (real):**
- **Bias mismatch**: tuning synthesized `bias=true` shapes while live calls are `bias=false` → 100%
  lookup miss → 0 hits despite a populated DB. The DB key must match live calls exactly.
- **Wrong env / file path**: `AITER_CONFIG_GEMM_BF16` pointing at a stale/missing CSV → silent miss.
- Always verify with the `grep` count > 0 *before* believing a flat A/B means "no win".

## Why TunableOp shows 0 engagement on aiter/sglang/vllm
PyTorch **TunableOp** (and `HIPBLASLT_TUNING_FILE`) hook the **PyTorch GEMM dispatch**. But in
sglang/vllm with aiter, the live GEMM is dispatched **by aiter and bypasses PyTorch dispatch entirely**
→ TunableOp's tuned solutions are never consulted → **0 engagement**, no matter how good the CSV
([`../backends/rocblas_tunableop/tunableop.md`](../backends/rocblas_tunableop/tunableop.md),
[`../backends/rocblas_tunableop/when_wins.md`](../backends/rocblas_tunableop/when_wins.md)).
- Smoking gun: `PYTORCH_TUNABLEOP_VERBOSE=1` shows the CSV loaded, yet the tuned `Gemm_*` rows are never
  hit for your live shapes → the aiter-bypass. TunableOp only engages on paths that go through PyTorch.
- Corollary: the **only** lever that engages the live sglang/vllm GEMM is **aiter's per-shape DB**.

## Env-flag / config sanity checks (generic)
- Confirm the flag was *seen*: log it at startup, or `grep` the server log for the tuned-config load line.
- Confirm the **file exists and is non-empty** and matches the dtype/`cu_num` of the box.
- Confirm `cu_num` matches: a config tuned for 304-CU MI300X won't key-match a partitioned (CPX) device.

## ncu-equivalent: prove a custom kernel ran
There is no `ncu` on ROCm; the equivalents:
```bash
rocprofv3 --kernel-trace --stats -- ./app   # lists every dispatched kernel by name + time
```
Grep the dispatch list for *your* kernel's mangled name; if it's absent, a fallback/library kernel ran
instead. For per-kernel internals, profile that named dispatch with
[`rocprof_compute_workflow.md`](rocprof_compute_workflow.md) (`--dispatch K`). This is the "did my
kernel actually execute, and was it the one that took the time" check.

## The verification gate (use before every A/B)
1. **Engagement**: `grep -c 'is tuned on cu_num' server.log > 0` (aiter) **or** your kernel name appears
   in `rocprofv3 --kernel-trace` output.
2. **Then** the same-session 2-launch A/B, accept iff the delta clears the **0.5%** noise band over
   REPEATS=7 ([`benchmarking_methodology.md`](benchmarking_methodology.md)).
Order matters: a "no change" A/B with **0 engagement** tells you nothing about the kernel — only that it
never ran.

## Pitfalls
- Believing a flat A/B before checking engagement (the #1 wasted-day failure).
- Tuning/benchmarking via TunableOp on an aiter serving stack → guaranteed 0 engagement.
- `cu_num` / dtype / bias key mismatch silently zeroing out a populated DB.

## Sources
- `AITER_LOG_TUNED_CONFIG`, `is tuned on cu_num`, bias-mismatch 0-engagement, 246 hits + +2.23% e2e: perf_knowledge aiter cards (e2e run 2026-06-08).
- TunableOp / HIPBLASLT_TUNING_FILE 0-engagement-on-aiter (PyTorch-dispatch bypass): perf_knowledge rocblas_tunableop cards.
- `rocprofv3 --kernel-trace --stats` as ncu-equivalent dispatch check: ROCm Blogs profilers intro.
