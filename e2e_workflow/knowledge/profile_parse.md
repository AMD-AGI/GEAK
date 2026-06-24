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

## ⚠️ Per-call distribution sanity — fix misleading `total_ms`/`pct_gpu_time` BEFORE you Amdahl-rank
The Top-N's `total_ms`/`pct_gpu_time` is a **sum of per-call durations**. For several kinds of kernel
that sum is NOT the steady-state optimizable cost you want to route on — so check the per-call
*distribution* of any top kernel before trusting its `pct`. This is **not just communication kernels**;
the same sampling + judgment applies to any suspicious entry.

**When to look:** any top-N kernel you're about to route on whose `avg_us` is surprising for its class,
or whose `pct_gpu_time` is large, or that you suspect from its name/role. **This is a JUDGMENT recipe,
not a hard pipeline step — apply with graceful degradation, never let it crash or block the Top-N.**

**How to look (one cheap sample of the per-call trace; works for ANY kernel name — pass its core token):**
```bash
# one rank's per-call trace is enough (distribution is rank-invariant); robust to huge files.
python3 - "$ROCPROF_DIR" 'cross_device_reduce_1stage' <<'PY' 2>/dev/null || true
import csv,glob,os,sys,statistics as st
d,core=sys.argv[1],sys.argv[2]
f=sorted(glob.glob(os.path.join(d,'**','*kernel_trace*.csv'),recursive=True))
if not f: sys.exit()
ds=[]
with open(f[0],newline='') as fh:
    r=csv.reader(fh); h=next(r); kn=h.index('Kernel_Name'); s=h.index('Start_Timestamp'); e=h.index('End_Timestamp')
    for row in r:
        if len(row)>e and core in row[kn]:
            try: ds.append(int(row[e])-int(row[s]))
            except: pass
if len(ds)>20:
    ds.sort(); n=len(ds); q=lambda p: ds[min(n-1,int(n*p))]
    m=ds[n//2]; mean=sum(ds)/n
    print(f"n={n} median={m/1000:.1f}us mean={mean/1000:.1f}us skew={mean/m:.1f}x "
          f"p10={q(.10)/1000:.1f} p90={q(.90)/1000:.1f} p99={q(.99)/1000:.1f} max={ds[-1]/1000:.1f}us")
PY
```
Read the shape (skew = mean/median; one-tail vs two-cluster vs uniform) and diagnose the cause —
different causes get different handling:

1. **Busy-wait / synchronization** (collective all-reduce/NCCL/RCCL `cross_device_reduce*`,
   `ncclDevKernel*`, `*all_reduce*`, `*all_gather*`, `*reduce_scatter*`, barriers — or ANY kernel whose
   job is to *wait* on peers/host). Heavy right tail, skew ≫ 3 (M3 `cross_device_reduce`: median 12µs,
   P99 12ms, skew **18×** → shown as ~51% GPU when intrinsic transfer is ~8%). The tail is peer-wait
   spin, not work.
   **🔴 DETERMINISTIC RULE — do NOT use judgment here:** ANY kernel whose name matches a collective/
   barrier (`cross_device_reduce*`, `ncclDevKernel*`, `*all_reduce*`, `*all_gather*`, `*reduce_scatter*`,
   `*barrier*`, `custom_all_reduce`, `one/two-stage reduce`) is busy-wait BY DEFINITION and **MUST be
   de-inflated** whenever skew (mean/median) > 3 — regardless of how "steady" the call rate looks. NEVER
   tag a collective as category-4 "healthy/steady" or leave its raw sum in the Amdahl total. A common
   failure is de-inflating `*_2stage` but leaving `*_1stage` (or vice-versa) at its raw % — de-inflate
   EVERY collective that crosses the skew bar, then recompute the table (step below).
   → report a robust **effective** `pct_gpu_time` = median-cap winsorize (clip each call
   at ~10×median then sum; `median×calls` is a fine shortcut), **keep the raw** in
   `raw_pct_gpu_time`/`notes`, and route it as a comm-overlap/load-imbalance **CONFIG** lever (AR
   backend/quant, comm-compute overlap, NCCL channels) — never a kernel rewrite.

2. **One-time warmup / JIT / autotune / graph-capture outliers** (Triton/Inductor JIT-compile +
   autotune on first launch; HIP-graph capture; first-touch allocation). Signature: a *handful* of giant
   first-calls, the rest tight — high skew but the tail is a few calls, not a fixed fraction. → use the
   **steady-state** estimate (median×calls, or drop the first-K outliers) for the optimization ranking;
   the giant first-calls are real but ONE-TIME, not what a rewrite/tune changes. Note it; keep raw.

3. **Bimodal = two regimes under one name** (prefill large-M + decode small-M; or context-len buckets).
   Signature: **two clusters**, not a single tail (e.g. p10 and p90 differ by ~10×+ with a gap). This is
   REAL compute in both — do **NOT** de-inflate. → instead **split** it into per-regime entries (carry
   each regime's `shapes`) so the Extractor builds regime-specific unittests and the **decode** regime
   (steady-state, e2e-critical) is ranked on its own rather than averaged away. (See the "Same name,
   many shapes" note above.)

4. **Healthy / honest** — skew ≈ 1, uniform. Use the summed `pct_gpu_time` as-is.

**🔴 RECOMPUTE THE WHOLE TABLE after any de-inflation — do NOT discount one kernel in isolation.**
De-inflating a kernel shrinks `total_gpu_time`, so EVERY kernel's `pct_gpu_time` must be re-expressed
against the NEW (de-inflated) total. The editable heads MUST rise correspondingly. Worked example (M3
TP=4): comm `cross_device_reduce` 51%→~8% means the total drops ~52% → MoE GEMM **16%→~31%** and dense
GEMM **11%→~21%**. A Top-N that shows the collective discounted to ~1.5% but leaves the GEMMs at their
raw 16%/11% is **INCONSISTENT and WRONG** — the displayed `%gpu` no longer sums against one total, and
the Architect will under-rank the real targets (a "+8% ceiling" instead of the true ~+17%). Concretely:
`effective_total = Σ effective_ms` (de-inflated where flagged, raw elsewhere); then every row's
`pct_gpu_time = 100 * effective_ms / effective_total`. The editable GEMM heads becoming the clear #1/#2
is the SIGNAL that you did it right.

**Always KEEP the raw value** (annotate, don't overwrite silently). **GRACEFUL DEGRADATION (the point of
a recipe over rigid code):** if the per-call trace is missing / a renamed rocprofv3 schema / too large to
sample — do NOT fail. Fall back to a *qualitative* flag from the Top-N alone, e.g. "collective with high
`avg_us` + huge `calls` → likely spin-inflated, discount in Amdahl routing; comm-config lever"; "Triton
kernel with one giant first-call → likely JIT/autotune warmup, discount that call"; "same name at both
large-M and small-M `avg` → split prefill vs decode". A qualitative flag beats a crashed profile or a
wrong routing target.

## Reliability notes
- Profile with the SAME ISL/OSL/concurrency as the throughput benchmark, after warmup.
- Use a short, bounded profiling window (`--profile-num-steps`) so traces stay parseable.
- `total_gpu_time_ms` is summed kernel duration in the captured window, not wall-clock — use it for
  RELATIVE ranking (%gpu), not as the throughput number.
