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

## ⚠️ De-inflate busy-wait collectives BEFORE you Amdahl-rank (do this every multi-rank trace)
A synchronizing collective — vLLM custom all-reduce (`cross_device_reduce*`), NCCL/RCCL
(`ncclDevKernel*`, `*all_reduce*`, `*all_gather*`, `*reduce_scatter*`), barriers — **busy-waits on the
GPU for peer ranks to arrive, and rocprofv3 counts that idle spin as kernel GPU time.** So its summed
`total_ms`/`pct_gpu_time` in the Top-N is NOT its optimizable cost — it is mostly peer-wait bubble.
Real MiniMax-M3 TP=4 trace: `cross_device_reduce_1stage` had a *median* call of ~12µs but a P99 of
~12ms (mean/median skew ≈ **18×**), inflating it to **~51%** of GPU when its intrinsic transfer cost is
**~8%**. Left raw, this **buries the editable GEMM heads** (MoE + dense mxfp8 GEMM) and sends the
Architect chasing a ~50% "comm" target that is mostly synchronization slack at fixed TP.

**This is a JUDGMENT recipe, not a hard pipeline step — apply it with graceful degradation, never let it
crash or block the Top-N.** For each collective-class kernel in the top entries with `pct_gpu_time` ≳ 5%:

1. **Decide if it is actually spin-inflated** (a plain transfer is fine; a spinning barrier is not).
   Best-effort: sample its per-call durations from the rocprofv3 per-call trace and compare mean vs
   median. A healthy compute kernel has mean/median ≈ 1; a spin-inflated collective is ≫ 3 (M3 was 18).
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
   if len(ds)>50:
       m=st.median(ds); mean=sum(ds)/len(ds)
       print(f"median={m/1000:.1f}us mean={mean/1000:.1f}us skew={mean/m:.1f}x n={len(ds)}")
   PY
   ```
2. **If skew > ~3 → de-inflate.** Report a robust **effective** cost = median-cap winsorize: clip each
   call at ~10×median, then sum (≈ `median × calls` is a fine shortcut). Put the EFFECTIVE %gpu in
   `pct_gpu_time` so the Architect Amdahl-ranks on it, and **keep the raw** in a `raw_pct_gpu_time` /
   `notes` field + say it was spin-deinflated Nx. Nothing hidden.
3. **Route it as a CONFIG lever, not a kernel rewrite.** The clipped time is comm-overlap / load-
   imbalance bubble → Config Tuner (AR backend/quant, comm-compute overlap, NCCL channels), never the
   kernel squad. The editable GEMM heads it was hiding are the real source-rewrite targets.
4. **GRACEFUL DEGRADATION (the point of doing this as a recipe, not rigid code):** if the per-call
   trace is missing / a different rocprofv3 schema / too large to sample — do NOT fail. Fall back to a
   *qualitative* flag: when a collective-class kernel shows a high `avg_us` AND a huge `calls` count AND
   tops the list, note in `notes`/`opt_hint` that its %gpu is "likely spin-inflated — discount in Amdahl
   routing; comm-config lever, not a rewrite" and let the Architect treat the editable heads beneath it
   as the real targets. A qualitative flag is better than a crashed profile or a wrong 50% target.

## Reliability notes
- Profile with the SAME ISL/OSL/concurrency as the throughput benchmark, after warmup.
- Use a short, bounded profiling window (`--profile-num-steps`) so traces stay parseable.
- `total_gpu_time_ms` is summed kernel duration in the captured window, not wall-clock — use it for
  RELATIVE ranking (%gpu), not as the throughput number.
