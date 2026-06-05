# Profiling & Bottleneck Triage — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target MI300X (gfx942, CDNA3); gfx950 (CDNA4) notes inline. This file is the measurement discipline: how to collect traces (`rocprofv3`), build a per-kernel roofline and counter profile (`rocprof-compute`, formerly Omniperf), find the Top-N kernels by GPU time, compute arithmetic intensity, classify memory-bound vs compute-bound, and apply Amdahl reasoning to decide what to optimize. **Profile before you optimize — every other file in this directory assumes you have a profile.**

---

## The triage loop

```
1. TRACE      rocprofv3 --sys-trace ... → Perfetto/CSV → Top-N kernels by gpu_time
2. RANK       sort by total gpu_time (count × duration). Amdahl: optimize the biggest %.
3. ROOFLINE   rocprof-compute --roof-only → is the hot kernel mem-bound or compute-bound?
4. DRILL      rocprof-compute analyze → VALU/MFMA util, LDS/L2 hit, mem BW %, occupancy, spills
5. FIX        apply the matching technique (fusion / tiling / split-K / quant / occupancy)
6. RE-PROFILE confirm the kernel moved (AI off the HBM roof, util up, kernel left the Top-N)
```

---

## 1. Tracing with rocprofv3 (rocprofiler-sdk)

`rocprofv3` (in `/opt/rocm/bin`) is the current CLI. Unlike old `rocprof`, **it traces nothing by default** — request each trace type explicitly.

### Whole-application system trace → Perfetto

```bash
rocprofv3 --sys-trace --output-format pftrace -d run1 -o app -- python serve.py
# open run1/..._app.pftrace in https://ui.perfetto.dev
```
`--sys-trace` = kernel + HIP API + memcpy + HSA. Default output format is **rocpd** (SQLite); convert later via the `rocpd` module to CSV / OTF2 / PFTrace.

### Targeted kernel trace + counters, multiple formats at once

```bash
rocprofv3 --kernel-trace \
  --pmc GPU_UTIL MfmaUtil BANDWIDTH_EA \
  --output-format csv pftrace -d run1 -o kern -- python bench.py
# emits both .csv (for Top-N scripting) and .pftrace (for visual), plus *_agent_info.csv
```

| Flag | Use |
|---|---|
| `--kernel-trace` | per-kernel start/stop/duration |
| `--hip-trace` / `--hsa-trace` / `--memory-copy-trace` | API / runtime / copies |
| `--sys-trace` | all of the above |
| `--pmc <counters>` | hardware perf counters |
| `--output-format <csv\|json\|pftrace\|otf2>` | one or many (`csv pftrace`) |
| `-d <dir>` `-o <name>` | output dir / prefix |
| `--kernel-include-regex` | filter to kernels of interest |

> Large traces (>10 GB) → use `otf2` (Vampir). Counter tracks appear per-agent in Perfetto with `pftrace`.

### Advanced Thread Trace (ATT) — instruction-level

```bash
rocprofv3 --att --att-simd-select 0x0 \
  --kernel-include-regex "gemm.*" -- python bench.py
# decode via rocprofv3; visualize in ROCprof Compute Viewer
```
ATT traces **one kernel instance** by default (use `--kernel-iteration-range` for more; `--att-consecutive-kernels` to merge). Stats CSV columns: **Hitcount** (instr executions over traced waves), **Latency** (stall+issue cycles, gfx9), **Stall** (cycles the pipe couldn't issue). Use ATT to find the exact stalling instructions once you've localized the hot kernel. Errors `INVALID_SHADER_DATA` / `Agent not supported` mean the HW/data prerequisites aren't met.

---

## 2. Top-N kernels by GPU time (the ranking step)

From the rocpd DB or the kernel-trace CSV, aggregate **total GPU time = count × mean duration** per kernel name and sort descending:

```bash
# from a rocpd SQLite DB:
rocpd --input run1/*.db --output-format csv --top   # or query kernel-dispatch table
# or from CSV:
python - <<'PY'
import pandas as pd
df = pd.read_csv("run1/kern_kernel_trace.csv")
df["dur"] = df["End_Timestamp"] - df["Start_Timestamp"]
g = df.groupby("Kernel_Name")["dur"].agg(["count","sum","mean"]).sort_values("sum",ascending=False)
g["pct"] = 100*g["sum"]/g["sum"].sum()
print(g.head(15))
PY
```

This `pct_gpu_time` column drives **Amdahl**: a kernel at 4% of GPU time, even made infinitely fast, can return at most 4% end-to-end. Always optimize the top of this list first.

---

## 3. rocprof-compute (Omniperf) — roofline & per-kernel metrics

`rocprof-compute` (formerly Omniperf; built into ROCm 7+) runs two stages: counter collection + roofline ceilings. SoC-specific results land in an `MI300X/` target dir.

### Generate a roofline

```bash
# full profile (counters + roofline):
rocprof-compute profile -n myrun -- python bench.py
# roofline only (fast, emits a standalone PDF plot):
rocprof-compute profile -n myrun --roof-only -- python bench.py
# (--no-roof skips the roofline stage)
```

The roofline benchmark measures ceilings: peak **MFMA** IOPs/FLOPs, **HBM**, **L2**, **L1**, **LDS** bandwidths, and peak FP32 FLOPs. Each kernel is plotted as a point at (arithmetic intensity, achieved throughput).

### Analyze counters per kernel

```bash
rocprof-compute analyze -p workloads/myrun/MI300X/        # text report
rocprof-compute analyze -p workloads/myrun/MI300X/ --gui  # interactive
```

---

## 4. The roofline model & arithmetic intensity

**Arithmetic intensity (AI)** = total FLOPs / total bytes moved (FLOP/byte) — the x-axis.

```
            ┌──────────── compute roof (peak MFMA FLOP/s) ─────────
 perf       │            ╱
(FLOP/s)    │          ╱  ← memory roof: slope = peak BW (FLOP/s = BW × AI)
            │        ╱
            │      ╱
            └────╱──────────────────────────────────────────────────
                 AI_ridge          arithmetic intensity (FLOP/byte) →

   AI < ridge  → MEMORY-BOUND (point sits on the BW slope)
   AI > ridge  → COMPUTE-BOUND (point sits under the flat roof)
```

**GEMM AI:** `2·M·N·K / (bytes(A)+bytes(B)+bytes(C))`. For `M=N=K=4096` fp16: `2·4096³ / (3·4096²·2) ≈ 1365` FLOP/byte → deep in compute-bound territory (large square GEMM should be compute-bound; if it's not, you have a tiling/occupancy bug).

**FlashAttention AI:** account for Q,K,V loads + O store + softmax FLOPs; FA tiles cluster around ~10–15 FLOP/byte → near the ridge, often memory-bound at decode.

**Ridge point (MI300X):** ~peak FLOP/s ÷ ~5.3 TB/s. Anything below that AI is fundamentally bandwidth-limited no matter how good the kernel.

> Counters feeding AI are precision-specific: `SQ_INSTS_VALU_{ADD,MUL,FMA,TRANS}_F16/BF16/F32` + `SQ_INSTS_VALU_MFMA_MOPS_*` for FLOPs, and TCC (L2) traffic counters for bytes. `rocprof-compute` derives AI automatically.

---

## 5. Memory-bound vs compute-bound: the key metrics

| Metric (rocprof-compute) | Meaning | Compute-bound looks like | Memory-bound looks like |
|---|---|---|---|
| **MFMA_util** / `SQ_VALU_MFMA_BUSY_CYCLES` | matrix-engine busy | high (→peak) | low |
| **VALU_util** | vector-ALU busy | high if VALU-heavy | low–mid |
| **HBM/mem BW %** | fraction of 5.3 TB/s used | low | high (→roof) |
| **L1 / L2 (TCC) hit rate** | cache reuse | high | low (re-reading HBM) |
| **LDS_util** + bank conflicts | LDS pressure / conflicts | — | conflicts → stalls |
| **Occupancy (waves/CU)** | latency hiding | "enough" | often low |
| **VGPR/AGPR spill / scratch** | register overflow | 0 (good) | >0 = redesign |

**Decision:** plot on the roofline. **On the BW slope** → memory-bound → cut traffic (fuse memory-bound neighbors, quantize, improve coalescing/L2 reuse). **Under the flat roof but below it** → compute-bound but under-utilizing MFMA → raise occupancy, fix pipelining/`num_stages`, eliminate stalls, use `mfma_16x16`.

> Real MI300X example (quantized LLaMA-3.3-70B prefill, rocprof-compute): all kernels — GEMMs, quantized GEMMs, FlashAttention — sat **well below** the MFMA/VALU ceilings; the FA tile (~15 FLOP/byte) reached only ~10 TFLOP/s, most kernels at ~1–3 FLOP/byte and ~1–7 TFLOP/s. Conclusion: **memory/utilization-bound, not compute-bound** → the fix is reducing traffic and raising MFMA utilization, not more FLOPs.

---

## 6. Amdahl-driven prioritization

```
max_end_to_end_speedup_from_fixing_kernel_K
        = 1 / ( (1 - pct_gpu_time_K) + pct_gpu_time_K / kernel_speedup_K )

  pct_gpu_time_K = K's share of total GPU time (from Top-N, §2)
  kernel_speedup_K = how much faster you can make K (from roofline headroom, §4)
```

- A kernel at **40%** of GPU time, made 4× faster → end-to-end `1/(0.6 + 0.4/4) = 1.43×`.
- A kernel at **3%** of GPU time, even made ∞ fast → at most `1/0.97 = 1.03×`. Skip it.
- **Roofline bounds `kernel_speedup_K`:** a memory-bound kernel already on the HBM roof has ~0 compute headroom — its only speedup is *less traffic* (fusion/quant), not faster math.

Pick the kernel maximizing `pct_gpu_time × achievable_speedup`. Re-rank after each fix (the bottleneck moves).

---

## 7. Worked triage example

```
1. rocprofv3 --sys-trace -d r1 -o m -- python serve.py
   Top-N:  fused_moe 31% | attn_decode 22% | rmsnorm 9% | o_proj_gemm 8% | ...
2. Amdahl: fused_moe (31%) is the target.
3. rocprof-compute profile -n moe --roof-only -- python moe_bench.py
   → fused_moe point sits on the HBM BW slope (AI ~2 FLOP/byte) → MEMORY-BOUND.
4. rocprof-compute analyze -p workloads/moe/MI300X/
   → MFMA_util 18%, mem BW 78% of peak, L2 hit 41%, 0 spills, occupancy 2 waves/CU.
   Diagnosis: re-reading expert weights from HBM; permute/combine buffers add traffic.
5. Fix: fuse permute/combine into grouped GEMM (eliminate buffers), per-expert tuned configs,
   FP8 expert weights (halve weight traffic), GROUP_M swizzle for L2 reuse, raise waves_per_eu.
6. Re-profile: fused_moe AI ↑, mem BW % ↓, MFMA_util ↑, kernel drops to 19% → re-rank, next.
```

---

## 8. Other tools & env

- **omnitrace / rocprof-systems** — full system trace (CPU+GPU+threads) when the bottleneck might be host-side (launch overhead, Python, dataloader, RCCL gaps). Use when GPU is idle in the Perfetto timeline.
- **rocprof-compute-viewer** — visualizes ATT and counter data; exposes derived `MFMA_util`, `VALU_util`, `LDS_util`.
- **roctx markers** — annotate code regions (`roctxRangePush/Pop`) so traces map to logical phases (prefill vs decode vs collective). `HIPBLASLT_ENABLE_MARKER=1` adds hipBLASLt markers.
- Useful env: `HIP_FORCE_DEV_KERNARG=1` (lower launch overhead), `AMD_LOG_LEVEL` (runtime debug).

---

## Profiling checklist

```
[ ] rocprofv3 --sys-trace → Top-N kernels by total gpu_time (count × dur)
[ ] Pick target by Amdahl (pct_gpu_time × achievable_speedup), not by single-call duration
[ ] rocprof-compute --roof-only → classify target: memory-bound (on BW slope) vs compute-bound
[ ] rocprof-compute analyze → MFMA_util, VALU_util, mem BW %, L1/L2 hit, LDS conflicts, occupancy, spills
[ ] Compute AI = FLOPs / bytes; compare to MI300X ridge (~peak FLOP/s ÷ 5.3 TB/s)
[ ] Confirm 0 register spills (scratch == 0) before chasing compute
[ ] Memory-bound → cut traffic (fuse / quant / coalesce / L2 reuse); Compute-bound → occupancy / num_stages / mfma_16x16 / stalls (ATT)
[ ] Re-profile after each change; verify the kernel moved on the roofline and re-rank Top-N
[ ] Record ROCm version + counters with results (counters/indices change across versions)
```

---

## Sources

- Using rocprofv3 (rocprofiler-sdk: kernel/sys trace, pftrace, output formats): <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
- Using thread trace / ATT (rocprofiler-sdk): <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html>
- ROCm Compute Profiler — performance model & roofline (MFMA/HBM/L2/L1/LDS ceilings, AI counters): <https://rocm.github.io/rocprofiler-compute/performance_model.html>
- rocprof-compute profile mode (`--roof-only`, `--no-roof`, MI300X target dir): <https://rocm.docs.amd.com/projects/omniperf/en/docs-6.3.0/how-to/profile/mode.html>
- rocprof-compute pipeline descriptions (VALU/MFMA/LDS definitions, SQ_VALU_MFMA_BUSY_CYCLES): <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html>
- rocprof-compute-viewer (MFMA_util / VALU_util / LDS_util derived counters): <https://github.com/ROCm/rocprof-compute-viewer>
- ROCm profiling of quantized LLMs on MI300X (memory-bound roofline finding): <https://medium.com/@afsara.benazir/what-rocm-profiling-revealed-about-quantized-llms-on-amds-fastest-gpu-c0edfab9624f>
- AMD MI300X workload optimization (profiling workflow + tuning context): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
