# Profiling Analysis Guide

## Reading the raw profiler dump (START HERE — the script does NOT parse for you)

`scripts/profile_kernel.sh` is intentionally thin: it warms up, picks the best available profiler, runs
it, and dumps the **raw, unparsed** output. YOU (the profile engineer) extract the metrics and classify
the bottleneck. The script never greps for version-specific section names, so it stays portable — which
means the parsing responsibility is yours, and you must adapt to whichever profiler actually ran.

1. **Entry point**: read `<profile_output_dir>/profile_report.txt`. Its tail prints `Profiler used: <name>`
   and an `Artifacts:` list. Branch your parsing on which profiler produced it (the four cases below).
   Native artifacts (e.g. rocprofv3 CSVs, the `*_profile_raw.log`) are also left in the dir for deeper
   parsing if `profile_report.txt` is not enough.
2. **Always extract the dispatch count** (kernels launched per call) regardless of profiler — it is the
   key geomean/overhead signal (see `geomean_levers.md`). How to find it differs per profiler (below).
3. **Degrade gracefully**: if a metric/field is absent in the available profiler, say so explicitly in
   your summary and classify from whatever IS present (at minimum: per-case latency + dispatch count).
   Never block on a field that this toolchain doesn't emit.

### Per-profiler extraction

- **`rocprof-compute` / `omniperf`** (richest): `profile_report.txt` holds the full `analyze` text —
  Speed-of-Light, Wavefront, Compute Pipeline, cache hierarchy. Parse it with the section tables further
  down this guide. Dispatch count = number of distinct kernel rows in the kernel/dispatch breakdown.
- **`rocprofv3`** (modern, trace-based): the report embeds the run log + every CSV/JSON artifact. Use the
  **kernel-stats CSV** (a `*kernel*stats*`-style file, but do NOT rely on the exact name — scan the
  artifact list): each row is a kernel with a call/dispatch **count** and total/avg **duration**.
  Dispatch count = sum of per-kernel counts per call (or distinct kernels × calls); top kernels =
  highest total-duration rows. No SoL/cache fields here — classify from durations + dispatch shape +
  the per-case latency table.
- **`rocprof`** (legacy `--stats`): a stats CSV/table of kernels with counts + durations. Same approach
  as rocprofv3 (counts → dispatch, durations → top kernels); no SoL/cache fields.
- **`benchmark-only`** (no profiler on the box): only the benchmark stdout. Classify from the per-case
  latency table + `geomean_levers.md` heuristics: cases of very different sizes at near-equal latency ⇒
  **overhead-bound** (floor); a large-N case far above the floor ⇒ likely **compute-bound**. State that
  no profiler was available.

## rocprof-compute (formerly omniperf) Output Interpretation

### Section 2: System Speed-of-Light (SoL)

The most important section. Shows overall utilization as percentage of peak.

| Metric | What it means | Threshold |
|--------|--------------|-----------|
| VALU Utilization | Vector ALU usage | > 60% = compute-bound |
| MFMA Utilization | Matrix unit usage | > 40% = MFMA-active workload |
| VMEM Utilization | Vector memory pipe | > 60% = memory-bound |
| LDS Utilization | Local data share | > 50% = LDS-heavy |
| Bandwidth (GB/s) | Effective HBM BW | Compare to this card's HBM peak (≈5300 GB/s MI300X/300A, ~6000 MI325X, ~8000 MI350/355 — see `amd_instinct.md`) |

**Classification from SoL:**
- VALU > 60% AND VMEM < 40% → **compute-bound**
- VMEM > 60% AND VALU < 40% → **memory-bound**
- Both < 40% → **latency-bound**
- LDS > 50% → **lds-bound** (check bank conflicts)
- Both 40-60% → **balanced**

### Section 7.2: Wavefront Runtime Stats

Shows how wavefronts spend their time.

| Metric | What it means |
|--------|--------------|
| Active Cycles | Cycles actually computing |
| Dependency Wait | Stalled waiting for data |
| Issue Wait | Stalled on instruction issue |
| Total Wave Cycles | Total cycles alive |

**Key ratios:**
- `Active / Total` = Kernel efficiency (< 20% = CRITICAL inefficiency)
- `Dependency Wait / Total` = Memory stall fraction
- `Issue Wait / Total` = Instruction scheduling stall

**Diagnosis:**
- High Dependency Wait → memory-bound or cache miss
- High Issue Wait → instruction-level parallelism needed
- Low Active + Low Wait → occupancy too low

### Section 11: Compute Pipeline

| Metric | What it means |
|--------|--------------|
| VALU Active Threads | Average active threads per VALU instruction |
| VALU Utilization % | How much of peak VALU is used |
| Branch Divergence | Fraction of divergent branches |

**Key checks:**
- Active Threads < 64 → wavefront divergence (threads disabled by branches)
- Branch Divergence > 10% → significant divergence penalty
- VALU Util close to SoL → compute is the bottleneck

### Sections 13-16: Cache Hierarchy

#### Section 13: L1 Cache (vL1D)
| Metric | What it means | Threshold |
|--------|--------------|-----------|
| Hit Rate | L1 cache hit % | < 60% = likely memory-bound |
| Bandwidth | L1 effective BW | Compare to peak |
| Coalescing | Memory coalescing efficiency | < 50% = fix access patterns |

#### Section 14: L2 Cache
| Metric | What it means | Threshold |
|--------|--------------|-----------|
| Hit Rate | L2 cache hit % | < 50% = heavy HBM traffic |
| Read/Write BW | L2 bandwidth used | |

#### Section 16: HBM
| Metric | What it means |
|--------|--------------|
| Read BW | HBM read bandwidth achieved |
| Write BW | HBM write bandwidth achieved |
| Total BW | Should be < this card's HBM peak (≈5300 GB/s MI300X; higher on MI325X/MI350/MI355 — `amd_instinct.md`) |

## Bottleneck Classification Decision Tree

```
1. Check SoL VALU vs VMEM utilization
   ├─ VALU > 60%, VMEM < 40% → COMPUTE-BOUND
   ├─ VMEM > 60%, VALU < 40% → MEMORY-BOUND
   ├─ Both > 50% → BALANCED
   ├─ Both < 40% → go to step 2
   └─ LDS > 50% → LDS-BOUND

2. Check Wavefront stats (Active / Total ratio)
   ├─ < 20% → LATENCY-BOUND (critical inefficiency)
   ├─ 20-50% → check Dependency vs Issue wait
   │   ├─ Dependency dominant → MEMORY-BOUND (cache miss stalls)
   │   └─ Issue dominant → LATENCY-BOUND (ILP needed)
   └─ > 50% → check cache hit rates
       ├─ L1 < 60% → MEMORY-BOUND (poor locality)
       └─ L1 > 60% → BALANCED (likely small kernel, launch overhead)
```

## Bottleneck Shift Analysis (for re-profiling after optimization)

After each optimization round, compare before/after metrics:

1. **What changed**: Which metrics improved/degraded?
2. **New bottleneck**: Did the bottleneck shift? (e.g., compute-bound → memory-bound)
3. **Why**: What optimization caused the shift? (e.g., "Template params freed registers, now memory latency is exposed")
4. **Next action**: What strategy should target the new bottleneck?

Format the analysis as:
```
BEFORE: [bottleneck type] - [key metric value]
AFTER:  [bottleneck type] - [key metric value]
SHIFT:  [old] → [new] because [reason]
NEXT:   Target [new bottleneck] with [strategy]
```
