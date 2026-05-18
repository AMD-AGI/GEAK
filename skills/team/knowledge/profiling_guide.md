# Profiling Analysis Guide

## rocprof-compute (formerly omniperf) Output Interpretation

### Section 2: System Speed-of-Light (SoL)

The most important section. Shows overall utilization as percentage of peak.

| Metric | What it means | Threshold |
|--------|--------------|-----------|
| VALU Utilization | Vector ALU usage | > 60% = compute-bound |
| MFMA Utilization | Matrix unit usage | > 40% = MFMA-active workload |
| VMEM Utilization | Vector memory pipe | > 60% = memory-bound |
| LDS Utilization | Local data share | > 50% = LDS-heavy |
| Bandwidth (GB/s) | Effective HBM BW | Compare to peak 5300 GB/s |

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
| Total BW | Should be < 5300 GB/s peak |

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
