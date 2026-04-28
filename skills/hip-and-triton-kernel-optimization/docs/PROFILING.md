# PROFILING.md — rocprofv3 reference for AMD CDNA-3/4

Companion to [SKILL.md](../SKILL.md). Full invocations and counter
reference for the three tiers of profiling. All examples target gfx950
(MI355X / CDNA-4); the same flags work on gfx942 (CDNA-3).

## Driver pattern

Every `rocprofv3` invocation needs a deterministic driver that runs
your kernel a known number of times. Pattern:

```python
# driver.py
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=["asm", "hip"], required=True)
ap.add_argument("--ctx", type=int, default=4000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()

# Build inputs once, then loop the kernel `iters` times.
# rocprofv3 will see N dispatches; we drop the cold one in post-processing.
inputs = build_inputs(args.batch, args.ctx)
for _ in range(args.iters):
    run_kernel(args.target, inputs)
torch.cuda.synchronize()
```

`iters = 10` is a good default: enough to median-filter the cold
dispatch, few enough that PC-sampling output stays manageable.

## Tier 1 — Perfetto timeline (`--sys-trace`)

### Invocation

```bash
rocprofv3 --sys-trace -d timeline_out -o timeline_<tag> \
  -- python3 driver.py --target asm --ctx 4000 --batch 4 --iters 10
```

### Outputs

| File | Use |
|---|---|
| `timeline_<tag>_results.pftrace` | Open in [ui.perfetto.dev](https://ui.perfetto.dev) for visual timeline |
| `timeline_<tag>_csv_kernel_stats.csv` | Per-kernel total / avg / min / max in ns |
| `timeline_<tag>_csv_kernel_trace.csv` | Per-dispatch start/end timestamps |
| `timeline_<tag>_csv_domain_stats.csv` | HIP API vs. kernel time breakdown |

### Reading kernel_stats.csv

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps",23,205401,8930,45.40,8520,11640,681
"_Z16kn_mla_reduce_v1...",23,179482,7803,39.67,7120,10680,746
```

`AverageNs` is your headline kernel time. `MaxNs - MinNs` quantifies
launch jitter; a stdev > 10 % of average usually means warmup is
incomplete or PMC overhead is contaminating timing.

### Decision template

Write down the percentage breakdown across kernels in the launch
chain. If one stage is > 80 % of total, focus there. If two stages
are roughly balanced, profile both before picking.

## Tier 2 — PMC counter sweep (`--pmc`)

### Why four groups

gfx950 has a fixed number of HW counter slots per pass and a strict
group-of-counters whitelist. A useful set (waves, ALU busy, LDS, VMEM,
HIT/MISS, instruction mix) does not fit in one pass. The fix is a
four-group sweep run as separate `rocprofv3` invocations, then merged
in post-processing.

### Counter groups

```
GROUPS=(
  "a:SQ_WAVES,GRBM_GUI_ACTIVE,VALUBusy,SALUBusy"
  "b:LDSBankConflict,MemUnitStalled,FetchSize,WriteSize"
  "c:SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_LDS,SQ_WAVES"
  "d:SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,TCC_HIT_sum,TCC_MISS_sum"
)
```

Per-group rationale:

- **a (timing)** — `GRBM_GUI_ACTIVE` is the front-end-active count
  (≈ kernel duration in cycles). `SQ_WAVES` lets you normalize all
  per-wave metrics. `VALUBusy` / `SALUBusy` are the headline ALU
  occupancy numbers.
- **b (memory pressure)** — `LDSBankConflict` is the ratio of
  conflicted to total LDS accesses; `MemUnitStalled` is the fraction
  of cycles the memory unit was blocked. The ratio
  `LDSBankConflict / MemUnitStalled` tells you whether the memory
  stalls are LDS-side or VMEM-side. `FetchSize` / `WriteSize` are
  HBM bytes per launch.
- **c (instruction mix per wave)** — divide each `SQ_INSTS_*` by
  `SQ_WAVES` to get per-wave instruction counts. Compare to a
  reference kernel (ASM, vendor) to see which class is bloated.
- **d (VMEM + L2)** — `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`
  is the L2 hit rate. A high hit rate combined with high `FetchSize`
  signals redundant fetches.

### Sweep invocation

```bash
for tgt in asm hip; do
  for ctx in 1000 2500 4000 7000 9000; do
    for grp in "${GROUPS[@]}"; do
      gname=${grp%%:*}
      counters=${grp#*:}
      counters_sp=$(echo "$counters" | tr ',' ' ')
      tag="${tgt}_b4_ctx${ctx}_g${gname}"
      rocprofv3 --pmc $counters_sp -f csv \
        -d pmc_runs -o "$tag" \
        --kernel-include-regex "${tgt}_kernel_pattern.*" \
        -- python3 driver.py --target $tgt --ctx $ctx --batch 4 --iters 10
    done
  done
done
```

`--kernel-include-regex` filters to your kernel only. Without it,
rocprofv3 collects PMC for every kernel in the process (including
PyTorch internals), bloating the CSVs.

### Worked example (Kimi-K2 MLA decode at b=4)

| ctx | LDSBankConflict (ASM) | LDSBankConflict (HIP-baseline) | LBC/MemUnitStalled (ASM) | LBC/MemUnitStalled (HIP) |
|---:|---:|---:|---:|---:|
| 1000 | 0.98 | 1.45 | 0.0035 | **0.118** |
| 4000 | 1.93 | 3.13 | 0.0029 | 0.049 |
| 9000 | 2.66 | 4.13 | 0.0029 | 0.034 |

The `0.118` at ctx=1000 was the smoking gun: HIP-baseline LDS pressure
was 35× the ASM kernel's. This drove the v9k transposed-LDS rewrite.

| Counter | ASM (per wave) | HIP (per wave) | Ratio |
|---|---:|---:|---:|
| `SQ_INSTS_VALU` | 236 | 1130 | 4.8× |
| `SQ_INSTS_SALU` | 107 | 297 | 2.8× |
| `SQ_INSTS_LDS` | 65 | 408 | 6.3× |
| `SQ_INSTS_MFMA` (scaled to NHEAD-equivalent) | 15.5 | 7.3 | 0.47× |

The 4.8× VALU and 6.3× LDS gap meant we were doing too much data
shuffling. The MFMA pipe at 0.47× ASM meant it was *under-fed*, not
saturated — confirmed by Tier 3 below.

### Aggregation

CSV format: `*_counter_collection.csv` per `(target, ctx, group)`.
Aggregate by computing the median across the 10 dispatches (drop the
cold first). Standard pandas pattern:

```python
import pandas as pd
from pathlib import Path

rows = []
for csv in Path("pmc_runs").glob("*_counter_collection.csv"):
    parts = csv.stem.replace("_counter_collection", "").split("_")
    tgt, ctx, grp = parts[0], int(parts[2][3:]), parts[3][1:]
    df = pd.read_csv(csv)
    for counter, sub in df.groupby("Counter_Name"):
        vals = sub["Counter_Value"].astype(float).values
        dur_ns = (sub["End_Timestamp"].astype(int)
                  - sub["Start_Timestamp"].astype(int)).values
        rows.append(dict(target=tgt, ctx=ctx, counter=counter,
                         median=float(pd.Series(vals).median()),
                         median_dur_ns=float(pd.Series(dur_ns).median())))
agg = pd.DataFrame(rows)
```

## Tier 3 — PC sampling (`--pc-sampling-method host_trap`)

### Invocation

```bash
rocprofv3 --pc-sampling-method host_trap \
          --pc-sampling-interval 50us \
          -d pc_out -o pc_<tag> \
          --kernel-include-regex "your_kernel_regex.*" \
          -- python3 driver.py --target asm --ctx 4000 --batch 4 --iters 200
```

`50us` is a good default for kernels in the 5–30 µs range. Drop to
`10us` for kernels < 5 µs. `iters=200` because each sample is a single
PC value; you need enough dispatches to get statistical coverage of the
hot regions.

### Output

`pc_<tag>_pc_sampling_host_trap.csv` with rows like:

```
Sample_Timestamp,Exec_Mask,Dispatch_Id,Instruction,Instruction_Comment,Correlation_Id
267256155006931,0xffff...,0,"v_mfma_f32_16x16x32_fp8_fp8 ...","",0
267256155007131,0xffff...,0,"s_waitcnt vmcnt(0) lgkmcnt(0)","",0
```

### Post-processing: bucket by instruction class

Annotate the disassembly first:

```bash
# Get disassembly of the kernel object
roc-obj-extract <kernel.co>
llvm-objdump --disassemble <kernel.co> > disasm.s
```

Then bucket the PC samples:

```python
import pandas as pd, re
df = pd.read_csv("pc_<tag>_pc_sampling_host_trap.csv")

def classify(insn):
    insn = (insn or "").strip()
    if insn.startswith("s_waitcnt"):     return "s_waitcnt"
    if insn.startswith("v_mfma"):        return "MFMA"
    if insn.startswith(("ds_read", "ds_write")): return "LDS"
    if insn.startswith(("buffer_load", "buffer_store",
                        "global_load", "global_store",
                        "flat_load", "flat_store")): return "VMEM"
    if insn.startswith("v_"):            return "VALU"
    if insn.startswith("s_"):            return "SALU"
    if insn.startswith(("v_cmp", "s_cbranch", "s_branch")): return "ctrl"
    return "other"

df["class"] = df["Instruction"].apply(classify)
hist = (df["class"].value_counts(normalize=True) * 100).round(1)
print(hist)
```

### Worked example (Kimi-K2 MLA decode, ASM kernel)

```
class       %
s_waitcnt   27.7
VALU        16.4
LDS         15.8
VMEM        12.1
SALU         9.0
MFMA         7.8
ctrl         7.0
SMEM         3.9
```

Reading: 7.8 % MFMA means the MFMA pipe is **idle 92 % of the time**
even on hand-tuned ASM. 27.7 % `s_waitcnt` means the kernel is
sitting on synchronization waits, not arithmetic. **Conclusion: this
kernel is wait-counter-bound, not arithmetic-bound. Adding MFMA
throughput will not help; cutting staging traffic will.**

### Decision template

| MFMA % | s_waitcnt % | Interpretation | Next step |
|---:|---:|---|---|
| > 50 | < 10 | Arithmetic-bound | Use larger MFMA opcode (recipe 3) |
| 20–50 | 10–20 | Balanced | Tune scheduling (recipe 6) |
| < 20 | > 25 | Wait-counter-bound | Cut staging traffic (recipes 1, 2, 9) |
| < 20 | < 10 | Memory-bound | Check FetchSize, L2 hit rate (recipe 9) |

## Tier 4 — ATT thread trace (often unavailable)

### What it would give you

- Per-CU per-wave-slot timeline of VALU / SALU / LDS / MFMA / VMEM
  issue events.
- Per-instruction `s_waitcnt` stall attribution (vs. PC-histogram
  aggregate).
- LDS bank-conflict heat map per access.

### Why it often fails

`rocprofv3 --att` requires `librocprof-trace-decoder.so`, which is
only installed in some ROCm distributions. Symptom:

```
Fatal error: rocprof-trace-decoder library path not found in
  ['/opt/rocm/lib', '/usr/local/lib', '', '/opt/rocm-*/lib']
```

If the package is not on pip and the bundled installer's tarball is
corrupted, you cannot decode the raw `.att` blobs even if rocprofv3
captures them.

### Workaround

Substitute Tier 2 group `c` (`SQ_INSTS_*`) per dispatch. You lose:

- Per-wave-slot resolution (you get per-dispatch aggregates).
- Per-instruction `s_waitcnt` attribution (you get the PC histogram
  from Tier 3 instead).

You keep: total instruction counts per class, which is enough for
99 % of optimization decisions.

## Counter reference (only what you'll actually use)

| Counter | Meaning | Useful for |
|---|---|---|
| `SQ_WAVES` | Total waves launched | Normalizing per-wave metrics |
| `GRBM_GUI_ACTIVE` | Front-end active cycles | Kernel duration in cycles |
| `VALUBusy` | % cycles VALU busy | ALU occupancy |
| `SALUBusy` | % cycles SALU busy | Scalar pipeline occupancy |
| `LDSBankConflict` | Conflicted LDS accesses (ratio or count, varies) | LDS pressure |
| `MemUnitStalled` | % cycles memory unit blocked | Detecting memory-side stalls |
| `FetchSize` | KB read from HBM per launch | HBM bandwidth |
| `WriteSize` | KB written to HBM per launch | HBM bandwidth |
| `SQ_INSTS_VALU` | VALU instructions | Instruction mix |
| `SQ_INSTS_SALU` | SALU instructions | Instruction mix |
| `SQ_INSTS_LDS` | LDS instructions | Instruction mix |
| `SQ_INSTS_MFMA` | MFMA instructions | MFMA throughput proxy |
| `SQ_INSTS_VMEM_RD` | VMEM reads | HBM read pattern |
| `SQ_INSTS_VMEM_WR` | VMEM writes | HBM write pattern |
| `TCC_HIT_sum` | L2 hits | L2 effectiveness |
| `TCC_MISS_sum` | L2 misses | L2 effectiveness |

Skip `MemUnitBusy`, `L2CacheHit` (use the TCC sums instead — they
agree with `rocm-smi` and `rocprof v1` data while `L2CacheHit` is
sometimes scaled differently between ROCm releases).

## Common pitfalls

- **rocprofv3 vs rocprof.** Always use `rocprofv3`. The legacy
  `rocprof` (v1) has different flags, different output format, and
  does not support `--pc-sampling-method` reliably on gfx950.
- **Forgetting `-f csv`.** Default output is JSON, which is harder to
  pipe into pandas. Add `-f csv` to all PMC and timeline runs.
- **Wrong `--kernel-include-regex`.** The regex matches the demangled
  kernel name; check `_kernel_stats.csv` after a Tier 1 run to confirm
  the exact name. ASM kernels have C-style names; HIP kernels have
  C++-mangled names.
- **PMC overhead inflating timing.** PMC adds ~5–15 % overhead per
  collected counter. Don't read kernel duration from a PMC run; use
  Tier 1's `--sys-trace` for timing and Tier 2 for counters only.
- **Cold dispatch contamination.** Always run ≥ 10 iterations and
  drop the first one in post-processing (median, not mean).
