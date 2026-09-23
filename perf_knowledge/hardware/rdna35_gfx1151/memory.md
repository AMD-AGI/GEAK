---
title: gfx1151 memory hierarchy — 32 MB Infinity Cache, LPDDR5X UMA, and the two rooflines
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/repro/bw.py (streaming read-read-write, warmed, 20 iterations)
  - GEAK/kernel_workflow/knowledge/repro/rdna_roofline.py
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §3, §3b, §4
---

# gfx1151 memory hierarchy

> Peaks and the compute roof in [peak_tables.md](peak_tables.md); the part overview in
> [arch.md](arch.md).

## TL;DR
> **Three** levels, not two: L1 32 KB → **L2 2 MB** → **32 MB Infinity Cache (MALL)** → LPDDR5X
> shared with the CPU. A working set that fits the 32 MB LLC runs at `[measured]` **~790 GB/s**
> against **229–233 GB/s** streaming from DRAM — **3.4×**. That step is the biggest lever on this
> part *for bandwidth-bound work only*; for high-arithmetic-intensity kernels it is worth **nothing**,
> and the window where it changes a verdict is narrow and computable.

## The measured bandwidth table
`[measured]` `repro/bw.py`, streaming read-read-write, warmed, 20 iterations:

| Working set | Bandwidth |
|---|---|
| ≤ 32 MB (fits LLC) | **~790 GB/s** |
| 64 MB | 230 GB/s |
| 256 MB | 233 GB/s |
| 1024 MB | 229 GB/s |

| Figure | Value | Note |
|---|---|---|
| DRAM, paper | 256 GB/s | `[vendor]` — **~12% optimistic** |
| DRAM, achievable streaming | **229–233 GB/s** | `[measured]` `bw.py` |
| DRAM, real KV-streaming kernel | 222 GB/s | `[measured]` llama.cpp hip-ep PR #675, RGP |
| LLC, read-read-write | ~790 GB/s | `[measured]` `bw.py` |
| LLC, **pure read** | **782 / 913 / 945 GB/s** | `[measured by ablation]` — see below |

## Concepts

### 790 GB/s is a read-read-write number. Do not use it as a read ceiling.
`[measured by ablation]` A same-grid / same-byte-stream kernel with the math deleted reached
**782 / 913 / 945 GB/s** on LLC-resident pure-read working sets. One GQA decode kernel was called
"saturated" against 790 and was in fact at **83–86%** of its real ceiling, with ~17% left on the
table. **Measure the ceiling for YOUR access mix by ablation** — keep the grid and the byte stream,
delete the math, time that — rather than taking a number from the table above.

### "Under 32 MB" is not one performance class
`[measured]` 6.3 MB gives ~35.1 TFLOP/s on a compute-bound fp16 GEMM but 25.2 MB — still nominally
LLC-resident — gives ~28.4. Those are *compute* numbers on a compute-bound shape, so the variation
is tile/occupancy efficiency rather than the cache roof itself. **Where the 790 GB/s figure starts
to degrade within the 32 MB has not been measured.** Do not assume a flat plateau.

### The window where the LLC/DRAM choice changes a verdict
With a 59.4 TFLOP/s compute roof, the two bandwidth roofs cross the compute roof at

```
59.4e12 / 790e9 =  75 FLOP/byte      (LLC)
59.4e12 / 230e9 = 258 FLOP/byte      (DRAM)
```

**Only between ~75 and ~258 FLOP/byte does the LLC-vs-DRAM choice flip a bound/not-bound verdict.**
Above it, both roofs clear the compute roof and the choice is irrelevant. Below it, both bind and
the kernel is memory-bound either way. **Compute the window before arguing about which bandwidth to
use.**

Worked counter-example `[measured]` `rdna_roofline.py`, fp16 square GEMM: 1024³ (6.3 MB, deep in
LLC) reaches ~35.1 TFLOP/s while 3072³ (56.6 MB, well into DRAM) reaches **~38.4** — the
DRAM-regime shape is ~10% *faster*. At AI ≈ 683 FLOP/byte even the DRAM roof (683 × 230 GB/s =
157 TFLOP/s) sits far above the compute roof, so bandwidth never binds and cache residency buys
nothing.

### The hierarchical roofline is NOT implemented
Selecting one bandwidth by "does the total working set fit in 32 MB" is a simplification. The
honest form needs traffic measured per level:

```
attainable = min(P_compute, AI_DRAM × BW_DRAM, AI_LLC × BW_LLC, AI_L2 × BW_L2)
```

and this part has **three** cache levels to account for. A kernel whose hot set fits the 2 MB L2 is
in a different regime again, and the same working set behaves differently warm vs cold — `[measured]`
the cold/warm gap on real decode weights is **3–4×**. `GL2C_HIT` / `GL2C_MISS` **are** collectable
here (a real run measured **GL2 hit 50.37%**), so the per-level traffic this needs is measurable;
it simply is not wired up. Until it is, use these numbers to calibrate expectations and **do not
wire them to an automatic keep/reject decision**.

### UMA: the CPU is on the same memory
LPDDR5X is shared with the CPU (96 GB carve-out of 128 GB). A busy CPU steals GPU bandwidth, and
the two also share a power envelope — see [clocks_power.md](clocks_power.md).

## The levers
1. **Split/tile so the hot set drops under 32 MB** — but only after checking the AI window above.
2. **Ablate to find your real ceiling** before declaring a read-dominated kernel saturated.
3. **Keep write-once output out of the LLC.** If a result will not be re-read, letting it occupy
   LLC evicts the stream that will. `[measured]` a contributor to the 1.81× add+rmsnorm win, whose
   two streaming shapes moved from ~81% to 99–100% of the measured DRAM wall.
4. **Count bytes against the fused contract floor.** `[measured]` a 3-dispatch baseline was already
   at 96/101/83% of the correct wall *on the bytes it actually moved* — and only ~25–40% of it on
   ideal bytes. Efficiency against the wrong byte count hides a 4× traffic bug.

## Pitfalls
- **Rooflining on 256 GB/s.** Use 229–233.
- **Using 790 GB/s as a read ceiling.** It is read-read-write; pure read reaches 913–945.
- **Treating ≤32 MB as a single class**, or assuming the LLC lever helps compute-bound work.
- **Porting CDNA memory intuition.** This part has ~1/25th the bandwidth of an MI350X and a
  completely different hierarchy; what counts as "memory-bound" moves accordingly.

## Verify
- `repro/bw.py` for the streaming table; re-run it rather than trusting these numbers on a
  different RDNA part.
- `rocprofv3` with `GL2C_HIT` / `GL2C_MISS` for L2 hit rate — but mind the PMC budget, see
  [tooling_and_profiling.md](tooling_and_profiling.md).
- Ablation harness (same grid, same bytes, math deleted) for the per-access-mix ceiling.

## Sources
- `GEAK/kernel_workflow/knowledge/repro/bw.py`, `rdna_roofline.py` — on-box, ROCm 7.2.3.
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §3, §3b, §4.
- llama.cpp hip-ep PR #675 (RGP capture of a real KV-streaming kernel at 222 GB/s).
