---
title: RDNA3.5 / gfx1151 (Strix Halo) — pitfalls
kind: hardware
gens: [gfx1151]
dtypes: []
regimes: [both]
updated: 2026-11-06
sources:
  - ../rdna4_gfx1201/pitfalls.md
  - ../../expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md
  - repro/pmc_slots.sh
  - repro/pmc_work.py
  - repro/blas_backend.sh
  - https://github.com/ROCm/rocm-systems/blob/1efd9a9c026cdc6e503607be0d20d5383fec166b/projects/rocprofiler-compute/CHANGELOG.md
---

# RDNA3.5 / gfx1151 — pitfalls

RDNA-wide items (CDNA PMC names may be absent, check `rocprofv3-avail list --pmc`, no MFMA / MX /
FNUZ) are in [`rdna4_gfx1201/pitfalls.md`](../rdna4_gfx1201/pitfalls.md) and
[`platform-known-issues.md`](../../expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md)
and apply here. These are the ones specific to gfx1151. Measured on a Radeon 8060S, ROCm 7.2.3,
2026-11-06; the one-line checks are in [repro/README.md](repro/README.md).

## Profiling
- **`amd-smi static` reports `VRAM TYPE: GDDR7`** and no memory clock. It is LPDDR5X shared with the
  CPU; take bandwidth from [memory.md](memory.md), not from `amd-smi`.
- **`rocprof-compute` 3.4.0 (ROCm 7.2.3) refuses the part**: `Cannot find a supported arch in
  rocminfo`. Use `rocprofv3`, which works.
- **Newer `rocprof-compute` has gfx1151 known issues** (vendor CHANGELOG, not measured here):
  - 3.6.0 adds gfx1151; its roofline table shows `N/A` for the peak field.
  - 3.7.0 renames the output directory from `strix_halo` to `rdna35_halo`, so a script that
    hard-codes the old name finds no data.
  - 3.7.0 does not populate `$max_mclk`; bandwidth metrics need `--specs-correction`.
  - 3.7.0: `TCP_REQ_sum` reads zero in single-pass collection.
  - 3.10.0: `TCP_REQ*` can read zero at the `AUTO` performance level, because the perfmon clock is
    power-gated.
- **The PMC limit is a hardware-counter budget, not a metric count.** One cumulative set
  (`SQ_WAVES`, `GRBM_GUI_ACTIVE`, `FETCH_SIZE`) fits, with every counter reading non-zero at the
  `AUTO` performance level. Adding `WRITE_SIZE` is rejected with `error code 38: Request exceeds the
  capabilities of the hardware to collect` ([`repro/pmc_slots.sh`](repro/pmc_slots.sh); the same
  result on 2026-09-17 and 2026-11-06). A metric can expand into several counters: on gfx1151
  `FETCH_SIZE` is four `GL2C_EA_RDREQ_*B` counters (`rocprofiler-sdk/counter_defs.yaml`). Probe the
  set you need and split it into passes.
- **A pass over the counter budget does not exit.** `[measured]` After error 38 the workload keeps
  running on the CPU (~200%, no GPU work, no output) until it is killed. In
  [`repro/pmc_slots.sh`](repro/pmc_slots.sh), accepted passes take 1–2 s and each rejected pass runs
  to the script's timeout. Run every profiling pass under a timeout, and read high CPU / idle GPU /
  no output after error 38 as this hang, not as start-up.
- **Available counters.** `rocprofv3 -L` lists RDNA-native `GL2C_HIT` / `GL2C_MISS`, the derived
  `L2CacheHit`, `SQ_INSTS_LDS` and `SQC_LDS_BANK_CONFLICT`. It does not list the CDNA `TCP_*` /
  `TCC_*` names.
- **`torch.cuda.get_device_name()` returns an empty string**, and so does the `rocminfo`
  marketing name; identify the part by `gcnArchName` or the `rocminfo` ISA name.

## Measurement
- **Time short calls in batches.** A single ~0.6 ms GEMM per event pair spread 14–37% round to round;
  see [clocks_power.md](clocks_power.md).
- **Evict with reads.** A write-based cache flush leaves dirty lines whose write-back is charged to
  the timed kernel; see [memory.md](memory.md).
- **Know which BLAS you measured.** `torch.mm` here runs rocBLAS. The hottest fp16 kernel is a
  Tensile `Cijk_..._MI16x16x16x1_..._ISA1151_...` kernel found in rocBLAS's library and not in
  hipBLASLt's ([`repro/blas_backend.sh`](repro/blas_backend.sh)). A `torch.mm` number says nothing
  about hipBLASLt.

## Numerics / ISA
- fp8 executes on an emulated path here; see [`amd_ryzen.md` §2](../../../kernel_workflow/knowledge/amd_ryzen.md).
- Spills land in shared DRAM; see [isa_notes.md](isa_notes.md).
