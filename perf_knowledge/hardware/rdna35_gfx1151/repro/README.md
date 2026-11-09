# gfx1151 measurement scripts

The `[measured]` numbers in `../*.md` were taken with these scripts and commands on a Strix Halo box
(Radeon 8060S, gfx1151) inside a ROCm 7.2.3 container with torch 2.11.0 and Triton 3.6.0, on
2026-11-06. Each script is standalone and finishes in seconds to minutes, so a doubted number can be
re-taken.

**Measure on a quiet box.** Check host load from outside the container first; see
`../clocks_power.md`. Hold the GPU for the whole run (for example under `gpu_lock.sh`).

| Script | Backs | Page |
|---|---|---|
| `ceilings.py` | read bandwidth (cold DRAM, LLC-resident); fp16 / bf16 / int8 WMMA peaks per size (`PEAK_SIZES`, `PEAK_DTYPES`), Triton sweep vs `torch.mm` / `torch._int_mm`; clock and power under sustained load | peak_tables, memory, clocks_power |
| `bw.py` | read-read-write streaming bandwidth vs working set (host-timed: small cases are launch-bound) | peak_tables, memory |
| `wmma_check.py` | `tl.dot` lowers to `v_wmma` (no `v_mfma`); fair Triton-vs-`torch.mm` comparison at 2048³ | matrix_core, peak_tables, clocks_power |
| `triton_caps.py` | 6-point Triton capability matrix, each case checked against a reference | matrix_core |
| `noise.py` | idle run-to-run dispersion, the 2σ noise floor | clocks_power |
| `drift.py` | 4 minutes of sustained memory-bound load | clocks_power |
| `pmc_slots.sh` | how deep one cumulative counter set goes in a single rocprofv3 pass, the counters' values, and how long each pass takes, including the hang of a rejected set (uses `pmc_work.py`) | pitfalls |
| `blas_backend.sh` | which library `torch.mm` runs: traces the hottest kernel and finds it in rocBLAS's or hipBLASLt's code objects | pitfalls, isa_notes |

## One-line checks
Facts that a single command shows, run in the same container:

| Command | Shows | Page |
|---|---|---|
| `amd-smi static` | `VRAM TYPE: GDDR7`, no memory clock | pitfalls |
| `rocminfo` | gfx1151, empty marketing name, 32 KB L1, 128 B cache line | pitfalls, arch |
| `python3 -c "import torch; print(repr(torch.cuda.get_device_name()))"` | `''` | pitfalls |
| `rocprof-compute --version; rocprof-compute profile -n t -- python3 pmc_work.py` | 3.4.0; `Cannot find a supported arch in rocminfo` | pitfalls |
| `rocprofv3 -L` | which counters exist (`GL2C_HIT`, `GL2C_MISS`, `SQ_INSTS_LDS`, ...) | pitfalls |
| `echo '<instruction>' \| /opt/rocm/llvm/bin/llvm-mc -arch=amdgcn -mcpu=gfx1151` | `v_dot4_i32_iu8`, `v_wmma_i32_16x16x16_iu8 ... neg_lo:[1,0,0]` assemble; repeat with `-mcpu=gfx942` for the CDNA side | isa_notes |

## Claims with no script here
These rest on campaign logs only. Treat them as the weakest numbers on the pages and re-measure
before relying on them:

- the kernel-campaign speedups and profiles behind the levers in `matrix_core.md` (LDS load counts,
  in-thread transpose, decode tile, int4 K block, epilogue dequant) and `memory.md` (fusions) —
  measured against frozen baselines by GEAK runs, not by a script in this directory;
- the dirty-write-back and cold/warm figures in `memory.md`, from gfx1151 serving traces;
- the leaked-process figures in `clocks_power.md`.

The occupancy table these pages point to is computed by
[`amd_occupancy.py`](../../../../kernel_workflow/scripts/kernel_tools/amd_occupancy.py), which is not duplicated here.
