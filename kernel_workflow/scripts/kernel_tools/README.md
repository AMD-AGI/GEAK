# Shared kernel analysis and screening tools

Evidence CLIs for AMD Triton/Gluon/HIP kernels. They report evidence; they do not pick a tactic,
and GEAK's verify step decides. Each tool's `--help` is its interface. Python 3.10+.

| Tool | Answers | Needs |
| --- | --- | --- |
| `hw_budget.py` | floor, bound and measured/floor multiple for a kernel | SKU or hardware context |
| `extract_sku.py` | per-SKU facts (CUs, XCDs, caches, peaks, ridge); renders SKU tables | `perf_knowledge/hardware/data/sku.json` |
| `mem_bw_probe.py` | in-shape HBM ceiling for an access pattern | GPU (under `gpu_lock.sh`) |
| `amd_isa.py` | page-cited ISA facts (`--arch` required) | committed databases; gfx942/RDNA matrix layout needs the AMD Matrix Instruction Calculator |
| `dump_ir.sh` + `collect_ir.py` | TTGIR/LLIR/AMDGCN/metadata of one compiled kernel | the workload's compile command |
| `probe.py` | `measure`: VGPR/AGPR/spill/LDS/occupancy of a compiled kernel; `plan`: resident VGPRs of a tile plan | compiled artifacts |
| `amd_occupancy.py` | VGPR/LDS -> waves/workgroups per CU (library + CLI) | — |
| `asm_loop_audit.py` | hot-loop op mix, waits, barriers, spills; `--diff` between variants | AMD assembly |
| `ab_bench.py` (+ `ab_task_adapter.py`) | interleaved A/B screening with a byte-identical control arm | kernel runtime + `e2e_workflow/scripts/harness_lib.py` |

Libraries (imported or sourced, not commands): `_hwdata.py`, `_quickhelp.py`, `amd_resources.py`,
`amd_artifacts.py`, `amd_targets.py`, `_amd_isa_db.py`, `_amd_matrix.py`.
Hardware data lives in `perf_knowledge/hardware/data/` (override with `GEAK_HW_DATA_DIR`).
ISA data maintenance: [isa_tools.md](isa_tools.md).

## Contracts

- **GPU work** runs under `kernel_workflow/scripts/gpu_lock.sh` (the GPU tools refuse otherwise or lock
  themselves with `--gpu/--dev <id>`).
- **A/B screening** (`ab_bench.py`) times through `harness_lib.time_op`; GEAK verify on the frozen task
  is the acceptance step. JSON v2: read `screen` only when `validity.valid` is true. Exit 0 = valid
  screening (not a speedup), 2 = invalid screening/input, 1 = run failure. Adapters provide
  `variants()` and either `oracle(name)` or `outputs(name)` + `reference(name)`; `__control__` is reserved.
- **Evidence limits**: static instruction order does not measure overlap; `s_nop` counts are encodings,
  not stall time; planning outputs (`hw_budget.py`, `probe.py plan`) carry `may_gate: false`.
- **Kernel selection**: multi-kernel inputs need `--kernel`; `--loop-label` pins the loop.

## Offline tests

```bash
python -m pytest -q kernel_workflow/scripts/tests -k "kernel_tools or isa or audit or shared"
```

They need no GPU or network; GPU results are recorded separately.
