# ISA, matrix layout and static occupancy tools

Static facts and planning bounds for Triton, Gluon and HIP kernels. Measured utilization and
calibration stay with GEAK's profiling workflow.

## Entry points

- [amd_isa.py](amd_isa.py): the AMD ISA CLI; `--arch` is required. `capabilities --json` reports the
  target, ISA family, data source and availability of each query. gfx950 answers every query from
  committed data (encoding, MFMA layout, ISA prose facts). gfx942/gfx1201 answer encoding queries from
  committed data; their matrix facts/layout/locate need a preinstalled AMD Matrix Instruction
  Calculator (`TILE_HW_SEED` / `TILE_HW_TOOLS` point at the directory holding
  `amd_matrix_instruction_calculator/`; `GEAK_MATRIX_PYTHON` selects its interpreter). A missing
  source prints `SCOPED TOOL-GAP` (rc 3); facts are never borrowed from another target.
  For gfx950 F8F6F4, `facts OPCODE --as fp4|fp6|fp8` reports equal-format A/B register spans;
  `tables scale.16x16x128` / `scale.32x32x64` print source excerpts.
- [amd_targets.py](amd_targets.py): target aliases, query coverage, asset lookup.
  [_amd_isa_db.py](_amd_isa_db.py) and [_amd_matrix.py](_amd_matrix.py) are the two providers.
- [dump_ir.sh](dump_ir.sh) runs the caller's compile command with a fresh cache;
  [collect_ir.py](collect_ir.py) accepts one complete compilation group (TTGIR, LLVM IR, AMDGCN,
  metadata), checks the assembly kernel against its metadata and writes a manifest.
- [amd_resources.py](amd_resources.py) parses resources (architectural, accumulator and combined
  registers, per-field sources); [amd_artifacts.py](amd_artifacts.py) binds launch/LDS metadata through
  the manifest or a same-stem sidecar. [amd_occupancy.py](amd_occupancy.py) owns register/LDS occupancy
  arithmetic and compiler sweeps. [probe.py](probe.py) renders the resource record (`measure`) or
  estimates resident tensors for a tile plan (`plan`; not compiler allocation).

The Gluon pack keeps `scripts/probe.py` as a legacy entry (gfx950 default for `plan` when `--arch` is
omitted, with a stderr notice; `GEAK_HARDWARE_CONTEXT` overrides it). It needs a full GEAK checkout
(`GEAK_ROOT` for a relocated entry). `dump_ir.sh --emit-gluon` calls the pack's optional
`recover_gluon.py` after a successful capture.

```bash
python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx1201 encoding v_wmma_f32_16x16x16_f16
python3 kernel_workflow/scripts/kernel_tools/probe.py plan --arch gfx942 --warps 4 --acc acc=128x128
bash kernel_workflow/scripts/kernel_tools/dump_ir.sh --out ir --variant candidate --kernel-name my_kernel -- python compile_candidate.py
python3 kernel_workflow/scripts/kernel_tools/probe.py measure --dir ir/candidate --json ir/candidate/probe.json
```

Run the compile command in the workload environment, under GEAK's GPU lock if it launches GPU work.
Pin one autotune configuration when inspecting its IR.

## Data maintenance

Databases live in [perf_knowledge/hardware/data](../../../perf_knowledge/hardware/data) with source
release/page per record. `GEAK_HW_DATA_DIR` selects an exclusive data directory (missing files report a
gap). Matrix data schema v2 keeps queried records in `gfx950-mfma-layout.json`; source figures and
conversion transcriptions live in the [reference evidence fixture](../tests/fixtures/gfx950-isa-evidence.json)
(`reference_evidence`), which normal queries do not load. A successful lookup is not GPU numerical
validation; new CDNA4 low-precision and sparse layouts keep their recorded caveats.

When updating committed ISA data: record the upstream document/XML release, page or instruction id,
license, date and verification scope; update the database and `amd_targets.py`; run the selftests and
the suites below; keep known errors visible. No XML/PDF-to-database generator is included, so review
the source diff per record.

```bash
python3 -m pytest -q kernel_workflow/scripts/tests/test_kernel_tools_isa.py kernel_workflow/scripts/tests/test_isa_workflow_boundaries.py kernel_workflow/scripts/tests/test_shared_amd_tools.py kernel_workflow/scripts/tests/test_amd_matrix_interpreter.py
```

`amd_occupancy.py --compiler-sweep --arch <gfx> --format json` compiles LLVM IR and reads LLVM's
occupancy report (no kernel launch).

Register and LDS occupancy from `amd_occupancy.py` / `probe.py measure` are static upper bounds, not achieved
residency. Use the launch's workgroup size for waves per workgroup: `.max_flat_workgroup_size` is a compile-time
bound, not the launch.

## Coverage

| Query | gfx950 / CDNA4 | gfx942 / CDNA3 | gfx1201 / RDNA4 |
| --- | --- | --- | --- |
| Instruction encoding | committed | committed | committed |
| Matrix facts/layout/locate | committed | calculator | calculator |
| Register/waitcnt/memory prose | committed, with caveats | not extracted | not extracted |
| Static occupancy | registers + LDS | registers + LDS | registers; LDS withheld |

ISA queries and IR inspection need no rocprofv3, rocprof-compute or ATT.
