# ISA, matrix layout and static occupancy tools

These shared tools support Triton, Gluon and HIP inspection. They supply static
facts and planning bounds; measured utilization, native roofline models and
calibration remain with GEAK's profiling workflow and TraceLens.

## Minimal interface ownership

Compiler layout construction, scale layouts and available layout views belong to
installed Triton. These shared tools provide hardware facts and artifact/resource
checks. Ordinary native Gluon layout queries do not need the optional AMD
calculator. Inspect installed API signatures because vendor builds can backport
compiler features.

`amd_isa.py` is the normalized hardware interface. `hw_sources.sh layout` is an
explicit raw calculator passthrough for modifiers, sparse K and inverse queries;
compatibility entrypoints do not implement a separate mapping algorithm.
For gfx950 F8F6F4, `facts OPCODE --as fp4|fp6|fp8` reuses hardware layout rules to
report equal-format A/B register spans. An unspecified format stays unknown;
XML operand-width declarations remain separate from format-specific use and
compiled allocation. `tables scale.16x16x128` / `scale.32x32x64` expose source
excerpts and placement caveats without reimplementing Triton's scale generator.

Default ISA queries read only the committed data, independent of installed
Triton/HIP/CK/FlyDSL packages. Seed/cache XML does not override `facts` or
`encoding`; use the explicit `hw_sources.sh xml` / decoder path when inspecting
external primary sources. `GEAK_HW_DATA_DIR` selects an exclusive ISA data
directory: missing files report a gap, rather than mixing it with the checkout.
This ISA-specific rule does not change the locator policy of SKU/planning tools.

Matrix data schema v2 keeps queried records in `gfx950-mfma-layout.json`.
Source figures and conversion transcriptions live in the
[reference evidence fixture](../tests/fixtures/gfx950-isa-evidence.json), linked
by `reference_evidence`; normal queries do not load it. Figure cells use an
explicit column list and one JSON row per source cell; citations and known-bad
figures are retained. Raw schema-v1 consumers of `figure_layouts`/`conversions`
must follow this reference; the query CLI and legacy entrypoints remain.
`--selftest` checks runtime integrity without test assets. Detailed source/grid
regressions run in [the existing test suite](../tests/test_kernel_tools_isa.py).

## Entry points and ownership

- [amd_isa.py](amd_isa.py) is the common AMD ISA CLI. New callers must pass
  `--arch`. `capabilities --json` reports the requested target, ISA family, data
  source and availability of each query. gfx942/gfx1201 matrix facts, layout and
  coordinate queries use a preinstalled AMD Matrix Instruction Calculator;
  encoding queries use the committed databases. gfx950 additionally has committed
  layout and ISA prose facts. Missing facts are not borrowed from another target.
- [amd_targets.py](amd_targets.py) owns target aliases, query coverage and the
  seed/tools/cache asset lookup. [_amd_isa_db.py](_amd_isa_db.py) and
  [_amd_matrix.py](_amd_matrix.py) implement the two query providers.
- [gfx950_isa.py](gfx950_isa.py) and [layout_facts.py](layout_facts.py) preserve
  previous command paths. The former retains its historical gfx950 default;
  both use the common dispatcher. Do not add separate per-chip query scripts.
- [hw_sources.sh](hw_sources.sh) checks optional assets and exposes source XML,
  decoder queries and PDF pages. Downloads require `TILE_HW_ALLOW_FETCH=1`.
  Queries do not install dependencies. Set `TILE_HW_SEED` or `TILE_HW_TOOLS` to
  existing assets, and `TILE_HW_CACHE` to a writable workload cache when needed.
- [dump_ir.sh](dump_ir.sh) runs the caller's command with a fresh absolute cache
  path. [collect_ir.py](collect_ir.py) requires one complete compilation group:
  nonempty TTGIR, LLVM IR, AMDGCN and named metadata. It checks the observable
  assembly kernel against metadata before copying, and rejects empty, partial
  or ambiguous captures, preserves the previous variant and writes a manifest.
- [amd_resources.py](amd_resources.py) is the shared resource parser. It keeps
  architectural, accumulator and combined register counts distinct, handles zero
  occupancy, rejects multi-kernel assembly, and records per-field sources. Actual
  launch geometry never comes from the compiler's maximum workgroup bound.
  [amd_artifacts.py](amd_artifacts.py) binds launch/LDS metadata through a manifest
  or same-stem cache sidecar, checking the assembly symbol independently at read
  time. Renamed files are supported through the manifest. Missing symbols leave
  launch data unknown; matching names in unrelated directories are insufficient.
  Probe text and JSON use the same per-artifact records, so different compilations
  of the same kernel retain their own resources. The JSON `kernels` list is
  authoritative; the name-keyed LDS summary omits conflicting configurations.
  [amd_occupancy.py](amd_occupancy.py) owns register/LDS model arithmetic and
  compiler sweeps; its assembly entry point uses this same parser.
- [probe.py](probe.py) renders the shared resource record or estimates resident
  tensors for a tile plan. Planning is not actual compiler register allocation.

Old Gluon-pack command paths forward to these shared tools. `probe.py` and
`dump_ir.sh` require a **full GEAK checkout**; copying only the skill directory is
not a self-contained installation. Their relocated entries accept `GEAK_ROOT`
pointing to that checkout (the path must exist inside the workload container).
Missing shared files produce an actionable error; they never download code.

The legacy Gluon CLI `probe.py plan` retains its historical **gfx950 reference**
when `--arch` is omitted, with a stderr notice. If `GEAK_HARDWARE_CONTEXT` is set,
it validates that context and uses its observed ISA instead; an invalid context
fails. An explicit `--arch` is an offline analysis override. The shared CLI and
Python planning API require an architecture; `measure` never assumes gfx950 for
unnamed artifacts. Python imports expose the shared API, including these stricter
resource semantics. No numeric implementation is duplicated. Plain
querying and inspection do not require Gluon. Transcription stays in the pack's
`emit_gluon.py` / `recover_gluon.py`; `dump_ir.sh --emit-gluon` is an explicit
compatibility hook that invokes that optional adapter after successful capture.

## Workflow use

Analyze/Plan can query the ISA and estimate register pressure before generating
code. After a candidate compiles, inspect its IR and run the probe; include its
target and unresolved terms in the result handed to Optimize/Profile. These
helpers do not add agents or change automatic skill selection. Existing default
knowledge in `amd_instinct.md` and `profiling_guide.md` also receives corrected
occupancy facts; that guidance is read by existing kernel lanes even if they do
not call a new tool. It applies to HIP, Triton and Gluon. Static bounds inform
analysis; they do not prescribe a tuning tactic or replace profiling/A-B gates.

```bash
python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx1201 encoding v_wmma_f32_16x16x16_f16
python3 kernel_workflow/scripts/kernel_tools/probe.py plan --arch gfx942 --warps 4 --acc acc=128x128
bash kernel_workflow/scripts/kernel_tools/dump_ir.sh --out ir --variant candidate --kernel-name my_kernel -- python compile_candidate.py
python3 kernel_workflow/scripts/kernel_tools/probe.py measure --dir ir/candidate --json ir/candidate/probe.json
```

Run the compile command in the workload environment. If that command also
executes a GPU workload, run it under the existing GEAK GPU lease/lock. The dump
helper preserves all arguments after `--`, including options such as `--arch`
that also exist on the wrapper, and does not turn a benchmark into a
compile-only operation. Pin one autotune configuration when inspecting its IR.
Optional `--emit-gluon` requires the Gluon pack's `recover_gluon.py`; plain IR
inspection and occupancy analysis do not require Gluon.

## Data maintenance and validation

The databases live in [perf_knowledge/hardware/data](../../../perf_knowledge/hardware/data).
They retain source release/page information and verification limitations. In
particular, new CDNA4 low-precision and sparse layouts retain their recorded
caveats; a successful lookup is not GPU numerical validation.

When updating committed ISA data:

1. Record the upstream document/XML release, URL, page or instruction identifier,
   license, retrieval date and verification scope with the data change.
2. Update the appropriate database and the target/provider declarations in
   `amd_targets.py`. Keep encoding, matrix rules and prose facts separate.
3. Run the offline runtime selftests and source/grid regressions in the CLI suites
   below. For calculator-backed
   targets, run actual facts/layout/locate queries using the pinned dependency.
4. Preserve known errors and unverified low-precision/sparse rules. A cross-check
   against CDNA3 does not validate new CDNA4 behavior; use numerical GPU tests
   before relying on those layouts for correctness.

A complete automatic XML/PDF-to-database rebuild pipeline is not included.
Database updates therefore require review of the source diff and per-record
provenance, not an assumption that a generator reproduced the files.
The evidence fixture retains the original layout-file hash and all relocated
records. Changes to a source row should remain visible as data changes; do not
minify or compress the fixture to hide its review surface.

Run the CPU regression suites from the repository root:

```bash
python3 -m pytest -q kernel_workflow/scripts/tests/test_kernel_tools_isa.py kernel_workflow/scripts/tests/test_isa_workflow_boundaries.py kernel_workflow/scripts/tests/test_shared_amd_tools.py kernel_workflow/scripts/tests/test_matrix_dependency.py
```

For a toolchain-specific VGPR allocation check, use `amd_occupancy.py
--compiler-sweep --arch gfx942 --format json`. This compiles LLVM IR and reads
LLVM's occupancy output; it does not launch a kernel or measure achieved occupancy.

ROCm LLVM 22 sweeps cover all 256 clobbered-VGPR counts on each of gfx942, gfx950
and gfx1201. Target-explicit compilation with Triton 3.5.1 (ROCm 7.2 image) and
3.7.0 and 3.8.0 also exercises IR capture and metadata association for Triton/Gluon vector
addition, Triton matrix multiplication on all three targets, and Gluon MFMA on
gfx942/gfx950, using four and eight warps. These are compiler checks, not GPU
correctness or performance claims. Gluon source APIs vary by compiler version;
for example, `AMDMFMALayout.instr_shape` is `[M, N]` in the tested 3.5.1 build
and `[M, N, K]` in 3.7.0/3.8.0.

## Dependencies and supported architecture facts

| Query | gfx950 / CDNA4 | gfx942 / CDNA3 | gfx1201 / RDNA4 |
| --- | --- | --- | --- |
| Instruction encoding | Committed database | Committed database | Committed database |
| Matrix facts/layout/locate | Committed database and rules | Optional AMD calculator | Optional AMD calculator |
| Structured register/waitcnt/memory prose | Committed facts, with caveats | Not extracted | Not extracted |
| Static occupancy | Registers/LDS, subject to launch data | Registers/LDS, subject to launch data | Registers; LDS scope unverified |

Use [dependencies.md](dependencies.md) for pinned calculator preparation, offline
installation, interpreter selection and real query checks. ISA queries and IR
inspection do **not** require rocprofv3, rocprof-compute or ATT. Optional
`hw_sources.sh decode` uses **isa_spec_cli for static instruction bytes**; it is
not an ATT trace decoder. Runtime profiling and ATT environment checks belong to
the separate profiling integration (PR4), with SDK/decoder packages supplied by
the workload image.

## Coverage limits

Encoding queries support gfx942/CDNA3, gfx950/CDNA4 and gfx1201/RDNA4. Matrix
layout queries use their declared providers; calculator availability is checked
at invocation. A supported query is not a hardware numerical validation claim.
Structured register/waitcnt/memory prose facts for gfx942/gfx1201 are not yet
extracted; capabilities reports this explicitly. RDNA LDS occupancy requires
verified WGP/CU scope handling and remains withheld. No gfx950 default is used
to fill either gap. Shared GPU measurements remain owned by GEAK profiling and
TraceLens, not by these static bounds.

## Profiling integration (PR4)

Container execution lives in shared `locus.sh`; the historical Gluon path sources
that implementation. `TILE_KERNEL_*` environment names remain compatible. GEAK
embedded runs use their existing workload container and GPU lease. The complete
GEAK tools directory must be mounted for IR collection; `capture.sh` no longer
copies a shell script without its Python dependencies. Capture preserves the
application argv after `--`, pins the requested kernel, and archives old evidence
before collecting again. Non-Triton applications may report the static IR layer
unavailable; this does not invent Triton IR for HIP or replace their build tooling.

Profiling parsers, decoded ATT analysis, GPU guards and native roofline delegation
are shared. `profile_payload.py` and its schema are shared capture interfaces;
language packs supply payloads and keep their campaign records. An optionally installed TraceLens runner owns its roofline models and calibration;
GEAK retains benchmark acceptance;
static resource estimates are planning evidence and never a fallback roofline engine.
