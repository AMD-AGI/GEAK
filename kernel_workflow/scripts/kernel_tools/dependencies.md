# Optional AMD hardware query dependencies

GEAK's committed hardware and encoding databases require no calculator, GPU or
network. Matrix facts/layout/locate for gfx942 (CDNA3) and gfx1201 (RDNA4) use the
AMD Matrix Instruction Calculator. Gfx950 uses its committed layout database.
These are shared GEAK tools, usable from Triton, Gluon and HIP workflows.

## Prepare the calculator explicitly

Use Python 3.10+ with `pip`, `venv` and git. The installer creates a new prefix,
clones a fixed commit, checks source and license hashes, and installs the pinned
pure Python dependency into a private venv. It does not change workload Python,
Triton, PyTorch or an existing prefix. Standard-library venv without ensurepip is
sufficient: the caller's pip installs the wheel into the private site-packages.

From the GEAK repository root:

```bash
python3 kernel_workflow/scripts/kernel_tools/prepare_matrix.py \
  --prefix /abs/geak-tools
export TILE_HW_TOOLS=/abs/geak-tools
python3 kernel_workflow/scripts/kernel_tools/matrix_dependency.py
python3 kernel_workflow/scripts/kernel_tools/amd_isa.py \
  --arch gfx942 facts v_mfma_f32_16x16x16_f16
python3 kernel_workflow/scripts/kernel_tools/amd_isa.py \
  --arch gfx1201 layout v_wmma_f32_16x16x16_f16 -A
```

`matrix-install.json` records the selected source, hashes, status and actual CLI
checks. A failed install leaves a marked partial prefix for inspection. Choose
a new prefix for a retry or version upgrade; existing directories are refused.
Source files and the upstream MIT LICENSE stay together in the calculator clone.

The pins are [matrix-calculator.json](dependencies/matrix-calculator.json) and
[matrix-requirements.txt](dependencies/matrix-requirements.txt). Update them
together and verify real CDNA3/RDNA4 detail/layout/locate queries. The tested
calculator needs `tabulate`, not numpy; do not infer dependencies from unrelated
plotting tools. The Python support policy is for the GEAK query scripts as well
as the upstream calculator.

## Offline or image preparation

Prepare a local git checkout containing the pinned commit and a wheelhouse on a
networked build machine. Then run without network access:

```bash
python3 kernel_workflow/scripts/kernel_tools/prepare_matrix.py \
  --prefix /abs/geak-tools --offline \
  --source /abs/amd_matrix_instruction_calculator \
  --wheelhouse /abs/wheels
```

The same command can be used while building a workload image. It is an explicit
environment preparation step, never invoked by runtime queries or skill discovery.
To prepare the wheelhouse: `python3 -m pip download --only-binary=:all:
--require-hashes -r kernel_workflow/scripts/kernel_tools/dependencies/matrix-requirements.txt
--dest /abs/wheels` (join these lines into one command).

## Use existing assets

Lookup order remains `TILE_HW_SEED`, `TILE_HW_TOOLS`, then `TILE_HW_CACHE`, each
containing `amd_matrix_instruction_calculator/matrix_calculator.py`. For managed
prefixes, queries automatically use `matrix-venv/bin/python`. Otherwise they use
the query's Python; `GEAK_MATRIX_PYTHON=/abs/python` overrides it explicitly.
Both `amd_isa.py` and `hw_sources.sh layout` use this selection. A read-only layout
query needs no writable cache. XML/PDF/decode paths still use `TILE_HW_CACHE`.

`matrix_dependency.py` runs the actual version and CDNA3/RDNA4 queries, reports
the interpreter, source hash and whether it matches the pin. Compatible external
versions can run, but are labeled by their observed hash rather than the pin.
`hw_sources.sh preflight` reports the broader hardware-source/cache inventory;
`amd_isa.py capabilities` is a quick asset inventory, not an execution check.

When an external container executes the query, the assets and selected Python
must be present at container-visible paths. A host installation does not populate
the container. Prefer preparing the assets inside the image; mount the whole
prefix when sharing an ABI-compatible environment.

Other optional primary-source operations are independent: PDF rendering uses
Poppler's `pdftoppm`; raw ISA decode uses a prebuilt `isa_spec_cli` and the selected
XML; these are not prerequisites for committed encoding queries or matrix layout.
