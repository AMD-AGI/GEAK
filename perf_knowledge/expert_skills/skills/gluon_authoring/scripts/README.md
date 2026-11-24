# Gluon tools

The tool table by purpose, with commands, is [`skill.md ## Tools`](../skill.md#tools); every tool's
`--help` is its interface, and [USAGE.md](USAGE.md) has the long-form usage. Run from the GEAK root and
pass `--arch` explicitly.

## Shared AMD tools and Gluon adapters

Shared tools live in `kernel_workflow/scripts/kernel_tools/` and are called by that path; this
directory keeps only Gluon-specific tools and `probe.py` (legacy CLI with a gfx950 default for `plan`).
Use `kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx942|gfx950|gfx1201 capabilities --json`
to check ISA query availability, then `encoding`, `facts`, `layout` or `locate`. Gfx942/gfx1201
structured ISA prose facts and the gfx1201 LDS occupancy model remain unavailable.

Shared resource parsing (`amd_resources.py`) supplies probe, assembly audits and
ATT diagnostics; `emit_gluon.py` owns the optional `dump_ir.sh --emit-gluon` adapter.
Container execution (`locus.sh`) and profiling parsers are shared GEAK tools.

## Not used under GEAK

The upstream standalone workflow's record, gate and stage tools were removed; GEAK's rounds, records and
acceptance replace them.
