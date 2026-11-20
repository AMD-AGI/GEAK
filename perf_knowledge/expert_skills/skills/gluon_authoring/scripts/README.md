# Gluon runtime tools

Start with [USAGE.md](USAGE.md) only after `toolctl` has supplied the current
`stage-context`. Use each selected tool's `--describe` and `--help` output as its callable
interface. Script-body reads are not part of normal execution.

For a concrete suspected defect, submit `source_read_request` with the question and symbol, then
inspect only the resulting `source_excerpt`.

## Shared AMD tools and Gluon adapters

Use `amd_isa.py --arch gfx942|gfx950|gfx1201 capabilities --json` to check
availability, then `encoding`, `facts`, `layout` or `locate`. This pack entry
forwards to GEAK's shared implementation. `gfx950_isa.py` retains its historical
default; `layout_facts.py` retains its legacy argument order. Neither duplicates
a database or calculator. Gfx942/gfx1201 structured ISA prose facts and the
gfx1201 LDS occupancy model remain unavailable.

Shared resource parsing (`amd_resources.py`) supplies probe, assembly audits,
ATT diagnostics and Gluon parity. `parity_gate.py` owns transcription decisions,
and `emit_gluon.py` owns the optional `dump_ir.sh --emit-gluon` adapter. Container
execution (`locus.sh`) and profiling parsers are shared GEAK tools.
