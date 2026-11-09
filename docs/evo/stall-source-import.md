# Stall-source import (default off)

`evolution_import_stall_sources: true` applies only to `optimizer: evo` with the default
`evolution_start: first_stall`. When the first non-improving round roots the continuation
archive at the incumbent, the retained sources of that round become ordinary archive rows.
Only that round's sources are imported, and only when their recorded base equals the new
archive base commit. Earlier rounds stay delivery-only, including a round whose failed commit
left its sources on the same base.

Each row is named by its engineer slot (`round_K/engineer_I` becomes `rK_dI`, the round's
integration becomes `rK_integrated`), uses the pinned delivery snapshot as its source, and has
`ORIGINAL` (the archive base) as parent. A settled verifier result keeps its verified fitness;
every other source, including a verification whose custody check never settled, is an
unverified row: selectable as a parent or donor, never ranked as best or default parent.
Empty patches are not imported. The retained delivery descriptor is renamed with the row so
end-of-run delivery records and selects that same row.

The first post-stall plan therefore sees these rows and the retained-approach policy. Rounds
before the stall are unchanged. This document is not injected into role prompts.
