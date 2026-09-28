# Archive — superseded / disproven cards

Cards evicted from the live index land here with their refuting evidence, so a future run can see "this
was tried and why it was dropped" without it polluting the active KB. Not loaded during a run.

Format: `- [date archived] <title> — REASON (superseded-by <slug> | disproven by <source>)`.

<!-- (empty — first curated edition 2026-06-18; the 44 legacy append-entries were distilled into the
     active cards, with NULLs/duplicates collapsed into confidence counters rather than archived) -->

- [2026-08-31] Every fusion phase needs a denominator, or the misses are invisible — SUPERSEDED BY CODE:
  the rule is machine-enforced by the four `scripts/fusion_*_harness.py` coverage gates and restated in
  the roles that must obey it (`roles/kernel_fusion_analyst.md`, `fusion_unit_validator.md`,
  `fusion_integrator.md`, `semantics_mapper.md`, `kernel_extractor.md`, `op_benchmarker.md`). A card is
  advice the box can overrule; a harness gate is not, so this did not belong in the KB.
  (was method-fusion-coverage-denominator.md, ★★★, 2026-08-26)
- [2026-08-31] A prefill/decode split capture means TWO traces must reach the table — SUPERSEDED BY ROLE:
  `roles/semantics_mapper.md` already carries `--table-phases all`, the both-traces requirement and the
  rows-without-shapes/eager-probe caveat. Duplicating it here made the INDEX line the stale copy.
  (was method-split-trace-two-phase.md, ★★, 2026-08-26)
