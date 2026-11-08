# Metric evidence handoffs

The optimizer already ranks candidates by the declared primary, but some role and history inputs used a geometric-mean label for that primary or exposed only a secondary geomean. This common GEAK/GEAK-EVO correction names the primary explicitly in planner, integrator, memory and history handoffs while preserving available legacy aliases. Missing score evidence stays `null` rather than becoming a measured zero.

`verified_primary` identifies the verifier-returned metric; it does not itself establish correctness, measurement validity or source custody. Existing verification status and source evidence retain their authority. External objectives retain their checked native reader and diagnostic fields.

The legacy population reader now requires positive finite weighted evidence when its declared objective is `time_weighted`. Its existing rejection path handles unavailable weighted evidence; it no longer substitutes geomean. Geomean and external-primary objectives retain their existing paths.

This branch starts from `b403ba93200fb29862c35a0ee37b345e087c5fd4` and preserves its first-round engineer-parity behavior and archive source-identity fields. The existing build script regenerates the single-file workflow from its source parts.

Review evidence:

- [Delta 01 revision 02 and source review](/home/sapmajum/research-collab/codex/evo_direction_causal_audit_20261108_01/prospective_metric_handoffs_20261108_02/READOUT.md), patch SHA256 `4223fd099db63b86584e7a3a715bb7dd38db3df4657a06ff40ba8f8983d9c2bd`.
- [Delta 02 source review](/home/sapmajum/research-collab/codex/evo_direction_causal_audit_20261108_01/prospective_metric_handoffs_20261108_01/legacy_population/SOURCE_REVIEW_02.md), patch SHA256 `312248c01100230904631e3a0be446a09aa978a6712614453a006dc437decd9f`.

Budget units are unchanged: `BUDGET_REMAINING` is direction-cost credits (`BUDGET - dispatched`), whereas `MINUTES_REMAINING` is a rounded wall-time snapshot. Neither is a count of successful candidates or an engineer-only time guarantee.

This is prospective source integration. Syntax and generated-source consistency checks do not establish runtime behavior, improved search, comparative superiority or workload benefit. Final review of the diff against its parent remains separate. Existing frozen experiments and active runtimes are unchanged.
