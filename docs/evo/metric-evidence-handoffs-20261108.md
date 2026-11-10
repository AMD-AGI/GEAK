# Metric evidence handoffs

The optimizer already ranks candidates by the declared primary, but some role and history inputs used a geometric-mean label for that primary or exposed only a secondary geomean. This common GEAK/GEAK-EVO correction names the primary explicitly in planner, integrator, memory and history handoffs while preserving available legacy aliases. Missing score evidence stays `null` rather than becoming a measured zero.

`verified_primary` identifies the verifier-returned metric; it does not itself establish correctness, measurement validity or source custody. Existing verification status and source evidence retain their authority. External objectives retain their checked native reader and diagnostic fields.

The legacy population reader now requires positive finite weighted evidence when its declared objective is `time_weighted`. Its existing rejection path handles unavailable weighted evidence; it no longer substitutes geomean. Geomean and external-primary objectives retain their existing paths.

This branch starts from `b403ba93200fb29862c35a0ee37b345e087c5fd4` and preserves its first-round engineer-parity behavior and archive source-identity fields. The existing build script regenerates the single-file workflow from its source parts.

Budget units are unchanged: `BUDGET_REMAINING` is direction-cost credits (`BUDGET - dispatched`), whereas `MINUTES_REMAINING` is a rounded wall-time snapshot. Neither is a count of successful candidates or an engineer-only time guarantee.
