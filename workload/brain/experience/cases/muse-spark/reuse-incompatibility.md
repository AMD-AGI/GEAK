# Reuse incompatibility

Origin: `muse-spark`. Case: `muse-spark/reuse-incompatibility`. Kind: retrospective public/source case.

**Signal:** Draft sliding-window storage repeats state, but different implementations use different slots.

**Historical action:** Reuse target SWA slots only on the compatible ordinary path.

**Observed result and limit:** The mono route does not use this reuse; the two benefits cannot simply be added.

**Next discriminator:** Which state and cache assumptions remain valid after changing the route?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/muse-spark.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `muse-spark/reuse-incompatibility`;
original chapter `reports/muse-spark/1027.md`.
