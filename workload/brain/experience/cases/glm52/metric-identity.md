# Metric identity

Origin: `glm52`. Case: `glm52/metric-identity`. Kind: retrospective public/source case.

**Signal:** A claimed throughput improvement uses a different metric from total output.

**Historical action:** Compare input-plus-output per GPU separately from total output and native P90.

**Observed result and limit:** Total output moves from 326.96976 to 226.54701 while P90 rises slightly; the reported gain is not an output-throughput gain.

**Next discriminator:** Does fixed request composition distinguish processing progress from a real speed change?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/glm52.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `glm52/metric-identity`;
original chapter `reports/glm52/EVIDENCE_NOTES.md`.
