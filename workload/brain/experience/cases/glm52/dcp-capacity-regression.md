# Dcp capacity regression

Origin: `glm52`. Case: `glm52/dcp-capacity-regression`. Kind: retrospective public/source case.

**Signal:** DCP enlarges logical KV capacity but adds communication and scheduling costs.

**Historical action:** Evaluate each workload regime under the actual resource count.

**Observed result and limit:** DCP4/C48 output 206.63048 and P90 2.2495 do not support predicting throughput from capacity multiples.

**Next discriminator:** Is capacity pressure, uncached prefill or DCP communication dominant?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/glm52.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `glm52/dcp-capacity-regression`;
original chapter `reports/glm52/EVIDENCE_NOTES.md`.
