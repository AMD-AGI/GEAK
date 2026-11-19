# Communicator memory

Origin: `kimi-k31-pro`. Case: `kimi-k31-pro/communicator-memory`. Kind: retrospective public/source case.

**Signal:** TP and DCP use identical rank groups but allocate duplicate communicators.

**Historical action:** Share compatible communicator resources and inspect derived host pools again.

**Observed result and limit:** Released device memory can trigger host/GTT over-allocation; the later host ratio is reduced.

**Next discriminator:** How do device and host capacity jointly constrain the best configuration?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/kimi-k31-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `kimi-k31-pro/communicator-memory`;
original chapter `reports/kimi-k31-pro/SGLANG_ROUTE.md`.
