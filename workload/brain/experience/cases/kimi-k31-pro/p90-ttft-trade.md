# P90 ttft trade

Origin: `kimi-k31-pro`. Case: `kimi-k31-pro/p90-ttft-trade`. Kind: retrospective public/source case.

**Signal:** Prefill repeatedly interrupts the decode token stream.

**Historical action:** Evaluate admission and prefill interval under all declared serving objectives.

**Observed result and limit:** C32 P90 improves 40.54% and output 0.77%, while P90 TTFT grows from 4.70 to 12.20 seconds.

In the timed send windows, 35.09% of 1,439 TP0 decode samples had queued work
before the change and 69.51% of 1,745 afterwards. Observed active-width P90 fell
from 23 to 18. Admission, prefill cadence and verify depth all changed. The queue
and width observations support testing one policy at fixed compute.
[Before](https://inferencex.semianalysis.com/api/v1/reproduce?id=447679) · [After](https://inferencex.semianalysis.com/api/v1/reproduce?id=448018).

**Next discriminator:** Which waiting-time objective constrains the useful plan?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/kimi-k31-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `kimi-k31-pro/p90-ttft-trade`;
original chapter `reports/kimi-k31-pro/SGLANG_ROUTE.md`.
