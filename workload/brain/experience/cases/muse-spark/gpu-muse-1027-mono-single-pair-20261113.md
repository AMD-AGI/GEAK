# Local intervention: muse-1027-mono-single-pair-20261113

Origin: `muse-spark`. Kind: one local MI355X same-image toggle pair.

**Intervention:** Add --monokernel-backend amd at the same explicit 1027 policy

**Fixed context from the original record:**

```json
{
  "gpu_type": "MI355X",
  "gpu_count": 8,
  "tp": 8,
  "ep": 8,
  "client_concurrency": 8,
  "admission": 2,
  "server_seed": 42,
  "client_seed": 42,
  "dataset": "semianalysis_cc_traces_weka_062126",
  "sending_window_seconds": 900
}
```

| Native metric | Observed change |
| --- | ---: |
| Total output throughput | +18.708727% |
| P90 full-response interactivity | +18.130253% |

Evidence: same-image toggle, one sequential pair; treatment executed before reference.

- Not repeated or randomized.
- Changing the mono route includes its automatic memory/layout effects.
- Closed-loop replay completed different request sets; no individual kernel credits are inferred.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

Related public/source case: [muse-spark/large-fusion-memory](../../cases/muse-spark/large-fusion-memory.md).

[Route evidence](../../routes/muse-spark.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `muse-1027-mono-single-pair-20261113`.
Original report `experiments/muse/DECOMPOSITION.md`; native reference `experiments/muse/006-1027-no-mono-tp8-c8-15m/native/aggregate.json`;
treatment `experiments/muse/005-1027-tp8-c8-full-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
