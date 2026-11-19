# Local intervention: deepseek-v41-flash-4fab-mono-single-pair-20261114

Origin: `deepseek-v41-flash`. Kind: one local MI355X same-image toggle pair.

**Intervention:** Add --monokernel to activate eligible target and DSpark draft mono paths together

**Fixed context from the original record:**

```json
{
  "gpu_type": "MI355X",
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "tp": 4,
  "ep": 1,
  "client_concurrency": 1,
  "admission": 1,
  "server_seed_resolved": 0,
  "client_seed": 42,
  "dataset": "semianalysis_cc_traces_weka_062126",
  "sending_window_seconds": 900,
  "engram_cpu_offload": false,
  "speculative_tokens": 5,
  "synthetic_acceptance_length": 3.51,
  "local_harness_version": "1.0.13"
}
```

| Native metric | Observed change |
| --- | ---: |
| Total output throughput | +31.001661% |
| P90 full-response interactivity | +55.025598% |

Evidence: same-image toggle; one sequential pair, reference executed before treatment.

- Target and draft effects not separated.
- Not repeated or randomized; compilation cache/order remain possible influences.
- 126 shared logical requests have identical output lengths; treatment completes21 additional requests.
- Includes automatic memory/layout effects.
- Image userspace with host glibc differs from official container execution.
- Both legs have one matched warmup content error, zero profiling errors.
- Synthetic AL is not a quality evaluation.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `deepseek-v41-flash-4fab-mono-single-pair-20261114`.
Original report `experiments/deepseek-v41-flash/MONO_COMPARISON.md`; native reference `experiments/deepseek-v41-flash/002-4fab-no-mono-tp4-c1-15m/native/aggregate.json`;
treatment `experiments/deepseek-v41-flash/001-4fab-mono-tp4-c1-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
