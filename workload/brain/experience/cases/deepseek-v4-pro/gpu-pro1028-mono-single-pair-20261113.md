# Local intervention: pro1028-mono-single-pair-20261113

Origin: `deepseek-v4-pro`. Kind: one local MI355X same-image toggle pair.

**Intervention:** Same SGL1028 image; enable-amd-monokernel for target61layers,width7; draft remains normal

**Fixed context from the original record:**

```json
{
  "gpu_type": "MI355X",
  "gpu_count": 8,
  "tp": 8,
  "ep": 1,
  "client_concurrency": 1,
  "admission": 2,
  "server_seed": 42,
  "client_seed": 42,
  "sending_window_seconds": 900,
  "harness_version": "1.0.13"
}
```

| Native metric | Observed change |
| --- | ---: |
| Total output throughput | +0.690802% |
| P90 full-response interactivity | -3.559959% |

Evidence: One sequential on/off pair; mixed median/tail result, repeat planned.

- Identical output volume; output ratio uses different native record spans.
- P90 worsened while median paired ITL improved.
- Automatic KV pool and monokernel layout memory differ.
- Synthetic AL and host-native execution do not establish model quality or exact historical CI equivalence.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

[Route evidence](../../routes/deepseek-v4-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `pro1028-mono-single-pair-20261113`.
Original report `experiments/deepseek-v4-pro/MONO_COMPARISON.md`; native reference `experiments/deepseek-v4-pro/002-sgl1028-no-mono-tp8-c1-15m/native/aggregate.json`;
treatment `experiments/deepseek-v4-pro/001-sgl1028-fused-tp8-c1-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
