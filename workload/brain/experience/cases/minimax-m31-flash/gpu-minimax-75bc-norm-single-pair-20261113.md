# Local intervention: minimax-75bc-norm-single-pair-20261113

Origin: `minimax-m31-flash`. Kind: one local MI355X same-image toggle pair.

**Intervention:** Same 75bc image; enable amd_kernels norm route versus disabled_routes norm

**Fixed context from the original record:**

```json
{
  "gpu_type": "MI355X",
  "gpu_count": 4,
  "tp": 4,
  "ep": 1,
  "client_concurrency": 16,
  "admission": 32,
  "server_seed": 0,
  "client_seed": 42,
  "sending_window_seconds": 900
}
```

| Native metric | Observed change |
| --- | ---: |
| Total output throughput | +0.942490% |
| P90 full-response interactivity | +10.982442% |

Evidence: One sequential on/off pair; replication pending.

- Small output difference is not established as a stable improvement.
- KV capacity changes automatically with route memory.
- Two matching profiling content errors in each leg; native validity true.
- Same host-native image mode, different from official container libc.
- Synthetic acceptance is not a model-quality evaluation.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

[Route evidence](../../routes/minimax-m31-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `minimax-75bc-norm-single-pair-20261113`.
Original report `experiments/minimax/NORM_COMPARISON.md`; native reference `experiments/minimax/002-75bc-no-norm-tp4-c16-15m/native/aggregate.json`;
treatment `experiments/minimax/001-75bc-full-tp4-c16-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
