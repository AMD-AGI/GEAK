# Local intervention: kimi-d73d-dsa-single-pair-20261113

Origin: `kimi-k31-pro`. Kind: one local MI355X same-image toggle pair.

**Intervention:** same d73d image; VLLM_KIMI_SD_DSA off versus default auto/fused

**Fixed context from the original record:**

```json
{
  "gpu_type": "MI355X",
  "gpu_count": 8,
  "tp": 8,
  "dcp": 1,
  "ep": 1,
  "client_concurrency": 1,
  "admission": 1,
  "server_seed": 7,
  "client_seed": 42,
  "sending_window_seconds": 900,
  "draft_steps": 7,
  "synthetic_acceptance_length": 3.69
}
```

| Native metric | Observed change |
| --- | ---: |
| Total output throughput | -0.668187% |
| P90 full-response interactivity | +3.441998% |

Evidence: One sequential on/off pair; first-pass compilation/cache asymmetry, repetition planned.

- Same token volume; output difference comes from native record-span timing.
- TTFT tail differs; first reference created 5 Triton binaries during profile.
- JIT warning count is not compilation count or duration.
- Automatic KV pool differs by995 reported tokens.
- Host-native image execution; synthetic AL is not a quality evaluation.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

Related public/source case: [kimi-k31-pro/specialized-dsa](../../cases/kimi-k31-pro/specialized-dsa.md).

[Route evidence](../../routes/kimi-k31-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `kimi-d73d-dsa-single-pair-20261113`.
Original report `experiments/kimi/DSA_COMPARISON.md`; native reference `experiments/kimi/002-d73d-dsa-off-tp8-c1-15m/native/aggregate.json`;
treatment `experiments/kimi/001-d73d-dsa-fused-tp8-c1-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
