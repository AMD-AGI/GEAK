# Local intervention: glm52-1028-indexer-single-pair-20261114

Origin: `glm52`. Kind: one local MI355X same-image toggle pair.

**Intervention:** Same SGL1028 image; enable fused DSA indexer path, including its Hadamard/prefill-writer/norm/workspace linkage

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
| Total output throughput | -0.592646% |
| P90 full-response interactivity | +2.064440% |

Evidence: one sequential on/off pair, modest decode gain with TTFT tradeoff; source counting bug separately audited.

- Native OSL overcount differs across43/39 requests.
- Both groups produce92152actual completion tokens; native lengths match80/106.
- TTFT effect not separated from format/writer linkage and first-pass cache/order.
- Synthetic AL not a quality evaluation.

**Next discriminator:** Repeat under matched state, verify actual route coverage and
separate remaining format/memory/scheduling factors before transferring the gain.
Complete implementation and adoption cost for a new workload is unknown.

Related public/source case: [glm52/fused-indexer-dependency](../../cases/glm52/fused-indexer-dependency.md).

[Route evidence](../../routes/glm52.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/gpu-attributions.jsonl`, record `glm52-1028-indexer-single-pair-20261114`.
Original report `experiments/glm52/INDEXER_COMPARISON.md`; native reference `experiments/glm52/002-sgl1028-no-indexer-fusion-tp8-c1-15m/native/aggregate.json`;
treatment `experiments/glm52/001-sgl1028-fused-tp8-c1-15m/native/aggregate.json`. These are local screening results, not a model-quality pass.
