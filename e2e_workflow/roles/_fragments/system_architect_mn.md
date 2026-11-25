# System Architect — multi-node serving

Read this only when `SERVING_TOPOLOGY` is in your inputs. A `tp_span` run needs nothing from this
file: its Top-N is one engine's, read exactly as the role file says.

## PD (`SERVING_TOPOLOGY=pd_disagg`)

**PHASE=strategize step 1, when the Top-N has a top-level `per_engine` block and an `engine` on every
row.** `pct_gpu_time` is the kernel's share of all the GPU time one request costs across both engines —
the same meaning as on one node — so rank the merged table as a single table, exactly like a
single-node one. Around it:
- **One candidate per kernel, across engines.** Both engines run one code tree, so the same kernel
  on both is ONE candidate whose cases span both engines' shapes (chunk-sized M and batch-sized M);
  its `pct_gpu_time` is the sum of its rows. Its `engine` is the row with the larger share — it names
  where an env/flag winner binds and which log proves engagement, never where code goes. (The
  candidate's `role` key names the seam kind, not the engine.)
- Rank on `pct_gpu_time` and use it as the ceiling for `plausible_speedup`; explain with
  `pct_gpu_time_in_engine` too.
- **Bottleneck:** quote `bottleneck_engine` and each `per_engine[].weight_pct`.
- **Tag each `config_directions` entry with `engine`** (`prefill`: chunk / batched-token budget;
  `decode`: running batch, decode graph shape). `""` only when the same value is right on both — the
  Tuner applies that string to both machines.
- **Read config directions off each engine's own table** (`PROFILE_TOPN_BY_ROLE` or
  `per_engine[].topN_json`); the merged 25 rows can drop the smaller engine entirely. Heads still come
  from the merged table. Every profiled engine needs at least one direction; there is no re-ask, so
  if an engine truly has none, say why in `regime_summary` (it is reported as a gap).

**Return JSON.** Every `config_directions`, `head_candidates` and `kernel_candidates` entry also
carries `"engine"`: for a direction, `prefill|decode`, or `""` when the same value belongs on both
engines; for a candidate, the `engine` of its larger Top-N row (`prefill|decode`), `""` when the
Top-N has none.
