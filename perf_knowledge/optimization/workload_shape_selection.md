---
title: workload shape selection — choosing the shape set to tune for agentic serving
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4, int4]
regimes: [prefill, decode, both]
updated: 2026-11-13
sources:
  - https://inferencex.semianalysis.com/agentx/methodology
  - https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
---

# workload shape selection (agentic / multi-turn serving)

## TL;DR
Before tuning, decide **which shapes to tune for**. On a fixed ISL/OSL benchmark one M often carries most of the GPU time, so "capture the live `(M,N,K)` and tune it"
(`[[optimization/autotuning_methodology.md]]`, `[[workflows/gemm_tuning_workflow.md]]`) is sound. On an **agentic multi-turn** workload it is not: shapes are a distribution, and which part of it you see depends on the serving config you happened to profile. A profile gives you N and K reliably; it gives a set of sampled values of  **M and N_KV** from the steady-state. While these values are sampled uniformly from the distribution, the sample size might not be statistically significant. Take N/K from the profile, take the **range** of M and N_KV from the rules below to be aligned with the provided shapes from the profiled region, and tune across that range. For example, measured on one agentic run, **4 distinct M values carry 50% of GPU time, 14 carry 90%, and 285 are needed for 99%** — so a handful of shapes is enough to matter, but the untuned fallback path must stay sane.

## Concepts

- **Prompt length ≠ prefill work.** Multi-turn agents resend the whole conversation each turn, so
  prompts are enormous but almost entirely prefix-cached. On the AgentX v1.0 1M corpus (393 sessions,
  98,827 requests) the median request presents **142,016 prompt tokens and prefills 1,600 of them**
  (98.3% corpus-wide hit rate with ideal KV cache). Prompt length sets **N_KV**; the cache-missing suffix sets **prefill M**.
  They differ by ~90× at the median.
- **Session-tree concurrency.** The client's concurrency setting counts *sessions*, not in-flight
  requests. Sub-agents fan out without taking their own slot, so in-flight requests can be higher that the
  configured concurrency. Typical range is **0.5x to 2x** of the concurrency. 
- **Step class.** One engine step is decode-only, prefill-only, or mixed (chunked prefill packed with
  decodes — `[[operators/chunked_prefill/overview.md]]`). The mix is a config outcome, not a constant.
- **GEMM M vs attention shape.** Linear layers see one GEMM with `M` = the whole step's token count.
  Attention sees per-request `(N_Q, N_KV)`. 

## The levers (config → shape)

Ranges below come from a dependency-only simulation over the AgentX 1M corpus, cross-checked against a
measured run. Unless a table says otherwise client idle delay is set to zero, which
keeps the engine backlogged and makes these **upper bounds on batch occupancy** — real runs sit at or
below them. Lever 1 quantifies how far below by replaying the corpus's own recorded think time.

**1. Decode M is driven by concurrency and capped by `max_num_seqs`** — whichever binds first. There is
no useful single multiplier from concurrency to M: the ratio is not constant across the range, and the
distribution is right-skewed, so its mean and median disagree.

Concurrency determines how many sessions will be scheduled in parallel. However, this does not necessarily mean that a decode-only step will have atleast one request from each session as a session might be waiting through client-side delay.  Each session can send more than one request at a step. `max_num_seq` determines the upper cap on how many requests can be packed together. 

The cap is one knob under two names:

| engine | flag | note |
|---|---|---|
| vLLM | `--max-num-seqs` | `SchedulerConfig` default 128, but resolved per usage context in `EngineArgs.create_engine_config` — read the value the engine logs, not the class default |
| ATOM | `--max-num-seqs` | same name, **default 512** |
| SGLang | `--max-running-requests` | unset by default, derived as `min(max_total_num_tokens // 2, req_to_token_pool size)`, so it lands wherever the KV pool puts it; divided by `dp_size` under `--enable-dp-attention`. Not `--max-queued-requests`, which bounds admission rather than the batch |

On SGLang especially, the cap that binds is often not the number you configured, so read the effective
value off the startup log before concluding a shape is cap-pinned.

How much that delay matters is itself a choice, so measure both ends of it. The first table deletes
client-side delay entirely — every session is always ready to submit — and is therefore the occupancy ceiling, not a prediction.

| concurrency | mean | p10–p90 | max | three most common M (share of decode steps) |
|---|--:|---|--:|---|
| 8 | 9.0 | 8–11 | 18 | 8 (59%), 9 (19%), 10 (8%) |
| 16 | 19.4 | 16–22 | 64 | 16 (34%), 17 (17%), 18 (11%) |
| 32 | 37.9 | 32–44 | 114 | 32 (20%), 35 (11%), 33 (10%) |
| 64 | 73.4 | 66–82 | 148 | 71 (9%), 68 (9%), 70 (9%) |
| 128 | 148.1 | 136–162 | 237 | 148 (6%), 140 (6%), 139 (6%) |
| 256 | 288.0 | 271–307 | 385 | 282 (5%), 277 (4%), 283 (4%) |

The second replays each request's recorded `think_time` instead, clipped to 10s and charged as
`ceil(think / 100ms)` waiting iterations, so a session that thinks for 2.3s sits out 23 scheduler steps.
Note that 100ms is a heuristic here, just as an exmaple of how client side delay impacts the shape distribution. 

| concurrency | mean | p10–p90 | max | three most common M (share of decode steps) |
|---|--:|---|--:|---|
| 8 | 8.7 | 7–11 | 18 | 8 (46%), 9 (16%), 7 (15%) |
| 16 | 18.5 | 15–22 | 64 | 16 (25%), 15 (17%), 17 (13%) |
| 32 | 36.4 | 30–44 | 114 | 32 (13%), 31 (12%), 33 (8%) |
| 64 | 70.4 | 63–79 | 147 | 66 (8%), 65 (8%), 67 (8%) |
| 128 | 139.6 | 128–149 | 237 | 131 (5%), 139 (4%), 138 (4%) |
| 256 | 275.7 | 256–298 | 380 | 268 (3%), 269 (3%), 270 (3%) |

What idle does is widen the band downwards (p10 at concurrency 32 falls 32 → 30) and flatten the mode
further. So the two tables are the same shape target, and a deployment measuring far below the second
one is arrival-starved rather than shape-limited.

Across the range the modal share collapses from **59% to 5%**: at concurrency 8 one tuned M covers over
half of all decode steps, while at 256 the top three values together cover barely an eighth, over a
p10–p90 band 36 values wide with a tail running 25% past p90. The value of tuning any single decode M
therefore falls as concurrency rises — above ~32 it is a band of shapes that needs to be tuned. 

**2. Concurrency sets the step-class mix.** Decode-only steps fall 99.0% → 83.9% → 69.5% as
concurrency goes 8 → 128 → 256; prefill's share of steps grows ~30× across that range. A profile taken
at low concurrency contains almost no prefill shapes. Note that the contribution to e2e runtime is not proportional to the number of steps since prefill-decode mix or prefill-only steps take more time than decode-only steps.  
Prefill-decode mix or prefill-only step typically have 1 prefill request. With DES based on no client delay, in 85–99% of cases, mean 1.01–1.21 prefill requests are packed in a prefill-decode mix or prefill-only steps. It weakens slightly with concurrency (99.1% at conc 8 → 84.7% at 256) but ≤2 stays above 97% everywhere. 

**3. The per-step token budget chunks prefill and does not touch decode M.** Same knob, three names:

| engine | flag | note |
|---|---|---|
| vLLM | `--max-num-batched-tokens` | needs `--enable-chunked-prefill` on pre-V1 builds |
| ATOM | `--max-num-batched-tokens` | same name, **default 16384**; `--no-enable_chunked_prefill` disables |
| SGLang | `--chunked-prefill-size` | effective budget is `min(chunked_prefill_size, max_prefill_tokens)`, and is divided by `dp_size` under `--enable-dp-attention` — set both flags together |

Decode M p50 is not impacted by the chunks size and remains similar across 8k/32k/64k. Prefill M p90 is limited by the chunk size for chunk sizes till ~8k. However, for larger chunk sizes, the prefill p90 (M) is limited by the request as majority of the context size is cached. 

**4. N_KV is set by the corpus/context cap, not by serving config.** Essentially unmoved by
concurrency, `max_num_seqs` or the budget:

Here, N_KV is per request and the total N_KV for a step depends upon the number  of requests batched together. 
| corpus variant | decode N_KV p50 | p90 |
|---|---:|---:|
| AgentX v1.0 full (1M cap) | 228,352 | 637,952 |
| AgentX v1.0 256k | 134,144 | 227,328 |

On the 1M corpus, **roughly 70% of all decode work attends more than 128k tokens**. If the stack shards
KV across ranks — decode context parallel, `[[operators/context_parallel_attention/overview.md]]` —
divide by the shard count for the extent a kernel actually walks, and read the factor off the profile
rather than assuming. `max_model_len` and `max_context_len` act as a request-level and session-level filter, respectively.

**5. Prefix caching dominates every other knob — but it moves the *mix*, not decode M.** 

The prefil caching and hit rate determines the prefill M as well as the number of decode-only to prefill-mixed steps. For example,  
Disabling prefilx caching entirely moves prefill M p50 from 1,920 to 8,152 (8192 is chunk limit) and
takes decode-only steps from 92.2% to **0.0%**. Degrading it
partially does the same thing in miniature: dropping the hit rate from the corpus-native 98.3% to 90% reduces decode-only step by 15% and pushes prefill chunks to the budget ceiling, while leaving the
decode-M histogram essentially untouched. Hit rate is a *step-mix* lever; concurrency is the M lever.

**6. Speculative decoding multiplies decode M.** With draft length *k*, each decode request contributes
up to *k*+1 query tokens per step, moving decode attention from a GEMV-like regime toward a small-`N_Q`
GEMM (`[[operators/speculative_decode_verify/overview.md]]`).

**7. CUDA graph capture re-quantises decode M after the scheduler is done with it.** A decode batch is
padded up to the nearest captured size, so the M a kernel actually runs is drawn from the capture list,
not from the scheduler's batch:

| engine | flag | note |
|---|---|---|
| vLLM | `cuda_graph_sizes` / `cudagraph_capture_sizes` (compilation config) | a single value *N* expands to `[1,2,4,8] + range(16, N+1, 16)`; `--enforce-eager` disables capture |
| ATOM | `--cudagraph-capture-sizes` | **default `[1,2,4,8,16,32,48,64,128,256]`**; same expansion rule from `cuda_graph_sizes` |
| SGLang | `--cuda-graph-max-bs-decode`, `--cuda-graph-bs-decode` | batches above the captured maximum fall back to the eager path instead of padding |

This cuts both ways. It is why a measured decode batch can take only a handful of distinct values while
the scheduler's own M is spread across dozens — on one SGLang run capped at 20 running requests, the
traced decode batch was only ever 12, 14 or 16. It also means the default ladders, powers of two then a
step of 16, round away exactly the non-power-of-two spread that lever 1 describes. So the capture list
is itself a tuning target: match it to the p10–p90 band for your concurrency rather than leaving the
default, and read M off the kernel trace rather than the scheduler log.

### Shape histograms

Percentiles hide multi-modality, so these are the full distributions on **one shared set of bins** —
rows are directly comparable across every table below. Cells are % of steps (decode/prefill M) or % of
decode-step-requests (N_KV); `·` means below 0.05%. Idle-zero DES with session recycling, 400k steps,
seed 42; baseline config is concurrency 64, budget 8k, `max_num_seqs` 256 unless the row says otherwise.

**Decode M vs concurrency** (`max_num_seqs` deliberately non-binding at 4× concurrency):

| concurrency | 5–8 | 9–16 | 17–32 | 33–64 | 65–128 | 129–256 | 257–512 | p50 | decode-only % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 58.6 | 40.6 | 0.8 | · | · | · | · | 8 | 99.0 |
| 32 | · | · | 19.6 | 79.1 | 1.3 | · | · | 36 | 95.6 |
| 64 | · | · | · | 2.8 | 96.5 | 0.7 | · | 71 | 92.2 |
| 128 | · | · | · | · | · | 100.0 | · | 145 | 83.9 |
| 256 | · | · | · | · | · | · | 100.0 | 283 | 69.5 |

Read this table against lever 1 rather than instead of it. Concentration *rises* with concurrency here —
at 8 the distribution straddles two bins 59/41, while at 128 and 256 a single bin holds every decode
step — but that is the bin edges doubling faster than the band widens, not the batch settling down. In
absolute terms the band is getting wider, so a row that sits 100% in one bin is still not evidence that
one tuned M will cover it.

**Decode M vs `max_num_seqs`** (concurrency fixed at 128):

| max_num_seqs | 5–8 | 9–16 | 17–32 | 33–64 | 65–128 | 129–256 | 257–512 | p50 | decode-only % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | · | · | 100.0 | · | · | · | · | 32 | 96.5 |
| 64 | · | · | · | 100.0 | · | · | · | 64 | 92.8 |
| 128 | · | · | · | · | 100.0 | · | · | 128 | 85.7 |
| 256 (non-binding) | · | · | · | · | · | 100.0 | · | 145 | 83.9 |

A binding cap collapses the distribution to a single value in **100%** of decode steps. This is the one
configuration where tuning exactly one decode M is not a gamble.

**Decode N_KV vs corpus context cap** (the two AgentX v1.0 variants, identical bins):

| corpus | 0–4k | 4k–16k | 16k–32k | 32k–64k | 64k–128k | 128k–256k | 256k–512k | ≥512k | p50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1M cap | 4.3 | 1.7 | 1.5 | 8.5 | 13.9 | 24.5 | 28.0 | 17.5 | 228,352 |
| 256k cap | 7.5 | 2.7 | 2.7 | 13.6 | 22.3 | 51.2 | · | · | 134,144 |

Both corpora put three-quarters or more of decode work beyond 64k. The cap does not rescale the
distribution, it **truncates** it: the 256k variant piles 51.2% into one bin against the cap and has no
mass above it, while the 1M variant puts 45% above 256k. Decode M is unaffected by the swap (~97% in
65–128 either way) — corpus changes N_KV only.

**Prefill chunk M vs the per-step token budget:**

| budget | 1–256 | 257–1k | 1k–2k | 2k–4k | 4k–8k | 8k–16k | 16k–32k | ≥32k | p50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8k | 10.0 | 22.3 | 20.5 | 22.5 | 24.6 | · | · | · | 1,920 |
| 32k | 11.2 | 24.8 | 22.9 | 24.6 | 8.5 | 3.7 | 4.3 | · | 1,664 |
| 64k | 11.4 | 25.2 | 23.2 | 25.0 | 8.5 | 3.6 | 1.4 | 1.7 | 1,664 |

An 8× budget increase moves ~7% of chunks above 8k and **lowers** the median. Roughly 75–85% of chunks
sit under 4k whatever the budget, because with prefix caching most requests simply do not have 8k of
uncached suffix to contribute. The budget sets where the distribution is *clipped*, not where its mass
is. The decode-M histogram is bit-identical across all three budgets (96.5% in 65–128, p50 71).

**Prefill chunk M vs prefix-cache hit rate** (the hit rate is degraded synthetically; the achieved
corpus-wide rate is shown because the model floors each request at its genuinely-new tokens):

| target hit rate | 1–256 | 257–1k | 1k–2k | 2k–4k | 4k–8k | p50 | achieved | decode-only % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 90% | 1.0 | 3.1 | 3.8 | 9.4 | 82.8 | 8,117 | 0.893 | 76.9 |
| 92% | 1.1 | 3.3 | 4.1 | 11.9 | 79.6 | 8,116 | 0.913 | 80.5 |
| 94% | 1.3 | 4.0 | 5.5 | 14.8 | 74.5 | 8,112 | 0.932 | 84.0 |
| 96% | 1.3 | 4.2 | 8.2 | 19.2 | 67.0 | 6,866 | 0.951 | 87.5 |
| 98.3% (corpus native) | 10.0 | 22.3 | 20.5 | 22.5 | 24.6 | 1,920 | 0.983 | 92.2 |

The last 2 points of hit rate are worth more than the previous 8: going 98.3% → 96% multiplies prefill
chunk p50 by 3.6×, while 96% → 90% adds only a further 1.2×. Across this whole range the decode-M
histogram does not move (~96.6% in 65–128 at every hit rate) — what changes is how *often* a decode-only
step happens, falling 92.2% → 76.9%. So a cache regression does not invalidate your decode shapes; it
changes how much they matter and saturates every prefill chunk against the budget.

### Deriving the sweep set

- **Decode GEMM / GEMV** (`[[operators/skinny_gemv_decode/overview.md]]`,
  `[[operators/fused_moe_grouped_gemm/overview.md]]`): take the decode-M histogram row for your
  concurrency and `max_num_seqs`, sweep the occupied bins including non-powers-of-two, times
  (draft length + 1).
- **Prefill GEMM**: the observed quantised chunk values up to the observed cap, plus scattered M in
  100–2,000 for request tails.
- **Prefill attention** (`[[operators/attention_prefill_fmha/overview.md]]`): tune `(N_Q, N_KV)` pairs,
  never square configs. The query is the uncached suffix while keys span the whole conversation, so the
  ratio runs **50–90×** — representative 1M-corpus pairs at an 8k budget are (0.6k, 41k), (1.6k, 110k),
  (8k, 429k), (8k, 729k).
- **Decode attention** (`[[operators/attention_decode_paged/overview.md]]`,
  `[[operators/mla_attention/overview.md]]`): sweep N_KV {32k, 64k, 128k, 256k, 512k}, divided by the KV
  shard count; batch from the decode rule.
- **Weighting**: score a time-weighted geomean across the set using GPU-time share, not step count. On
  the measured run the two disagree by more than an order of magnitude in the tail — prefill steps were
  1.3% of steps but 21.3% of GPU time.

### Operator cards carrying these ranges

The `## Shape regimes` sections of these four cards previously described chat-style serving only and
now state the agentic ranges as well. Go to the card for the per-operator detail; come back here for
how the range moves with config.

| card | what its Shape regimes section now covers |
|---|---|
| `[[operators/attention_decode_paged/overview.md]]` | N_KV beyond the old 1k..128k envelope, and the two corrections (KV shard count, `sq = draft_len+1`) to apply before reading a profile |
| `[[operators/skinny_gemv_decode/overview.md]]` | the decode-M rule, the measured ladder, and the crossover to dense GEMM above ~M=64 |
| `[[operators/chunked_prefill/overview.md]]` | quantised chunk values, the observed-vs-configured budget gap, and the `N_KV/N_Q` asymmetry |
| `[[operators/context_parallel_attention/overview.md]]` | decode CP as a first-class path, its constraints, and why its shape and comm pattern differ from prefill CP |

## Pitfalls

- **Treating one profile window as the workload.** It fixes N and K; it samples M and N_KV. The whole
  point of this doc.
- **Tuning the configured budget instead of the observed chunk.** 8192 configured, 7,680 observed.
- **Assuming powers of two.** Real decode batches are 71, 72, 77 — not 64 or 128.
- **Letting a tuned kernel regress the fallback.** With 285 shapes needed for 99% of time, unmatched
  shapes are ~10% of GPU time rather than ~0.1%. A 3× regression off-shape is a net loss.
- **Regressing small-M decode for a prefill win.** Decode-only steps are 70–99% of all steps across the
  entire config range; treat them as mandatory no-regression gates.
- **Disturbing the cache and not noticing.** A short warmup, an undersized KV pool that forces eviction,
  or a cache-bust change does not shift the distribution, it replaces it (lever 5). Check the reported
  prefix-cache hit rate before trusting a window's shapes.
- **Letting a fixed session pool drain.** The corpus holds 393 sessions; replay it for longer than the
  sessions last and the engine runs progressively emptier, so the batch you measure is a property of
  your observation window rather than of the workload. Recycle sessions for the whole run, or keep the
  window short enough that occupancy is still flat at the end of it.

## Sources
- AgentX methodology (corpus construction, replay graph, session-tree concurrency, warmup/determinism):
  https://inferencex.semianalysis.com/agentx/methodology
- AgentX v1.0 corpora: `semianalysisai/cc-traces-weka-062126` (1M) and `-256k`.
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
- Per-step token budget flag names and semantics: vLLM
  https://docs.vllm.ai/en/stable/configuration/optimization/ · ATOM (default 16384)
  https://rocm.docs.amd.com/projects/atom/en/main/configuration_guide.html · SGLang
  (`min(chunked_prefill_size, max_prefill_tokens)`, `dp_size` division)
  https://docs.sglang.io/advanced_features/server_arguments.html
- Running-batch cap flag names and defaults: vLLM `--max-num-seqs`
  (`SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128`, resolved in `EngineArgs.create_engine_config`)
  https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/ · ATOM `--max-num-seqs` (default 512)
  https://rocm.docs.amd.com/projects/atom/en/main/configuration_guide.html · SGLang
  `--max-running-requests` (unset by default, derived from the KV pool, divided by `dp_size` under
  `--enable-dp-attention`) https://docs.sglang.io/advanced_features/server_arguments.html
- CUDA graph capture sizes as a decode-M quantiser: ATOM `--cudagraph-capture-sizes` and the
  `cuda_graph_sizes` expansion rule
  https://rocm.docs.amd.com/projects/atom/en/main/configuration_guide.html · SGLang
  `--cuda-graph-max-bs-decode` https://docs.sglang.io/advanced_features/server_arguments.html.
  Observed three-valued decode batch: SGLang replay of the AgentX 1M corpus at concurrency 10,
  `--max-running-requests 20`, captured in `TraceLens/agentic_workload_understand`.
