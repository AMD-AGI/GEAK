# GLM-5.2-MXFP4 on MI355X — SGLang, TP=8, five source patches in the DSA prefill and MLA absorb paths

**Measured win: +4.23% output throughput (1468.358 → 1530.461 tok/s)**, gsm8k 5-shot strict-match
0.970432 → 0.968916 (passes, 2 problems out of 1319). Carried by **five source patches** — no server
flag, environment variable or workload parameter differs from the baseline in any arm reported here.
The configuration on this model was already exhausted before either round started; everything below
is code.

This is the only model in the campaign taken through **two rounds**, and the two rounds do not carry
equal weight. Read the status table before the headline.

| round | reference level | result level | delta | claimed? |
| --- | --- | --- | --- | --- |
| **round 1** — patches 01 + 02 + 03 | 1468.358 (stock source, 7 runs / 3 sessions) | **1509.241** (5 runs / 2 sessions) | **+2.78%** | yes, all three patches individually clear the floor |
| **round 2a** — first-slot guard | 1508.187 (`r1ref`, 9 runs / 3 sessions) | 1512.166 (15 runs / 5 sessions) | +0.264% | **no — inside the floor.** Exported as a *correctness* result |
| **round 2b** — absorb bmm token-major | 1512.166 (`fs`, 15 runs / 5 sessions) | **1530.461** (9 runs / 3 sessions) | **+1.210%** | yes |
| round 2 combined vs the round-1 arm | 1508.187 | 1530.461 | +1.477% | yes, but see the split above |
| **cumulative, both rounds** | 1468.358 | **1530.461** | **+4.23%** | yes, with the cross-boot caveat below |

## Reproduction status — read this first

**Round 1's result reproduced across boots. Round 2's has not yet been re-measured on a second
boot.** Stated precisely, because the difference matters:

- **Round 2 re-measured the round-1 stack on its own node and boot**, from the exported
  `patches/applied/` tree, as the `r1ref` arm. Its first session gave **1507.629 / 1506.475 /
  1507.979**, and the pooled figure over 9 runs and 3 sessions is **1508.187** against round 1's
  **1509.241** — **0.070% low, well inside either round's noise floor.** That **is** a cross-boot
  reproduction of round 1's result from the artifact alone, and it is the strongest reproduction
  evidence in this entry.
- **What it reproduced is the absolute patched level, not round 1's delta.** Round 2 never ran a
  stock arm, so the +2.78% itself was not re-derived on boot 2. The cumulative +4.23% therefore
  chains a baseline measured on boot 1 to a result measured on boot 2, and its justification is
  exactly the agreement above: the two boots land within 0.070% of each other at the round-1 level.
- **Round 2's own +1.210% rests on one boot**, with 3 `fl` sessions interleaved against 5 `fs`
  sessions, 9 runs against 15, and non-overlapping distributions at both run and session level. It
  has not been re-measured after a reboot or on a different node.
- **Artifact integrity was re-verified while writing this entry**, from the copies in `artifacts/`
  and nothing else: seeding a throwaway tree from `artifacts/round1/base/` and applying all five
  patches with `git apply -p1` succeeds for each, and the resulting three files are byte-identical
  to `artifacts/round2/arms/fl/` — `forward_mla.py` at md5 `41824f82b4d625edf600645661ddbae8`, the
  value round 2 records for the live tree it left running. That verifies the *tree*, not the
  throughput.

So: treat +2.78% as reproduced, +1.210% as measured once and well separated, and +4.23% as the
composition of the two. If you deploy on a matching stack, budget one hour to re-measure `fl`
against `r1ref` and you will have the second boot this entry is missing.

## What the baseline already contains, and the three numbers not to quote

Get this straight before comparing anything, because five figures are in circulation for this model
and three of them are wrong for most purposes.

| layer | tok/s | what it is |
| --- | --- | --- |
| untouched configuration | 1158.043 | source session, before any config search |
| FP8 KV cache only | 1429.925 | source session, the middle step |
| `BASELINE.md` figure — FP8 KV + INT4 quick-reduce | **1462.337** | **+26.28% over untouched, already banked**, a config search that is done |
| locally measured stock source, this node | **1468.358** | 7 runs / 3 sessions — **the number every round-1 claim is made against** |
| round-2 result | 1530.461 | 9 runs / 3 sessions on round 2's boot |

The +26.28% is two flags — an FP8 KV cache and INT4 quick-reduce — and it is not re-discoverable;
it is the starting point. **Quote gains against 1468.358**, the locally measured stock-source mean,
not against the document's 1462.337 (the same result reads +4.66% against it) and never against
1158.043 (+32.16%). Round 1's own headline is stated both ways: +2.78% locally, +3.21% against the
document figure.

Two things the baseline also contains, both worth knowing:

- **INT4 quick-reduce is genuinely lossy** — rel-L2 **1.2e-01** against an exact all-reduce, at every
  message size round 1 measured. The frozen configuration runs it for all of prefill's all-reduce
  traffic and gsm8k still scores 0.970432. That robustness is a property of the given baseline, not
  of anything either round added.
- **The baseline arm had no accuracy figure before round 1.** The source session evaluated gsm8k only
  on the untouched configuration (0.965883 ± 0.005000). Round 1's `baseline_20260820_112214` run is
  the first accuracy number for the configuration being beaten.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 8 × AMD Instinct MI355X, `gfx950`, **256 CU** each | **yes** — 256 appears twice as a literal: the split-K heuristic's one-wave grid is `256 CU × 2 = 512` blocks (round 1 §6), and the MoE tuned table is keyed `cu_num=256` (round 2 §4h). Off 256 and both stop being true. |
| host | `crsuse2-m2m-068` | descriptive |
| container | tag `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`, **probable, not verified** — see gap note | descriptive |
| OS / ROCm | Ubuntu 22.04.5 LTS / ROCm 7.2.0 | descriptive |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 | descriptive, but see the absorb patch's caveat — the `out=`-strided bmm's kernel selection is a rocBLAS/torch property |
| SGLang | **0.5.17**, editable install from `/sgl-workspace/sglang/python`, git HEAD **`29481685462732237d80d86076d6563e1f658102`** | **yes** — all five patches are line diffs against three files at this commit. A different commit is where "the patch does not apply" comes from, and the pristine copies in `artifacts/round1/base/` are the fallback. |
| Triton | 3.6.0 | descriptive |
| aiter | `/sgl-workspace/aiter`, a source checkout; **version and commit not recorded** | **yes in effect** — it supplies the MoE (22.0% of decode) and the `ca` all-reduce, and both were checked as already-optimal against *this* build. The missing sha is a real gap. |
| tilelang | `/opt/tilelang/build`; **version not recorded** | **yes** — patches 01 and the round-2 guard are TVM-Script edits inside a tilelang kernel factory, and the round-2 engagement check reads tilelang's on-disk HIP cache. |
| model | GLM-5.2-MXFP4, **TP=8**, 78 attention layers, 75 MoE layers (`first_k_dense_replace = 3`), 21 of 78 indexer layers `full` | **yes** — TP=8 gives `tp_q_head_num = 8`, which is what makes patch 02's shape gate pass and what round 1 §5/§8 are about. Layer counts are how both rounds recognised their targets (624 = 2 × 78 × 4 passes; 79/step = 78 + 1). |
| quantization | `quark` MXFP4 (group 32) weights, bf16 activations | **yes** — and specifically: the checkpoint's `quantization_config.exclude` lists `o_proj`, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` **for every layer**, so the MLA linears are bf16. That is what selects the bf16 absorb path, which is the only path round 2b fixes, and it is what makes `w_scale == 1.0`, which is the only condition under which patch 03 does anything. |
| KV cache | `fp8_e4m3` | **yes** — it selects the tilelang DSA prefill/decode kernels (`Set DSA backends for fp8_e4m3 KV Cache: prefill=tilelang, decode=tilelang`), which are the files patches 01/02 and the guard edit. |
| attention backend | `dsa`, resolved — **not requested** | **yes** |
| `dsa_topk_backend` | **`sgl-kernel`**, resolved from the default because the launcher does not pass `--dsa-topk-backend` | **yes, and load-bearing for correctness, not for speed** — the round-2 guard's invariant was verified against this producer only, and round 2 demonstrated that the `flashinfer` backend *breaks* it. See "the guard's correctness is configuration-bounded". |
| page size | **64**, resolved — `--page-size 1` was requested | **yes** |
| `disable_radix_cache` | True | descriptive, but part of the frozen contract |

Verified against the live server: every `r1ref`/`fs`/`fl` result directory in round 2 carries a
`server_info.json` captured next to it, which is where `page_size = 64`,
`attention_backend = 'dsa'`, `dsa_prefill_backend = dsa_decode_backend = 'tilelang'` and
`dsa_topk_backend = 'sgl-kernel'` are read from. Round 1's result directories do **not** carry one —
round 2 added that.

### Where a config label disagrees with what ran

**`--page-size 1` is requested and ignored.** SGLang logs `Setting page size to 64 for DeepSeek DSA`
before printing its resolved arguments, and the live server reports `page_size: 64`.
`launch_server.sh` asserts 64, deliberately: **if you ever see page size 1 here, the DSA path did not
engage and the number is not comparable to anything in this entry.** The sparse-attention path picks
its own page size, so paging experiments on this model are constrained.

Two more, upstream of this entry: the source session passed `--block-size 1`, which is the vLLM
spelling and required an argparse alias patch to be accepted at all; the harness here passes the real
flag name instead. And `--mem-fraction-static 0.8` may legitimately read back as 0.68, because SGLang
rescales it by 0.85 when aiter is combined with a context length above 8192 — `launch_server.sh`
accepts either.

## Launch configuration

Exactly this, and nothing added. `artifacts/harness/launch_server.sh` is the copy that was used.

```bash
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

python3 -m sglang.launch_server \
  --model-path /shared_nfs/hyperloom/models/GLM-5.2-MXFP4 \
  --host 0.0.0.0 --port 43111 \
  --tp-size 8 \
  --context-length 11264 \
  --watchdog-timeout 1800 \
  --page-size 1 \
  --quantization quark \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3
```

**Two environment variables, and that is the whole env recipe.** `SGLANG_USE_AITER=1` is not
optional bookkeeping: it is the condition the quark-MXFP4 MoE import already tests, so with it unset
you are on a different stack and a second enablement patch from the source session becomes relevant
again. `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` is what makes quick-reduce usable at all at these sizes
(round 1 §7). Both rounds treated the environment as frozen and therefore could not use any
env-gated diagnostic — no `AITER_LOG_TUNED_CONFIG`, no `SGLANG_DSA_TRITON_PREFILL`, no profiler
env var. Every engagement check below is flag-free for that reason.

Resolved values not visible in the invocation, read out of `server_info.json`:
`max_total_num_tokens = 3283264`, `max_req_input_len = 11258`, `max_prefill_tokens = 16384`,
decode CUDA graphs captured to `max_bs = 512` on the `full` backend, prefill graph capture 0.0 s.
Startup on this node: `load_weight` 58.6 s, `scheduler_e2e` 148.6 s, decode graph capture 43.1 s.

**Budget ~4 minutes per server start and ~2.5 minutes per benchmark.** An interleaved A/B leg is
therefore about 40 minutes per candidate — affordable, and the reason both rounds could interleave at
all. Round 2 ran eleven sessions on a single boot.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, **8 warmups**, seed 0, `random` dataset,
`random_range_ratio 1.0`, `random_prefix_len 0`, `--ignore-eos`, InferenceX `benchmark_serving`
against `/v1/completions`. `artifacts/harness/run_bench.sh` is the copy that was used, and every
figure in this entry came out of it unmodified. Round 2's `summarize.py` re-checks the 192-prompt /
concurrency-64 contract on every result JSON it averages, which is a cheap guard worth copying.

Note the warmup count: **8, not DeepSeek-V4-Pro's 128.** Comparisons against this directory's other
entries are like-for-like on that axis except for DeepSeek.

What sets the shapes each patch depends on:

- `--chunked-prefill-size 16384` gives the prefill chunk **16384 tokens** = 2 × ISL 8192. That is
  what makes 12.1% of index blocks all-padding, which is patch 01's entire premise, and it is what
  puts prefill query-token counts above patch 01's `_SKIP_EMPTY_MIN_TOKENS = 1024` threshold.
- Concurrency 64 gives **decode batch 64**, and the largest decode query-token count observed over
  12,168 decode calls is **512** — comfortably below 1024, which is why the guard is never compiled
  into the decode kernel.
- `index_topk = 2048` with `block_I = 64` gives 32 index blocks per query row.
- TP=8 gives `tp_q_head_num = 8`, `nope_dim = 512`, `rope_dim = 64` — all powers of two, which is
  patch 02's shape gate, and 8 heads is what gates round 1 §5 and §8 off.
- The MoE tuned table's key: `model_dim = 6144`, `expert = 257`, `inter_dim = 256`
  (`moe_intermediate_size` 2048 / TP 8).

## Baseline and noise floor

### Where the time goes (round 1, corrected profile)

Both rounds' target selection came from this split, so it is worth carrying. Window 134.09 s; 12
captured prefill passes totalling 8322 ms → 693.5 ms per pass × 96 chunks = **66.6 s of prefill,
49.7% of end-to-end**; decode is the other ~50.3% and is **99.7% GPU-busy** — there are no bubbles
to reclaim, only work to delete.

| prefill budget | | decode budget | |
| --- | --: | --- | --: |
| DSA attention (`main_kernel`) | 48.2% | MoE | 36.0% |
| all-reduce (INT4 quickreduce) | 15.1% | dense GEMM | 19.9% |
| MoE | 12.3% | DSA attention | 13.41% |
| dense GEMM | 10.2% | all-reduce (`cross_device_reduce_2stage`) | 8.92% |
| indexer | 3.65% | norm / rope / cache | 6.6% |
| norm / quant | 2.6% | elementwise | 4.9% |
| `CatArrayBatchedCopy` | 1.42% | indexer | 3.6% |

Round 1's three patches came out of the prefill column plus one item inside decode's `elementwise`
4.9%. Round 2's win came out of a fresh capture on the round-1 arm, where **31 of 45 distinct decode
kernels average under 10 µs and together are 37.99% of decode**, at 2134 kernel launches per decode
step.

### Round 1's arms

| arm | runs / sessions | tok/s | vs baseline | vs previous arm |
| --- | --: | --: | --: | --: |
| baseline (stock source) | 7 / 3 | 1468.358 | — | — |
| A only | 2 runs (sessions not recorded) | 1487.89 | +1.33% | — |
| B only | 2 runs (sessions not recorded) | 1478.94 | +0.72% | — |
| A + B | 9 / 4 | 1497.662 | +2.00% | +2.00% |
| **A + B + C** | 5 / 2 | **1509.241** | **+2.78%** | **+0.77%** |

Individual A+B runs: 1496.902 / 1497.500 / 1495.810 (s1), 1495.463 / 1495.658 (s2), 1500.528 /
1497.377 (s3), 1498.381 / 1501.338 (s4). Individual A+B+C runs: 1507.291 / 1509.758 / 1508.101 (s1),
1510.233 / 1510.825 (s2).

### Round 2's arms

Eleven sessions on one boot, interleaved `r1ref / fs / r1ref / fs / r1ref / fs`, then
`fl / fs / fl / fs / fl`. One fresh server per session, three benches per server, nothing dropped.

| arm | session | runs (tok/s) | session mean | within-instance p2p |
| --- | --- | --- | --: | --: |
| `r1ref` | s1 | 1507.63 / 1506.47 / 1507.98 | 1507.36 | 0.100% |
| `r1ref` | s3 | 1505.86 / 1509.79 / 1507.64 | 1507.76 | 0.261% |
| `r1ref` | s5 | 1511.92 / 1506.50 / 1509.89 | 1509.44 | 0.359% |
| `fs` | s2 | 1511.95 / 1511.13 / 1510.27 | 1511.11 | 0.111% |
| `fs` | s4 | 1508.72 / 1512.41 / 1513.39 | 1511.51 | 0.309% |
| `fs` | s6 | 1512.70 / 1511.48 / 1507.19 | 1510.46 | 0.365% |
| `fs` | s8 | 1510.65 / 1518.05 / 1513.05 | 1513.91 | 0.489% |
| `fs` | s10 | 1511.75 / 1514.62 / 1515.14 | 1513.84 | 0.224% |
| `fl` | s7 | 1529.51 / 1526.78 / 1527.92 | 1528.07 | 0.179% |
| `fl` | s9 | 1527.78 / 1533.25 / 1531.53 | 1530.85 | 0.358% |
| `fl` | s11 | 1530.07 / 1534.30 / 1533.02 | 1532.46 | 0.276% |

### The floor, measured separately in each round

| noise floor | round 1 | round 2 |
| --- | --- | --- |
| repeating the benchmark within one process | **≤0.21%** (worst of 9 sessions; best 0.161%) | **0.100%–0.489%** per session; worst per arm 0.359% / 0.489% / 0.358% |
| across restarts, spread of *session means* | 0.160% / 0.287% / 0.142% by arm | 0.138% (`r1ref`) / 0.229% (`fs`) / 0.287% (`fl`) |
| **across restarts, run-level peak-to-peak (the floor actually used)** | **0.39%** (worst arm, 21 runs / 9 sessions) | **0.402% / 0.718% / 0.491%** by arm |

**The restart floor is the one that applies**, in both rounds, for the ordinary reason: every change
here is a source change, the tilelang kernels are JIT-compiled at first use and the decode CUDA
graphs are captured at startup, so two arms can never share a process. Round 1's within-process floor
reads 0.21% and round 2's reads 0.100% — either would have licensed claims that do not survive.

**Round 2 re-derived its own floor rather than inheriting round 1's, and got a worse one.** Round 1
measured 0.39% run-level across 21 runs and 9 sessions; round 2 measured **0.402% / 0.718% / 0.491%**
across 33 runs and 11 sessions on a single boot. This is the campaign's cross-cutting lesson landing
concretely on one model: **the floor is a property of the boot and the node, not of the model.** Two
corollaries round 2 recorded that are worth more than the numbers:

- **The per-arm floor grows as you add sessions.** Adding `fs_s8` and `fs_s10` moved the `fs` point
  estimate *up* (+0.188% → +0.264%) and widened its run-level spread from 0.410% to **0.718%**,
  because `fs_s8` contained a 1518.05 run. More data made the estimate more favourable and the bar it
  had to clear worse. That is the reason not to shop for the most flattering spread statistic after
  the fact.
- **Restart spread of session means (0.138–0.287%) is a different, tighter statistic** than run-level
  peak-to-peak, because averaging three runs cancels most within-instance noise. It is the right unit
  for *comparing* arms, and it cannot be substituted for round 1's floor to make a small delta look
  claimable. Round 2 held the guard to the run-level figure and it failed; the absorb fix clears both
  by 2.5× and 4.2× respectively.

### Are the arms disjoint?

| comparison | run level | session-mean level | verdict |
| --- | --- | --- | --- |
| baseline → A+B (round 1) | disjoint — worst A+B run 1495.463 beats best baseline run 1470.671 by **+1.69%** | disjoint | claimed |
| A+B → A+B+C (round 1) | disjoint — worst A+B+C run 1507.291 beats best A+B run 1501.338 by **+0.40%** | disjoint | claimed |
| `r1ref` → `fs` (round 2a) | **overlapping** — worst `fs` run 1507.19 is below best `r1ref` run 1511.92 | disjoint — min `fs` session 1510.46 > max `r1ref` session 1509.44; Welch t = 4.15, df = 5.7, +3.98 tok/s, SE 0.96 | **not claimed** |
| `fs` → `fl` (round 2b) | disjoint — worst `fl` run 1526.78 beats best `fs` run 1518.05 by **+0.58%**; all 9 `fl` runs above all 15 `fs` runs | disjoint — worst `fl` session 1528.07 > best `fs` session 1513.91; Welch t = 12.45, df = 3.3, +18.29 tok/s, SE 1.47 | claimed |

### Drift, and two measurement traps specific to this stack

**Throughput drifts upward over hours with no source change at all.** Round 1 saw A+B session means
climb from 1496.737 / 1495.561 in its early sessions to 1498.953 / 1499.860 in the two late-afternoon
ones — a **+0.29%** wander on the same source, larger than its within-instance spread and a third of
candidate C's whole effect. Round 2 saw the
`fs` arm go 1511.11 / 1511.51 / 1510.46 / 1513.91 / 1513.84 over four hours, **~0.2% upward on the
same binary**. Both rounds handled it by interleaving rather than blocking, and in both cases the
drift ended up working *against* the claimed result: round 1's late A+B control sessions raised the
control mean from 1496.267 to 1497.662, and round 2's two late `fs` sessions are the two fastest of
five, so the +1.210% is measured against an inflated baseline.

**A profiled run poisons the next run on the same process.** Round 1 excluded `ab_s3_r1` = 1486.704
(mean TTFT 11363.6 against ~11085 for its arm) from every figure: it was the first bench on a server
that had just serviced two `with_stack` captures, and runs 2 and 3 on the same process returned to
1500.5 / 1497.4. The baseline sessions show no such first-run climb, so this is profiler residue, not
warm-up. Round 2's response was to profile on a server that was then torn down rather than reused.

**"Discard the first run" is not right here, and this entry disagrees with the house rule.** Round 2
checked: `fs_s4` climbed across its three runs (1508.72 → 1513.39) while `fs_s6` fell (1512.70 →
1507.19). Run 1 of a session is not reliably the slow one on this stack, so dropping it would be a
fudge. Every round-2 run is kept. Two runs *were* dropped in round 1 and both for stated,
non-numerical reasons — profiler residue and profiler-driver runs (1074.74 and 1152.675 tok/s, both
invalid as throughput measurements by construction).

## The five patches, and what each is worth

| # | patch | file | worth | evidence class |
| --- | --- | --- | --: | --- |
| 01 | `artifacts/round1/patches/01-dsa-prefill-skip-empty-index-blocks.patch` | `tilelang_kernel.py` | **+1.33%** (1487.89) | bit-exact by construction |
| 02 | `artifacts/round1/patches/02-dsa-prefill-fused-q-fp8-prep.patch` | `dsa_backend.py` | **+0.72%** (1478.94) | bitwise identical (`torch.equal`, rel-L2 0.00000) |
| 03 | `artifacts/round1/patches/03-mla-absorb-drop-unit-weight-scale.patch` | `forward_mla.py` | **+0.77%** on top of A+B (1509.241) | numerically inert (`x * 1.0`) |
| G | `artifacts/round2/patches/dsa-prefill-first-slot-empty-block-guard.patch` | `tilelang_kernel.py`, `dsa_backend.py` | **+0.264%, NOT claimed** | correctness proven empirically; **no throughput claim** |
| A | `artifacts/round2/patches/mla-absorb-bmm-write-token-major.patch` | `forward_mla.py` | **+1.210%** (1530.461) | not bit-identical; gsm8k is the primary evidence |

**01 — skip index blocks that are entirely padding.** DSA selects `index_topk = 2048` keys per query.
In a 16384-token chunk every query at position < 2048 in its sequence has a causal prefix shorter
than topk, so its index row is partly `-1`. The stock kernel *masks* that padding instead of skipping
it: it still gathers 64 × 576 B of KV per block and still issues all five GEMMs, then discards the
result via a `-inf` score. Counting whole 64-blocks, `sum_p min(ceil((p+1)/64), 32)` = 230400 needed
against 262144 processed per sequence — **12.1% of all blocks processed are provably dead.** The
patch adds a block-uniform any-valid reduce over the validity mask the kernel already builds. Eliding
an all-padding block is bit-exact, not an approximation: every score in it is `-inf`, `exp2` gives 0,
so it adds nothing to `sumexp` and nothing to `acc_o`, and `m_i` is reduced with `clear=False` so a
block of `-inf` cannot lower it either. Isolated: **4.415 → 4.248 ms, 1.039×.**

**01's prefill-only dispatch is load-bearing, and this is the most transferable lesson in round 1.**
The first version applied the guard unconditionally and **regressed decode by 8%** (34.55 → 37.32 µs,
0.926×) — at decode every sequence is longer than topk, so no block is ever all-padding, the guard
never fires, and it is pure overhead. Netted against the prefill gain that is roughly +0.36%, at or
below the floor: **unfixed, the patch was not reportable.** The fix is a parse-time switch.
`skip_empty` is a Python bool read by the TVM Script parser, so with it `False` the branch, the
cross-lane reduce and the `valid_i32` write are all absent from the emitted HIP — the factory
produces a **byte-identical kernel to the pre-patch one**, verified by string comparison of
`get_kernel_source()` at both splits (32483 chars, exact match), with decode output bitwise identical
by `torch.equal`. Dispatch is `skip_empty = q.shape[0] >= _SKIP_EMPTY_MIN_TOKENS` with the constant
at **1024**, which sits in an empty gap: decode graphs are captured at ≤ 512 and prefill chunks are
16384. Because both variants compute the same thing, a misdispatch costs speed, never answers.

**02 — fuse the concat and the fp8 cast on the tilelang prefill q path.** `tilelang_sparse_fwd` casts
q to the KV dtype itself, so the prefill path built a bf16 `[T, H, 576]` with
`concat_mla_absorb_q_general` and then made a second full fp8 copy of the same shape — two passes and
an extra ~18 MB temporary per layer at a 16384-token chunk, visible in the prefill profile as
`CatArrayBatchedCopy` at 1.42%. `concat_and_cast_q_fp8_pad` already existed in the tree, its own
docstring calls it a bit-exact replacement for exactly this pair, and the decode and flashmla paths
in the same file already used it. **This tilelang prefill branch was the last caller still paying the
unfused version.** Isolated: **4.415 → 4.337 ms, 1.018×**, output bitwise identical. Guarded by a
power-of-two shape check on `num_heads / nope_dim / rope_dim`; GLM-5.2 at TP=8 gives 8 / 512 / 64, and
anything else falls back to current behaviour.

**03 — drop the loop-invariant `w_scale` multiply on the MLA absorb path.** The kernel is
`vectorized_elementwise_kernel<8, AUnaryFunctor<BFloat16,BFloat16,BFloat16,MulFunctor<float>>>` —
**46800 calls at 4.60 µs = 3.08% of decode**, exactly 2 per layer per decode step across 78 layers.
The mangled name says only "bf16 tensor × Python scalar"; it does not say where from. Both operands
are load-time constants, so the product is loop-invariant, and for this checkpoint it is the
identity: `DeepseekV2AttentionMLA.__init__` sets `w_scale = 1.0` and the loader overwrites it only on
the fp8 and int8 paths, none of which a quark-MXFP4 `kv_b_proj` dequantising to bf16 can reach.
`_scaled_absorb_weight(w, w_scale)` returns the cast weight unchanged when the scale is a plain
Python 1 and otherwise evaluates the stock expression, so a model whose loader sets a real scale keeps
exactly the old behaviour and the old cost. A 0-dim *tensor* scale is deliberately left alone —
reading its value mid-forward would force a device sync costing more than the multiply.

**Why 03 hides in prefill, which is the reusable part:** its cost is set by the *weight* size
(~3.5 MB/layer), not the batch, so it is ~0.1% of a 693 ms prefill chunk and 3.08% of a ~34 µs decode
step. Round 1's first profile was mis-segmented as 100% prefill and would have dismissed it as noise.

**G — the first-slot guard.** See the dedicated section below. **A — the absorb bmm.** See the next
section.

## The round-2 win: a `transpose().flatten()` that was a full copy

Round 1 left "~26% of decode is in kernels averaging ~5 µs" as a fusion *surface*. Round 2 profiled
the arm actually under measurement and found that **the part of that surface reachable from framework
Python was one bug**, and this was it.

The live absorb path for this checkpoint is the plain-bf16 one — established from the checkpoint's
own `quantization_config.exclude` list rather than assumed. `bmm` writes `(heads, tokens, dim)`;
`o_proj` wants `(tokens, heads*dim)`. So the epilogue read:

```python
attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)   # [T, 8, 256] -> [T, 2048]
```

Transposing first makes the flatten non-view-able, so it **silently degrades to `aten::copy_`** —
materialising the whole attention output once per attention layer per decode step, with nothing in the
source suggesting a memcpy. Profiled dims confirm it: `[[16258, 8, 256], [16258, 8, 256], []]`. At
decode that is **23700 calls at 5.05 µs = 1.915% of decode GPU time**, 79 per step (78 layers plus one
unrelated sampler-path call). 5.05 µs to move ~256 KB is about 100 GB/s on a part that does ~5 TB/s:
**launch and occupancy overhead, not bandwidth.**

The fix is nine lines and was already in the same function, twelve lines above, on the aiter MXFP4
branch — *"Allocate in (batch, heads, dim) so the post-GEMM transpose+flatten is a free view instead
of a copy."* The bf16 branch never got it. Allocate `_bmm_buf` as `(T, H, D)`, pass its transpose as
`out=` to `bmm`, and fall into the `if _bmm_buf is not None:` block that already exists, whose bf16
arm is `_bmm_buf.flatten(1, 2)` — a view.

| | calls | per step | µs/call | % of decode |
| --- | --: | --: | --: | --: |
| before (`fs` arm) | 23700 | 79.00 | 5.05 | **1.915%** |
| after (`fl` arm) | 300 | 1.00 | 17.60 | 0.084% |

All 78 per-layer copies are gone and the survivor is the sampler one. The strided `out=` costs the
bmm a little — 78 launches/step, 11.66 → 11.84 µs, about **14 µs/step** — against roughly **394
µs/step of copy removed**.

**How the kernel was attributed to a line, because this is the method and it is not in the skillset.**
A kernel-name profile ranks time and cannot say which line launched
`elementwise_kernel_manual_unroll<…>`. Round 1 had already established that a `with_stack` capture
must be taken during **prefill** — decode runs inside a HIP graph, so the decode trace shows 6884
kernels against 517 `cpu_op`s, all tagged `"kind": "Dispatch Task"`, with no Python frames at all;
prefill runs eagerly and executes the same layer code. But round 1's linking recipe did **not**
transfer: on this torch/ROCm build `args["Call stack"]` does not exist on `cpu_op` at all, and the
first implementation returned 429/429 unattributed. The chain that works is

```
GPU kernel --args["correlation"]--> cuda_runtime (hipLaunchKernel)
           --args["External id"]--> cpu_op
           --> innermost containing python_function on the same tid, by timestamp interval
           --> out via args["Python parent id"]
```

which is `artifacts/round2/analysis/attribute.py`, **429/429 attributed, 0 unattributed**. Two
details made the output readable rather than merely correct: `--collapse` strips the
`nn.Module: GLM5DecoderLayer_<n>` index so all 78 layers aggregate into one row, and **the output that
matters is a per-kernel-family count, not a ranking**. Ranked by absolute time the copy is 1.9% and
unremarkable; counted, it is *exactly one per attention layer per pass*, which has no legitimate
explanation.

| calls | share of `at::native` time | launching op | site |
| --: | --: | --- | --- |
| **312** | **79.3%** | `aten::copy_` | `forward_mla.py` `forward_absorb_core`, `Tensor.flatten` |
| 84 | 9.8% | `aten::fill_` | `dsa_indexer.py:953` `_get_topk_ragged`, `torch.full` |
| 84 | 2.0% | `aten::copy_` | `index_buf_accessor.py:186` `.to()` on a `[3]` tensor |
| 84 | 1.8% | `aten::fill_` | `index_buf_accessor.py:186` `torch.zeros([4])` |

312 = 78 layers × 4 captured passes.

**The bug class generalises and has a mechanical detector.** A chained reshape that *reads* like
metadata manipulation and is silently a full copy: `transpose(0, 1).flatten(1, 2)` cannot be a view
once the transpose broke contiguity. Grep hot paths for `.transpose(...).reshape(` and
`.permute(...).flatten(`, check `.is_contiguous()` on the intermediate, or look for
`elementwise_kernel*` / `aten::copy_` in a profile **with a call count that matches your layer
count**. The fix is usually to hand the producing op an `out=` in the layout the consumer wants.

## The first-slot guard: correct, exported, and NOT a throughput claim

This is round 2's other outcome and it must not be conflated with the win above. **It is recorded here
as a correctness result. A reader should not expect throughput from it.**

Round 1 measured a cheaper variant of patch 01's guard — test only the block's first slot instead of
reducing over all 64 — at **1.067× prefill against the reduce's 1.039×**, extrapolated it to roughly
**+0.64% more end-to-end**, and **declined to adopt it**. Not because it was slow but because it is
correct only if the `-1` padding is a contiguous *suffix* of every index row, which is a property of
the top-k producer and not of the kernel file. Four backends could produce that tensor and each would
have to preserve the invariant in every future version, and **a violation does not fail loudly: it
silently drops real KV keys**, degrading answers in a way neither an assertion nor a 1319-problem
gsm8k run would reliably catch. Round 1 kept it as a reviewable rejected diff and called it the
largest single unclaimed gain in the run.

### What round 2 proved

Round 2 stopped reading producers and instrumented the operative tensor — `page_table_1` at the exact
call site that feeds the tilelang sparse kernel as `indices` — over one full unmodified benchmark:

```
calls=8250  rows=126,861,936  blocks=4,059,581,952
reduce_skips=491,606,928   first_skips=491,606,928
UNSAFE_BLOCKS=0            NONMONOTONIC_POS=0
all_pad_rows=0             all_valid_rows=95,161,470
shapes={(6,2048):78, (1,2048):156, (8192,2048):546, (16384,2048):7470}
decode_max_qtok=512        decode_calls=12168
```

- **4.06 billion 64-wide index blocks checked**, across 126.9 million rows, in 8250 kernel calls.
- **`UNSAFE_BLOCKS = 0`** — zero blocks where slot 0 is `-1` but the block still holds a real key.
  That is exactly the set the cheap guard would wrongly drop, counted directly.
- **`first_skips == reduce_skips`, exactly.** Bit-identical block sets, not "nearly the same". The
  guard is neither unsafe nor conservative.
- **`NONMONOTONIC_POS = 0`** — the strictly stronger property holds too: across 126.9M rows there is
  not one position anywhere where a `-1` is followed by a valid slot.
- **The boundary was genuinely exercised.** **31,700,466 rows (24.99%) are partially padded** — the
  case that discriminates the two guards. This is what makes the zero a positive result rather than
  an assert that failed to fire; an all-valid or all-padding corpus would have proved nothing.
- **12.11% of blocks are skipped**, reproducing round 1's independently derived 12.1% dead-block
  figure to three significant figures — good evidence the probe is reading the right tensor.

The producer was identified by direct observation, not deduction: rebinding the entry points that
`DSATopKBackend.topk_transform` imports inside its function body counted
`{'sglk:fast_topk_transform_fused': 2221}` — **one** producer, the sgl-kernel fused PAGED transform.
`topk_transform_512_v2` never ran at prefill, consistent with its decode-shaped dispatch condition.

The probe's own result directory is prefixed `probe_` because the instrumentation perturbs throughput
(**1454.2 tok/s against 1507.4 unmodified, −3.5%**) and is not a valid measurement of anything.

### And it measured +0.264%, which is inside the floor

| | `r1ref` | `fs` | delta |
| --- | --: | --: | --: |
| pooled mean | 1508.187 (9 runs, 3 sessions) | 1512.166 (15 runs, 5 sessions) | **+0.264%** |

**Not claimed.** The run-level peak-to-peak noise on this boot is 0.402% (`r1ref`) / 0.718% (`fs`) —
the same statistic round 1 used to set its 0.39% floor — and 0.264% is under it. Individual runs
overlap: the worst `fs` run (1507.19) is below the best `r1ref` run (1511.92).

What the data does support, stated no more strongly than it deserves: the sign is consistent across
all five independent restarts of `fs` and all three of `r1ref`; **every `fs` session mean (min
1510.46) exceeds every `r1ref` session mean (max 1509.44)**; Welch on session means gives t = 4.15,
df = 5.7, +3.98 tok/s, SE 0.96; and the change is *strictly less work* — the emitted kernel loses a
cross-lane AllReduce and its LDS workspace and gains one scalar load, so there is no mechanism by
which it is slower.

**Round 1's +0.64% extrapolation was optimistic by roughly 2.4×.** Converting it into a measured
+0.264% is most of the value of closing this lead: the "standing uncertainty" was mostly not there.
This is a clean instance of the campaign rule that a kernel multiple is not a result.

### The guard's correctness is configuration-bounded, and the boundary is one flag away

`dsa_topk_backend` is a **server argument**; the frozen launcher does not pass it, so it defaults to
`sgl-kernel`. The `torch` and `flashinfer` backends are unreachable without changing the launch
configuration. **That genuinely narrows the correctness argument**, and round 2 found the concrete way
it breaks — in code that ships today, not hypothetically. `_topk_unfused` maps padding to `-1`
**in place**, after the top-k, without re-sorting:

```python
topk_local_indices = topk_local_indices.masked_fill(topk_scores == float("-inf"), -1)
```

Whether the `-1`s land in a suffix depends entirely on the ordering `topk_op` returns. The `torch`
backend survives only because `torch.topk` defaults to `sorted=True` descending, which pushes `-inf`
to the tail — **by accident of a default**. The `flashinfer` backend passes `{"sorted": False, ...}`,
so the `-1`s can land anywhere. Demonstrated rather than argued, by running that exact masking
sequence under both orderings:

```
sorted=True  (torch backend):  non-suffix positions = 0   row0 = [2, 0, 1, -1, -1, -1, -1, -1]
sorted=False (flashinfer):     non-suffix positions = 4   row0 = [-1, 0, -1, -1, -1, 1, 2, -1]
```

The second row is exactly the shape that drops live keys: slot 0 is `-1` while slots 5 and 6 hold
real pages. **That is round 1's feared failure mode, one launch flag away.** Hence the shipped patch
carries `SGLANG_DSA_ASSERT_TOPK_SUFFIX=1` (default off, cost when off is one predicted branch on a
module-level constant per prefill layer call — about 8k times per benchmark, ~0.25 ms total), which
converts the silent failure into a loud one naming the offending row. The assertion itself was
verified against three cases: suffix-padded (silent), all-padding (silent), scattered `-1` (raises).
An assertion that cannot fire would be worse than none.

The guard is also inert at decode by measurement, not assumption: `skip_empty` is decided at
kernel-build time from `q.shape[0] >= 1024`, and the largest decode query-token count over 12,168
decode calls is **512**. So the invariant is load-bearing for prefill only, which is the population
that was instrumented. **Patch 01's prefill-only dispatch must stay** — the first-slot guard still
costs ~4% at decode (0.961×) despite having no reduce at all, which says the *branch*, not the
reduce, is most of the decode penalty.

## Deploy

The three files, at SGLang `29481685462732237d80d86076d6563e1f658102`:

- `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`
- `python/sglang/srt/layers/attention/dsa_backend.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`

### Order and interdependencies

**Apply in this order.** 01, 02 and 03 are mutually independent — each applies alone to stock, which
is how they were measured separately. The two round-2 patches are not:

- **The guard stacks on 01.** It edits the `skip_empty` guard that 01 introduces and **will not apply
  to stock SGLang.** It also touches `dsa_backend.py`, which 02 has already modified.
- **The absorb patch depends on 03 textually, not semantically.** 03 replaced the inline
  `self.w_vc.to(torch.bfloat16) * self.w_scale` with `_scaled_absorb_weight(...)` on the same two
  lines the absorb hunk rewrites. To apply it on stock instead, substitute the stock expression back
  into its `_w = ...` line; the optimisation itself is independent of 01, 02 and 03.

```bash
cd /sgl-workspace/sglang

git apply -p1 <KB>/artifacts/round1/patches/01-dsa-prefill-skip-empty-index-blocks.patch
git apply -p1 <KB>/artifacts/round1/patches/02-dsa-prefill-fused-q-fp8-prep.patch
git apply -p1 <KB>/artifacts/round1/patches/03-mla-absorb-drop-unit-weight-scale.patch
git apply -p1 <KB>/artifacts/round2/patches/dsa-prefill-first-slot-empty-block-guard.patch
git apply -p1 <KB>/artifacts/round2/patches/mla-absorb-bmm-write-token-major.patch
```

**If any patch does not apply**, do not fight it — copy the measured files in whole. That is what the
run itself did to switch arms, so the file that was measured and the file you get are the same bytes
by construction:

```bash
cp <KB>/artifacts/round2/arms/fl/tilelang_kernel.py \
   /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
cp <KB>/artifacts/round2/arms/fl/dsa_backend.py \
   /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py
cp <KB>/artifacts/round2/arms/fl/forward_mla.py \
   /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
```

`artifacts/round2/analysis/set_arm.sh` is the run's own arm switcher and does exactly this, plus the
cache drop below, plus printing md5s. `artifacts/round1/base/` (stock) and `artifacts/round1/applied/`
(stock + 01/02/03) are the other two states, for diffing and for reverting to the round-1 arm.

### Cache invalidation

```bash
# 1. Python bytecode for the DSA package — MANDATORY. set_arm.sh does exactly this.
find /sgl-workspace/sglang/python -name '__pycache__' -path '*dsa*' -prune -exec rm -rf {} +

# 2. TileLang's on-disk JIT cache — NOT required for these five patches, see below.
#    rm -rf ~/.tilelang/cache      # forces a rebuild; costs JIT time on the first prefill chunk

# 3. Restart the server — MANDATORY.
bash <KB>/artifacts/harness/launch_server.sh --stop
bash <KB>/artifacts/harness/launch_server.sh     # must print "config verified"
```

**On the TileLang cache.** TileLang JIT-caches compiled kernels on disk under `~/.tilelang/cache`, so
a kernel-source change whose cache key did not move would silently reuse the old binary. Round 2
checked rather than assumed: **exactly one new cache entry appeared** the moment the patched server
first ran a prefill chunk, and its generated HIP contains the new predicate. So the key does move for
these patches and clearing is not required — but if you edit the kernel *further*, clear it, because
this is the one place on this stack where a source edit can be defeated by a cache.

**On `~/.triton/cache` and `/tmp/aiter_configs`: neither bundle records clearing either, and neither
should matter for these five patches.** Patch 02 switches a caller to an *already existing* Triton
helper rather than editing a Triton kernel, and no patch here touches an aiter config table. The two
caches are the standard hazards on this stack and on any *other* change you make they apply in full
force — `/tmp/aiter_configs` is derived and is not regenerated if it already exists, and
`~/.triton/cache` will independently serve you the old kernel. **That they were not needed here is an
inference from what the patches touch, not a recorded observation.** Treat it as untested.

### Every way this deploy silently does nothing

Each one produces a clean, plausible, wrong number.

1. **No restart.** Decode CUDA graphs are captured at startup and the tilelang kernels are
   JIT-compiled at first use. A live drop-in benchmarks perfectly and changes nothing.
2. **Stale `__pycache__`.** `dsa_backend.py` is imported from a package with bytecode caches; the
   run's own arm switcher deletes them on every switch for this reason.
3. **You copied some files but not all three.** You get a mixed arm that is not any measured
   configuration. `set_arm.sh` prints md5s of all three files so the arm is recorded, not assumed —
   copy that habit. Expected `fl` md5s are in the engagement check below.
4. **`--dsa-topk-backend` set to anything other than `sgl-kernel`.** The guard's invariant was
   verified against that producer *only*, and under `flashinfer` it silently drops live KV keys.
   This one is not a no-op — it is a correctness failure. Set
   `SGLANG_DSA_ASSERT_TOPK_SUFFIX=1` if you must run another backend, or revert the guard to
   patch 01's reduce.
5. **Page size resolves to 1 instead of 64, or `attention_backend` is not `dsa`.** The DSA path did
   not engage, patches 01/02/G are unreachable, and the number is not comparable to anything here.
   `launch_server.sh` refuses to proceed.
6. **KV cache dtype is not `fp8_e4m3`.** That is what selects `prefill=tilelang, decode=tilelang`.
   Another dtype takes different kernels and 01/02/G go unused.
7. **`--chunked-prefill-size` below 1024.** Prefill query-token counts fall under
   `_SKIP_EMPTY_MIN_TOKENS`, `skip_empty` is `False` at kernel build, and 01 and G both compile out
   entirely — to a byte-identical kernel, by design. Silent, and it also destroys 01's premise:
   fewer padded rows means fewer dead blocks.
8. **A decode step exceeding 1024 query tokens** (a much larger batch, or speculative decoding) flips
   the guard *on* at decode, where round 1 measured it costing 8% (reduce) or 4% (first-slot).
   Silent, and in the wrong direction.
9. **TP ≠ 8.** `num_heads` stops being 8, patch 02's power-of-two shape gate may fail and fall back
   to the unfused concat — silently, since the fallback is stock behaviour.
10. **A checkpoint whose `kv_b_proj` is fp8 or MXFP4 rather than excluded from quantization.** Then
    `w_scale` is not 1.0 and 03 keeps the stock expression and the stock cost, and the absorb path
    takes the aiter branches, which already write token-major — so the round-2 win is already there
    and the patch does nothing. Both are correct-but-inert.
11. **Benching against a dead server.** Two real failures on this stack: `launch_server.sh --stop`
    returns before the listening socket is released, and **port 43111 sits inside the default
    ephemeral range**, so an outbound socket opened during the ~2 minutes of load and graph capture
    can steal it. uvicorn then dies with `EADDRINUSE` *after* a startup that logged as entirely
    healthy, including completed CUDA-graph capture. Seen twice in ~12 starts; it aborted one
    session. `artifacts/round2/analysis/session.sh` polls `ss` until the port clears, checks
    `PIPESTATUS` rather than a piped `tail`'s exit status, and retries the launch.
12. **Measuring right after profiling.** Profiler residue depressed the *next* run on the same
    process by about 0.9%. Tear the server down after a capture.

## Engagement check

Three checks, cheapest first. All are flag-free, because the environment is the measurement contract
and no diagnostic env var may be added.

### 1. File identity — before you even restart

```bash
md5sum /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py \
       /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py \
       /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
```

| state | `tilelang_kernel.py` | `dsa_backend.py` | `forward_mla.py` |
| --- | --- | --- | --- |
| **engaged (`fl`, all five)** | `95b4881a4b8aaf9346af5495e4b019ab` | `8015825cecb7ea72e0d39b6a4740ff67` | `41824f82b4d625edf600645661ddbae8` |
| round 1 only (01+02+03) | `68d3a7bfc35f502dbf9a7a57683a25e7` | `6fae0184107930674e734a7778d705ca` | `a0707cea836304d4aabcbc61244ba125` |
| **not engaged (stock)** | `a02636bc6d30adc6b035315893ce36d8` | `6b39f9685eb4d56965c01b56533bf1a5` | `2e1d6cdce2ffb9a5d7bcb44e08bc8b94` |

These are md5s of the artifact files themselves, computed while writing this entry; the `fl`
`forward_mla.py` value `41824f82…` is the one round 2 records for the live tree it left running. This
proves the *source* is right and nothing else — it is exactly the check that cannot distinguish a
deploy from a no-op, which is why the next two exist.

### 2. The guard — kernel identity out of TileLang's HIP cache

The strongest available check, because it reads the emitted device code rather than a log line.
After the patched server has served at least one prefill chunk:

```bash
grep -rl 'bool guard = (0 <= indices\[' ~/.tilelang/cache/*/device_kernel.cu
grep -rlE 'int any_valid\[1\]' ~/.tilelang/cache/*/device_kernel.cu
```

- **Engaged:** at least one `device_kernel.cu` matches the first grep — the predicate is
  `bool guard = (0 <= indices[((blockIdx.x * 2048) + (k_i * 64))]);`, where
  `blockIdx.x * 2048 + k_i * 64` is slot 0 of index block `k_i`. That same file has **no `any_valid`
  symbol at all** and only the **two** softmax `tl::AllReduce`s. Round 2 observed exactly one new
  cache entry appear, timestamped at the moment the `fs` server first ran a prefill chunk.
- **Not engaged (patch 01's reduce, i.e. the round-1 arm):** the first grep is empty and the prefill
  build of the same kernel signature — q `[1, seq_len, 8, 576]`, indices `[1, seq_len, 1, 2048]` —
  declares `int any_valid[1]` and carries a **third** `tl::AllReduce`.
- **Not engaged (stock):** neither predicate is present in any build, and there is no `skip_empty`
  parameter in the factory at all.

Note that this is a two-sided check on kernel *content* — the reduce disappearing from the emitted
code, not merely from the Python — which is what makes it stronger than confirming the file changed.

### 3. The absorb fix — kernel call count from a profile

The recorded check, and the one that separates "the number moved" from "my change is why". Capture
300 decode steps through the `/start_profile` POST body, which needs no env var and no flag, so the
profiled server is configuration-identical to the measured one:

```bash
python3 <KB>/artifacts/round2/analysis/profile_decode2.py --steps 300 --out /tmp/prof_decode
# then count the copy kernel in the rank-0 summary
```

| | calls over 300 steps | per step | % of decode |
| --- | --: | --: | --: |
| **engaged** | **300** | **1.00** | 0.084% |
| **not engaged** | **23700** | **79.00** | 1.915% |

79/step is 78 attention layers plus one unrelated sampler-path call; engaged, only the sampler one
survives. The kernel to count is
`void at::native::elementwise_kernel_manual_unroll<…, gpu_kernel_impl_nocast<…>>`. Reference
summaries for both directions are checked in as
`artifacts/round2/analysis/profile_decode_r2_tp0.json` (before) and `profile_decode_fl_tp0.json`
(after).

**And a mirror check for patch 03**, which must stay at zero: a `with_stack` prefill capture counted
**624 `aten::mul` ops and 624 `AUnaryFunctor…MulFunctor` kernels before the patch and 0 of each
after** (624 = 2 × 78 layers × 4 passes). If those reappear, `w_scale` is no longer 1.0 on your
checkpoint and 03 is correctly inert — which also means you are not on the configuration this entry
measured.

## Accuracy gate

gsm8k 5-shot, 1319 problems, **lm-eval 0.4.12** (`lm-eval[api]==0.4.12` in a venv of its own),
`--apply_chat_template`, `temperature=0, top_p=1`, `max_tokens=9216`, seeds `0,1234,1234,1234`,
`num_concurrent=64`, `max_length=11264`. `fewshot_as_multiturn` is not passed and defaults to `True`
in 0.4.12; the eval log records it. `artifacts/harness/run_eval.sh` unmodified — three settings in it are
load-bearing and each was learned by getting it wrong: `max_tokens=9216` (lm-eval's default 256
truncates the reasoning and scores 0.0318, which reads as a broken model rather than a broken
measurement), a `sitecustomize` fallback to `reasoning_content` when `content` is empty, and the
fixed seeds.

| arm | strict-match | flexible-extract |
| --- | --: | --: |
| **gate (round-1 stock-source baseline)** | **0.970432 ± 0.004666** | 0.972707 ± 0.004488 |
| round 1, A+B | 0.973465 ± 0.004427 | 0.975739 ± 0.004238 |
| round 1, **A+B+C** | 0.968916 ± 0.004780 | 0.969674 ± 0.004723 |
| round 2, `fs` (+ guard) | 0.970432 ± 0.004666 | 0.969674 ± 0.004723 |
| round 2, **`fl`** (final) | **0.968916 ± 0.004780** | 0.968916 ± 0.004780 |
| *source session, untouched configuration (given, not measured here)* | 0.965883 ± 0.005000 | 0.968158 ± 0.004836 |

**Threshold: the win must not regress strict-match below 0.970432 ± 0.004666.** The final arm is
0.968916 — **2 problems out of 1319 below the gate, under a third of one standard error. Pass.** The
`fs` arm returned 0.970432, identical to the reference to every digit recorded. (The `fl` row's two
columns are equal at full precision, 0.9689158453373768 ± 0.004780296718393364, read out of that
gate's `results_*.json`; round 2's `FINDINGS.md` records only the strict-match figure.)

**The gate has a resolution, and this run measured it.** Candidate C is numerically inert — `x * 1.0`
is the identity on every finite bf16 value — so A+B and A+B+C execute **bit-identical arithmetic**.
Their gsm8k scores nevertheless differ by 0.00455, **6 problems out of 1319**, which is therefore a
direct measurement of the eval's own run-to-run variance rather than an effect of any change. The
mechanism is the obvious one: at `num_concurrent 64` batch composition depends on arrival timing,
batch composition changes reduction order in the MoE and the all-reduce, and greedy decoding turns a
last-bit difference into a different token wherever two logits are near-tied.

Consequences, which are the reusable part:

- **±0.0047 stderr is the right yardstick and it is not conservative** — it is roughly the observed
  spread. All arms plus the given stock row span 0.9659–0.9735, ten problems, with at most six
  attributable to anything but rerunning the eval.
- **A single gsm8k run here cannot resolve a regression smaller than about 0.005.** Every accuracy
  statement in this entry is "passes the gate", never "matches" or "improves". Read each one as *no
  regression larger than ~0.5pp*.
- **For four of the five patches the gate is a backstop, not the evidence.** 01 is bit-exact by
  construction and byte-identical-kernel-verified when off; 02 is `torch.equal`; 03 is `x*1.0`; the
  guard skips a bit-identical block set (4.06e9 blocks, 0 violations). **The absorb patch is the
  exception** — a strided `out=` can select a different rocBLAS kernel and therefore a different
  accumulation order, so it is *not* bit-identical by construction and the gate is the primary
  correctness evidence for it. That is worth knowing before you trust it on a different torch build.
- One bit-level caveat on 03: the sole difference is a nan payload — stock canonicalises to `0x7fc0`,
  the patch returns it untouched — which no weight tensor carries. Also, `torch.equal` returns False
  in the presence of any nan, which cost time; an int16-view comparison isolated it.

## What was tried and did not work

Eight rows, all with measured numbers: round 1's four documented negatives, two near-misses that
would have been published had they not been re-measured, and round 2's two. Read this before spending
a day on any of them.

| attempt | kernel / op-level result | end-to-end | verdict |
| --- | --- | --- | --- |
| **Prefill launch geometry** — `block_I ∈ {32,64,128}` × `threads ∈ {128,256,512}` × `h_per_block ∈ {8,16}`, interleaved | **stock `64 / 256 / 16` beats every config that compiles**; `h_per_block = 8` does not compile at all (`no matching constructor for 'fp8_e4_2_t'`, an fp8×2 vectorisation constraint at an 8-wide tile) | not run | Geometry is not the problem. The 16-row head tile *is* half padding at 8 heads, but the MFMA count is unchanged at M=8 — only register pressure would improve. |
| **Decode split-K** — `inner_iter` over every power-of-two divisor, HIP-graph timed | stock `inner_iter=4 / n_groups=8 / grid 512` = 1.000×; **every alternative worse, monotonically in both directions**: 2→0.898×, 1→0.825×, 8→0.728×, 16→0.415×, 32→0.220× | not run | The heuristic is right. Decode DSA moves 75.5 MB in ~34 µs = 2.2 TB/s, ~28% of the part — but that is an indexed gather with essentially no KV reuse across queries, not a launch-geometry problem. More waves buy less than the wider combine pass costs. |
| **Quickreduce for the decode all-reduce** — 8 ranks under `torchrun`, each candidate in a HIP graph of 50 calls | at decode's 768 KB, **`ca` 15.76 µs vs `qr` 32.12 µs — `ca` is 2.04× faster** | would have **cost ~9% of decode ≈ 4.5% end-to-end** | The stock dispatch order is correct at both ends of this workload, not an oversight. Two source gates keep `qr` out and both were fair game; timing them directly settled it for one script instead of two restarts and four bench runs. |
| **Stock Triton sparse-MLA kernel at 8 heads** | **0.603×** the tilelang path (7.326 ms vs 4.415 ms), rel-L2 0.0275 | not run | The `tp_q_head_num == 16` gate is protecting a genuine shape limitation, not an untested path. Reaching it would also need `SGLANG_DSA_TRITON_PREFILL`, an env var and therefore frozen. |
| **Patch 01's guard applied unconditionally** (the first version) | **decode 34.55 → 37.32 µs, 0.926× — an 8% regression** | net ≈ **+0.36%, at or below the 0.39% floor** | **Not a dead end but a near-miss:** unfixed, round 1's largest patch was unreportable. Fixed by the parse-time prefill-only switch. The reusable rule: a change in shared code must be proven not to cost anything in the regime it was not aimed at. |
| **The first-slot guard, round 1** — cheaper predicate, no reduce | **prefill 1.067× vs the reduce's 1.039×**; decode 0.961× (still 4% worse) | extrapolated **+0.64%, never measured** | Rejected on correctness: correct only under an invariant four topk backends would each have to preserve, and a violation silently drops KV keys. |
| **The first-slot guard, round 2** — same change, invariant proven, measured properly | 4.06e9 blocks, **0 violations**; emitted kernel loses one AllReduce and its LDS workspace | **+0.264% against a 0.402–0.718% run-level floor — inside the noise. Runs overlap.** | **Adopted for correctness, NOT claimed as a gain.** Round 1's 0.64% extrapolation was optimistic by ~2.4×. This is the row to read if you are tempted to quote an isolated kernel speedup. |
| **The MoE tuned config tables** (round 2, checked because they were named fair game) | `glm5_fp4_tuned_fmoe.csv` is **complete over the relevant space** — `gfx950`, `cu_num=256`, `model_dim=6144`, `expert=257`, `inter_dim=256`, every power-of-two token count 1…32768 — so decode's ~64 tokens/step and prefill's 16384 both hit a tuned row. `block_m=32` at decode matches the `t32x128x256` in the observed kernel name. | not run | No gap to fill. The MoE expert GEMMs are 22.0% of decode and already tuned for *this* model; re-tuning means re-running aiter's autotuner, a rebuild-and-benchmark project of its own. **A cheap, evidenced negative** — and the last selection-shaped question on this model. |

Two harness bugs that had to be fixed before the split-K table meant anything, both worth recording
because either would have produced confident nonsense:

- **A fake 1.31× for every candidate**, because the reference re-ran `concat_mla_absorb_q_general`
  and the fp8 cast as eager ops on every call while the candidates used a precomputed q. At this
  shape that pair costs more in host dispatch than the attention costs on device.
- **The harness was then host-dispatch bound**: 73.79 µs measured against 38.2 µs of device time in
  the profile. A decode call is ~38 µs of work behind two eager launches, so the CPU set the pace and
  every candidate collapsed to the same number. Fixed by capturing 20 back-to-back calls into a HIP
  graph, which is also what the server does — after which the harness reproduced the profile exactly
  (34.29 µs vs 34.0 µs).

One live defect found in passing and deliberately not patched: **`ca` shadows `qr` between 3 MB and
16 MB**, where `qr` is up to **1.72× faster** (at 12 MB, 101.50 vs 58.93 µs), because
`_resolve_outplace_all_reduce_method` tests `ca` first and thereby shadows a `qr` gate that has
already opened. This workload almost never lands there — 768 KB at decode, 192 MB at prefill, with
the 3–16 MB band reached only by a partial trailing prefill chunk — so it is not worth a patch here.
It would matter at a chunked-prefill size near 1024 or a much larger decode batch. Also worth
keeping: sglang's `_QR_MIN_SIZE` puts the INT4 threshold at 2 MB and the measured crossover is 3 MB,
one shape away — the table is well calibrated on this hardware.

## Sizing: what the profile share is worth, and what it is not

Both rounds tested the same prediction and it failed in the same direction twice, which makes it a
finding rather than an artefact.

| change | profile-share prediction | measured end-to-end | ratio |
| --- | --: | --: | --: |
| A + B (prefill) | 1.039× / 1.018× on an op that is 48.2% of a regime that is 49.7% of the run → ~+1.5% | **+2.00%** | better than predicted |
| C (decode) | 3.08% of decode at a 50.3% decode share → ~+1.55% | **+0.77%** | **half** |
| first-slot guard | +0.64% extrapolated from isolated kernel timings | **+0.264%** | **~0.42×** |

**Treat a profile share as an upper bound on the value of deleting a kernel, not an estimate.**
Removing a 4.6 µs kernel from a 99.7% GPU-busy loop also removes its launch and its share of a
dependency chain, and the remaining kernels do not simply close up. The isolated harnesses predicted
end-to-end well in prefill and badly in decode.

## Latency

Round 1's arms, over the same runs used for the throughput figures. No metric moved the wrong way at
any step, so none of these is a throughput-for-latency trade — and the split is a useful check on
attribution: A+B is a prefill change and takes most of its gain in TTFT, while C is decode-dominated
and inverts the ratio.

| | baseline (n=7) | A+B (n=9) | A+B+C (n=5) | C alone | total |
| --- | --: | --: | --: | --: | --: |
| mean TTFT | 11492.7 ms | 11085.1 ms | 11052.8 ms | −0.29% | **−3.83%** |
| mean TPOT | 32.38 ms | 31.93 ms | 31.63 ms | −0.94% | **−2.31%** |
| p99 TTFT | 22224.0 ms | 21426.9 ms | 21360.0 ms | −0.31% | −3.89% |
| p99 TPOT | 42.88 ms | 41.92 ms | 41.61 ms | −0.74% | −2.96% |

**Round 2 records no latency table.** Per-run `mean_ttft_ms` and `mean_tpot_ms` are in every
`results/*/inferencex_result.json` in its bundle, but no arm means were computed, and this entry does
not derive them after the fact. That is a gap against this template.

## How to get a usable profile on this stack

No usable profile existed before round 1 — both TraceLens attempts failed identically
(`steady_state_chunk_empty: requested --steady-state-mode=mixed but the selected chunk has zero GPU
events`), leaving only two graph-capture traces at batch 1, concurrency 1, which describe nothing
about this workload. Producing one was a deliverable. Three things were needed, and all three
generalise.

**Capture without touching the configuration.** The environment is frozen, so the profiler cannot be
enabled by env var. SGLang's `/start_profile` HTTP endpoint accepts `output_dir`, `num_steps`,
`activities` and `record_shapes` in the POST body — no env var, no flag, so the server under the
profiler is configuration-identical to the server under measurement. Drivers:
`artifacts/round1/analysis/profile_decode_stack.py` and
`artifacts/round2/analysis/profile_decode2.py` (the same script with a `--stack` toggle, so one tool
serves both the timing and the attribution capture).

**Segment the two regimes, or the profile is worse than none.** Round 1's first capture was
mis-segmented and reported "prefill 100.00% `main_kernel`", which is impossible. The diagnosis came
from counting layer-characteristic kernels: `alloc_extend_kernel` 11 times,
`write_req_to_token_pool_triton` 11 times, `mfma_moe` 900 times = 12 × 75 MoE layers — **all 12
captured passes were prefill chunks**, and the two duration clusters the segmenter had split on were
tilelang's *partial* and *combine* kernels, **both of which are emitted as `main_kernel`**. A
duration-bimodality split therefore mis-classifies everything. A second capture was taken during pure
decode. This matters directly: patch 03's target is 3.08% of decode and ~0.1% of a prefill chunk, so
the mis-segmented profile would have dismissed it outright.

**Attribute a kernel to a line by capturing during prefill.** Decode runs inside a HIP graph, so a
`with_stack` decode trace carries 6884 kernels against 517 `cpu_op`s, all tagged
`"kind": "Dispatch Task"`, with no Python frames at all. Prefill runs eagerly and executes the same
layer code: 68202 `cpu_op`s with full `python_function` stacks. See the round-2 section above for the
linking chain, which is build-specific and must be *discovered*, not copied — histogram the event
categories and arg keys actually present in your trace before writing any linker.

**A profiled run is not a measurement run, and the residue outlives it.** Round 1's two
profiler-driver runs (1074.74 and 1152.675 tok/s) are kept only because they are where the traces
came from, and the first bench *after* a capture on the same process was 0.9% low. Profile on a
server you then tear down.

## When this entry stops applying

Silently, in every case except the fourth:

- **`gfx950` or CU count ≠ 256** — 256 is a literal in the split-K one-wave grid and in the MoE
  table key. Do not deploy; the analysis behind three of the negative results stops holding.
- **SGLang commit ≠ `2948168546`** — the patches are line diffs against three files. Fall back to the
  whole-file copies, then re-verify engagement, then re-measure.
- **TP ≠ 8** — `tp_q_head_num` moves off 8, patch 02's shape gate may fail into its stock fallback,
  and the geometry findings (§5, §8) are about 8 heads specifically.
- **`--dsa-topk-backend` ≠ `sgl-kernel`** — **not silent-inert but a correctness hazard.** The
  round-2 guard drops live KV keys under `flashinfer`. Enable
  `SGLANG_DSA_ASSERT_TOPK_SUFFIX=1` or revert to patch 01's reduce.
- **KV cache dtype ≠ `fp8_e4m3`, or page size resolves to anything but 64, or the attention backend
  is not `dsa`** — the tilelang DSA kernels are not the ones running and 01/02/G are unreachable.
- **`--chunked-prefill-size` < 1024, or a decode step above 1024 query tokens** — the guard compiles
  out (no gain) or compiles in at decode (4–8% loss), both silently.
- **A checkpoint that does not exclude `kv_b_proj` from quantization** — 03 and the absorb fix are
  both correctly inert, because the aiter fp8/MXFP4 absorb branches already write token-major.
- **Concurrency, ISL or OSL changed** — the absorb win is decode-side and scales with layer count,
  not batch, so it is worth more at low tokens-per-step and nothing at very large ones; patch 01's
  12.1% dead-block figure is a function of ISL 8192 against `index_topk` 2048.
- **A different tilelang or torch build** — the guard's engagement check reads tilelang's HIP cache
  format, and the absorb patch's strided `out=` selects a rocBLAS kernel whose accumulation order is
  not guaranteed to be the one gsm8k was measured on.

**Still reusable when inert**, and this is most of the entry's value on a near-miss:

- The **regime split and the two budget tables** — where the time goes on a DSA/MLA model at this
  operating point.
- The **method that found all five patches**: profile, take the largest kernel whose *name* does not
  explain it, attribute it to a source line, and ask whether it needs to exist. Every change that
  worked here removed provably dead work; every change that tuned a number failed.
- `artifacts/round2/analysis/attribute.py` — the kernel-to-source-line tool, and the note that its
  linking chain is build-specific.
- `artifacts/round1/analysis/tl_dsa_blockskip.py` — the standalone block-skip analysis and
  correctness argument, with `skip_mode="first"` as well as `"reduce"`.
- `artifacts/round2/probe/dsa_backend.probe_suffix.py` — the live-tensor invariant prober. Overlay it
  on the round-1 applied tree (`set_arm.sh probe_suffix`) to re-verify the suffix invariant against
  any topk producer on any workload. This is the transferable answer to "I have an invariant I cannot
  close by reading code."
- The **isolated harnesses**, all HIP-graph timed and interleaved:
  `bench_dsa_prefill.py`, `sweep_dsa_tl.py`, `sweep_dsa_decode.py`, `bench_allreduce.py`.
- The **measurement protocol**: interleave arms across restarts, re-derive the floor on your own
  boot, report run-level *and* session-mean spreads separately, and do not discard the first run on
  this stack.
- `artifacts/round2/analysis/summarize.py` and `session.sh` — the audit and the session runner,
  including the two `EADDRINUSE` guards.

## Gaps — what the bundles did not record that this template asks for

Listed so the next reader does not go looking.

- **Container image digest: not recorded.** Only a probable tag, and round 1 states explicitly that
  the image could not be confirmed from inside the running container — not in the environment, not in
  `/etc/`, not in any session state. Treat the tag as unverified.
- **aiter version and commit: not recorded.** Only the path `/sgl-workspace/aiter`, noted as a source
  checkout. This matters more than usual, because aiter supplies 22.0% of decode (the MoE) and the
  `ca` all-reduce, and both were checked as already-optimal against *this* build.
- **tilelang version or commit: not recorded.** Only `/opt/tilelang/build`.
- **Round 2 ran no stock arm**, so round 1's +2.78% was not re-derived on round 2's boot. What round 2
  reproduced is the round-1 *level*, not the round-1 *delta*.
- **Round 2's +1.210% has not been re-measured on a second boot**, and neither round-2 patch has been
  re-measured on a clean instance by anyone other than the run that produced it.
- **Round 2 records no latency table.** The per-run values exist in its result JSONs.
- **No accuracy figure exists for the baseline configuration prior to round 1** — the source session
  evaluated only the untouched configuration. Round 1's baseline gate is the first one.
- **The `~/.triton/cache` and `/tmp/aiter_configs` hazards were never exercised.** Their irrelevance
  here is inferred from what the patches touch, not observed.
- **Round 1's result directories carry no `server_info.json`**; only round 2's do. Round 1's live
  configuration is attested by `launch_server.sh` printing `config verified`, not by a captured
  snapshot.
- **`patches/base/` and `patches/applied/` are byte-identical between the two bundles** (verified by
  md5), so `artifacts/round1/base/` and `artifacts/round1/applied/` serve both rounds. Round 2 took no
  new pristine copies because it edits the same three files.

## Artifacts

```
artifacts/
  round1/
    patches/01-dsa-prefill-skip-empty-index-blocks.patch      +1.33%, bit-exact
    patches/02-dsa-prefill-fused-q-fp8-prep.patch             +0.72%, bitwise identical
    patches/03-mla-absorb-drop-unit-weight-scale.patch        +0.77%, numerically inert
    patches/04-dsa-prefill-first-slot-guard.REJECTED.patch    round 1's rejected variant, carrying
                                                              round 2's forward-pointer annotation
    patches/tilelang_kernel.first-slot.py                     that variant's whole file
    base/{tilelang_kernel,dsa_backend,forward_mla}.py          pristine at 2948168546
    applied/{tilelang_kernel,dsa_backend,forward_mla}.py       stock + 01/02/03 — the round-1 arm,
                                                              and the base of both round-2 diffs
    analysis/tl_dsa_blockskip.py                               block-skip analysis + both skip modes
    analysis/bench_dsa_prefill.py                              isolated DSA prefill harness
    analysis/sweep_dsa_tl.py, sweep_dsa_decode.py              the two geometry sweeps (§5, §6)
    analysis/bench_allreduce.py, allreduce_bench.txt           the 8-rank all-reduce bench (§7)
    analysis/trace_{summary,detail,regime}.py                  trace post-processing + regime split
    analysis/profile_decode_stack.py                           /start_profile driver, with_stack
    analysis/profile_conc64_tp0.json, profile_decode_tp0.json  the two corrected profiles
  round2/
    patches/dsa-prefill-first-slot-empty-block-guard.patch     +0.264%, NOT CLAIMED; stacks on 01
    patches/mla-absorb-bmm-write-token-major.patch             +1.210%, claimed; textual dep on 03
    arms/fs/{tilelang_kernel,dsa_backend}.py                   the exact files that produced `fs`
    arms/fl/{tilelang_kernel,dsa_backend,forward_mla}.py        the exact files that produced `fl`
    probe/dsa_backend.probe_suffix.py                          live-tensor suffix-invariant prober
    analysis/attribute.py                                      kernel -> source line
    analysis/profile_decode2.py                                /start_profile driver, --stack toggle
    analysis/profile_decode_r2_tp0.json                        decode profile, before (`fs`)
    analysis/profile_decode_fl_tp0.json                        decode profile, after (`fl`)
    analysis/set_arm.sh, session.sh, summarize.py               arm switch, session runner, audit
  harness/launch_server.sh, run_bench.sh, run_eval.sh          the measurement contract, unmodified
```

`arms/fs/` holds only two files because the run's arm switcher overlays them on the round-1 applied
tree; `fs`'s `forward_mla.py` is `round1/applied/forward_mla.py`.

## Provenance

**Round 1:** `tuning_workspace/experiment_standalone/glm_52_mxfp4_tuning/`. `FINDINGS.md` — local
baseline and both spreads, the accuracy gate and its own noise, "Getting a usable profile" and the
corrected budget tables, §1/§1a/§1b candidate A and the rejected first-slot variant, §2 candidate B,
§3 A+B, §4 candidate C, §5–§8 the four negatives, the latency table, Conclusions and Open threads.
`BASELINE.md` — where 1462.337 comes from, the `--block-size`/`--page-size` story, and the caveats.
`patches/` — the three adopted diffs with measurement and accuracy in each header, plus
`rejected/04`. `results/` — 28 bench runs. `eval_results/` — three gsm8k gates. `analysis/` — profile
drivers, post-processing, five isolated harnesses, saved traces.

**Round 2:** `/shared_nfs/ethany/home/tuning_workspace/experiment_standalone/glm_52_r2_tuning/`.
`FINDINGS.md` §0 the reference level on this boot, §1a–§1f the invariant (reachability, the observed
producer, the probe, decode inertness, the latent `flashinfer` hazard), §2 the guard's end-to-end
measurement and why it is not claimed, §3 the noise floor and the drift, §4a–§4h lead 2 from the
fresh profile to the exported patch and what was left on the surface, §6 reproducing and the pristine
bases, §7 the skillset assessment. `round1/` — round 1's archived `FINDINGS.md`, `PROMPT.md`,
`analysis/` and `results/`. `results/` — 33 on-contract runs across `r1ref_*`, `fs_*`, `fl_*`, plus
`probe_suffix_*`, each with a `server_info.json`. `eval_results/` — `fs_gate_20260820_202453` and
`fl_gate_20260820_215927`. `patches/` — both round-2 diffs, `arms/` for every measured arm,
`rejected/04` re-annotated.
