# Llama-3.1-8B-Instruct on MI355X — SGLang + aiter, TP=1, ragged-gather prefill attention

**Measured win: +3.185% ± 0.916% output throughput** (2640.618 → 2724.708 tok/s), gsm8k 5-shot strict-match
0.8431 ± 0.010 → **0.8491 ± 0.0099** against a ≥ 0.8213 gate. Mean TTFT 3986.8 → 3571.7 ms
(−10.4%), mean TPOT 20.353 → 20.011 ms (−1.7%). The win is carried by **one framework source
patch**, `artifacts/0004-prefill-attn-ragged-gather.patch`, which gathers the paged KV span
contiguously so that aiter's hand-written gfx950 v3 ASM prefill kernel becomes reachable.

Found 2026-08-20 on `crsuse2-m2m-172`, in round 2 of a two-round campaign.

## Reproduction status — read this before quoting the number

**Not reproduced from the artifact alone on a clean instance.** This entry does not meet the
directory's own bar (`../README.md`, "Only reproduced results go in") and is filed the way the
DeepSeek entry is: measured, honestly caveated, with the promotion steps listed at the end.

What *does* stand behind the number:

| evidence | strength |
| --- | --- |
| Six restart-paired, order-flipped rounds, one bench run per fresh server, all six paired deltas positive (min +1.524%, max +4.332%) | strong — and the arm ranges are **disjoint** (base 2628.009–2664.149, candidate 2684.980–2757.107) |
| Engagement classified correctly on **12 of 12** arms from the scheduler log *of the benchmark run itself*, with no overlap between the two prefill-throughput bands | strong, and it doubles as proof no arm was ever half-applied across twelve apply/revert cycles |
| An independent single smoke run on a fresh server, 2723.804 tok/s (`results/r2rgv2_smoke_20260820_162817`) | consistent |
| Applying the exported patch to the recorded pristine base reproduces the measured candidate file **byte-for-byte** (sha256 `19dcc694390ed23b9b5e34c694deea8197799e75e6f5f38a60f6c12910e10c3f`; base is `45b09e64f84defea67aa30a0c26c42c63322eaf0ddd845d8b59a994197c23180`) — verified while writing this entry | the artifact is the thing that was measured; this is *not* a performance reproduction |

What does not: nobody has started from a pristine tree on a second instance, applied
`artifacts/0004-…patch`, and re-derived +3.185% with its own restart floor. Budget an hour and do
that before you rely on this entry.

One caveat the run stated rather than buried: **the six base runs of the winning campaign span
1.37%** (sd 0.48%), much worse than the 0.09–0.14% restart floor inherited from round 1. The
paired, order-flipped design absorbs it — the effect is 6.6× the sd of the paired deltas — but a
future round quoting a 0.3% result must re-derive the floor rather than inherit it. And r5's
+1.524% is a genuine outlier against the other five (+3.2% to +4.3%) that the run could not explain;
it is reported, not dropped, and it is what widens the sd.

## Why this model matters more than its 3%

Llama-3.1-8B was the campaign's **standing null**. A configuration-space search spent four hours
forty-nine minutes and kept nothing — the accepted stack is empty, cumulative gain `0.0`, best still
tagged `baseline` (`BASELINE.md`). Round 1 then spent a full code-level session on seven measured
attempts and also kept nothing; all three of its patches are in `artifacts/rejected/`. Round 2 found
+3.185% **on its first real candidate**.

Three things made the difference, and they are the transferable part of this entry:

1. Round 1 spent itself almost entirely inside the **19% of decode** taken by dense GEMMs, and
   **never profiled prefill at all** — which is 28.7% of the wall clock here.
2. It was given the **wrong noise floor** (1.49% instead of 0.09–0.14%) and discarded candidates on
   size grounds that it should have measured.
3. The lead that found the win arrived **from another model on the same cluster** (see
   "Cross-model transfer", below), and was wrong in its specifics while being right about the region.

## Environment fingerprint

Diff this before deploying. If a **load-bearing** field differs, the patch's predicate stops
matching and the branch is skipped silently — the server serves correctly, at the old speed.

**This is a TP=1, single-GPU configuration.** Every other entry in this directory except Qwen3-8B is
multi-GPU, so the usual per-rank engagement counting does not apply here: there is exactly one rank,
and "one rank failed to engage" is not a failure mode you have to exclude.

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | **1×** MI355X, `gfx950`, CDNA4, **256 CU** (one of eight on the node, `renderD128`) | **yes** — the patch's branch is gated on `is_gfx95_supported()`; on any other arch it is skipped and stock prefill runs. 256 CU is the `cu_num` key column in every aiter config row on this box |
| node | `crsuse2-m2m-172` | descriptive — but the noise floor is a property of the *node*, not the model (`../README.md`) |
| container | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` | descriptive. **Image digest not recorded** — a real gap for exact matching; the source session recorded no image at all and `preflight.sh` cannot assert one |
| SGLang | **0.5.17**, installed as a plain package | **yes** — the patch is a source diff against `python/sglang/srt/layers/attention/aiter_backend.py` from this tree, and it inserts at a specific anchor. **No commit sha exists**: there is no git metadata, which is why the base is a pristine `cp` (`artifacts/base/aiter_backend.py.orig`) rather than a commit |
| aiter | commit `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | **yes** — it supplies both the incumbent `mha_batch_prefill_func` and the replacement `flash_attn_varlen_func` path, and it is the package that ships `hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_causal_group.co`. No ASM kernel, no win |
| torch / ROCm / Triton / python | 2.9.1+rocm7.2.0.git7e1940d4 / 7.2.0 / 3.6.0 / 3.10.12 | descriptive — Triton is not on this patch's path at all (see the Triton decode result below for why that matters) |
| model | Llama-3.1-8B-Instruct, `LlamaForCausalLM`, 32 layers, hidden 4096, GQA 32 q → 8 kv heads, **head_dim 128**, vocab 128256, **TP=1** | **yes** — the branch tests `layer.qk_head_dim == 128 and layer.v_head_dim == 128` literally, and the ASM kernel exists only for hd128. TP=1 means no sharding changes the head dim |
| precision | **bf16 weights, bf16 KV cache** | **yes** — the branch requires `q.dtype == torch.bfloat16` and `self.kv_cache_dtype != fp8_dtype`. On an fp8 KV cache the pre-existing fp8 branch handles it and this one never runs |
| attention backend | **`--attention-backend aiter`** | **yes** — the patched file *is* the aiter backend. Any other backend and the artifact is inert |
| page size | **`--page-size 1`** | **yes, and it is the whole story** — see "Why `--page-size 1` is why this exists" |
| KV layout | not vectorized-5d (`self.kv_cache_is_vectorized_5d` false) | **yes** — an explicit clause in the predicate |

No config label was found to disagree with what ran. `scripts/launch_server.sh` verifies the live
server's `ServerArgs` against the intended values through `/get_server_info` and refuses to let you
benchmark on a mismatch; it prints `config verified`.

## Launch configuration

Reproduce this **verbatim** — `scripts/launch_server.sh` in the bundle, unmodified:

```bash
MODEL=/shared_nfs/hyperloom/models/Llama-3.1-8B-Instruct
PORT=43101

python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 1 \
    --context-length 11264 \
    --watchdog-timeout 1800 \
    --mem-fraction-static 0.68 \
    --chunked-prefill-size 16384 \
    --page-size 1 \
    --disable-radix-cache \
    --attention-backend aiter
```

Resolved values that are not visible in the invocation, recovered from the live server rather than
assumed:

- **`mem_fraction_static` may read back as 0.578, not 0.68.** SGLang rescales it by 0.85 on builds
  that combine aiter with a context length above 8192, so the launcher accepts either value. A
  reader who asserts 0.68 exactly will fail a correct server.
- **`max_num_partitions` is computed once from `--context-length 11264` → 44**, while the workload
  never exceeds 9216 tokens → 36. The extra blocks early-return; round 1 measured this as a
  non-issue (~4096 empty block dispatches over 256 CUs).
- The four flags that do not look like tuning knobs — `--mem-fraction-static`,
  `--chunked-prefill-size`, `--disable-radix-cache`, `--attention-backend` — were set by the
  *harness*, not by SGLang defaults. They are part of the configuration you are comparing against,
  and `--chunked-prefill-size 16384` in particular is load-bearing for this win in a way nobody
  expected (see the BOS-token story).

**Environment variables: none are set by the launch script, and the frozen contract forbids adding
any.** That is itself a finding — this stack has no env recipe, unlike Qwen3-8B's six aiter
variables. **Gap:** the run never dumped `/proc/<pid>/environ` for the live server, so what the
*container image* exports is not recorded. `../README.md` warns specifically about this ("the launch
script setting no environment variables does not mean the process has none"), and it is the one
preflight step this campaign skipped. Do it before assuming a clean environment.

One env var appears in this entry and is **not** part of the recipe: `SGLANG_USE_AITER_UNIFIED_ATTN`
was flipped in an off-contract probe to measure the Triton decode path (rejected, below). Leave it
unset.

## Workload

| | |
| --- | --- |
| harness | InferenceX `benchmark_serving` fork, in the bundle's `bench/` |
| ISL / OSL | 8192 / 1024 |
| concurrency | 64 (`--max-concurrency`) |
| prompts | 192 |
| warmups | 8 |
| dataset | random, `--random-range-ratio 1.0`, `--random-prefix-len 0`, `--ignore-eos`, seed 0 |
| endpoint | `/v1/completions`, backend `vllm` |

**Never mix rigs.** A second benchmark rig measured this identical configuration 8.71% faster
(2813.63 tok/s). Every figure in this entry comes from `bench/` against a baseline from `bench/`.

What the workload parameters set, and why it matters here more than usual:

- **ISL 8192 against `--chunked-prefill-size 16384` is what creates the batch shape the win is
  tuned for.** The prompts are exactly 8192 tokens (`total_input_tokens` 1572864 = 192 × 8192), the
  server's tokenizer prepends BOS, and 2 × 8193 = 16386 overshoots the 16384-token budget by two
  tokens. From the second batch onward **every** prefill batch is
  `[2-token continuation | whole 8193-token request | ~8189-token head]` — three sequences, one of
  them carrying a prefix. The dominant census line under `bench/` is
  `#new-seq: 3, #new-token: 16384, #cached-token: 0`, 92 of them.
- Concurrency 64 with `--ignore-eos` makes the run **three waves**, not an interleaved stream: 64
  requests in flight, all 1024-token outputs finishing together, so each wave is a prefill burst
  followed by 1024 decode steps. A 200-step profiler window taken 40 s in contains **zero** prefill
  kernels, which is how the wave structure was confirmed.
- Phase split, from that trace: decode 17.28 ms/step × 3072 steps = 53.1 s of a 74.6 s run, leaving
  **~28.7% prefill**. Within a prefill batch: GEMM 66.0%, **attention 23.3%**, silu 5.08%, rmsnorm
  2.25%, rope 1.59%, elementwise 0.73%, store_kvcache 0.34%. So prefill attention — the thing this
  patch replaces — is roughly **6.7% of wall clock**, and the largest single thing round 1 never
  looked at.

## Baseline and noise floor

### The arms

| | tok/s |
| --- | --- |
| reference figure, another rig-day (`BASELINE.md`) | 2588.104 |
| round 1, local, 6 fresh-server run-#1s (§7 campaign) | 2640.788 |
| round 2, local, single fresh-server run #1 (`results/r2_base_probe1_20260820_151927`) | 2636.160 |
| **stock, 6 fresh-server run-#1s interleaved with the candidate** | **2640.618** |
| **with `0004-prefill-attn-ragged-gather.patch`, same 6 restarts** | **2724.708** |
| **delta** | **+3.185% ± 0.916%** (sd of the six paired deltas; SEM 0.374%) |

Every round, nothing dropped:

| round | base tok/s | cand tok/s | Δ | TTFT ms | TPOT ms |
| --- | --: | --: | --: | --: | --: |
| r1 | 2636.608 | 2721.170 | +3.207% | 3971.3 → 3573.0 | 20.406 → 20.039 |
| r2 | 2634.722 | 2719.865 | +3.232% | 3986.8 → 3588.9 | 20.406 → 20.033 |
| r3 | 2664.149 | 2757.107 | +3.489% | 3985.1 → 3552.7 | 20.142 → 19.750 |
| r4 | 2635.536 | 2749.712 | +4.332% | 3977.3 → 3583.8 | 20.409 → 19.783 |
| r5 | 2644.684 | 2684.980 | +1.524% | 3993.5 → 3556.2 | 20.308 → 20.373 |
| r6 | 2628.009 | 2715.415 | +3.326% | 4007.0 → 3575.9 | 20.448 → 20.085 |
| **mean** | **2640.618** | **2724.708** | **+3.185%** | 3986.8 → 3571.7 (−10.4%) | 20.353 → 20.011 (−1.7%) |

**The two arms are disjoint**: the worst candidate run (2684.980) is above the best base run
(2664.149). That is a stronger statement than the margin, and it is what makes a result with a 0.916%
sd defensible.

### The noise floor, and the direction that surprised everyone

| noise floor | spread | how measured |
| --- | --: | --- |
| repeating the benchmark **within one process** | **1.20%**, and still falling at run 5 | `artifacts/spread_within.sh`, one server start, five consecutive `run_bench.sh` runs: 2639.832 → 2635.880 → 2625.267 → 2621.238 → 2608.093 |
| **across restarts**, always taking run #1 | **0.09 – 0.14%** | round 1's two 6-restart base arms: mean 2640.788, sd 1.381, spread 0.142%; the other campaign's base arm 0.09% |

**Restart-to-restart is about 8.5× TIGHTER than within-instance here.** That is the opposite of what
`../../tuning-core/measurement.md` Rule 3b's worked examples show (26× and 1.1× *wider*), the opposite of
what the surrounding prose implies ("restarting adds variance"), and the opposite of what every other
run in this campaign assumed. **Check which spread is actually larger on your stack before you design
an A/B. Do not inherit the direction from anywhere, including from here.**

The mechanism is mechanical and worth understanding, because it predicts when the inversion happens:

- Throughput **falls** within an instance, monotonically, over five consecutive runs, and had not
  plateaued by run 5. Whatever drives it — thermal, or KV-pool/allocator state accumulating across
  runs — it is **directional, not random**, so it does not average out. Note that this is the
  opposite of MiniMax's and DeepSeek's climb-to-plateau behaviour, where the *first* run is the bad
  one; here the first run is the good one.
- Every restart sample is taken at the same point on that settle curve (run #1, cold instance), so
  the drift that dominates the within-instance number is held **constant** rather than sampled.

Two consequences, both of which shaped everything above:

1. **Take exactly one benchmark run per server start, always run #1.** One run per server lifetime is
   not wasteful; it is the control. `artifacts/ab_campaign.sh` does this, and flips the arm order
   every round so drift cancels rather than landing on whichever arm goes first.
2. **The applicable floor here is the restart floor**, because the change is a source edit consumed
   at import and HIP-graph capture time — Rule 3b's own criterion. At 0.14%, the +3.185% headline is
   about 23× the floor and even the worst single round (+1.524%) is 11×.

**Round 1 was given the wrong floor and paid for it.** Its brief said to treat **1.49%** as the
floor, a number that came from four ad-hoc runs at mixed run indices on a warm instance — i.e. it was
mostly the within-instance settle curve, misread as a restart floor. At 1.49% a 0.3% candidate is
unmeasurable and gets dropped without a campaign; at 0.14% it is reportable. The concrete cost:
round 1 left the small-kernel bucket (rmsnorm / silu / rope / elementwise, which it measured at 3.7%
of decode) **unattempted on size grounds** — "even a 20% win there is 0.7% end-to-end, and §4 says
that would not survive the server". At the corrected floor that bucket is worth 26× the floor, and
round 2 re-measured it at **6.7% of decode, larger than advertised rather than smaller**. It is now
the strongest open lead on this model, and it stayed closed for a whole session because of a floor
that was off by an order of magnitude.
The same misreading produced the largest single number of round 1: the identical configuration
measures 2595.555 cold-and-ad-hoc versus 2638–2641 on the one-run-per-fresh-server cadence, **+1.94%
from changing nothing but when the benchmark was run**.

Round 2's own data agrees with the inherited floor where it was measured under the same discipline:
across the three restart-paired probe rounds the within-arm spread was 0.10% (candidate) and 0.13%
(base). The winning campaign's base arm, on a different day-part, spread 1.37% — reported above as
the caveat it is.

## The win, and the gap between the kernel number and the result

### Mechanism

SGLang's aiter backend serves Llama prefill with **`mha_batch_prefill_func`** (ck_tile), which walks
the paged KV pool **through a page table inside the softmax loop**. aiter also ships a hand-written
**gfx950 v3 ASM kernel for exactly this shape** — `hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_causal_group.co`,
bf16, head_dim 128, causal, grouped — reachable through `flash_attn_varlen_func`. The backend even
has a branch that uses the ragged path already, but it is an **fp8 / head_dim-256** branch gated on
`not any(forward_batch.extend_prefix_lens_cpu)`. Llama is bf16 / head_dim-128, so it never qualifies.

The ASM kernel **cannot read the paged pool at all** on this configuration: `flash_attn_varlen_func`
accepts a `block_table`, but at `--page-size 1` it raises *"Paged KV cache block size must be
divisible by 128"* — recorded as a hard failure in `artifacts/data/prefill_attn2.json`, not as a
slow arm.

So the patch **materialises the KV span contiguously first**: two `index_select`s from the pool into
a persistent scratch pair, then `flash_attn_varlen_func` with `cu_seqlens_k != cu_seqlens_q` — causal
attention is bottom-right aligned in that form, which is precisely the correct mask for a chunk
continuation. The gather moves ~100 MB per layer and is **inside the timed region** of every number
below.

### The numbers, kept separate on purpose

| level | measurement | value |
| --- | --- | --- |
| **kernel, isolated** | `artifacts/bench_prefill_attn2.py`, at the shape the sealed benchmark actually produces (3 seqs, q 2/8193/8189, kv 8193/8193/8189, production-sized 1,236,694-token pool, graph-captured, interleaved) | paged 1724.25 µs → gather + v3 ASM 1082.51 µs = **+37.22%**, gather included |
| **phase, in the live server, during the benchmark** | scheduler `input throughput` line, n≈92 batches per run, 6 restarts per arm | 70616 → 79019 tok/s = **+11.90%** (base band 70531–70687, cand band 78594–79507 — disjoint) |
| **end to end** | `bench/`, 6 restart-paired rounds | 2640.618 → 2724.708 = **+3.185% ± 0.916%** |

**Why the gap.** Prefill is 28.7% of the run, so +11.90% on the prefill phase predicts about **+3.5%**
end to end against the +3.185% measured — model and measurement agree inside their own error bars,
which is the check that was missing in the first attempt. The isolated +37.22% shrinks to +11.90% at
the phase level because prefill attention is only 23.3% of a prefill batch (GEMM is 66.0%), and to
+3.185% overall because prefill is 28.7% of the wall. **That is Amdahl's share, not a measurement
failure** — but +37.22% is the number never to quote as the result.

An arm-position control (`paged_self`, a byte-identical copy of the incumbent registered last) came
out at −0.08%, so the isolated race is not reporting harness bias. Numerics against the paged arm:
**max_rel_err 6.443e-04, cos 0.99999976**.

Worth noting the sign of the phase gain: **+11.90% under the benchmark exceeds the +8.2%** the
earlier batch-gated version got on a chunk-free probe. That is the right direction — the chunked
batches the benchmark produces are the *worse* case for the paged kernel (three sequences, one of
them reading a full 8193-token KV span to compute 2 query rows), so they are the *better* case for
replacing it.

### Why `--page-size 1` is why this exists

The single most useful sentence to carry to another model on this stack: `--page-size 1` is
simultaneously what makes aiter's fastest prefill kernel unreachable through the paged dispatcher
(this win) and what makes the Triton decode kernel unusable (a 6.33× loss, below). **The gap is not a
missing optimisation inside a kernel; it is a dispatcher that gives up when the KV is not already
contiguous, when making it contiguous costs about 6% of what the better kernel saves.** Before
tuning anything on a frozen configuration, enumerate the branches in the stack that key on page size,
dtype, head dim and alignment: the config has already pre-selected candidates for you. Doing that
took an hour here and produced both of round 2's decisive results.

## Deploy

The patch touches exactly one file:
`/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py`. It touches **no** aiter
config CSV, no Triton kernel, and nothing that needs rebuilding.

```bash
# 1. Stop the server. The change is consumed at import time and at HIP-graph capture time;
#    applying it to a live server is a silent no-op that benchmarks perfectly.
./scripts/launch_server.sh --stop

# 2. Apply. The diff carries ABSOLUTE paths, so -p0 targets /sgl-workspace/... directly and
#    the working directory is irrelevant.
patch -p0 < artifacts/0004-prefill-attn-ragged-gather.patch

#    Or, reversibly — this is what the A/B campaign used twelve times, and it also handles
#    step 3 for you. It needs artifacts/base/ next to it.
python3 artifacts/deploy_ragged_prefill.py apply     # | revert | status
#    -> prints e.g. "apply: live=19dcc694390e -> cand(ragged)"

# 3. Invalidate stale bytecode. A whole-file copy can defeat mtime-based invalidation, and a
#    stale .pyc is a restart that imports the old code with no warning.
rm -f /sgl-workspace/sglang/python/sglang/srt/layers/attention/__pycache__/aiter_backend.*.pyc

# 4. Caches that this patch does NOT need, but that anything you stack on it will:
rm -rf /tmp/aiter_configs   # aiter merges configs/*.csv + model_configs/* into this and
                            # regenerates it ONLY when absent -> a stale merge silently
                            # serves the old table. Mandatory for the rejected GEMM patches.
rm -rf ~/.triton/cache      # only if you have touched a Triton kernel; nothing on this
                            # model's hot path is Triton (see the Triton decode result).

# 5. Restart — mandatory.
./scripts/launch_server.sh          # must print "config verified"
```

Verify what is live at any time:

```bash
python3 artifacts/deploy_ragged_prefill.py status
# cand(ragged) | base(paged) | UNKNOWN  (UNKNOWN means a hand edit crept in — stop and fix it)
sha256sum /sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py
# 19dcc694390ed23b9b5e34c694deea8197799e75e6f5f38a60f6c12910e10c3f  = patched
# 45b09e64f84defea67aa30a0c26c42c63322eaf0ddd845d8b59a994197c23180  = pristine
```

To regenerate the candidate file (and hence the patch) from the pristine base rather than trusting
the diff: `python3 artifacts/make_ragged_patch.py`, which performs a scripted, idempotent text
insertion at a fixed anchor.

### Every way this deploy silently does nothing

Each of these produces a clean, plausible benchmark at the *old* speed. None of them errors.

1. **Applying it to a running server.** Python has already imported the module and the decode graphs
   are captured. Restart or you are measuring the old path.
2. **A stale `__pycache__/aiter_backend.*.pyc`.** The deploy script deletes it deliberately; a manual
   `patch` does not.
3. **Patching a different tree than the one on `sys.path`.** SGLang is installed as a plain package
   and the patch hard-codes `/sgl-workspace/sglang/python/...`. Confirm with
   `python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"` — `preflight.sh` prints
   this for exactly this reason.
4. **Any single clause of the predicate failing.** The branch requires *all* of:
   `is_gfx95_supported()`, extend mode, `window_size == (-1, -1)`, `sinks is None`,
   `logits_soft_cap == 0.0`, `qk_head_dim == 128`, `v_head_dim == 128`, KV dtype not fp8, KV layout
   not vectorized-5d, `q.dtype == torch.bfloat16`, and both `forward_metadata.kv_indices` and
   `forward_metadata.max_kv_len` present. Miss one and control falls through to the paged call with
   no log line and no error.
5. **A different attention backend.** The file is only imported when `--attention-backend aiter` is
   in force.
6. **A workload whose prefill share is small.** The gain is +11.90% on a phase worth 28.7% of the
   wall. Halve the ISL, or run a decode-heavy operating point, and most of the win goes with it.
7. **Measuring on run #2 or later against a warm server.** The within-instance settle curve is 1.20%
   and directional; comparing a candidate's run #3 against a baseline's run #1 mixes a −1% drift into
   a +3% effect and can produce any answer you like.
8. **Quoting against the wrong baseline.** 2588.104 is another rig-day's figure and 2813.63 came from
   a different harness entirely. Both are 2–8% away from the local `bench/` baseline, i.e. larger than
   this win.

## Engagement check

**The lesson this run paid two hours for: check engagement under the contract workload, not under a
reproduction of it.** The first version of this change passed the strongest engagement check
`../../tuning-core/engagement_verification.md` describes — profiler kernel identity, new kernel
present and old kernel absent — while firing on
**0 of 92** of the benchmark's prefill batches. So run all three checks below, in this order.

### 1. Kernel identity from a profile (the check to publish)

The change swaps which attention kernel is dispatched, so kernel identity is the right primary
evidence, not a log line.

Capture a profiler window while the server is under load, then count the two kernel names in the
exported trace:

```bash
zcat <trace>.trace.json.gz \
  | grep -o -e 'fmha_fwd_hd128_bf16_causal_group' -e 'FmhaBatchPrefill' | sort | uniq -c
```

| | expected output |
| --- | --- |
| **engaged** | `fmha_fwd_hd128_bf16_causal_group` present (the gfx950 v3 ASM kernel, reached via `flash_attn_varlen_func`), **zero** `FmhaBatchPrefill` — the ck_tile paged kernel |
| **not engaged** | the exact mirror image: `FmhaBatchPrefill` present, `fmha_fwd_hd128_bf16_causal_group` absent |

Both directions were observed. **Gap: the exact profiler invocation is not recorded in the bundle** —
`FINDINGS.md` cites trace paths (`/tmp/r2prof/r2p1-TP-0.trace.json.gz`, `/tmp/r2prof3`) but not the
command that produced them. `bench/benchmark_serving.py` has a `--profile` flag that POSTs to the
server's `/start_profile`, but `scripts/run_bench.sh` does not pass it and the harness is sealed, so
a profile has to be taken under an off-contract driver such as `artifacts/probe_load.py` — which is
precisely why this check alone is **necessary but not sufficient** here.

### 2. Engagement under the contract workload (the check that actually decided it)

The scheduler prints per-prefill-batch `input throughput` unconditionally, n≈92 per benchmark run,
and its per-arm median reproduces across restarts to 0.22% on the base arm and 1.16% on the
candidate. `artifacts/ab_campaign.sh` copies the server log per arm
because `launch_server.sh` writes to a fixed path and truncates it on the next start.

```bash
cp /tmp/sglang_server_llama_3_1_8b_instruct.log /tmp/srv_arm.log   # BEFORE the next restart
python3 artifacts/prefill_stats.py /tmp/srv_arm.log cand r1
```

| | expected output (median over n≈92 full-size prefill batches) |
| --- | --- |
| **engaged** | `prefill ~78600–79500 tok/s  ~207 ms/batch` |
| **not engaged** | `prefill ~70500–70700 tok/s  ~232 ms/batch` |

Those are the ranges the per-arm medians occupied over six restarts each (base spread 0.22%, cand
spread 1.16% across restarts; the within-run scatter of the individual batch samples was measured at
1.8% under the probe driver, and the median reproduces to four significant figures across restarts). The
two bands do not overlap, and they classified **12 of 12** arms correctly across the campaign.
Anything between them means a partially-applied tree.

### 3. Batch census (the check that would have caught the failure)

Before trusting *any* reproduction of the workload, diff its batch census against the benchmark's:

```bash
grep -o '#new-seq: [0-9]*, #new-token: [0-9]*, #cached-token: [0-9]*' /tmp/srv_arm.log \
  | sort | uniq -c
```

| driver | dominant line | count |
| --- | --- | --: |
| `bench/` (the contract) | `#new-seq: 3, #new-token: 16384, #cached-token: 0` | 92 |
| a chunk-free probe | `#new-seq: 2, #new-token: 16384, #cached-token: 0` | 95 |

If your reproduction shows `#new-seq: 2`, it is not sending BOS and it is not the workload the win
was measured on. Note the trap that makes this hard to see: **`#cached-token` is 0 on every one of
those lines**, because with the radix cache disabled a chunk continuation reports zero cached tokens
while still carrying a non-zero `extend_prefix_lens`.

## Accuracy gate

gsm8k 5-shot, greedy, 1319 problems, **lm-eval 0.4.12** (`[api]` extra, in its own venv),
`--apply_chat_template`, `max_tokens=9216`, `temperature=0`, `top_p=1`, seeds `0,1234,1234,1234`,
via `scripts/run_eval.sh`. **The gate is ≥ 0.8213**, set at two standard errors below round 1's
baseline.

| config | strict-match | flexible-extract | source |
| --- | --- | --- | --- |
| round 1 baseline (defines the gate) | 0.8415 ± 0.0101 | 0.8431 ± 0.0100 | `eval_results/baseline_gsm8k_20260820_060747` |
| round 2 baseline, re-established | 0.8431 ± 0.010 | 0.8446 ± 0.010 | `eval_results/r2_baseline_20260820_152801` |
| **with the patch** | **0.8491 ± 0.0099** | **0.8499 ± 0.0098** | `eval_results/r2_ragged_gather_20260820_170637` |

**Pass, with room.** The +0.006 against the round-2 baseline is inside one standard error — no
measurable accuracy change in either direction, which is what a 6.4e-04 max relative error on the
attention output predicts. Two settings in `run_eval.sh` are worth copying rather than re-deriving:
`max_tokens=9216` (lm-eval's default 256 truncates the reasoning and scored **0.0318** here, which
reads as a broken model rather than a broken measurement), and the `sitecustomize` shim that falls
back to `reasoning_content` when `content` is empty.

## What was tried and did not work

Thirteen rows across two rounds — round 1's seven closed attempts, then round 2's rejected leads.
This is the longest section on purpose: it is what stops a third round from re-running any of it.
One row (11) is deprioritised rather than closed, and it is flagged as such.

### Round 1 — seven attempts, nothing kept

| attempt | kernel-level result | end-to-end | verdict |
| --- | --- | --- | --- |
| **1. Paged-attention decode `QKV_VERSION` GOLDEN vs EXPERIMENTAL** | GOLDEN **376.01 µs** vs EXPERIMENTAL **376.63 µs**, 6 interleaved graph-captured rounds at the exact trace shape | not deployed | Dead tie inside a 0.1% round spread. A prior session's "2.2% faster" (367.12 vs 375.19 µs) was **clock-ramp ordering noise** — measured one variant after the other on a part whose clock ramps 13–17%. |
| **2. `_AITER_PARTITION_SIZE_ROCM` sweep {128, 256, 512}** | ps512 looks like **247 µs**, a 34% win | not deployed | **Correctness trap.** `T_PAR_SIZE` is hard-coded to 256 inside `pa_ragged.cuh`; the argument reaches only the reduce kernel and the grid's y extent. At ps512 the grid computes 22 of 34 partitions and returns — 376 × 22/34 = 243 µs against the 247 measured, and relative error against fp32 is **1.16**, i.e. garbage. ps128 is correct and slower. Worth an upstream report. |
| **3. Attention's missing 15% of roofline** | QKV kernel **390.21 µs (98.1%)** vs reduce 7.41 µs (1.9%); KV-layout race: contig 389.73 / **production 389.24** / strided 397.05 / random 407.44 µs | not deployed | Closed. The kernel is at **85% of the 6.87 TB/s roofline** and the production page layout is *already* optimal (ISL 8192 under a 16384 chunk budget hands out each prefill KV as one contiguous run, ~94% contiguous). Trusting the naive strided model would have chased a 1.9% phantom. |
| **4. Retune the M=64 decode GEMMs** (`artifacts/rejected/0001-…`) | isolated graph race, LLC-overflowing: o_proj **+7.70%**, down_proj **+4.55%**, qkv **+2.15%**, gate_up 0% — 2.74 µs of 102.6 µs of GEMM per layer-step, ⇒ ~+0.45% predicted | **−0.394% ± 0.066%**, 6 paired rounds, **all six negative**, sd 0.055%; TPOT +0.52% | **Rejected — the main negative result.** Faster kernels, reproducibly slower server. |
| **5. Tune the M=16384 prefill GEMMs** (`artifacts/rejected/0002-…`) | isolated: o_proj **+3.64%**, qkv **+1.67%**, down_proj **+0.27%**, gate_up 0% — 25.7 µs of 4677 µs of prefill GEMM = 0.55% of prefill, **0.11% end to end** | TTFT −0.19%, ≈+0.03% throughput, measured only stacked on attempt 4 | Not shipped: a fifth of the rig's resolution. Also: prefill GEMMs already run at 1450–1585 TFLOPS and an exhaustive search of 2084 hipBLASLt solutions could not beat the `torch` heuristic on the two largest shapes. |
| **6. Does the inherited FlyDSL `o_proj` row (err_ratio 0.0198) cost accuracy?** | n/a — the hipBLASLt replacement is exact | gsm8k 0.8446 ± 0.0100 vs 0.8415 ± 0.0101 baseline | No. The numerically sloppier kernel is not measurably worse on gsm8k, and this is not a reason to accept a 0.394% regression. |
| **7. Race the same GEMMs in situ, inside a whole decoder layer** (`artifacts/rejected/0003-…`) | in-situ, bias-corrected: down_proj `flydsl:3522` **+0.54%** of a 508.3 µs layer step — while the isolated harness calls the same kernel **0.88% slower** and the tuner picks hipBLASLt | **−0.049% ± 0.088%**, 6 paired rounds, 2 up / 4 down, base and cand ranges **overlapping** | Rejected: a third of the 0.142% floor. Engagement was verified (0 vs 1 `not found tuned config` lines), so this is a null about a live change. |

Two structural facts from round 1 that decide whether a GEMM result is even *shippable* on this
model, and which cost real time to discover:

- **Llama-3.1-8B owns no aiter tuned-GEMM config file.** It inherits `(64,4096,4096)` from
  `qwen3_5_397b_bf16_tuned_gemm.csv` and `(64,6144,4096)` from `glm5_bf16_tuned_gemm.csv` purely by
  shape collision, and misses `(64,4096,14336)` entirely (falls to `torch`). Because
  `aiter/jit/core.py` raises on a duplicate shape key and auto-resolves by the tuner's eager `us`
  column, its rows **cannot** ship as a clean new `model_configs/` file — they have to be edited into
  *other models'* configs, which changes those models' behaviour on this GPU. That is a rejection
  reason independent of any measurement.
- **`solidx` is not stable across tuner runs** (the deployed qkv FlyDSL kernel is 2331 in the shipped
  CSV and 2415 in this session's tuner output). Harmless for FlyDSL, which dispatches by
  `kernelName`; not harmless for hipBLASLt, which dispatches by `solidx`.

### Round 2 — the rejected leads

| attempt | kernel-level result | end-to-end | verdict |
| --- | --- | --- | --- |
| **8. Ragged prefill, gated the obvious way** (bf16 twin of the fp8 branch, gated on `not any(extend_prefix_lens)`) | isolated **+34.71%** at 2×8192 with a self-control arm at +33.35% and max_rel_err 7.267e-04; profiler shows the ASM kernel present and **zero** ck_tile prefill kernels | **+0.035%** (2 paired rounds: −0.034%, +0.105%) and **+0.111%** on a repeat round — all inside the 0.14% floor | **Rejected.** Verifiably engaged under a probe, and engaged on **0 of 92** benchmark prefill batches, because one BOS token turns every batch into a 3-sequence batch with a prefix. The 51-line "obvious" version of the win is worth nothing. |
| **9. `flash_attn_varlen_func` reading the paged pool directly via `block_table`** | **fails**: `RuntimeError('Paged KV cache block size must be divisible by 128')` | n/a | Impossible at `--page-size 1`. This is *why* the gather exists; recorded in `artifacts/data/prefill_attn2.json` as a failed arm rather than a slow one. |
| **10. Lead 1 — Triton `unified_attention` / `pa_decode_gluon` decode path** (`SGLANG_USE_AITER_UNIFIED_ATTN=1`) | decode **3710.2 → 586.0 tok/s** median over n=68 batches = **6.33× slower**; probe wall 72.70 s → 353.53 s (4.86×) | not taken to a bench campaign — this is not a marginal call | **Rejected and closed.** Cause: the unified path builds its page table as `torch.zeros(bs, max_kv_len)`, so at page size 1 the Triton kernel does one block iteration per *token* where it was designed to do one per 16 or 64. Page size is sealed, so no version of this lead works. Prefill simultaneously measured **+0.18%**, which localises the collapse to the decode branch and rules out a sick server. |
| **11. Lead 3 — the small-kernel bucket** (rmsnorm / silu / rope / elementwise / store_kvcache / reduces) | **6.7% of decode, not the briefed 3.7%.** All launch-bound: rope 4.92 µs against 0.19 µs of bytes (**26×**), `add_rmsnorm_quant` 4.16 vs 0.23 (18×), silu 7.08 vs 0.80 (9×), `store_kvcache` 4.33 vs 0.04 (**100×**) — 1154 µs of a 17281 µs decode step | not measured end to end | **Deprioritised on expected value, not closed on evidence.** Driving the whole bucket to zero is worth ~4.8% of wall and needs real fusion (rope+store_kvcache, silu into a GEMM epilogue). **This is the strongest remaining lead for a round 3.** |
| **12. Lead 4 — re-read round 1's GEMM results against the corrected 0.14% floor** | §4 −0.394% ± 0.066% (a tighter floor makes it *more* firmly a regression); §5's +0.11% prediction is still under 0.14%; §7 −0.049% ± 0.088% overlaps zero at either floor | unchanged | Nothing to revisit. An independent roofline pass agrees: paged attention 5.78 TB/s (84% of 6.87), gate_up 4.81 (70%), down 4.88 (71%), qkv 3.10 (45%), o_proj 2.44 (36%) — and the two apparently-slack shapes are 50 MB and 34 MB of weights at 13–16 µs, i.e. **latency-bound, not bandwidth-bound**. |
| **13. The brief's literal lead 2** — a gfx95 launch-config predicate in `python/sglang/kernels/ops/attention/extend_attention.py:65-78` giving head_dim 128 a fallback `(BLOCK_M, BLOCK_N) = (64, 64)` instead of the tuned `(128, 64)` | the predicate is real and Llama would take the fallback | n/a | **That file is not on Llama's prefill path under the aiter backend at all.** The lead was right about the region and wrong about the mechanism. See below — this is the most interesting negative in the list. |

## No kernel-level proxy predicted the end-to-end sign

Round 1 tested this in both directions and round 2 confirmed it from a third angle. On this stack,
**deploy-and-A/B is not diligence on top of kernel benchmarking; it is the only measurement that
decides.**

- An isolated race — graph-captured, LLC-overflowing via `--footprint-mb`, interleaved,
  min-of-round-medians, at the correct production working set — said +2.15…+7.70% on three decode
  GEMMs. The server measured **−0.394%**: the **sign** was wrong, at 0.055% sd over six paired rounds.
- A strictly better proxy (`artifacts/bench_decode_layer.py`, the same GEMMs inside a whole
  graph-captured decoder layer with 2.12 GiB of attention traffic flushing the 256 MB LLC around
  them) **reproduced that regression from kernels alone** — −0.53% of the layer step against the
  measured +0.52% TPOT regression, and it decomposed by shape. Then the same harness manufactured a
  +0.54% that deployed to **−0.049% ± 0.088%**. A proxy that catches real regressions does not
  thereby license the gains it reports.
- The same shape, M=64 N=4096 K=14336, measured three ways gives **three different winners**:

| method | verdict |
| --- | --- |
| aiter tuner, eager | `hipblaslt:440151` 28.96 µs beats `flydsl:3522` 30.05 µs → **hipBLASLt** |
| isolated graph, 1120 MB working set | `torch` 28.991 µs beats `flydsl:3522` 29.245 µs → **torch** |
| in situ, whole decoder layer | `flydsl:3522` +0.54% of the layer step → **FlyDSL** |

- And in round 2, the *winning* line of enquiry produced two consecutive kernel wins worth nothing
  before the third one landed: **+34.71% isolated → +0.035% end to end**, then **+0.111%**, then
  **+37.22% isolated → +3.185% end to end**. The isolated number barely moved between the losing and
  winning versions; the end-to-end result moved by two orders of magnitude.

Two harness lessons that generalise, both cheap:

- **Race a new harness against itself before believing it.** Six byte-identical copies of the live
  config, registered as separate arms, come out at **+0.19% ± 0.03%, always the same sign**:
  `gbench.race()` pre-captures arms in insertion order and systematically penalises the
  first-registered one (`artifacts/data/insitu_selfcontrol.json`). That bias is larger than three of
  the four "wins" the in-situ search reported. This is `../../tuning-core/measurement.md` Rule 6b applied as a
  commissioning step to a harness you *wrote*, and it cost one run.
- **Cross-trace kernel comparison is unsound on this box.** The *unchanged* prefill GEMM
  `Cijk MT256x256x64` averages 1414 µs in a pure-prefill profiler window and 1058 µs in a mixed one —
  a 25% swing with no code change, from clock/power state. Every speedup in this entry is either an
  interleaved isolated race or a restart-paired A/B, never a difference between two traces.

## Cross-model transfer: where the winning lead came from

Round 2's brief pointed at prefill because **a sibling run on the same cluster had found +14.30%
there on a different model.** The gpt-oss-120b campaign found two defects in SGLang's
`extend_attention.py`: a loop bound that ignored the sliding window, and **a gfx950 launch
configuration whose tuned constants sat behind the wrong architecture predicate**. The first cannot
apply to Llama, which has no sliding window. The second was the lead: gpt-oss is head_dim 64, Llama
is head_dim 128, so *check what that branch selects for head_dim 128 on gfx950*.

What happened next is the part worth recording:

- **The lead was literally wrong.** The predicate is real and would give Llama the fallback
  `(64, 64)` tiles — but `extend_attention.py` is **not on Llama's prefill path under the aiter
  backend**, so it cannot cost anything.
- **The lead was directionally right, and that was enough.** It sent someone to profile prefill for
  the first time in two rounds, and a *different* instance of the same class of defect —
  **a gfx950-specific fast path sitting behind a predicate that this model does not satisfy** — was
  waiting one dispatcher up, in the aiter backend's fp8/head_dim-256 gate.

The transferable rule: **carry the defect class across models, not the file and line.** "A
hand-written gfx950 kernel exists for this shape and the dispatcher's predicate excludes us" is
portable; `extend_attention.py:65-78` is not. And the corollary from round 1's conclusion, which is
what makes this entry exist at all: **when two rounds have found nothing, the next thing to check is
not another candidate in the region you have mapped — it is whether there is a region you have not
mapped.**

## Artifacts

| file | what it is |
| --- | --- |
| `artifacts/0004-prefill-attn-ragged-gather.patch` | **The win.** +83 lines against `aiter_backend.py`; header carries the base, apply commands and every measurement |
| `artifacts/base/aiter_backend.py.orig` / `.ragged` | The pristine base (`45b09e64f84d…`) and the exact candidate that was measured (`19dcc694390e…`). Required by the deploy script, and the only way to verify the patch reproduces what ran |
| `artifacts/deploy_ragged_prefill.py` | `apply` / `revert` / `status` by whole-file swap, `filecmp`-verified, drops the stale `.pyc`. Idempotent — `ab_campaign.sh` flips arm order, so `apply` can be called twice in a row |
| `artifacts/make_ragged_patch.py` | Regenerates `.ragged` from `.orig` by scripted insertion, so the patch is never hand-edited |
| `artifacts/bench_prefill_attn2.py` | The isolated race **at the shape the sealed benchmark produces** (paged / gather+ASM / block_table / position-control arms), with numerics against the paged arm |
| `artifacts/bench_prefill_attn.py` | The earlier race at the shape a chunk-free probe produces — kept because the +34.71% it reports is the trap |
| `artifacts/gbench.py` | Graph-captured interleaved timing (many reps per graph, one sync per replay, min-of-round-medians). Required on gfx950, where the clock ramps 13–17% and pinning silently fails in this image. **Has the +0.19% arm-position bias** |
| `artifacts/prefill_stats.py` / `decode_stats.py` | Two ~50-line phase-local instruments reading the scheduler's own `input throughput` / `gen throughput` lines, at n≈92 and n≈68 samples per run. Between them they cover 100% of the wall clock, and the run rated them above any harness it built |
| `artifacts/ab_campaign.sh` | The end-to-end rig: full restart per arm, one bench run per start (always run #1), arm order flipped every round, per-arm server log kept |
| `artifacts/spread_within.sh` | The other half of the noise floor: N consecutive runs against one server |
| `artifacts/probe_load.py`, `prefill_probe_ab.sh`, `decode_probe_ab.sh` | Off-contract phase-local A/B drivers. **`probe_load.py` sends raw token ids and gets no BOS**, which is exactly why it disagreed with the benchmark — keep it, and keep the batch census check with it |
| `artifacts/bench_decode_layer.py`, `race_gemm_candidates.py` | Round 1's in-situ and isolated GEMM harnesses — the evidence behind "no kernel proxy predicts the sign" |
| `artifacts/rejected/0001…0003` | Round 1's three rejected patches, kept for their measurements. `0001` also **edits two other models' config CSVs**; do not apply it |
| `artifacts/data/*.json` | Raw race and probe output: `prefill_attn2.json` (the +37.22% and the block_table failure), `prefill_attn.json` (+34.71%), `insitu_selfcontrol.json` (the position bias), `decode_layer_insitu.json`, `decode_triton_probe_result.json` (the 6.33×) |

Scripts assume the bundle's layout (`analysis/…`, `scripts/…`) and the container's absolute paths
(`/sgl-workspace/sglang`, `/sgl-workspace/aiter`); adjust the two constants at the top of each rather
than trusting them to run from `artifacts/` unchanged.

## When this entry stops applying

Silently, in every case — the branch is skipped and the server serves correctly at the old speed:

- **arch ≠ gfx950** (`is_gfx95_supported()` is the first clause), or the aiter build lacks
  `hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_causal_group.co`.
- **head_dim ≠ 128**, either query or value. The ASM kernel exists only for this shape.
- **fp8 KV cache, or a vectorized-5d KV layout, or non-bf16 q.** Each is an explicit clause.
- **A different attention backend**, or a Triton/ck_tile prefill path — the patched file is not
  imported.
- **`--page-size` a multiple of 128.** Then `flash_attn_varlen_func` accepts a `block_table` and the
  gather is probably the wrong design; re-measure the block_table arm, which fails outright here.
- **A different SGLang tree.** The patch anchors on a specific spot in `forward_extend`; on another
  version `patch` will reject rather than fail silently, which is the one failure mode in this list
  that is loud.
- **A workload with a materially different prefill share**, or one that does not chunk. The gain is
  a phase gain; ISL, concurrency and `--chunked-prefill-size` together determine how much of it
  reaches the end-to-end number.

Still reusable when the artifact is inert: the **defect class** (a gfx950 hand-written kernel behind
a dispatcher predicate your model does not satisfy), the **inverted noise floor** and the
one-run-per-fresh-server protocol, `prefill_stats.py`/`decode_stats.py` as a pattern for finding a
phase-local instrument the framework already prints, the **batch census** check, and the whole
"what was tried" table — which is a map of where the time is on any dense bf16 model on this box.

## What would promote this entry to a verified win

1. **Reproduce from `artifacts/0004-…patch` alone on a clean instance**, deriving that instance's own
   restart floor rather than inheriting 0.09–0.14%, and reporting the arm ranges.
2. **Explain or bound r5** (+1.524% against +3.2…+4.3% elsewhere), and the winning campaign's 1.37%
   base-arm spread, which is roughly 10× the inherited floor.
3. **Record the profiler invocation** so the kernel-identity check is runnable as written rather than
   described.

## Gaps — things the template asks for that the bundle does not record

Listed rather than guessed, because a wrong number here is worse than a missing one.

- **Container image digest.** The tag is recorded; no `sha256:` anywhere in the bundle.
- **SGLang commit sha.** None exists — installed as a plain package with no git metadata. The base is
  a file hash, not a commit.
- **The server process's actual environment.** `/proc/<pid>/environ` was never dumped, so what the
  image exports is unknown. `../README.md` flags this as a recurring trap.
- **The profiler command** behind the kernel-identity evidence (see Engagement check).
- **Whether the 23.3% prefill-attention share was measured under `bench/` or under the probe
  driver.** `FINDINGS.md` says "a live server under a full-shape load" without disambiguating, and
  the two drivers produce different batch shapes — which is this run's central lesson, so the
  ambiguity matters.
- **A per-arm accuracy run for the rejected round-2 candidates.** Only the shipped patch and round
  1's two campaign arms were gated.
- **A transcription slip to be aware of when reading the source.** `FINDINGS.md` § 3's first
  end-to-end table transposes the base and cand columns for its round 2 row: the raw results are
  `r2rag_base_r2` 2635.748 and `r2rag_cand_r2` 2638.504, so the recorded Δ of +0.105% is correct and
  the two columns are swapped. There is also an unpaired extra candidate run in that campaign
  (`r2rag2_cand_r2`, 2635.980) that the table does not list. Neither changes any conclusion — the
  attempt is within noise either way.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/llama_31_8b_tuning/`.

- `EXPERIMENT_COMPLETE` — the one-line outcome.
- `FINDINGS.md` — round 2. § "Where the time actually goes" for the phase split, § 1 the small-kernel
  bucket, § 2 the GEMM re-examination, **§ 3 the win and the BOS-token story**, § 4 the Triton decode
  rejection, § "Measurement" for the protocol.
- `round1/FINDINGS.md` — the seven attempts, the noise-floor derivation (§ "Noise floor — the two
  spreads, measured separately"), and the "no proxy predicts the sign" argument in § 4 and § 7.
- `BASELINE.md` — the 2588.104 reference, the empty configuration search, and the 2813.63 rig warning.
- `patches/README.md` and each patch header — per-patch measurements and deploy contracts.
- `scripts/launch_server.sh`, `scripts/run_bench.sh`, `scripts/run_eval.sh` — the frozen launch,
  workload and gate, reproduced above.
- `results/` — 55 result directories; `r2rgv2_*` are the twelve arms of the winning campaign,
  `spread_within_*` the within-instance floor, `abd_*` and `ab_*` round 1's campaigns.
- `eval_results/` — five gsm8k runs with stderr.
- `analysis/` — the harnesses and raw JSON, subset copied to `artifacts/`.
