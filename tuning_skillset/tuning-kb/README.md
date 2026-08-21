# tuning-kb — known-good tuning results, per model

The rest of this skillset teaches you how to *find* a win. This directory records wins that were
already found and measured, so that a run landing on an environment we have already tuned can
**deploy the known answer in minutes and spend its time somewhere new** instead of rediscovering the
same rows.

One subdirectory per model. Each entry is self-contained: the environment it was measured on, the
launch configuration it assumes, the artifact that carries the win, and the numbers to expect.

## Evidence tiers

Read the tier before the percentage. It is the difference between "deploy this" and "this is a
starting hypothesis".

| tier | what it means | entries |
| --- | --- | --- |
| **Verified win** | The exported artifact was re-applied to a tree it had not been measured on and re-measured, reaching the number again | 3 |
| **Measured win** | Interleaved arms across restarts against a stated floor, with disjoint distributions — but nobody has yet deployed the artifact from scratch and reproduced it | 8 |
| **No win** | Nothing shipped. The entry exists for its negatives and its cost model | 1 |

The old rule in this file said only reproduced wins go in. That bar turned out to exclude most of a
twelve-model campaign, so the rule is now: **everything goes in, and the header says which tier it
is.** A measured win with an honest floor and an artifact is worth having. What is not acceptable is
a measured win *presented as* a verified one, and the tier label exists to make that impossible.

## The entries

| model | hardware | recipe | measured win | tier |
| --- | --- | --- | --- | --- |
| [Qwen3-14B-FP8](qwen3-14b-fp8/) | MI355X ×1, TP=1 | SGLang + aiter, FP8 — tuned CK block-scale/B-preshuffle GEMM table, `pa_ragged` decode by default, quant-kernel scale layout | **+29.05%** (1538.009 → 1985.566 tok/s), gsm8k 0.9454 → 0.9454 *identical on all 1319* | measured |
| [Gemma-4-26B-A4B-it](gemma-4-26b-a4b-it/) | MI355X ×2, TP=2 | SGLang 0.5.17, Triton attention/MoE, bf16 — six patches | **+24.88%** (2916.54 → 3642.20 tok/s), gsm8k 0.9386 ± 0.0066 (−0.35σ, pass) | **verified** |
| [Kimi-K3](kimi-k3/) | MI355X ×8, TP=8 | SGLang 0.5.15.post1 (K3 build) — five stacked code-lane patches: MLA split-K decode, attn-residual Triton pair, KDA warps, fused-front GEMM + tuned CSV | **+18.19%** (803.786 → 949.964 tok/s), gsm8k 0.9765 → 0.9788 | measured |
| [gpt-oss-120b](gpt-oss-120b/) | MI355X ×2, TP=2 | SGLang 0.5.17 — three Triton attention patches: SWA loop bound, extend-attention launch config, gfx950 decode segments | **+15.66%** (3938.109 → 4554.813 tok/s), gsm8k 0.962851 → 0.962851 *identical* | measured |
| [MiniMax-M3-MXFP4](minimax-m3-mxfp4/) | MI355X ×8, TP=8 | vLLM 0.26.0 + aiter, MXFP4 experts + FP8 KV — split-K in the Triton paged-decode kernel | **+7.71%** (4101.59 → 4417.64 tok/s), gsm8k 0.9447 → 0.9500 | measured |
| [DeepSeek-V4-Pro](deepseek-v4-pro/) | MI355X ×8, TP=8 | vLLM 0.26.0 + aiter, `deepseek_v4_fp8` + FP8 KV + MTP depth 1 — 56-row block-scaled GEMM table | **+5.8%** (987.67 → 1046.01 tok/s), gsm8k 0.9575 ± 0.0056 | **verified** |
| [GLM-5.2-MXFP4](glm-5.2-mxfp4/) | MI355X ×8, TP=8 | SGLang — five source patches in the DSA prefill and MLA absorb paths, over two rounds | **+4.23%** (1468.358 → 1530.461 tok/s), gsm8k 0.970432 → 0.968916 (pass, 2 of 1319) | measured |
| [Qwen3-8B](qwen3-8b/) | MI355X ×1, TP=1 | SGLang 0.5.17 + aiter, FP8 weights + FP8 KV — four rows of tuned-GEMM CSV | **+3.95%** (3642.1 → 3786.1 tok/s), gsm8k 0.9348 above a 0.9280 gate | **verified** |
| [Qwen3.5-397B-A17B-MXFP4](qwen3.5-397b-a17b-mxfp4/) | MI355X ×4, TP=4 | SGLang 0.5.17 + aiter — TP=4 GEMM table, topk, GDN, fp4 fused-MoE rows | **+3.93%** (2505.96 → 2604.42 tok/s), gsm8k 0.9727 ± 0.0045 vs a 0.9691 gate — **0.80σ, the thinnest margin here** | measured |
| [Llama-3.1-8B-Instruct](llama-3.1-8b-instruct/) | MI355X ×1, TP=1 | SGLang + aiter, bf16 — KV gather making aiter's gfx950 v3 ASM prefill kernel reachable | **+3.185% ± 0.916%** (2640.618 → 2724.708 tok/s), TTFT −10.4%, gsm8k 0.8431 → 0.8491 | measured |
| [Mixtral-8x7B-Instruct](mixtral-8x7b-instruct/) | MI355X ×8, TP=8 | SGLang + aiter, bf16 weights + FP8 KV — two rows of tuned bf16 GEMM CSV | **+0.474%** (6918.43 → 6951.22 tok/s), gsm8k 0.6482 ± 0.0132 vs 0.6543 base | measured |
| [Qwen3.8-2.4T-A95B-MXFP4](qwen3.8-24t-a95b-mxfp4/) | MI355X ×8, TP=8/EP=8 | SGLang + aiter, MXFP4 experts + FP8 KV | **no win.** Best arm +0.17% against a 1.03% floor. Read it for the negatives and the cost model | **no win** |

All twelve were measured on the same frozen workload — ISL 8192, OSL 1024, concurrency 64, 192
prompts, InferenceX `benchmark_serving` — so the *gains* are comparable across entries even though
the stacks are not. DeepSeek is the one exception on warmups: 128, not 8.

## How to use an entry

**Read the fingerprint before the result.** Every entry opens with an environment fingerprint, split
into fields that are load-bearing for the win and fields that merely describe the run. A tuned config
is a **lookup keyed on a tuple**; if a load-bearing field differs, the lookup misses and the artifact
does nothing. It does not warn you — it falls back and serves at the old speed
(`../tuning-core/engagement_verification.md`).

1. **Fingerprint the environment you are on**, then diff it against the entry's table.
2. **On a full match**: deploy the artifact, restart, run the entry's engagement check, and
   **re-derive the noise floor on your own node** before believing the number. Budget one hour, not
   one day. Then go look for something the entry does not already cover.
3. **On a partial match**: treat the entry as a *starting hypothesis*, not an answer. The shapes and
   the winning kernel families usually still point the right way even when the exact rows miss —
   re-tune those shapes rather than re-deriving the target list from scratch.
4. **On a mismatch in arch or CU count**: do not deploy. Those are literally part of the config key,
   so the rows are unreachable. Use the entry only for its shape list and its record of what turned
   out not to matter.

**Reproduce, do not assume.** An entry is evidence that a win existed on a specific stack, not a
guarantee about yours. Every entry states the noise floor it was measured against so you can tell a
real reproduction from a coincidence.

## What twelve campaigns agree on

Read this before starting a thirteenth. These findings repeated across entries, and are stated once
here rather than twelve times below.

### The noise floor is a property of the machine, not the model — and this is the campaign's most expensive lesson

DeepSeek-V4-Pro proved it on itself. Round 1 measured a **5.5–6.4%** restart-to-restart spread and
concluded that this model was inherently noisy, that effects under 3% were unmeasurable on it, and
that its win was barely claimable. Round 2 ran the **same patch, same stack, same workload** on a
quieter host and measured **0.148%** — a floor forty times tighter, from changing nothing but the
machine. The same effect went from "wider than the noise" to roughly 40× the floor.

The corollaries cost real time elsewhere in the campaign:

- **Llama-3.1-8B inherited a floor and lost a round to it.** Round 1 was handed 1.49%, discarded
  candidates on size grounds, and shipped nothing. That figure was mostly the within-instance settle
  curve misread as a restart floor; the true restart spread was **0.09–0.14%**, and round 2 found
  +3.185% on its first real candidate.
- **Do not assume which spread is larger.** On Llama, restart-to-restart was about **8.5× tighter**
  than within-process, because every restart samples the same point on a falling settle curve. That
  inverts the worked examples in `../tuning-core/measurement.md` Rule 3b. Measure both and find out.
- **Re-derive it per boot, not just per node.** GLM-5.2 round 2 re-derived **0.718%** run-level where
  round 1 measured 0.39% across restarts, on the same model.

**Never inherit a floor from a previous run, a sibling bundle, or this file. Measure your own across
at least three restarts before attributing anything.** It is the highest-leverage hour in a tuning
run and the easiest to skip.

### Rank by gain ÷ floor, not by gain

The floors span an order of magnitude, so the percentages are not comparable on their own. A +0.474%
result on a 0.070% floor is better evidence than a +7.4% result on a 6.4% one.

| model | gain | its floor | ratio |
| --- | --- | --- | --- |
| Qwen3-14B-FP8 | +29.05% | 0.21% | **138×** |
| Gemma-4-26B | +24.88% | 0.25% | ~100× |
| DeepSeek-V4-Pro | +5.8% | 0.148% | ~40× |
| gpt-oss-120b | +15.66% | 0.501% | ~31× |
| Kimi-K3 | +18.19% | 0.713% | ~26× |
| Llama-3.1-8B | +3.185% | 0.09–0.14% | ~23× |
| Qwen3-8B | +3.95% | 0.36% | 11× |
| GLM-5.2 | +4.23% | 0.39% | 10.8× |
| MiniMax-M3 | +7.71% | 0.87% | ~9× |
| Mixtral-8x7B | +0.474% | 0.070% | 6.8× |
| Qwen3.5-397B | +3.93% | 0.65% | ~6× |
| Qwen3.8-2.4T | +0.17% | 1.03% | **inside the floor** |

Ratios are derived from the two columns beside them; each entry states its own arithmetic and its own
disjointness evidence, which is the claim that actually matters. **And quote the ratio on your
smallest claimed increment, not on the total** — Kimi's entry does this deliberately, reporting that
its last arm is 1.95× the floor rather than advertising 26× on the stack.

### Tuned constants stranded behind an architecture predicate is the most repeatable bug in the stack

Four models hit this independently, and it produced the two largest single-patch wins in the
campaign. The pattern: upstream ships a tuned launch configuration or kernel table, then guards it
with a predicate that is false on this hardware — an `SM100+` test, a CUDA-only branch, an arch key,
a TP size. The guard silently selects an untuned fallback, which is then **the only path that ever
runs**, so the tuned constants are dead code and nothing in any log says so.

| model | the predicate | what it was worth |
| --- | --- | --- |
| Kimi-K3 | attn_residual launch config reachable only behind an SM100+ test; on CDNA the untuned fallback always runs. One wave64 per CTA beat the shipped shape by **1.83×** on the pair | +5.31% |
| gpt-oss-120b | gfx950 attention path selected behind the wrong predicate at head_dim 64, plus the extend-attention launch config | +14.30% with the SWA loop bound |
| Gemma-4-26B | the *same* one-case predicate in the *same* function as gpt-oss, and the *same* SWA loop bound in the same file, found independently | +7.20% and +1.18% |
| Qwen3.5-397B | a TP=2 tuned table simply unreachable at TP=4 — 956 logged misses; plus aiter's FlyDSL GDN rows walled off because SGLang permits only Triton on ROCm | +1.60% |

**Go looking for this first on any new model.** Grep the kernel you care about for arch tests,
capability tests and TP-keyed table lookups, and check which branch actually executes. The tell is a
tuned-config lookup that misses every call — usually visible as a repeated miss line in the server
log, which is also the cheapest engagement check you will ever write.

### Cross-model leads transfer, even when the exact line does not

gpt-oss found the head_dim 64 predicate defect. That lead was handed to Llama-3.1-8B, which had
shipped nothing in round 1; `extend_attention.py` turned out not to even be on Llama's prefill path
under the aiter backend, but the *phase* was right, and looking at prefill found the same class of
defect at head_dim 128 for +3.185%. **A transferred lead tells you which phase to open, not which
line to edit** — and that is still most of the value, because choosing the phase is the hard part.

### A real kernel win is not a result

Every campaign produced at least one, and most were dropped. The gap is always Amdahl's share of the
pass, not a measurement failure.

| the kernel win | what it moved end to end |
| --- | --- |
| MiniMax prefill tile, **1.58×**, engagement confirmed on all 8 ranks | **+0.062%** — 14× *below* the floor |
| Qwen3-8B prefill M=8192, **−24.2%** on the kernel | +0.083% |
| Qwen3.8-2.4T dense bf16 GEMMs, **1.044–1.301×**, installed and gsm8k-gated | **+0.08%**, inside the floor |
| Qwen3.8-2.4T prefill MoE stage 2, **1.0182×**, four reproductions | **+0.05%** — and it was predicted at 0.05% beforehand, which is the right way to spend an instance: not at all |
| Mixtral fused MoE, 8.3% at the op level | nothing in decode |
| DeepSeek block-scaled GEMMs, **3.29×** aggregate | ~7% — the entry's standing instruction is *never quote 3.29× as the result* |
| Llama prefill attention, **+37.22%** isolated → +11.90% on the phase in situ | **+3.185%** |
| MiniMax split-K decode, **5.98×** | **+7.71%** |

**Measure the share before you tune the kernel.** Qwen3.8-2.4T's entry gives the arithmetic as
`f·(1 − 1/x)`: a surface at 10% of wall time needs 1.11× to move 1%, at 5% it needs 1.26×, at 2% it
needs 2.06×. Compute that number first and most leads die for free.

### Microbenchmarks overstate, and sometimes get the sign wrong

- Mixtral's isolated harness reported 5.75 µs for a kernel that measured 7.85 µs inside the server.
  Round 2 then **withdrew round 1's explanation** of that gap: it is not a 21× overstatement of the
  improvement but a consistent ~2.1 µs additive dispatch offset between two harnesses. The raw datum
  survives; the story did not.
- Mixtral's isolated A/B ranked the gate at −17.8% and `qkv_proj` at −4.5%; the deployed graph said
  −3.9% and −13.1%. **The ranking inverted.**
- Qwen3.5-397B's tuner liked four rows at 1.05–1.23×; re-timed interleaved they came back
  **0.976/1.017/0.997/0.965×** — three of four were losses.
- Llama's isolated LLC-modelled graph race got the **end-to-end sign backwards** in round 1. In round
  2, the first two ragged-prefill attempts measured +0.035% and +0.110% before the third reached
  +3.185%. No kernel-level proxy predicted which.

Re-time winners in situ, on the cold shipping harness, interleaved.

### A correctness proof is not a throughput win, and both are worth exporting

GLM-5.2's first-slot guard was verified against **4.06 billion live index blocks with zero
violations** — a real result that closed an invariant question round 1 could only answer by reading
four topk producers. It is worth **+0.264%** against a 0.718% floor: inside the noise. It is exported
and explicitly **not claimed**. Keep the distinction visible in both directions; an entry that
quietly folds a sub-floor patch into a headline is the thing that makes the rest of it unusable.

### The launch script setting no environment variables does not mean the process has none

Mixtral's container image exports twelve, including `SGLANG_USE_AITER=1`, and a whole attempt was
built on assuming it was unset. Gemma's image exports the `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` its
all-reduce actually runs. Qwen3-14B's patch 2 lets `QKV_VERSION` override its new default, so an
image exporting `QKV_VERSION=GOLDEN` makes the patch inert **silently**.
**`cat /proc/<pid>/environ | tr '\0' '\n'` is the ground truth** — and note that most entries in this
directory record it as *not dumped*, which is the most common gap in the whole collection.

### Anything that touches graph-captured decode needs a restart

A live drop-in is a silent no-op that benchmarks perfectly. True in SGLang and vLLM alike, for config
tables and source changes both. It is the most common way to produce a clean, plausible, wrong
number. Pair it with the cache hazards, each of which will independently serve you the old kernel:
aiter's `/tmp/aiter_configs` merge is derived and is **not** regenerated if it already exists;
Triton's `~/.triton/cache`; a stale `__pycache__`. In-process `lru_cache` on config lookups means
even a correct file on disk cannot reach a running server.

### Verify engagement two-sided and per rank

Positive marker present the expected number of times *and* negative marker exactly zero, with the
control arm asserting the mirror image, counted across all ranks. Gemma's harness gated on both TP
ranks; Mixtral counted `named=80 default=0` across eight; MiniMax counted 24 markers as 8 ranks × 3
shapes and had to check they landed *inside* the graph capture window; DeepSeek's check is that 1928
miss lines go to 0 while a deliberately untouched shape's 8 misses **survive**. One rank engaging is
a real failure mode, and at TP>1 it can read as no gain at all because the fast ranks wait on the
slow one.

### Discard the first benchmark after every restart, and never use a fixed run window

MiniMax's cold runs land at ~57% of warm. DeepSeek's instances split into fast-born and slow-born
classes and one was still climbing at its ninth benchmark — a fixed b3–b5 window would have
understated it by 2.4%. Run until the last three are flat within the within-instance scatter. Also
**interleave arms across restarts**: every entry that did this can defend its result, and the one
that ran arms in blocks (MiniMax) carries a caveat in its header for exactly that reason.

### Price an instance before choosing a lead

On big models the cost model *is* the strategy. Qwen3.8-2.4T: ~20 minutes per server start, ~3.1
minutes per benchmark round, about 1.2 instances per hour achieved, so a 3-vs-3 interleaved design is
two hours of pure measurement and resolves roughly 1%. DeepSeek's reference server took 1927 s to
accept traffic. **Work out the smallest effect you can resolve within the run before you pick what to
tune**, or you will spend the allocation proving nothing. And profile with care on these models:
Qwen3.8's profiled rounds ran 2.8% and 8.3% below baseline and are not poolable with unprofiled ones,
and `/stop_profile` hung every rank until a 1200 s watchdog killed the server.

### Watch for hazards that have nothing to do with your patch

DeepSeek lost an entire measurement instance when a co-tenant filled the NFS volume backing `/home`
to 100%. The signature is worth memorising because it is designed to fool a mean: one benchmark
printed a plausible 992.48 tok/s whose JSON was never written, the next recorded **0.00**, and the
rest produced nothing. The instance was quarantined with a `WHY.md` rather than silently dropped.
Check free space, foreign GPU tenants and clocks before attributing a surprise.

### Record the negatives with their numbers, and be willing to withdraw

Mixtral withdrew three claims in round 1 and two more in round 2. DeepSeek's update withdrew seven,
including its own headline and its "effects under 3% are unmeasurable on this model" rule. In every
case the claim survived as long as it did because it had a story attached. The best sentence in this
knowledge base came out of that: **when a result needs a story to make sense, measure it again.**

### Check the accuracy gate margin, not just the pass

Most entries pass comfortably, and two pass while scoring *below* their nominal reference — Gemma at
−0.35σ and GLM two problems light — soundly, because each measured its own harness resolution first.
Qwen3.5-397B is the exception worth flagging: it passes with **0.80σ**, about 4.76 answers of 1319,
after the score moved 6 answers against a ±4-answer reproducibility band established on that exact
stack. Its entry names the numerical risk (an fp4 fused-MoE row that keeps a stage-1 intermediate in
fp4) and tells you to re-run the gate before shipping. **Do that whenever the movement is larger than
the harness's resolution**, regardless of which side of the threshold you land on.

## Reproduction debt

Eight of the eleven wins here have never been deployed from their own artifact onto a tree they were
not measured on. That is the single largest gap in this directory, and it is concentrated in the
newest and largest entries. Several entries did verify **artifact integrity** — applying the exported
patch to the recorded pristine base and reproducing the measured tree byte-for-byte — which is worth
having and is not the same thing as reproducing the throughput.

If you have a spare instance and no better lead, promoting a measured win to a verified one is an
hour well spent, and every entry states the exact steps and the number to expect.

## What every entry must record

Entries exist to be *reused*, which means an entry that omits the boring parts is not usable.
`ENTRY_TEMPLATE.md` is the skeleton; the load-bearing sections are:

- **Environment fingerprint** — container digest, framework and library commits, GPU arch and CU
  count, TP, quantization. Marked load-bearing or descriptive, per field.
- **Launch configuration** — the exact server flags, because they determine the shapes. This is the
  part most often left out and most often the reason a reuse silently fails: on Qwen3-8B the tuned
  rows are keyed on M values that come directly from `--chunked-prefill-size` and the benchmark's
  concurrency, so changing either makes the entry inert.
- **Workload** — ISL, OSL, concurrency, prompt count, warmups. A win at one operating point is not a
  win at another.
- **Baseline and noise floor** — both spreads separately, which one applies, and whether the arms are
  disjoint. Without the floor the delta is unreadable.
- **The artifact** — checked in under `artifacts/`, deployable as-is, plus the exact commands to
  apply it and to invalidate every derived cache.
- **Engagement check** — the command that proves the win is live, with expected output in *both*
  directions.
- **Accuracy gate** — the score, the harness and its pinned version, the margin in σ.
- **What was tried and did not work** — the most valuable and most often discarded section. Record
  negative results with their measured numbers on both the kernel and end-to-end side.
- **What the bundle did not record** — an explicit gap list. "Not recorded" is a finding; a guessed
  value is a trap.

## Adding an entry

Copy `ENTRY_TEMPLATE.md` to `<model-name>/README.md`, fill it in, and put the deployable artifact in
`<model-name>/artifacts/`. Three rules:

- **Label the tier in the first line.** `Verified win` only if the artifact was re-applied to a tree
  it had not been measured on and re-measured. Otherwise `Measured win`, with a reproduction-status
  section saying exactly what was and was not done.
- **Record the measured spread, and never a single run.** A number without its noise floor cannot be
  checked by the next reader, and this whole directory is only worth anything if its claims are
  checkable.
- **Ship what is deployable, and separate it from what is only evidence.** Mixtral's `artifacts/`
  splits `deploy/` from `evidence_only_001/` for exactly this reason: a reader deploying a withdrawn
  patch is the failure an entry must prevent.

## If you are running a blind evaluation, delete this directory first

This directory is an answer key. That is the point of it in production, and it is disqualifying when
the same task bundles are being used to *measure* whether an agent can find a win on its own.

**Every entry carries deployable artifacts under `<model>/artifacts/`** — patches, tuned CSVs, tuner
and verification scripts, and in several cases the pristine diff bases. Hand an agent this skillset
alongside the matching bundle and it can deploy the result in minutes: a correct outcome for a
production run, and a meaningless one as an evaluation.

The entries are also mutually informative in a way that matters more than any single one of them. The
cross-cutting section above tells a reader which phase to open on a model they have never seen, names
the most repeatable defect class in the stack, and hands over the measurement protocol that took
twelve campaigns to work out. **Excluding only the entry for the model under test leaves a
substantial hint in place.**

```bash
rsync -a --exclude tuning-kb/ tuning_skillset/ <bundle>/tuning_skillset/
```

Entries for *other* models are harmless and worth keeping — they carry method and environment detail
without the answer. The rule is narrow: **exclude the entry for the model under test**, and prefer
excluding the whole directory when the evaluation is scored.
