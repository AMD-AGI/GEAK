# Mixtral-8x7B-Instruct-v0.1 on MI355X — SGLang, TP=8, bf16, aiter tuned bf16 GEMM rows

**Measured win: +0.474% output throughput** (6918.43 → 6951.22 tok/s), gsm8k strict-match
0.6482 ± 0.0132 against a same-node baseline of 0.6543 ± 0.0131 — no regression, 0.4 σ. The win is
**2 rows of CSV** — a data-only drop-in, no compilation, no source change. It is
`artifacts/deploy/`, and it is *not* the patch this entry shipped a round ago.

**Read this entry for the negatives, not the win.** Half a percent is not why this is here. It is
here because two independent rounds characterised the noise floor on this stack better than any
other bundle in this directory, and then between them withdrew five of their own claims after
re-measuring. The hours you save by reading "What was tried and did not work" and "Claims made and
then withdrawn" are worth an order of magnitude more than the 32.8 tok/s.

**Two rounds ran, on different hosts, and their numbers must not be mixed.** Round 1 (2026-08-19)
kept `001_aiter_tuned_fmoe_mixtral` and measured +0.269% on `crsuse2-m2m-110`. Round 2 (2026-08-20)
ran on a **third node whose name is not recorded in the bundle**, kept
`002_aiter_tuned_gemm_mixtral`, and did not carry round 1's patch forward. **The headline above is
round 2's, measured entirely on round 2's node against a baseline re-measured there from scratch.**

**Reproduction status: not reproduced on a clean instance, and this falls short of the house bar.**
Round 2's +0.474% rests on six independently booted server instances on one node, arms interleaved
boot by boot, plus one final confirmation boot from a tree carrying nothing but the drop-in
(`final_kept_r1`: 6944.86 tok/s, TTFT 2057.81 ms, TPOT 7.20 ms — in the candidate range and
carrying the candidate's TPOT signature). What is missing is the same thing that was missing
before, one step further out: a fresh container on a different machine, applying
`artifacts/deploy/` and nothing else, by someone who did not run the experiment. Round 1's patch
001 *did* clear that bar across two machines — see "Two rounds" — which is why the weaker claim now
carries the headline and the stronger one sits in `artifacts/evidence_only_001/`.

## Two rounds

| | round 1 | round 2 |
| --- | --- | --- |
| dates | 2026-08-19 | 2026-08-20 |
| hosts | `crsuse2-m2m-261`, then `crsuse2-m2m-110` | **not recorded** — stated only as "a third node, distinct from m2m-110 and m2m-261" |
| kept change | `001_aiter_tuned_fmoe_mixtral` — 12 tuned fused-MoE CK 2-stage rows | `002_aiter_tuned_gemm_mixtral` — 2 tuned bf16 GEMM rows |
| what it speeds up | **prefill** (the 16384 / 32768 buckets); decode contributes nothing | **decode** (`qkv_proj` and the router gate, inside the captured HIP graph); TTFT unchanged |
| baseline / patched | 6914.54 → 6933.13 (n110) | 6918.43 → 6951.22 |
| delta | +18.59 tok/s = **+0.269%** (n110); +30.08 = +0.436% (n261) | +32.79 tok/s = **+0.474%** |
| restart floor on that node | 5.62 tok/s = 0.081% (n110) | **4.45 tok/s = 0.064%** |
| gain ÷ floor | 3.3× (n110, against that node's own restart sd) | **6.8×** (against the pooled restart sd, 4.85) |
| status now | **superseded.** Retained as evidence, not deployed | **deployable** |

The two changes touch different kernels — MoE stage 1/2 versus `qkv_proj` and the router gate — so
they are not obviously in conflict. Round 2 measured them together anyway, interleaved boot by
boot: **002 alone 6950.41 (4 instances, restart sd 4.57); 001+002 6939.56 (2 instances, restart sd
9.50)**. Stacking adds nothing. The point estimate is −10.9 tok/s, inside the stack arm's own
restart spread at n=2, so the defensible statement is *no measurable benefit* — not that 001 is
harmful. Patch 001 genuinely engaged on that node (8 named `fused_moe` selections per rank, 0
defaults, at every bucket including decode token=64), so this was a real comparison and not
cand-versus-cand.

**Ship 002 alone.** Do not install both.

**001 alone was never measured on round 2's node.** Its +0.269% stands on round 1's two hosts and
was never refuted; it simply was not re-tested against pristine aiter on the third. The simplest
reading round 2 offers is that 001's effect on that node is smaller than its own restart spread.

## Environment fingerprint

Identical across both rounds: the image is pinned by digest in `scripts/start_container.sh` and
round 2 re-verified the aiter SHA on its own tree before starting.

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 8× MI355X, `gfx950`, **256 CU** each | **yes** — `gfx` and `cu_num` are literal columns in both config keys; the rows carry `gfx950,256` |
| container | `rocm/sgl-dev@sha256:95a933896aeab2a431521ece6ebe90c1db37a3aaf1e32a938d56ef7ccf6603a5` | descriptive |
| SGLang | 0.5.17 (`0.5.17.dev20260812+gdc5f6c4883`), commit `197832bcf536543092e621e03d61ae2602a392d0` | descriptive |
| aiter | commit **`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`** | **yes** — merge path, schema and kernel names are version-specific; round 2 confirmed the tree was at this SHA with 001 *not* applied |
| CK submodule | `f33252cebe5a52362ec1ee12c124dde7800dda3a` | descriptive — not touched by either patch, but it **is** a real submodule, so a root-level `git diff` would not capture a CK edit |
| torch / ROCm | 2.9.1+rocm7.2.0 / 7.2.0 | descriptive |
| model | Mixtral-8x7B-Instruct-v0.1, **TP=8**, `ep_size=1` | **yes** — for 001, TP=8 sets `inter_dim = 14336/8 = 1792`; for 002 it sets `qkv_proj` N = 6144/8 = 768 |
| weights | **bf16**, no `--quantization` | **yes** — both tables are keyed on `torch.bfloat16` |
| KV cache | `fp8_e4m3` | **yes** — and for 002 it is also the risk: `qkv_proj` is the path that fills it |
| MoE op | aiter fused-MoE 2-stage CK, `tuned_fmoe` table | relevant to 001 only |
| dense GEMM path | `aiter.tuned_gemm.tgemm`, `bf16_tuned_gemm` table | **yes** for 002 — this is which table is read |
| attention / MoE backend | aiter / aiter | descriptive (frozen flags) |

The reference labels this arm `moe_swa`. **That label is wrong** — `config.json` has
`sliding_window: null`. There is no sliding-window attention on this model.

## Launch configuration

Confirmed unchanged between the two rounds: `scripts/launch_server.sh` is byte-for-byte the script
both rounds ran, and round 2 verified `config verified` and `ranks reporting KV pool: 8` on its own
boots.

```bash
python3 -m sglang.launch_server \
  --model-path <Mixtral-8x7B-Instruct-v0.1> \
  --tp-size 8 \
  --context-length 11264 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend aiter \
  --attention-backend aiter \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --disable-radix-cache \
  --watchdog-timeout 1800 \
  --trust-remote-code
```

Resolved from the server log, not assumed: `mem_fraction_static` **0.68** (SGLang rescales the 0.8
at TP=8), `page_size=1`, `quantization=None`, `max_total_num_tokens=23,249,868`, KV pool 88.69 GB
for K and the same for V per rank in `float8_e4m3fn`, weights ~10.93 GB per rank, decode HIP graph
captured up to batch 512, prefill graph disabled. On its own node round 2 re-confirmed the parts
its boots print — `config verified`, the 0.8 → 0.68 rescale, and 23,249,868 tokens with K 88.69 GB
+ V 88.69 GB on 8 of 8 ranks — alongside a 375 s cold boot, VRAM ~71% per GPU on all eight, and no
foreign tenant.

### The launch script sets no environment variables. The process has twelve.

This distinction cost round 1 an entire attempt, so it is stated first. `launch_server.sh` exports
nothing and says so. But the container image's own `ENV` block puts all of this into the server's
environment:

```
SGLANG_USE_AITER=1                      ROCM_QUICK_REDUCE_QUANTIZATION=INT8
SGLANG_MOE_PADDING=1                    SGLANG_USE_ROCM700A=1
SGLANG_SET_CPU_AFFINITY=1               SGLANG_INT4_WEIGHT=0
SGLANG_DISABLE_CUDNN_CHECK=1            SGLANG_ROCM_FUSED_DECODE_MLA=1
SGLANG_ROCM_DISABLE_LINEARQUANT=0       SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
AITER_USE_SYSTEM_TRITON=1               BUILD_AITER_ALL=1
```

**Always read `/proc/<server-pid>/environ` before reasoning about which code paths are live.**
Round 1's attempt 2.6 was built entirely on the assumption that `SGLANG_USE_AITER` was unset
because the launch script did not set it. It was set, the branch was already live, and the
resulting patch was a literal no-op.

`tuning/relaunch.sh` exists because a bare relaunch is not reliable here: it waits for three
consecutive all-zero VRAM and KFD readings before starting (a single zero reading lies, there is a
~200 s reap delay), then asserts the KV pool is 23,249,868 tokens on **8 of 8** ranks. The trap it
guards is a short 8,144,599-token pool that still passes the launcher's own rank count and would
quietly change what you are measuring.

## Workload

Confirmed unchanged between the two rounds — `scripts/run_bench.sh` is the same sealed script, and
round 2 ran it with nothing overridden.

ISL 8192, OSL 1024, concurrency 64, 192 prompts, 8 warmups, seed 0, `random_range_ratio 1.0`,
`--ignore-eos`, InferenceX `benchmark_serving` fork. Reference headline 6923.6 tok/s, mean TTFT
2047.6 ms, mean TPOT 7.241 ms, 28.4 s wall.

Shapes, and which patch cares about which:

- decode token bucket **64** from the concurrency, and the padded decode batch **M=64**
  (`get_padded_m` rounds 40/48/56 up to 64). **Both of 002's rows are keyed on M=64** — this is the
  whole of 002's gain.
- prefill buckets **16384 and 32768** from `--chunked-prefill-size` and `--max-prefill-tokens`.
  **All of 001's end-to-end gain lives here** — see the withdrawn claims.
- MoE geometry `model_dim=4096`, `inter_dim=1792`, `expert=8`, `topk=2` (001's key).
- dense GEMM shapes `qkv_proj` M=64 N=768 K=4096 and router gate M=64 N=8 K=4096 (002's key).
  N=768 is the TP=8 shard of the fused QKV output width — (32 Q + 8 K + 8 V) heads × 128 = 6144,
  divided by 8.

The two patches therefore optimise **opposite phases** of the same benchmark. That is worth
holding onto: it is why "concurrency ≠ 64" is fatal to 002 and irrelevant to 001, and why
"chunked-prefill-size ≠ 32768" is the reverse.

## Baseline and noise floor

The most valuable section of the entry. Three machines: round 1 measured across **14 independently
booted instances** on two of them, and round 2's kept-change campaign is **six boots** on the third
(plus three more for the stacking test). **Do not average across hosts.** The patched *level* has been
remarkably stable and the *baseline* level is what moves between nodes, so a cross-host percentage
is a difference of two things measured under different conditions.

### Round 2's node — quote this one for the headline

| arm | instances | instance means (tok/s) | within-instance sd | arm mean |
| --- | --- | --- | --- | --- |
| base | baseA (preflight), baseB2, baseB3 | 6913.50, 6919.67, 6922.13 | 9.52, 2.56, 5.51 | **6918.43** |
| cand (002) | candB1, candB2, candB3 | 6953.23, 6955.15, 6945.29 | 4.74, 7.67, 6.31 | **6951.22** |

| noise floor | spread |
| --- | --- |
| within one server process, across every five-run block in the campaign | **2.56–9.52 tok/s = 0.037%–0.139%** |
| restart-to-restart, sd of the 3 base instance means | **4.45 tok/s = 0.064%** |
| restart-to-restart, sd of the 3 cand instance means | 5.23 tok/s = 0.075% |
| pooled restart sd across both arms | **4.85 tok/s** |

**Delta +32.79 tok/s = +0.474%, which is 6.8× the pooled restart sd (4.85).** On the
three-versus-three instance means, t = 8.28 with 4 df, **95% CI +21.8 .. +43.8 tok/s**.

**The arms are disjoint, and at run level, not just at instance level.** The worst candidate
instance (6945.29) beats the best base instance (6922.13) by 23 tok/s, and the worst of fifteen
candidate runs (6934.66) still beats the best of fourteen base runs (6927.27). There is no overlap
anywhere in the campaign. Dropping `baseA` — the one instance that came from preflight rather than
from the interleaved campaign — leaves base 6920.90 (n=2) against cand 6951.22 (n=3), **+30.32
tok/s = +0.438%**; the result does not rest on mixing protocols.

Protocol, which is why the above is worth anything: arms alternated **boot by boot**
(`cand, base, cand, base, cand`) so drift in machine state is shared rather than assigned to one
arm, and every boot discarded one warm-up run before five measured runs, so a first run against a
fresh server is never compared against a fifth.

### Round 1, `crsuse2-m2m-110` — quote this one for patch 001

| arm | instances | instance means (tok/s) | mean |
| --- | --- | --- | --- |
| baseline | baseP1, baseP3, baseP5 | 6912.05, 6920.97, 6910.59 | **6914.54** |
| patch 001 | moe001P2, moe001P4, moe001P6 | 6932.14, 6942.65, 6924.59 | **6933.13** |

| noise floor | spread |
| --- | --- |
| within one server process, pooled sd over 6 instances × 3 runs | **8.42 tok/s = 0.122%** |
| restart-to-restart, sd of the 3 baseline instance means | **5.62 tok/s = 0.081%** |
| spread of the 3 baseline instance means | 10.38 tok/s = 0.15% |

**Delta +18.59 tok/s = +0.269%.** Standard error of the difference is `sqrt(2/3) × 5.62 = 4.59`, so
**4.05σ**. The arms are completely separated — the lowest patched instance mean (6924.59) is above
the highest baseline instance mean (6920.97) — which under a permutation test on six instances
gives exactly p = 1/20 = 0.050.

### Round 1, `crsuse2-m2m-261` — the first host

Baseline 6 instances, mean **6905.67**, sd of instance means **6.91 tok/s = 0.100%**, spread of the
six means **16.89 tok/s = 0.245%**. Patched 2 instances, mean **6935.75**. Delta **+30.08 tok/s =
+0.436%**, ~4.4 restart-sigma, again completely separated (p = 1/28 = 0.036). Combined Fisher p ≈
0.013 across the two round-1 hosts.

**Why the round-1 hosts give different percentages, which is the interesting part:** the patched
level is essentially identical on both machines — 6935.75 against 6933.13, agreeing to 2.6 tok/s.
It is the *baseline* that moved, by 8.9 tok/s. A higher baseline on n110 shrinks the same absolute
gain into a smaller percentage.

### The floor is a property of the node, and you must measure it

The three nodes give restart sds of 6.91, 5.62 and 4.45 tok/s — a 1.55× range on the quantity every
claim is divided by. Round 1's gate, derived on n261, was **"anything under about 15 tok/s (0.22%)
is not distinguishable from restart noise."** On round 2's tighter node that gate would have been
too conservative: a 20 tok/s win genuinely *was* claimable there, because the spread really is
0.064% and it was measured across six independent boots rather than assumed. Both rounds reached
the same rule from opposite directions — **re-measure the floor on your node before you quote a
ratio against it**, and use the *restart* floor, never the within-process one: decode runs from a
HIP graph captured at startup, so a config change cannot take effect without a restart.

## Deploy

The artifact is **`artifacts/deploy/mixtral8x7b_bf16_tuned_gemm.csv`, 2 rows**, deployed the same
way aiter ships its own per-model tables. Nothing vendored is modified. It is a pure data change:
`"rebuild": []`.

```bash
cp artifacts/deploy/mixtral8x7b_bf16_tuned_gemm.csv \
   /sgl-workspace/aiter/aiter/configs/model_configs/

rm -rf /tmp/aiter_configs      # MANDATORY
# then start the server
```

Equivalently `git apply artifacts/deploy/002_aiter_tuned_gemm_mixtral.patch` in `/sgl-workspace/aiter`.
The deployed CSV hashes to `c85d3ec60d2d2dd0ad6e61fee6ac7c4b`.

**`/tmp/aiter_configs/` is a derived merge cache, not a deploy target.** aiter auto-discovers any
`*bf16_tuned_gemm*.csv` not containing "untuned" under `aiter/configs/model_configs/` and merges it
with `aiter/configs/bf16_tuned_gemm.csv` into `/tmp/aiter_configs/` on first use — see
`AITER_CONFIG.get_config_file` in `aiter/jit/core.py`. The merge is cached and will not regenerate,
so skipping the removal means your rows are ignored with no message at all. Round 2 names this as
"the single most likely way to get a false null" and reports being bitten by it during bring-up.

Note the removal here is **`rm -rf /tmp/aiter_configs`**, the whole directory — broader than round
1's `rm -f /tmp/aiter_configs/tuned_fmoe.csv`, which was scoped to the fmoe table. Different table,
same hazard, different command.

**A restart is mandatory.** The CSV must be in place before launch, because decode's HIP graph is
captured at startup. Dropping it next to a running server does nothing.

No environment variable is required, which matters because the frozen launch line sets none.

**Do not also install `artifacts/evidence_only_001/`.** See "Two rounds".

## Engagement check

Two independent checks were run for 002, because either alone is insufficient.

### 1. The dispatch-log miss line, two-sided, per rank

aiter emits one `not found tuned config` line per rank per missing shape when the dispatch is
built. Same boot procedure, same benchmark, drop-in absent versus present:

| shape | drop-in absent | drop-in present |
| --- | --- | --- |
| `M:64, N:768, K:4096` (`qkv_proj`) | **8** | **0** |
| `M:64, N:8, K:4096` (router gate) | **8** | **0** |
| `M:64, N:4096, K:512` (`o_proj`, control) | 0 | 0 |

Eight is one line per rank on an eight-way TP server; the message carries no TP tag, so the count
*is* the per-rank evidence. `o_proj` reads 0 in both arms because it already hits a tuned row by
shape collision with the kimi / qwen3.5-397B drop-ins — it is the control, and it is deliberately
not in this patch.

### 2. Kernel identity inside the captured decode graph, all eight ranks

The log line is emitted from Python when the dispatch is built; a row could be read there and still
lose to a fallback at runtime. So round 2 captured the decode HIP graph on all eight ranks in each
arm and diffed kernel identity (`round2/kernel_ident.py round2/prof_base round2/prof_cand`). On
**every rank**, the two baseline `MT16x16x1024` kernels are absent from the candidate's graph and
the two tuned kernels are present, at exactly 32 calls each — one per layer.

Shapes were attributed to kernel symbols by dispatching each shape *in isolation* under the
profiler (`round2/gemm_tune/which_kernel.py`), not by assuming the slower baseline kernel was
`qkv_proj`: under the baseline both are hipBLASLt `MT16x16x1024` differing only late in the mangled
name, and graph-replayed dispatches carry no grid/block args to tell them apart. The attribution is
`qkv_proj → …_NTB0_…_SKXCCM0_…`, `gate → …_NTB4_…_SKXCCM8_…`.

### Round 1's check for patch 001 — keep it, it is the model for the above

```bash
for t in 1 2 4 8 16 32 64 128 256 512; do
  printf 'token=%-6s default=%s named=%s\n' $t \
    $(grep -c "using 2stage default for ('gfx950', 256, $t," /tmp/sglang_server_mixtral.log) \
    $(grep -c "using 2stage (kernelName1=.*for ('gfx950', 256, $t," /tmp/sglang_server_mixtral.log)
done
```

- **Engaged:** every bucket reports `default=0 named=8` — one line per rank. Totals **named=80,
  default=0**.
- **Not engaged:** every bucket reports `default=8 named=0`. Totals **named=0, default=80**.

The two prefill buckets (16384, 32768) show 0/0 at boot and only become `named=8` after the first
benchmark run, because nothing has driven a prefill of that size yet. Round 2 re-ran this during
its stacking test and confirmed 001 still engages at `d9e5ef7c`: 8 named `fused_moe` selections and
0 defaults at every token bucket, including decode token=64.

**Do not grep for the "hit" line alone.** It is gated behind `AITER_LOG_TUNED_CONFIG=1`, which the
frozen configuration does not set, so it returns zero against a perfectly working deploy. The
*miss* line prints unconditionally, which is why both checks above are built around the miss line
disappearing and a positive signal appearing together. Counting both directions is what makes them
two-sided.

## Accuracy gate

gsm8k 5-shot, lm-eval pinned at `b315ef3b05176acc9732bb7fdec116abe1ecc476`, task from
`eval/gsm8k.yaml`, via the sealed `scripts/run_eval.sh` against a live server on each arm.

### Round 2 — the gate for what ships

| config | `exact_match,strict-match` | flexible-extract |
| --- | --- | --- |
| bundle reference | 0.6611 ± 0.0130 | 0.6755 ± 0.0129 |
| base, round 2's node | 0.6543 ± 0.0131 | 0.6694 ± 0.0130 |
| **patch 002 (kept)** | **0.6482 ± 0.0132** | **0.6641 ± 0.0130** |

Candidate versus that node's base: −0.0061 strict and −0.0053 flexible, about **0.4 σ** each.
Candidate versus the reference: −0.0129 and −0.0114, both under 1 σ. The gate passes.

**This was the check that mattered, and it is not a formality here.** The `qkv_proj` row is a
flydsl kernel whose mean relative error against an fp32 reference is **0.01740 versus 0.00141 for
the incumbent — 12× worse**, max abs error 0.615 against 0.242. The incumbent accumulates in fp32;
the split-K flydsl kernel does not. And `qkv_proj` is precisely the path that fills an `fp8_e4m3`
KV cache, where a numerics regression degrades output while throughput still looks fine.

Be precise about what the gate establishes: gsm8k at n=1319 cannot resolve a degradation smaller
than about 1.3 points, so what this shows is that the effect is **below that resolution on this
task**. It is not a proof of numerical safety in general. Anyone adopting 002 should re-gate on
their own task.

### Round 1 — the gate patch 001 passed

| config | `exact_match,strict-match` | flexible-extract |
| --- | --- | --- |
| same-node baseline (n110) | 0.6505 ± 0.0131 | 0.6657 ± 0.0130 |
| patch 001 | 0.6664 ± 0.0130 | 0.6793 ± 0.0129 |

No regression. The movement is inside eval scatter and no improvement is claimed. Note the low
absolute score compared to the other entries — this is a 2023 base model, not a regression.

Context for why gates get run at all here: round 1 found the fused-MoE tuner's own accepted
candidates carrying `err2` (cosine diff versus a torch reference) of 0.8–1.1% under its default
`errRatio=0.1` (**10%**) acceptance threshold. "The tuner said it was correct" is not a correctness
statement.

## The deployable artifact in detail — 002

Two rows, both keyed on `gfx950, cu_num=256, M=64, bf16 in / bf16 out, no bias, no scaleAB, no
bpreshuffle`:

| shape | layer | winner | tuner µs | err_ratio |
| --- | --- | --- | --: | --: |
| M=64, N=768, K=4096 | `qkv_proj` | **flydsl** `solidx 3343`, tile 16×64×128, **splitK 4** | 7.2969 | **0.0194** |
| M=64, N=8, K=4096 | MoE router gate | **hipblaslt** `solidx 440197`, `MT16x16x512` | 5.7991 | 0.0 |

`o_proj` (M=64, N=4096, K=512) is deliberately **not** in the file: it already hits a tuned row and
serves as the untouched control in every A/B below.

### Why exactly these two shapes were untuned, established from the live system

With `AITER_LOG_TUNED_CONFIG=1` the server emits an unconditional miss line per lookup. Counted
over a full run:

| (N, K) | layer | miss lines | misses at the decode shape M=64 |
| --- | --- | --: | --- |
| (768, 4096) | `qkv_proj` | 5552 | **yes** (8 = one per rank) |
| (8, 4096) | router gate | 5552 | **yes** (8 = one per rank) |
| (4096, 512) | `o_proj` | 40 | **no** |

`aiter/configs/bf16_tuned_gemm.csv` has **112 rows** and covers only
N/K ∈ {128/6144, 2048/6144, 3072/6144, 50016/6144, 6144/2048, 6144/3072, 256/5120} — nothing
Mixtral-shaped. On a miss, `tuned_gemm.get_GEMM_A16W16_config` falls through to
`default_config["libtype"] = "torch"`, i.e. hipBLASLt's own heuristic.

**`o_proj` is the control that proves the mechanism, and it is the finding worth carrying to every
other model.** It is tuned — flydsl, `solidx 4183`, 4.7372 µs — but only *by luck*: that row exists
because `model_configs/kimi_bf16_tuned_gemm.csv` and `qwen3_5_397b_bf16_tuned_gemm.csv` happen to
contain the same (4096, 512) shape. **Thirteen model drop-ins ship with aiter and none of them is
Mixtral, so the two shapes no sibling model shares are exactly the two that miss.** A merged table
serves other models' rows to you; "no tuned rows for my model" does not mean "no tuned rows on my
path," and a grep of the base file tells you less than a grep of the merged one.

### Getting to the tuner at all

The correct tuner at this commit is **`csrc/gemm_a16w16/gemm_a16w16_tune.py`** — backends asm,
opus, flydsl, triton, skinny, torch, with hipBLASLt **opt-in** via `--with-hipblaslt`, and
`--libtype` available as an input flag. Two shapes, all backends, **317.6 s**.

`gradlib/gradlib/GemmTuner.py` — which three skill files route you to, and which is capitalised
differently from what they say — is **hipblaslt-only** at `d9e5ef7c`; its own docstring says the
non-hipblaslt backends have been removed. It therefore could not have found this bundle's
`qkv_proj` winner, which is flydsl. Round 2 ran the wrong tuner first and lost ~40 minutes.
`pretune.py --list` surfaces neither tuner.

### The three-tier evidence hierarchy this patch establishes

The same substitution measured three ways, and the three disagree in a way that is the single most
transferable lesson in this entry:

| | `qkv_proj` | router gate | `o_proj` (untouched control) |
| --- | --- | --- | --- |
| tuner µs (ranks candidates only) | 7.2969 | 5.7991 | — |
| isolated same-harness A/B, HIP-graph captured | 6.104 ± 0.003 → 5.828 ± 0.011, **−4.5%** | 5.890 ± 0.019 → 4.840 ± 0.007, **−17.8%** | 2.478 ± 0.007 → 2.463 ± 0.003, −0.6% |
| **in-graph, deployed server, 8 ranks** | **−1.091 ± 0.133 µs/call, −13.1%** | **−0.316 ± 0.194 µs/call, −3.9%** | −0.059 ± 0.117, n.s. |

**The isolated A/B inverted the ranking.** An isolated GEMM owns the caches, the memory system and
all 256 CUs; in the server the same kernel runs between a 49 µs MoE GEMM and an all-reduce on a
memory system that eight ranks are saturating. The gate is a 0.72 TFLOP/s shape that is pure
launch-and-latency, so its isolated advantage largely evaporates under contention, while
`qkv_proj`, the bigger bandwidth-bound shape, *gains* relative to isolation. The isolated harness
also runs ~2.1 µs faster than the server for both changed shapes (6.10 against 8.35, 5.89 against
7.95) — a consistent additive offset which is the in-graph dispatch floor.

The rule: **isolated timing establishes sign, in-graph timing establishes magnitude, and only the
restart-paired end-to-end A/B is the claim.**

### And it closes end-to-end two independent ways

Net over 32 layers: **−45.0 ± 9.8 µs per decode step**, on a graph body measured at 5808.5 µs/step.
The `o_proj` control moves −0.059 ± 0.117 µs/call — consistent with zero, which is what licenses
reading the other two as the change rather than as drift.

- −45.0 µs on a 5808.5 µs graph body is −0.775% of decode step time; decode is roughly 78% of
  per-request latency at this ISL/OSL, predicting ≈ +0.60% end to end against **+0.474% observed**.
- Measured mean TPOT falls **7.238 → 7.196 ms**, i.e. **−42 µs per decode step**, against −45.0 ±
  9.8 µs predicted from the profile. **Mean TTFT is unchanged** (~2050 ms both arms) — the correct
  signature for a change that only touches the captured decode graph.

**Honest note on the split: the `qkv_proj` row carries the win and the gate row contributes
little.** The gate improves on 7 of 8 ranks but by only −0.32 µs/call, and on TP-0 it is
+0.09 µs/call. No qkv-only ablation was run, so there is **no confidence interval on the gate row's
end-to-end contribution alone**. On the in-graph evidence it is small but positive and it costs
nothing numerically (`err_ratio` 0.0), so it ships. Round 2 names this as the first thing it would
run next.

**There is no numerically-clean speedup available for `qkv_proj`.** Re-running the tuner at
`--errRatio 0.003` returns `hipblaslt solidx 439688`, `MT16x16x1024`, **8.4974 µs, err_ratio 0.0** —
the same tile family as the incumbent and slower than it. So the only available speedup at this
shape is the one that costs precision, and the justification for shipping it is the empirical gsm8k
gate above, not the error metric.

## The retained artifact in detail — 001, not deployed

Kept in `artifacts/evidence_only_001/` because it still carries evidence. 12 rows, one per token
bucket: **1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 16384, 32768**, each keyed on
`gfx950, cu_num=256, model_dim=4096, inter_dim=1792, expert=8, topk=2, ActivationType.Silu`, bf16
activations and weights, `QuantType.No`, `use_g1u1=1`, `doweight_stage1=0`. The bucket list was
enumerated from the server log rather than guessed — each appears 8 times, once per rank. md5
`3910458156ceae313ad9d8a41f7edd87`.

The rows name explicit `kernelName1` and `kernelName2` CK instances, selecting
`moe_ck2stages_gemm1_256x32x64x**128**_...` where the heuristic picks the `...x**64**_...` variant —
a doubled K-tile. `block_m` is left at the default 32 for the decode bucket. Produced with the
vendor tuner, which took **13–16 seconds for all 12 shapes**:

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i <untuned.csv> -o <tuned.csv>
```

Note the flag spelling: `-i`/`-o`. The a16w16 tuner in the same tree uses `--input_file`/`--tuned_file`
instead.

Op-level, the tuned instance is a real improvement: median **88.1 → 80.8 µs, −8.3%**, disjoint
distributions, 4.00 → 4.36 TB/s, measured under HIP-graph replay in interleaved fresh processes. In
situ, the same substitution measured **+1.32%** — that is, no decode speedup at all. Both numbers
are correct; see the withdrawn claims. One side effect worth knowing if you ever do deploy it:
deploying any tuned fmoe row disables `use_non_temporal_load`, measured at about 0.5%. On this
model it happens to help.

Round 2 independently closed the obvious follow-up: the hypothesis that round 1 tuned the fused-MoE
config only at prefill token counts and never at the decode batch is **false** — the CSV contains a
token=64 row, and TPOT was still flat. **Instance selection on this kernel is exhausted.** What
remains is real but is not a config problem: MoE stage 1 moves 234.9 MB in 49.2 µs = **4.77 TB/s,
about 60% of the ~8 TB/s peak**. Recovering that needs better CK kernel code, not a better table
row — the highest ceiling in the bundle (~4.6% end to end for stage 1 alone) and the highest
effort. Stated as a lead, not a result.

## Where decode time actually goes

Round 2's in-situ budget on its own node, rank 0, graph body **5808.5 µs/step over 25 kernels**,
ranked by summed device duration from `cat=="kernel"` records. This independently reproduces round
1's corrected table on a different node, including its correction that `MT16x16x1024` is
**`qkv_proj` and the gate**, not `o_proj`.

| kernel | impl | µs/call | calls | µs/step | % |
| --- | --- | --: | --: | --: | --: |
| MoE stage 1 (w13) | `ck::kernel_moe_gemm` | 49.211 | 32 | 1574.7 | 27.11 |
| paged attention | `paged_attention_ll4mi_QKV_mfma16` | 30.159 | 32 | 965.1 | 16.62 |
| MoE stage 2 (w2) | `ck::kernel_moe_gemm` | 27.033 | 32 | 865.1 | 14.89 |
| all-reduce | `aiter::cross_device_reduce_2stage` | 10.070 | 65 | 654.5 | 11.27 |
| rmsnorm+quant | `aiter::add_rmsnorm_quant_kernel` | 4.215 | 64 | 269.8 | 4.64 |
| **`qkv_proj`** | hipBLASLt `MT16x16x1024` | **8.352** | 32 | 267.2 | **4.60** |
| **router gate** | hipBLASLt `MT16x16x1024` | **7.949** | 32 | 254.4 | **4.38** |
| MoE sorting | `aiter::opus_moe_sorting_entry` | 6.794 | 32 | 217.4 | 3.74 |
| `o_proj` | aiter flydsl `hgemm_bf16_16x64x64x6` | 4.237 | 32 | 135.6 | 2.33 |
| rotary | `rotary_embedding_kernel` | 4.197 | 32 | 134.3 | 2.31 |
| KV store | `reshape_and_cache_flash` | 4.163 | 32 | 133.2 | 2.29 |
| attn reduce | `paged_attention_ll4mi_reduce` | 4.111 | 32 | 131.6 | 2.27 |
| gate softmax | `vllm::moe::topkGatingSoftmax` | 3.817 | 32 | 122.1 | 2.10 |

Patch 002 is the 8.98% those two GEMM rows add up to.

**The launch floor is real and its value here is ~3.8–4.2 µs.** Six kernels doing almost no
arithmetic — rotary 4.197, `reshape_and_cache_flash` 4.163, attention reduce 4.111, `o_proj` 4.237,
rmsnorm+quant 4.215, gate softmax 3.817 — sit in a 0.4 µs band regardless of how much work they do.
That band is the in-graph dispatch cost on this stack. You cannot make a 4.1 µs kernel that moves
0.1 MB into a 2 µs kernel; you can only stop launching it. Fusion is the structurally correct
attack on that fifth of the budget, and neither round attempted it.

A tooling note that bit both rounds in mirror-image ways: the graph body is identified by call
**multiplicity**, and the off-by-one kernels are the ones that bite. Round 2's first budget script
filtered on `calls % 32 == 0` and silently dropped `cross_device_reduce_2stage` (65 calls),
understating the denominator by 11%; round 1 made the opposite error by keeping only the 32-call
kernels. Dump all kernels with call counts.

## What was tried and did not work

| attempt | round | kernel / op level | end to end | verdict |
| --- | --- | --- | --- | --- |
| route bf16 dense linears through `aiter.tuned_gemm.tgemm` | 1 | **identical kernels in both arms** — profiler shows the same hipBLASLt `Cijk_...MT16x16x1024`; `o_proj` was *already* on a flydsl `hgemm_bf16_*` path | **+0.29 tok/s (+0.004%)** | **A literal no-op.** The premise was that `_use_aiter=False` because the launch script sets no `SGLANG_USE_AITER`. The image sets it. The whole attempt rested on not reading the process environment. |
| tuned router-gate GEMM, `M:64, N:8, K:4096` | 1 | tuner said **5.7518 µs**; in situ the substituted kernel went 7.954 → 7.851 µs, **−1.29%** | ~+2.5 tok/s predicted | **Not shipped in round 1** — below the 5.6 tok/s restart sd. The instructive part: byte-identical *untouched* control kernels drifted −0.93% mean, sd 1.81%, worst 5.30% between instances. The substituted kernel moved *less than the kernels that did not change*. And the isolated harness reported **5.75 µs for a kernel that measured 7.85 µs in the server** (5.7518 against 7.851). Round 2 re-found bit-for-bit the same solution and shipped it as one of 002's two rows — see the corrected reading below. |
| force INT8 quickreduce onto the decode all-reduce | 1 | engaged perfectly — **65 quickreduce, 0 cross_device_reduce_2stage on all 8 ranks**, an exact 1:1 substitution — and ran **610.4 → 2107.6 µs per rank per step, 3.45× slower** | not run | **Rejected.** SGLang's `ca`-before-`qr` ordering and its 4 MB INT8 floor are already correct for a 512 KB decode message. Prefill *does* use quickreduce Q8; decode does not, and that is deliberate. |
| lowering the quickreduce INT8 threshold 4 MB → 256 KB | 1 | **0 quickreduce calls, 65 two-stage, on all ranks** | none | A no-op, because `ca` is tested *before* `qr` regardless of threshold. **Tuning a threshold is useless when the ordering short-circuits it** — check the dispatch order before the constants. |
| aiter's EXPERIMENTAL paged-attention kernel (`QKV_VERSION=EXPERIMENTAL`) — the transfer that a sibling Qwen3-14B run got +2.84% from | 2 | at the **production** fp8_e4m3 KV shape: GOLDEN 34.731 µs and correct, EXPERIMENTAL 37.714 µs and **100% NaN**. At bf16 KV, where it is supported: 55.437 → 54.838 µs, **−1.1%** | not run — paged attention is 16.6% of decode, so the ceiling on a −1.1% kernel is **~0.18% end to end**, below the floor | **Rejected, and it is a safety bug, not just a negative.** It compiles cleanly at the production configuration (`~/.aiter/build/pa_ragged_09f9db1ae880d7461e800099fb909a2b`, `VERSION_ID = 1`, `kFp8E4M3`) and returns all-NaN. **aiter's own log line is factually false**: it prints `EXPERIMENTAL pa_ragged kernel requires head_size=128 and kv_dtype=bf16. Fallback to original kernel`, and there is no fallback in the code — `version=version` is passed to `compile_template_op` regardless. **A user who trusts that message ships silent NaNs at full throughput.** The Qwen3-14B win does not transfer, for a specific and checkable reason: Mixtral runs an FP8 KV cache and the experimental kernel has no working FP8 path. |
| re-tuning `qkv_proj` for a numerically clean win, `--errRatio 0.003` | 2 | returns `hipblaslt solidx 439688`, `MT16x16x1024`, **8.4974 µs, err_ratio 0.0** — same tile family as the incumbent and slower than it | not run | **No clean speedup exists at this shape in any backend aiter ships.** That is itself the result. The fast row ships and the gate is empirical. |
| stacking round 1's patch 001 on top of 002 | 2 | 001 engaged: 8 named `fused_moe` selections per rank, 0 defaults, every bucket including token=64 | 002 only **6950.41** (4 instances, restart sd 4.57); 001+002 **6939.56** (2 instances, restart sd 9.50); point estimate **−10.9 tok/s** | **No measurable benefit.** Inside the stack arm's own restart spread at n=2, so this does not support "001 is harmful" either. 001 is not carried forward. Side observation: the stack arm's within-instance sd was 11.36 and 12.79 against 2.56–7.67 everywhere else — adding the fmoe table made run-to-run behaviour noticeably noisier, unexplained. |
| MoE decode config — "round 1 only tuned prefill token counts" | 2 | hypothesis **false**: round 1's CSV contains a token=64 row and TPOT was still flat | none | **Closed.** Instance selection on the fused-MoE kernel is exhausted. What is left is a 4.77 TB/s stage-1 kernel at ~60% of peak, which needs kernel code, not a table row. |
| `gradlib/gradlib/gemm_tuner.py` for dense bf16 GEMM, as three skill files instruct | 2 | the file is `GemmTuner.py` and is **hipblaslt-only** at `d9e5ef7c` — it cannot race flydsl, so it cannot find 002's `qkv_proj` winner | — | ~40 minutes lost. Use `csrc/gemm_a16w16/gemm_a16w16_tune.py`. Three agreeing documents are not a cross-check if they share a source. |

## Claims made and then withdrawn

The most instructive content in the bundle, and the reason this entry exists in the form it does.
Each was caught by re-measuring rather than by re-reasoning. **Round 1's three are kept verbatim
below; round 2 added two more, one of which is a correction to round 1's stated reasoning.**

### Round 1

**1. "MoE stage 1 improved 16.9%, and two independent measurements agree."** The 49.70 µs came from
the bundle's stock trace and the 41.29 µs from the agent's own microbenchmark — different
harnesses, different profiling overhead, different machines — presented as a single controlled A/B.
In-situ profiling gave 49.740 → 50.398 ± 1.264 µs, **+1.32%**, and the stage-2 kernel, whose name
is byte-identical between arms, drifted −1.28% on its own. Corrected: the K-tile substitution is
real, **the decode speedup is not**. Two numbers from two harnesses are not two independent
measurements.

**2. "The microbenchmark overstates by about 7×,"** explained by single-GPU versus 8-GPU saturation
and clock limits. This was a category error, not a magnitude error: a physical story invented to
reconcile two numbers instead of re-measuring them. Splitting the benchmark by phase showed TPOT
flat at 7.2367 → 7.2333 ms and TTFT down 23.63 ms. **The decode bucket contributes nothing; 001's
gain is prefill, from the 16384 and 32768 rows.** The patch is a prefill optimization that was
defended for half the run as a decode optimization. Phase-attribute before you believe a kernel
win.

**3. "The `o_proj` win is invisible end-to-end because the saved time is absorbed by the downstream
all-reduce."** A plausible mechanism, and completely inapplicable: the routing patch was the no-op
above, so no kernel changed and there was nothing to absorb. Corrected: +0.29 tok/s is **zero
effect, correctly measured**. Absorption remains plausible in general at TP=8; it is simply not
what happened here.

### Round 2

**4. "The isolated harness overstated the available gate win by 21×" — the *mechanism* is withdrawn;
the conclusion stands.** Round 1 predicted its gate win by comparing the *tuner's* µs for the
substituted kernel (5.7518) against the *trace's* µs for the incumbent (7.954), concluded −28%,
measured −1.29% in situ, and blamed the isolated harness. Round 2 measured both arms in one harness
and found a **consistent ~2.1 µs additive offset** between harness and server for both changed
shapes (6.104 against 8.352 for `qkv_proj`, 5.890 against 7.949 for the gate) — the in-graph
dispatch floor. **The comparison was invalid because it crossed harnesses, not because either
harness was bad.** Round 1's decision not to ship the gate alone was right; its stated reason was
not, and as stated it discourages isolated measurement generally, which is the wrong lesson.

Keep round 1's raw observation — **the isolated harness reported 5.75 µs for a kernel that measured
7.85 µs in the server** (5.7518 against 7.851) — because it is the datum that makes the additive
offset visible. Discard the "21× overstatement" framing.

**5. "The kernel deltas will probably land below the noise floor, and the gate will carry whatever
gain there is."** Round 2 recorded this prior before measuring, which was the right thing to do, and
it was wrong in both magnitude and mechanism. The result was +0.474%, and it is carried by
`qkv_proj`, not the gate. The isolated same-harness A/B ranked the gate at −17.8% and `qkv_proj` at
−4.5%; the deployed graph says **−3.9% and −13.1%**. The ranking inverted. Round 2 had corrected
round 1's cross-harness error and then made a smaller version of the same mistake: **removing the
harness difference is not the same as removing the context difference.** The only kernel
measurement that predicted the end-to-end result was the one taken inside the deployed graph.

The rule both rounds converged on, from opposite directions, is the one to carry forward: **when a
result needs a story to make sense, measure it again.**

## When this entry stops applying

Silently, in every case.

**For 002, the deployable win:**

- **`cu_num` ≠ 256 or arch ≠ gfx950** — the rows are literally keyed on them.
- **Concurrency ≠ 64** — this kills the *entire* 002 gain. Both rows are keyed on **M=64**, which is
  the padded decode batch for this workload (`get_padded_m` rounds 40/48/56 up to 64). A deployment
  that steadily runs a different concurrency will hit neither row, and will see neither the gain nor
  the numerics risk.
- **TP ≠ 8** — `qkv_proj` N moves off 768 and that row misses. The gate row (N=8, the expert count)
  is TP-independent, but it is the row that contributes little.
- **A more numerically sensitive task, or longer context** — the `qkv_proj` row is 12× dirtier in
  mean relative error and feeds an fp8 KV cache. gsm8k holds at 0.4 σ but resolves nothing finer
  than ~1.3 points. If your own gate fails, deleting the `qkv_proj` row and keeping only the gate
  row is numerically free — and gives up most of the gain.
- **FP8 weights instead of bf16** — different table entirely.
- **A different aiter commit** — merge path, schema or the set of shipped model drop-ins may change,
  and the drop-in set is exactly what determines which of your shapes are already covered by
  collision.
- **Stale `/tmp/aiter_configs`**, or deploying after the server is up.

**For 001, if you ever reinstate it as a record-keeping exercise:**

- **TP ≠ 8** — `inter_dim` moves off 1792 and every row misses.
- **`--chunked-prefill-size` or `--max-prefill-tokens` ≠ 32768** — this kills the *entire* 001 gain,
  since all of it lives in the prefill buckets.
- **Concurrency ≠ 64** — moves the decode bucket, which contributes nothing anyway.

Still reusable when both are inert: the 12-bucket shape list, the two decode GEMM shapes, the
finding that the CK 2-stage heuristic under-sizes the K-tile on this geometry, the finding that a
model silently inherits another model's tuned row when shapes collide, the `model_configs/` drop-in
method, the three-tier evidence hierarchy, and every row of the negatives table.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/hold_mixtral_8x7b_tuning/`. It holds both
rounds; round 1's report is preserved at `round1/FINDINGS.md` with its parking note at
`round1/HOLD.md`, and the top-level `FINDINGS.md` is round 2's.

**Round 2** (`FINDINGS.md`, 661 lines): §1.4 the noise floor, §2.1 the paged-attention negative and
the false-fallback bug, §2.2 the decode budget, §2.3 the closed MoE-config avenue, §2.4 the kept
change end to end, §3.2 the throughput campaign, §3.3 the latency signature, §3.5 the 001+002
stacking test, §3.6 what ships, §4 the skillset assessment. `EXPERIMENT_COMPLETE` is the one-line
summary. `patches/002_aiter_tuned_gemm_mixtral/RESULT.md` has every per-run number and five named
risks; its `MANIFEST.json` has the deploy and cache-invalidation commands. Round-2 tooling is under
`round2/`: `kernel_ident.py` for the eight-rank graph diff, `gemm_tune/which_kernel.py` for shape
attribution, `gemm_tune/tgemm_ab.py` for the isolated A/B, `pa_ab.py` for the paged-attention
harness, `prof_budget.py` for the decode budget, `prof_base/` and `prof_cand/` for the eight rank
traces.

**Round 1** (`round1/FINDINGS.md`, ~1650 lines): §1.3 and §2.5 the floors, §2.5.1 the in-situ
profiling that forced withdrawal 1, §2.6.3 the no-op proof, §2.7 the router gate, §2.9 the
quickreduce experiment. `patches/001_aiter_tuned_fmoe_mixtral/RESULT.md` has every per-run number.
`PREFLIGHT.md` records the 15 tok/s significance gate as derived on m2m-261, and `HANDOFF.md`
records the mid-run node change. The harness is `tuning/relaunch.sh` and `tuning/measure.sh`; the
microbenchmarks are `analysis/bench_moe.py` and `analysis/bench_linear.py`; the no-op profiles are
`analysis/prof_route_baseline.txt` and `analysis/prof_route_patched.txt`.

Round-2 node name: **not recorded** anywhere in the bundle.
