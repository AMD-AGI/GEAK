# gpt-oss-120b on 2× MI355X — SGLang 0.5.17, TP=2, three Triton attention source patches

**Measured win: +15.66% output throughput** (3938.109 → 4554.813 tok/s), gsm8k 5-shot strict-match
**0.962851 → 0.962851** — numerically identical to the baseline. The win is carried by three source
patches to two files in two repositories: two in SGLang's Triton prefill extend-attention kernel and
one in aiter's Triton decode attention. No configuration flag, no environment variable, no tuned
config table, and nothing that needs a rebuild.

Found 2026-08-20 over a single-day run on host `crsuse2-m2m-224`.

> **Reproduction status: not reproduced on a clean instance.** Read this before the result.
>
> Everything below was measured on one host, in one container, over one session. What *was*
> repeated, and repeated from these exact patch files: `artifacts/ab_restart.sh` applies and reverts
> them with `git apply` around every server restart, and the 01+02 arm was re-measured **14 restarts
> later** as the base arm of the patch-03 campaign, landing at 4520.931 tok/s against the earlier
> 4522.506 — **0.03% apart**. That is a real reproduction of the artifact's effect *within the
> campaign*, and it is why this entry exists at all. What has not
> happened is a fresh container on a fresh host reaching 4554.813 from `artifacts/` alone. The house
> bar (`../README.md`, "Adding an entry") is that stronger thing. Treat the number as solid and the
> *transfer* as untested, exactly as `minimax-m3-mxfp4/` is labelled.

## Which number to quote, and against what

There are two defensible headline figures here and they differ by more than seven points, so state
the base every time.

| comparison | baseline | tuned | delta |
| --- | --- | --- | --- |
| **against the locally measured baseline** | **3938.109** | **4554.813** | **+15.66%** |
| against the bundle's reference figure | 4201.601 | 4554.813 | +8.41% |

**Quote +15.66%.** The bundle's 4201.601 is the *source session's* number, measured on a different
machine and never reproduced here: five local rounds of the identical frozen configuration averaged
3938.109, which is **6.27% below** the reference and far outside the reference's own 0.86%
three-round spread. `BASELINE.md`'s own rule covers this case — "if your mean lands outside the
reference spread, your figure becomes the baseline" — so the local mean governs and 4201.601 is
quoted alongside rather than used. Note the tuned figure passes the reference number too, which is
the useful thing about carrying both: the win is not an artifact of a low local baseline.

The reference figure is itself the *ceiling of a configuration search*, not a stock arm. The
untouched configuration served **1451.311 tok/s**; `--page-size 64` alone took it to 4201.601, a
factor of 2.9. That configuration win is already banked in this entry's baseline and is not
re-discoverable. Never quote a gain against 1451.311.

*(The bundle records that percentage two different ways — `BASELINE.md` says +183.50%, its own
`README.md` says +189.50%, and the two throughputs give +189.51%. Quote the throughputs.)*

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | **2× MI355X, `gfx950`, 256 CU** | **yes** — patch 02 gates on `_is_gfx95`, patch 03 on `DEVICE_ARCH == "gfx950"`; both are dead on any other arch. CU count enters patch 03's guard through `get_num_sms()` and its tuning was measured at 256 |
| model | gpt-oss-120b, **TP=2**, 36 layers (18 sliding-window(128) + 18 full attention), `head_dim` **64**, 8 KV heads globally → **32 q / 4 kv heads per rank** | **yes** — `head_dim` 64 is literally the predicate patch 02 widens and patch 03 tests; the 4 local KV heads are a term in the segment-count formula patch 03 overrides; the 18/18 interleave is what patch 01 exploits |
| precision | **MXFP4 expert weights, bf16 activations and attention, bf16 KV cache** | **yes** — patch 03 requires `kv_cache_dtype == torch.bfloat16` and skips otherwise. bf16 KV is confirmed by the profiled kernel name (`IS_KV_FP8_0`) and by patch 03's guard, not inferred from a flag |
| page size | **64** (`--page-size 64`) | **yes** — it sets `BLOCK_SIZE_64` / `TILE_SIZE_64` in the decode kernel, and patch 03 requires `TILE_SIZE == 64`. It is also the entire configuration win underneath this baseline |
| prefill attention backend | **triton** | **yes** — patches 01 and 02 edit SGLang's Triton extend kernel. On any other prefill backend that file is not on the path and both patches are inert *without failing* |
| decode attention backend | **aiter** | **yes** — patch 03 edits aiter's Triton `unified_attention`. Same silent-inert failure on any other decode backend |
| MoE runner | aiter (ck_tile MXFP4 path) | descriptive — 32.9% of decode busy time, and untouched by this campaign |
| SGLang | 0.5.17, commit **`29481685462732237d80d86076d6563e1f658102`** at `/sgl-workspace/sglang` | **yes** — patches 01/02 are context diffs against this tree |
| aiter | commit **`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`** at `/sgl-workspace/aiter`, an editable checkout | **yes** — patch 03 is a context diff against this tree |
| ROCm / torch / Triton / python | 7.2.0 / 2.9.1+rocm7.2.0.git7e1940d4 / 3.6.0 / 3.10.12 (`/opt/venv/bin/python3`) | descriptive — but see `gemma-4-26b-a4b-it/` for a baseline that came in 4.43% low on a newer image, most of it attributed to a Triton 3.6.0 → 3.7.0 bump |
| host | `crsuse2-m2m-224`; model at `/shared_nfs/hyperloom/models/gpt-oss-120b`, 15 safetensors shards | descriptive |
| container | **not recorded** — see below | descriptive |

**The container digest is a gap, and the run says so rather than guessing.** There is no `docker`
binary inside the container and no image label on the filesystem, so the image could not be
confirmed from where the measurements were taken. `artifacts/start_container.sh` defaults to
`harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` and the
stack reported inside matches that tag, but that is inference from the launcher, not confirmation,
and **no `sha256:` digest exists anywhere in the bundle.** A reader trying to match this environment
exactly cannot, and would have to re-derive it from the two commit shas and the version table above.
The two shas are the stronger pins in practice, because they are what the patches actually apply to.

Both repositories are real git checkouts here, which is worth noting because it is not always true:
patch 03's header records that `git status --porcelain` was empty on the aiter file before editing,
so the diff is against upstream rather than against an earlier experiment's leftovers.

## Launch configuration

Reproduce verbatim. This is `artifacts/launch_server.sh`, run inside the container:

```bash
export SGLANG_USE_AITER=1

python3 -m sglang.launch_server \
    --model-path /shared_nfs/hyperloom/models/gpt-oss-120b \
    --host 0.0.0.0 --port 43112 \
    --tp-size 2 \
    --context-length 11264 \
    --watchdog-timeout 1800 \
    --prefill-attention-backend triton \
    --decode-attention-backend aiter \
    --moe-runner-backend aiter \
    --mem-fraction-static 0.68 \
    --chunked-prefill-size 16384 \
    --disable-radix-cache \
    --page-size 64
```

**`SGLANG_USE_AITER=1` is the whole environment recipe as far as the launcher is concerned** — and
per `../README.md`'s standing warning, that is a statement about the launcher, not about the process.
The image's own exports were **not** enumerated in this run (`cat /proc/<pid>/environ | tr '\0' '\n'`
was not recorded), which is a small gap: it means "no other env vars" is an assumption here, not a
verified fact. It did not affect any result, because both arms of every A/B ran under the same
process environment.

`artifacts/launch_server.sh` does not merely launch; it queries `/get_server_info` and refuses to
exit 0 unless the live server reports `context_length 11264`, `tp_size 2`, **`page_size 64`**,
`moe_runner_backend aiter`, `prefill_attention_backend triton`, `decode_attention_backend aiter`,
`chunked_prefill_size 16384`, `disable_radix_cache true`, and a `mem_fraction_static` of either 0.68
or 0.578.

**Why that check earns its place, and why it is not optional here.** `page_size 64` is the entire
1451.311 → 4201.601 configuration win sitting underneath this entry. If the flag is dropped or
silently rescaled, the server comes up healthy, answers correctly, and serves at roughly a third of
the rate — a failure indistinguishable from "my patch destroyed the model". The
`mem_fraction_static` exemption is the second half of the same lesson: SGLang rescales it by 0.85 on
builds that combine aiter with a context length above 8192, so a naive equality check on 0.68 fails
on a perfectly good server and would have blocked every measurement in this run. Check the values
that matter, and know which ones the framework is allowed to rewrite.

Every A/B round in this entry gated on the string `config verified` appearing in the launch log
before the benchmark client was allowed to start (`artifacts/ab_restart.sh`, `artifacts/gate*.sh`).

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, **8 warmups**, seed 0, `random_range_ratio 1.0`,
`--ignore-eos`, `--random-prefix-len 0`, InferenceX `benchmark_serving` fork against
`/v1/completions` (`artifacts/run_bench.sh`). A round takes about 44–50 s, which makes this the
cheapest bundle in this directory to measure — run more repeats than you would elsewhere.

What sets the shapes the patches were tuned against:

- `--chunked-prefill-size 16384` with **ISL 8192** means prefill batches are whole sequences: chunks
  of one or two complete 8192-token requests. That is the `bs 2 × extend 8192` shape both prefill
  sweeps used, and it is why `BLOCK_M = 128` pays — 128 M-blocks each re-reading the prefix at
  `BLOCK_M = 64` is what makes the kernel traffic-bound.
- **Concurrency 64** sets decode batch size 64, which is the `bs` that puts patch 03's guard term
  `num_2d_prgms * 4 >= 2 * get_num_sms()` on the right side of its threshold. At bs 8 the guard
  deliberately does not fire.
- `--random-range-ratio 1.0` makes every prompt unique, which is why `--disable-radix-cache` costs
  nothing: prefix caching has nothing to hit and only pays bookkeeping.
- **`--page-size 64`** is what makes the decode kernel's `BLOCK_SIZE`/`TILE_SIZE` 64.

Change ISL, concurrency or chunked-prefill size and you are off the tuned point in a way the patches
will not warn you about.

## Baseline and noise floor

### The arms

| | value |
| --- | --- |
| stock, this stack (5 rounds over 2 server processes) | **3938.109 tok/s**, TTFT 4164.1 ms, TPOT 12.182 ms |
| with patches 01 + 02 (3-pair interleaved A/B) | **4522.506 tok/s**, TTFT 3090.6 ms, TPOT 11.132 ms |
| with patches 01 + 02 + 03 (14 clean pairs of 16) | **4554.813 tok/s**, TTFT ~3096 ms, TPOT 11.037 ms |
| delta | **+15.66%** output throughput, **≈−25.7%** TTFT, **−9.4%** TPOT |

The five baseline rounds, every one recorded: 3927.966, 3943.354, 3951.163 (process A, no restart
between them), then 3934.449, 3933.613 (process B, after a full stop and relaunch).

### The two floors, measured separately

| noise floor | how | n | mean | min | max | spread | stdev |
| --- | --- | --- | --: | --: | --: | --: | --: |
| repeating the benchmark **within one process** | consecutive rounds, no restart | 3 | 3940.828 | 3927.966 | 3951.163 | **0.589%** | 0.300% |
| **across restarts** | one round each against 6 freshly started servers, identical unpatched code | 6 | 3954.967 | 3947.258 | 3967.072 | **0.501%** | 0.186% |

**The restart floor governs, and it is 0.501%.** All three patches are Python/Triton source changes
that take effect only at process start — Triton JIT plus HIP-graph capture — so the arms must
alternate at restart granularity (`../../tuning-core/measurement.md` Rule 3b). Every base and
candidate sample in this entry is one benchmark round against one freshly started server.

Two things about this stack are worth carrying to the next run:

**The restart spread came out *smaller* than the within-process spread**, a ratio of 0.85×. That is
the opposite of the Qwen3-8B stack that motivates Rule 3b, where the ratio is 26×. Importing the 26×
figure here would have implied a floor near 15% and thrown away both prefill wins. Measure your own
ratio; do not inherit one.

**The within-process samples are monotonic, +0.591% from round 1 to round 3** — throughput climbs
against a fresh server and then settles. Round *position* is therefore a confound, and this run
designs it out rather than correcting for it: `artifacts/ab_restart.sh` runs exactly **one** bench
round per server start, so every sample in both arms is a round-1-against-a-fresh-server sample and
like is compared with like by construction. The 6 restart samples above are not a separate
experiment — they are the pooled base arms of the two prefill A/Bs, which is why they are guaranteed
to be on identical code.

### Disjointness, per patch

| arm pair | base rounds | candidate rounds | disjoint? |
| --- | --- | --- | --- |
| 01 vs base | 3947.258 / 3958.471 / 3954.008 | 4194.478 / 4252.199 / 4316.892 | **yes** — worst candidate beats best base by 5.96%, ~10× the floor |
| 01+02 vs base | 3955.210 / 3967.072 / 3947.785 | 4506.781 / 4526.655 / 4534.082 | **yes** — no overlap at all; arm spreads 0.49% and 0.60%, both sitting on the floor |
| 01+02+03 vs 01+02 | 14 clean pairs, mean 4520.931, spread 1.086% | 14 clean pairs, mean 4554.813, spread 0.907% | **no**, and it cannot be — see below |

**Patch 03 does not satisfy the disjoint-distributions criterion and the run says so plainly.** The
reason is a tail, not the patch. Across the 32 restarts of the patch-03 campaign, **two instances
came up sick** — 4428.7 tok/s (candidate) and 4351.4 tok/s (base) against a ~4520 norm — and each
was identifiable *independently of its throughput* by a mean TTFT 200–330 ms above the ~3100 ms
norm. One landed on each arm, which is also the evidence that they are restart lottery rather than
patch-related. So the honest restart distribution on this stack is ~0.5% wide **with a heavy left
tail at roughly 6% incidence**, and any effect under several percent is non-disjoint by construction
once a campaign runs long enough to sample the tail.

The criterion that does work is distribution-free: **15 of 16 pairs favour the candidate, two-sided
sign test p = 0.00052.** That is the stated basis for the +0.750% claim, and it is the most portable
methodological result in this entry.

## Per-patch attribution

Attribution is stated exactly as measured, including which base each delta is against. Do not add
the three percentages together.

| # | artifact | repo / file | kernel-level | end-to-end | measured against |
| --- | --- | --- | --- | --- | --- |
| 01 | `artifacts/01-prefill-swa-loop-bound.patch` | sglang `python/sglang/kernels/ops/attention/extend_attention.py` | window 128: **2531.6 → 378.1 µs (6.70×)**; window −1: 2638.2 → 2641.1 µs, unchanged | **+7.62%** (3953.246 → 4254.523), 3 interleaved restart pairs | pristine base |
| 02 | `artifacts/02-prefill-extend-launch-config.patch` | same file | full-attn **2641.1 → 1254.5 µs (2.11×)**, windowed **372.0 → 188.5 µs (1.97×)**, per 36-layer pass **54.24 → 25.97 ms (2.088×)** | **+6.30%** marginal (4254.523 → 4522.506) | patch 01 alone |
| **01+02** | both | same file | — | **+14.30%** (3956.689 → 4522.506), 3 interleaved restart pairs, arms disjoint | pristine base |
| 03 | `artifacts/03-decode-attn-segments.patch` | **aiter** `aiter/ops/triton/attention/unified_attention.py` | bs 64: **108.60 → 100.44 µs (1.081×)**; bs 32: 58.08 → 54.32 µs (1.069×); sweep best 108.32/108.44 → **99.62 µs** (1.087× per the patch header, 1.089× in FINDINGS §4) | **+0.750%** (4520.931 → 4554.813), 14 clean pairs of 16 order-balanced restart pairs | **01+02**, not the bare baseline |
| **01+02+03** | all three | two repos | — | **+15.66%** (3938.109 → 4554.813) | local 5-round baseline |

**Patches 01 and 02 were measured together as +14.30%** and that is the number to trust for the
prefill work; the +7.62% / +6.30% split comes from two separate A/B campaigns and is reported so a
reader can decide whether one patch alone is worth deploying. They touch disjoint hunks of the same
file, apply in either order (`git apply --check` verified both ways), and each is a win alone.

**Patch 03's +0.750% is the marginal gain over 01+02**, and it needed 32 server restarts to
establish where the prefill patches needed 6. Its full statistical record:

| estimator | value |
| --- | --- |
| paired delta, 14 clean pairs | **+0.750%** (paired stdev 0.397%, **t = 7.08**) |
| difference of arm means, all 16 pairs | +0.818% |
| difference of medians, all 16 pairs | +0.748% |
| 10% trimmed mean, all 16 pairs | +0.765% |
| paired mean, all 16 pairs | +0.829% |
| **sign test, all 16 pairs** | **15/16 favour candidate, two-sided p = 0.00052** |
| order effect | base-first pairs +0.780%, candidate-first pairs +0.721% — **absent** |

Four independent estimators agree to within 0.08 percentage points and the effect clears the 0.501%
restart floor by 1.5×. The mechanism corroborates from three directions, which is what makes a
sub-1% claim believable: **TPOT falls 0.75% while TTFT does not move**, the exact signature of a
decode-only change (both prefill patches did the reverse); the measured +0.750% sits just under the
≈1.0% the kernel benchmark predicted, as it should, because not every decode step runs at the bs=64
the guard is tuned for; and the effect is present in all four batches in the same direction.

The order-balancing deserves its own note. Rule 6b prescribes interleaving but does not name the
residual confound that survives it: if the base arm is always the earlier of the two runs in a pair,
any drift across the sequence accrues to the candidate. This run built `ORDER=rev` into
`artifacts/ab_restart.sh` and ran two batches each way to *measure* the confound rather than assume
it away. It is not there — 0.06 pp between the two orderings. That is worth one control run, not a
standing worry.

## The transferable pattern: tuned constants stranded behind an architecture predicate

This is the part of the entry most likely to be useful on a model that is not gpt-oss-120b.

**The shape of the defect.** A kernel ships with two things: a set of hand-tuned launch constants
that someone measured on this architecture, and a predicate deciding who gets them. The constants
are correct. The predicate is narrower than the argument that justified them. Everything outside it
silently takes a generic default that nobody tuned, and *nothing logs anything* — the server is
fast, correct, and leaving a factor of two on the floor.

Here it was `_get_block_sizes_for_extend_attention`, which already carried a gfx950 tuning —
`BLOCK_M, BLOCK_N = (128, 64)`, `num_warps = 8`, with an in-tree comment recording −36% kernel time
and 28% → 44% MFU on MI350X — gated on `128 < Lq <= 256`. gpt-oss-120b runs `head_dim` **64** and
fell through to the generic AMD default `(64, 64)` / 4 warps. The justification in the comment is
about KV bytes re-streamed per workgroup, which does not depend on head dim at all, and the measured
traffic said so louder at 64 than at 256: **8.66 GB of K/V reads against 8.4 MB of unique KV**,
sustaining 3.28 TB/s against the ~4.6 TB/s this machine can actually copy, at 208 TFLOPS ≈ 8% of
bf16 peak. Widening the predicate to `Lq <= 64 or 128 < Lq <= 256` was worth **2.088× on a prefill
pass and +6.30% end to end.**

**Why the finding is more than a local fact — three independent sightings, one of them in this
directory.**

- `../gemma-4-26b-a4b-it/artifacts/003_gfx95_extend_attn_launch.patch` diagnoses the *same predicate in
  the same function*, independently, on a different model: "The HIP branch … has exactly one tuned
  case — `128 < Lq <= 256` → (128, 64, 8 warps) — and sends everything else to a generic (64, 64, 4
  warps)." It was worth **+7.20%** there, at head dims 256 and 512. Two campaigns, two different
  head dims, same one-case branch.
- The Qwen3.5-397B-A17B run
  (`tuning_workspace/experiment_standalone/hold_qwen35_397b_a17b_mxfp4_tuning/`) hit the same class
  from the other side: its headline finding is "AMD tuned this model's dense GEMMs for TP=2, and the
  frozen config is TP=4", plus a `gdn_chunk_h_launch_config.patch` for launch geometry. Same
  failure mode — tuned constants keyed on a tuple the deployment does not match — with a topology
  key instead of an arch key.
- Kimi-K3's `03-kda-packed-decode-warps16.diff`
  (`tuning_workspace/experiment_standalone/kimi_k3_tuning/`) is the untuned-launch-geometry half of
  the same idea: 8 warps → 16 in a shipped decode kernel, 1.095× isolated, bitwise identical output.

*(The Gemma numbers above are from this knowledge base and are checkable here. The Qwen3.5 and
Kimi-K3 figures are quoted from their own bundles and have not been re-verified as part of this
entry.)*

**The companion defect, in the same file: a kernel that is told a constraint and only uses it as a
mask.** Patch 01 is that. `_fwd_kernel` receives `sliding_window_size` and applies it — as a mask,
with a `SKIP_TILE` guard that avoids the loads and the two dots for a fully dead tile. But the
stage-2 loop bound was still the causal diagonal, so at window 128 with extend 8192 the kernel
*visits* about 64 dead tiles for every live one and reduces each of their masks. The guard was worth
4%; starting the loop at the first tile that can be live is worth **6.70×**. And
`../gemma-4-26b-a4b-it/artifacts/004_swa_window_loop_bound.patch` is the same fix to the same loop in
the same file, found independently, worth +1.18% there — the difference in end-to-end value between
the two entries is Amdahl's share, not a difference in the defect.

**The detection method, which is the actually reusable part.** Both wins came from one question
asked of a stage-split profile: *why do two things that should differ cost the same?* The decode
path knew the difference — windowed layers 8.24 µs against full-attention layers 95.01 µs, 11× — and
the prefill path showed no bimodality at all, all 36 layers at the same ~1440 µs per 1-sequence
chunk. That flat profile is the signature. Run the profile with `profile_by_stage: true`, rank
device kernels by summed duration per kernel name, and look for **groups of layers that are
architecturally different and empirically identical**.

**The one caveat, which is load-bearing.** This finding was passed to the Llama-3.1-8B run
(`tuning_workspace/experiment_standalone/llama_31_8b_tuning/`) as a lead pointing at the same
predicate at `head_dim` 128, and its FINDINGS records that **the lead did not literally apply**: the
predicate is real and Llama would take the fallback, but under SGLang's aiter prefill backend
`extend_attention.py` is not on Llama's prefill path at all. That run still found a prefill win
(+3.185%) by a different mechanism. The lesson is precise: **this pattern tells you which phase to
open, not which line to edit.** Confirm the file is on the path before sizing a result on it — the
"backend" rows in the fingerprint table above are marked load-bearing for exactly this reason.

## Where the time actually goes

Recorded here because the bundle's own headline profile numbers are wrong in both directions, and a
reader who trusts them will pick the wrong target.

`BASELINE.md` reports "97.35% of decode is exposed communication" and offers ~11% as its own
corrected value. Both are artifacts of a single `cross_device_reduce_1stage` of 14,220,915.6 µs
(14.2 s) on TP-1, sitting ~0.9 ms into the capture window — rank skew, not communication. (That is
the event `BASELINE.md` describes as 9063.5 ms; the two figures disagree, and `FINDINGS.md` asserts
they are the same event. It changes nothing downstream — either way it is one skew outlier and it is
dropped.) Dropping that one
event (`artifacts/steady_state.py`) and recomputing over the 32 decode steps in the window:

**Post-skew steady state, TP-1, 32 decode steps: span 273.56 ms, busy 271.29 ms, exposed gap 0.83%,
653 kernels/step, 8549 µs span and 8478 µs busy per step.** The GPU is **99.2% busy** during decode.
This is not a bubble problem and not a communication-exposure problem. It is a kernel-time problem,
and the only levers are faster kernels or fewer of them.

| µs/step | % busy | calls/step | µs each | kernel |
| --: | --: | --: | --: | --- |
| 1800.1 | 21.23 | 18 | 100.01 | `kernel_unified_attention_3d … TILE_SIZE_64_HEAD_SIZE_64_NUM_SEGMENTS_PER_SEQ_16` (full-attn layers) — **patch 03's target** |
| 1768.4 | 20.86 | 36 | 49.12 | MoE ck_tile MXFP4 gemm1 |
| 1019.0 | 12.02 | 36 | 28.31 | MoE ck_tile MXFP4 gemm2 |
| 942.4 | 11.12 | 73 | 12.92 | `cross_device_reduce_1stage` (p50 12.24 µs) |
| 355.0 | 4.19 | 36 | 9.86 | QKV `hgemm_bf16_32x64x64x7_SPK3` (M64 N2560 K2880) |
| 353.0 | 4.16 | 73 | 4.84 | `direct_copy_kernel_cuda` bf16 — origin unresolved, see below |
| 345.1 | 4.07 | 36 | 9.59 | o_proj `Cijk_…MT32x32x512` (M64 N2880 K2048) |
| 300.1 | 3.54 | 72 | 4.17 | `aiter::add_rmsnorm_quant_kernel` |
| 209.2 | 2.47 | 1 | 209.23 | `aiter::allgather_vec` |
| 149.6 | 1.77 | 18 | 8.31 | `kernel_unified_attention_2d` (SWA layers) |
| 118.5 | 1.40 | 18 | 6.59 | `reduce_segments_…NUM_SEGMENTS_PER_SEQ_16` |
| 101.8 | 1.20 | 1 | 101.82 | lm_head GEMM (M64 N100544 K2880) |

Grouped: **attention 22.6%** (3d + 2d + reduce_segments), **MoE 32.9%** (gemm1 + gemm2),
**all-reduce and its companion copy 15.3%**. Roughly 400 of the 653 kernels per step sit at a ~4 µs
dispatch floor and are worth about 1.4 ms/step (≈17%) in aggregate — a launch-count problem no
single kernel rewrite touches, and the largest structural item left.

Prefill, before any patch: the extend kernel was **39.87% of prefill busy time** (258 calls ×
2560.65 µs), and prefill is roughly 47% of wall clock. That ratio is what made prefill the right
first target, and it is why patch 01's +7.62% was predictable in advance: 0.53 + 0.47 × 0.834 =
0.932 → +7.3% predicted against +7.62% measured.

**The exchange rate for decode work on this stack is 8% of the largest decode kernel → 0.75% of the
run.** Know that before spending a day there.

## Deploy

Two repositories. Both are git checkouts in this image, so `git apply` works and `git checkout --`
is a complete revert.

```bash
# 1. patches 01 and 02 -> the SGLang tree
cd /sgl-workspace/sglang
git checkout -- python/sglang/kernels/ops/attention/extend_attention.py     # start clean
git apply --whitespace=nowarn <kb>/artifacts/01-prefill-swa-loop-bound.patch
git apply --whitespace=nowarn <kb>/artifacts/02-prefill-extend-launch-config.patch
git diff --stat -- python/sglang/kernels/ops/attention/extend_attention.py
#   expect: 1 file changed, 49 insertions(+), 7 deletions(-)

# 2. patch 03 -> the aiter tree (NOT the sglang tree)
cd /sgl-workspace/aiter
git checkout -- aiter/ops/triton/attention/unified_attention.py
git apply --whitespace=nowarn <kb>/artifacts/03-decode-attn-segments.patch
git diff --stat -- aiter/ops/triton/attention/unified_attention.py
#   expect: 1 file changed, 25 insertions(+)

# 3. invalidate every derived cache before restarting
rm -rf ~/.triton/cache
rm -rf /tmp/aiter_configs
find /sgl-workspace/sglang/python/sglang/kernels/ops/attention -name '__pycache__' -prune -exec rm -rf {} +
find /sgl-workspace/aiter/aiter/ops/triton/attention   -name '__pycache__' -prune -exec rm -rf {} +

# 4. RESTART. Not optional.
cd /work && ./scripts/launch_server.sh --stop && ./scripts/launch_server.sh
#   must print: [server] config verified
```

Per-patch line counts if you deploy a subset: 01 alone is 21 insertions / 1 deletion, 02 alone is 28
/ 6, 03 is 25 / 0.

`/tmp/aiter_configs` is listed for completeness rather than necessity: this entry ships no aiter CSV,
so nothing merges into it. It is in the list because aiter's merge is derived and **is not
regenerated if it already exists**, which makes it the standing hazard on any aiter stack — and
because a reader deploying this alongside another entry's CSV will need it. Clearing it costs
nothing.

### Every way this deploy silently does nothing

Each of these produces a healthy server, a correct model, and a benchmark number that looks exactly
like the arm you thought you left behind.

1. **Applying anything to a running server.** Decode replays a HIP graph captured at startup and
   Triton compiles at first call. A live drop-in benchmarks *perfectly* like the previous arm. This
   is the single most expensive trap on this stack because it yields a clean, plausible, wrong
   number.
2. **Applying all three patches in one tree.** Patch 03 is the only one in the aiter repo, and it is
   easy to loop over `artifacts/*.patch` from `/sgl-workspace/sglang` and deploy two of three. That
   particular mistake fails loudly; the quiet version is deploying 01+02 and *believing* you
   deployed the full stack, which reads as "patch 03 was worth nothing". Each patch declares its own
   `# Repo:` header line so that `artifacts/ab_restart.sh` can route it, and that is the mechanism to
   copy if you automate this.
3. **A stale `~/.triton/cache`.** Both files are Triton kernels. A cached compilation keyed on
   unchanged metadata serves the old kernel.
4. **A stale `__pycache__`.** Patch 02's first hunk and all of patch 03 are ordinary Python executed
   at launch config selection time; a stale `.pyc` reverts them without touching the `.py`.
5. **`--prefill-attention-backend` not `triton`.** Patches 01 and 02 are then not on the path at
   all. This is the failure mode the Llama-3.1-8B run actually hit with the same lead, and it fails
   with no error and no log line.
6. **`--decode-attention-backend` not `aiter`.** Same for patch 03.
7. **`--page-size` not 64.** Patch 03 requires `TILE_SIZE == 64`; a different page size moves the
   tile and the guard declines. It also costs you the ~2.9× configuration win underneath the
   baseline, so you will notice — but for the wrong reason.
8. **`head_dim` ≠ 64, KV dtype ≠ bf16, arch ≠ gfx950, or `shuffled_kv_cache` true.** Each is a
   literal term in patch 02's or patch 03's guard. All decline silently by design; that is what
   makes the patches safe to carry on other models, and what makes them worthless there.
9. **Decode batch size below the saturation threshold.** Patch 03's `num_2d_prgms * 4 >= 2 *
   get_num_sms()` term means bs 8 keeps the shipped default. Verified: bs 64 and bs 32 take the
   tuning, bs 8 does not. At concurrency well under 64 the patch is a no-op *and that is correct* —
   small batches need the extra splits to fill the GPU.
10. **Benchmarking the first round after a restart, or using a fixed round window.** Throughput
    climbs +0.591% over the first rounds within a process. Either take exactly one round per fresh
    server, as `artifacts/ab_restart.sh` does, or run until the last three are flat.

## Engagement check

**The strongest check is kernel identity in a stage-split profile**, because patch 03 encodes its
entire effect in the Triton kernel's specialized name. Capture with `profile_by_stage: true` and
rank kernels with `artifacts/rank_kernels.py`:

```bash
python3 artifacts/rank_kernels.py <capture>-TP-0-DECODE.trace.json.gz | grep -E 'unified_attention_3d|reduce_segments'
```

**Not engaged** — these are the literal kernel names in this run's own baseline capture
(`analysis/prof_stage/1787221409.378875-TP-0-DECODE.trace.json.gz`):

```
kernel_unified_attention_3d_num_query_heads_32_num_queries_per_kv_8_BLOCK_SIZE_64_TILE_SIZE_64_HEAD_SIZE_64_NUM_SEGMENTS_PER_SEQ_16_num_warps_2_waves_per_eu_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_0_IS_Q_FP8_0_IS_KV_FP8_0
reduce_segments_num_query_heads_32_TILE_SIZE_64_HEAD_SIZE_64_NUM_SEGMENTS_PER_SEQ_16
```

**Engaged** — the same names with `NUM_SEGMENTS_PER_SEQ_4`, `num_warps_4`, `num_stages_3`, and
`reduce_segments_…NUM_SEGMENTS_PER_SEQ_4`. Every other component of the name must be unchanged; if
`TILE_SIZE` or `HEAD_SIZE` moved, you are not on the shape this entry tuned.

> **Gap, stated rather than glossed:** the engaged string above is *derived*, from the patch's own
> assignments (`num_segments = 4`, `attn_warps = 4`, `attn_stages = 3`, confirmed by the `TUNED`
> dict in `artifacts/check_decode_cfg_bs.py`) and from the naming scheme visible in the baseline
> capture. **No profile of the patched server was captured in this run.** The baseline half of this
> check is observed; the engaged half is predicted. Capture one on first deploy and the check
> becomes fully two-sided.

**No environment variable is needed for any of this**, which matters because the configuration is
frozen and adding `AITER_LOG_TUNED_CONFIG=1` to a run you intend to publish is not allowed.

Two flag-free checks that need no profile at all:

**Structural, exact, and two-sided** — the line counts above:

```bash
cd /sgl-workspace/sglang && git diff --numstat -- python/sglang/kernels/ops/attention/extend_attention.py
#   engaged (01+02): 49  7   |  01 only: 21  1  |  02 only: 28  6  |  not engaged: no output
cd /sgl-workspace/aiter  && git diff --numstat -- aiter/ops/triton/attention/unified_attention.py
#   engaged: 25  0   |   not engaged: no output
```

This catches a failed apply and a reverted tree. It does **not** catch a stale `__pycache__` or a
live drop-in, so it is necessary and not sufficient — pair it with the behavioural check.

**Behavioural, and the one that cannot be faked by bookkeeping** — read the benchmark client's own
mean TTFT, which moves far outside any noise floor when the prefill patches engage:

| arm | mean TTFT | mean TPOT |
| --- | --: | --: |
| not engaged (pristine) | **4164.1 ms** | 12.182 ms |
| 01 engaged | 3554.4 ms | 11.572 ms |
| 01+02 engaged | **3090.6 ms** | 11.132 ms |
| 01+02+03 engaged | ~3096 ms | **11.037 ms** |

A TTFT near 4160 ms means the prefill patches are not live no matter what `git diff` says. And the
last row is the check for patch 03 specifically: **TPOT falls while TTFT holds flat**. If TTFT moved
when you added 03, something other than patch 03 changed.

## Accuracy gate

gsm8k 5-shot, **lm-eval 0.4.12** (`[api]` extra, in its own venv at `/tmp/lmeval_venv`), task
version 3.0 from the bundle's `eval/gsm8k.yaml`, `local-chat-completions` against
`/v1/chat/completions`, `--apply_chat_template`, `fewshot_as_multiturn=True`, `num_concurrent=64`,
`temperature=0`, `top_p=1`, `max_tokens=9216`, `max_length=11264`, seeds `0,1234,1234,1234`, 1319
problems (effective 1319). Reproduce with `artifacts/run_eval.sh`; the per-arm drivers are
`artifacts/gate01.sh`, `gate12.sh` and `gate123.sh`.

| arm | `exact_match,strict-match` | flexible-extract | vs baseline |
| --- | --- | --- | --- |
| **gate (this run's own baseline)** | **0.962851 ± 0.005210** | 0.962851 ± 0.005210 | — |
| source session, stock configuration | 0.962092 ± 0.005260 | 0.961334 ± 0.005311 | −0.15σ |
| patch 01 | **0.962851 ± 0.005210** | 0.962092 | ±0 problems |
| patches 01 + 02 | **0.965125 ± 0.005053** | 0.965125 | +3 problems |
| **patches 01 + 02 + 03** | **0.962851 ± 0.005210** | 0.962092 | ±0 problems |

**Threshold: strict-match ≥ 0.95764**, i.e. baseline − 1σ. All three arms pass.

**The baseline arm had never been evaluated** (`RUN_EVAL: false` in the reference session), so this
run's own baseline measurement *is* the gate — the correct construction, and the reason it is
trustworthy is the control: it lands 0.000759 above the source session's stock-configuration figure,
0.15σ, indistinguishable. Since the only difference between those two arms is a paging parameter,
that agreement is what says the stack is not suspect.

**Why the full stack coming back numerically identical to the baseline is expected, and why it is
not the whole story.** Patch 01 is a pure loop-bound tightening: the tiles it skips are exactly the
tiles `SKIP_TILE` already discarded without touching `e_max`, `deno` or `acc`, and the patch header
carries the proof that the tile immediately before `extend_start` is provably dead. It was verified
rather than argued — output **bitwise identical** at both window settings (`torch.equal` → True, max
abs diff 0.0), checked against the genuine unpatched kernel via `git stash` rather than against a
reimplementation. A bitwise-identical kernel *should* reproduce the baseline score exactly, and it
does.

Patch 02 is different: `BLOCK_M` and `num_stages` change accumulation order, so its score is not
expected to be bitwise stable, and the 01+02 arm lands 0.44σ **above** the baseline. Three problems
out of 1319 is exactly what an accumulation-order change looks like. **It is reported as noise in
the direction that happens to be favourable, not as an accuracy gain.** Patch 03 changes the decode
segment count, which changes the order of the `reduce_segments` combine, so it too is not expected
to be bitwise identical — that the full stack lands back on 0.962851 is **fortunate rather than
structural**, and the gate is the ≥ 0.95764 threshold either way. Do not read the identical numbers
as proof of numerical equivalence for the stack; that claim holds for patch 01 alone.

## What was tried and did not work

| attempt | kernel / op level | end to end | verdict |
| --- | --- | --- | --- |
| `direct_copy_kernel_cuda` is the all-reduce staging memcpy — remove the copy, remove the cost | 353 µs/step, **4.16% of decode**, 73 calls/step matching `cross_device_reduce_1stage` exactly (73 = 36×2 + 1) | not measured | **Refuted by source before spending a restart.** `custom_all_reduce.py:316` selects `registered = not self.tms_cudagraph`, and `tms_cudagraph` is `SGLANG_MEMORY_SAVER_CUDA_GRAPH`, default `False` and unset here — so capture already takes `registered=True` → `ops.all_reduce_reg`, which contains no `hipMemcpyAsync`. **The optimisation is already on.** Two other origins ruled out: the `gpt_oss.py:290/:344` contiguous trims are gated behind `SGLANG_AITER_FUSE_RMSNORM_PAD` (default False) *and* hard-disabled for `tp_size != 1`; the `_all_reduce_out_place` clone paths are only reached by `pymscclpp` and no-communicator branches. Origin still unresolved — the best remaining lead in this bundle. |
| prefill all-reduce: swap the algorithm | at 90 MB: rccl 1654.2 µs, **qr 917.2 µs**, ca (max_size→128 MB) 1548.7 µs | not measured | **Nothing to win, three ways.** SGLang already picks the winner at both sizes this workload produces: prefill batches are 8192/16384 tokens (45.00/90.00 MB), above the 16 MB `ca` cutoff, routed to `qr`, fastest there by 1.75–1.80×; decode is 64 tokens / 0.35 MB, routed to `ca`, fastest there by 1.48×. The sizes where `ca` is picked but `qr` would win (512 and 2048 tokens) are shapes this workload never generates. |
| raise `_MAX_CAR_SIZE` to 128 MB so `ca` competes at 90 MB | `ca` becomes available and is **1.69× slower** than what is already chosen (1548.7 vs 917.2 µs) | not measured | The in-tree comment "crossover is at 16MB buffer size for ROCm" is **still true on MI355X**. The cutoff is not stale. |
| beat quickreduce at all | 90 MB in 917.2 µs = **98 GB/s per direction, 196 GB/s across the link**, against a 2-rank algorithmic minimum of N bytes per direction | — | **At the algorithmic bound, not merely best available.** No algorithm left to swap in; the only lever is raw fabric speed. Standalone 917.2 µs corroborates the profile's 948 µs to within 3%, which also validates the attribution. |
| quantized (INT8/INT6/INT4) all-reduce | would roughly halve the 917 µs, ≈7% of wall — **the largest single number left in this system** | not run | **Available and declined, with the reason.** It is selected by an environment variable, so flipping it in code is materially the same act as flipping the frozen setting, and unlike everything else here it changes the *arithmetic* of every all-reduce rather than only the schedule. The source session's measurements point the same way: page 64 + INT6 measured **4080.414 tok/s, −3.36%**, page 64 + INT8 **−5.98%**, page 64 + all-reduce fusion **−1.37%**. Every quick-reduce arm retested on top of page size 64 lost. |
| decode attention `TILE_SIZE` 128 / 256 | in the 318-config sweep; reachable on gfx950 via the kernel's generic gather branch even though `select_3d_config:196` clamps to `min(64, next_pow2(block_size))` | — | **No help.** Worth chasing on paper, measured flat. Full ranking in `artifacts/unified_attn_sweep_tile.json`. |
| extend kernel `BLOCK_M = 256` | in the 103-config sweep | — | **Lost to LDS pressure.** The remaining 2× under the roofline needs a kernel change (two-level tiling keeping the 8.4 MB unique K/V resident across M-blocks), not another config. Ranking in `artifacts/extend_cfg_sweep.json`. |
| CPU→GPU attribution through `ac2g` flow events, so each device kernel could be tied to a Python frame | 3328 `cpu_op` records against **20,928 device kernels**; no flow event reaches any kernel replayed inside the HIP graph — 2336 `direct_copy` dispatches carry `"kind": "Dispatch Task"` with no `grid` and no `External id`, while the few running outside the graph carry `"kind": "Dispatch Kernel"` with both | — | **Cannot work on this stack**, and it cost real time. Attribution must come from reading source, or from a capture taken with graph replay disabled. |

Two more negatives inherited from the source session, settled and not worth re-testing: two
framework-stage patches measured **+0.28%** and **+0.53%** and were reverted under a 1% keep
threshold, and the code-level stage ran 7h14m and **attempted nothing at all** because graph replay
hid every kernel from the profiler. That last one is not a negative result about the model; it is
what `artifacts/rank_kernels.py` and `artifacts/steady_state.py` exist to fix.

**Nothing in this run reached patch form and then lost.** The three ideas that got as far as a
working diff all held up under A/B, which is why there is no `rejected/` directory of measured-and-
discarded patches — the losing ideas died at hypothesis or at kernel-sweep stage, and inventing
diffs after the fact to fill a directory would misrepresent what was measured.

## When this entry stops applying

Every one of these fails **silently** — the guard declines, the server serves correctly at the old
speed, and nothing is logged:

- **Arch ≠ gfx950.** Patch 02 gates on `_is_gfx95`, patch 03 on `DEVICE_ARCH == "gfx950"`.
- **`head_dim` ≠ 64.** Patch 02's widened predicate is `Lq <= 64 or 128 < Lq <= 256`; head dims in
  `64 < Lq <= 128` are *deliberately* left on the default because they were not measured here. Patch
  03 requires `head_size <= 64`.
- **KV cache dtype ≠ bf16**, or **`shuffled_kv_cache` true**, or **`TILE_SIZE` ≠ 64** — three more
  literal terms in patch 03's guard.
- **`--page-size` ≠ 64**, which moves `TILE_SIZE`.
- **A model with no sliding-window layers.** Patch 01 does nothing when `SLIDING_WINDOW_SIZE <= 0` —
  verified, 2638.2 → 2641.1 µs, unchanged. On this model that means it is worth nothing on 18 of 36
  layers by construction.
- **Prefill backend ≠ triton, or decode backend ≠ aiter.** The patched file is then not on the path.
- **Decode batch size well below 64**, which drops patch 03 below its saturation guard.
- **A different SGLang or aiter commit.** These are context diffs; `git apply` will at least fail
  loudly, which is the one failure in this list that announces itself.
- **Concurrency, ISL, OSL or `--chunked-prefill-size` changed.** The tuned launch geometry was scored
  on the `bs 2 × extend 8192` prefill shape and the bs-64 decode shape those parameters produce.

What stays reusable when they do occur, and it is most of the entry: **the detection method** (split
the profile by stage, rank device kernels by summed duration, look for architecturally-different
layer groups that cost the same); **the pattern** — tuned constants stranded behind a narrow
architecture predicate, now sighted in three independent campaigns; **the measurement protocol**
(one bench round per fresh server, order-balanced pairs, sign test when the tail makes disjointness
impossible); and the two sweep rankings under `artifacts/`, which record every config that lost with
its timing rather than merely asserting that it lost.

## Artifacts

| file | what it is |
| --- | --- |
| `artifacts/01-prefill-swa-loop-bound.patch` | **the win**, part 1 — sglang, extend-attention loop bound |
| `artifacts/02-prefill-extend-launch-config.patch` | **the win**, part 2 — sglang, gfx950 launch config for head_dim 64 |
| `artifacts/03-decode-attn-segments.patch` | **the win**, part 3 — aiter, gfx950 decode segment count |
| `artifacts/launch_server.sh` | the exact launch configuration, with the live-config verification gate |
| `artifacts/run_bench.sh` | the frozen workload — the measurement contract |
| `artifacts/run_eval.sh` | the gsm8k gate, including the three settings that separate a real accuracy number from a meaningless one on a model that reasons before answering |
| `artifacts/preflight.sh` | stack assertion — fails on what the source session *did* record (sglang version, ROCm, arch, device count, shard count) and prints what it did not (aiter sha, sglang sha, torch, triton, host). Those printed lines are where this entry's two commit shas come from |
| `artifacts/start_container.sh` | container launch, including the render-node mapping trap on these hosts |
| `artifacts/ab_restart.sh` | the A/B harness: one bench round per fresh server, per-repo patch apply/revert via each patch's `# Repo:` header, `ORDER=fwd\|rev` for order balancing |
| `artifacts/gate01.sh`, `gate12.sh`, `gate123.sh`, `gate_then_sweep.sh` | per-arm accuracy-gate drivers (note the scoped `PATH` for the lm-eval venv — `launch_server.sh` must not see it) |
| `artifacts/bench_extend_swa.py` | the kernel harness that found patch 01, at the exact served shape |
| `artifacts/bench_extend_cfg.py` | the 103-config extend launch sweep behind patch 02 |
| `artifacts/bench_unified_attn.py` | the 318-config decode sweep behind patch 03 |
| `artifacts/check_decode_cfg_bs.py` | patch 03's boundary check at bs 64 / 32, run against the `git stash`ed pristine function |
| `artifacts/bench_allreduce.py`, `bench_p2p.py` | the all-reduce negative result, and the misleading DMA measurement that preceded it |
| `artifacts/rank_kernels.py`, `steady_state.py` | the two tools that made this stack readable at all |
| `artifacts/extend_cfg_sweep.json`, `unified_attn_sweep.json`, `unified_attn_sweep_tile.json` | full rankings — every config that lost, with its timing |

## What the bundle did not record

Listed because the template asks for them and a missing value is better than a guessed one:

- **Container image digest.** No `sha256:` anywhere. Only the launcher's default tag, and no way to
  confirm it from inside. This is the one field a reader genuinely cannot recover.
- **The server process environment.** `cat /proc/<pid>/environ` was never captured, so "the launch
  script exports only `SGLANG_USE_AITER=1`" is a statement about the script, not the process.
  Harmless for the deltas here — both arms shared it — but it is exactly the gap that cost the
  Mixtral run a whole attempt.
- **A profile of the patched server.** The engaged half of the kernel-identity check is derived
  rather than observed.
- **Any reproduction on a second host or a fresh container.**

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/gpt_oss_120b_sglang_tuning/`.

`EXPERIMENT_COMPLETE` carries the one-line claim. `FINDINGS.md` is the full report: local baseline
and both noise floors at the top, accuracy gate next, the decode kernel budget under "Where the time
actually goes", then §1 the refuted staging-copy hypothesis, §2 patch 01, §3 patch 02, §4 patch 03
and its 16-pair campaign, §5 the all-reduce negative. `BASELINE.md` documents the 4201.601 reference
figure, its six reference rounds, and why the baseline carries no `ROCM_QUICK_REDUCE_QUANTIZATION`.
Each patch header in `patches/` carries its own base sha, apply command, kernel-level and end-to-end
measurements, and stacking relationship. `patches/rejected/README.md` records what was discarded and
at what stage it died. Raw per-run throughput for all 52 benchmark rounds is in `results/`; the four
accuracy runs are in `eval_results/`; the stage-split profile this entry's kernel names come from is
`analysis/prof_stage/`.
