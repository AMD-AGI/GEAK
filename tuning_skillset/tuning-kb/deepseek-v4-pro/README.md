# DeepSeek-V4-Pro on MI355X — vLLM, TP=8, FP8 block-scaled GEMM tuning

**Verified win: +5.8% output token throughput** (987.67 → 1046.01 tok/s on the arms' per-instance
plateaus, +5.91%; **+5.77%** on the matched b5–b7 window, which is the figure the run headlines),
bounded **+5.5% to +6.3%** across every reasonable choice of window and instance pairing. gsm8k
0.9575 ± 0.0056 strict-match against a 0.9606 ± 0.0054 reference — 0.57σ, **pass**. The win is
carried by one file: 56 rows of aiter config data, no code change, no rebuild, no new environment
variables.

Measured against a **restart-to-restart noise floor of 0.148%** on the host it was measured on,
making the effect **~40× the floor**. The arms' distributions are disjoint over all 37 benchmark
runs: the best single baseline run anywhere (996.66) is below the worst single tuned run anywhere
(1006.63).

**Reproduction status, stated precisely.** The exported artifact
(`artifacts/a8w8_blockscale_tuned_gemm_dsv4_pro.csv`, md5 `f83564daa3a74abebd6b1e33b20d98bc`) was
deployed **on a second host** and re-measured there from scratch: two fresh tuned server instances
interleaved against three fresh baseline instances, the drop-in physically added and removed between
instances, engagement re-verified on every one, and the baseline re-established on that host rather
than carried over. The deployed file was byte-identical to the exported one — same md5. What was
**not** done: an independent re-derivation of the CSV by anyone else, and a measurement on an
uncontended node (see the hazards section). The two hosts agree on effect size (+5.8–6.0%) while
disagreeing by ~17% on absolute level.

> **Read this entry for the noise floor before you read it for the result.** Not because this model
> is noisy — it is not, necessarily — but because this campaign is the directory's clearest
> demonstration that *the floor belongs to the machine*. See the next section; it is the most
> valuable thing here.

## The single most important finding: the noise floor is a property of the machine

The same patch, same model, same stack, same frozen flags, same workload, measured on two hosts:

| | round 1, `crsuse2-m2m-150` | round 2, container `e984b47b12aa` |
| --- | --- | --- |
| restart-to-restart spread, unmodified baselines | **6.4%** (1113.64 … 1184.84) | **0.148%** (987.11 … 988.57, three restarts spanning 1.46 tok/s) |
| within-instance spread | ~1% | 0.25–1.87% |
| baseline absolute level | 1113–1185 tok/s | ~987 tok/s |
| baseline mean TPOT | 46.50–47.89 ms (r1) | 56.04–56.51 ms |
| instance "birth class" bimodality | present and strong | **absent** |
| what the campaign could say | bound the effect: +5.8% to +7.6% | resolve it: +5.8%, bounded +5.5–6.3% |

Nothing about the change moved between those two columns. A result that was barely claimable became
solidly claimable because the machine underneath it was quieter.

**Explicit correction — a claim this entry previously made and now withdraws.** The earlier version
of this entry taught that DeepSeek-V4-Pro on this stack is a high-noise model with a 5.5–6.4%
restart floor, "twenty times Gemma's", and repeated round 1's §1.4 conclusion that *effects smaller
than about 3% are not measurable on this model within a reasonable run*. **That is wrong as stated
and is withdrawn.** The bundle's own words: that conclusion "was true of `crsuse2-m2m-150` and is
false here", on a host whose floor is 0.148%. The lesson generalises in the opposite direction to
how it was written.

**The actionable rule:** re-derive the restart floor on the node you are actually on, every time,
from your own back-to-back baseline restarts. Never inherit a floor from a previous run, from a
sibling bundle, or from an earlier round of your own campaign. Round 1's *procedure* — measure the
floor before believing a delta — transferred perfectly; its *calibration* transferred not at all.

The two-host structure also means: **do not mix the hosts' numbers.** Every absolute figure below is
labelled with the host it came from. The headline comes from round 2.

## The baseline already contains a large win that is not this campaign's

Get this straight before comparing any number, because there are several figures in circulation and
most of them are wrong for most purposes.

| layer | tok/s | what it is |
| --- | --- | --- |
| stock vLLM, no MTP | **931.22** | before any config search |
| frozen baseline, MTP depth 1 | **1192.9** promoted (three-run mean 1184.1) | **+28.10% over stock, from a config change** — already banked, not re-discoverable |
| locally reproduced warm mean, round 1's host | **1187.20** | five warm runs on one server, −0.47% from promoted |
| this campaign's code patch | **+5.8%** | measured **on top of** the frozen MTP baseline |

The +28.10% came from adding MTP speculative decoding, and it needs two flags together, not one:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' --gpu-memory-utilization 0.85
```

At 0.95 there is no room for the draft head.

**Never quote a gain against 931.22, and never against the promoted 1192.9** (which is a best of
three, not a mean). Quote against a warm mean you measured yourself on the host you are on. The
earlier version of this entry said "quote against 1187.20"; that instruction is **narrowed**, not
deleted — 1187.20 is round 1's host's local warm mean and it is the right comparator *there*. On
round 2's host the baseline plateaued at ~987 tok/s, about 17% lower, so a round-2 tuned number
compared against 1187.20 would read as a large regression that does not exist. Same rule as the
noise floor: the reference level is a property of the machine.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 8× MI355X, `gfx950`, **256 CU** each, CDNA4 | **yes** — the CSV rows are keyed `(gfx950, 256, M, N, K)` |
| container | `vllm-openai-rocm:v0.26.0` @ `sha256:5709fafe47123becb2f5e61c32d0b97beff1a629bb40bb753c15464f69a97a18` | descriptive, pins the stack |
| vLLM | **0.26.0+rocm723** (banners say bare `0.26.0`) | descriptive |
| aiter | **0.1.16.post3**, wheel only — **no git metadata, no `csrc/`, no tuner** | **yes** — schema, merge path and kernel IDs; and its incompleteness shaped the whole method (see below) |
| torch / ROCm / Triton | 2.11.0+gitd0c8b1f / 7.2.3 / 3.6.0 | descriptive |
| model | DeepSeek-V4-Pro, `DeepseekV4ForCausalLM`, **TP=8** — 61 layers, 384 experts, MLA | **yes** — sharding sets every N and K |
| quantization | bf16+quark checkpoint resolving to **`deepseek_v4_fp8`** | **yes** — selects the `gemm_a8w8_blockscale` table |
| KV cache | `fp8` | **yes** |
| **MTP** | `{"method":"mtp","num_speculative_tokens":1}` | **yes** — it is what creates the `7168×7168` shapes, and it is the entire +28% config delta |
| hosts | round 1 `crsuse2-m2m-150`; round 2 container `e984b47b12aa` | **yes for absolute levels, and for the noise floor** — see above |

The stack was asserted identical on both hosts by `scripts/preflight.sh`; only the machine differed.

aiter's commit sha is **not recorded** — the wheel carries no git metadata, and the patch manifest
records `base_sha: null`, with a note that the base tree's identity is the wheel version plus the
container digest. That is a real gap for anyone trying to match this environment exactly.

## Launch configuration

```bash
export VLLM_ROCM_USE_AITER=1

vllm serve <DeepSeek-V4-Pro> \
  --host 0.0.0.0 --port 42323 \
  --tensor-parallel-size 8 \
  --max-model-len 13312 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

That one variable is the whole env recipe — with it unset you are measuring a different stack, since
aiter supplies the fused MoE, the FP8 GEMMs and the RMSNorm path on this model. The frozen contract
forbids adding others, which rules out `AITER_LOG_TUNED_CONFIG=1` and means engagement has to be
checked without it (see below).

**Startup cost is also a per-machine property — budget it, then measure it.** The reference server
logged its banner at 18:12:09 and accepted traffic at 18:44:16, **1927 s**, and the launch script's
health wait is 2700 s for that reason. Nothing in this campaign reproduced it: round 1's node was
ready in **462 s**, the local baseline reproduction in **450 s**, and round 2's `base n2r5` in
**389 s**. Round 1 measured one instance to protocol (five benchmarks) at **~26 minutes**, or ~31
with gsm8k — roughly two instances an hour. Why the reference took 32 minutes is **not established**:
a storage-bandwidth hypothesis was tested and did not reproduce, so treat it as unexplained rather
than as a known lever.

Round 1 drew from that arithmetic the conclusion that *effects below ~3% are not measurable here*.
**That conclusion is withdrawn as a general statement** (see the noise-floor section): it was a
consequence of that host's 6.4% floor, not of the model. What survives is the planning discipline —
decide what you are measuring before spending an instance, because instances cost half an hour.

The launch script verifies the resolved engine config out of the startup log and refuses to let you
benchmark on a mismatch: vLLM version, `tensor_parallel_size=8`, `max_seq_len=13312`,
`kv_cache_dtype=fp8`, `quantization=deepseek_v4_fp8`, `dtype=torch.bfloat16`, and — the one that
matters most — `speculative_config` must be a `SpeculativeConfig` with `method='mtp'` and one
speculative token, not `None`.

**Why that check earns its place:** MTP is the entire 931.2 → 1192.9 difference. If the draft head
fails to load, vLLM logs `speculative_config=None` and then serves perfectly correctly at roughly the
stock rate. That failure is indistinguishable from "my change did nothing" — or, worse, from a 22%
regression caused by your patch. Note also that the log's repr field is `num_spec_tokens`, not the CLI
spelling `num_speculative_tokens`. MTP engagement was verified from the script's own output on every
launch in both rounds, not assumed.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, seed 0, `random_range_ratio 1.0`, `--ignore-eos`,
InferenceX `benchmark_serving` fork. Unchanged across both rounds — it is part of the frozen
measurement contract.

**Warmups are 128 here, not 8.** Every other entry in this directory uses 8. The reference run used
128 and the frozen `run_bench.sh` keeps it, so a comparison against any other model's numbers is not
like-for-like, and reducing it to 8 changes what you measure.

What sets the tuned shapes: concurrency 64 gives decode **M=64**, hit on every token step; graph
capture walks non-power-of-two M buckets (8, 16, … 104, 112, 120, 128, … 2048), so there are hundreds
of distinct M per `(N,K)`; TP=8 shards the shared-expert N and replicates the fused QKV projection;
and **MTP depth 1 is what creates the 7168×7168 shapes at all.**

## Baseline and noise floor — two campaigns, two hosts

Both tables are reported in full. They are **not** merged, averaged or compared to each other in
absolute terms; only the *effect size* is comparable across them.

### Round 2 — the host the headline comes from (container `e984b47b12aa`)

Every run, nothing dropped. Execution order was base → tuned → base → tuned → base, so the baseline
was re-measured *between* the treated instances rather than taken once at the start.

| instance | drop-in | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | plateau (last 3) | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base n2r1 | no | 951.82 | 973.42 | 991.30 | 986.16 | 984.57 | | | | | **987.34** | 0.68% |
| **tuned n2r2** | **yes** | 1020.48 | 1033.22 | 1039.12 | 1049.98 | 1036.45 | 1048.43 | 1043.55 | | | **1042.81** | 1.15% |
| base n2r3 | no | 952.24 | 976.13 | 975.27 | 978.93 | 985.88 | 988.33 | 987.12 | | | **987.11** | 0.25% |
| **tuned n2r4** | **yes** | 1010.44 | 1029.03 | 1006.63 | 1032.27 | 1052.01 | 1052.86 | 1039.20 | 1049.57 | 1058.87 | **1049.22** | 1.87% |
| ~~base n2r5~~ | no | ~~959.87~~ | ~~977.56~~ | — | — | — | — | — | — | — | **VOID** — filesystem incident, see hazards | — |
| base n2r6 | no | 959.94 | 985.05 | 971.71 | 986.65 | 987.21 | 985.24 | 996.66 | 985.16 | 983.89 | **988.57** | 1.29% |

| arm | n | instances | mean plateau |
| --- | --- | --- | --- |
| baseline | 3 (21 runs) | 987.34, 987.11, 988.57 | **987.67** |
| tuned | 2 (16 runs) | 1042.81, 1049.22 | **1046.01** |

| floor, round 2 | spread |
| --- | --- |
| within one server process | 0.25–1.87% |
| **restart-to-restart, unmodified baselines** | **0.148%** — three independent restarts spanning **1.46 tok/s** |
| restart-to-restart, tuned arm | 0.612% (1042.81 … 1049.22, n=2) |

### Round 1 — superseded for the headline, kept for the method (`crsuse2-m2m-150`)

This arm was cut off when the allocation was lost; `base r8` never produced a single result.

| instance | drop-in | benchmark runs (tok/s) | last-3 mean |
| --- | --- | --- | --- |
| base r1 | no | 1160.86, 1182.66, 1179.97, 1191.89 | **1184.84** |
| base r2 | no | 963.63, 1071.85, 1125.75, 1122.60, 1113.43, 1104.89 | **1113.64** |
| **tuned r3** | yes | 1259.36, 1278.33, 1266.55, 1280.83, 1278.60 | **1275.33** |
| base r4 | no | 1130.22, 1125.35, 1154.08, 1176.30, 1149.83 | **1160.07** |
| **tuned r5** | yes | 973.50, 1071.76, 1154.03, 1159.75, 1202.71, 1195.52, 1218.70, 1192.62, 1190.07 | **1200.46** |
| base r6 | no | 957.44, 1108.59, 1101.44, 1088.53, 1067.20, 1138.12, 1142.71, 1111.37, 1109.82 | **1121.30** |
| **tuned r7** | yes | 1012.63, 1138.49, 1183.65, 1110.25, 1196.49, 1131.09, 1161.22, 1190.44, 1233.85, then 1219.80, 1247.53, 1263.12, 1223.38 | **1195.17** (first nine) |
| base r8 | no | — | *never completed — node lost* |

| floor, round 1 | spread |
| --- | --- |
| within one server process | **~1%** (r1 runs b2–b4 span 1.01%; the local reproduction spanned 1.10%) |
| **restart-to-restart, unmodified baselines** | **6.4%** (1113.64 … 1184.84) — **wider than the effect being measured** |
| restart-to-restart, tuned arm | **6.2%** (1200.46 … 1275.33) |

On that host the tuned arm scattered exactly as much as the baseline arm, which was the key
observation available at the time: **the scatter is a property of the instance, not of the change.**
Throttling, tuner-corrupted CSVs, node degradation between instances and KV-cache size differences
were all checked and ruled out as causes. Round 2 completed the thought — the scatter is a property
of the *machine*, and on a quiet one it nearly vanishes.

### Slow-born instances, the run-until-flat protocol, and where the taxonomy stops applying

On round 1's host, server instances fell into two classes that persisted for the life of the process:

- **Fast-born** (r1, r3): the first benchmark is already ~98% of the eventual plateau.
- **Slow-born** (r2, r5, r6, r7): the first benchmark is 86% of plateau or worse — r2 started at
  963.63 against a ~1120 plateau, about 14% low — and the instance climbs for several runs.

Whatever decides it is fixed at startup and is invisible from inside a running process. Candidates
consistent with everything measured, **none confirmed**: HIP-graph capture decisions taken once at
startup, the physical page layout the weights land in, and worker-to-XCD/NUMA placement.

**Five runs is not always enough.** Instance r5 was still climbing at its fifth benchmark (1202.71 was
its highest so far) and only flattened around runs 5–9 at ~1200. Applying a fixed b3–b5 window to it
would have reported **1172.16 and understated the instance by 2.4%** — larger than most wins anyone
is looking for. r7 had to be extended to thirteen runs, and its last three of thirteen (1244.68) sit
4% above its first nine (1195.17).

The practical rule this establishes, and it is the most portable thing in this entry: **discard the
first run, then keep benchmarking until the last three are flat within the within-instance scatter —
do not use a fixed window.**

**The birth-class taxonomy itself did not transfer to round 2's host, and that is worth knowing.**
There, baseline first runs land within **0.85%** of each other (951.82 / 952.24 / 959.94, and the
voided n2r5 opened at 959.87 on top of them) and tuned first runs within **1.00%** (1020.48 /
1010.44). The only thing predicting an instance's birth number is which arm it is in. There is no
fast-born cell to fill because there is no fast-born cell. Round 1 had to compare the arms *blocked*
by birth class; round 2's design reduces to a straight interleaved A/B, which is a **stronger**
comparison, not a weaker one. Same pattern as the floor: check whether the taxonomy exists on your
machine before designing around it.

## Measurement hazards on this stack

Three things went wrong or nearly wrong during the campaign that had nothing to do with the change
being measured. Check for all three before trusting a number.

**1. A co-tenant filled the shared filesystem and corrupted an entire arm.** The NFS volume backing
`/home/ethany` (`172.27.255.2:/volumes/b0a55a09-.../ethany`, 10 T) reached **100% full with 0 bytes
available** partway through baseline instance `base_n2r5`'s benchmark sequence. The run's own
footprint on that volume is ~74 MB, so this was another tenant. The failure mode is nastier than "the
job crashed":

| run | recorded | state |
| --- | --- | --- |
| b1 | 959.87 | executed and written cleanly, before the volume filled |
| b2 | 977.56 | executed and written cleanly |
| b3 | 992.48 **printed to stdout** | `inferencex_result.json` **never written** — `Disk quota exceeded` |
| b4 | 0.00 | ran into the full volume; harness reported zero throughput |
| b5–b9 | nothing | no output at all; the server was gone by the end of the sequence |

By the time space freed (35 G available again) the vLLM server was **DOWN** (`/health` unreachable),
so the instance could not be resumed. Note what this looks like if you are not paying attention: a
real-looking number on the terminal with no file behind it, then a **0.00** that would poison any
mean, then silence. The instance was quarantined rather than silently dropped — its directories carry
an `_invalid_base_n2r5_` prefix and `results/_invalid_base_n2r5_WHY.md` records exactly what happened.
Its two valid runs (959.87, 977.56) sit right on top of the other baselines' first two runs, so
nothing inconvenient is being hidden; they are excluded because a two-run prefix is not a plateau
under the protocol. `base_n2r6` is its replacement, run to the full nine benchmarks. **Check free
space on the volume your results are written to before an arm, and treat a `0.00` or a
stdout-without-a-file as an infrastructure event rather than a data point.**

**2. A foreign tenant on two of eight GPUs.** Round 2's host carried a non-Slurm process (host PID
269756, ~17.6 GiB, **not in the container's PID namespace**, so unstoppable from inside) holding
9.76 GB on GPUs 6 and 7 and running a steady **17–21% GFX load**. `preflight.sh` returned FAIL on
exactly that check and on nothing else. At TP=8 every layer carries a collective, so two slowed ranks
set the pace for all eight: absolute throughput on that host is ~17% below round 1's and mean TPOT is
56.3 ms against 46.9 ms in the bundle's reference. The A/B stays internally valid because both arms
are exposed equally and the arms are interleaved — and to make that checkable rather than assumed,
per-GPU utilisation and VRAM were sampled every 5 s for the whole round to
`/tmp/n2logs/gpu_timeline.csv`, with each instance recording the VRAM state it launched into. KV cache
size was verified unchanged across arms (91.69–91.71 GiB), so the tenant was not silently resizing the
cache under one arm. What this costs the claim is stated below under what is not claimed.

**3. Extending only one arm's instance biases the result.** Round 1 extended `tuned r7` to thirteen
runs because it had not flattened, and it kept climbing — the only instance in the campaign to get
thirteen runs, and it was in the tuned arm. Every round-1 figure therefore uses r7's **first nine
runs** so that r5/r6/r7 share an identical window. Round 2 hit the same shape — `tuned n2r4` was
still climbing at run 9 (1039.20 → 1049.57 → 1058.87) — and ran `base n2r6` to the full nine
benchmarks specifically to find out whether the late drift is a property of the stack (in which case
it cancels) or of the instance:

| runs | tuned n2r4 | base n2r6 |
| --- | --- | --- |
| b7 | 1039.20 | 996.66 |
| b8 | 1049.57 | 985.16 |
| b9 | 1058.87 | 983.89 |
| trend | **+1.9% over two runs** | **−1.3% over two runs** |

It answers cleanly: the late climb belongs to that one tuned instance, so folding it in would flatter
the tuned arm. The headline therefore uses the **matched b5–b7 window**, and the nine-run plateau is
reported only as the upper end.

## The delta, stated five ways

All five are in the patch's own `RESULT.md`, deliberately. All are round 2, on round 2's host.

| comparison | n | baseline | tuned | delta |
| --- | --- | --- | --- | --- |
| **matched b5–b7 window** — every instance in it has ≥7 runs | 2 v 2 | 988.41 (n2r3 987.11, n2r6 989.70) | 1045.42 (n2r2 1042.81, n2r4 1048.03) | **+5.77%** |
| each instance's own plateau (last 3 at plateau) | 3 v 2 | 987.67 | 1046.01 | **+5.91%** |
| worst tuned instance vs best baseline instance | 1 v 1 | 988.57 | 1042.81 | **+5.49%** — *lower bound* |
| best tuned instance vs worst baseline instance | 1 v 1 | 987.11 | 1049.22 | **+6.29%** — *upper bound* |
| single worst tuned run vs single best baseline run | 16 v 21 runs | 996.66 | 1006.63 | **+1.00%** — *distributions disjoint* |

**The honest headline is +5.8%, bounded +5.5% to +6.3%.** Every window agrees to within 0.8
percentage points, which is the sense in which the effect is *resolved* here rather than merely
bounded — and the reason it resolves is the 0.148% floor, against which +5.9% is **~40×**.

**Latency corroborates, which rules out a scheduling artefact.** A throughput change with unchanged
latencies usually means the workload changed; both move, uniformly, in the right direction.

| instance | arm | mean TTFT (ms), plateau | mean TPOT (ms), plateau |
| --- | --- | --- | --- |
| n2r1 | baseline | 7584.6 | 56.04 |
| n2r3 | baseline | 7447.7 | 56.51 |
| n2r6 | baseline | 7480.0 | 56.34 |
| **n2r2** | **tuned** | **6910.4** | **53.65** |
| **n2r4** | **tuned** | **6694.3** | **53.45** |

TPOT falls **5.0%** (56.30 → 53.55 ms, arm means) and TTFT falls **9.2%** (7504 → 6802 ms); the three
baseline instances agree with each other to 0.8% on TPOT. It is the same signature round 1 measured
on the other host (43.65–44.48 tuned vs 46.50–47.89 baseline).

### What this measurement does not license

- **Not claimed: that this is the figure on an uncontended node.** Round 2's host has a foreign tenant
  on two of eight ranks. Amdahl cuts both ways — the four tuned GEMMs are a fixed slice of work, and
  contending two of eight ranks changes the fraction they represent. **The direction of that bias is
  not established.**
- **Not claimed: round 1's +7.6% fast-born figure.** It rested on one instance, `r3`, and round 2
  could not test it because that host produces no fast-born instances. The honest reading is that
  +7.6% was the high end of a wide spread on a noisy machine, and the reproducible effect is ~+6%.
- **Cross-host agreement is on effect size only.** Two hosts, four tuned and five baseline instances
  in total, all point the same way; the range that survives both is **+5.8% to +6.0%**. Absolute
  levels differ by ~17% and are never mixed.

### Claims withdrawn from the earlier version of this entry

Recorded rather than deleted, because a reader who saw the old numbers needs to know they are gone.

| withdrawn claim | why | replacement |
| --- | --- | --- |
| "**+7.4% on arm means**" (1152.85 → 1237.90, 3 base vs 2 tuned) | a round-1 arm-mean over r1/r2/r4 against r3/r5, which excludes r6 and r7; the completed bundle carries no arm-mean framing for round 1 at all, only the class-blocked table and a +5.8–7.6% bound | **+5.8%** (987.67 → 1046.01), round 2 |
| "**worst-case floor of +1.3%**" (1184.84 → 1200.46) | that framing existed only because round 1's restart spread was wider than the effect; it has no analogue once the floor is 0.148% | **lower bound +5.49%**, worst tuned instance vs best baseline instance |
| "+4.3% against the bundle's quoted local warm mean 1187.20" | 1187.20 is round 1's host's level; round 2's absolutes are ~17% lower and not comparable to it | dropped — compare within a host only |
| "+7.6% (fast-born) / +7.54% over the stronger baseline instance" | rested on n=1 (`r3`); the bundle marks it superseded and says **do not quote it** | the replicated slow-born cell, **+5.8%**, is the part that survived |
| "restart-to-restart noise floor on this stack is 5.5–6.4%, twenty times Gemma's" | host property misattributed to the stack | 6.4% on round 1's host, **0.148%** on round 2's |
| "effects smaller than about 3% are not measurable on this model" | true of round 1's host only | re-derive the floor per machine |
| "result provisional, run still in flight" / "a third tuned instance and a matched baseline were running" | round 1's `base r8` never produced a result — the allocation was lost | run complete; `EXPERIMENT_COMPLETE` written |

## The artifact

`artifacts/a8w8_blockscale_tuned_gemm_dsv4_pro.csv` — md5 `f83564daa3a74abebd6b1e33b20d98bc`, 57
lines, 56 rows, being 14 power-of-two M buckets (M=1…8192) × 4 shapes, split **42 `ck` and 14
`cktile`** instances. All M values are reached via aiter's `get_padded_m(..., gl=1)` retry, which is
why 14 buckets cover hundreds of captured M sizes. This is the same file, by md5, that was deployed
on both hosts.

The four shapes, and where they come from:

| N × K | layer | calls per forward |
| --- | --- | --- |
| 2048 × 7168 | `attention.fused_wqa_wkv` (replicated on all ranks) | ×61 |
| 768 × 7168 | `mlp.shared_experts.gate_up_proj` | ×61 |
| 7168 × 384 | `mlp.shared_experts.down_proj` | ×61 |
| 7168 × 7168 | `mtp.e_proj`, `mtp.h_proj` | ×2 per draft step |

Each is a **near-miss by one architecture change** against a V3 shape that *is* tuned: the shipped
table has `(2112, 7168)` — V3's `q_a+kv_a+rope` fusion — but V4 drops the rope component, giving
2048; it has `(7168, 256)` and `(7168, 512)` but not `(7168, 384)`. This is the concrete form of the
warning not to reason by analogy from a sibling model.

They feed `gemm_a8w8_blockscale`. Without a matching row it falls through to
`gemm_a8w8_blockscale_ck(...)`, the CK heuristic, with no `kernelName` and no `splitK`. **The
production dispatch accepts only `ck` and `cktile` libtypes — there is no Triton or FlyDSL path for
block-scaled GEMM on this model**, which is worth knowing before going looking for one.

Aggregate kernel-level improvement across all 56 buckets: **6516 µs → 1983 µs, 3.29×**. Per-bucket it
ranges from 1.07× to 4.29×.

```bash
cp artifacts/a8w8_blockscale_tuned_gemm_dsv4_pro.csv \
   /usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/

rm -rf /tmp/aiter_configs     # MANDATORY
# restart the server — mandatory
```

Equivalently, `git apply -p1 artifacts/001_aiter_dsv4_blockscale_gemm_tune.patch` from
`/usr/local/lib/python3.12/dist-packages/aiter` — the patch just adds that same CSV.

The merged table grows 37,404 → 37,460 rows, exactly +56, which is the cheapest possible confirmation
that the merge took. A live server will not pick this up: there is an in-process `lru_cache` on
`get_CKGEMM_config` and vLLM's decode graphs are captured at startup.

**Deploy as a uniquely-named drop-in, never as an edit to a shipped table.** The filename must contain
`a8w8_blockscale_tuned_gemm` and must not contain `untuned`, because `get_config_file()` globs
`model_configs/*a8w8_blockscale_tuned_gemm*.csv` and filters on that substring. Before deploying, all
37,404 merged rows were checked for a `(gfx950, 256, M, N, K)` collision with any of the 56 new ones:
**zero**. That check matters because aiter's merge does not merely pick a winner on collision — it
**rewrites the source CSVs on disk** and then raises `RuntimeError(... Please re-run)`, so a collision
would have silently mutated AMD's shipped tables. Afterwards every `aiter/configs/**/*.csv` was
md5-compared against a pristine copy: one diff, the new drop-in itself.

## Engagement check

No environment variable required, because the *miss* line is unconditional while the hit line is
gated behind a flag the frozen config forbids. So the check is that the misses disappear:

```bash
grep -a a8w8_blockscale_tuned_gemm.csv /tmp/vllm_server_dsv4pro.log \
  | grep -oP 'N:\d+, K:\d+' | sort | uniq -c
```

- **Not engaged:** 1928 miss lines over the instance's life — 416 each for `N:2048, K:7168`,
  `N:768, K:7168` and `N:7168, K:384`, plus 680 for `N:7168, K:7168`.
- **Engaged:** empty. 1928 → 0.

Pair it with a negative control that must *survive*, which is what makes the check two-sided rather
than just "the log went quiet":

```bash
grep -a bf16_tuned_gemm.csv /tmp/vllm_server_dsv4pro.log | grep -c 'not found tuned config'
```

→ **8**, one per rank, for the deliberately untouched bf16 `lm_head` shape
`M:1024, N:16160, K:7168`. If that also goes to zero, something other than your drop-in changed.

**Round 2 ran this on every instance, and the count toggles in lockstep with the drop-in being added
and removed:** 1928 → 0 → 1928 → 0 → 1928 (voided n2r5) → 1928, with the negative control at 8 on all
six. The per-shape breakdown reproduced exactly on the second host. That toggle — engaged, disengaged,
re-engaged — is stronger evidence than a single before/after pair, and it costs nothing beyond the
restarts an interleaved A/B already requires.

A second, independent check answers "is the row *selected*, not merely *found*". `gate_selection.py`
calls the production entry point `aiter.gemm_a8w8_blockscale(...)` with the server down and checks it
reproduces the CSV's timing rather than the untuned one:

| N × K | M | CSV µs | production µs | untuned µs | ratio |
| --- | --- | --- | --- | --- | --- |
| 2048×7168 | 128 | 15.72 | 16.70 | 24.35 | 1.06 |
| 7168×7168 | 256 | 31.60 | 33.21 | 77.43 | 1.05 |
| 7168×384 | 2048 | 21.83 | 22.29 | 33.23 | 1.02 |
| 768×7168 | 4096 | 42.26 | 42.67 | 112.63 | 1.01 |

The 1–6% excess is Python dispatch plus the memoised lookup; had `kernelName` been dropped anywhere
in the dispatch, the production column would have tracked the *untuned* column instead.

## Accuracy gate

gsm8k 5-shot, lm-eval 0.4.9.2, `--fewshot_as_multiturn`, custom task, 1319 problems, none truncated.

| config | `exact_match,strict-match` | flexible-extract | source |
| --- | --- | --- | --- |
| reference | 0.9606 ± 0.0054 | 0.9598 ± 0.0054 | bundle |
| baseline, round 1 (r1) | 0.9598 ± 0.0054 | 0.9591 ± 0.0055 | FINDINGS §1.5 |
| **tuned, round 1 (r3)** | **0.9598 ± 0.0054** | **0.9591 ± 0.0055** | `eval_results/run_20260819_135734` |
| **tuned, round 2 (n2r2)** | **0.9575 ± 0.0056** | **0.9568 ± 0.0056** | `eval_results/dsv4_resume_20260820_164914` |

Round 1's tuned figure is numerically identical to its own baseline and 0.15σ from the reference.
Round 2 re-ran the gate on the second host — not required, but a new host deserves its own
correctness evidence — and got **0.9575, 0.57σ from the reference**, three problems out of 1319 from
round 1's figure. Both sit inside the ±0.0054 stderr band. **Pass on both hosts, with the same
drop-in by md5.**

The gate was run *before* the throughput benchmarks on the tuned instance, deliberately, because the
winning rows at small M for the `2048×7168` and `7168×7168` shapes use **`splitK > 0`**, and split
partials are accumulated in bf16. Their `errRatio` against an fp32 reference is **~2.5e-3, against
~8e-6 for the `splitK=0` rows** — one to two bf16 ULP, a real numerical change on a path that runs 61
times per token. No accuracy movement was observed on either host, but this is the one thing in the
patch that could have moved it. Caveat carried forward: gsm8k is a 1319-problem greedy-decode test
and not a sensitive numerical probe. A long-context, multi-turn or logit-tie-sensitive workload could
plausibly behave differently; if that matters more than the throughput, a `splitK = 0`-only table can
be regenerated with `tune_blockscale.py`, expecting the kernel win to drop to ~1.2× at small M while
staying ~2–3× at large M.

## How the tuning was done, and the deviation from the skillset

**No official tuner was used, and this is a documented deviation rather than a shortcut.** The aiter
wheel in this image ships compiled CK instances but not the tuner: there is no `csrc/`, no
`op_tests/`, no tuner driver at all. So `tune_blockscale.py` (shipped in `artifacts/`) enumerated the
already-AOT-compiled kernel instances straight out of the shared objects:

```bash
strings -n 20 aiter/jit/module_gemm_a8w8_blockscale.so        | grep '^a8w8_blockscale_1x128x128_'   # 19
strings -n 20 aiter/jit/module_gemm_a8w8_blockscale_cktile.so | grep '^a8w8_blockscale_cktile_'      # 22
```

then rebuilt vLLM's exact tensor layout, screened every candidate for numerics against an fp32
reference, timed the survivors interleaved across three passes, and took the minimum. `splitK` was
swept 0–4. `artifacts/gate_selection.py` then confirms that the production entry point
`aiter.gemm_a8w8_blockscale()` actually reaches the CSV's timing rather than the untuned timing —
which is the check that distinguishes "I found a fast kernel" from "the server will use it."

**What should have happened instead:** the `env-setup` skill prescribes cloning the aiter source
matching the wheel version, which takes about two minutes, and running the official tuner. The run
believed no source and no network were available; the network was in fact reachable. The substitute
is methodologically sound — enumerated from the binary, numerics-screened, selection-verified — but
it searched a **narrower space** than the official tuner would have, so the **+5.8%** is a floor on
what this surface offers rather than its ceiling. Nothing here was compiled: the tune is pure
selection among instances already in the shipped `.so`, and the deployment is a CSV drop-in.

One caveat on the kernel table itself: the candidates were timed on an idle GPU with one kernel in
flight, and eager timing amortises no launch overhead. The 1.07× "wins" at `7168×384` / small M
(6.06 → 5.67 µs) are inside launch-overhead territory and should not be believed at face value. The
end-to-end measurement is the arbiter, not the per-kernel table.

## What was tried and did not work

Attempt 2 examined four surfaces and rejected all four **before spending a server instance** — the
right call on round 1's host, where an instance cost half an hour and the floor was 6%. **Two of the
four rejections are floor-dependent and should be re-derived against your own floor rather than
inherited** — the lm_head and sampler surfaces were bounded at <1% and ~1.5% respectively and then
dismissed as sub-noise, which they are at 6% and are not at 0.148%. The other two are rejected on
structure, not on size, and stand regardless.

| surface | evidence | verdict |
| --- | --- | --- |
| bf16 `lm_head` GEMM `M:1024, N:16160, K:7168` | misses `bf16_tuned_gemm.csv`, falls through to `torch.mm`, but only **8 miss lines total** — once per rank at startup, never again during 5–9 benchmark runs | Prefill-dominated workload (~1.6M prefill against 197k decode tokens in a ~165 s run). Eliminating the GEMM *entirely* is worth **<1%**. |
| fused MoE | all 11 M buckets already resolve to named FlyDSL two-stage kernels via the shipped `dsv4_fp8fp4_tuned_fmoe.csv`; the 384 routed experts are **MXFP4** (`float4_e2m1fn_x2`, `per_1x32`), not fp8 — which is why the a8w8 block-scale table never covered them | The dominant kernel time is already AMD-tuned. Beating it without the tuner or `csrc/` is unrealistic. |
| sampler falling back to PyTorch | `aiter sampler does not support per-request generators` — a Python loop over the batch, one kernel launch per request per step | Optimistically ~1.5% of a 165 s run, and the fix risks gsm8k for a sub-noise gain. |
| DSA indexer decode, `use_flattening=False` | with MTP depth 1, `next_n=2` is already on the native multi-token fast path | Flattening would be a workaround, not an optimization. Nothing to do. |

**And the number never to quote: 3.29×.** The tuned GEMMs really are 3.29× faster in aggregate, and
that is worth about 6% end to end, because these four dense GEMMs are a real but modest slice of a
forward pass dominated by the fused MoE (already tuned) and by attention. That is Amdahl's share, not
a measurement failure — but a 3.29× headline over a ~6% result is the most misleading way to describe
this patch.

One defect found in passing and not acted on: 449 shapes in the merged MoE table have duplicate rows
after `act_type` was dropped from the key, and aiter keeps the **first** match rather than the
fastest — unlike the GEMM merge, which resolves duplicates on lowest `us`. For those 449 shapes the
kernel chosen is a function of file-glob order. None of this model's 11 buckets are affected, but it
is a live hazard for other models.

## What remains outstanding

The reproduction and confirmation items from the earlier version of this entry are **done**: the run
completed, the arms were re-interleaved on a second host from the exported artifact, and the
rank-order claim is stated over the final instance count. What genuinely remains:

1. **The obvious next increment, left undone for time rather than merit.** The CSV carries rows only
   at the 14 powers of two, so an M of 1088 is served by the kernel tuned for M=2048. Adding rows at
   the actual graph-capture buckets (104, 112, 120, 128, …) would let the first, exact lookup hit.
   Expected gain is second-order — which is precisely why it was not worth an A/B against a 6% floor,
   and is worth reconsidering on a host with a 0.148% one.
2. **A measurement on an uncontended node.** Round 2's host carried a foreign tenant on two of eight
   ranks; the direction of that bias on the effect size is not established.
3. **A wider search.** The tune selected among the instances compiled into the shipped `.so`. The
   official tuner searches a wider space, so +5.8% is a floor on this surface, not its ceiling.
4. **aiter's commit sha**, which is not recoverable from a wheel with no git metadata.

## When this entry stops applying

Silently, in every case: **arch ≠ gfx950 or CU count ≠ 256** (both are literal columns in the key);
**TP ≠ 8** (N and K shard differently); **quantization ≠ `deepseek_v4_fp8`** (different op, different
table); **MTP disabled or a different `num_speculative_tokens`** (the `7168×7168` rows go unused and
the gain shrinks); **concurrency, ISL or OSL changed** (the M distribution moves off the tuned
buckets); a **stale `/tmp/aiter_configs`** or **no restart**; a **different aiter version**.

Untested rather than known-inert: other concurrency levels, other input/output lengths, and non-MTP
operation. On a mismatched arch or CU count the rows simply do not match — a no-op, not a wrong
kernel.

Still reusable when inert: the four-shape target list, `artifacts/tune_blockscale.py` as a method for
tuning against a wheel with no tuner in it, `artifacts/gate_selection.py` as the selection gate, the
add/remove/re-add engagement toggle with a negative control, and the measurement protocol —
re-derive the floor per host, run until the last three are flat, interleave the arms, extend both
arms or neither. That protocol is the most portable thing in this entry.

## Skillset gaps this run exposed

Recorded here because they are environment-shaped rather than model-shaped, and the next vLLM run
will hit them too.

- **`tuning-aiter` §3's tuner recipe assumes a source checkout.** This wheel has no tuner. Either the
  page needs a no-source path, or `env-setup`'s "clone aiter matching the wheel" step needs to be
  impossible to miss. This run missed it and wrote its own tuner. The missing page is: enumerate the
  AOT-compiled instances out of the shipped `.so` with `strings`, then drive them through the public
  entry points with an explicit `kernelName`.
- **There is no page on tuning a stack with no source and no network.** Pristine-copy discipline
  (`cp -a` the site-packages tree *before* the first edit, because there is no `git diff` to fall back
  on), how to inventory an AOT `.so`, and how to express a patch when the "repo" has no SHA.
- **`tuning-in-vllm` §1's Triton fused-MoE JSON path is never entered on this model** — zero
  `Using configuration from` or `Using default MoE config` lines across a full server startup, in
  every log captured. The MoE goes through aiter CSVs keyed on `(gfx, cu_num)`. Five of the skill's
  eight checklist items are dead here.
- **`tuning-in-vllm` §4's `VLLM_TUNED_CONFIG_FOLDER` is irrelevant here**, and aiter's per-table env
  vars (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`, `AITER_CONFIG_FMOE`, …) have **replace-not-add**
  semantics — the opposite of what the page teaches. Within a single vLLM process there are two
  tuned-config override variables with opposite semantics, and which one you get depends on which
  library owns the op. The frozen launch forbids env overrides anyway, leaving only the route §4
  discourages: editing the package directory.
- **The documented shape-capture and engagement methods are all env-gated** (`HIPBLASLT_LOG_MASK`,
  `AITER_LOG_TUNED_CONFIG`), and `tuning-in-vllm` §5's `grep -c "is tuned on cu_num"` additionally
  counts distinct `(M,N,K)` rather than kernel invocations, because the lookup is `lru_cache`d. Under
  a frozen configuration the only thing that works is the unconditional miss-line grep used above.
  **Every framework page needs a frozen-configuration section**, covering deployment, shape capture
  and engagement verification together, because they all go off the table at once.
- **`measurement.md` Rule 3b's *procedure* is exactly right and its *examples* are badly calibrated.**
  The page tabulates restart spreads of 0.36% and 0.16%; this campaign saw **5.5–6.4% on one host and
  0.148% on another, for the identical stack**. The page is careful to say the ratio must be measured
  rather than predicted, and that advice saved this run twice — but every worked example is a small
  dense model on 1–2 GPUs, and the set has no example of the same configuration having two floors two
  orders of magnitude apart. **That is the row this campaign should contribute.** Related and also
  missing: on this stack throughput climbs across the first two or three runs against a fresh server
  — up to 14% below plateau on run 1 — so "repeats" must mean "repeats after plateau", and the
  repeat-and-spread discipline needs a plateau-detection step before it.
- **`graph_captured_benchmarking.md` diagnoses the problem and then hands you a SGLang-shaped tool.**
  Its harness wraps a callable in `torch.cuda.graph`; what needed measuring here is a kernel selected
  by a CSV lookup *inside* vLLM's 51 captured decode graphs, which cannot be reproduced by capturing
  one op. Its concrete warning did bite, though: eager timing overstates small-kernel wins.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/deepseek_v4_pro_tuning/`, marked complete by its
`EXPERIMENT_COMPLETE` file. `FINDINGS.md` §1.1–§1.5 preflight and baseline reproduction on
`crsuse2-m2m-150`, §1.6 the second host and its foreign tenant, §2.1 where the untuned kernels are,
§2.2 attempt 1 and round 1's arm table, §2.3 the four rejected surfaces, **§3 the completed
end-to-end result and the final claim**, §4 skillset assessment.
`patches/001_aiter_dsv4_blockscale_gemm_tune/RESULT.md` carries the five delta framings, both hosts'
arm tables and the per-bucket kernel timings; `MANIFEST.json` has the deploy contract, the
`base_sha: null` note and the engagement expectations. `reference/README.md` documents the
931.22 → 1192.9 MTP config gain and `reference/local_reproduction.md` the 1187.20 local warm mean.
Raw output for all instances is in `results/`; the voided baseline instance is under
`results/_invalid_base_n2r5_*` with `results/_invalid_base_n2r5_WHY.md`. Accuracy runs are
`eval_results/run_20260819_135734` (round 1) and `eval_results/dsv4_resume_20260820_164914` (round 2).
