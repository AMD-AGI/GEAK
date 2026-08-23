# Gemma-4-26B-A4B-it on MI355X — bf16 serving, TP=2, six-patch Triton stack

**Verified win: +24.88% output throughput** (2916.54 → 3642.20 tok/s), gsm8k strict-match 0.9386 ±
0.0066 against a 0.9409 ± 0.0065 requirement — a pass at −0.35σ. Mean TPOT 17.345 → 14.373 ms
(−17.1%). The win is a **stack of six patches**: four Triton launch/kernel changes, two tuned config
tables, one of which needs a rebuild.

This is the largest gain in the knowledge base and the only entry where the *stack* matters more
than any single artifact. Do not cherry-pick patches expecting proportional gains.

Found over three runs, 2026-08-18 to 2026-08-19. **Reproduction status: re-measured from the
exported patches on a second machine.** Run 3 started from a pristine tree on `crsuse2-m2m-052`
(`git status --porcelain` matched the recorded pristine manifest; sglang at `197832bc`, aiter at
`d9e5ef7c`), applied 001–005 from their exported `change.patch` files with `git apply --check` clean
on every one, and re-anchored both the baseline and the stack there. The rebuilt `sgl_kernel` shared
object hashes to `17b764881536ded99566aeae86b0c1ab` on both machines, so the compiled patch is
bit-reproducible across hosts. What has *not* happened is an independent reproduction outside this
campaign — see "Reproducing this" below for the exact steps and the numbers to expect.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | MI355X, `gfx950`, **256 CU**, **two GPUs** | **yes** — patches 001 and 003 gate on `is_gfx95_supported()`; on any other arch they skip and stock launches run |
| container | `rocm/sgl-dev@sha256:95a933896aeab2a431521ece6ebe90c1db37a3aaf1e32a938d56ef7ccf6603a5` | descriptive |
| SGLang | 0.5.17, commit `197832bcf536543092e621e03d61ae2602a392d0` | **yes** — the engagement markers below are log strings from this tree |
| aiter | commit `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | descriptive — not on the hot path for this model |
| torch | 2.9.1+rocm7.2.0, ROCm 7.2.0 | descriptive |
| Triton | **3.7.0** in this image; the reference run ran **3.6.0** | **yes** — it is the `configs/triton_3_7_0/` path component of the MoE config filename, and see the baseline note |
| model | Gemma-4-26B-A4B-it, **TP=2** | **yes** — sharding sets every N and K, and fixes the MoE `N=352` in the config filename |
| precision | **bf16 throughout** — weights *and* KV cache | **yes** |
| attention backend | **Triton** (SGLang's own default for Gemma4, not an override) | **yes** |
| MoE runner | **Triton** (`--moe-runner-backend triton`) | **yes** |
| quantization | none | **yes** |

**The upstream sweep labels this arm `precision: fp8`. That label is wrong** — resolved ServerArgs
show `quantization=None` and `kv_cache_dtype='auto'`, the checkpoint is 51.6 GB for ~26 B params
(2 bytes/param), and the server log reports `KV Cache is allocated. dtype: torch.bfloat16` for both
pools. Everything is bf16. Do not go looking for FP8 GEMM tables on this model; the FP8 config
tables that carry the Qwen3-8B win are not on this path at all.

## Launch configuration

```bash
ROCR_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path <gemma-4-26B-A4B-it> \
  --tp-size 2 \
  --context-length 11264 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --moe-runner-backend triton \
  --attention-backend triton \
  --disable-radix-cache \
  --watchdog-timeout 1800 \
  --trust-remote-code
```

Resolved: `page_size=1`, `max_prefill_tokens=16384`, `ep_size=1`, `swa_full_tokens_ratio=0.8`,
`max_running_requests` unset (4096 effective), decode HIP-graph capture enabled.

**The launch script exports nothing, but the process environment is not empty — read it, do not infer
it.** The measured runs export only a GPU pin, from `analysis/env.sh`:

```bash
export ROCR_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
```

and the benchmark wrapper adds `BENCH_TRUST_REMOTE_CODE=1` and `HF_HUB_TRUST_REMOTE_CODE=1`. The
container image itself, however, exports `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (which is what the
TP=2 all-reduce actually runs, and it is part of this baseline) and `HSA_NO_SCRATCH_RECLAIM=1`. The
sibling Mixtral entry burned a whole attempt on exactly this confusion, so the rule is:
**`cat /proc/<server-pid>/environ | tr '\0' '\n'` before believing any claim about what is or is not
set.**

The GPU pin is load-bearing for a different reason than performance. In run 2 neither the leg script
nor the launch script sourced it, so which physical pair a leg landed on depended on the invoking
shell; legs C1, C2 and D1 ran on physical GPUs 2,3 while C3, D2, E1, D3 and E2 ran on 0,1. The audit
found the effect inside the noise floor and no conclusion moved, but a leg that can choose its own
hardware invalidates the A/B by construction. Pin it in the harness, not the shell.

**There are no `SGLANG_*` or aiter environment variables in the recipe, and no config deltas.**
Unlike the Qwen3-8B entry,
where the recorded baseline already has a config sweep baked into it, **this baseline is stock.** Do
not read the empty env as a tuned state — see *Config headroom deliberately not in this baseline*
below. The reference session reached it by re-baselining from scratch after its warm-start replay
was skipped (`warm_replay_outcome = {status: skipped, reason: "legacy native records do not satisfy
the current replay contract"}`), leaving `optimization_stack` empty of config entries.

## Architecture, and why it matters for target selection

30 layers, 128 experts, and a **5:1 interleave of sliding-window and full attention** — 25 sliding
layers at `head_dim` 256, 5 full-attention layers at `global_head_dim` 512. Two separate KV pools
result, per rank:

| pool | tokens | K | V |
| --- | --- | --- | --- |
| full attention | 2,307,417 | 11.0 GB | 11.0 GB |
| sliding window | 1,845,933 | 88.0 GB | 88.0 GB |

The two attention paths are different kernels with different shapes, so neither layer count nor
pool size predicts their share of runtime. **Measure the split before choosing a target** rather
than reasoning about it from the architecture.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, 8 warmups, seed 0, `random_range_ratio 1`,
InferenceX `benchmark_serving` fork.

## Baseline and noise floor

| | value |
| --- | --- |
| output throughput (measure_round, 192 requests, 64.14 s) | **3065.1 tok/s** |
| second instance of the same config (warmup_round) | 3070.9 tok/s |
| third instance, same config and client (GEAK cold basis) | 3081.7 tok/s |
| restart-to-restart spread across those three | 16.6 tok/s = **0.54%** |
| total token throughput | 27,585.9 tok/s |
| mean TTFT / TPOT / E2EL | 4130.0 ms / 16.851 ms / 21,368.8 ms |
| gsm8k `exact_match,strict-match` | **0.9424 ± 0.0064** |

Three instances bound the floor at ~0.54% but do not give a distribution. Measure your own across
≥3 restarts before judging a candidate (`../../tuning-core/measurement.md` Rule 3b); on the
comparable Qwen3-8B stack the true restart spread was 0.36%.

**Reproducing 3065.1 is image-sensitive, and it is the image rather than the hardware.** On the
pinned image above this configuration measures **2927.0 tok/s** — 8 benchmarks over 3 server
instances on `m2m-018`, restart-to-restart spread **0.16%** — and a 9th run on a different machine
(`m2m-059`, fresh container and fresh server) landed at 2925.9, agreeing to within 0.04%. That is a
flat **−4.43% against the reference**, with the shortfall concentrated in prefill (TTFT 4674 ms vs
4130, +13.2%) while decode is nearly unchanged (TPOT 17.31 ms vs 16.85, +2.7%) and gsm8k is
unaffected (0.9409 ± 0.0065 against the reference's 0.9424 ± 0.0064). Two machines agreeing that
closely rules out a bad node or a topology accident. The
workload is provably identical (same `total_input_tokens` 1,572,864 / `total_output_tokens` 196,608 /
192 completions) and the resolved ServerArgs are identical in **all 458 keys the two runs share** —
the pinned image simply carries a *newer* stack: 6 additional ServerArgs fields absent from the
reference, and **Triton 3.7.0 against the reference's 3.6.0** (read straight off the MoE config path
each build searches — `configs/triton_3_7_0/…` here versus `configs/triton_3_6_0/…` there). On this
model both the attention backend and the MoE runner are Triton, so a Triton compiler bump reaches
essentially every hot kernel, and a prefill-weighted regression is what you would expect from it.
The reference build is not recoverable from the digest. Treat 3065.1 as the reference build's number
and re-establish a local baseline on whatever image you actually run.

### The run that produced the win, and the floor that made it reportable

Measured on `crsuse2-m2m-052`, a node verified idle on all eight GPUs at container start. Arms were
alternated across server restarts throughout, two benchmarks per server instance:

| arm | instances | runs | mean tok/s | vs base | gsm8k strict |
| --- | --- | --- | --- | --- | --- |
| pristine base | 2 (G2, G4) | 4 | **2916.54** | — | 0.9431 ± 0.0064 |
| patches 001–005 | 4 (G1, G3, H2, H4) | 8 | **3602.40** | **+23.52%** | 0.9386 ± 0.0066 |
| patches 001–006 | 3 (H1, H3, H5) | 6 | **3642.20** | **+24.88%** | 0.9386 ± 0.0066 |

| noise floor | spread |
| --- | --- |
| restart-to-restart, 4 pristine legs on this node | **0.25%** |
| restart-to-restart, run 2's shared node | **2.05%** (59.3 tok/s on a 2893.6 mean) |

**The floor is the story of this entry.** Run 2 measured 2.05% on a node shared with a neighbour
cycling ~265 GB of allocations. Three of the six patches — 004 at +1.18%, 005 at +1.24%, 006 at
+1.10% — are individually *inside* that floor and were unreportable there. Moving to a quiet node and
re-measuring the floor at 0.25% is what made them defensible, and it is the single highest-leverage
action available on this stack. Do not inherit a floor across machines; measure your own, and do it
before you start attributing patches.

The 001–006 arm is disjoint from 001–005: the slowest 001–006 run (3637.00) is 30.25 tok/s above the
fastest 001–005 run (3606.75). Both are far outside the 0.25% floor.

## Headroom

Memory roofline puts the decode ceiling at **7224.6 tok/s** against the 2927.0 baseline achieved —
**40.5% of ceiling** (`PROMPT.md` §4). That is a bound, not a forecast, and the gap
includes everything the roofline ignores: collective communication across the TP=2 pair, kernel
launch structure, and the prefill share of the run.

## The MoE Triton config — the lever, now taken by patches 002 and 006

**This section describes the lever as it was found: no tuned Triton fused-MoE config existed for this
GPU, so all 30 layers ran on the heuristic default.** Patches 002 and 006 are what took it, worth
+3.92% and +1.10% respectively. Everything below is retained because it is what you need in order to
regenerate those tables for a different topology, Triton version or device name — all of which move
the filename and silently un-tune the model. The five traps are still live traps.

The upstream static analysis enumerated all 7 shipped `triton_*` config directories
(358 JSON files) and found **zero** matching `AMD_Instinct_MI355X`; the nearest AMD entries are
MI300X, MI325X and Radeon. The server says so itself, four times per launch:

```
Using default MoE kernel config. Performance might be sub-optimal! Config file not found at
  …/configs/triton_3_7_0/E=128,N=352,device_name=AMD_Instinct_MI355X.json
  …/configs/triton_3_7_0/E=128,N=352,device_name=AMD_Instinct_MI355X_down.json
```

A TraceLens attribution taken on the stock TP=2 baseline put the Triton fused-MoE path at **~24.5%
of GPU time** (13.71% + 7.95% + 0.82% for the fused-MoE kernels, 2.03% for `moe_align`). That figure
comes from the earlier session's profile, not from a trace of this exact image — treat it as
"MoE is a first-rank target here", and confirm the share against your own trace before sizing a
result on it. Gemma4's experts are very
narrow — `moe_intermediate_size` 704, sharded to **N=352 at TP=2** — and narrow experts are the case
the default heuristic handles worst: it picks `BLOCK_SIZE_N` 128 or 256, against which N=352 leaves a
ragged tail tile on every expert. This is the `tuning-in-sglang` lever, not an aiter one, and it is
a missing *artifact* rather than a code defect.

Five traps, all recorded from the reference session's own static recon. **The first three apply to
this bf16 config; the last two do not, and are listed so they are not applied by mistake:**

1. **The device-name miss cannot be worked around by borrowing a donor.** The lookup key embeds
   `get_device_name().replace(' ','_')`, and the triton-version fallback loop retries *the same
   device-specific filename* in the other `triton_*` directories. It can never fall back to the
   H200 or MI325X file, so a correct-shape donor from another vendor is unreachable. The file has
   to be produced for this device name.
2. **`SGLANG_MOE_CONFIG_DIR` hard-crashes the server unless the two-level nesting exists.** The
   version-fallback scan does a bare `os.listdir(os.path.join(config_dir,'configs'))` with no
   `try/except`, so pointing the variable at a directory without a `configs/` subdirectory raises
   `FileNotFoundError` **on the first MoE forward** rather than degrading to the default. This is
   the natural mistake, because the tuning script writes a bare filename into the CWD with no
   nesting. Create `$DIR/configs/triton_3_7_0/` before exporting the variable. Note the override
   also *replaces* the shipped directory, hiding all 358 bundled files.
3. **The filename's `N` tracks the topology, not the model, and a mismatch fails silently.** The
   tuner derives `N` from the sharded intermediate size, so `N` is 352 at TP=2 and 704 at dp2-tp1.
   Its `--tp-size` default of 2 happens to be right for *this* config and wrong the moment the
   topology changes. Always confirm the emitted basename against what the server logged as missing —
   `E=128,N=352,device_name=AMD_Instinct_MI355X.json` here — before moving it into place. Get it
   wrong and the file is produced, the variable is set, the server still logs `Config file not
   found`, and the result reads as "the tuned config gave 0%".
4. *(Not this config.)* Under `--quantization w8a8_fp8` the filename gains a `,per_channel_quant=True`
   suffix and a `dtype=fp8_w8a8` component — only 10 of the 358 shipped configs carry that suffix.
   The frozen bf16 config here needs **neither**, as the log above confirms. If the quant path ever
   changes, re-derive the filename; do not reuse.
5. *(Not this config.)* `W8A8FP8MoEMethod.create_moe_runner` hardcodes `MoeRunnerBackend.TRITON` and
   its `apply()` never checks the backend, so `--moe-runner-backend aiter` is silently discarded
   under `w8a8_fp8`. Only reachable if the FP8 quantization path is enabled.

## Config headroom deliberately not in this baseline

Two different facts here, and conflating them is the trap:

- **This session's own config search was genuinely exhausted** — 0 of 5 accepted across two full
  cycles, with real negative results (below). Re-running those levers is waste.
- **A prior session validated a substantial config stack that this baseline does not contain.**
  Session `20260813T091400Z` recorded `validated_gain_pct` **23.57%** over 4 stack entries:
  `dp2-tp1-alone` (+6.78%), `quantization-fp8-online` (+11.21%), `sched-occupancy-cg64` (+16.22%),
  `quant-w8a8-fp8-per-token` (+23.57%, cumulative). **None of those four is among the five this
  session tested.** The recipe survived only as names: `replayable: false`,
  `replay_material_available: false`, `replay_disabled_reason: "legacy native records do not satisfy
  the current replay contract"`.

Treat that 23.57% as *direction, not magnitude*. Its `best_throughput` of 6108.5 tok/s implies a
prior-session baseline near 4944 tok/s, which is nothing like this session's 3065.1 — so the two were
measured on different bases and the percentage cannot be transplanted onto 3065.1. What it does
establish is that **the flag axis on this model is not exhausted**, in particular the dp2-tp1
topology and the FP8 online-quant path. Note that switching to dp2-tp1 also moves the MoE shape from
N=352 to N=704, which changes the config filename above.

## Levers already tested negative — do not re-test

From the reference session, with magnitudes. These are settled unless the root cause changes:

| lever | result |
| --- | --- |
| `SGLANG_USE_AITER=1` (RMSNorm fusion only) | **−7.61%** |
| NEXTN MTP speculative decode | **−44.38%** — architectural miss, not a tuning problem |
| `mixed-chunk` with explicit prefill size | **−15.19%** |
| scheduler overhead / poll combo | **−7.60%** |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | no lever exists — decode traffic is ~0.34 MiB, below any QuickReduce threshold |
| `--moe-runner-backend` variants, `fp8-gemm-runner-aiter`, `ep-size-2`, `torch-compile-decode-bs64` | all failed warmup outright |

## Why the recorded "+8.15% GEAK gain" must not be trusted

The reference state carries `cumulative_gain = 8.146%` with provenance
`geak_orch_harness_validated`, and an `optimization_stack` entry claiming **3314.8 tok/s**. Neither
number is a measurement of this configuration, and the state file says so itself:

- `baseline_alignment` on that entry is `status: "warning"`, `divergence_pct: **48.41**` against a
  `warning_threshold_pct: 3.0`. GEAK's own hot baseline was **4548.95 tok/s** where the orchestrator's
  was 3065.1 — its reported TTFT of 794.9 ms against the baseline's 4130 ms shows it was measuring a
  prefix-warm regime, not the frozen cold workload.
- The 3314.8 figure is **synthetic**: 3065.1 × 1.08146, i.e. GEAK's percentage applied to the
  orchestrator's baseline. No run ever produced it.
- The accepted config also adds `--context-length 13312`, against the baseline's 11264.

This is the canonical example of `../../tuning-core/measurement.md`'s rule that tuner winners must be
re-measured on the cold shipping harness. The underlying kernel change — an authored Triton
`_fwd_grouped_kernel_stage1` for the full-attention layers, isolated 1.66×, claimed 10.38% E2E — may
still be real. It has simply never been measured on this baseline.

## The reference run never tested SGLang-side source changes at all

Worth knowing before assuming the source axis is picked over. The reference session was structurally
unable to patch SGLang: its patch specialists were bound to a worktree that *was* the aiter repo, so
every SGLang-side patch failed `patch_safety missing_target` regardless of quality — confirmed 3/3
across genuinely distinct targets, and recorded as a session-wide conclusion ("aiter-side patches can
succeed where sglang-side ones structurally cannot"). Several SGLang-side targets were therefore
confirmed real and then abandoned unfixed, not disproven. A run with a writable SGLang tree is
working an axis the reference never reached.

## Notes for a run on this stack

- **TP=2 means collective communication is in the critical path** and it is not transparent in a
  torch trace. A per-rank kernel-time attribution will not sum to wall time, and time spent waiting
  on a peer can look like a slow kernel. Establish where the collectives sit before attributing a
  regression to a kernel.
- **Both ranks appear in a trace.** De-interleave by rank before ranking kernels.
- The available reference trace was captured on a 776-prompt profiled run, not the 192-prompt
  benchmark, and profiling adds overhead. Use it for relative relationships and shape discovery,
  never for absolute durations.
- Decode runs under HIP-graph capture here too, so every candidate needs a server restart and the
  benchmarking constraints in `../../tuning-core/graph_captured_benchmarking.md` apply unchanged.

## The six patches

Order matters, and two of them have hard dependencies. Deltas are as measured in each patch's own
arm; the stack total of +24.88% is the number to trust.

| # | artifact | what it changes | kind | delta | needs |
| --- | --- | --- | --- | --- | --- |
| 001 | `001_gfx95_grouped_decode_attn_launch.patch` | gfx95 launch table for the grouped decode attention stage-1 kernel, in `ops/attention/decode_attention.py` | interpreted Triton launch params | +7.61% | — |
| 002 | `002_moe_triton_config_e128_n352_mi355x.patch` | tuned Triton fused-MoE config table, `E=128,N=352,device_name=AMD_Instinct_MI355X.json` | data only, untracked file | +3.92% | — |
| 003 | `003_gfx95_extend_attn_launch.patch` | gfx95 launch table for extend (prefill) attention, in `ops/attention/extend_attention.py` | interpreted Triton launch params | +7.20% | — |
| 004 | `004_swa_window_loop_bound.patch` | sliding-window lower bound on the extend-attention key loop | Triton kernel body | +1.18% | **003** (textual dependency; will not apply to a pristine tree) |
| 005 | `005_moe_token_sort_two_level.patch` | two-level counting sort for MoE token routing, in `aot/csrc/moe/moe_align_kernel.cu` | **compiled — rebuild required** | +1.24% | — |
| 006 | `006_moe_split_down_config_table.patch` | separate down-projection tile table, plus one retuned key in the up table | data only, two untracked files | +1.10% | **002** (it edits that table) |

## Deploy

Patches 002 and 006 are *not* ordinary source patches: they create untracked JSON files inside the
SGLang tree, which is why `artifacts/moecfg/` carries them as plain files as well. Either route
works; the file copy is the one that survives a `git checkout`.

```bash
cd /sgl-workspace/sglang

git apply .../artifacts/001_gfx95_grouped_decode_attn_launch.patch
git apply .../artifacts/003_gfx95_extend_attn_launch.patch
git apply .../artifacts/004_swa_window_loop_bound.patch      # after 003, not before
git apply .../artifacts/005_moe_token_sort_two_level.patch

# the two MoE tables (equivalently: apply the 002 and 006 patches)
cp .../artifacts/moecfg/006/*.json \
   python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0/

# 005 is compiled: rebuild and install, or it does nothing at all
cd python/sglang/kernels/aot && python3 setup_rocm.py build_ext
cp build/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so \
   /opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so.new
mv -f /opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so{.new,}

rm -rf ~/.triton/cache      # optional but cheap
# then start the server. A restart is mandatory — see below.
```

Regenerate the tables instead of copying them with `artifacts/make_moe_config.py --write` (002) and
`artifacts/make_moe_down_config.py --write` (006); both carry self-tests.

`artifacts/moecfg/` holds both states of that directory, because 006 is a *second state* of the same
untracked files rather than an addition to them: `moecfg/002/` is the single up table as patch 002 left
it, and `moecfg/006/` is the retuned up table plus the new down table. Deploy `006/` for the full
stack; `002/` is there for anyone reproducing the 001–005 arm, which is the intermediate 3602.40
figure.

**Every way this deploy silently does nothing:**

1. **Dropping any of it into a running server.** Decode replays a captured HIP graph, so a live
   drop-in benchmarks *exactly* like the previous arm — this is the single most expensive trap on this
   model, because it produces a clean, plausible, wrong number. A restart is mandatory for all six.
2. **Editing the `.cu` for 005 without rebuilding and reinstalling the `.so`.** The server keeps
   mmapping the old binary. The source-side check passes and the kernel never changes.
3. **Wrong MoE filename.** Any drift in device name, `E`, `N` or the `triton_3_7_0` directory and the
   loader falls back to the heuristic with a log line you have to be looking for.
4. **`_down.json` missing while the retuned up table is present.** This is worse than not deploying:
   the 006 up-table key was tuned *expecting* the down table, and without it the M=64 shape measures
   **−10.96%**. Deploy 006 as both files or neither.
5. **`USE_TMA` absent from the down table.** The mechanism is
   `down_moe_use_tma = _down_moe_use_tma() and down_config is not None and down_config.pop('USE_TMA', False)`
   — supplying a `_down` file is a precondition for the TMA path but does not enable it; the key must
   also be present *and* truthy. A `_down` file without it is a tile-shape-only change, which is exactly
   what 006 ships: the key is deliberately omitted and `make_moe_down_config.py` self-tests that it is
   absent. The trap is expecting TMA on the strength of 002's manifest, not a missing field in this table.
6. **Applying 004 to a pristine tree**, which fails loudly, or **006 without 002**, which does not.

## Engagement check

The harness gated every arm on counted markers on **both TP ranks** before any benchmark ran,
two-sided: the positive marker present the expected number of times *and* the negative marker exactly
zero, with the control arm asserting the mirror image. No environment variable is needed to make any
of these appear.

```bash
L=/tmp/sglang_server_gemma.log
grep -c 'gfx95 grouped decode'                 $L   # 001: engaged 4, not engaged 0
grep -c 'gfx95 extend attention tuning active' $L   # 003: engaged 4, not engaged 0
grep -c 'two-level MoE token sort active'      $L   # 005: engaged 2, not engaged 0
grep -c 'Using MoE kernel config from'         $L   # 002: 2   006: 4   base: 0
grep -c 'Using default MoE kernel config'      $L   # 002/006: 0        base: 4
grep 'Using MoE kernel config from' $L | grep -c '_down\.json'   # 006: 2, otherwise 0
```

Four rather than two for the attention markers because each fires once per shape per rank: the
`BLOCK_DMODEL=256` sliding path and the `BLOCK_DMODEL=512` full-attention path, on TP0 and TP1.
**Counts, not presence** — a single rank engaging is a real and easy failure, and it leaves the fast
rank waiting on the slow one at every all-reduce, so it can read as no gain at all.

Patch 004 has no log marker, because it is a change to a Triton kernel body with no Python-side
branch. Two checks stand in for it. Structurally, `git diff --numstat` on `extend_attention.py`
must show **99** added lines for 003+004 against **72** for 003 alone. Behaviourally — and this is
the one that cannot be faked by bookkeeping — the profiled sliding-layer duration drops to ~410 µs
from ~740 µs while the full-attention layers hold at ~2085 µs.

For 005, the `.so` md5 discriminates within this campaign: `17b764881536ded99566aeae86b0c1ab`
patched against `ccd8e42ce308128f32e4bf82d01a06df` for a pristine rebuild. Note that neither equals
the shipped wheel's `0a6648b0f233bea61360332510c4ebec`, so md5 alone cannot tell "patched" from
"never rebuilt" — use it together with the banner count.

## Accuracy gate

gsm8k 5-shot, lm-eval pinned at `b315ef3b05176acc9732bb7fdec116abe1ecc476`, InferenceX task variant
from `eval/gsm8k.yaml`, `num_concurrent=64`, `temperature=0`, `max_tokens=9216`.

| config | `exact_match,strict-match` | flexible-extract |
| --- | --- | --- |
| requirement (bundle contract) | 0.9409 ± 0.0065 | — |
| pristine base, this node | 0.9431 ± 0.0064 | 0.9424 |
| patches 001–005 | 0.9386 ± 0.0066 | 0.9378 |
| **patches 001–006** | **0.9386 ± 0.0066** | 0.9378 |

−0.35σ against the requirement, and the pristine base on this node reads +0.34σ high, so the whole
spread sits inside the cross-machine baseline envelope. **Pass.**

## What was tried and did not work

The most reusable part of this entry. Run 3's own negatives first:

| attempt | kernel / op level | end to end | verdict |
| --- | --- | --- | --- |
| TP=2 all-reduce | 895 µs per all-reduce, **86% of a directly measured 60 GB/s peer link**; already INT8 via the image's `ROCM_QUICK_REDUCE_QUANTIZATION` | 42.4% of the prefill chunk after the patches | **Closed, not merely unprofitable.** Link-bound. The only levers are fewer bytes (INT4 quick-reduce, which overrides the frozen container env) or fewer reductions (impossible at TP=2). Do not re-profile this. |
| patch 006 key M=8 | +1.28% against a 0.7–0.9% within-run repeatability control | not measured separately | **Deliberately shipped as an exact no-op.** A 1.4× margin over a microbenchmark's own repeatability is not a result. |
| full-attention prefill as a standalone arm | ~2.6% of benchmark wall time | ~0.6% at a 30% kernel win | Below run 2's floor; folded into 003 instead of pursued alone. |
| INT4 quick-reduce override | would roughly halve collective wire time, est. 3–6% end to end | not run | Not pursued: it overrides a frozen container env var and changes collective numerics. Flagged as real headroom for a run whose contract allows it. |

And the flag-level levers, already settled from the reference session — re-testing these is waste:

| lever | result |
| --- | --- |
| `SGLANG_USE_AITER=1` (RMSNorm fusion only) | **−7.61%** |
| NEXTN MTP speculative decode | **−44.38%** — architectural miss, not a tuning problem |
| `mixed-chunk` with explicit prefill size | **−15.19%** |
| scheduler overhead / poll combo | **−7.60%** |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | no lever exists — decode traffic is ~0.34 MiB, below any QuickReduce threshold |
| `--moe-runner-backend` variants, `fp8-gemm-runner-aiter`, `ep-size-2`, `torch-compile-decode-bs64` | all failed warmup outright |

**One methodological failure is worth more than any of these.** Patch 002's manifest claimed that
shipping a `_down` config file would enable an untested TMA path. That was wrong — `USE_TMA` must
also be set — and because the claim went unchecked, roughly 1.1% sat unclaimed for an entire run
before patch 006 collected it. A manifest is a claim about a mechanism, and an unverified one costs
exactly as much as a wrong measurement.

## Reproducing this

Budget one hour on a node you have verified idle, not one day.

1. Fingerprint against the table above. `gfx950`, 256 CU, two GPUs, TP=2, bf16, Triton 3.7.0.
2. Measure your own baseline and your own restart floor across **at least three** server instances
   before applying anything. If your floor is worse than about 1%, patches 004, 005 and 006 are not
   measurable for you and you should fix the node before spending time on kernels.
3. Deploy all six, restart, and run the engagement checks. All of them, counting.
4. Expect **≈3642 tok/s against a ≈2917 baseline**, TPOT near 14.4 ms, and gsm8k ≈0.939 strict.
5. Interleave arms across restarts if you re-measure the delta. Two benchmarks per instance.

## When this entry stops applying

Every one of these fails **silently**, by falling back to the heuristic or the stock launch:

- **TP ≠ 2.** Changes the MoE `N` in the filename (352 at TP=2, 704 at TP=1) and every sharded N and
  K. The tuned tables become unreachable.
- **Triton ≠ 3.7.0.** The config directory component changes, and the version-fallback loop retries
  *the same device-specific filename* in the other directories rather than borrowing a donor, so it
  can never fall back to a working file.
- **A different device name or CU count.** `device_name=AMD_Instinct_MI355X` is a literal in the
  filename.
- **Arch ≠ gfx950.** Patches 001 and 003 gate on `is_gfx95_supported()` and simply skip.
- **`--chunked-prefill-size` ≠ 16384**, which moves the MoE M keys off 8192/16384, or **concurrency
  ≠ 64**, which moves the decode M off 64.
- **ISL or OSL changes**, which moves the prefill/decode mix the launch tables were tuned against and
  invalidates the sliding-window loop bound's assumptions.

When they do occur, what remains reusable is the **shape list** (M = 8, 64, 8192, 16384), the
**target ranking** (fused MoE first, then extend attention, then decode attention, then `moe_align`),
the two config generators under `artifacts/`, and the gfx95 launch-table audit pattern — the finding
that SGLang shipped no gfx95 entries for either attention kernel is the reason two of these patches
are each worth more than 7%.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/gemma_26b_tuning/`. Baseline transcribed from
the reference run's `baseline_config.with_envs.yaml` and verified against the resolved ServerArgs in
its `server.log` rather than assumed.

`FINDINGS.md` sections 0–3 are run 2 on `crsuse2-m2m-110` and remain attributed to that machine;
section 4 is run 3 on `crsuse2-m2m-052`, which is where every number in this entry comes from, and
nothing is pooled across runs. Per-patch measurements and apply notes are in
`patches/NNN_*/RESULT.md` and `MANIFEST.json`. The measurement harness is `analysis/ab_leg.sh` (arm
setup, rebuild, two-sided engagement gate, two benchmarks) with `analysis/bench_logged.sh` recording
node-wide GPU state for the duration of every leg. Raw output for run 3 only is in `results/` and
`eval_results/`.
