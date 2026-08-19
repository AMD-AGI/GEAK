# Gemma-4-26B-A4B-it on MI355X — bf16 serving, TP=2

**Status: no verified win recorded yet.** The environment, baseline, and roofline are documented
below so a run on this stack starts from a known reference point instead of re-deriving it. Fill in
the result sections when a win has been reproduced from an artifact.

Nothing here should be deployed — there is nothing to deploy. Use it as the fingerprint to check
against, and as the record of what the baseline is.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | MI355X, `gfx950`, **256 CU**, **two GPUs** | **yes** |
| container | `rocm/sgl-dev@sha256:95a933896aeab2a431521ece6ebe90c1db37a3aaf1e32a938d56ef7ccf6603a5` | descriptive |
| Triton | **3.7.0** in this image; the reference run ran **3.6.0** | **yes** — see the baseline note |
| model | Gemma-4-26B-A4B-it, **TP=2** | **yes** — sharding sets every N and K |
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

**No `SGLANG_*` or aiter environment variables, and no config deltas.** Unlike the Qwen3-8B entry,
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

## Headroom

Memory roofline puts the decode ceiling at **7224.6 tok/s** against 3065.1 achieved — **42.4%**
utilization, i.e. roughly 2.3× nominal headroom. That is a bound, not a forecast, and the gap
includes everything the roofline ignores: collective communication across the TP=2 pair, kernel
launch structure, and the prefill share of the run.

## The MoE Triton config is untuned here — the largest documented unexploited lever

**No tuned Triton fused-MoE config exists for this GPU, so all 30 layers run on the heuristic
default.** The upstream static analysis enumerated all 7 shipped `triton_*` config directories
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

## To fill in when a win is verified

Sections this entry is missing, in the order `../ENTRY_TEMPLATE.md` expects them: **Deploy**,
**Engagement check**, **Accuracy gate result**, **What was tried and did not work**, **When this
entry stops applying**. Add the deployable artifact under `artifacts/` and only then update the
index table in `../README.md`.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/gemma_26b_tuning/`. Baseline transcribed from
the reference run's `baseline_config.with_envs.yaml` and verified against the resolved ServerArgs in
its `server.log` rather than assumed.
