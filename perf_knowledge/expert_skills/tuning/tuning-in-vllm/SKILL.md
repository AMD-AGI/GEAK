---
name: tuning-in-vllm
description: Tune ops inside a vLLM serving deployment on AMD Instinct — capture the shapes the model actually issues, tune fused-MoE and GEMM paths, deploy configs where vLLM reads them via VLLM_TUNED_CONFIG_FOLDER, and confirm from the server log that the tuned config was loaded. Use when tuning for a live vLLM workload.
---

# Tuning inside vLLM

Read `../tuning-core/SKILL.md` first, and set the environment up per `../env-setup/`.

Tuning inside a serving framework differs from tuning a kernel in one respect that governs
everything else: **you do not choose the shapes, the model and the traffic do.** Your job is
to find out what is actually being issued, tune those, and prove the framework picked up the
result.

Measured on `primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix` — vLLM 0.21.0, ROCm 7.2.2 —
on gfx942 / MI300X and on gfx950 / MI355. The same image serves both parts per
`docker_select.json`. `tuning_workspace/verify_vllm_950.py` reproduces every claim below against
a live image.

## 1. The device-name trap — read this before anything else

vLLM's tuned configs are files named by device. In this image, on both parts:

```python
torch.cuda.get_device_name()                       # -> ''   <- empty string
vllm.platforms.current_platform.get_device_name()  # -> 'AMD Instinct MI300X'
                                                   #    'AMD Instinct MI355 OAM'
```

**`torch.cuda.get_device_name()` returns an empty string here.** vLLM itself uses
`current_platform.get_device_name()` (see `get_config_file_name` in
`vllm/model_executor/layers/fused_moe/fused_moe.py`), which produces the real name.

Anyone who builds a config filename from the torch call gets
`E=128,N=1024,device_name=.json`, which matches nothing. Verified on gfx950 — the file is
written, the lookup misses, and the only sign is the startup warning in §5. Always take the
device name from `current_platform`, and verify your filename against one that ships:

```bash
ls .../vllm/model_executor/layers/fused_moe/configs/ | grep MI355
# E=160,N=384,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json
```

Two further reasons not to hand-build the name. **`gcnArchName` does not appear in it** — that
reads `gfx950:sramecc+:xnack-` and is not what the lookup uses, so an arch-based filename misses
too. And **the same silicon answers to different names in different frameworks**: sglang reports
`AMD Instinct MI355X` for the identical GPU (`../tuning-in-sglang/` §3), so configs do not
transfer between the two frameworks.

How much of the shipped surface is actually yours, measured on gfx950:

| | gfx942 / MI300X | gfx950 / MI355 |
| --- | --- | --- |
| total configs shipped | 317 | 316 |
| matching this device | 34 | **2** |

The two on gfx950 are `E=160,N=384,…,dtype=fp8_w8a8` and `E=384,N=256,…,dtype=int4_w4a16`.
Presence of a large `configs/` directory says nothing about your device being covered — on MI355
it covers two shapes.

Worse, the MI35x configs that *do* exist are split across four names, and only one of them is
reachable:

| device_name in filename | configs | reachable from this device |
| --- | --- | --- |
| `AMD_Instinct_MI355_OAM` | 2 | **yes** |
| `AMD_Instinct_MI355X` | 1 | no |
| `AMD_Instinct_MI350_OAM` | 3 | no |
| `AMD_Instinct_MI350X` | 1 | no |

Seven MI35x-family configs ship; two load. One of the unreachable ones — `E=160,N=192,fp8_w8a8`
for `MI350_OAM` — covers a shape this device has no config for at all. Before tuning that shape
from scratch, try renaming that file: it is a near-neighbour part and a copy costs nothing to
test against a real tune.

## 2. Capture the shapes the model actually issues

Do not tune a generated sweep if you can tune the real thing.

```bash
export HIPBLASLT_LOG_MASK=32
vllm serve <model> ... 2> hipblaslt.log
# drive representative traffic, then:
grep -o '\-\-api_method.*' hipblaslt.log | sort | uniq -c | sort -rn | head -20
```

Each line is a complete, runnable `hipblaslt-bench` command; the counts rank them by
importance. Tune in that order. Replay the captured command **verbatim** — a torch matmul of
`(512,1024)×(1024,2048)` logs as `-m 2048 -n 512 -k 1024`, with M and N swapped relative to
the torch call, so hand-reconstruction tunes a different problem (`../tuning-hipblaslt/` §1).

For the aiter path, `AITER_LOG_TUNED_CONFIG=1` reports every GEMM's shape and whether a tuned
config was found — which doubles as your untuned-shape inventory:

```
[aiter] shape is M:4096, N:4096, K:4096 ... not found tuned config ... will use default config!
```

Generated shapes (`../benchmark/shapes.py`) remain the fallback when you cannot run the real
model, and the smoke test for your harness.

## 3. Two tuning surfaces

**Dense GEMM** goes through aiter and hipBLASLt. Tune with the aiter gradlib tuner across
libtypes, deploy into aiter's package config directory — including the `/tmp/aiter_configs`
trap, which is the single most likely way to spend a day producing nothing
(`../tuning-aiter/` §5).

**Fused MoE** is vLLM's own Triton path with its own config format — JSON keyed by M bucket:

```json
{"1":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2, "waves_per_eu": 0}, ...}
```

A shipped MI300X file carries 18 buckets: `1, 2, 4, 8, 16, 24, 32, 48, … 2048, 3072, 4096`.
That ladder is worth studying as a model for your own M-bucketing: dense near the decode end
where behaviour changes fastest, sparse toward prefill where it is stable. And note the M=1
entry uses `BLOCK_SIZE_M=16` with `num_warps=2` — the decode corner of the space, nothing like
what the 4096 bucket wants (`../tuning-core/search_strategy.md`).

Tune it with the benchmark script shipped in the image:

```bash
python3 /app/vllm/benchmarks/kernels/benchmark_moe.py --model <model> --tune
```

Note `waves_per_eu` in that schema — the AMD occupancy knob, exposed as a first-class field.
Its authoring-time behaviour is covered in `../tuning-triton/`.

## 4. Deploy where vLLM looks

vLLM checks two locations, in this order:

1. `$VLLM_TUNED_CONFIG_FOLDER/<filename>` — **user-defined, takes priority**
2. the package's own `fused_moe/configs/<filename>`

```bash
export VLLM_TUNED_CONFIG_FOLDER=/work/tuned_configs
```

Both are searched, so setting the variable adds a directory rather than replacing the shipped
one — the miss warning in §5 names both paths it tried. Worth stating because **sglang's
equivalent variable does the opposite**: it replaces the package root for both the exact lookup
and the version fallback, so a sparse override directory turns shipped configs into misses
(`../tuning-in-sglang/` §6). Do not carry the assumption across.

Prefer the env var over editing the package directory. It survives image updates, keeps your
work separable from what shipped, and makes A/B testing a matter of unsetting one variable.
The filename must match exactly — see §1.

## 5. Prove engagement from the server log

vLLM states plainly which path it took. Both lines below were reproduced on gfx950 by planting a
config in a scratch directory and reading what `get_moe_configs` logged. On a hit:

```
INFO [fused_moe.py:1060] Using configuration from /work/tuned_configs/E=128,N=1024,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json for MoE layer.
```

On a miss — here caused by naming the file from `torch.cuda.get_device_name()`, i.e. the §1 trap:

```
WARNING [fused_moe.py:1073] Using default MoE config. Performance might be sub-optimal! Config file
not found at /work/tuned_configs/E=128,N=1024,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json,
/usr/local/.../fused_moe/configs/E=128,N=1024,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json
```

Note what the warning prints: the filename vLLM *wanted*, in both locations. Diff that against
the filename you wrote and the mismatch is immediate — this is the fastest way to debug a
misnamed config, faster than re-deriving the device name.

That warning names every path it tried, which tells you exactly how your filename differs
from what it wanted. Grep for both at startup:

```bash
vllm serve ... 2>&1 | grep -E "Using configuration from|Using default MoE config"
```

Note these are `info_once`/`warning_once` — they appear once per config, not per request, so
look at startup rather than steady state.

For the aiter path, the corresponding proof is the log transition described in
`../tuning-aiter/` §2:

```bash
AITER_LOG_TUNED_CONFIG=1 vllm serve ... 2>&1 | grep -c "is tuned on cu_num"
```

Backend-agnostic fallback when neither is available — check what actually ran:

```bash
rocprofv3 --kernel-trace --stats -f csv -d ./prof -- <short workload>
```

The kernel names in `*_kernel_stats.csv` are ground truth (`../tuning-hip/` §4).

## 6. Only then measure end to end

Once engagement is proven, measure serving metrics rather than kernel times:

```bash
vllm bench serve --model <model> ...      # throughput / TTFT / ITL
vllm bench latency --model <model> ...    # single-batch latency
```

A kernel-level win does not automatically become a serving win — it can be diluted by other
bottlenecks, or concentrated in a phase your traffic barely exercises. Run the A/B by
toggling `VLLM_TUNED_CONFIG_FOLDER`, keeping everything else fixed, and apply the same
repeat-and-spread discipline as at kernel level: serving benchmarks are noisier, not less
noisy, than microbenchmarks.

## Checklist

- [ ] device name from `current_platform`, never `torch.cuda.get_device_name()` or `gcnArchName`
- [ ] filename verified against a shipped config for this device — on MI355 only 2 match
- [ ] near-neighbour configs (`MI350_OAM`, `MI355X`) checked before tuning a shape from scratch
- [ ] shapes captured from real traffic and ranked by call count
- [ ] dense GEMM tuned via aiter; MoE via `benchmark_moe.py --tune`
- [ ] deployed via `VLLM_TUNED_CONFIG_FOLDER`, not by editing the package
- [ ] startup log shows `Using configuration from …`, not `Using default MoE config`
- [ ] end-to-end A/B by toggling the env var, with repeats
