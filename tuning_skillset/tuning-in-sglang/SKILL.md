---
name: tuning-in-sglang
description: Tune ops inside an SGLang serving deployment on AMD Instinct — capture real shapes, tune the Triton fused-MoE path into the correct triton_<version> config directory, deploy via SGLANG_MOE_CONFIG_DIR, and read the server log to distinguish an exact config hit from a silently degraded version fallback. Use when tuning for a live SGLang workload.
---

# Tuning inside SGLang

Read `../tuning-core/SKILL.md` first, and set the environment up per `../env-setup/`.

SGLang's tuning surface is close to vLLM's — capture real shapes, tune, deploy a config, prove
engagement — with one structural addition that changes both where artifacts go and what
"engaged" means: **configs are keyed by Triton version as well as by device.**

**Before measuring anything on the decode path, read
`../tuning-core/graph_captured_benchmarking.md`.** SGLang runs decode under captured HIP graphs
by default, which has two consequences that invalidate the obvious harness: a kernel or config
change takes effect **only at capture time**, i.e. only across a server restart, and a kernel
timed alone in eager mode carries launch overhead it never pays inside a ~790-kernel graph.
Because every candidate needs a restart, your noise floor is the restart-to-restart spread —
measured at 0.36% here against 0.014% within a single process, a 26× difference
(`../tuning-core/measurement.md` Rule 3b). A harness that misses this reports wins that are
really drift.

Measured on gfx942 / MI300X in `primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix` and on
gfx950 / MI355X in `primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix` — SGLang 0.5.12,
Triton 3.6.0, ROCm 7.2.0 in both. `tuning_workspace/verify_sglang_950.py` reproduces every
claim below against a live image.

## 1. Configs are keyed by Triton version, and the shipped ones do not match

The MoE config path has an extra directory level:

```
.../moe/moe_runner/triton_utils/configs/triton_<version>/E={E},N={N},device_name={name}[,dtype=…][,block_shape=…].json
```

Because — per the comment in SGLang's own lookup code — a config tuned under one Triton
version can produce *negative* gains under another. The version is part of the key for the
same reason the architecture is.

What actually ships, against an installed Triton of **3.6.0** in both images:

| config dir | mi30x files | MI300X files | mi35x files | MI355X files |
| --- | --- | --- | --- | --- |
| `triton_3_1_0` | 127 | **7** | 127 | **0** |
| `triton_3_2_0` | 35 | 0 | 35 | 0 |
| `triton_3_3_0` / `3_3_1` | 1 / 21 | 0 | 1 / 21 | 0 |
| `triton_3_4_0` | 33 | 0 | 33 | 0 |
| `triton_3_5_1` | 103 | 0 | 103 | 0 |
| `triton_3_6_0` | 8 | 0 | 8 | 0 |
| **total** | 322 | 7 | **328** | **0** |

Two separate problems, one per column.

**On MI300X, every matching config is in `triton_3_1_0` — five minor versions behind the
installed Triton.** They still get used, via a fallback that walks version directories
newest-first, but by the codebase's own reasoning that is the situation the versioned layout
exists to flag. Re-tuning into `triton_3_6_0` is concrete headroom, not speculative.

**On MI355X there is nothing to fall back to.** All 328 files are present, none of them names
this device — the only AMD names anywhere under `configs/` are `AMD_Instinct_MI325X` (15) and
`AMD_Instinct_MI300X` (7). So every fused-MoE lookup on MI355 in this image reaches the third
case below and runs the default config. Verified directly:

```
Using default MoE kernel config. Performance might be sub-optimal! Config file not found at
.../configs/triton_3_6_0/E=256,N=256,device_name=AMD_Instinct_MI355X,dtype=fp8_w8a8,block_shape=[128, 128].json
```

That makes MoE tuning the single highest-value item for sglang on MI355, and it is starting from
zero rather than from a stale baseline. Note also the generated filename contains a **literal
space** in `block_shape=[128, 128]` — only `device_name` gets its spaces stripped. Generate the
name with `get_config_file_name()` rather than by hand.

## 2. Read the log — a hit and a fallback are different outcomes

SGLang distinguishes three cases, and the distinction is the whole point of §1. All three were
reproduced on gfx950 by planting files in a scratch directory and reading the emitted log:

Exact match (`logger.info`):

```
Using MoE kernel config from .../configs/triton_3_6_0/E=256,N=256,device_name=AMD_Instinct_MI355X,dtype=fp8_w8a8,block_shape=[128, 128].json.
```

Version fallback (`logger.warning`):

```
Config file not found at .../triton_3_6_0/<file>. Fallback to triton version 3.1.0 and use
MoE kernel config from .../triton_3_1_0/<file>. Performance might be sub-optimal!
```

No config at all — defaults, and you are leaving the most on the table. This is the current
state of every MoE layer on MI355 in this image (§1).

A fourth case exists in this version and is not in the docs: with `down_moe=True` and no
`…_down.json` present, it logs `Using MoE kernel config with down_moe=False`, which is a partial
hit — the up-projection config gets reused for the down-projection. Treat it as a miss on the
down side; `tuning_fused_moe_triton_sep.py` (§5) is what produces the missing half.

```bash
python3 -m sglang.launch_server ... 2>&1 | grep -E "Using MoE kernel config from|Fallback to triton version|sub-optimal"
```

**Treat the fallback warning as a miss, not a hit.** It is the failure mode that looks most
like success: a config *was* loaded, the server runs, performance is plausible. A naive check
for "did it find a config" passes. Only the version in the path tells you whether the config
was tuned for the compiler you are running.

## 3. The device name

```python
from sglang.srt.utils import get_device_name
get_device_name()      # -> 'AMD Instinct MI300X' / 'AMD Instinct MI355X'
```

Without GPUs passed in (`--device=/dev/kfd --device=/dev/dri --group-add video`) this returns
`None` and the filename becomes `device_name=None.json`. If you see `None` or an empty string,
fix the container before doing anything else (`../env-setup/` §2).

**The two frameworks report different names for the same silicon.** Measured on one physical
MI355 node, same GPU, two containers:

| source | reports |
| --- | --- |
| sglang `get_device_name()` (mi35x image) | `AMD Instinct MI355X` |
| torch `get_device_name()` (mi35x image) | `AMD Instinct MI355X` |
| vLLM `current_platform.get_device_name()` (vllm image) | `AMD Instinct MI355 OAM` |
| torch `get_device_name()` (vllm image) | `''` — empty string |

Consequences worth being explicit about:

- **A tuned MoE config does not transfer between the two frameworks**, even on the same node with
  the same Triton version, because the device name is in the filename. Tune twice, or rename.
- **`torch.cuda.get_device_name()` works here and is empty in the vllm image.** So a helper
  script that builds filenames from torch is correct in sglang and silently wrong in vLLM — which
  is exactly how that bug survives. Always use the framework's own helper.
- The vLLM image ships two `MI355_OAM` configs and one `MI355X` config; the `MI355X` one is
  unreachable from vLLM and would be the reachable name from sglang, where it does not exist.
  The same silicon has configs stranded on both sides of the name split.

Then check your filename against one that ships:

```bash
ls .../triton_utils/configs/triton_*/ | grep MI3 | head
```

## 4. Capture real shapes

```bash
export HIPBLASLT_LOG_MASK=32
python3 -m sglang.launch_server --model-path <model> ... 2> hipblaslt.log
# drive representative traffic, then rank by frequency:
grep -o '\-\-api_method.*' hipblaslt.log | sort | uniq -c | sort -rn | head -20
```

Replay captured commands verbatim — M and N appear swapped relative to the torch call
(`../tuning-hipblaslt/` §1). For the aiter path, `AITER_LOG_TUNED_CONFIG=1` inventories which
GEMM shapes have no tuned config.

Tune in call-count order. `../benchmark/shapes.py` is the fallback when you cannot run the
real model.

## 5. Tune

**Fused MoE** — SGLang ships its tuner:

```bash
python3 /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model <model> --tune
```

(`tuning_fused_moe_triton_sep.py` alongside it tunes the up/down projections separately, and
is what produces the `_down.json` whose absence causes the partial hit in §2.) Output is the
JSON-per-M-bucket format; write it into the `triton_<installed version>` directory so it is an
exact hit rather than a fallback. On MI355 this is the highest-value work in this skill,
because §1 shows there is nothing shipped to fall back to.

**Dense GEMM** goes through aiter, and the sglang image is the *easy* case: aiter is a git
checkout at `/sgl-workspace/aiter`, so `gradlib/gradlib/gemm_tuner.py` and the seven per-op CK
tuners are already present — no source to fetch (`../env-setup/image_tool_matrix.md`). Run and
deploy per `../tuning-aiter/`, including the `/tmp/aiter_configs` trap.

All seven CK tuners were exercised on gfx950 in this image; results, the two real wins, and
three traps — a tuner that installs regressions, a shipped shape list that aborts on this part,
and an environment-variable name that silently does nothing — are in `../tuning-ck/` §3 and
§3b. Read that before running one, not after.

The authoring-time knobs behind those MoE configs — `num_warps`, `waves_per_eu`, block sizes
— are covered in `../tuning-triton/`. Worth knowing which of them actually move the number
before spending tuning time on a wide space.

## 6. Deploy without editing the package

```bash
export SGLANG_MOE_CONFIG_DIR=/work/sglang_tuned
# expects: $SGLANG_MOE_CONFIG_DIR/configs/triton_3_6_0/<filename>.json
```

Note the layout: SGLang appends `configs/triton_<version>/` **beneath** the directory you
name. Two things about getting that wrong, both measured on gfx950:

**Pointing the variable at the version directory raises, it does not miss.** With
`SGLANG_MOE_CONFIG_DIR` set to the `triton_3_6_0` directory itself, the exact-path check fails,
and the fallback scan then calls `os.listdir(<dir>/configs)` unguarded:

```
FileNotFoundError: [Errno 2] No such file or directory: '/work/sglang_tuned/configs'
```

That is an unhandled exception at the first MoE layer construction, so it surfaces as a server
that fails to start rather than as a quiet performance loss. Of the two failure modes this is
the good one — but do not expect a warning.

**The override replaces the package directory, it does not extend it.** `config_dir` is used as
the root for both the exact lookup *and* the newest-first fallback scan, so with the variable
set, none of the 328 shipped files are on any search path. On MI355 that costs nothing, since
none of them match this device anyway. On MI300X it means a directory holding one tuned config
turns the other six shipped MI300X configs into misses. If you deploy by env var there, copy the
shipped tree in first:

```bash
cp -r .../triton_utils/configs "$SGLANG_MOE_CONFIG_DIR"/
```

This is the opposite of vLLM's behaviour, which checks the override *then* the package directory
and names both in its miss warning (`../tuning-in-vllm/` §4). Same intent, two different
semantics; do not carry an assumption across.

Using the env var still keeps tuned artifacts separate from what shipped, survives image updates,
and makes A/B a matter of unsetting one variable.

## 6b. One switch that discards all of it

```python
if get_global_server_args().enable_deterministic_inference:
    logger.warning("Deterministic inference is enabled, using default MoE kernel config.")
    return None
```

`get_moe_configs` returns before it looks at any file. Verified on gfx950: with a valid
exact-hit config in place, `enable_deterministic_inference=True` loads nothing and logs only the
line above. If a reproducibility flag gets turned on later in the deployment's life, the MoE
tuning silently stops applying — and the warning does not mention configs going unused in a way
that reads as a problem. Check that flag before investigating a regression in tuned MoE
performance.

Related, for anyone scripting against this: `get_moe_configs` reads `get_global_server_args()`,
which is only populated inside a scheduler or tokenizer process. Calling it from a plain script
to check your filename raises `ValueError: Global server args is not set yet!` — stand one up
with `set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))` first.

## 7. Verify, then measure end to end

Order matters — engagement first, timing second:

1. Startup log shows `Using MoE kernel config from …/triton_3_6_0/…` — an exact path, no
   fallback warning.
2. aiter path: `AITER_LOG_TUNED_CONFIG=1 … | grep -c "is tuned on cu_num"` matches the number
   of tuned shapes your traffic hits.
3. Backend-agnostic confirmation of what actually ran:
   `rocprofv3 --kernel-trace --stats -f csv -d ./prof -- <short workload>` (`../tuning-hip/` §4).

Then benchmark serving, A/B by toggling `SGLANG_MOE_CONFIG_DIR` with everything else fixed.
Serving metrics are noisier than microbenchmarks, so the repeat-and-spread rule matters more
here, not less (`../tuning-core/measurement.md`).

## Checklist

- [ ] GPUs passed in; `get_device_name()` returns a real name, not `None`
- [ ] filename generated by `get_config_file_name()`, not by hand (it contains a space)
- [ ] installed Triton version identified; target config dir is `triton_<that version>`
- [ ] shipped configs checked for this device's name — on MI355X there are none
- [ ] shapes captured from real traffic, ranked by call count
- [ ] MoE tuned with `tuning_fused_moe_triton.py`; dense GEMM via aiter
- [ ] deployed via `SGLANG_MOE_CONFIG_DIR` pointing at the parent of `configs/`, with the
      shipped tree copied in — the override replaces the package dir, it does not extend it
- [ ] `enable_deterministic_inference` off, or none of this applies
- [ ] log shows an exact hit — a version-fallback warning counts as a miss
- [ ] end-to-end A/B with repeats, after engagement is proven
