---
name: env-setup
description: Get a container ready to tune GPU ops on AMD Instinct — audit which tuning tools are present, install the missing bench clients (hipblaslt-bench, ckProfiler, aiter tuners) at versions that match the runtime, and avoid the version-skew failures that silently produce useless tuned artifacts. Use before any tuning run in a vllm or sglang image.
---

# Environment setup for tuning

Serving images ship what is needed to **run** tuned kernels, not to **produce** them. The
tuned config files are there; the tools that write them are not. This skill closes that gap.

Budget: ~1 minute for ckProfiler, ~2 for aiter source, ~20 for hipblaslt-bench. Install only
what your task needs.

## 1. Audit first

Never assume. Run the inventory in a **pristine** container from the image you will tune in:

```bash
docker run --rm --entrypoint bash \
  -v $PWD/audit_tools.sh:/tmp/a.sh:ro \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  <image> /tmp/a.sh
```

`audit_tools.sh` prints `OK` / `MISSING` / `INFO` per tool and changes nothing. Grep for
`MISSING` to get your work list.

Two things that will bite you here:

- **`--entrypoint bash`** — the vllm image's entrypoint is the `vllm` CLI. Without the
  override your script is parsed as vllm arguments and you get
  `vllm: error: unrecognized arguments`.
- **A container you have already worked in is not evidence.** It carries your earlier
  installs. Audit a fresh one.

`image_tool_matrix.md` records what the two target images shipped when measured, and which
image differences change your workflow.

## 2. Pass the GPUs in

```bash
--device=/dev/kfd --device=/dev/dri --group-add video
```

Without these, `rocminfo` fails and arch detection fails with it. The symptom surfaces far
from the cause: aiter raises `ImportError: cannot import name 'dtypes' from 'aiter'`, which
reads like a broken install but is arch detection failing and disabling the CK/HIP op
registrations. **When a GPU library throws a strange import error, run `rocminfo` before
debugging the library.**

## 3. Install what is missing

| tool | how | cost | needed for |
| --- | --- | --- | --- |
| ckProfiler | `apt-get install composablekernel-ckprofiler` | ~1 min | CK instance selection → `ckprofiler_install.md` |
| aiter source | `git clone` + checkout the tag matching the installed wheel | ~2 min | aiter tuners (vllm image only) |
| hipblaslt-bench | source build, clients-only | ~20 min | hipBLASLt solution racing → `hipblaslt_bench_build.md` |

Do them in that order and stop when your task is covered. Only `hipblaslt-bench` requires a
build, and only because no package ships it; the clients-only path avoids regenerating
kernels and turns a multi-hour build into a coffee break. `hipblaslt_bench_build.md` has the
recipe plus each dependency mapped to the distinct stage it fails at — the build fails
three separate times on three separate missing packages, one of them at 98%.

## 4. Match versions to the runtime, not to HEAD

This is the rule that separates a tuning run that helps from one that appears to.

| tool | match it to | how to read it |
| --- | --- | --- |
| hipBLASLt clients | the container's ROCm | `cat /opt/rocm/.info/version` → use branch `release/rocm-rel-7.2` for 7.2.x |
| ckProfiler | the ROCm apt repo already in the image | take the candidate, do not pin across ROCm versions |
| aiter tuners | the installed aiter | `python3 -c "import aiter._version as v; print(v.__version__)"`, then resolve to a tag with `git ls-remote --tags origin` |

A tuned artifact is a lookup-table entry keyed by shape, dtype and architecture. Produced by
a mismatched tool, it lands where the runtime never looks, or is looked up and rejected.
**Neither failure raises an error.** You get baseline performance and a report claiming a
win. This is why every tuning run in this skillset ends with
`../tuning-core/engagement_verification.md` rather than with a timing.

Corollary: never carry a tuned artifact between architectures. gfx942 and gfx950 differ in
CU count and FP8 dialect; the lookup keys include the arch for a reason.

### The version string is not the compiler

Same architecture, same GPU, same kernel source, same config, same reported library version —
and 1.89x apart. Measured on one MI355X, aiter's Gluon MXFP4 GEMM at M512 N4096 K4096, both
images reporting `triton 3.6.0`:

| | vllm image | sglang image |
| --- | --- | --- |
| median | **0.0616 ms** | **0.1163 ms** |
| triton path | `dist-packages/triton` | `/sgl-workspace/triton-custom` |
| VGPRs | 254 | 256 (at the cap) |
| VGPRs spilled | **0** | **63** |
| scratch traffic | none | 55 stores + 55 reloads |
| MFMA count | 64 | 64 |
| LDS | 69 632 B | 69 632 B |

Identical instruction mix and identical LDS. The whole difference is that one compiler build
lands two registers over the 256-VGPR cap and spills 63 values into the inner loop. Both
report `3.6.0` because a vendored fork keeps the version string of the release it forked
from.

Three things follow, in increasing order of how much they cost:

- **Read `triton.__file__`, not `triton.__version__`,** when recording what produced a result.
  The version answers a question nobody was asking.
- **Engagement verification will not catch this.** The config is found, the lookup hits, the
  intended kernel runs. Every check in `../tuning-core/engagement_verification.md` passes.
  The kernel is simply compiled worse, and nothing in the Python layer can see it. What sees
  it is `.vgpr_spill_count` in the generated `.amdgcn` — dump it with `TRITON_CACHE_DIR=/tmp/tc`
  and read the metadata, which is worth doing once per image for any kernel you care about.
- **Tuning does not fully recover it.** Sweeping the sglang image lifted that shape from
  0.1163 to 0.0822 ms — a real 1.4x — and it is still slower than the vllm image *untuned*.
  The compiler build sets a ceiling the config space cannot reach past. If you are comparing
  a tuned result in one image against a baseline in another, you are measuring the images.

## 5. Confirm before tuning

Re-run `audit_tools.sh`. Everything your task needs should read `OK`, and `gpu-arch` should
name a real target (`gfx942`), not a `rocminfo failed` message. Then pin an idle GPU
(`rocm-smi --showuse`; `export HIP_VISIBLE_DEVICES=<n>`) and take a baseline with
`../benchmark/run_case.py --smoke` — if the harness cannot produce a sane number on a known
shape, nothing downstream of it is trustworthy.

## Files

- `audit_tools.sh` — read-only inventory; run it first and again at the end
- `image_tool_matrix.md` — measured contents of both target images, and the aiter wheel-vs-source split
- `ckprofiler_install.md` — CK bench client: install, interface, and what it does and does not prove
- `hipblaslt_bench_build.md` — the one real build, with its three failure modes

## Next

Environment ready → `../tuning-core/SKILL.md` for the loop itself, then the skill for your
source language or serving framework.
