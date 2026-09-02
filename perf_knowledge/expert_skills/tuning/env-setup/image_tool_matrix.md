# What each image ships, and what you must add

Measured by running `audit_tools.sh` inside a **pristine** container from each image.
Do not trust a container you have already worked in — it carries your earlier installs.

```bash
docker run --rm --entrypoint bash \
  -v $PWD/audit_tools.sh:/tmp/a.sh:ro \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  <image> /tmp/a.sh
```

`--entrypoint bash` matters: the vllm image's entrypoint is the `vllm` CLI, and without the
override your script is parsed as vllm arguments (`vllm: error: unrecognized arguments`).

## The matrix

| | vllm `v0.21.0-rocm720-profilerfix` | sglang `v0.5.12-rocm720-mi30x-profilerfix` |
| --- | --- | --- |
| ROCm | 7.2.2 | 7.2.0 |
| torch | 2.10.0+git8514f05 | 2.9.1+rocm7.2.0 |
| triton | 3.6.0 | 3.6.0 |
| flydsl | 0.1.4 | 0.1.5 |
| framework | vllm 0.21.0+rocm722 | sglang 0.5.12 |
| **hipblaslt-bench** | **missing** — build | **missing** — build |
| **ckProfiler** | **missing** — apt | **missing** — apt |
| rocblas-bench | missing (rarely needed) | missing |
| torch TunableOp | present | present |
| triton autotune | present | present |
| aiter python pkg | 0.1.13 (**wheel**) | present (**source**) |
| aiter tuner scripts | **missing** | present at `/sgl-workspace/aiter` |
| aiter tuned CSVs | present | present |
| gtest/gmock headers | **missing** | present |
| boost headers | present | present |

Neither image ships a single bench client. Both ship every deploy target. That asymmetry is
the whole reason this skill exists: the images are built to *run* tuned kernels, not to
produce them.

## The aiter split — the one that actually changes your workflow

This is the difference that costs the most time if you miss it.

- **sglang** has aiter as a git checkout at `/sgl-workspace/aiter` (commit `a6bb4993`,
  2026-04-29). Tuner scripts are right there: `gradlib/gradlib/gemm_tuner.py`,
  `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`. You can tune in place.
- **vllm** has aiter as an installed wheel under
  `/usr/local/lib/python3.12/dist-packages/aiter`. `aiter/configs/*.csv` — the files a tuner
  *writes* — are present, but no `gradlib/`, no `csrc/`, no tuner. Wheels ship the runtime,
  not the build tree.

So in the vllm image you must supply the source yourself, and it must match the wheel:

```bash
pip show aiter 2>/dev/null   # often absent; fall back to:
python3 -c "import aiter._version as v; print(v.__version__)"   # -> 0.1.13
```

Check out the matching tag. For `0.1.13` that is commit `cdcfa833bdf554ca75594c90dde4316ea9b50199`:

```bash
git clone https://github.com/ROCm/aiter.git /work/aiter
cd /work/aiter && git checkout cdcfa833bdf554ca75594c90dde4316ea9b50199   # tag v0.1.13
```

Resolve the tag yourself rather than copying a hash — the mapping is
`git ls-remote --tags origin | grep v0.1.13`. The point is the *method*: read the installed
version, resolve it to a tag, check out that tag. A tuner from a different aiter generation
can emit CSV rows the installed runtime will not parse, and you get silent no-ops rather
than errors.

Do not `pip install` the checkout over the wheel. Run the tuner from the source tree and let
it write into the *installed* package's `configs/` directory — that is the path the runtime
reads. See `../tuning-aiter/` for the deploy step.

## Install order

Cheapest first; stop when you have what your task needs.

1. **ckProfiler — apt, ~1 min.** Available in both images from the already-configured ROCm
   repo (`/etc/apt/sources.list.d/rocm.list`). No build.
   ```bash
   apt-get update && apt-get install -y composablekernel-ckprofiler
   /opt/rocm/bin/ckProfiler          # prints the op list
   ```
   Candidate versions track the image's ROCm: `1.2.0.70202` on vllm, `1.2.0.70200` on
   sglang. Take the repo's candidate — do not pin across ROCm versions.

2. **aiter source — git, ~2 min.** vllm only; sglang already has it. See above.

3. **hipblaslt-bench — source build, ~20 min.** The only genuinely expensive item, and only
   if you are tuning the hipBLASLt path. Full recipe with the failure modes in
   `hipblaslt_bench_build.md`. On the vllm image, install `libgtest-dev libgmock-dev` first —
   the sglang image already has them.

## Version skew is the standing hazard

Every tool here must match the runtime it will feed, not the newest release:

| tool | match it to |
| --- | --- |
| hipBLASLt clients | the container's ROCm (`cat /opt/rocm/.info/version`) — branch `release/rocm-rel-7.2` for 7.2.x |
| ckProfiler | the ROCm apt repo already in the image |
| aiter tuners | the installed `aiter.__version__` |

A tuned artifact is a lookup-table entry keyed by shape, dtype and *architecture*. Produce
it with a mismatched tool and it will be written somewhere the runtime never looks, or
looked up and rejected. Neither failure raises an error — you just get baseline performance
and a report claiming a win. Always finish with the engagement check in
`../tuning-core/engagement_verification.md`.

## Passing GPUs in

Every container that must tune needs:

```bash
--device=/dev/kfd --device=/dev/dri --group-add video
```

Omit them and `rocminfo` fails. The downstream symptom is misleading — aiter raises
`ImportError: cannot import name 'dtypes' from 'aiter'`, which reads like a broken install
but is actually arch detection failing, disabling the CK/HIP op registrations. If you see an
odd import error from a GPU library, run `rocminfo` before debugging the library.
