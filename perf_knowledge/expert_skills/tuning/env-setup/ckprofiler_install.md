# ckProfiler

The CK bench client. Enumerates Composable Kernel's compiled instances for an op and races
them, so it answers "which CK instance is fastest for this shape" without you writing a
harness.

## Install

It is an apt package. Both target images already have the ROCm repo configured, so:

```bash
apt-get update && apt-get install -y composablekernel-ckprofiler
/opt/rocm/bin/ckProfiler          # no args -> prints the op list
```

~1 minute. Do not build CK from source for this — a source build is hours and produces the
same binary. (The binary is large, ~2.5 GB, because it statically contains every compiled
instance; that is expected, not a broken package.)

Take whatever version the repo offers rather than pinning: it is built against the image's
ROCm and the candidate differs per image (`1.2.0.70202` on the vllm image, `1.2.0.70200` on
sglang). A ckProfiler from another ROCm generation may enumerate instances the installed
`libck` cannot dispatch.

## Shape of the interface

```
ckProfiler <op> <datatype> <layout> <verify> <init> <log> <time> <shape args...>
```

Positional and undocumented in `--help` style — run `ckProfiler <op>` with no further
arguments and it prints its own argument legend. Do that first for any op you have not used
before rather than guessing; the legend is authoritative for your build, this document is
not. `ckProfiler` with no args at all lists every op (`gemm`, `batched_gemm`,
`grouped_gemm`, `gemm_add_relu`, conv variants, …).

Two arguments matter for tuning discipline:

- **verify** — turn it on. It checks the instance against a reference before you trust its
  timing. An instance that is fast because it computes the wrong thing is a real failure
  mode, and this is the cheapest place to catch it.
- **time** — turn it on, or you get no timings at all.

## Reading the result

The output is one row per instance with a timing, then a best-instance line. The instance
name is the artifact: it encodes the tile shape, block size, and pipeline of the winning
configuration. That name — not the raw milliseconds — is what you carry forward, because it
is what identifies the configuration to CK.

Apply `../tuning-core/measurement.md` to the numbers before believing them. In particular
ckProfiler reports its own timing, so cross-check the winner against an independent
measurement of the same shape (`../benchmark/run_case.py`) before deploying. Two tools
disagreeing by more than the noise floor means one of them is measuring something you did
not intend — usually a layout difference.

## Where this fits

ckProfiler tells you which CK instance wins. It does **not** change what your framework
runs. Frameworks reach CK through aiter's per-op CK tuners, which have their own tuning
entry points and write their own config CSVs; that is the path that actually changes
dispatch. Use ckProfiler to:

- establish the ceiling a CK path could reach for a shape, before investing in wiring it up;
- decide whether the CK backend is even competitive for a shape versus hipBLASLt or Triton;
- sanity-check a suspicious result from a higher-level tuner.

For deploying a CK-backed change into a live framework, see `../tuning-aiter/`. As always,
finish with `../tuning-core/engagement_verification.md` — a fast ckProfiler number proves a
kernel exists, not that your workload calls it.
