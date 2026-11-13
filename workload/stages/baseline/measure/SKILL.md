---
name: measure
description: Measure the baseline on this machine — a trace replay against a running service, or a full reproduction of a named CI run from a fresh digest-pinned container with every pin proven afterwards from the run's own records. One skill, parameterised by which benchmark command runs and whether pins are declared.
---

# Measure the baseline

The alternative to `adopt`. This one really occupies the GPUs and really measures. It
exists because reusing a receipt presupposes an unchanged environment: after a new image,
new weights, a different node or changed serving parameters, the old receipt is not this
machine's baseline.

Two jobs live here and they are **one skill with parameters, not two directories**:

| Scenario config | What you do |
| --- | --- |
| `harness` absent | drive GEAK's `e2e_workflow/scripts/bench_e2e.sh` with `BENCH_CLIENT=agentx` against the already-running service at `base_url`. You do **not** start that service |
| `harness.dir` + `harness.cmd` present | drive the site's own CI harness inside a fresh container, so the thing being reproduced stays the thing being reproduced |
| `pins` absent | measure and record; there is no reproduction target |
| `pins` present | a named CI run is the target. Check every pin before spending GPU time, and **prove every pin afterwards from the run's own records** |

Everything else — the smoke gate, the cold-cache discipline, the canonical
`baseline.json`, the fact that you do not grade yourself — is the same either way.

## 1. The readiness check is not optional

Run `e2e_workflow/scripts/agentx_smoke.sh` first, with `BASE_URL` and `E2E_METRIC` set.
The failures it catches share one shape: **nothing goes wrong until the client has already
started and warmed the service**, and the error arrives twenty minutes later. The worst of
them never raises an error at all — a stock `aiperf` without `--scenario` support will
cheerfully run a *different* workload and hand back a perfectly plausible throughput
number. `aiperf` and InferenceX are on neither `$PATH` nor `/opt` on this machine, so
`inferencex_path` must be given. Any blocker is `blocked`, not a warning.

## 2. When `pins` are declared: refuse before you spend an hour

Each check below costs milliseconds; the run it guards costs an hour on eight GPUs and is
the reference every later comparison is measured against. Report **every** blocker at
once, not the first. Write the plan to `$OUT/alignment.json` before launching anything.

| Pin | Why it moves the number | Check before | Prove after |
| --- | --- | --- | --- |
| **Image digest**, never a tag | `v0.5.19-rocm720-mi35x-20260913` can be repushed, and the only evidence afterwards is throughput that drifted for no reason anybody can name | well-formed `sha256:…`, present locally, and among the tag's `RepoDigests` | `docker inspect` of the container that actually served. It is started by digest reference, so `Config.Image` carries it; `Image` is the local image id, a different hash — record it, do not compare it |
| **Data + seed** | the corpus decides the prompt-length mix, hence the prefill/decode ratio, hence most of the throughput; the seed decides replay order | the corpus exists and its entry count matches, or — for a named public-dataset `loader` — the generation is pinned. `semianalysis_cc_traces_weka_062126` and `…_061526` are different prompt-length mixes | both read back from the client's own export or recorded CLI |
| **`benchmark_id`** | the AgentX scenario builds its cache-bust marker from AIPerf's per-invocation id, so the id **is** the KV keyspace. Reuse one and the second run starts on warmed prefixes: it measures high and nothing in its output says so | refuse any id already in the ledger | read back from the run, then retire it to the ledger. `"auto"` is for a client that draws its own id (agentx-harness ≥ 1.0.12 has no way to set one) — then what must be proven is that the drawn id is *new*, not that it matches |
| **Cache-bust target** | decides how much of each prompt can be served from cache | the AgentX scenario plugin locks `first_turn_prefix`; any other value is refused | read back from the client's recorded CLI/config |
| **CPU binding** | two ranks on one core's execution resources is not the CI topology | one logical CPU per physical core taken from the machine's own `lscpu -p` — guessing that siblings are `cpu` and `cpu + ncores/2` is right on this node and wrong on the next — and only from cores this process was actually granted | `sched_getaffinity` sampled **inside** the container. A `--cpuset-cpus` the daemon silently narrowed looks identical from outside |
| **Fresh container + cold compile caches** | a warm Triton/XDG cache can hold kernels compiled from *another campaign's overlay*. This stack was once profiled with `glm52_optimized` kernels still in the Triton cache, and the only reason anybody noticed was the MLIR source locations inside the cache entries | the container name is unused, and each cache directory is created empty | the caches are **not empty**. See below |

**An empty compile cache after the run is a failure, not tidiness.** If
`TRITON_CACHE_DIR` never reaches the server process, the server compiles into its own warm
cache, the run completes normally, and every log line still says the cache was cold. On
this cluster an environment variable has already failed to cross into a benchmark
subprocess — a HIP-only device mask that silently landed on the wrong GPU — so "we set it"
is not "it arrived".

**`-e` is not always enough.** A harness that exports the variable itself wins: AgentX's
`tools/environment.sh` sets `TRITON_CACHE_DIR=$IX_RUN/cache/triton` unconditionally. Where
the scenario gives `isolation.cache_dirs`, create a fresh empty host directory and
bind-mount it onto the container path the harness names, so the harness's own export lands
cold. The harness's real cache tree is shadowed, never emptied.

**Model weights are deliberately not in the cold set.** Emptying `HF_HOME`,
`HF_HUB_CACHE`, `TRANSFORMERS_CACHE` or `MODEL_PATH` re-downloads tens of gigabytes and can
change the revision underneath the run. Pin `pins.model_revision` instead, and refuse any
config that lists a weights cache among the compile caches.

**`SGLANG_SET_CPU_AFFINITY=1` under a restricted cpuset is a trap.** sglang derives
per-rank masks from the *full* topology, so inside a narrower cgroup it binds to CPUs the
cgroup then denies — which is how a prior run on this cluster had to set it to 0 in both
arms. Either grant the whole node or set it to 0, and then set it to 0 in **every** arm it
is compared against.

**A KV-offload point must be handed every `KV_OFFLOAD*` variable the client reads**, and
checked for all of them before the run starts. The AgentX client reads `KV_OFFLOADING`,
`KV_OFFLOAD_BACKEND`, `KV_OFFLOAD_BACKEND_METADATA` and `TOTAL_CPU_DRAM_GB`, but it checks
the backend when it *starts* and the metadata only when it *aggregates* — after the whole
measurement window. A hicache baseline (c64, 2026-10-28) hit both halves: one attempt
stopped at client start having already paid the server's startup, the next ran the full
hour and left no throughput at all. Apply the stricter rule up front, reading the
environment the client will actually be given: `params.env` first, then each
`harness.env_files` entry (the harness sources them with `set -a`, so they win), defaulting
to the harness's `pins.env` where it exists. Copy the values from the CI job's benchmark
env rather than inventing them.

A run under 900 s is a screening diagnostic on this workload, not a baseline.

## 3. Run it

`REUSE_SERVER` is part of the number. **Results from `REUSE_SERVER=1` and `0` are not
comparable**, and how far a round on a warm service differs from one on a cold isolated
service — or in which direction — cannot be derived from the configuration. Record which
was used in `provenance.measurement_mode`; applyback must use the same one.

**Do not open the profile window as the load starts.** At that point aiperf is still
parsing the corpus and staging per-lane warm-up, so the window captures the ramp — batch
1, none of the steady-state MoE/attention mix. The agentx adapter places the window at the
75% mark by itself; do not override it with `AGENTX_PROFILE_WARMUP_S` unless you know why.

For a pinned CI reproduction: start a **new** container, addressed as `repo@sha256:…`,
with the planned cpuset, the cold cache mounts, and the pin variables named by
`env_map` — the variable names a client reads are site-specific, so they come from config.
This skill will not pass a pin it cannot name.

## 4. Prove it, then do not grade yourself

Re-read each pin from an artifact the run itself produced, write
`$OUT/alignment_verified.json` check by check, and retire the `benchmark_id` to the ledger
**whether or not the run reproduced** — its cache-bust marker has already warmed a
keyspace. A check you could not run is `unproven`, which is its own verdict and never
renders as a tick.

Write `$OUT/baseline.json` in the canonical schema (`../README.md`), carrying every pin,
the per-check alignment result and `provenance.alignment_verdict`. Then stop:
`../finalize.py` writes `result.json` and it is the stage's conclusion
(`engine/core/README.md`). A run that completed but drifted outside the band is
`blocked`, not `ok` — a baseline that is not the CI point silently relabels every
downstream delta — and that rule is written there, not here. You can run it early to see
what it will say.

## Three things this skill will not do

- **Reuse a container.** Not even a clean-looking one: a container carries whatever a
  previous session installed in place, and none of that is in the image.
- **Reissue a `benchmark_id`.**
- **Report a tick for a check it could not run.**

## Outputs

`$OUT/baseline.json`, `$OUT/bench/` or `$OUT/replay.log` (whatever the benchmark wrote),
`$OUT/smoke.log`, and — when `pins` are declared — `$OUT/alignment.json`,
`$OUT/alignment_verified.json`, `$OUT/container_info.json`, `$OUT/affinity.json`.
`$OUT/result.json` is written by the slot, not by you.
