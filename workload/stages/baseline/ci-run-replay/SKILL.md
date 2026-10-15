---
name: ci-run-replay
description: Reproduce a named CI run as this pipeline's baseline, and accept the result only if it aligns with that run. Pins the image by digest rather than tag, pins data/seed/benchmark_id/cache-bust, binds one rank per physical core from the machine's own topology, and runs in a brand-new container with empty Triton/XDG compile caches. Every pin is proven afterwards from the run's own records; an unproven alignment is reported blocked rather than adopted as a baseline.
---

# Reproduce a CI run, and prove it was the same run

The other two baseline implementations answer "what was the number?" — `adopt-measured-receipt`
by reusing one, `agentx-trace-replay` by measuring against a service someone else started.
This one answers a stricter question: **can this machine reproduce the CI point from
scratch, to within a declared band?** It is the right choice when the CI run is the thing
being optimized, because everything downstream is a delta against it.

Output throughput is the only number that has to line up, and it will not line up by
itself. The six pins below are each worth tenths of a percent to several percent, which is
the entire size of the effects this pipeline is built to detect.

## The six alignments

| Pin | Why it moves the number | Checked before | Proven after |
| --- | --- | --- | --- |
| **Image digest** (not tag) | a tag is a moving pointer; a repush changes the stack with no trace | digest is well-formed, present locally, and is one of the tag's `RepoDigests` | `docker inspect` of the container that actually served |
| **Data + seed** | the corpus decides the prompt-length mix, hence the prefill/decode ratio; the seed decides replay order | corpus exists, entry count matches, seed/warmup pinned | seed read back from the client's own export |
| **`benchmark_id`** | the AgentX cache-bust marker is built from it, so the id *is* the KV keyspace | refused if the id appears in the ledger of ids already spent | id read back from the run, then retired to the ledger |
| **Cache-bust target** | decides how much of each prompt can be served from cache | must be `first_turn_prefix`, which the scenario plugin locks | read back from the client's recorded CLI/config |
| **CPU binding** | two ranks on one core's execution resources is not the CI topology | one logical CPU per physical core from `lscpu -p`, only from granted cores | `sched_getaffinity` sampled **inside** the container |
| **Fresh container + cold compile caches** | a warm Triton/XDG cache can hold kernels compiled from another campaign's overlay | container name unused, cache dirs created empty | the caches are **not empty** afterwards — the only proof the variables reached the server |

The last row is the one that catches the quietest failure. If `TRITON_CACHE_DIR` never
reaches the server process, the server compiles into its own warm cache, the run completes
normally, and every log line still says the cache was cold. An empty cache directory after
an hour of serving is therefore treated as a failure, not as tidiness.

Model weights are deliberately **not** in the cold set. Emptying `HF_HOME` re-downloads
tens of gigabytes and can change the revision underneath the run; `preflight.py` refuses to
treat a weights cache as a compile cache. Pin `pins.model_revision` instead.

## Input `$PARAMS`

```json
{
  "ci": {"run_id": "34783501425", "run_url": "https://…/runs/34783501425/job/…",
         "output_throughput": 446.56324, "max_drift_pct": 0.5},
  "pins": {"image": "repo/name:tag", "image_digest": "sha256:…",
           "model": "amd/GLM-5.2-MXFP4", "model_revision": "386bd0e4…",
           "seed": 42, "benchmark_id": "wl_ci_replay_01",
           "cache_bust": "first_turn_prefix", "duration_s": 3600,
           "warmup_requests_per_lane": 10,
           "trace_corpus": {"loader": "semianalysis_cc_traces_weka_062126",
                            "entries": 393}},
  "cpu": {"bind": "physical-cores", "sglang_set_cpu_affinity": 1},
  "isolation": {"fresh_container": true, "container_name": "wl_ci_replay",
                "cache_vars": ["TRITON_CACHE_DIR", "XDG_CACHE_HOME",
                               "TORCHINDUCTOR_CACHE_DIR", "TRITON_HOME"],
                "cache_dirs": {"TRITON_CACHE_DIR": {"container": "…/cache/triton"}},
                "cache_root": null, "ledger": null},
  "env_map": {"benchmark_id": "AIPERF_BENCHMARK_ID", "seed": "AIPERF_RANDOM_SEED"},
  "docker": {"volumes": ["/shared_nfs:/shared_nfs"], "extra_args": []},
  "harness": {"dir": "…", "cmd": "tools/run_tp8_c10.sh",
              "args": ["CASE", "RUN_TAG", "DURATION"], "case": "baseline",
              "run_tag": null, "result_dir": null}
}
```

- `cpu.bind`: `physical-cores` (default), `inherit` (record the mask without changing it),
  or `explicit` with `cpu.cpus`.
- `pins.trace_corpus`: a local `path` (entry count is checked) **or** a named public-dataset
  `loader`. The AgentX harness takes the latter, and the name selects the corpus
  *generation* — `062126` and `061526` are different prompt-length mixes.
- `isolation.cache_vars` exports the variables and nothing more. Use `cache_dirs` when the
  harness exports them itself: a fresh empty directory is created under `cache_root` and
  bind-mounted onto the container path the harness names, so its own export lands cold.
  The harness's real cache tree is shadowed, never emptied. Without this, a harness like
  `tools/environment.sh` — which sets `TRITON_CACHE_DIR` unconditionally — simply wins over
  the `-e` pin, and the run compiles into whatever that shared directory already held.
- `env_map` exists because the environment-variable names a client reads are site-specific.
  The skill refuses to guess them; what it will not do is pass a pin it cannot name.
- `harness.args`: literal arguments in which the bare tokens `CASE`, `RUN_TAG` and
  `DURATION` are substituted. This skill does not reimplement the client — it drives the
  site's own CI harness, so the thing being reproduced stays the thing being reproduced.
- `allow_unproven: true` downgrades "could not check" from a failure to a note. Use it only
  with a written reason; the default is that an unproven pin is a failure.

## Output

`$OUT/baseline.json` (the canonical schema, with every pin and the alignment verdict in
`provenance`), `$OUT/alignment.json` (the plan), `$OUT/alignment_verified.json` (the
proof, check by check), `$OUT/replay.log`, `$OUT/container_info.json`, `$OUT/affinity.json`,
`$OUT/result.json`.

`status` is `ok` only when the throughput lands inside `ci.max_drift_pct` **and** every pin
was proven. A run that completed but drifted is `blocked`, not `ok`: a baseline that is not
the CI point silently relabels every downstream delta.

## Three things this skill will not do

- **It will not reuse a container.** Not even a clean-looking one: a container carries
  whatever a previous session installed in place, and none of that is in the image.
- **It will not reissue a `benchmark_id`.** The id is spent the moment it is used, whether
  or not the run reproduced, because its cache-bust marker has already warmed a keyspace.
- **It will not report a tick for a check it could not run.** Unproven is its own verdict.
