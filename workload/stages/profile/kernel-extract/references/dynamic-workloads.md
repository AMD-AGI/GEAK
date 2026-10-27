# AgentX workload measurement

Use the sections matching the current run. The [core skill](../SKILL.md) owns
activation, ranking artifacts and the weighting contract.

## Distribution and phase sampling

Measure active concurrency and server batches independently of requested
concurrency. Arrival gaps, think-time and scheduling change occupancy. Preserve
the configured replay duration, arrival schedule and session policy; inspect
occupancy for drain-down rather than silently recycling a finite session pool.
Establish how many iterations each log row represents, including logging
intervals and gaps; cross-check expanded counts against completed work.

When caching exists, distinguish nominal input length, cache hits and newly
executed tokens. Sample the observed cache regime. For attention, distinguish
query length, per-request context, aggregate occupancy and rank-local KV extent.
A changed cache capacity or hit distribution can invalidate old case weights.

When speculation exists, separate request batch, proposed/verified/accepted
tokens and padded kernel M. Derive their mapping from actual launches. Discover
cycle boundaries and omit partial endpoints; a named marker is valid only if
its cadence matches the current algorithm. Attribute subphases separately when
shapes or dispatch differ, including conditional layers and reused results.

Use the original joint distribution to select common cases and material tails,
including non-power-of-two values, chunk tails and served dispatch boundaries.
Record both call mass and baseline-time mass. Configured caps and graph capture
lists constrain possible shapes; runtime evidence establishes their mass and
served eager fallbacks. Keep unobserved envelope probes at zero observed weight.

## Turn observed shapes into trace sampling points

**Run the scripts below rather than reimplementing them.** Everything in this
section used to be prose only, and the predictable result was that each agent wrote its
own log parser, its own sampling heuristic and its own coverage arithmetic — so two
runs against the same deployment produced different points, different denominators and
different coverage claims, none of them comparable. The pipeline is now executable, and
the prose that follows is the reasoning behind it, not a specification to implement.

```bash
# 1. log + config -> what the measured leg actually executed
python3 "$SKILL_ROOT/scripts/observed_workload.py" \
    --log <serving log> --rank 0 --server-info server_info.json \
    --bench-log <client bench log> --completed-output-tokens <N> \
    --out "$EVAL_DIR/evidence/observed_workload.json"

# 2. distribution + budget -> N sampling points, each annotated with the mass it stands for
python3 "$SKILL_ROOT/scripts/plan_sampling_points.py" \
    --observed "$EVAL_DIR/evidence/observed_workload.json" -n 6 \
    --server-info server_info.json --out "$EVAL_DIR/sampling_points.json"

# 3. one server (probe_server.py, from serving.json), every point back to back, census in
#    the background, relabel, ledger, ranking, selection, then the evidence a task needs
#    (shapes, source map, adjacency) and a draft task_decls.json. Resumable.
python3 "$SKILL_ROOT/scripts/probe_server.py" run --serving "$EVAL_DIR/serving.json" \
    --run-dir "$EVAL_DIR/probe" -- \
  python3 "$SKILL_ROOT/scripts/run_points.py" --server-state "$EVAL_DIR/probe/probe_server.json" \
    --points "$EVAL_DIR/sampling_points.json" \
    --observed "$EVAL_DIR/evidence/observed_workload.json" --eval-dir "$EVAL_DIR"
# -> evidence/{probe,census,shapes}/*.json, evidence/{source_map,adjacency}.json,
#    sampling_points_final.json, coverage.json, kernel_ranking.json,
#    selection_targets.json, task_decls.json (draft), profile_run.json (per-step timings)

# the same steps one at a time, for a single point or a re-run:
python3 "$SKILL_ROOT/scripts/serving_probe.py" --points "$EVAL_DIR/sampling_points.json" \
    --point-id de_5 --base-url http://127.0.0.1:PORT --trace-dir <dir>/de_5 \
    --expect-files 8 --vocab-size <config.json vocab_size> \
    --out "$EVAL_DIR/evidence/probe/de_5.json"
python3 "$SKILL_ROOT/scripts/point_census.py" --trace-dir <dir>/de_5 --phase DECODE \
    --point-id de_5 --out "$EVAL_DIR/evidence/census/de_5.json"
python3 "$SKILL_ROOT/scripts/coverage_ledger.py" --points "$EVAL_DIR/sampling_points.json" \
    --census de_5="$EVAL_DIR/evidence/census/de_5.json" ... --out "$EVAL_DIR/coverage.json"
python3 "$SKILL_ROOT/scripts/build_ranking.py" --coverage "$EVAL_DIR/coverage.json" \
    --eager-captured DECODE --probe-result "$EVAL_DIR/evidence/probe/*.json" \
    --out "$EVAL_DIR/kernel_ranking.json"
```

Properties of this path worth knowing before you deviate from it:

- **Traces are read with `trace_scan.py`, never `json.load`.** An eager decode trace is
  ~280 MB of gzip (~4 GB of JSON) per rank; `json.load` takes ~60 s and ~16 GB of memory
  per rank, which also rules out parsing the ranks in parallel. The streaming scan takes
  ~20 s and ~300 MB, checks its row counts against the raw category counts and falls
  back to `json.load` itself if they disagree. Measured on GLM-5.2 TP8: an 8-rank decode
  census in 22 s, matching the hand-written census's compute time to 0.03%.
- `serving_probe.py` verifies on one rank, closes when the trace files settle rather than
  waiting out `--window-timeout`, aborts its decode requests and returns only on an empty
  batch, so the next point starts clean on the same server. Its decode prompts depend on
  the request index only, so a larger-batch point re-prefills only its new requests. It
  defaults to 8 steps in both phases.
- A decode window whose `step[VERIFY|DECODE ...]` span appears only on the CPU timeline
  was captured under the CUDA graph: `serving_probe.py` fails it as `graph_hidden` and
  `run_points.py` does not spend a recapture on it. Restart the server eagerly.
- `probe_server.py` keeps decode under its CUDA graph and adds
  `--enable-profile-cuda-graph`, so decode runs as deployed and the profiler sees the
  replayed kernels. If the server rejects the flag, or a decode window still comes back
  graph-hidden, it restarts eagerly and `run_points.py` recaptures only those points with
  `--decode-eager`. probe_server.json records which mode was used.
- Decode kernels replayed from the CUDA graph are visible with `--enable-profile-cuda-graph`
  (verified on sglang 0.5.20: five decode points in ~20 s each, collective wait 4% of
  decode) but carry no launching op and no source frame. Their costs stay from the graph
  windows; `run_points.py` asks for one eager evidence window at the smallest decode
  batch and `probe_server.py` serves it (~4 min with a warm JIT), after which
  `source_map.py` uses the eager frames. That window never enters the ledger.
- A prefill point is driven as ONE request with the batch's new tokens over one
  sequence's share of the cached prefix: sglang admits concurrent requests one extend
  at a time, so splitting a two-sequence batch measured half of it (measured: 32768 ->
  16416), and the old one-request-per-sequence form measured twice it.
- `relabel_points.py` sets every point to the shape its probe achieved (a two-sequence
  batch admitted one request at a time, a prefix residue) and re-partitions the observed
  instances over the achieved shapes; the instances no point covers within the ledger's
  tolerance are reported as bands, which `plan_ut_cases.py` turns into bucket cases.
  `point_census.py` takes the window's shape nearest the planned one, not the most
  expensive one, so a merged extend in the same window cannot replace the point.
- `kernel_shapes.py` (record_shapes + with_stack, which serving_probe arms by default)
  gives each selected kernel's launch variants, launches per instance, launching op with
  operand dims/dtypes/strides/scalars, and Python frames; `source_map.py` resolves the
  frames to installed files and `module:Class.method` (run it in the serving container);
  `adjacency.py` gives consecutive-launch evidence between selected kernels.
- `point_census.py` defines the instance (one prefill batch = target EXTEND + its draft;
  one decode iteration = anchor to anchor) and splits collectives per launch: the
  min-over-ranks duration is transfer, each rank's excess is `__collective_wait__`.
  Under eager decode that wait is ~90% of a step; `build_ranking.py --eager-captured
  DECODE` takes it out of the denominator (it does not exist under the graph) and reports
  how much it was. `__untracked_tail__` and `__collective_wait__` are never rankable.


- `parse_profile.py` phase-classifies forwards from the step-span dialect its
  `STEP_DIALECTS` registry knows, and a trace whose step spans are in an unregistered
  dialect **raises**. It used to return an empty phase split, which is indistinguishable
  from a trace with no forwards in it, and silently zeroed every per-phase number
  downstream. A new serving stack costs one registry entry.

- `observed_workload.py` **refuses to guess `decode_log_interval`**. Each decode row
  stands for that many iterations, so a wrong value scales the entire decode mass.
  It also cross-checks the expansion against completed output tokens — two routes to
  request-steps that share no term — and prints the disagreement.
- `plan_sampling_points.py` splits the budget **evenly between phases by default**,
  because the prefill proxy (tokens) and the decode proxy (iterations) are different
  units and comparing them as device time would silently decide the split. After the
  first census, re-run with `--phase-budget` to put the points where the time is.
- `serving_probe.py` encodes the arming rules in *Trace integrity* below as code:
  confirm occupancy from the server's own counters before arming, arm **once**, never
  `profile_by_stage`, close in a `finally`, then parse the trace back and assert the
  intended phase and shape are in it.
- `coverage_ledger.py` **refuses to report mass coverage for a phase measured at one
  point**, because a UT at that point covers 100% of a denominator built from that same
  measurement. It reports instance coverage instead and says why. It reads
  `parse_profile.py`'s own output as a census directly — dividing the window total by
  that file's per-phase step count, keeping only the point's phase, and booking the mass
  below the top-N cut to `__untracked_tail__` so a truncated census cannot shrink the
  denominator. It refuses a census that is both mixed-phase and truncated, because the
  remainder below the cut then belongs to no phase in particular.

Start from the completed baseline's logs and measured-leg exports. Reuse matching
windows already captured from that replay. If kernel costs or dispatch boundaries
are unknown, take a short targeted capture, then refine the points using measured
time mass. Do not run a new full replay or generic full-model profile. Derive
probe request controls from observed work and validate the resulting launches:

| Phase | Choose from configs/logs | Set and verify for each probe |
|---|---|---|
| Prefill | Joint uncached-token/context distribution, effective chunk budget, common chunks and tails | Prepare the observed cached prefix; choose total tokenized ISL to produce the target uncached suffix/context; arm before the intended extend; verify actual chunk/query/KV lengths |
| Decode | Joint active-batch/per-request-context distribution, speculation and graph padding | Prepare target context and cache state; choose request concurrency to reach the observed server batch; choose OSL long enough to pass admission and capture complete steady-state cycles; arm after decode begins and verify executed M/KV |

Record phase, request ISL/OSL, concurrency, cache preparation, server flags,
capture trigger/length, expected and observed shapes, selection reason, trace
path and covered mass in `kernel_workload.json`'s sampling plan. ISL is not
necessarily uncached work; OSL controls how long decode remains active, not its
kernel M. Reuse relevant quantization, routing and speculation behavior rather
than inferring it from request lengths. State approximations and missing coverage.

Targeted probes measure identity/cost; their synthetic request counts do not
replace original replay weights. Keep their controls local to the probe and
leave canonical replay config, workload/state and fixed ISL/OSL paths unchanged.

## Trace integrity

Arm short captures at observed phase boundaries: starting before admission can
produce a prefill trace labeled decode. Check actual markers and launches;
fixed delays from another run do not establish steady state. Resolve nested
annotations by phase semantics and containment, not timestamp ordering alone.
Mixed windows require attribution by complete phase instances.

Compare expected forwards and per-family events across ranks only when the
execution should be symmetric. Use the modal clean pattern rather than the
largest count, which may include compilation/autotuning. Rank agreement alone
cannot detect a phase omitted everywhere; also check workload progress and
active-graph call counts. Validate phase-exclusive markers on both matching
and nonmatching captures before using them to reject a window.

Busy/span is diagnostic evidence, not a universal integrity threshold. Peer
wait can inflate collective duration; host stalls can lower activity despite
complete events. Discover collective families from all ranked launches and
inspect significant unclassified work. A rank with the smallest summed time
is not automatically an unbiased compute reference.

Use bounded eager diagnostics if production graphs hide events, then validate
final timing in deployment graph/compile mode. On affected backends, disable
stage profiling only when it demonstrably omits work and explicitly close the
capture. Before restarting an owned failed server, verify worker exit and
allocation release; stop health polling when the process exits. Preserve raw
captures and point-specific logs, placing retries in separate directories.

## Cost attribution

Complete observed counts with per-case latency need no parametric model. For
sampled costs, fit verified logical families or separate dispatch branches;
retain branch identities when assigning editable shares. Do not treat a kernel
name disappearing at a dispatch boundary as zero operation cost.

Choose axes and fixed-cost terms supported by measurements. Examples include
`a + b*batch + c*batch*context` or `a + b*new_tokens + c*new_tokens*context`, but
sparse/windowed attention or other architectures may have different dependencies.
Inspect absolute and relative residuals, validate additional points and report
extrapolation. Fit objectives must match the deployment distribution rather
than letting a few large samples dominate. Normalize per call/phase instance
before comparing captures of different lengths.

For gap-based throughput logs, inferred phase occupancy requires validation
against spans or independent progress; another backend may define its counters
differently. Keep rank/stream scope consistent and preserve collective/wait and
unattributed mass. Ledger closure checks accounting, not representativeness.

Cross-check predictions with independent benchmark counters or device telemetry
where available over the same interval/ranks. Document units, sampling error
and overlap: integrated `gfx_activity` is activity time, not summed device
kernel time or non-collective compute. Missing telemetry is a stated validation
limit. Production duty and critical-path assumptions are needed for wall-time
projections; eager profiler span and a scalar utilization factor cannot by
themselves establish an e2e gain.
