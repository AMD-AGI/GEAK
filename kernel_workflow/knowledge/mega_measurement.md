# Mega measurement guide — where does the time go, measured instead of read from code

You get this file as `MEGA_MEASUREMENT_GUIDE` only in a whole-operator fusion run (`mode=mega`) with
Expert Skills disabled. It adds three measurements to that run, plus an operator-neutral graph
contract runner (§4). It never changes how a candidate is
scored: the paired, meter-off, rank-max e2e A/B stays the only number that ranks anything. Every
number here is advisory and describes where time is spent.

The rule behind all three measurements: **a number you did not measure is omitted, never
estimated, zero-filled or copied from source-code reasoning.** "Not measured, because …" is a valid
answer. A plausible guess is not.

Read the section for your role:

| role / phase | section |
|---|---|
| `profile_engineer`, PHASE=`mega_baseline` | §1 |
| mega engineer (`optimize`) | §2 |
| `verify_engineer` (`verify`) | §2 + §2.3 |
| `analysis_engineer` (label `mega:analyze`) | §3 |
| `benchmark_engineer`, PHASE=`setup` | §4.1 |
| `verify_engineer` (`verify`), `director` (`select_mega`) | §4.2 |
| mega engineer (`optimize`), optional self-check | §4.3 |
| mega engineer (`optimize`), fast A/B while iterating | §5 |

---

## §1 Baseline per-stage profile (profile_engineer, PHASE=mega_baseline)

Your role file's `PHASE=mega_analysis` section refers to bench flags this harness may not have.
It does **not** apply here. Follow this section instead.

The frozen baseline runs as several separate, serialized launches. Before fusion, per-stage time can
be measured directly, so measure it now: the planner needs to know which stage dominates before it
authors anything.

Inputs: `WORKSPACE`, `EVAL_DIR`, `COMMANDMENT`, `GPU_ID` (the whole multi-GPU lease),
`BASELINE_PER_CASE`, `TARGET_GUARDS`, `REGRESSION_GUARDS`, `SKILL_DIR`.

**Budget.** Finish within about 45 minutes. The orchestrator stops waiting at 60 minutes and falls
back to an unmeasured stub, which loses everything you measured. Order of work: the target guard
first, then the smallest regression guard, then one mid-size guard. Skip the rest if time is short
and list them as not measured.

Wrap every GPU command in `bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID …`. Do not edit any tree.

0. **Regime map (no GPU, a minute).** Place every guard on the shape's roofline before measuring
   stages (MEGA_GENERIC_LESSONS §5.1):
   ```
   python3 $SKILL_DIR/tools/mega_regime.py --guards <TARGET_GUARDS + REGRESSION_GUARDS>
     --experts-per-rank <E/world> --topk <k> --world <ranks> --hidden <model dim> --inter <inter dim>
     --weight-bits <stored bits per weight incl. scales> --dispatch-bytes <b> --combine-bytes <b>
     --hbm-tbps <achieved> --math-tflops <peak for the operand type> --link-gbps <per rank>
     --hardware-source "<measured how / datasheet>" --baseline-ms <guard=BASELINE_PER_CASE ms ...>
     --json-output $EVAL_DIR/regime.json
   ```
   - Take the shape from the task's test or COMMANDMENT, never guess it.
   - Prefer measured bandwidth (a large device copy, or a link copy between two ranks) over
     datasheet peaks. If you only have a datasheet figure, say so in `--hardware-source`.
   - The baseline figures are the rank-max e2e of `BASELINE_PER_CASE`.
1. **Stage timers.** Take the frozen-baseline benchmark command for each guard from `COMMANDMENT`
   (frozen tree, activation switch off). Run it once and parse the rank-reduced stage timers the
   harness prints: here that is the `[RESULT]` line's `stage1=<mean>/<max>` and
   `stage2_combine=<mean>/<max>` along with the e2e. Use the **max** (rank-max). These timers are
   meaningful **only for the unfused baseline**. On a fused candidate, a stage that starts early
   charges its own waiting to itself, so these timers must never be used for fused candidates.
2. **Per-kernel split.** Stage timers usually lump stage2 and combine together. To split them, run
   the same command with the harness's trace option (here `--profile-dir <EVAL_DIR>/mega_profile/<guard>`
   writes one torch-profiler chrome trace per rank, covering 3 graph replays). Alternatively use the
   per-rank rocprofv3 wrapper `SKILL_DIR/scripts/multi_rank_analysis/run_rank_kernel_trace.sh`.
   - Bucket the kernel durations per rank with
     `scripts/multi_rank_analysis/trace_categories.py` (`load_category_map` + `bucket_trace_events`).
   - Use a category map with keys `stage1`, `stage2`, `combine`. Start from
     `knowledge/analysis_skills/moe_bottleneck/default_moe_category_map.json`. If the observed
     kernel names do not match it, write your own map from the observed names, save it under
     `EVAL_DIR`, and record its path.
   - Divide by the replay count, then take the rank-max per stage.
   - Also record the median gap between consecutive kernels (per rank, rank-max): that is launch
     and serialization cost that fusion can remove.
3. **Pipe utilization (optional, only if time remains).** Use `run_rank_pmc.sh` on the target guard
   to measure per-stage functional-unit utilization (valu / mfma / vmem / lds). Only measured rows go
   into `resource_timeline.pipes`.
4. Write `EVAL_DIR/mega_baseline_stages.json` in this shape, omitting any key you did not measure:
   ```json
   {"baseline_stages": [{"name": "<guard>", "e2e": 0.0, "stage1": 0.0, "stage2": 0.0,
     "combine": 0.0, "stage2_combine": 0.0, "gap_ms": 0.0, "replays": 3,
     "source": "<exact commands + trace/log paths>"}]}
   ```
   All values are rank-max ms per forward.
5. Return the PROFILE schema:
   - `bottleneck`: the measured dominant stage of the target guard with its share of e2e (e.g.
     `"combine 41% of e2e"`), or `"unknown"`.
   - `profiler_used`: the tools you actually used.
   - `device`.
   - `dispatch_count`: measured launches per forward.
   - `key_metrics`: `{baseline_stages_path, stage_ms_by_guard, interkernel_gap_ms, regime_path,
     regime_by_guard: {<guard>: {regime, binding_resource, baseline_over_bound, unexplained_share}},
     buckets: [{regime, guards}], not_measured: [...]}`.
   - `top_kernels`: `[{name, stage, rank_max_ms, share_pct}]` for the target guard.
   - `top_opportunities`: measured facts only, not advice copied from source.
   - `resource_timeline`: only if step 3 ran, with shape `{pipes: [{stage, pipe, utilization_pct, source}],
     interkernel_gap_us: {median}}`.
   - `summary_path`, `shift_note`.

---

## §2 Per-stage timestamp meter for the fused candidate (engineer + verify)

After fusion there is one kernel. Nothing outside the kernel can show which stage is slow or how
long workgroups spin waiting on each other. Only an in-kernel meter can. The design and its traps
are in `knowledge/overlap_instrument.md` (per-workgroup phase log, `s_memrealtime`, preallocated
buffer, XCD clock skew, perturbation, rank-local clocks). This section fixes one **standard
contract**, so every candidate is measured the same way and the numbers can be compared across rounds.

### 2.1 Contract (engineer builds it, keeps it working)

- **Switch.** `GEAK_MEGA_STAGE_METER=1` arms the meter. `GEAK_MEGA_STAGE_METER_OUT=<dir>` names
  the output directory. When the switch is unset, the meter is compiled out: a compile-time
  constant, a separate JIT identity, and zero instructions in the timed kernel. Timing always comes
  from the meter-off build.
- **Roles.** Each workgroup role gets one enter/leave timestamp pair, written by one lane. Every role
  is tagged with:
  - `stage`: one of `stage1`, `stage2`, `combine`.
  - `kind`: `busy` means doing that stage's work (loads, MMA, stores, sends). `wait` means spinning
    on a readiness or arrival counter, a barrier, or an empty-queue poll that belongs to that stage.
- **Nesting.** Roles must not nest within a workgroup. The reducer rejects a workgroup whose role
  time exceeds the kernel span.
- **Dump.** Each rank writes `<dir>/stage_meter_rank<r>.json` with `{rank, tick_hz, n_workgroups,
  overflow, roles:[{name,stage,kind}], events:[[wg, role_index, t_begin, t_end], ...]}`.
  - `tick_hz` must be the counter's real rate. Confirm it once against a host timer; do not assume it.
  - `overflow > 0` voids the dump.
- **Engineer.** Build the meter into the first fused candidate and keep it compiling and correct in
  every later edit (it is part of the candidate tree). State in your return notes that it is present
  and give the env switch. A candidate without the meter is still scored normally, but its round
  produces no stage data, so the next plan is made blind.

### 2.2 What the numbers mean

For each rank and stage:

- `busy_ms` = Σ busy interval over all workgroups / `n_workgroups`
- `wait_ms` = Σ wait interval / `n_workgroups`
- `span_ms` = wall-clock union of that stage's busy intervals

Each value is then reduced with **rank-max**, the same reduction as the e2e metric. How to read
them:

- High `wait_ms` means workgroups sit idle on dependencies. Look at the scheduling and ordering
  between producers and consumers, not at the math.
- A stage's `busy_ms` above the baseline's time for that stage means its work got slower inside the
  fusion (tile shape, occupancy, LDS or register pressure, contention with co-resident roles).
- Stages may overlap, so busy times need not add up to e2e. Never subtract them from e2e.

### 2.3 Verify (after the score arms, never instead of them)

1. Finish every meter-off timing arm first. The meter never produces a scored latency.
2. Run the target guard once with the meter armed. If time allows, also run one small regression
   guard. Also time the same guard meter-off once, and report
   `meter_overhead_pct = metered / meter_off − 1`.
3. `python3 $SKILL_DIR/scripts/multi_rank_analysis/reduce_stage_meter.py --dump '<dir>/stage_meter_rank*.json'
   --guard <guard> --evidence '<log path>; overhead=<pct>' --output <VERIFY_DIR>/stage_meter_<guard>.json`
4. If `valid` is true, copy its `component_breakdown` rows unchanged into your return
   `component_breakdown` (`method: "timestamp"`, `guard` set). If `valid` is false, or the candidate
   has no meter, return no timestamp rows and give the reason (the reducer's `problems`, or
   "meter absent") in your notes. Never fill rows by estimate.

---

## §3 Analysis (analysis_engineer, label `mega:analyze`)

In the route-report pre-step, add the measured inputs to the builder call. Only add the ones that
exist:

- `--baseline-stages EVAL_DIR/mega_baseline_stages.json`, if §1 wrote it.
- `--component-breakdown <file holding candidate_evidence.component_breakdown>
  --breakdown-guard <target guard>`, if the rows exist. Use this flag **instead of** passing those
  rows as `--component-timings`.

With these inputs, each category carries its measured baseline and candidate ms, and the fused
candidate's in-kernel wait appears as its own `fused_wait` category. Confidence rises to `medium`
only when stage2 and combine were split on both sides and every candidate row is `timestamp`.
Report the confidence the builder stamps; do not raise it.

For the baseline call (`analysis_engineer:baseline`, round 0) there is no candidate yet. Summarize
the measured baseline stage shares from `PROFILE_SUMMARY` and stop there.

---

## §4 Graph contract runner (the generic `GRAPH_CONTRACT_TOOL`)

In this run `GRAPH_CONTRACT_TOOL` is `tools/generic_graph_contract.py`. The runner knows nothing
about the operator. It owns the measurement:

- graph-captured accuracy per case (relL2, rank-max);
- replay liveness with fresh inputs on every replay;
- per-capacity (`_mtpr<M>`) cases, with fallback allowed only where declared;
- a cross-bucket switch test (every operator and size interleaved, graph and eager, ragged ranks);
- a repeat-same-input bitwise check per operator;
- path-marker activation on every rank;
- launches per forward;
- rank-max graph timing;
- a hang watchdog that names the stuck phase.

All operator knowledge lives in one task-owned **contract adapter**,
`EVAL_DIR/commandment_tools/graph_contract_adapter.py`, written once at setup. The protocol
(5 functions) is in the runner's docstring:

- `setup()`
- `build(capacity)`
- `make_inputs(tokens, route, iteration)`
- `forward(op, inputs)`
- `reference(inputs)`
- `cleanup()` (optional)
- `RAGGED = True` (optional) when `make_inputs` works with a different token count on each rank.
  Set it whenever the official test or bench supports per-rank token counts.

### 4.1 benchmark_engineer, PHASE=setup: write and prove the adapter

1. **Write the adapter from the task's own test/reference code.**
   - Load test helpers (reference, weight preparation, distributed setup) **by file path from the
     frozen tree** `EVAL_DIR/baseline`, using `importlib.util.spec_from_file_location`, never from a
     candidate tree. That way a candidate can never change its own reference.
   - Import the operator class normally. It resolves to the tree under test, because the runner puts
     `--candidate-tree` first on `sys.path`.
   - `build(capacity)` uses the official test's sizing convention. A `_mtpr<M>` guard is capacity `M`.
   - `make_inputs`:
     - accepts every route named in the guard ids (`<tokens>_<route>[_mtpr<M>]`);
     - is deterministic per (rank, tokens, route, iteration);
     - iteration 0 gives the canonical inputs, and every later iteration gives different values with
       the same shapes and dtypes.
   - `forward` returns this rank's valid output rows. `reference` returns the same rows.
2. **Prove it on the frozen tree, in one lease.** Use the COMMANDMENT environment and the lease
   wrapper:
   `torchrun --standalone --nproc_per_node=<ranks> $SKILL_DIR/tools/generic_graph_contract.py --self-test
   --candidate-tree $EVAL_DIR/baseline --runtime-file <adapter> --accuracy-cases <REQUIRED_ACCURACY_CASES>
   --routes <all guard routes> --mtpr-cases <tokens of every _mtpr guard> --target-guard <first TARGET_GUARDS entry>
   --switch-rounds 2 --control-env <baseline switch-off KEY=VALUE>
   --import-modules <CANDIDATE_IMPORT_MODULES> --json-output $EVAL_DIR/graph_contract_selftest.json`.
   - Require `self_test_pass: true`, with every accuracy and replay row passing.
   - Record `switch_test.status` and every `determinism` row. If the **frozen baseline** fails the
     switch test, the test cannot gate candidates: record the failing step in COMMANDMENT and add
     `--switch-advisory` to the §4.2 command (the result is then reported, not gated). If the
     frozen baseline is not bit-exact on repeated input, do not add `--require-deterministic`.
   - For reference: on an 8-rank operator of this size, a self-test typically takes about a minute.
3. **Record it.** Add a `## GRAPH_CONTRACT` section to COMMANDMENT with:
   - the adapter path and its sha256;
   - the routes;
   - the self-test JSON path;
   - the §4.2 command with the run's values filled in.

   The adapter is immutable from then on, like COMMANDMENT.
4. **If it fails.** If the self-test does not pass within about 30 minutes:
   - do not ship a half adapter: `mv` it to `graph_contract_adapter.py.broken`;
   - say so in COMMANDMENT.

   Verification then uses the COMMANDMENT correctness command. Nothing else in setup depends on
   the adapter.

### 4.2 verify_engineer (verify) and director (select_mega): the candidate claim

**When this command applies.** Use it when both hold:

- the adapter exists;
- its sha256 equals the COMMANDMENT record.

In that case it **replaces** the role's `GRAPH_CONTRACT_TOOL` command line. There is no
`EXPERT_SKILL_RUNTIME_FILE` in this run; the adapter takes its place. Otherwise take the role's
empty-tool branch (the COMMANDMENT graph-capable correctness command for every required case and
replay count). Say which branch you used.

```
torchrun --standalone --nproc_per_node=<ranks> GRAPH_CONTRACT_TOOL --candidate-tree "$WS"
  --runtime-file EVAL_DIR/commandment_tools/graph_contract_adapter.py
  --accuracy-cases <REQUIRED_ACCURACY_CASES> --liveness-cases <REQUIRED_ACCURACY_CASES>
  --routes <COMMANDMENT GRAPH_CONTRACT routes> --replays <GRAPH_CONTRACT_REPLAYS>
  --mtpr-cases <as the role derives them> --mtpr-fallback-max <as the role derives it>
  --target-guard <first TARGET_GUARDS entry> --fallback-capacities <see below>
  --switch-rounds 2 [--switch-advisory, only if COMMANDMENT says so]
  --rtol <ACCURACY_THRESHOLD> --frozen-baseline-ms <BASELINE_PER_CASE latency of the first TARGET_GUARDS entry>
  --min-speedup <MEGA_MIN_SPEEDUP or 1.0>
  --path-marker <candidate ACTIVATION marker> --activation-env <each candidate switch KEY=VALUE>
  --control-env <BASELINE_ACTIVATION switch KEY=VALUE> --launch-target <LAUNCH_TARGET>
  --import-modules <CANDIDATE_IMPORT_MODULES, comma-separated> --hang-timeout-s 600
  --json-output "$VERIFY_DIR/graph_contract.json"
```

**Fallback buckets.** `--fallback-capacities` lists the capacities (`_mtpr<M>`, or the shared
capacity) of the buckets the candidate's own notes declare as honest baseline fallbacks. A capacity
that belongs to the target guard is never a fallback; the runner refuses it. Leave it empty when the
candidate declares none. A fallback bucket still has to pass accuracy, replay, the switch test and
its regression floor in the scored A/B.

**Launch target.** `--launch-target` is a ceiling: fewer launches per forward pass, more fail.

Run it in one lease with the COMMANDMENT environment (`$WS` first on `PYTHONPATH`) and an outer
wall timeout.

**Reading the JSON:**

- **Claim status.** `claim_complete: true` exactly when `claim_blockers` is empty. Exit 1 with
  blockers is incomplete evidence to report, not a harness crash.
- **Evidence rows.** Copy these unchanged as your evidence:
  - `accuracy_results` and `replay_results` rows;
  - per-rank `activation` / `mtpr_activation`;
  - `launch_count`.
- **Shape adaptivity.** Quote `switch_test` (status, steps, `failed_step` if any), the
  `determinism` rows, `target_activation`, and `launch_count_by_capacity`. A `switch_test_failed`
  blocker means one path corrupts another; it is a correctness failure of the candidate.
- **Hangs.** `hang_phase` means liveness failed in that phase. Quote it verbatim; it is the
  cheapest hang localization you will get.
- **Timing.** `paired_readings`, `absolute_speedup` and `incremental_switch_speedup` are in-process
  graph timings. They are diagnostic only; the scored number stays the COMMANDMENT paired A/B. Report
  any disagreement over 5% and any `denominator_mismatch`.
- **Adapter integrity.** `adapter_sha256` different from the COMMANDMENT record voids the run.
- **Resources.** `--resource-evidence` is not gated here. Do not build a Skill resources file.

### 4.3 mega engineer: optional self-check

Before handing off, you may run the §4.2 command on your own candidate with `--replays 30`. It
catches accuracy, replay and activation failures early, and the watchdog names the phase of a hang.

Never edit the adapter. If the adapter cannot drive your candidate (for example, a changed
constructor or call signature), your candidate broke the operator's interface; fix the candidate.

## §5 Fast paired A/B while iterating (`MEGA_AB_TOOL`, mega engineer)

`MEGA_AB_TOOL` times arms against each other in CUDA graphs: one torchrun job per arm and pass,
rotated arm order, rank-max graph time, the paired delta against the first arm, an output relL2 /
bit-exact screen against the first arm, and `[megamoe] path=` marker counts per leg.

```
python3 $MEGA_AB_TOOL --tokens <T> --mtpr <M> \
    --arm base=<frozen tree> --arm cand=<candidate tree>#<switch>=<value> \
    --expect-path cand=MEGA --passes 3 --work-dir $EVAL_DIR/ab_<tag> \
    --jit-root <a directory outside every tree> --json-output <file>
```

- Wrap it in the GPU lock like every GPU command; give each arm only the switches the candidate
  reads.
- Graph timing noise is about ±1 µs. Keep a change only after at least 3 passes with a consistent
  sign; below about 1 µs use at least 5.
- It is for choosing what to keep. The score still comes from the workflow's paired A/B and the
  graph contract (§4.2).
