# vLLM observation and bounded profiling

Use this path for vLLM AgentX extraction. `server.log` throughput summaries do not
contain per-iteration request geometry. Do not parse them as SGLang batch rows,
invent `decode_log_interval`, or turn average ISL/OSL into observed call weights.
The Nov 2 DeepSeek-V4.1-Flash and Nov 5 Qwen3-8B-FP8 runs both hit this mismatch;
the latter also sent the SGLang `/generate` RPC to vLLM and received HTTP 404.

## Record the workload

Prefer iteration evidence from the completed baseline. For an owned serving run,
install the bundled observer before the workers import vLLM:

```bash
export GEAK_ITERATION_LOG_DIR="$EVAL_DIR/iterations"
export PYTHONPATH="$SHARED_TOOLS/adapters/iteration_env${PYTHONPATH:+:$PYTHONPATH}"
# Run the user's serving launcher with the unchanged model, TP and graph settings.
```

The observer wraps `vllm.v1.worker.gpu_worker.Worker.annotate_profile`, preserving
its return value. The installed scheduler output must expose `scheduled_new_reqs`,
`scheduled_cached_reqs.is_context_phase`, computed-token counts and scheduled-token
counts (validated on the pinned vLLM DeepSeek-V4.1-Flash image). Unsupported builds
must supply an equivalent observer; do not claim full coverage from an empty log.
Only CPU metadata is recorded: timestamp, rank, phase, query and prior context
length. No prompt content, request IDs, tensor values or arbitrary object reprs.
It chains the next `sitecustomize`, preserving the serving overlays. The default
byte cap is 256 MiB per worker (`GEAK_ITERATION_LOG_MAX_BYTES`); exhaustion writes
an error record, continues serving and makes the affected observation fail parsing.
For separate P/D engines or DP instances use distinct directories and retain their
engine/rank scopes; do not merge unrelated rank-0 streams into one distribution.

```bash
python3 "$SKILL_ROOT/scripts/observed_workload.py" \
  --log "$EVAL_DIR"/iterations/rank0_pid*.jsonl --log-format vllm-iterations \
  --window-start '<measured UTC start>' --window-end '<measured UTC end>' \
  --out "$EVAL_DIR/observed_workload.json"
```

Each record is one iteration; no log-interval multiplier is applied. Mixed steps
are owned once by PREFILL with total scheduled M, retaining the per-request split.
Prior computed tokens are context extent, not prefix-cache hits. Cache hit rate
and graph mode are left unknown rather than inferred from these records.
`--accept-len` is the measured acceptance used only for the decode cross-check;
with mixed steps, completed output tokens need an independent phase reconciliation.
Pass the effective chunk limit and graph capture sizes from the installed server
configuration to sampling; requested concurrency is not observed batch size.

If an old baseline has only aggregate logs, its missing geometry cannot be restored
by changing a parser flag. Reuse compatible recorded evidence or take a bounded,
explicitly labelled observation under the actual workload. Keep its interval and
coverage separate from the saved baseline score; do not silently rerun a full replay.

## Execute a point

```bash
python3 "$SKILL_ROOT/scripts/serving_probe.py" \
  --framework vllm --served-model '<served model name>' \
  --base-url http://127.0.0.1:8000 --points "$EVAL_DIR/sampling_points.json" \
  --point-id de_2 --trace-dir "$EVAL_DIR/traces/de_2" \
  --iteration-log-dir "$EVAL_DIR/iterations" --vocab-size '<actual vocabulary size>' \
  --expect-files '<worker rank count>' --out "$EVAL_DIR/de_2.journal.json"
```

vLLM requests use `/v1/completions` with token-ID prompts. Occupancy uses fresh
worker observations when supplied, otherwise vLLM running-request and token
counters. `/start_profile` receives `profile_prefix` and `max_iterations`;
`/stop_profile` closes the window. Enable the torch profiler at server launch,
including shape and detailed phase annotations on builds that expose them.
On builds without the per-request profiler fields, use the existing time-bounded
vLLM serving adapter and explicitly account for that alternative capture scope.

The trace directory above is a point subdirectory of the server's configured
profiler root. vLLM writes prefix-named files to that root; the probe links the
current point's new traces into its subdirectory. Request and window limits remain
finite. No SGLang abort RPC is sent to vLLM; finite requests must drain before the
next point. A trace must contain actual GPU kernels at the requested phase/shape.
The GPU lease and serving lifecycle rules in setup/capture still apply.

For main's `bench_e2e.sh`, AgentX + vLLM automatically stages this observer through
`PROFILE_PYTHONPATH`, also when `PROFILE=0`, and defaults the output to
`$OUT_DIR/iterations`. The extraction handoff supplies `BASELINE_ITERATION_DIR`.
Custom/remote launchers must propagate the environment and make that directory
accessible. Frozen UT execution does not install this observer or launch serving.
