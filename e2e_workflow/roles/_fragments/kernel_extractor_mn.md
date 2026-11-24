# Kernel Extractor — multi-node serving

Read this only when `SERVING_TOPOLOGY` is in your inputs. On a multinode run `CAPTURE_ON_ENGINE`
(`tp_span` / `pd_disagg`) is always in your inputs too: the shape capture of PHASE=extract step 2 and
any capture of PHASE=extract_op run as the serving leg below, never as the role file's one-card
server. That server is not the served model: its N/K are unsharded, `--dp-size` /
`--enable-dp-attention` / `--ep-size` are refused at TP=1, and a model bigger than one card never
boots. So never fall back to it, not even "just for the shapes".

## The capture leg (replaces the `bench_e2e.sh` command of PHASE=extract step 2)

Build the capture overlay `--from "$CAPTURE_ENGINE_BASE_OVERLAY"` (the current shared overlay). Run it
as a SERVING leg: keep the `BENCH_LAUNCHER` / `GEAK_MN_STATE` part of your prompt's SERVING env line (it
is what puts the leg on the pinned machines), at `TP="$CAPTURE_TP" GPU="$CAPTURE_GPU"`:
```bash
BACKEND="<backend>" <BENCH_LAUNCHER=... GEAK_MN_STATE=... from the SERVING env line> \
OUT_DIR="$TASK/_capture" TP="$CAPTURE_TP" GPU="$CAPTURE_GPU" MODEL="$MODEL_PATH" \
ISL=<WORKLOAD.isl> OSL=<WORKLOAD.osl> CONC=<WORKLOAD.conc> PROFILE=0 \
GEAK_REPEAT_MODE=warm_server REPEATS=1 MEASUREMENT_PURPOSE=search \
OVERLAY_PYTHONPATH="$TASK/_capture_overlay" \
EXTRA_SERVER_ARGS="$CURRENT_FLAGS" \
EXTRA_ENV="$CURRENT_ENV CAPTURE_TARGET=... (the capture vars of the role file's command)" \
  bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/capture_<short_name>.log"
```
Keep `EXTRA_SERVER_ARGS="$CURRENT_FLAGS"` verbatim — the DP/EP flags in it are what make the captured
shapes the served ones. `CAPTURE_OUT=$TASK` must be on the shared filesystem (it is: `EVAL_DIR`); every
rank writes its `capture.pid-*.rank-*` dir there from whichever host serves it (the names carry the
host). Each holds that rank's `meta.json` and selection traces; only rank 0 also holds
`reference_io.pt`. Run `kernel_selection.py` exactly as the role file says, over all the metas: it
promotes rank 0's oracle.

**`tp_span`.** One engine. Record `"capture_engine": "tp_span"` and `"capture_lifecycle": "warm_server"`
in `meta.json`.

## PD (`SERVING_TOPOLOGY=pd_disagg`)

**Capture on BOTH real engines, in the MEASURED lifecycle.** A colocated one-box server batches
prefill differently and misses the warm rounds, so its M set is not the engines'. Both engines run
one code tree, so the patch you are building a case set for runs on both, and the capture leg above
puts the hook on both.
- Verify in every engine's `log` that `$TASK/_capture/mn_launch.json` records under `pd.engines` that
  the hook ran there.
- The M set of this op is the union of `meta.shape_counts` over both engines' rank-0 metas (cold
  round + warm rounds, with their call counts). An engine whose meta has no calls does not run this
  op; say so in `notes`. Record `"capture_engine": "pd_disagg"` and `"capture_lifecycle": "warm_server"`
  in `meta.json`. Do not replace the captured M with chunk/ISL buckets; add those only when the
  capture saw them.

**PHASE=extract_op: the colocated both-regimes rule holds.** Prefill and decode run on different
machines but one code tree, so a patch to this kernel runs at the prefill engine's M AND the decode
engine's M; a head tuned on one engine's M alone can regress the other. Build the case set over both
engines: `PROFILE_WORKLOAD_JSON` is the merged model (`mn_traces.py topn-merge --workload-out`: each engine's
cases, weighted by that engine's share of a request's GPU time, each tagged with its engine's
regime), so let `attribute_weights.py` derive the served regimes from it. Drop a regime only when the
capture shows this kernel never runs on that engine, and say so in `notes`.

**The M set comes from the capture on both engines, even for a synthesized GEMM.** `GEMM_SYNTH` only
means the operands may be synthesized; the shapes may not be derived from ISL/chunk. In GEMM step 2,
still run the capture leg above for the M set (not for the oracle), and take every `(M,N,K)` in its
`meta.shape_counts` as a case, weighted by its count. A prefill engine under the warm measured
lifecycle runs mostly few-token M (prefix-cache hits), which no config-derived bucket predicts. Put
the captured M lists in `meta.prefill_m_buckets` / `decode_m_buckets`; if the capture failed, say so
in `notes`.
