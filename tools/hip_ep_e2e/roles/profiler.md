# hip-ep Profiler — Profile

You record one kernel trace of the model on `PKG` (the release package) and turn it
into per-phase kernel shares. You do not interpret beyond what the trace says.

## PHASE=profile

1. Run (LONG; DIR `EVAL_DIR/runs`, NAME `profile`):
   ```
   python3 TOOLS_DIR/kprof.py record --model MODEL_PATH --pkg PKG \
       --out EVAL_DIR/runs/profile --len LEN --gen 32
   ```
   It runs rocprofv3 around one `model_benchmark` and writes `EVAL_DIR/runs/profile/heads.json`.
   On failure, read `EVAL_DIR/runs/profile/record.log` and report it. Do not rerun with
   debug or profiling environment variables.
2. Read `heads.json`. `phases.prefill` is the timed prompt's forward and `phases.decode`
   its 32 one-token forwards; `by_base` lists their kernels with `share`, `ms`, `calls`
   and `source`.
3. Sanity check: `phases.decode.busy_ms_per_step` should be within about 10% of the release's
   decode ms/token in `AA_JSON` (`gate.per_len.<LEN>.decode_base`, us). If not, say so.

## Return
```json
{"heads_json": ".../heads.json",
 "prefill": [{"base": "...", "share": 0.0, "ms": 0, "calls": 0, "source": "lib/..."}],
 "decode":  [{"base": "...", "share": 0.0, "ms": 0, "calls": 0, "source": "lib/..."}]}
```
Copy every `by_base` row of each phase as written (kprof keeps the top 15).
