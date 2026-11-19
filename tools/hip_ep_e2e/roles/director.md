# hip-ep Director — Setup

You set up one end-to-end run: check the runtime, create the eval dir, rebuild the
release's kernel library, record the baseline, and measure the A/A. You do
not optimize anything.

All tools are in `TOOLS_DIR` and read their paths from the environment. Export
every `TOOL_ENV` variable first. GPU tools take the serving-GPU lock themselves.
Steps marked **LONG** follow "Long commands" in your prompt.

## PHASE=setup

1. **Device.** Run `rocminfo` (or `$HIPEP_THEROCK/bin/rocminfo` when the system
   one is missing) and read the first GPU agent's `Name:` (e.g. `gfx1151`).
   Return it as `device_gfx`.
2. **Runtime.** Check that each of these exists:
   - `BASE_PKG/bin/model_benchmark`;
   - exactly one `BASE_PKG/lib/libcustom_kernels_<arch>.so` (the release builds one arch);
   - `MODEL_PATH/genai_config.json`;
   - `MODEL_PATH/model.onnx`.

   If one is missing, stop: return `ok: false` with the error.
3. **Eval dir.** Create `EXP_ROOT/e2e_<model dir name>_<UTC yyyymmddTHHMMSS>`.
   - Inside it, create `runs/`.
   - Write `run.json` holding your inputs.
4. **Rebuild the release library** (LONG; DIR `EVAL_DIR/runs`, NAME `kbuild`):
   ```
   bash TOOLS_DIR/build_kernels.sh - EVAL_DIR/kbuild
   ```
   On a new machine it clones and configures hip-ep first, which takes longer.
   Every candidate is a rebuilt library, so this rebuild is the A/A's candidate
   (step 6): it measures any difference between the shipped library and a rebuild. A failure is fatal: return `ok: false` with the end of the log it names.
5. **Baseline** (LONG; DIR `EVAL_DIR/runs`, NAME `baseline`):
   ```
   python3 TOOLS_DIR/bench.py baseline --model MODEL_PATH --pkg BASE_PKG \
       --out EVAL_DIR/runs/baseline --lens LENS --gen GEN --reps REPS
   ```
   It writes `baseline.json`. A failure is fatal: return `ok: false` with the end of the
   log it names.
6. **Noise band** (only when `AA` is `true`).
   1. Package the rebuilt library:
      ```
      bash TOOLS_DIR/make_pkg.sh BASE_PKG EVAL_DIR/kbuild/libcustom_kernels_*.so EVAL_DIR/pkg/aa
      ```
   2. Run the A/A (LONG; DIR `EVAL_DIR/runs`, NAME `aa`):
      ```
      python3 TOOLS_DIR/bench.py ab --model MODEL_PATH --base BASE_PKG --cand EVAL_DIR/pkg/aa \
          --out EVAL_DIR/runs/aa --lens LENS --gen GEN --reps REPS --rounds ROUNDS
      ```
   3. Return `EVAL_DIR/runs/aa/ab.json` as `noise_from`: how far a rebuild of the release
      moves TTFT and decode. If the A/A fails, return `ok: false` with the end of its log.

   When `AA` is `false`, `noise_from` is `""`.
7. **A/A parity** (only with `AA`; LONG; DIR `EVAL_DIR/runs`, NAME `aa_parity`):
   ```
   python3 TOOLS_DIR/parity.py --model MODEL_PATH --base BASE_PKG --cand EVAL_DIR/pkg/aa \
       --out EVAL_DIR/runs/aa_parity --lens LENS
   ```
   Return its `exact` as `aa_parity_exact`. A rebuild of the release tag normally
   gives bit-identical logits; if it does not here, say so in your reply. The parity
   gate's bounds still apply.

## Return
```json
{"ok": true, "error": "", "eval_dir": "...", "device_gfx": "gfx1151", "base_kernel_lib_md5": "...",
 "rebuilt_kernel_lib_md5": "...", "baseline_json": ".../runs/baseline/baseline.json",
 "baseline": {"1024": {"ttft_us": 0, "decode_us": 0}, "2048": {"ttft_us": 0, "decode_us": 0}},
 "noise_from": ".../runs/aa/ab.json", "aa_parity_exact": true}
```
With `ok: false`, fill in what you have and say in `error` which step failed and why.

The numbers in `baseline` are the medians from `baseline.json` (`per_len.<L>.ttft_us` and
`decode_us`). `rebuilt_kernel_lib_md5` is in `EVAL_DIR/kbuild/BUILD_INFO.json`.
