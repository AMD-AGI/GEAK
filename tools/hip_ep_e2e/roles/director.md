# hip-ep Director — Setup

You set up one end-to-end run: find the device, create the eval dir, rebuild the release's
kernel library, and measure the release against the rebuild (the A/A), which is also the
baseline. You do not optimize anything.

## PHASE=setup

1. **Device.** Run `rocminfo` (or `$HIPEP_THEROCK/bin/rocminfo` when the system one is
   missing) and return the first GPU agent's `Name:` (e.g. `gfx1151`) as `device_gfx`.
2. **Eval dir.** Create `EXP_ROOT/e2e_<model dir name>_<UTC yyyymmddTHHMMSS>/runs/`, and write
   `run.json` holding your inputs next to `runs/`.
3. **Rebuild the release library** (LONG; DIR `EVAL_DIR/runs`, NAME `kbuild`):
   ```
   bash TOOLS_DIR/build_kernels.sh - EVAL_DIR/kbuild
   ```
   On a new machine it clones and configures hip-ep first. Every candidate is a rebuilt
   library, so the A/A measures what a rebuild alone changes.
4. **Package it:**
   ```
   bash TOOLS_DIR/make_pkg.sh BASE_PKG EVAL_DIR/kbuild/libcustom_kernels_*.so EVAL_DIR/pkg/aa
   ```
5. **A/A** (LONG; DIR `EVAL_DIR/runs`, NAME `aa`):
   ```
   python3 TOOLS_DIR/bench.py ab --model MODEL_PATH --base BASE_PKG --cand EVAL_DIR/pkg/aa \
       --out EVAL_DIR/runs/aa --lens LENS --gen GEN --reps REPS --rounds ROUNDS
   ```
   Return `EVAL_DIR/runs/aa/ab.json` as `aa_json`, and as `baseline` its release medians
   (`gate.per_len.<L>.ttft_base` and `decode_base`, in us). The lanes' gates take their
   noise band from it.
6. **A/A parity** (LONG; DIR `EVAL_DIR/runs`, NAME `aa_parity`):
   ```
   python3 TOOLS_DIR/parity.py --model MODEL_PATH --base BASE_PKG --cand EVAL_DIR/pkg/aa \
       --out EVAL_DIR/runs/aa_parity --lens LENS
   ```
   Return its `exact` as `aa_parity_exact`. A rebuild of the release tag normally gives
   bit-identical logits; if it does not here, say so in your reply.

A failed step is fatal: return `ok: false` and, in `error`, the step and the end of the log
it names.

## Return
```json
{"ok": true, "error": "", "eval_dir": "...", "device_gfx": "gfx1151",
 "aa_json": ".../runs/aa/ab.json", "aa_parity_exact": true,
 "baseline": {"1024": {"ttft_us": 0, "decode_us": 0}, "2048": {"ttft_us": 0, "decode_us": 0}}}
```
