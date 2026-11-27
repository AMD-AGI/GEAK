# hip-ep Tuner — Tuning lane

You refresh hip-ep's autotune tables for `OPS` on this model, rebuild the kernel
library with them, and gate the candidate. You never pick configs yourself.
hip-ep's own sweep drivers pick them, and its `update_lut.py` writes each table.

## PHASE=tuning
1. **Refresh the tables and rebuild the library** (LONG; DIR `LANE_DIR`, NAME `lut`).
   ```
   python3 TOOLS_DIR/lut_refresh.py --model MODEL_PATH --out LANE_DIR/lut --arch ARCH --ops OPS
   ```
   The tool sweeps each op on the model's own shapes, rebuilds the library into
   `LANE_DIR/lut/kbuild/` when a table changed, and writes the changed tables to the
   overlay `LANE_DIR/lut/overlay`.

   Read `LANE_DIR/lut/lut_refresh.json`. Each op under `ops` has `changed` and, when
   the tool skipped it, `skipped` with the reason. If no op changed, return
   `accepted: false` with reason `no table change` and the skip reasons. Do not build
   a package in that case.
2. **Package.**
   ```
   bash TOOLS_DIR/make_pkg.sh BASE_PKG LANE_DIR/lut/kbuild/libcustom_kernels_*.so LANE_DIR/pkg
   ```
3. **Timing gate** (LONG; DIR `LANE_DIR`, NAME `ab`), against the release:
   ```
   python3 TOOLS_DIR/bench.py ab --model MODEL_PATH --base BASE_PKG --cand LANE_DIR/pkg \
       --out LANE_DIR/ab --lens LENS --gen GEN --reps REPS --rounds ROUNDS --noise-from AA_JSON
   ```
4. **Parity gate** (LONG; DIR `LANE_DIR`, NAME `parity`), against the release:
   ```
   python3 TOOLS_DIR/parity.py --model MODEL_PATH --base BASE_PKG --cand LANE_DIR/pkg \
       --out LANE_DIR/parity --lens LENS
   ```
5. **Verdict.** `accepted` is true only if both of these hold:
   - `LANE_DIR/ab/ab.json` has `gate.accepted == true`;
   - `LANE_DIR/parity/parity.json` has `pass == true`.

   Never override the tools' verdicts. If a gate fails, `reason` names it with its
   numbers (e.g. `TTFT@2048 -0.6% inside noise 1.0%`).
   Name the ops whose tables changed in `reason` either way.

## Return
```json
{"accepted": true, "reason": "...", "pkg": "LANE_DIR/pkg", "overlay": "LANE_DIR/lut/overlay",
 "per_len": {"1024": {"ttft_gain": 0.0, "decode_delta": 0.0}}, "parity_top1": 1.0, "parity_rel_l2": 0.0}
```
