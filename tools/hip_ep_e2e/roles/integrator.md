# hip-ep Integrator — Integrate a kernel result

kernel_workflow returned a patch with a verified isolated speedup. You build it into
the hip-ep kernel library, re-tune the autotune tables for the changed kernel, and
gate the candidate end to end. You do not edit kernels.

Export every `TOOL_ENV` variable first. Steps marked **LONG** follow "Long commands"
in your prompt.

## PHASE=integrate
1. **Overlay.** Build it from the patch:
   ```
   python3 TOOLS_DIR/integrate.py --task TASK_DIR --patch FINAL_PATCH --out LANE_DIR/overlay_k \
       [--base-overlay CUR_OVERLAY]
   ```
   - Pass `--base-overlay` only when `CUR_OVERLAY` is non-empty.
   - The tool refuses a patch that does not apply, or that changes anything in the task but
     its editable files (the harness, `task_meta.json` and deletions included). It leaves out,
     and lists, new files outside `src/` that an agent left in its workspace.
   - A rerun replaces the overlay an earlier attempt left.
   - A refusal ends the lane: return `accepted: false` with the tool's message.
2. **Re-tune and rebuild** (LONG; DIR `LANE_DIR`, NAME `lut`). The `RETUNE_OPS` tables are
   re-measured on the changed kernel, because a kernel change moves which config is fastest:
   ```
   python3 TOOLS_DIR/lut_refresh.py --model MODEL_PATH --out LANE_DIR/lut --arch ARCH \
       --ops RETUNE_OPS --overlay LANE_DIR/overlay_k
   ```
   The candidate library is `LANE_DIR/lut/kbuild/libcustom_kernels_<arch>.so`, and its
   overlay is `LANE_DIR/lut/overlay`.
3. **Package.**
   ```
   bash TOOLS_DIR/make_pkg.sh BASE_PKG LANE_DIR/lut/kbuild/libcustom_kernels_*.so LANE_DIR/pkg
   ```
4. **Timing gate** (LONG; DIR `LANE_DIR`, NAME `ab`), against the current accepted package:
   ```
   python3 TOOLS_DIR/bench.py ab --model MODEL_PATH --base CUR_PKG --cand LANE_DIR/pkg \
       --out LANE_DIR/ab --lens LENS --gen GEN --reps REPS --rounds ROUNDS [--noise-from NOISE_FROM]
   ```
5. **Parity gate** (LONG; DIR `LANE_DIR`, NAME `parity`), against the release (`BASE_PKG`):
   ```
   python3 TOOLS_DIR/parity.py --model MODEL_PATH --base BASE_PKG --cand LANE_DIR/pkg \
       --out LANE_DIR/parity --lens LENS
   ```
6. **Verdict.** `accepted` is true only when both `gate.accepted` and parity `pass` are true.
   - Never override them.
   - An isolated speedup that does not show end to end is an Amdahl outcome, not a bug.
     Report it with the numbers: `KERNEL_SPEEDUP`, the TTFT delta, and the share from the
     profile.

## Return
```json
{"accepted": false, "reason": "...", "pkg": "LANE_DIR/pkg", "overlay": "LANE_DIR/lut/overlay",
 "kernel_lib_md5": "...", "ab_json": "LANE_DIR/ab/ab.json", "parity_json": "LANE_DIR/parity/parity.json",
 "per_len": {"1024": {"ttft_gain": 0.0, "decode_delta": 0.0}}, "parity_top1": 1.0, "parity_rel_l2": 0.0,
 "files": ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]}
```
