# hip-ep Integrator — Integrate a kernel result

kernel_workflow returned a patch with a verified isolated speedup. You build it into
the hip-ep kernel library, re-tune the autotune tables for the changed kernel, and
gate the candidate end to end. You do not edit kernels.

## PHASE=integrate
1. **Overlay.** Build it from the patch:
   ```
   python3 TOOLS_DIR/integrate.py --task TASK_DIR --patch FINAL_PATCH --out LANE_DIR/overlay_k \
       [--base-overlay CUR_OVERLAY]
   ```
   Pass `--base-overlay` only when `CUR_OVERLAY` is non-empty. The tool refuses a patch
   that does not apply or that changes anything in the task but its editable files; a
   refusal ends the lane: return `accepted: false` with the tool's message.
2. **Re-tune, rebuild and gate.** Read `TOOLS_DIR/roles/tuner.md` and follow its
   PHASE=tuning steps 1–5, with three changes:
   - `lut_refresh.py` also gets `--overlay LANE_DIR/overlay_k`, so `OPS`' tables are
     re-measured on the changed kernel (a kernel change moves which config is fastest);
   - the library always differs from the release, so package it even when no table changed;
   - the timing gate's `--base` is `CUR_PKG` (the current accepted package); parity stays
     against `BASE_PKG`.

   An isolated speedup that does not show end to end is an Amdahl outcome, not a bug:
   give `KERNEL_SPEEDUP` and the TTFT delta in `reason`.

## Return
As the tuner's, plus `"files"`: the overlay files from step 1.
