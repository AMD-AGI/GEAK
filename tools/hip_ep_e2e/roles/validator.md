# hip-ep Validator — Validate and report

You re-measure the final package against the release, independently of the lanes'
own A/Bs, and write the report. You change nothing, and you return the report as text:
the workflow saves it.

## PHASE=validate
1. If `FINAL_PKG` equals `BASE_PKG`, nothing was accepted: set `validation_status` to
   `no_change`, skip the measurements, and still write the report.
2. Otherwise, run both gates against the release. Both are LONG (DIR `EVAL_DIR/runs`,
   NAME `final_ab` and `final_parity`):
   ```
   python3 TOOLS_DIR/bench.py ab --model MODEL_PATH --base BASE_PKG --cand FINAL_PKG \
       --out EVAL_DIR/runs/final_ab --lens LENS --gen GEN --reps REPS --rounds ROUNDS --noise-from AA_JSON
   python3 TOOLS_DIR/parity.py --model MODEL_PATH --base BASE_PKG --cand FINAL_PKG \
       --out EVAL_DIR/runs/final_parity --lens LENS
   ```
   Set `validation_status` from the result:

   | Result | Status |
   |---|---|
   | Both gates pass | `validated` |
   | The A/B rejects it (no win beyond noise, or the other metric regressed) | `not_reproduced` |
   | Parity fails | `parity_failed` |
3. Write the report in markdown and return it as `report_md`. Do not write it to a file.
   Keep it short and factual, with these sections:
   - **Setup:** the model, the release package, the device (`DEVICE_GFX`), the A/A's TTFT
     and decode deltas and the noise band they set (`gate.per_len` in `AA_JSON`), and
     whether the A/A's logits were identical (`AA_PARITY_EXACT`).
   - **Baseline:** the release's TTFT at each length and decode ms/token (`ttft_base`,
     `decode_base` in the same `gate.per_len`).
   - **Profile:** the prefill and decode top kernels, from `HEADS_JSON`.
   - **Lanes:** one row per entry of `HISTORY`: the lane, its ops, each tunable op's
     largest phase share (`shares`), whether it was accepted, the TTFT and decode deltas,
     the parity top-1 and relL2, and the `reason` exactly as `HISTORY` records it (a lane
     that did not run is not a lane that found nothing).
   - **Final:** release vs final, from your own measurement in step 2, with the final
     package path and its kernel library md5.

   Only numbers from the JSON files the tools wrote go in the report. Do not round a
   rejection into a win, and do not estimate.

## Return
```json
{"validation_status": "validated", "report_md": "# hip-ep end-to-end report\n...",
 "per_len": {"1024": {"ttft_gain": 0.0, "decode_delta": 0.0}}}
```
