# hip-ep Validator — Report

You write the run's report. You change nothing, and you return the report as text:
the workflow saves it.

## PHASE=validate
Write the report in markdown and return it as `report_md`. Do not write it to a file.
Keep it short and factual, with these sections:
- **Setup:** the model, the release package, the device (`DEVICE_GFX`), the A/A's TTFT
  and decode deltas at each length (`gate.per_len` in `NOISE_FROM`; "not measured" when
  it is empty), and whether the A/A's logits were identical (`AA_PARITY_EXACT`).
- **Baseline:** TTFT at each length and decode ms/token, from `BASELINE_JSON`.
- **Profile:** the prefill and decode top kernels with their shares and source files,
  from `HEADS_JSON`.

Only numbers from the JSON files the tools wrote go in the report. Do not estimate.

## Return
```json
{"report_md": "# hip-ep end-to-end report\n..."}
```
