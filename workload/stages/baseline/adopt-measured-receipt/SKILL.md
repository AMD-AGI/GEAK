---
name: adopt-measured-receipt
description: Freeze an already-measured serving receipt (a measured_baseline.json or similar) into this pipeline's canonical baseline.json. No GPUs, no tokens, fully replayable. Use it when the baseline has already been measured and the environment has not changed, instead of measuring again.
---

# Adopt a measured receipt as the baseline

This step measures nothing. It reads a receipt someone else produced, checks whether it
is admissible as a reference, and translates it into the only file downstream reads,
`baseline.json`.

**Its value is in the checking, not the translating.** A receipt silently adopted while
non-compliant, or produced under a different configuration, anchors the next 48 hours of
comparisons to the wrong reference — without ever raising an error.

## Input

`$PARAMS` (written to `$OUT/params.json` by `stage.py prepare`):

| Field | Required | Meaning |
| --- | --- | --- |
| `receipt` | yes | path to the receipt JSON. **Comes from config; never hardcoded** |
| `metric` | no | metric name recorded in the output; default `output_token_throughput` |
| `unit` | no | default `tok/s` |
| `noise_band` | no | keep threshold; default `1.045` (calibrated by measurement, not the documented 1.02) |
| `throughput_key` | no | which field to read; by default each known alias is tried in turn |
| `reference_key` | no | reference-value field; default `ci_output_tokens_per_second` |
| `max_reference_drift` | no | maximum relative deviation from the reference; default `0.05`, beyond which the stage reports `blocked` |
| `require_valid` | no | default `true`: `submission_valid`/`power_valid` being false yields `blocked` |
| `node` | no | which node the receipt was measured on; `null` if omitted, and applyback will warn about cross-node risk |

## Running it

```bash
python3 "$WORKLOAD_ROOT/scripts/stage.py" run baseline
```

Or manually: `source $OUT/env.sh && bash $SKILL_ROOT/run.sh`.

## Output

`$OUT/baseline.json` (schema in `../README.md`), `$OUT/receipt.copy.json` (a snapshot of
the original receipt with its sha256, so a later edit to the original cannot go
unnoticed), and `$OUT/result.json`.

## When this reports `blocked` rather than `ok`

- `receipt` is unset, or the file does not exist.
- No throughput field can be found in the receipt.
- `submission_valid` or `power_valid` is false.
- Deviation from the reference exceeds `max_reference_drift` — which means the
  reproduction itself never matched, and adopting it writes a known discrepancy into
  every subsequent comparison.

On `blocked`, `notes` states what is missing and which parameter to change before
re-running.
