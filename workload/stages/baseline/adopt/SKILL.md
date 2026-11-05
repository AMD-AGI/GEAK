---
name: adopt
description: Freeze an already-measured serving receipt into this pipeline's canonical baseline.json. No GPUs, no tokens, fully replayable. Use it when the number has already been measured and the environment has not changed, instead of measuring again.
---

# Adopt a measured receipt as the baseline

This step measures nothing. It reads a receipt someone else produced and translates it
into the only file downstream reads, `baseline.json`. Whether the receipt is *admissible*
is decided by `../finalize.py`, which belongs to the slot (`stages/CONTRACT.md` 3.1) —
read that file for the rules rather than looking for a second copy of them here.

**Nothing is retyped.** A receipt that nests its numbers, or splits them across two
documents, is read where it is. Hand-copying a figure into a flat file would produce a
receipt whose sha256 attests only to the copy, which is the one thing this step exists to
prevent.

## Input `$PARAMS`

| Field | Required | Meaning |
| --- | --- | --- |
| `receipt` | yes | path to the receipt JSON. **Comes from config; never hardcoded** |
| `throughput_key` | yes | which field holds the number. A **dotted path** (`request_metrics.throughput.output.tokens_per_second`, list indices allowed) reaches into a nested receipt |
| `reference_receipt` | no | a sibling document holding the reference figure, when the harness writes the throughput and the comparison into two files. Copied and hashed exactly like the receipt |
| `reference_key` | no | dotted path to the reference value, in `reference_receipt` if set, otherwise in the receipt |
| `valid_flags` | no | fields carrying the harness's validity verdict; defaults come from `config/default.json`. They are copied into `provenance` and the slot judges the same configured fields |
| `metric`, `unit` | no | recorded as written; default `output_token_throughput` / `tok/s` |
| `noise_band` | no | keep threshold; default `1.045`, calibrated by measurement rather than quoted |
| `config` | no | deployment shape (`tp`/`ep`/`gpus`/`concurrency`) when the receipt does not carry it under those names |
| `model_key`, `image_key` | no | dotted paths to what the receipt says it served; default `model` / `image`. Recorded in provenance, where `engine/core/target.py` cross-checks them against the declared target |
| `node` | no | which machine the receipt was measured on. An A/B is same-node or it is nothing, so the slot requires this |

**`throughput_key` is named, not guessed.** This step used to try five aliases in turn and
two more for the reference. An alias list is a silent choice between somebody else's field
names: when two of them are present and disagree, it picks whichever comes first in a list
written in this repository, and the baseline that anchors the next 48 hours of comparisons
is then decided here instead of in the scenario config.

## Running it

```bash
python3 "$WORKLOAD_ROOT/run.py" run baseline
```

Or manually: `source $OUT/env.sh && bash $SKILL_ROOT/run.sh`.

## Output

`$OUT/baseline.json` (schema in `../README.md`), `$OUT/receipt.copy.json` — a snapshot of
the original with its sha256, so a later edit to the original cannot go unnoticed —
`$OUT/reference.copy.json` when `reference_receipt` is set, and `$OUT/result.json`,
which `../finalize.py` writes after this script.
