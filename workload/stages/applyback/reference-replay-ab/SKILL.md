---
name: reference-replay-ab
description: Re-run the exact end-to-end command a reference run recorded, with the cold-re-verified kernels layered in through the reference harness's own python_overlay hook, and A/B against that reference — operator speedup and end-to-end delta stated separately.
---

# reference-replay-ab

An implementation of the applyback slot for the case where **the baseline was measured by
a harness that is not GEAK's `bench_e2e.sh`**.

## Why this exists alongside `overlay-e2e-ab`

An A/B is only an A/B if both legs were measured the same way. The default implementation
drives GEAK's own `bench_e2e.sh` with the `agentx` client, which assumes the server was
launched by GEAK's magpie phase. When the reference number came from somewhere else — a
CI reproduction directory with its own launcher, its own client flags, its own corpus
slice — measuring the candidate through `bench_e2e.sh` produces a number that differs
from the reference for reasons that have nothing to do with the kernels.

So this skill does not re-implement the measurement. It **re-runs the reference harness**,
unchanged, and changes exactly one thing: the server imports the overlay.

## How it installs the kernels

The slot's `build_overlay.py` layers every cold-re-verified kernel into one reversible
sitecustomize overlay, chain-seeded with `--from` (mounting two overlay directories on
`PYTHONPATH` at once does not work — only the first `sitecustomize` is imported), and
confirms with `overlay_setup.py check` that each target resolves **into** the overlay.
Run this skill inside the serving container, so that check is performed against the same
installation the server will import.

The overlay is then handed to the reference harness through the hook the harness already
has for this: a case entry carrying `python_overlay`. Nothing in the harness is patched.

## The one check that makes the result mean anything

A reference harness that silently ignores `python_overlay` would produce a clean "no
gain" that is really "the patch was never loaded". `summarize.py` therefore does not
trust the request: it reads back the `overlay_hashes` the launcher recorded in the
server's own `server_case.json` and requires them to match the overlay on disk, file by
file and hash by hash. A mismatch, or an absent record, makes the stage `fail` rather
than report a delta.

## Parameters

| Param | Meaning |
| --- | --- |
| `harness_dir` | Root of the reference harness (`IX_RUN`). |
| `harness_cmd` | Run script, relative to `harness_dir`; called as `<cmd> <case> <run_tag> <duration>`. |
| `case_file` | The harness's case table, relative to `harness_dir`. |
| `case_name` | Case to register and run. Created from `base_case` plus `python_overlay`. |
| `base_case` | The case whose server args the reference leg used — copied verbatim. |
| `overlay_subdir` | Where under `harness_dir` to place the overlay (the launcher requires it inside its own root). |
| `reference_run` | The reference leg's result JSON — the A/B's other half. |
| `reference_key` | Throughput field in it. |
| `run_tag_prefix`, `repeats`, `duration_s` | The candidate legs. |

## Output

`$OUT/overlay/`, `$OUT/applied.json`, `$OUT/reference.json`, `$OUT/e2e/bench_summary.json`,
and whatever the slot's `finalize.py` writes from those: `ab.json` and `result.json`.
