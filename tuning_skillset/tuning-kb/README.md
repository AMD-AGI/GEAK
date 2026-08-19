# tuning-kb — known-good tuning results, per model

The rest of this skillset teaches you how to *find* a win. This directory records wins that were
already found, verified, and reproduced, so that a run landing on an environment we have already
tuned can **deploy the known answer in minutes and spend its time somewhere new** instead of
rediscovering the same rows.

One subdirectory per model. Each entry is self-contained: the environment it was measured on, the
launch configuration it assumes, the artifact that carries the win, and the numbers to expect.

| model | hardware | recipe | verified win | entry |
| --- | --- | --- | --- | --- |
| Qwen3-8B | MI355X (gfx950, 256 CU), TP=1 | SGLang 0.5.17 + aiter, FP8 weights + FP8 KV | **+3.95%** output throughput (3642.1 → 3786.1 tok/s), gsm8k 0.9348 | [`qwen3-8b/`](qwen3-8b/) |
| Gemma-4-26B-A4B-it | MI355X (gfx950, 256 CU), TP=2 | SGLang + Triton attention/MoE, bf16 throughout | not yet run — environment documented, results pending | [`gemma-4-26b-a4b-it/`](gemma-4-26b-a4b-it/) |

## How to use an entry

**Read the fingerprint before the result.** Every entry opens with an environment fingerprint,
split into fields that are load-bearing for the win and fields that merely describe the run. A
tuned config is a **lookup keyed on a tuple**; if a load-bearing field differs, the lookup misses
and the artifact does nothing. It does not warn you — it falls back and serves at the old speed
(`../tuning-core/engagement_verification.md`).

1. **Fingerprint the environment you are on**, then diff it against the entry's table.
2. **On a full match**: deploy the artifact, restart, run the entry's engagement check, and
   reproduce the stated number before believing it. Budget one hour, not one day. Then go look for
   something the entry does not already cover.
3. **On a partial match**: treat the entry as a *starting hypothesis*, not an answer. The shapes
   and the winning kernel families usually still point the right way even when the exact rows
   miss — re-tune those shapes rather than re-deriving the target list from scratch.
4. **On a mismatch in arch or CU count**: do not deploy. Those are literally part of the config
   key, so the rows are unreachable. Use the entry only for its shape list and its record of what
   turned out not to matter.

**Reproduce, do not assume.** An entry is evidence that a win existed on a specific stack, not a
guarantee about yours. Every entry states the noise floor it was measured against so you can tell
a real reproduction from a coincidence.

## What every entry must record

Entries exist to be *reused*, which means an entry that omits the boring parts is not usable.
`ENTRY_TEMPLATE.md` is the skeleton; the load-bearing sections are:

- **Environment fingerprint** — container digest, framework and library commits, GPU arch and CU
  count, TP, quantization. Marked load-bearing or descriptive, per field.
- **Launch configuration** — the exact server flags, because they determine the shapes. This is
  the part most often left out and most often the reason a reuse silently fails: on Qwen3-8B the
  tuned rows are keyed on M values that come directly from `--chunked-prefill-size` and the
  benchmark's concurrency, so changing either makes the entry inert.
- **Workload** — ISL, OSL, concurrency, prompt count, warmups. A win at one operating point is not
  a win at another.
- **Baseline and noise floor** — what stock measures on this stack, and the restart-to-restart
  spread. Without the floor, the delta is unreadable.
- **The artifact** — checked in under `artifacts/`, deployable as-is, plus the exact commands to
  apply it and to invalidate any derived cache.
- **Engagement check** — the command that proves the win is live, with its expected output.
- **Accuracy gate** — the score the win must not regress, and how it was measured.
- **What was tried and did not work** — the most valuable and most often discarded section. It is
  what stops the next run from spending a day on a dead end. Record negative results with their
  measured numbers, not just as a list of names.

## Adding an entry

Copy `ENTRY_TEMPLATE.md` to `<model-name>/README.md`, fill it in, and put the deployable artifact
in `<model-name>/artifacts/`. Two rules:

- **Only reproduced results go in.** The bar is that someone else, on a clean instance, reached
  the number using nothing but the artifact and the entry. If it has not been reproduced from the
  artifact alone, it belongs in a run's `FINDINGS.md`, not here.
- **Record the measured spread, and never a single run.** A number without its noise floor cannot
  be checked by the next reader, and this whole directory is only worth anything if its claims are
  checkable.

## If you are running a blind evaluation, delete this directory first

This directory is an answer key. That is the point of it in production, and it is disqualifying
when the same task bundle is being used to *measure* whether an agent can find a win on its own.

The `qwen3-8b/` entry in particular contains the exact four CSV rows that a from-scratch Qwen3-8B
run is supposed to discover. Hand an agent this skillset alongside that bundle and it can deploy
the result in minutes — a correct outcome for a production run, and a meaningless one as an
evaluation.

```bash
rsync -a --exclude tuning-kb/ tuning_skillset/ <bundle>/tuning_skillset/
```

Entries for *other* models are harmless and worth keeping — they carry method and environment
detail without the answer. The rule is narrow: **exclude the entry for the model under test.**

