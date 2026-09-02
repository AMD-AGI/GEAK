<!-- Copy this file to <model-name>/README.md before filling it in. The ../../ links below
     resolve from that location, one level deeper than this template sits. -->

# <Model> on <hardware> — <one-line recipe summary>

**<Verified|Measured> win: <+X.XX%> <metric>** (<before> → <after>), <accuracy metric> <before> →
<after>. <One sentence on what carries the win: a config file, a patch, a set of flags.>

Found <date> over a <duration> run.

## Reproduction status

Say which tier this is in the first line, and back it with what actually happened.

- **Verified win** — the exported artifact was applied to a tree it had not been measured on and
  re-measured, reaching the number again. Give the instances and the numbers.
- **Measured win** — arms interleaved across restarts against a stated floor, distributions disjoint,
  but nobody has deployed the artifact from scratch and reproduced it. Say so plainly, and say what
  *was* done instead.

Then state separately whether **artifact integrity** was checked: does applying the exported patch to
the recorded pristine base reproduce the measured tree byte-for-byte? That is a different and weaker
claim than reproducing the throughput, and conflating the two is the failure this section exists to
prevent.

> A measured win with an honest floor and a deployable artifact is worth having. A measured win
> presented as a verified one is worse than nothing, because the next reader cannot tell which of
> your claims to check.

## Environment fingerprint

Mark every field **load-bearing** (part of a lookup key, or determines the shapes) or
**descriptive**. A reader diffs this table first; if a load-bearing field differs, the artifact will
silently do nothing.

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | arch (`gfxNNN`), CU count, count of devices | **yes** |
| container | image digest | |
| framework | name + version | |
| key library | name + **commit sha** | |
| model | name, TP size | **yes** |
| precision | weights and KV cache, separately | **yes** |
| backends | attention, MoE, anything selectable | |

State explicitly where a config label disagrees with what actually ran, and which one you verified.

## Launch configuration

The exact server invocation, plus resolved values that are not visible in it (recovered from the
server log, not assumed). Then the environment variables, if any — and if none, say so explicitly,
because "no env recipe" is itself a finding.

## Workload

ISL, OSL, concurrency, prompt count, warmups, seed, and **which benchmark harness**. Note which
workload parameters set the shapes you tuned.

## Baseline and noise floor

| | value |
| --- | --- |
| stock, this stack (N instances: …) | |
| with the artifact (N instances: …) | |
| delta | |

| noise floor | spread |
| --- | --- |
| repeating the benchmark within one process | |
| across restarts | |

State which floor applies and why. If the change requires a restart, it is the restart floor
(`../../tuning-core/measurement.md` Rule 3b). Report whether the two distributions are disjoint,
not just the difference of means.

## Deploy

Exact commands, in order, including **cache invalidation** and whether a restart is required.
Then list every way this deploy silently does nothing — a no-op deploy is the default failure mode
for anything config-driven, and each one you document is a debugging session the next reader skips.

## Engagement check

The command that proves the win is live, plus its **expected output in both directions** — engaged
and not engaged. Prefer kernel identity from a profile over a log line
(`../../tuning-core/engagement_verification.md`). If a log-based check needs an env var to emit
anything, say so, and give the flag-free alternative.

## Accuracy gate

The metric, the harness and its pinned version, and the before/after scores with stderr. State the
threshold the win must clear.

## What was tried and did not work

| attempt | kernel-level result | end-to-end | verdict |
| --- | --- | --- | --- |

Include the measured numbers, not just names, and say why each was dropped. Negative results with a
real kernel win but a within-noise end-to-end delta are the most valuable rows in the table. Where an
attempt was rejected before it cost a server instance, say so and give the number that killed it —
refusing to spend an instance is a result.

## What the bundle did not record

An explicit gap list, checked against the sections above: container digest, library commits, the
server process environment, anything the entry asserts from prose rather than from a captured file.
**"Not recorded" is a finding; a guessed value is a trap.** State why each gap matters to someone
trying to match this environment.

## When this entry stops applying

Enumerate the changes that make the artifact inert, and note that they fail **silently**. Then say
what is still reusable when they occur — usually the shape list, the target ranking, and the winning
kernel families.

## Provenance

Path to the task bundle, run log, per-patch manifests, and anything else needed to audit the claim.
