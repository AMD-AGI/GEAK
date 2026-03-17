# GEAK Dashboard Design

## Goal

The GEAK dashboard provides a live terminal UI for orchestration runs so users can
understand what is happening across GPUs without reading a wall of interleaved
agent logs.

The design goal is:

- show the current round and global state at a glance
- show what each worker is doing right now
- show the latest natural-language reasoning from each worker
- show important validation and speedup milestones
- suppress noisy raw sub-agent console spam when dashboard mode is enabled

## Activation

The dashboard is enabled by setting:

```bash
GEAK_DASHBOARD=1
```

When enabled, `run_orchestrator()` creates a `Dashboard` instance and routes
status updates into it.

## Main UI Regions

The dashboard has three major regions:

1. Header
2. Status + Checklist + Worker cards
3. Recent activity feed

### Header

The header shows:

- dashboard title
- kernel name
- elapsed run time

### Status Panel

The status panel shows run-level information:

- current round
- current phase
- best speedup so far
- per-round speedup summaries when available

### Checklist

The left-side checklist shows run milestones in a stable order so users can
quickly tell what has already happened and what is currently active.

Checklist items currently include:

- URL resolved
- generating tests
- test harness ready
- profiling context ready
- baseline metrics ready
- commandment ready
- round loop state
- dispatch state
- evaluation state
- final report state

Each item has a simple state:

- `OK`
- `NOW`
- `...`
- `ERR`

and may include a short detail such as a filename, test count, or latency.

### Worker Cards

Each GPU gets its own mini card.

Each worker card tracks:

- GPU id
- status (`idle`, `running`, `done`, `error`)
- current objective / what it is trying now
- stage
- step number
- current tool
- latest natural-language intent
- latest summarized result
- short per-worker recent history

The worker cards are the main improvement over the original GPU table because
they keep both state and reasoning together in one place.

### Activity Feed

The activity feed shows the latest cross-worker timeline:

- round starts
- dispatch starts
- worker task starts
- condensed worker LLM/tool updates
- task completion or task errors
- benchmark and evaluation milestones

The feed deduplicates repeated identical messages to reduce noise.

## Data Flow

There are two structured status paths feeding the dashboard.

### 1. Orchestrator-level events

The orchestrator uses `_print()` and dashboard parsing for:

- round transitions
- dispatch messages
- round-best evaluation messages
- benchmark/profile/finalize updates

These are mostly parsed from orchestrator text output.

### 2. Worker-level events

Homogeneous pool workers emit structured task lifecycle events through an
optional `task_status_hook`.

The pool currently emits:

- `task_start`
- `task_done`
- `task_error`
- `task_message`

`task_message` is used for condensed worker LLM and tool updates.

This path is the key mechanism that lets the dashboard reflect dynamic worker
progress instead of only global orchestrator milestones.

### 3. LLM-authored status intent

Homogeneous worker tasks are instructed to emit short natural-language updates
before major actions and may optionally emit `GEAK_DASH` markers. The dashboard
prefers these human-readable worker intents over raw command-shaped output
whenever available.

## Suppressing Raw Worker Spam

The original homogeneous run printed raw `InteractiveAgent` messages like:

- `mini-swe-agent (step N, $X.XX)`
- full assistant text
- full tool output

That output polluted the terminal and competed with the dashboard.

Dashboard mode now changes worker behavior:

- `InteractiveAgent` can emit structured UI messages through a callback
- `InteractiveAgent` can suppress direct console output
- `ParallelAgent` installs that callback on worker agents when a dashboard task
  hook is present

As a result, worker reasoning is now routed into the dashboard instead of
printed raw to the terminal.

## Natural-Language Worker Summaries

Worker assistant/tool messages are compressed into short natural-language
summaries before being shown in the dashboard.

Examples:

- `GPU 7 step 2 -> bash: view the profile data to understand hotspots`
- `GPU 4 step 11 -> save_and_test: run the test to establish a baseline`
- `GPU 6 bash: inspected file contents`
- `GPU 5 save_and_test: correctness passed`
- `GPU 4 save_and_test: benchmark 0.0338 ms`

The summary formatter tries to turn tool-shaped output into human-meaningful
signals such as:

- file inspected
- file edited
- correctness passed
- benchmark running
- benchmark `<latency>` ms
- test passed / failed
- profile run complete
- strategy updated

## Worker Stages

Each worker card includes a stage derived heuristically from the current tool
and summary content.

Current stages include:

- `starting`
- `planning`
- `analyzing`
- `editing`
- `testing`
- `benchmarking`
- `profiling`
- `validating`
- `working`
- `completed`
- `failed`

The stage is meant to be stable enough for quick scanning while still updating
as the worker changes activity.

## Real GPU ID Tracking

The dashboard no longer assumes GPUs are `0..N-1`.

Instead, it accepts the real `gpu_ids` list from the orchestrator and builds
worker state using those actual device ids. This is important when runs use
explicit devices like `4,5,6,7`.

## Self-Contained Run Directory Requirement

The dashboard does not require a folder named `homo`, but the run directory used
by `geak-orchestrate` must be self-contained.

A valid run directory must contain:

- preprocess artefacts (`resolved.json`, `discovery.json`, `profile.json`, etc.)
- `.geak_resolved` repo snapshot
- harness path and harness file
- internally consistent absolute paths

Copied preprocess bundles can fail if their embedded paths still point at the
original location. In practice, a dedicated run directory such as `homo` is
useful because it keeps orchestration outputs isolated and self-consistent.

## Logging and Color Semantics

Worker state colors currently follow:

- yellow: running
- green: done
- red: error
- dim: idle

Global best speedup is highlighted in green when it exceeds `1.0x`.

## Current Limitations

- Worker summaries are heuristic, not model-authored structured summaries.
- Some tool results, especially large `save_and_test` outputs, still compress
  imperfectly.
- The activity feed is global, so repeated high-frequency worker updates can
  still crowd out more important events.
- The dashboard currently uses terminal rendering only; there is no persisted
  structured event log yet.

## Recommended Next Improvements

Good follow-up dashboard improvements would be:

- emit explicit worker-side structured progress markers from the LLM itself
- persist dashboard events to `jsonl` or markdown for post-run review
- add per-worker metric fields such as latest latency and delta vs baseline
- improve `save_and_test` parsing for clearer benchmark/correctness summaries
- add a compact round summary strip for best worker / best latency / best patch

## Summary

The dashboard now does more than show raw logs:

- it tracks true worker lifecycle
- it suppresses raw sub-agent spam
- it shows condensed worker reasoning
- it presents per-GPU mini boxes
- it keeps a human-readable global activity feed

This makes homogeneous multi-GPU runs much easier to monitor and explain to
others in real time.
