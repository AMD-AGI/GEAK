---
name: run-report
description: Build or read the report for a GEAK run — one self-contained page (report/geak_run_report_<model>.html) with every LLM API call's cost, tokens, billed time and elapsed time, arranged by phase, by workflow invocation and as a role tree of agents. Use when asked where a run's time or money went, which phase or agent dominates it, or which work is cheap enough to delegate to a smaller model.
---

# GEAK run report

One run, one page: `<eval_dir>/report/geak_run_report_<model>.html`, with a
Markdown twin beside it. Everything it shows comes from GEAK's own code in this
checkout — no other tool is needed to build it or to read it.

| Question | Where on the page |
| --- | --- |
| What did the run cost, and in which bucket? | Header totals: cost split into cache-write / cache-read / uncached-context / output, tokens, per-model split |
| How long did it take? | *elapsed* (first request → last response) beside *billed span* (time spent waiting on the model) |
| Which phase took the time and the money? | *Time, cost and tokens by phase* |
| Which workflow invocations are counted? | *Workflow invocations counted* — a resumed or re-entered run is several |
| Which agent, and what did it say? | The role tree: click an agent for its cost, tokens, elapsed time, prompt snippet and output, then any row for one API call |

The model name comes from `kb_identity.json` (`dims.model`), falling back to
`env_report.json` and then the eval dir name.

## How it is built

GEAK issues almost no LLM calls itself: Claude Code runs the workflow and writes
every agent's transcript into its own config home (`$CLAUDE_CONFIG_DIR`, else
`~/.claude`). The report **reads** those transcripts; no GEAK code writes a token
ledger during the run.

1. `e2e_workflow/scripts/llm_ledger.py` turns the transcripts into
   `<eval_dir>/reports/trace/llm_calls.jsonl` (one row per API call) plus
   `token_stats.{json,md}`.
2. `interface/geak_call_tree_html.py` folds that ledger into the page.

`interface/geak_report.py` runs both, and it is the last step of every E2E and
kernel workflow, so on a normal run there is nothing to do but open the page. To
rebuild it:

```bash
python3 interface/geak_report.py --eval-dir <EVAL_DIR>
# the Claude home is gone, but the run mirrored it:
python3 interface/geak_report.py --eval-dir <EVAL_DIR> --claude-home <EVAL_DIR>/llm_trace
```

`--claude-home` adds a search root; the eval dir selects the run.

## Which transcripts count

A run is every workflow invocation whose **own** eval dir is this one — by its
`wf_<runId>.json` record, or, for an invocation that has no record (still
running, or killed before it returned), by its journal naming this directory as
its `eval_dir`. Only those invocations' `subagents/workflows/<runId>/` transcripts
are read, so a concurrent session that merely mentions the path is never billed
to the run. Journal-only evidence is marked INFERRED on the page.

Each agent is named by the label the workflow script gave it, read from the
runtime's `agent-<id>.meta.json` (`eng r2_d0:algorithm`, `extract_op <op>`), and
placed in the phase the runtime recorded for it. An agent reads `(driver)` only
when it is genuinely the launching session.

## Reading the numbers honestly

- **Read the Completeness block first.** It says how the transcripts were
  selected (`run-scoped`, `run-scoped-inferred`, `partial`, `substring-fallback`)
  and lists every caveat. A `substring-fallback` total may include other sessions.
- **Cost is estimated** from token buckets and the rate card in `llm_ledger.py`
  (`DEFAULT_RATES`, overridable with `--rates`). It is not an invoice.
- **Tokens are deduplicated by `message.id`.** A transcript repeats a response
  once per content block; counting rows overstates the bill by ~60%.
- **Output tokens include thinking and tool arguments.** Do not add thinking on top.
- **Elapsed is not billed span.** Elapsed runs from first request to last
  response and includes compiling and benchmarking; billed span is the sum of
  per-call durations. Phases overlap when work ran in parallel, so their elapsed
  times do not add up to the run's.
- **The role tree is organisational, not a spawn graph.** Nesting is by role rank
  and execution order; transcripts carry no cross-agent spawn edge.

## What the run bought

Throughput is not in the transcripts. `interface/geak_outcome_report.py` reads
it from the run's own artifacts (`reports/geak_outcome.{json,md}`):

```bash
python3 interface/geak_outcome_report.py <EVAL_DIR> --stdout
```

Read `—` as *the artifact was absent*, never as zero.

## When the page is missing

A report is written when a run finishes. A run that died first can be rendered
by hand with the command above, as long as its transcripts survive: check
whether the Claude home (or the run's `llm_trace/` mirror) outlived the run
before concluding anything about the run itself.
