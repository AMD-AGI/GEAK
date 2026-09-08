---
name: run-report
description: Build the report for a finished GEAK e2e run — the phase → task → subtask → sub-sub-task tree of every LLM API call (ISL/OSL, tokens, wall-clock, USD, tool calls) joined to what each phase actually bought in serving throughput. Use when asked where a run's time or money went, which phase paid for which speedup, or which work is cheap enough to delegate to a smaller model.
---

# GEAK run report

Answers two questions about one run, side by side:

- **What did it cost?** Every LLM API call, in a `phase → task → subtask →
  sub-sub-task` hierarchy.
- **What did it buy?** The throughput each phase measured, with its gate.

A cost number without the matching gain number is how a phase that spends 60% of
the bill for a measured 0.00% ceiling goes unnoticed for months. Always render
both.

## Where the two halves live

| Half | Written by | Lands in |
| --- | --- | --- |
| Outcome / throughput | GEAK itself | `<eval_dir>/reports/geak_outcome.{json,md}` |
| Per-LLM-call tree | Claude Code | `<eval_dir>/llm_trace/` (mirrored), rendered to `<eval_dir>/reports/` |

GEAK issues almost no LLM calls itself: `interface/run_e2e.py` opens one
`ClaudeSDKClient` and hands a single prompt to Claude Code, which runs
`e2e_workflow/e2e_workflow.js` and tags every `agent()` call with
`{phase, label}`. Claude Code records that structure in its own config home, so
the call tree is **read**, not instrumented — no GEAK code writes a token ledger
and none needs to.

That home is the fragile part. It is `$CLAUDE_CONFIG_DIR`, else `~/.claude`, and
if that path is a container overlay it dies with the container and takes an
entire run's cost record with it. `run_e2e` therefore mirrors it into the run's
own `eval_dir` as the run proceeds. **Read the mirror, not the home** — the
mirror has the run's durability, the home does not.

## Steps

1. **Outcome.** This needs nothing but the run directory:

   ```bash
   python3 interface/geak_outcome_report.py <EVAL_DIR>            # writes reports/
   python3 interface/geak_outcome_report.py <EVAL_DIR> --stdout   # just look
   ```

   Read `—` as *the artifact was absent*, never as zero. "We did not measure it"
   and "it contributed nothing" are different claims and the report keeps them
   apart.

2. **Call tree.** The renderer lives in Hyperloom (`dump_geak_call_report`);
   GEAK reaches it by subprocess when a checkout is present:

   ```bash
   PYTHONPATH=<HYPERLOOM_SRC> python3 -m \
     hyperloom.inference_optimizer.tools.dump_geak_call_report \
     --claude-home <EVAL_DIR>/llm_trace --output-dir <EVAL_DIR>/reports
   ```

   Selection is an identity match on `args.eval_dir` (falling back to
   `args.exp_root`), never a guess by mtime. `--list` with no selector lists
   every record in every home.

3. **Join them.** `geak_outcome.md` already carries the per-phase spend table
   when `geak_calls.jsonl` sits beside it. Rank phases by
   `USD ÷ measured delta`, and treat any phase with a measured Amdahl ceiling of
   0.00% as having bought nothing however much it cost.

## Reading the numbers honestly

- **Group by `message.id`.** A Claude Code transcript repeats a message once per
  content block. Counting rows instead of messages overstates the bill by ~60%.
- **`thinking ⊂ output`** in these transcripts, so `OSL = output_tokens` alone —
  the inverse of Hyperloom's own ledger. Do not add them.
- **`ISL = input + cache_read + cache_creation`.**
- **Cost is derived** from the shipped rate card, never read from a field.
- **Phases do not join end to end.** Each measures its own before/after in its
  own server session, so one phase's `after` need not equal the next phase's
  `before`. The outcome report prints those seams and marks the compounded
  speedup an estimate when they exist. Quote the observed first-to-last figure
  when you want a measured number.
- **Check coverage before quoting anything.** A run still in flight reports
  partial coverage and phase labels inferred from artifact mtimes; the
  authoritative `{phase,label}` tags only exist once the `wf_*.json` record is
  written at completion.

## When a report is missing

Check whether the Claude home outlived the run before concluding anything about
the run itself. On a build without the mirror, the ledger is gone and cannot be
reconstructed — the run directory holds no token or cost field anywhere. The
throughput outcome, however, survives independently: it comes from the run's own
artifacts, so `geak_outcome_report.py` still works on such a run.
