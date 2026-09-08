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
| Both, joined and readable | the HTML renderer | `<eval_dir>/reports/geak_report.html` |

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

3. **The HTML report — start here when someone asks "where did the money go?"**
   The tree from step 2 is exhaustive but flat: thousands of call rows answer
   *what happened*, not *where the cost is*. This renderer arranges the same
   data as a handful of questions, and joins in the outcome from step 1:

   ```bash
   PYTHONPATH=<HYPERLOOM_SRC> python3 -m \
     hyperloom.inference_optimizer.tools.render_geak_html_report \
     --reports-dir <EVAL_DIR>/reports
   ```

   It is written automatically at the end of a run when a Hyperloom checkout is
   reachable (`HYPERLOOM_SRC`, or `GEAK_HTML_REPORT_CMD` to override), so on a
   normal run there is nothing to do but open it. The page is self-contained and pure ASCII, so it
   survives being copied to shared storage and cannot render as mojibake in a
   viewer that ignores the charset declaration. Its sections:

   | Section | Answers |
   | --- | --- |
   | Headline cards | Throughput gained, baseline -> final tok/s, spend, calls, wall-clock |
   | What each phase contributed to throughput | The ladder in run order: from/to tok/s, the gain, and each phase's share of the summed measured gain |
   | What each phase bought | The same gains put beside what they cost, and `$ per +1%` |
   | Spend by phase | Where the bill is, ranked |
   | Inside each phase | Cost by position in the conversation, how few agents carry the total, what tools the work consisted of, and every agent drillable to its own API calls |
   | Delegation signals | Input/output ratio and tool variety per agent |

   **Share of measured gain is arithmetic, not attribution.** A phase's own
   tok/s gain over the summed tok/s gains of the phases that measured one. It
   does not reconcile with the end-to-end figure and is not meant to: a phase
   does not always start from where the previous one finished, and the page
   names the handoff seam that accounts for the gap.

   **Reading the deep dive.** Every call re-sends the conversation so far, so an
   agent's input grows as it works — on the Qwen3 run, median ISL went from
   51k tokens in the first tenth of a conversation to 164k in the last. That is
   the mechanism behind most of a long phase's bill, and it is why the position
   table is the first thing in each phase block.

   **The delegation table is signals, not a verdict.** A low output-to-input
   ratio means an agent spent its budget reading rather than writing, and a
   single-tool agent ran a mechanical loop. Both are reasons to *look*. Neither
   is evidence a smaller model would have reached the same result — only running
   the arm and comparing measured throughput settles that.

4. **Join them yourself if you need a number, not a page.** `geak_outcome.md` already carries the per-phase spend table
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
- **Read the HTML's Coverage banner before quoting anything from it.** It names
  what the ledger does not contain for that run — unpriced calls, calls with no
  recorded duration (wall-clock sums are then lower bounds, not elapsed time),
  and agents whose role could not be read.
- **Roles in the HTML are derived**, inferred from each agent's first prompt
  because `geak_calls.jsonl` truncates prompts and carries no label. The
  authoritative `{phase,label}` pair lives in the `wf_*.json` record. An agent
  whose role could not be read is shown `unlabelled`, never guessed at.
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
