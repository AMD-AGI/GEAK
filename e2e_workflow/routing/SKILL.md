---
name: cost-ladder-routing
description: GEAK's default routing policy on expt/3-routing — send every agent to the lowest sufficient Claude model (Haiku 5.5 → Sonnet 5.5 → Opus 4.6 → Opus 5.5), climb only on deterministic failure, cap spend in code, and report every dollar. Use when running, reviewing or tuning a routed GEAK run, or when asked where a routed run over- or under-spent.
---

# Cost-ladder routing

Ship the same kernel for less money. Every agent starts on the cheapest model that can do its job,
and moves up only when a tool — not a model's opinion — shows that it failed.

This file is the **policy**. The code in `kernel_workflow/kernel_lane.js` (the region between
`<<ROUTING-INLINE-START>>` and `<<ROUTING-INLINE-END>>`) **enforces** it. If they ever disagree, the
code is what ran; fix whichever one is wrong. `routing_dryrun.js` checks the code against this file.

**Off by default.** Nothing here happens unless the run passes `routing=1` (or sets `GEAK_ROUTING=1`).
With routing off, the run is byte-identical to a build without this feature.

## Goal

Cut spend 60–80% on a GEAK run **without a worse kernel**. That range is a target, not a promise.
It counts only when measured against a matched run with routing off (see *Cost report*). A cheaper run
with a slower kernel is a loss, not a saving.

Why routing is the right lever: in the 2026-09-15 compaction ablation, cache writes and output were
two-thirds of the bill, and shortening the conversation barely touched them (it saved 4%). A cheaper
model lowers the price of **every** bucket — Haiku 5.5's cache write is $0.125/M against Opus 5.5's $5.

## The three layers

Keep these apart. Never let one do another's job.

| layer | who | does | never does |
|---|---|---|---|
| **Brain** | Opus 5.5 | Director and TechLead: plan, split work, adjudicate, validate | implementation work a worker can do |
| **Workers** | Haiku 5.5 / Sonnet 5.5 / Opus 4.6 / Opus 5.5 | everything else: engineers, verify, integrate, commit, profile, research, KB | choose its own model |
| **Decider** | Sonnet 5.5 | answers one typed question per decision (below) | write code, patches or plans; answer anything a tool already answered |

Fixed-script helpers (`clock`, `storage:reclaim`, `warm_start:resolve`) always run on Haiku 5.5. Their
output is schema-checked, so there is nothing to decide.

**Jev.** The prompt this policy came from names Jev as the decider. As of 2026-09-28 the AI Gateway
refuses every Jev request (see `research/09_jev_typesafe_routing.md`), so Sonnet 5.5 answers the same
typed schema natively inside the workflow. `route_decider=jev` is a reserved slot: it logs once and
falls back to Sonnet 5.5. When Jev serves, wire it behind that switch; keep the schema and the gates.

## The ladder

| lane | complexity | model | $/M in · cache write · cache read · out |
|---:|---|---|---|
| 0 | small | `claude-haiku-5-5` | 0.10 · 0.125 · 0.01 · 0.50 (prompt over 100k: 0.50 · 0.625 · 0.05 · 2.50) |
| 1 | medium | `claude-sonnet-5-5` | 2 · 2.50 · 0.20† · 10 |
| 2 | high | `claude-opus-4-6` | 5 · 6.25 · 0.50 · 25 |
| 3 | escalate | `claude-opus-5-5` | 4 · 5 · 0.20 · 20 |

Official prices, read 2026-09-28 (lanes 0 and 1 moved to Haiku 5.5 and Sonnet 5.5 on 2026-10-09). Opus 5.5
is also GEAK's default model on this branch (`interface/run_e2e.py`, `GEAK_CLAUDE_MODEL`). Note Opus 5.5 is
**cheaper** than Opus 4.6; lane 2 is there for capability spread, not price. Haiku 5.5 is priced per request
by prompt length: once input plus cache reads and writes pass 100,000 tokens, that request pays the second
row, five times the first. Haiku 5.5, Sonnet 5.5 and Opus 5.5 cache prompts from 512 tokens; Opus 4.6 needs 4,096.

† The pricing page now lists $0.10 for Sonnet 5.5 cache reads. Claude Code (2.1.295) still charges $0.20, and so
does the ledger, so that its cross-check against Claude Code's own cost keeps working. Not yet decided.

## Routing rules

1. **Classify before the first dispatch of a scope, once.** A scope is the dispatch label with its
   round/tag removed: `eng d3:memory` and `eng d7:memory` share the ladder `eng:memory`.
2. **Default to SMALL; the task must prove it needs more.** The decider reports its confidence that
   the *current* lane (Haiku, for a new scope) will finish the task. Only if that is **below 0.70**
   does the scope start on its classified lane. An invalid or missing answer proves nothing: SMALL.
3. **Never route up because a stronger model is available.** Only the triggers below move a scope up.
4. **Re-classify only after a failed cycle**, never after a success and never pre-emptively.
5. **Climb one lane at a time**, and only when, after a deterministic failure, at least one holds:
   - the decider's confidence in the current lane is below 0.70;
   - this lane has now failed **3** times for this scope;
   - the decider raises `risk_flag` = `security` or `data_loss`.
   Otherwise, retry on the same lane.

## What counts as a failure — deterministic evidence only

A model is never asked something a tool can prove. GEAK's tools already decide:

| scope | failed cycle means | proved by |
|---|---|---|
| engineers (`eng:*`, `deep:*`) | no **verified, correct, above-floor** patch this round — including an honest below-floor result | the verify agent's oracle run: `status`, `correctness`, measured speedup vs `CANDIDATE_FLOOR` |
| every other worker | no result at all: timeout, API fault, lost StructuredOutput | the workflow's own hang/fault guards |

Evidence passed back to the decider is one short line of status words and numbers — never a log,
never a diff body.

## The typed decision

One call, one schema (`ROUTE_DECISION_SCHEMA`):

| field | type | used for |
|---|---|---|
| `complexity` | `small \| medium \| high \| escalate` | the starting lane |
| `confidence` | 0–1: will the **current** lane finish this task? | both gates (0.70) |
| `risk_flag` | `none \| security \| data_loss` | forces a climb after a failure |
| `action` | `continue \| retry \| verify \| escalate \| complete` | recorded for review only |
| `scope_drift` | bool | recorded for review only |

`action` and `scope_drift` do not steer anything: in GEAK the oracle already decides "done", and
the TechLead plus Director validation already decide when a run stops. The prompt's "complete only
above 0.85 with all checks passing" is therefore **stricter in GEAK than asked**: completion needs the
oracle, and no confidence score can substitute for it.

## Budget layer — in code, never overridable by a model

| cap | default | arg |
|---|---|---|
| failures per lane before a forced climb | 3 | `route_max_retries_per_lane` |
| worker dispatches on Opus 5.5 per run | 1 (further requests run on Opus 4.6) | `route_max_top_escalations` |
| escalation gate | 0.70 | `route_conf_escalate` |
| output-token kill switch | 1,000,000 | `route_max_output_tokens` (0 = off) |
| per-scope starting floors | none | `route_floors` (JSON `{scope: lane}`) |

**Kill switch.** Once the output tokens spent since the lane started pass the cap, no new round
starts and no worker is dispatched. `budget.spent()` is the whole session's running total (it read
4.7M before the first live run began), so the lane records it at start and caps the growth. Whatever
else the session spends during the run counts too, so the switch can trip early but never late. Brain calls still run, so the TechLead report and the Director's validation of work
already done are not thrown away. Output tokens are the only spend a workflow script can see
(`budget.spent()`); dollars are settled afterwards by the ledger. If the runtime has no `budget`, the
switch stays off and says so in the report. For scale: the 3.7× fused-MoE run used 472,918.

**Not enforceable here:** a per-call `max_tokens` (the runtime exposes no such option), and a live
dollar cap (the workflow cannot see prices).

## Context hygiene

GEAK already does most of this and routing does not change it: every dispatch is a fresh agent,
history reaches the next round as the TechLead's compact `INSIGHTS`, files are read from the workspace
rather than pasted, and repeated context is cached by the API. Routing adds one caution: a cheaper
lane only pays if its prompt is long enough to cache (512 tokens on lanes 0, 1 and 3; 4,096 on Opus 4.6), and
Haiku 5.5 costs five times as much on a request whose prompt passes 100,000 tokens.

## After each run — review, then you decide

Nothing tunes itself. After a run, generate the review:

```bash
python3 e2e_workflow/routing/route_review.py \
  --calls   <eval>/reports/geak_calls.jsonl \
  --routing <file holding the lane result, e.g. the workflow return JSON> \
  --control <matched routing-OFF run>/reports/geak_calls.jsonl \
  --out     <eval>/reports/
```

It writes `route_review.md` and `proposed_floors.json`. A floor is proposed only where a scope failed
on a lane and succeeded higher up. A lane that succeeded every time is flagged for a **trial** one lane
lower — untested is not proven, so that is never proposed as a floor. To accept a proposal, pass the
JSON as `route_floors` in the next run. Nothing reads the file on its own.

## Cost report

A routed run is not finished until it has this report (`route_review.md` plus the ledger's
`geak_run_report_<model>.html`):

- lane used for every dispatch, and why (the audit trail in the lane result under `routing`)
- spend by model, and the decider's own cost as the `router` line
- retries and escalations with their reasons
- the deterministic checks and their results (verify's status, correctness, speedup)
- cache reuse
- the final diff (`final_patch.diff`)
- savings: **measured** against a matched routing-OFF run, or clearly labelled a counterfactual
  when there is none. Repricing the same tokens at Opus 5.5 rates is not a measurement: other models
  write other outputs, and Opus 4.6 counts about 30% fewer tokens for the same text (so did Haiku 4.5, the
  small lane before 2026-10-09).
- every step where a cheap lane cost more than it saved
- the ledger's **cross-check against Claude Code's own cost**. The API returns tokens, never dollars;
  Claude Code prices them and reports `total_cost_usd` and per-model `costUSD` in each
  `ResultMessage`, which `run_e2e.py` saves to `reports/sdk_results.json`. Our rates applied to
  Claude Code's own tokens must reproduce its dollars for **every** model (a miss marks the run
  incomplete: some lane's rate card is wrong). If calls finished after the last result, that total
  is partial — quote the ledger's, and say so. Both are Claude Code's list-price arithmetic, not an
  invoice.

A matched control means: same kernel, same GEAK commit, same GPU, same budget and deadline, routing
off, Opus 5.5 default. Without one, report no savings figure as fact.

## Final check — before calling a routed run done

- Is the requested kernel speedup actually delivered and Director-validated?
- Is the cheaper result equivalent — measured on the same oracle — or just untested?
- Did every deterministic check pass? Say so plainly if any did not.
- Does the final diff match the requested scope?
- Did each escalation happen because evidence demanded it, or because it felt safer?
- Never hide a failed verification to make the run look cheap.

End every routed-run report with a section titled **WHERE DID I OVERSPEND?**. A system that stays
correct beats one that looks cheap.

## Honest limits

- The decider's confidence is uncalibrated. The JEV-as-a-Judge paper (arXiv:2609.26550) found
  thresholds fitted on one task point the wrong way on another, and judge confidence is weakest
  exactly on hard code correctness. That is why failure is always decided by a tool, and why 0.70 is
  a starting point for `route_review.py` to question, not a measured optimum.
- Starting kernel engineers on Haiku is the operator's explicit choice. It risks the speedup the way
  context compaction did (−34% speedup for −4% cost). Watch the first routed run's speedup before
  trusting its savings.
- The model IDs above have not all been served through this gateway yet. Check each one answers
  before a long run.
- Only the kernel lane has the full ladder. `e2e_workflow.js` keeps its two-scope verbatim-write
  cascade (Sonnet 5.5, with Opus 5.5 as the fallback).
