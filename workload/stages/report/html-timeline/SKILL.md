---
name: html-timeline
description: Summarize a workload campaign's actual decisions, module calls and parallel work in a compact HTML timeline for handoff or a requested route review.
---

# HTML timeline report

Follow the [direct-call contract](../../CONTRACT.md#direct-module-calls). Read the run's
brief, state, invocation plans/reports and native timestamps. Write a standalone
`timeline.html` in the requested output directory; link it from the campaign report.
Native receipts own results and time accounting; this module presents existing evidence.

Adapt the [example](assets/example.html)'s visual style, using the user's language.
Its events, task counts, methods, metrics and layout are example data, not workflow rules.

- Use a consistent muted color per module and a small legend. Put Reasoning above the
  main Module lane, ordered by recorded time. Hover or focus
  shows a short previous report, current plan, called module/implementation and time.
- Use as many lanes as actual concurrent branches require. Group tasks sharing a plan
  and outcome; split when their paths diverge and rejoin only at an actual handoff.
  Independent calls that do not overlap can reuse a lane; secondary detail may collapse.
- Label module bars with task and duration; expose exact intervals, results and relative
  evidence links on interaction. Keep returns to UT, retries, recovery and unfinished
  branches visible. Distinguish reconstructed decisions from recorded events; leave
  unknown times unknown. Label any compressed scale and overlapping duration totals.
- Keep one HTML with inline SVG/CSS and small interaction code, without external assets
  or a new generator/framework. Check readability, hover/focus, touch and evidence links.

Return the HTML link and any evidence or viewing limits in the invocation's `REPORT.md`.
