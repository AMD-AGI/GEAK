---
name: html-timeline
description: Summarize a workload campaign's actual decisions, module calls and parallel work in a compact HTML timeline for handoff or a requested route review.
---

# HTML timeline report

Read the [Report contract](../README.md) and the run's
brief, state, invocation plans/reports and native timestamps. Write a standalone
`timeline.html` in the requested output directory; link it from the campaign report.
Native receipts own results and time accounting; this module presents existing evidence.

Adapt the [example](assets/example.html)'s visual style, using the user's language.
Its events, task counts, methods, metrics and layout are example data, not workflow rules.

- Put Reasoning in one row above Module lanes, ordered by recorded time. Each node
  names its decision in a short action; hover or focus shows the previous report,
  current plan, module/implementation and time.
- Connect the actual loop: module report → Reasoning → module plan. A progress report
  may trigger a decision while its module runs; attach it at the observed time, not
  the eventual end. Continuing or adjusting a call points back to that call. Keep
  task input and final handoff explicit; do not invent calls or use Reasoning-to-
  Reasoning shortcuts to fill gaps. Highlight the selected loop and mute other edges.
- Use as many lanes as actual concurrent branches require. Group tasks sharing a plan
  and outcome; split when their paths diverge and rejoin only at an actual handoff.
  All lanes use the same module-card style and visual weight; color identifies the module,
  with a small legend. Calls that do not overlap can reuse a lane.
- Label calls with task and duration; expose exact intervals, results and relative
  evidence links on interaction. Keep repairs, retries and unfinished work visible;
  distinguish unexecuted plans from calls and reconstructed decisions from records.
  Leave unknown times unknown. A compact flow view must label its non-temporal widths;
  a time-proportional view keeps event points and intervals on one linear scale,
  moving labels only to avoid overlap. Both views use the same records.
- Keep one HTML with inline SVG/CSS and small interaction code, without external assets
  or a new generator/framework. Check readability, hover/focus, touch and evidence links.

Return the HTML link and any evidence or viewing limits in the invocation's `REPORT.md`.
