---
name: native-profile
description: Answer a kernel bottleneck question with source inspection, native timings or an available profiler.
---

# Native inspection and profiling

Read the [Profile contract](../README.md).
Choose the smallest useful observation: source inspection, launch/case timings, counters
or a focused trace. Use an available profiler when it distinguishes the competing
explanations, with the same measurement isolation as Verify.
Identify the actual entry, dispatch and effective settings relevant to the question.
For candidate screening, prefer faithful Task probes. Distinguish operator/per-case
cost from test-runner wall time, and record the workload contribution when available.
Use faithful entry evidence to decide whether a full Verify is worth its cost.

Reuse earlier observations with their original source/input identities. Revisit profiling
when changed source or unresolved evidence makes a fresh observation useful to the next
plan. Retain earlier versions for comparison and return the evidence, uncertainty and
next useful experiment.
