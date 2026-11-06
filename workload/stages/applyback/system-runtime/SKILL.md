---
name: system-runtime
description: Integrate a declared serving candidate into an isolated runtime and record what actually loaded, before handing its system identity and evidence to evaluation.
---

# Integrate a system candidate

Read the [invocation contract](../../CONTRACT.md#direct-module-calls). Use the candidate,
parent identity, native launch/build path and permitted resources named in the plan.
Keep the accepted system and its recovery artifacts intact. Apply source, build or
configuration changes to a separate runtime; use the selected engine's own mechanisms.

Record the actual image/build, effective settings and changed code paths. Exercise the
relevant serving paths before measured traffic; requested flags or a coordinator import
alone do not prove that workers used the change. Retain the native proof and define its
required values for Evaluate's `native_validation` when that format supports them.
A kernel overlay still uses [overlay-ab's integration checks](../overlay-ab/SKILL.md);
calling this module does not replace frozen-task or source-identity verification.

Return `REPORT.md`, the runtime/candidate identity, launch details and effect evidence,
or a blocker with the failed evidence preserved. Remove measurement-affecting probes
before handing the runtime to [Evaluate](../../evaluate/native-e2e/SKILL.md). Stop only
owned work within the task's permissions. This module claims integration, not a speedup.
