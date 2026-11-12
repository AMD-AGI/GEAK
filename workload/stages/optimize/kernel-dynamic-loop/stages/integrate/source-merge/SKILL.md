---
name: source-merge
description: Combine evidence-backed kernel changes into an isolated candidate using their common ancestor and retained diffs.
---

# Source merge

Read the [Integrate contract](../README.md).
Use the common source ancestor and retained diffs to combine changes in a separate
candidate. Resolve conflicts against the computation contract and retain the individual
candidates as fallbacks. Leave the accepted source untouched.

Return the combined source and constituent evidence for Verify. With no useful
combination hypothesis, no integration call is needed.
