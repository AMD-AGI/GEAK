# Verify

Establish candidate validity and performance under the [shared contract](../CONTRACT.md).
Initial implementation: [frozen-ut](frozen-ut/SKILL.md).

**Input:** an immutable candidate, comparison source, frozen task and test budget.
**Output:** native validity, full-task score, comparison basis, costs and source-bound evidence.
**Acceptance:** the exact candidate passes the native task with its complete cases and
device set. Generator claims and selected cases cannot substitute for this check;
changed or merged source cannot inherit an earlier verdict. Final cold checks belong to the parent.
