# Apply back calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls). For a kernel candidate,
use the integration steps of [overlay-ab](overlay-ab/SKILL.md): its source-identity and
actual-loading requirements remain in force. An integration-only call returns the
candidate/runtime identity and proof; it does not claim an E2E result or completion of
the standard applyback stage. For configuration/build candidates use
[system runtime](system-runtime/SKILL.md).

Hand the proven runtime and its evidence to [Evaluate](../evaluate/native-e2e/SKILL.md)
when the plan requests comparison. The standard recipe composes both operations: its
unchanged finalizer entry checks kernel integration and then invokes the shared Evaluate
criterion. Neither route replaces missing proof with an empty kernel record.
