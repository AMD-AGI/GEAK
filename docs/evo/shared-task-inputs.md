# Shared immutable task inputs

Large immutable fixtures can remain at one read-only task root while GEAK and
GEAK-EVO create private candidate source. The ordinary workspace materializer
and [native measurement adapter](direct-external-command.md) use the same
`geak-native-shared-inputs-v1` manifest.

Supply these values to the workflow host and its agent tools:

```text
GEAK_SHARED_INPUT_MANIFEST=/inputs/shared-inputs.json
GEAK_SHARED_INPUT_SHA256=<SHA-256 of the manifest bytes>
GEAK_SHARED_INPUT_ROOT=/task
```

The manifest contains the task artifact hashes and sizes, plus the exact
`shared_directories` and `shared_files`. Its artifact map must match `task.json`.
Shared files must be protected fixtures; editable `source_paths` stay private.
The manifest's `task_root` records the original location. The explicit launch
root maps those pinned artifacts into the current environment.

The materializer validates the manifest before clearing its destination. It
copies private files, then creates absolute links for the declared fixture
leaves. Their ordinary task-relative paths remain available to agents and tools.
Without this configuration, the existing copy behavior applies.

## Explicit options

The materializer accepts options that override the environment defaults:

```text
--shared-input-manifest /inputs/shared-inputs.json
--shared-input-sha256 <SHA-256 of the manifest bytes>
--shared-input-root /task
```

The native adapter accepts `--shared-input-manifest` and
`--shared-input-sha256`; its required `--task-root` supplies the root. That root
and shared targets must be on read-only mounts. Native execution links only
these declared fixture leaves into its work directory; candidate source and
other files remain private.

The source checks validate exact link targets, regular canonical fixture files,
and expected hashes. Unlisted links and protected edits remain errors. These
checks run before and after native task processes; they do not prove that no
temporary change occurred between checks.

Sharing fixtures avoids copying them into each candidate and native work tree.
Generated files, compiled artifacts, caches, and raw results still consume
storage. Plan capacity for those outputs and preserve evidence needed to inspect
the run. Fixture sharing alone does not establish a total storage bound.
